package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"
)

// defaultGeminiBaseURL is the first-party Gemini API (AI Studio) host. The
// request path is appended by the client, so BaseURL never includes the API
// version - unlike the OpenAI client, whose BaseURL convention carries /v1.
const defaultGeminiBaseURL = "https://generativelanguage.googleapis.com"

// geminiAPIVersion is the pinned path version. generateContent is served under
// v1beta on AI Studio; pinning it here keeps the request shape stable and is
// what makes a base_url that already ends in /v1beta a config error (the
// version would then be sent twice - see config.validateProvider).
const geminiAPIVersion = "v1beta"

// The two roles this wire has. There is no system role: the system prompt is a
// top-level systemInstruction Content (design doc §"Gemini-specific mechanics"
// item 6).
const (
	geminiRoleUser  = "user"
	geminiRoleModel = "model"
)

// GeminiClient talks to the Gemini API natively
// (POST /v1beta/models/{model}:generateContent, AI Studio key auth). It
// implements Provider.
//
// The client speaks raw HTTP instead of using the official genai SDK on
// purpose: that SDK alone is 15 MB, over amele's whole binary budget, while
// only one endpoint is needed (design doc §Rulings 2). Hand-rolled wire types
// also keep this client consistent with the house pattern the OpenAI and
// Anthropic clients established - typed statusError routing, injected Sleep,
// capped error snippets.
//
// SCOPE: this slice speaks conversations and tool calls, and round-trips the
// thought signatures Gemini 3 signs its steps with. The
// thinkingConfig/responseJsonSchema knobs declare their wire fields here but
// are not emitted yet (Task 4); toWire says so at its site.
type GeminiClient struct {
	// BaseURL is the API root without a trailing slash and WITHOUT the version
	// segment, e.g. "https://generativelanguage.googleapis.com". Empty means
	// the first-party host.
	BaseURL string
	// APIKey is sent as the x-goog-api-key header when non-empty.
	APIKey string
	// HTTPClient is injectable for tests; nil means a client bounded by
	// RequestTimeout.
	HTTPClient *http.Client
	// RequestTimeout bounds a single HTTP round-trip when HTTPClient is nil.
	// Zero means defaultRequestTimeout (120s). Wired from the config's
	// provider.request_timeout, same semantics as the other clients.
	RequestTimeout time.Duration
	// MaxAttempts overrides defaultMaxAttempts when > 0.
	MaxAttempts int
	// InitialBackoff is the wait before the second attempt, doubled for every
	// attempt after that. Zero means defaultInitialBackoff (1s). Wired from the
	// config's provider.retry.initial_backoff.
	InitialBackoff time.Duration
	// Sleep is injectable for tests; nil means context-aware sleeping.
	// Determinism rule (docs/engineering.md §5.4): time-dependent behavior must
	// be injectable.
	Sleep func(ctx context.Context, d time.Duration) error
}

// Wire types for the generateContent JSON body. Kept unexported: the rest of
// the codebase only sees the neutral types in llm.go.
// Every field amele owns is a struct member (not a map entry) so the encoded
// key order is fixed by the declaration order and the wire goldens are
// byte-stable; the caller's raw provider.params are merged afterwards (see
// encodeBody).
//
// GeminiOwnedWireFields lists these same keys for the params collision check -
// keep the two in step when a field is added here.
//
// The model is NOT a body field on this wire: it lives in the request path.
type gemRequest struct {
	Contents []gemContent `json:"contents"`
	// SystemInstruction is a pointer so the key vanishes when no system prompt
	// was given: protobuf-JSON is strict about the shapes it accepts, and an
	// empty Content is not one of them.
	SystemInstruction *gemContent `json:"systemInstruction,omitempty"`
	// Tools carries every function declaration in ONE entry (see gemTool).
	// Gemini rejects unknown JSON Schema keywords with hard 400s, so nothing
	// reaches this field without passing SanitizeGeminiSchema first.
	Tools            []gemTool            `json:"tools,omitempty"`
	GenerationConfig *gemGenerationConfig `json:"generationConfig,omitempty"`
}

// GeminiOwnedWireFields returns the request-body keys the native
// generateContent client writes itself - OwnedWireFields for a WIRE rather than
// a dialect, since the dialect is not consulted on this path (a dialect on the
// gemini wire is a validate error).
//
// CONTRACT: the list mirrors gemRequest's json tags plus the keys a later slice
// will write (toolConfig, safetySettings, cachedContent), which is what makes it
// the right answer for provider.params collisions on this wire
// (config.validateParams). Both spellings are listed on purpose: the wire is
// camelCase but protobuf-JSON accepts the snake_case form too, and a params key
// that reached the body under EITHER spelling would clobber a field amele owns.
//
// The model is absent because it is not a body field on this wire at all.
func GeminiOwnedWireFields() []string {
	return []string{
		"contents",
		"system_instruction", "systemInstruction",
		"tools",
		"tool_config", "toolConfig",
		"generation_config", "generationConfig",
		"safety_settings", "safetySettings",
		"cached_content", "cachedContent",
	}
}

// gemContent is one conversation turn (or the system instruction, which carries
// no role).
type gemContent struct {
	Role  string    `json:"role,omitempty"`
	Parts []gemPart `json:"parts"`
	// PartsRaw, when non-nil, REPLACES Parts: it is the parts array exactly as
	// the provider sent it, echoed back verbatim (see MarshalJSON and
	// appendAssistant). It never appears as a field of its own on the wire.
	PartsRaw json.RawMessage `json:"-"`
}

// MarshalJSON implements json.Marshaler.
//
// CONTRACT: this is the byte-exact echo path. When PartsRaw is set the turn is
// rendered with those bytes as its parts array instead of re-encoding the
// decoded parts, because Gemini 3 SIGNS thinking and function-call parts and
// requires the signature back unmodified - on the first functionCall part of
// every step - rejecting an altered or reordered array with a 400 (design doc
// §Rulings 3). Passing the raw region through means nothing here can reorder,
// re-escape or drop a signature, not even a field this client does not know.
//
// The one transformation the payload undergoes is the encoder's own compaction
// and its JSON-level escaping of <, > and &: value-preserving (the provider
// decodes the same strings it produced, which is what a signature is computed
// over), and the same trade-off the Anthropic client documents on its carrier.
func (c gemContent) MarshalJSON() ([]byte, error) {
	if c.PartsRaw != nil {
		return json.Marshal(struct {
			Role  string          `json:"role,omitempty"`
			Parts json.RawMessage `json:"parts"`
		}{Role: c.Role, Parts: c.PartsRaw})
	}
	// A defined type without gemContent's methods: marshalling it directly
	// would recurse into this function forever.
	type plain gemContent
	return json.Marshal(plain(c))
}

// gemPart is one piece of a turn. A single struct covers every part type this
// wire has; omitempty keeps the irrelevant fields off the wire for each one.
//
// A thought part must not leak into the answer text (see gemAssistantMessage),
// and ThoughtSignature is never read from this struct on the way back: a signed
// step is echoed from the raw parts array instead (gemContent.MarshalJSON), so
// the signature cannot be re-encoded by a decode/encode round trip.
type gemPart struct {
	Text             string               `json:"text,omitempty"`
	FunctionCall     *gemFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *gemFunctionResponse `json:"functionResponse,omitempty"`
	// Thought marks a thinking summary part rather than answer text.
	Thought bool `json:"thought,omitempty"`
	// ThoughtSignature is the opaque, base64 signature Gemini 3 attaches to
	// thinking and function-call parts and requires back unmodified.
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
}

// gemFunctionCall is a tool invocation requested by the model. Args stays raw:
// the neutral ToolCall carries the model's own JSON and parsing is deferred to
// the tool layer.
type gemFunctionCall struct {
	ID   string          `json:"id,omitempty"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

// gemFunctionResponse is one tool result travelling back to the model. Response
// must be an OBJECT on this wire, which is why the neutral string result is
// wrapped (Task 3).
type gemFunctionResponse struct {
	ID       string          `json:"id,omitempty"`
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

// gemGenerationConfig carries every per-request generation knob. The whole
// object is omitted when it would be empty (see empty): amele sends no key it
// was not asked for, which keeps the wire goldens honest about what a plain
// run costs.
type gemGenerationConfig struct {
	MaxOutputTokens int `json:"maxOutputTokens,omitempty"`
	// Temperature and TopP are pointers because 0 is a meaningful sampling
	// value: omitempty on a float64 would silently drop `temperature: 0`, the
	// exact setting a deterministic run asks for.
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"topP,omitempty"`
	// StopSequences has no neutral knob behind it today. It is declared because
	// it belongs to the object amele owns: were it left out, a params key could
	// not be refused for it (GeminiOwnedWireFields owns generationConfig whole).
	StopSequences []string `json:"stopSequences,omitempty"`
	// ResponseMimeType and ResponseJSONSchema are the structured-output pair
	// (Task 4); ThinkingConfig is the reasoning knob (Task 4). Declared here so
	// the object's key order - and therefore every golden - is fixed once.
	ResponseMimeType   string          `json:"responseMimeType,omitempty"`
	ResponseJSONSchema json.RawMessage `json:"responseJsonSchema,omitempty"`
	ThinkingConfig     *gemThinking    `json:"thinkingConfig,omitempty"`
}

// empty reports whether the object would encode to {}. In that case the key is
// dropped entirely rather than sent empty: this wire 400s on shapes it does not
// expect, and an empty generationConfig is noise no other amele wire sends.
func (g gemGenerationConfig) empty() bool {
	return g.MaxOutputTokens == 0 && g.Temperature == nil && g.TopP == nil &&
		len(g.StopSequences) == 0 && g.ResponseMimeType == "" &&
		len(g.ResponseJSONSchema) == 0 && g.ThinkingConfig == nil
}

// gemThinking is the reasoning control object (Task 4 maps the neutral knob
// onto it). ThinkingBudget is a pointer because 0 is the meaningful "turn
// thinking off" value on the 2.5-era models, which omitempty would drop.
type gemThinking struct {
	ThinkingLevel  string `json:"thinkingLevel,omitempty"`
	ThinkingBudget *int   `json:"thinkingBudget,omitempty"`
}

// gemTool is one tools entry. Gemini groups all function declarations under a
// single tool object rather than listing one tool per function (Task 3).
type gemTool struct {
	FunctionDeclarations []gemFunctionDecl `json:"functionDeclarations,omitempty"`
}

// gemFunctionDecl is one declared function. Parameters is an OpenAPI-3.0 SUBSET
// on this wire, not arbitrary JSON Schema, which is why it may only be filled
// through the sanitizer (Task 3).
type gemFunctionDecl struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type gemResponse struct {
	Candidates []gemCandidate `json:"candidates"`
	// PromptFeedback explains a response that carries no candidate at all: the
	// prompt itself was blocked. It is the only thing that can name that
	// failure, so it is decoded even though nothing else consults it.
	PromptFeedback *struct {
		BlockReason string `json:"blockReason"`
	} `json:"promptFeedback"`
	// UsageMetadata is a pointer so "provider omitted usage entirely" is
	// distinguishable from "zero tokens" - token budgets fail closed on the
	// former (see llm.Response.UsageMissing).
	// int64, not int: a 32-bit build must be able to DECODE an absurd count
	// (json rejects an int overflow outright) so parseUsage can clamp it.
	UsageMetadata *struct {
		PromptTokenCount     int64 `json:"promptTokenCount"`
		CandidatesTokenCount int64 `json:"candidatesTokenCount"`
		ThoughtsTokenCount   int64 `json:"thoughtsTokenCount"`
	} `json:"usageMetadata"`
}

type gemCandidate struct {
	Content struct {
		Role string `json:"role"`
		// Parts stays RAW at decode time and is parsed separately
		// (gemParts). The raw region is what Task 3's reasoning carrier will
		// need: Gemini signs thinking and function-call parts and rejects a
		// modified array, so the decoder must not be the only thing that ever
		// sees those bytes.
		Parts json.RawMessage `json:"parts"`
	} `json:"content"`
	FinishReason string `json:"finishReason"`
}

// gemErrorEnvelope is the google.rpc error body every non-2xx reply carries.
// Details are kept raw: only the RetryInfo entry is read (parseGoogleRetryDelay)
// and the rest are shapes this client has no reason to model.
type gemErrorEnvelope struct {
	Error struct {
		Code    int               `json:"code"`
		Status  string            `json:"status"`
		Message string            `json:"message"`
		Details []json.RawMessage `json:"details"`
	} `json:"error"`
}

// geminiErrorSignatures is the ordered table consulted for a non-retryable 400
// on the generateContent wire, the counterpart of errorSignatures on the OpenAI
// wire and anthropicErrorSignatures on the Messages API.
//
// It is EMPTY in this slice, and deliberately so: every signature the design
// names (a missing thought_signature, a thinkingLevel/thinkingBudget conflict,
// "thinking cannot be disabled" on Gemini 3, an unknown responseJsonSchema
// field) is advice about the REASONING knobs, which Task 4 owns. The parameter
// is threaded through now so that task adds a table entry and nothing else.
var geminiErrorSignatures []errorSignature

// Chat implements Provider. It retries 429 and 5xx with exponential backoff,
// honoring the retry delay this API carries in the error BODY rather than in a
// Retry-After header (design doc §"Gemini-specific mechanics" item 3).
//
// The retry loop mirrors the other two clients' rather than sharing a helper:
// they evolve independently - each carries its own capability fallback for the
// field its wire spells differently - and extracting the ~20 shared lines would
// couple their futures for no robustness gain. What IS shared is the machinery
// underneath: backoffDelay, statusFailure and encodeBody.
func (c *GeminiClient) Chat(ctx context.Context, req Request) (*Response, error) {
	wire, fields := c.toWire(req)
	body, err := encodeBody(wire, fields)
	if err != nil {
		return nil, fmt.Errorf("%w: encoding request: %v", ErrProvider, err)
	}

	attempts := c.MaxAttempts
	if attempts <= 0 {
		attempts = defaultMaxAttempts
	}

	var lastErr error
	var retryAfter time.Duration
	for attempt := 1; attempt <= attempts; attempt++ {
		if attempt > 1 {
			// Exponential backoff from InitialBackoff (1s, 2s, 4s... by
			// default), stretched to the provider's own wish when one was sent,
			// and bounded by the caller's context deadline (sleep aborts when
			// ctx is done).
			if err := c.sleep(ctx, backoffDelay(c.InitialBackoff, attempt, retryAfter)); err != nil {
				return nil, fmt.Errorf("%w: %v (last error: %v)", ErrProvider, err, lastErr)
			}
		}

		resp, retryable, ra, err := c.doOnce(ctx, req.Model, body)
		if err == nil {
			return resp, nil
		}
		lastErr = err
		retryAfter = ra
		if !retryable {
			return nil, err
		}
	}
	return nil, fmt.Errorf("%w: retries exhausted: %v", ErrProvider, lastErr)
}

// doOnce performs a single HTTP round-trip. retryable reports whether the
// failure is worth retrying; retryAfter carries the provider's wish (0 when
// absent). The model is a parameter rather than a body field because this wire
// puts it in the URL.
func (c *GeminiClient) doOnce(ctx context.Context, model string, body []byte) (resp *Response, retryable bool, retryAfter time.Duration, err error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint(model), bytes.NewReader(body))
	if err != nil {
		return nil, false, 0, fmt.Errorf("%w: building request: %v", ErrProvider, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		// SECURITY: the Gemini API authenticates through this header, not a
		// Bearer token; sending the key in the wrong header would leak it to
		// intermediaries expecting no credential there. The documented query
		// parameter form (?key=) is refused on purpose - a URL is logged by
		// every proxy on the path.
		httpReq.Header.Set("x-goog-api-key", c.APIKey)
	}

	httpResp, err := c.httpClient().Do(httpReq)
	if err != nil {
		// Context cancellation is not retryable - the run is being aborted.
		if ctx.Err() != nil {
			return nil, false, 0, fmt.Errorf("%w: %v", ErrProvider, ctx.Err())
		}
		// Neither is the client-side request timeout: a generation that
		// exceeded the per-request budget will exceed it again, so retrying
		// only multiplies the wasted wall-clock. Name the knob to turn.
		var netErr net.Error
		if errors.As(err, &netErr) && netErr.Timeout() {
			return nil, false, 0, fmt.Errorf("%w: request timed out after %s; raise provider.request_timeout if the model legitimately needs longer", ErrProvider, c.requestTimeout())
		}
		// Network-level failures (DNS, refused, reset) are retryable.
		return nil, true, 0, fmt.Errorf("%w: %v", ErrProvider, err)
	}
	defer func() { _ = httpResp.Body.Close() }()

	if httpResp.StatusCode != http.StatusOK {
		return c.failure(httpResp)
	}

	var wire gemResponse
	if err := decodeResponseBody(httpResp.Body, &wire); err != nil {
		return nil, false, 0, fmt.Errorf("%w: decoding response: %v", ErrProvider, err)
	}
	resp, err = geminiResponse(wire)
	if err != nil {
		return nil, false, 0, err
	}
	return resp, false, 0, nil
}

// failure turns a non-200 reply into the typed provider error, adding the one
// thing this wire does differently: the retry delay arrives in the BODY as a
// google.rpc.RetryInfo detail, not in a Retry-After header.
func (c *GeminiClient) failure(httpResp *http.Response) (*Response, bool, time.Duration, error) {
	// Shared with the other clients: same retry band (429 and every 5xx), same
	// typed statusError, same header reading - only the error-signature table
	// and the body-borne delay below differ.
	retryable, retryAfter, err := statusFailure(httpResp, geminiErrorSignatures)
	if !retryable || retryAfter > 0 {
		return nil, retryable, retryAfter, err
	}
	// The snippet statusFailure already read is the only copy of the body (it
	// consumed the reader), so the delay is recovered from there rather than by
	// re-reading. A body truncated at maxErrorBody no longer parses as JSON and
	// simply yields no wish - the exponential ladder then stands, which is the
	// same outcome as an endpoint that sent no RetryInfo at all.
	var se *statusError
	if errors.As(err, &se) {
		var env gemErrorEnvelope
		if json.Unmarshal([]byte(se.snippet), &env) == nil {
			retryAfter = parseGoogleRetryDelay(env.Error.Details)
		}
	}
	return nil, retryable, retryAfter, err
}

// geminiResponse turns one decoded generateContent body into the neutral
// response: the first candidate's parts become the assistant message, its
// finishReason is translated, and usageMetadata becomes the sanitized usage.
func geminiResponse(wire gemResponse) (*Response, error) {
	if len(wire.Candidates) == 0 {
		// A 200 with no candidate is a failed turn, not an empty answer: the
		// prompt itself was rejected. Naming the block reason is the only way
		// the operator learns why, so it rides in the message when present.
		if wire.PromptFeedback != nil && wire.PromptFeedback.BlockReason != "" {
			return nil, fmt.Errorf("%w: response has no candidates (prompt blocked: %s)", ErrProvider, wire.PromptFeedback.BlockReason)
		}
		return nil, fmt.Errorf("%w: response has no candidates", ErrProvider)
	}
	candidate := wire.Candidates[0]
	msg, err := gemAssistantMessage(candidate.Content.Parts)
	if err != nil {
		return nil, fmt.Errorf("%w: decoding response: %v", ErrProvider, err)
	}
	finish, err := mapGeminiFinishReason(candidate.FinishReason)
	if err != nil {
		return nil, err
	}

	resp := &Response{Message: msg, FinishReason: finish}
	if wire.UsageMetadata == nil {
		resp.UsageMissing = true
		return resp, nil
	}
	// CONTRACT: thinking tokens are BILLED AS OUTPUT (design doc
	// §"Gemini-specific mechanics" item 5), so they are added to the visible
	// output count. Counting only candidatesTokenCount would let a reasoning
	// run spend far past limits.max_tokens while the budget reported it as
	// cheap. Sanitizing happens in parseUsage, the same boundary every other
	// wire crosses.
	usage, trustworthy := parseUsage(
		wire.UsageMetadata.PromptTokenCount,
		geminiOutputTokens(wire.UsageMetadata.CandidatesTokenCount, wire.UsageMetadata.ThoughtsTokenCount),
	)
	resp.Usage = usage
	resp.UsageMissing = !trustworthy
	return resp, nil
}

// geminiOutputTokens folds the two output-side counters into the single number
// the neutral Usage carries.
//
// CONTRACT: it preserves the two signals parseUsage keys on. A negative counter
// (impossible, therefore untrustworthy) must stay negative rather than cancel
// against a positive sibling, and an absurd pair must saturate rather than
// overflow int64 into a small - or negative - total that reads as "cheap".
func geminiOutputTokens(candidates, thoughts int64) int64 {
	if candidates < 0 || thoughts < 0 {
		return -1
	}
	if thoughts > math.MaxInt64-candidates {
		return math.MaxInt64
	}
	return candidates + thoughts
}

// gemAssistantMessage turns the raw parts array of one candidate into the
// neutral assistant message.
//
// Only text parts contribute to the content, and a part marked thought is
// skipped: it is the model's thinking summary, and letting it into Content
// would hand it to the user, to output.schema validation and to the session log
// as if it were the answer. functionCall parts become tool calls in the order
// the model produced them, which is the order the loop answers them in.
func gemAssistantMessage(parts json.RawMessage) (Message, error) {
	decoded, err := gemParts(parts)
	if err != nil {
		return Message{}, err
	}
	msg := Message{Role: RoleAssistant}
	signed := false
	for _, part := range decoded {
		// A signature can ride on ANY part, and a thought part implies one is
		// coming: either means this array must go back untouched.
		if part.ThoughtSignature != "" || part.Thought {
			signed = true
		}
		if part.Thought {
			continue
		}
		if part.FunctionCall != nil {
			msg.ToolCalls = append(msg.ToolCalls, ToolCall{
				// The id is empty on the 2.5-era models, which send none. It is
				// tolerated rather than invented: the pairing then falls back
				// to call order (gemBuilder.takeCall).
				ID:        part.FunctionCall.ID,
				Name:      part.FunctionCall.Name,
				Arguments: compactJSONObject(part.FunctionCall.Args),
			})
			continue
		}
		// Multiple text parts concatenate: the neutral Message carries one text
		// body, and the API guarantees part order.
		msg.Content += part.Text
	}
	// CONTRACT: the carrier is the ENTIRE raw parts array, not the signed parts
	// alone. Gemini 3 wants the signature back on the first functionCall part
	// of the step, interleaved with the text and calls it was produced with, so
	// the array is the unit that round-trips - which makes signature position
	// and part order correct by construction rather than by a rule this code
	// has to remember. Nothing here parses the payload.
	if signed {
		msg.Reasoning = parts
	}
	return msg, nil
}

// gemParts decodes the raw parts array. An absent or null parts field is not an
// error - it is an empty turn, which a MAX_TOKENS or MALFORMED_FUNCTION_CALL
// candidate really does produce - but a parts field that is not an array is:
// this client cannot form a message from it, and failing here names the
// response instead of producing a silently empty answer.
func gemParts(raw json.RawMessage) ([]gemPart, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, nil
	}
	var parts []gemPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, fmt.Errorf("parts: %w", err)
	}
	return parts, nil
}

// mapGeminiFinishReason translates a Gemini finish reason into the
// OpenAI-compatible vocabulary the loop understands. Unknown values pass
// through lowercased: the loop's badFinish path already handles unrecognized
// reasons defensively, and inventing a translation here would hide new provider
// states.
//
// MALFORMED_FUNCTION_CALL is the one reason that becomes an ERROR. The API
// answers 200 with it, so nothing below this client would notice the turn
// failed; the model produced a function call that cannot be parsed, and the
// loop would see an empty answer and stop as if the task were done.
func mapGeminiFinishReason(reason string) (string, error) {
	switch reason {
	case "STOP":
		return "stop", nil
	case "MAX_TOKENS":
		// CONTRACT: truncation must map to "length" so the loop hard-fails the
		// run; the passthrough default would land it in badFinish's
		// unknown-reason branch, which accepts non-empty content - letting a
		// truncated answer exit 0 in unattended cron runs.
		return "length", nil
	case "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII":
		return "content_filter", nil
	case "MALFORMED_FUNCTION_CALL":
		return "", fmt.Errorf("%w: the model emitted a malformed function call (finish reason MALFORMED_FUNCTION_CALL); simplify the tool's parameter schema or retry the run", ErrProvider)
	default:
		return strings.ToLower(reason), nil
	}
}

// googleRetryInfoType is the type URL of the error detail carrying the wait the
// API wants before the next attempt.
const googleRetryInfoType = "google.rpc.RetryInfo"

// parseGoogleRetryDelay returns the wait a google.rpc.RetryInfo detail asks
// for, or 0 when the details carry none.
//
// This is the Gemini counterpart of parseRetryAfter: a 429 here carries NO
// Retry-After header, and the delay arrives in the error body instead (design
// doc §"Gemini-specific mechanics" item 3). The value is a protobuf Duration in
// its JSON form - seconds with an optional fraction and a trailing "s" ("3s",
// "3.5s") - which time.ParseDuration reads directly. Anything unparseable,
// zero or negative yields 0 so the exponential ladder stands; the caller caps
// whatever comes back (maxRetryAfter).
//
// The type URL is matched by SUFFIX because the prefix is a registry host
// ("type.googleapis.com/") that is not part of the identity of the message.
func parseGoogleRetryDelay(details []json.RawMessage) time.Duration {
	for _, raw := range details {
		var detail struct {
			Type       string `json:"@type"`
			RetryDelay string `json:"retryDelay"`
		}
		if err := json.Unmarshal(raw, &detail); err != nil {
			continue
		}
		if !strings.HasSuffix(detail.Type, googleRetryInfoType) {
			continue
		}
		delay, err := time.ParseDuration(strings.TrimSpace(detail.RetryDelay))
		if err != nil || delay <= 0 {
			return 0
		}
		return delay
	}
	return 0
}

// toWire converts the neutral request into the generateContent JSON shape: the
// system prompt moves to the top-level systemInstruction, the assistant role is
// renamed to "model", and the cap and sampling knobs land inside
// generationConfig. It returns the struct-encoded part of the body and the
// body-root fragments merged afterwards (the caller's raw provider.params).
//
// CONTRACT: params keys cannot collide with the fields amele owns - config
// validation rejects that at exit 2 (GeminiOwnedWireFields) - so merging them
// needs no further defense here.
//
// SCOPE: req.Reasoning and req.ResponseFormat are deliberately NOT read. They
// need the thinkingConfig/responseJsonSchema mapping (Task 4); emitting a
// half-mapped field would be a 400 in production rather than a missing feature.
// No config can reach this client before that task lands, because the cmd
// wiring is Task 5.
func (c *GeminiClient) toWire(req Request) (gemRequest, map[string]json.RawMessage) {
	var b gemBuilder
	for _, m := range req.Messages {
		b.appendMessage(m)
	}
	b.req.applyTools(req.Tools)
	b.req.applyKnobs(req)
	return b.req, extraFields(req.Extra)
}

// gemBuilder assembles the contents array turn by turn.
//
// It exists because this wire needs something the neutral message does not
// carry: a functionResponse must name the FUNCTION, while a RoleTool message
// only knows the call id. The builder remembers the calls each assistant turn
// announced so every result can be paired back to the name it answers.
type gemBuilder struct {
	req gemRequest
	// pending holds the announced calls that have not been answered yet, in
	// the order the model produced them - which is the order the loop replies
	// in, and therefore the fallback pairing when no id is available.
	pending []ToolCall
}

// appendMessage folds one neutral message into the wire request: system prompts
// hoist to systemInstruction, everything else becomes a content turn.
func (b *gemBuilder) appendMessage(m Message) {
	switch m.Role {
	case RoleSystem:
		// This wire has no system role; the prompt belongs in the top-level
		// systemInstruction Content. A second system message becomes another
		// PART of it rather than being dropped - parts concatenate on the
		// provider's side, so the operator's text survives intact and no string
		// surgery invents a separator.
		if b.req.SystemInstruction == nil {
			b.req.SystemInstruction = &gemContent{Parts: []gemPart{{Text: m.Content}}}
			return
		}
		b.req.SystemInstruction.Parts = append(b.req.SystemInstruction.Parts, gemPart{Text: m.Content})
	case RoleAssistant:
		b.appendAssistant(m)
	case RoleTool:
		b.appendToolResult(m)
	default:
		// RoleUser and anything unknown are caller-supplied content, which
		// this wire carries under the user role - the only role left once
		// system, assistant and tool have been handled above.
		b.appendUserText(m.Content)
	}
}

// appendAssistant adds one model turn: the carrier echo when the provider
// signed this step, the rebuilt text and functionCall parts otherwise.
func (b *gemBuilder) appendAssistant(m Message) {
	// The calls are recorded from the TYPED fields either way: the results that
	// follow need their names, and the raw echo path is opaque on purpose.
	b.pending = append(b.pending, m.ToolCalls...)

	// CONTRACT: a signed step is echoed VERBATIM - the array already holds the
	// text and functionCall parts in the order the model produced them, so
	// rebuilding it here would both duplicate those parts and break the
	// signatures Gemini checks (see gemContent.MarshalJSON).
	//
	// The array check is the one guard: a carrier captured from another wire (a
	// reasoning_content string replayed against this client) cannot be a parts
	// array, and reconstruction is a well-formed request where sending it would
	// be a guaranteed 400.
	if isJSONArray(m.Reasoning) {
		b.req.Contents = append(b.req.Contents, gemContent{Role: geminiRoleModel, PartsRaw: m.Reasoning})
		return
	}

	var parts []gemPart
	if m.Content != "" {
		parts = append(parts, gemPart{Text: m.Content})
	}
	for _, tc := range m.ToolCalls {
		parts = append(parts, gemPart{FunctionCall: &gemFunctionCall{
			ID:   tc.ID,
			Name: tc.Name,
			Args: geminiCallArgs(tc.Arguments),
		}})
	}
	// A turn with no parts is a 400 on this wire, and an empty text part
	// encodes to {} - a shape protobuf-JSON rejects. An assistant message with
	// neither text nor calls carries nothing to send, so it is dropped.
	if len(parts) == 0 {
		return
	}
	b.req.Contents = append(b.req.Contents, gemContent{Role: geminiRoleModel, Parts: parts})
}

// appendToolResult adds one tool output as a functionResponse part.
//
// CONTRACT: parallel results share ONE user turn, in call order. The loop emits
// a RoleTool message per result sequentially; this wire documents the parallel
// answer as several functionResponse parts of a single content, and keeping the
// order deterministic is what makes the request bytes reproducible.
func (b *gemBuilder) appendToolResult(m Message) {
	name, ok := b.takeCall(m.ToolCallID)
	if !ok {
		// An output with no call to answer cannot become a functionResponse:
		// this wire demands the function name, which only the call carries, and
		// an empty name is a guaranteed 400. The output still reaches the model
		// as plain user text - nothing is dropped, and no pairing is invented
		// that the transcript does not support.
		b.appendUserText(m.Content)
		return
	}
	part := gemPart{FunctionResponse: &gemFunctionResponse{
		ID:       m.ToolCallID,
		Name:     name,
		Response: geminiToolResponse(m.Content),
	}}
	if n := len(b.req.Contents); n > 0 && isGemToolResultContent(b.req.Contents[n-1]) {
		b.req.Contents[n-1].Parts = append(b.req.Contents[n-1].Parts, part)
		return
	}
	b.req.Contents = append(b.req.Contents, gemContent{Role: geminiRoleUser, Parts: []gemPart{part}})
}

// appendUserText adds a plain text turn under the user role, dropping an empty
// one (see appendAssistant on why an empty part cannot be sent).
func (b *gemBuilder) appendUserText(text string) {
	if text == "" {
		return
	}
	b.req.Contents = append(b.req.Contents, gemContent{Role: geminiRoleUser, Parts: []gemPart{{Text: text}}})
}

// takeCall consumes the announced call a result answers and returns its
// function name.
//
// An empty id is the 2.5-era models' normal case - they send none - so the
// oldest unanswered call is taken, which is the call the loop is replying to. A
// NON-empty id that matches nothing is not resolved by position: the id is
// evidence about which call this is, and guessing against it would mislabel the
// output as another function's result.
func (b *gemBuilder) takeCall(id string) (string, bool) {
	if len(b.pending) == 0 {
		return "", false
	}
	index := 0
	if id != "" {
		index = slices.IndexFunc(b.pending, func(tc ToolCall) bool { return tc.ID == id })
		if index < 0 {
			return "", false
		}
	}
	name := b.pending[index].Name
	b.pending = slices.Delete(b.pending, index, index+1)
	return name, true
}

// isGemToolResultContent reports whether a turn is one this client built from
// tool results, i.e. one that can absorb the next functionResponse part.
// Checking the first part suffices: such turns only ever contain those.
func isGemToolResultContent(content gemContent) bool {
	return content.Role == geminiRoleUser && len(content.Parts) > 0 &&
		content.Parts[0].FunctionResponse != nil
}

// geminiCallArgs renders a tool call's arguments as the args object.
//
// Absent arguments stay absent (a zero-parameter call is legitimate and
// omitempty keeps the key off the wire). Arguments that are not JSON at all -
// only reachable from a history captured on another wire - travel unchanged so
// the encoder fails the request loudly, rather than silently sending a call the
// model never made.
func geminiCallArgs(args string) json.RawMessage {
	trimmed := strings.TrimSpace(args)
	if trimmed == "" || trimmed == "null" {
		return nil
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, []byte(trimmed)); err != nil {
		return json.RawMessage(trimmed)
	}
	return buf.Bytes()
}

// gemToolOutput wraps a plain tool result, since functionResponse.response must
// be an OBJECT on this wire.
type gemToolOutput struct {
	Output string `json:"output"`
}

// geminiToolResponse renders one tool result as the response object.
//
// A tool that already answers with a JSON object passes through compacted - its
// structure is what the model asked for. Everything else (which is what every
// amele builtin returns: plain text) travels under "output", because a string,
// an array or a number is not a shape this field accepts.
func geminiToolResponse(result string) json.RawMessage {
	trimmed := strings.TrimSpace(result)
	if len(trimmed) > 0 && trimmed[0] == '{' && json.Valid([]byte(trimmed)) {
		var buf bytes.Buffer
		if err := json.Compact(&buf, []byte(trimmed)); err == nil {
			return buf.Bytes()
		}
	}
	// Marshalling a struct of one string cannot fail, so the error is discarded
	// deliberately rather than turned into an unreachable branch.
	wrapped, _ := json.Marshal(gemToolOutput{Output: result})
	return wrapped
}

// applyTools declares the offered tools, sanitizing every parameter schema on
// the way.
//
// CONTRACT: no schema reaches this wire unsanitized. Gemini answers an
// unsupported JSON Schema keyword with a hard 400 that fails the whole request,
// and amele's own fs builtins ship additionalProperties. The stripped-keyword
// report is discarded HERE on purpose: surfacing it (explain, and one warning
// line per run) is Task 5's, and this client prints nothing.
func (out *gemRequest) applyTools(defs []ToolDef) {
	if len(defs) == 0 {
		return
	}
	decls := make([]gemFunctionDecl, 0, len(defs))
	for _, def := range defs {
		clean, _ := SanitizeGeminiSchema(def.Parameters)
		decls = append(decls, gemFunctionDecl{
			Name:        def.Name,
			Description: def.Description,
			Parameters:  clean,
		})
	}
	// One tools entry holding every declaration: this wire groups them that
	// way, unlike the OpenAI-compatible one that lists a tool object per
	// function.
	out.Tools = []gemTool{{FunctionDeclarations: decls}}
}

// applyKnobs maps the cap and sampling knobs into generationConfig, which is
// sent only when it would carry something.
func (out *gemRequest) applyKnobs(req Request) {
	config := gemGenerationConfig{
		MaxOutputTokens: max(req.MaxOutputTokens, 0),
		// Sampling is passed through as given. Google recommends the default
		// 1.0 on Gemini 3 but does not reject other values, so amele sends what
		// the config asked for and `explain` carries the warning.
		Temperature: req.Temperature,
		TopP:        req.TopP,
	}
	if config.empty() {
		return
	}
	out.GenerationConfig = &config
}

// endpoint builds the versioned generateContent URL for one model.
func (c *GeminiClient) endpoint(model string) string {
	base := c.BaseURL
	if base == "" {
		base = defaultGeminiBaseURL
	}
	return strings.TrimSuffix(base, "/") + "/" + geminiAPIVersion + "/models/" + geminiModelPath(model) + ":generateContent"
}

// geminiModelPath renders a model name as ONE URL path segment.
//
// The "models/" prefix is trimmed because the API's own documentation prints
// model ids both ways ("gemini-3-pro" and "models/gemini-3-pro") and a config
// copied from the latter would otherwise request /models/models/... and 404.
//
// SECURITY: what remains is escaped as a single segment, so every separator a
// model name might carry - "/" included - is percent-encoded. A model name can
// then only ever name a model: it cannot append a query string, and it cannot
// climb out of the path this client chose, not even through a "../.." that a
// normalizing proxy would otherwise resolve away. Real model ids carry no
// slash, so single-segment escaping costs nothing; a slash-bearing name (a
// tuned model, which amele does not support today) fails loudly with a 404
// instead of silently addressing another endpoint.
func geminiModelPath(model string) string {
	return url.PathEscape(strings.TrimPrefix(model, "models/"))
}

func (c *GeminiClient) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	// A fresh Client per call is cheap: it shares http.DefaultTransport, so
	// connection pooling is preserved.
	return &http.Client{Timeout: c.requestTimeout()}
}

// requestTimeout returns the effective per-request ceiling, for both the client
// construction and the timeout error message.
func (c *GeminiClient) requestTimeout() time.Duration {
	if c.HTTPClient != nil && c.HTTPClient.Timeout > 0 {
		return c.HTTPClient.Timeout
	}
	if c.RequestTimeout > 0 {
		return c.RequestTimeout
	}
	return defaultRequestTimeout
}

func (c *GeminiClient) sleep(ctx context.Context, d time.Duration) error {
	if c.Sleep != nil {
		return c.Sleep(ctx, d)
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
