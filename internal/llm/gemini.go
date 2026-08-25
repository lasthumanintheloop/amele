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

// vertexAPIVersion is the pinned path version on the Vertex side, which is a
// DIFFERENT number from the AI Studio one above.
//
// v1 is the GA surface and its GenerateContentRequest is field-for-field
// identical to v1beta1's (verified by diffing the two discovery documents -
// docs/superpowers/specs/2026-08-25-vertex-adc-research.md §1.3), so beta buys
// nothing here that would justify a pre-GA path in an unattended run.
const vertexAPIVersion = "v1"

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
// SCOPE: this client speaks conversations, tool calls, the reasoning and
// structured-output knobs, and round-trips the thought signatures Gemini 3
// signs its steps with, against EITHER backend of this API - AI Studio by
// default, Vertex AI when Vertex is set. The two differ in the endpoint and the
// credential; the request body they receive is byte-identical (see the vertex
// note on gemRequest).
type GeminiClient struct {
	// BaseURL is the API root without a trailing slash and WITHOUT the version
	// segment, e.g. "https://generativelanguage.googleapis.com". Empty means
	// the first-party host.
	//
	// In vertex mode it means less than that: only its scheme and host are
	// read, because the rest of the URL is derived from the project and
	// location (see vertexEndpoint).
	BaseURL string
	// APIKey is sent as the x-goog-api-key header when non-empty. It is the AI
	// Studio credential and is NEVER sent in vertex mode - that endpoint
	// refuses API keys outright.
	APIKey string
	// Vertex, when non-nil, switches this client to the Vertex AI endpoint:
	// project- and location-addressed, API version v1, OAuth bearer auth from
	// TokenSource. nil is AI Studio.
	Vertex *VertexTarget
	// TokenSource supplies the OAuth access token for vertex mode. It is
	// required there and unused otherwise.
	TokenSource GeminiTokenSource
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

// VertexTarget names the Vertex AI deployment a request is addressed to: the
// Google Cloud project that owns the quota, and the location that serves it.
//
// It is a value rather than a set of fields on the client because the two
// coordinates are meaningless apart - neither addresses a model on its own -
// and because a nil *VertexTarget is how the client says "AI Studio".
type VertexTarget struct {
	// Project is the project id or project number. It becomes a URL path
	// segment.
	Project string
	// Location is the Vertex location: a region ("us-central1"), a
	// jurisdictional multi-region ("us", "eu"), or "global". It selects the
	// host AND appears in the path.
	Location string
}

// Host returns the Vertex API host that serves this target's location, the
// value a base_url override would replace. It is exported for the report
// commands: `amele explain` must be able to name the host a run will reach
// without building a request or duplicating the mapping.
func (t VertexTarget) Host() string { return vertexHost(t.Location) }

// Endpoint returns the full generateContent URL this target and model resolve
// to, with baseURL overriding the host exactly as it does on a real request
// (empty means the host Host reports). The error is the same one a run would
// fail with - a project or location that cannot become a URL segment, or a
// base_url that is not an absolute http(s) URL.
//
// CONTRACT: it is the client's OWN address, not a second copy of the mapping.
// `amele explain` prints the answer of the very function Chat builds its
// request with, so the report cannot promise a destination the run will not
// reach. That guarantee is the whole reason this is exported rather than
// re-derived in internal/explain.
func (t VertexTarget) Endpoint(baseURL, model string) (string, error) {
	return vertexEndpoint(baseURL, t, model)
}

// GeminiTokenSource supplies the OAuth access token a Vertex request
// authenticates with.
//
// The interface is declared here, in the package that CONSUMES it (the house
// rule, docs/engineering.md §5.1), and it is deliberately the smallest thing
// this client needs: one method, one string. Everything the credential machine
// actually involves - the ADC chain, a service-account key file, token caching
// and refresh - lives behind it, which is what keeps this file free of a
// dependency on any of it.
//
// CONTRACT for implementations: Token returns a token that is valid NOW
// (caching and refresh are the source's business), respects the context, and is
// safe for concurrent use. It is called once per HTTP attempt, so a retried
// request picks up a token refreshed in between rather than replaying an
// expired one.
type GeminiTokenSource interface {
	// Token returns the bearer token to authenticate one request with.
	Token(ctx context.Context) (string, error)
}

// GeminiQuotaProject is the OPTIONAL companion to GeminiTokenSource: a token
// source whose credential needs the quota-project header implements it, and a
// source that does not simply omits the method.
//
// It is a second interface rather than a second method on GeminiTokenSource
// because the header is a property of the CREDENTIAL, not of the wire: only the
// `authorized_user` legs of the ADC chain need it, every other credential is
// harmed by it (it demands serviceusage.services.use on the named project), and
// a fake token source in a test has no opinion at all. Keeping the required
// interface at one method is also what lets the simplest possible stub satisfy
// it.
//
// CONTRACT: QuotaProject is consulted AFTER Token on the same request, so a
// source that resolves its credential lazily has already decided by the time it
// is asked. It returns "" when no header should be sent, and it must be safe
// for concurrent use.
type GeminiQuotaProject interface {
	// QuotaProject returns the project id for the x-goog-user-project header,
	// or "" for no header.
	QuotaProject() string
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
//
// # Vertex body diffs
//
// The same struct is sent to BOTH backends, byte for byte
// (TestVertexBodyIsIdenticalToAIStudio). The AI Studio and Vertex bodies are
// not identical in general - the research names five differences - but every
// one of them concerns a field amele does not write, so the diff table is
// recorded here as a comment rather than implemented as a mode switch. Each row
// is INERT, and each says what would make it live:
//
//   - topK - typed float on Vertex, integer on AI Studio. amele exposes no
//     topK knob at all; adding one means emitting it as a float in vertex mode.
//   - responseFormat - an ARRAY of ResponseFormat on Vertex, a single object on
//     AI Studio; a shared struct would break outright. amele's structured
//     output travels as generationConfig.responseJsonSchema, which is shared
//     and identical, and the live smoke settled that it is the right field.
//   - safetySettings.method - Vertex-only third key on each entry. amele writes
//     no safetySettings (the key is OWNED, so provider.params cannot supply one
//     either), so there is nothing for method to attach to.
//   - labels - Vertex-only, for billing attribution. amele does not own it, so
//     it is reachable through provider.params exactly like any other extra; on
//     AI Studio the same key would be a 400, which is the operator's own
//     trade-off to make.
//   - serviceTier - AI Studio-only request field, likewise not owned and
//     likewise reachable through params, where it would 400 on Vertex.
//
// On the response side the pair is usageMetadata.trafficType (Vertex) vs
// serviceTier (AI Studio); this client decodes neither - it reads only the
// token counts, which carry the same names on both.
//
// The request tree diverges at more than its root - Part (§2.6) and Tool (§2.7)
// each carry backend-only keys too - and amele writes none of those either: its
// parts are text/functionCall/functionResponse/thought/thoughtSignature and its
// only tool kind is functionDeclarations, all shared.
//
// Source: docs/superpowers/specs/2026-08-25-vertex-adc-research.md §2 and §6.
// TestGeminiWireFieldsPinTheVertexDiffTable fails when a field is added to ANY
// struct this body is assembled from (request, generationConfig, thinkingConfig,
// content, part, functionCall, functionResponse, tool, functionDeclaration), so
// the next knob cannot skip this table.
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
	// ResponseMimeType and ResponseJSONSchema are the structured-output pair -
	// the schema alone does nothing without the JSON mime type - and
	// ThinkingConfig is the reasoning knob. Their declaration order fixes the
	// object's key order, and therefore every golden.
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

// gemThinking is the reasoning control object (mapGeminiThinking maps the
// neutral knob onto it). ThinkingBudget is a pointer because 0 is the
// meaningful "turn thinking off" value on the 2.5-era models, which omitempty
// would drop.
//
// CONTRACT: never both fields at once - the API answers a request carrying a
// level AND a budget with a 400 (the signature table explains that one too).
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
// CONTRACT: these are STRING HEURISTICS by necessity (design doc §"Error-
// signature detection") and the fixtures in gemini_reasoning_test.go are what
// pin them. They are safe in a way a general string match is not, because they
// change NOTHING but the human-facing text: no retry, no downgrade, no request
// rewrite. A signature that stops matching (Google reworded its 400) costs a
// hint, never correctness.
//
// Order matters - first match wins - so the generic unknown-field entry comes
// last: the specific failures below it are all unknown-field or invalid-value
// 400s too, and a generic "remove it from provider.params" would be the wrong
// door for a key amele itself wrote.
var geminiErrorSignatures = []errorSignature{
	{
		// A signed step reached the API without its signature. amele echoes
		// the raw parts array precisely so this cannot happen
		// (gemContent.MarshalJSON), which makes this 400 evidence of a bug
		// HERE rather than a config mistake - so the advice asks for a report
		// instead of naming a knob the operator could turn.
		//
		// The backticks are part of the match: they are how the API quotes the
		// field, and requiring them keeps the entry off a 400 that merely
		// mentions signatures in prose.
		match:  func(e *statusError) bool { return strings.Contains(e.snippet, "missing a `thought_signature`") },
		advice: "amele must echo signatures automatically - this is a bug, please report it with the session log",
	},
	{
		// Both thinking fields in one request. Unreachable through validate,
		// which refuses effort+budget (exit 2), but reachable when the second
		// field came from provider.params - so the advice names the config
		// pair rather than the wire fields.
		match: func(e *statusError) bool {
			return strings.Contains(e.snippet, "thinkingLevel") && strings.Contains(e.snippet, "thinkingBudget")
		},
		advice: "set only one of provider.reasoning.effort or budget_tokens",
	},
	{
		// The Gemini 3 half of the effort: none mapping. Thinking cannot be
		// switched off on that generation, and validate cannot know the model
		// generation from the name (docs/engineering.md's no-model-name-tables
		// rule), so the runtime 400 is where this becomes explainable.
		match:  func(e *statusError) bool { return matchesGeminiCannotDisableThinking(e.snippet) },
		advice: "this model cannot disable thinking; remove reasoning.effort: none",
	},
	{
		// The catch-all for protobuf-JSON strictness: this API rejects any key
		// it does not know, and the only key amele sends that it did not
		// choose itself is a provider.params entry. The advice says "if",
		// because the same 400 also names a field a future API version
		// dropped.
		match:  func(e *statusError) bool { return strings.Contains(e.snippet, "Unknown name") },
		advice: "the gemini API rejects unknown fields; if this key came from provider.params, remove it",
	},
}

// matchesGeminiCannotDisableThinking reports whether body is a "this model
// always thinks" 400, in either wording the API uses: the numeric complaint
// about a zero budget, and the prose form.
//
// The match is case-folded because the two phrasings differ in exactly that
// (a sentence-initial "Thinking" versus "cannot disable thinking" mid-message),
// and the prose form additionally requires the word "thinking" so an unrelated
// "cannot be disabled" cannot claim the hint.
func matchesGeminiCannotDisableThinking(body string) bool {
	lower := strings.ToLower(body)
	if strings.Contains(lower, "budget 0 is invalid") {
		return true
	}
	if strings.Contains(lower, "cannot disable thinking") {
		return true
	}
	return strings.Contains(lower, "thinking") && strings.Contains(lower, "cannot be disabled")
}

// rejectsResponseJSONSchema reports whether this failure looks like "I do not
// support responseJsonSchema".
//
// The gemini counterpart of rejectsResponseFormat, and a heuristic for a
// sharper reason than the OpenAI one: the design doc records an OPEN QUESTION
// about this very field - the SDK documents responseJsonSchema while a newer
// nested responseFormat shape appears elsewhere - and the live smoke (Task 6)
// is what decides it. Until then the client sends the documented form and
// treats a 400 naming it as capability discovery, so a wrong guess degrades to
// validate+retry instead of failing every structured-output run.
//
// Both spellings are matched: the wire is camelCase, but this API's error text
// echoes the protobuf field name in snake_case ("Unknown name
// \"response_json_schema\"") depending on which form the request used.
func (e *statusError) rejectsResponseJSONSchema() bool {
	return e.code == http.StatusBadRequest &&
		(strings.Contains(e.snippet, "responseJsonSchema") || strings.Contains(e.snippet, "response_json_schema"))
}

// Chat implements Provider. It retries 429 and 5xx with exponential backoff,
// honoring the retry delay this API carries in the error BODY rather than in a
// Retry-After header (design doc §"Gemini-specific mechanics" item 3).
//
// When the request carries a ResponseFormat and the API rejects the schema
// field, Chat repeats the call once without it; see the fallback comment below.
//
// The retry loop mirrors the other two clients' rather than sharing a helper:
// they evolve independently - each carries its own capability fallback for the
// field its wire spells differently - and extracting the ~20 shared lines would
// couple their futures for no robustness gain. What IS shared is the machinery
// underneath: backoffDelay, statusFailure, shouldFallback and encodeBody.
func (c *GeminiClient) Chat(ctx context.Context, req Request) (*Response, error) {
	wire, fields := c.toWire(req)
	body, err := encodeBody(wire, fields)
	if err != nil {
		return nil, fmt.Errorf("%w: encoding request: %v", ErrProvider, err)
	}

	// fallbackBody is the same request with the schema stripped, built up-front
	// only when one was actually requested. Capability is rediscovered per Chat
	// call rather than cached on the client (no global mutable state,
	// docs/engineering.md §5.1); the cost - one extra 400 round-trip - is paid
	// only by the combination of a schema and an endpoint that cannot honor it.
	//
	// The JSON mime type deliberately SURVIVES the strip: JSON mode alone still
	// gets an answer the validate+retry layer above can check, and the 400 this
	// fallback reacts to names the schema field, not the mime type.
	//
	// dropped remembers that every response produced after this point was
	// generated without provider-native schema enforcement
	// (Response.SchemaEnforcementDropped). A format that carried no schema is
	// known to be in that state before the first request - plain JSON mode
	// constrains nothing - so it is flagged immediately and there is nothing to
	// probe for, the same reading the OpenAI client applies to its json_object
	// dialects.
	var fallbackBody []byte
	dropped := false
	if req.ResponseFormat != nil {
		dropped = wire.GenerationConfig == nil || len(wire.GenerationConfig.ResponseJSONSchema) == 0
		if !dropped {
			// Mutating the already-encoded struct is safe: body holds its bytes.
			wire.GenerationConfig.ResponseJSONSchema = nil
			fallbackBody, err = encodeBody(wire, fields)
			if err != nil {
				return nil, fmt.Errorf("%w: encoding fallback request: %v", ErrProvider, err)
			}
		}
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
		if shouldFallback(err, fallbackBody, (*statusError).rejectsResponseJSONSchema) {
			// Capability discovery, not a transient failure: the API will
			// reject the field just as firmly on the next attempt, so the
			// schema-less repeat happens immediately, inside this same attempt.
			// It therefore consumes no MaxAttempts budget (reserved for rate
			// limits and 5xx) and no backoff sleep. Clearing fallbackBody bounds
			// it to exactly one extra round-trip per Chat call; the stripped
			// body is then used for any remaining retries too.
			body, fallbackBody = fallbackBody, nil
			dropped = true
			resp, retryable, ra, err = c.doOnce(ctx, req.Model, body)
		}
		if err == nil {
			resp.SchemaEnforcementDropped = dropped
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
	endpoint, err := c.endpoint(model)
	if err != nil {
		// A target that cannot be addressed will not become addressable on the
		// next attempt, so this is not retryable.
		return nil, false, 0, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, false, 0, fmt.Errorf("%w: building request: %v", ErrProvider, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if err := c.authorize(ctx, httpReq); err != nil {
		// Same reasoning: a missing or unobtainable credential is a wiring
		// problem, not a transient one. Failing here also means the prompt is
		// never spent on a request the endpoint would refuse.
		return nil, false, 0, err
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

// authorize sets the credential header for one request. Which credential that
// is follows from the backend, and the two are mutually exclusive.
//
// SECURITY: the AI Studio key never travels to Vertex and the bearer token
// never travels to AI Studio. The two backends authenticate differently and
// each rejects the other's credential, so sending the wrong one would leak it
// to a service that has no business seeing it - which is also why config
// refuses a config naming both (exit 2), and why this reads the mode rather
// than "whatever is set".
//
// The documented ?key= query-parameter form is refused on both paths: a URL is
// logged by every proxy between here and the API.
func (c *GeminiClient) authorize(ctx context.Context, req *http.Request) error {
	if c.Vertex == nil {
		if c.APIKey != "" {
			req.Header.Set("x-goog-api-key", c.APIKey)
		}
		return nil
	}
	if c.TokenSource == nil {
		return fmt.Errorf("%w: vertex needs google credentials but no token source is configured", ErrProvider)
	}
	token, err := c.TokenSource.Token(ctx)
	if err != nil {
		return fmt.Errorf("%w: obtaining a google access token: %v", ErrProvider, err)
	}
	if token == "" {
		// An empty token would go out as a bare "Bearer " and come back a 401
		// that describes the endpoint rather than the credential source.
		return fmt.Errorf("%w: the google credential source returned an empty access token", ErrProvider)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	// The quota-project header is asked for AFTER the token, which is the
	// ordering GeminiQuotaProject's contract promises: a source that resolves
	// the ADC chain lazily only learns its credential type by producing a
	// token. Sources that do not implement the interface send no header, which
	// is the right default - see quotaProjectFor in gemini_auth.go.
	if quota, ok := c.TokenSource.(GeminiQuotaProject); ok {
		if project := quota.QuotaProject(); project != "" {
			req.Header.Set("x-goog-user-project", project)
		}
	}
	return nil
}

// geminiBackendSignature is the backend-aware counterpart of errorSignature: an
// entry that may key on the status code, on the body, AND on which of this
// wire's two backends the request went to.
//
// The two tables are separate because they answer different questions.
// geminiErrorSignatures holds body heuristics consulted for 400s on BOTH
// backends through the shared adviceFor, whose 400 gate is what keeps a generic
// phrase like "Unknown name" off a 401 on the OpenAI and Anthropic wires as
// well. The entries here are about credentials and addressing: they need the
// statuses that gate rules out, and every one of them is mode-dependent - a 401
// on the AI Studio backend is about an api_key and has no Google Cloud project
// to name, and a 404 there has no location to blame.
type geminiBackendSignature struct {
	// match reports whether this entry explains the failure. vertex is the
	// client's target, nil on the AI Studio backend; body is the (possibly
	// truncated) error body statusFailure already read.
	match func(code int, body string, vertex *VertexTarget) bool
	// advice renders the hint. CONTRACT: it is called only after match
	// returned true for the same vertex value, so an entry whose match
	// requires a non-nil target may dereference it here.
	advice func(vertex *VertexTarget) string
}

// geminiBackendSignatures is the ordered table consulted for a failure the
// shared 400 gate cannot reach. First match wins.
//
// CONTRACT: like every signature table, these entries change NOTHING but the
// human-facing text - no retry, no rewrite. What they add is the vocabulary the
// fix lives in: the live 401 body says only "Request is missing required
// authentication credential" (research §2.10), which names neither the IAM
// role, nor the project, nor the credential sources amele searched.
var geminiBackendSignatures = []geminiBackendSignature{
	{
		// Express-mode keys. Google documents an API-key form of the Vertex
		// endpoint; the live service answers it with this 401 for both
		// :generateContent and :streamGenerateContent (research §3.3), which
		// is the generic "this method has no API-key auth" reply rather than
		// "bad key". amele can only arrive here on the KEYED backend - config
		// refuses api_key next to a vertex block, and vertex mode never sends
		// x-goog-api-key - so this is an AI Studio config aimed at a Vertex
		// host, and the fix is a vertex block, not another key.
		match: func(code int, body string, vertex *VertexTarget) bool {
			return code == http.StatusUnauthorized && vertex == nil &&
				strings.Contains(body, "API keys are not supported by this API")
		},
		advice: func(*VertexTarget) string {
			return "vertex requires OAuth credentials; api keys are not supported (express-mode keys are rejected " +
				"live too) - replace provider.api_key with a provider.vertex block naming project and location"
		},
	},
	{
		match: func(code int, _ string, vertex *VertexTarget) bool {
			return code == http.StatusUnauthorized && vertex != nil
		},
		advice: func(t *VertexTarget) string {
			return "vertex rejected the credential: check that it resolves (provider.vertex.credentials, " +
				"GOOGLE_APPLICATION_CREDENTIALS, or `gcloud auth application-default login`) and that its principal " +
				"has roles/aiplatform.user on project " + t.Project
		},
	},
	{
		match: func(code int, _ string, vertex *VertexTarget) bool {
			return code == http.StatusForbidden && vertex != nil
		},
		advice: func(t *VertexTarget) string {
			return "vertex authenticated the credential but refused the call: grant roles/aiplatform.user on project " +
				t.Project + ", confirm the aiplatform.googleapis.com API is enabled there and that the model is " +
				"served in location " + t.Location +
				", and - for gcloud user credentials - serviceusage.services.use for the x-goog-user-project header"
		},
	},
	{
		// A 404 on this endpoint is an ADDRESSING answer, and the API's body
		// says only that the publisher model was not found. The two candidates
		// are a wrong model id and a model that is simply not served where the
		// request was sent: region support is per-model and far narrower than
		// the 47-region host list (research §1.5 - gemini-3.5-flash is not
		// served in us-central1 at all, and two Gemini 3 models are
		// global-only). The advice names both, and repeats that the location
		// is the operator's to change: amele will not reroute it.
		match: func(code int, _ string, vertex *VertexTarget) bool {
			return code == http.StatusNotFound && vertex != nil
		},
		advice: func(t *VertexTarget) string {
			return "vertex has no such model at that address: check the model id, and that the model is served in " +
				"location " + t.Location + " (availability is per-model and narrower than the region list - some " +
				"models are served only on `global`). amele never reroutes a configured location"
		},
	},
}

// geminiBackendAdvice returns the hint for one failure, or "" when no entry
// claims it.
func geminiBackendAdvice(code int, body string, vertex *VertexTarget) string {
	for _, sig := range geminiBackendSignatures {
		if sig.match(code, body, vertex) {
			return sig.advice(vertex)
		}
	}
	return ""
}

// failure turns a non-200 reply into the typed provider error, adding the two
// things this wire does differently: advice that depends on which backend the
// request went to (geminiBackendSignatures), and a retry delay that arrives in
// the BODY as a google.rpc.RetryInfo detail rather than in a Retry-After
// header.
func (c *GeminiClient) failure(httpResp *http.Response) (*Response, bool, time.Duration, error) {
	// Shared with the other clients: same retry band (429 and every 5xx), same
	// typed statusError, same header reading - only the error-signature table
	// and the body-borne delay below differ.
	retryable, retryAfter, err := statusFailure(httpResp, geminiErrorSignatures)
	// One errors.As for both gemini-specific steps. The snippet statusFailure
	// already read is the ONLY copy of the body - it consumed the reader - so
	// everything below reads it from the typed error rather than from the
	// response.
	var se *statusError
	if errors.As(err, &se) {
		// The backend-aware advice is appended here rather than through the
		// shared table because it depends on the BACKEND and on statuses the
		// shared 400 gate excludes, neither of which statusFailure can see.
		if advice := geminiBackendAdvice(se.code, se.snippet, c.Vertex); advice != "" {
			err = fmt.Errorf("%w — %s", err, advice)
		}
		if retryable && retryAfter == 0 {
			// This wire carries the retry delay in the BODY as a
			// google.rpc.RetryInfo detail, not in a Retry-After header. A body
			// truncated at maxErrorBody no longer parses as JSON and simply
			// yields no wish - the exponential ladder then stands, which is the
			// same outcome as an endpoint that sent no RetryInfo at all.
			var env gemErrorEnvelope
			if json.Unmarshal([]byte(se.snippet), &env) == nil {
				retryAfter = parseGoogleRetryDelay(env.Error.Details)
			}
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
// Two reasons become ERRORS instead. The API answers 200 with them, so nothing
// below this client would notice the turn failed: the loop would take whatever
// text arrived - or an empty answer - and stop as if the task were done, which
// in an unattended cron run means exit 0 on a broken turn.
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
	case "MISSING_THOUGHT_SIGNATURE":
		// CONTRACT: this is the 200-shaped form of the failure the signature
		// table explains on the 4xx path - the API can report it EITHER way, and
		// only the error table would see the 4xx one. amele echoes the raw parts
		// array precisely so neither can happen (gemContent.MarshalJSON), which
		// makes this amele's bug rather than the operator's; and a turn that
		// carried a plausible-looking answer alongside it would otherwise exit 0.
		return "", fmt.Errorf("%w: the model's step is missing a thought signature (finish reason MISSING_THOUGHT_SIGNATURE); amele echoes thought signatures automatically - this is a bug; please report it with the session log", ErrProvider)
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
		// A schema the sanitizer emptied (an MCP tool that describes its input
		// purely through $ref/$defs, say) declares NO parameters key at all
		// rather than an empty Schema object. Both readings are guesses about
		// what a type-less Schema means to the service; the absent key is the
		// one already known to work - it is what a genuinely argument-less tool
		// sends - and this wire punishes an unexpected shape with a 400 that
		// fails every other tool in the request too.
		if isEmptyJSONObject(clean) {
			clean = nil
		}
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

// geminiJSONMimeType is the responseMimeType that turns on JSON mode. The
// schema field does nothing without it: the pair is what constrains decoding.
const geminiJSONMimeType = "application/json"

// applyKnobs maps the cap, sampling, structured-output and reasoning knobs into
// generationConfig, which is sent only when it would carry something.
func (out *gemRequest) applyKnobs(req Request) {
	config := gemGenerationConfig{
		MaxOutputTokens: max(req.MaxOutputTokens, 0),
		// Sampling is passed through as given. Google recommends the default
		// 1.0 on Gemini 3 but does not reject other values, so amele sends what
		// the config asked for and `explain` carries the warning
		// (GeminiSamplingNote).
		Temperature: req.Temperature,
		TopP:        req.TopP,
	}
	if req.ResponseFormat != nil {
		config.applyResponseFormat(*req.ResponseFormat)
	}
	if req.Reasoning != nil {
		config.ThinkingConfig = mapGeminiThinking(*req.Reasoning)
	}
	if config.empty() {
		return
	}
	out.GenerationConfig = &config
}

// applyResponseFormat turns on JSON mode and hands the schema to the API
// verbatim.
//
// CONTRACT: the schema is NOT sanitized, unlike a tool's parameters. Those are
// an OpenAPI-3.0 subset that 400s on an unknown keyword; a RESPONSE schema is
// read as JSON Schema and tolerates the keywords amele's own output.schema
// carries, so stripping them here would weaken the constraint the operator
// asked for. The format's Name has no field on this wire at all - only the
// OpenAI-compatible response_format names its schema.
//
// An empty schema sends the mime type alone: that is plain JSON mode, which the
// validate+retry layer above can still work with, whereas an empty
// responseJsonSchema would be a shape this API rejects.
//
// TODO(#17): the design doc records an OPEN QUESTION here - the SDK documents
// responseJsonSchema while a newer nested responseFormat shape appears
// elsewhere. The documented form is what ships; the live smoke (plan task 6) is
// the decider, and until then a 400 naming the field degrades through the
// fallback in Chat rather than failing the run.
func (g *gemGenerationConfig) applyResponseFormat(format ResponseFormat) {
	g.ResponseMimeType = geminiJSONMimeType
	if len(bytes.TrimSpace(format.Schema)) == 0 {
		return
	}
	g.ResponseJSONSchema = format.Schema
}

// mapGeminiThinking translates the neutral reasoning knob into this wire's
// thinkingConfig object, or nil when the config asked for nothing (the model's
// own default then stands). It is a pure function - the same spec always yields
// the same object - which is what lets GeminiReasoningNotes describe a request
// before it is sent without reading the spec a second time.
//
// The level vocabulary here is low/medium/high, a SUBSET of the neutral union,
// so xhigh and max round DOWN to high - the opposite direction from the
// openai-wire dialects, which round up. There is nothing above high on this
// wire to round to, and the rounding is reported in the notes rather than
// hidden.
//
// CONTRACT: a BudgetTokens wins over an Effort, and the two never travel
// together - the API rejects a request carrying both fields. The combination is
// already a validate error (exit 2), so this precedence is a defense against an
// impossible spec, not a reachable policy; the dropped effort is reported by
// GeminiReasoningNotes so it can never be silent.
func mapGeminiThinking(spec ReasoningSpec) *gemThinking {
	if spec.BudgetTokens > 0 {
		budget := spec.BudgetTokens
		return &gemThinking{ThinkingBudget: &budget}
	}
	switch spec.Effort {
	case "":
		return nil
	case effortNone:
		// "Thinking off" is spelled as a zero budget on this wire. It works on
		// the 2.5-era models and 400s on Gemini 3, which cannot disable
		// thinking - a difference validate cannot see (it would have to key on
		// the model NAME), so the error-signature table explains it at runtime.
		zero := 0
		return &gemThinking{ThinkingBudget: &zero}
	default:
		return &gemThinking{ThinkingLevel: geminiThinkingLevel(spec.Effort)}
	}
}

// geminiThinkingLevel rounds the neutral effort onto the low/medium/high levels
// this wire knows. Unknown values pass through untouched: the vocabulary is
// validated in config, and inventing a level for an unrecognized one would hide
// a typo the API's own 400 names precisely.
func geminiThinkingLevel(effort string) string {
	switch effort {
	case "xhigh", "max":
		return "high"
	default:
		return effort
	}
}

// endpoint builds the versioned generateContent URL for one model, on whichever
// backend this client is pointed at.
//
// It returns an error only in vertex mode, where the URL is assembled from
// operator-supplied coordinates that must be checked before they become a
// hostname; the AI Studio path cannot fail.
func (c *GeminiClient) endpoint(model string) (string, error) {
	if c.Vertex != nil {
		return vertexEndpoint(c.BaseURL, *c.Vertex, model)
	}
	base := c.BaseURL
	if base == "" {
		base = defaultGeminiBaseURL
	}
	return strings.TrimSuffix(base, "/") + "/" + geminiAPIVersion + "/models/" + geminiModelPath(model) + ":generateContent", nil
}

// vertexEndpoint builds the Vertex AI generateContent URL:
//
//	https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent
//
// Three host shapes exist and the location picks between them (research §1.1):
// the regional prefix above, the un-prefixed aiplatform.googleapis.com for
// "global", and aiplatform.{us|eu}.rep.googleapis.com for the two
// jurisdictional multi-regions. The path segment stays locations/{location} in
// all three - the host and the path are not alternatives, they travel together.
//
// SECURITY: the configured location is NEVER rewritten. Not to "global" when a
// region cannot serve the requested model, not to a default when it looks
// unfamiliar, not when base_url moves the host. The location decides where the
// prompt is PROCESSED, which is a data-residency commitment an operator makes
// deliberately (research §7); silently substituting another one is a compliance
// break rather than a convenience, and it is exactly the bug filed against
// gemini-cli as google-gemini/gemini-cli#27984 ("Vertex AI with API key ignores
// GOOGLE_CLOUD_LOCATION and uses the global endpoint"). A location that cannot
// serve the model must surface as the API's own loud error.
//
// SECURITY: baseURL overrides the HOST and nothing else - scheme, host and port
// are taken from it, and the resource path is still built here. That is what a
// VPC-SC deployment needs (a restricted VIP or a Private Service Connect DNS
// name in front of the same API, research §7) and it is all it gets: a path
// written in base_url cannot move the request, and config refuses that shape
// outright rather than dropping it quietly.
func vertexEndpoint(baseURL string, target VertexTarget, model string) (string, error) {
	if err := checkVertexID("project", target.Project); err != nil {
		return "", err
	}
	if err := checkVertexID("location", target.Location); err != nil {
		return "", err
	}

	scheme, host := "https", vertexHost(target.Location)
	if baseURL != "" {
		u, err := url.Parse(baseURL)
		if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
			return "", fmt.Errorf("%w: provider base_url %q is not an absolute http(s) URL; in vertex mode it names the host to reach instead of %s", ErrProvider, baseURL, vertexHost(target.Location))
		}
		scheme, host = u.Scheme, u.Host
	}

	// The ids are not percent-escaped: checkVertexID above is the stronger
	// gate. Escaping would turn a hostile value into one odd-looking segment,
	// while the charset check refuses it - and the location has to survive that
	// check anyway, since it also lands in the hostname, where escaping means
	// nothing at all.
	return scheme + "://" + host +
		"/" + vertexAPIVersion +
		"/projects/" + target.Project +
		"/locations/" + target.Location +
		"/publishers/google/models/" + geminiModelPath(model) + ":generateContent", nil
}

// vertexHost maps a location onto the host that serves it.
//
// The two special cases are documented endpoint shapes, not shortcuts: "global"
// is served by the un-prefixed host, and the "us"/"eu" multi-regions by their
// own .rep. names (research §1.1, all three live-probed). Building
// "eu-aiplatform.googleapis.com" for the EU multi-region would be a DNS failure
// on the one location an EU-residency deployment must be able to name.
//
// Anything else is treated as a region and gets the regional prefix. Vertex has
// 45 of them and the list moves; validating against a snapshot here would age
// into refusing a region Google had already launched, while an unknown one
// fails immediately and unmistakably at DNS.
func vertexHost(location string) string {
	switch location {
	case "global":
		return "aiplatform.googleapis.com"
	case "us", "eu":
		return "aiplatform." + location + ".rep.googleapis.com"
	default:
		return location + "-aiplatform.googleapis.com"
	}
}

// ValidVertexID reports whether a string may be used as a Vertex project or
// location: non-empty, lowercase letters, digits and hyphens, not starting with
// a hyphen.
//
// CONTRACT: this is the SINGLE definition of that charset. internal/config
// calls it for provider.vertex.project/location so that validate and the client
// can never disagree - a config that passes `amele validate` must not fail its
// CONFIGURATION at run time (docs/engineering.md §7), which a second copy of
// this rule would eventually break by drifting.
//
// SECURITY: the location is interpolated into the endpoint HOSTNAME, so a value
// carrying a dot, a slash or an @ could send a whole request - prompt included
// - to a server nobody configured; both values are also URL path segments. This
// check is what lets vertexEndpoint assemble the URL by concatenation, and
// percent-escaping is no substitute for it: escaping has no meaning inside a
// hostname. A hyphen may not lead because it cannot start a DNS label.
//
// The charset costs nothing real: Google project ids and locations are
// lowercase-alphanumeric with hyphens by construction, and a project NUMBER is
// digits.
func ValidVertexID(value string) bool {
	if value == "" {
		return false
	}
	for i, r := range value {
		alnum := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9')
		if alnum || (r == '-' && i > 0) {
			continue
		}
		return false
	}
	return true
}

// checkVertexID turns ValidVertexID into the client's typed error, separating
// the two reasons a coordinate can be unusable so the message says which.
//
// Config validation applies the same rule through the same function, so
// reaching this error means a caller constructed the client directly; it stays
// because the alternative failure is silent misrouting.
func checkVertexID(kind, value string) error {
	if value == "" {
		return fmt.Errorf("%w: vertex %s is empty: a vertex request is addressed by project AND location", ErrProvider, kind)
	}
	if !ValidVertexID(value) {
		return fmt.Errorf("%w: vertex %s %q is not addressable: lowercase letters, digits and hyphens only", ErrProvider, kind, value)
	}
	return nil
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
