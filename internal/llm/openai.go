package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"
)

// maxErrorBody bounds how much of an error response body is echoed into
// error messages, so a misbehaving proxy cannot flood logs.
const maxErrorBody = 2048

// maxResponseBody bounds how much of a SUCCESS response body is read. It is
// generous - an honest completion, tool arguments included, is orders of
// magnitude smaller - because its job is not to police size but to keep a
// hostile or broken endpoint from streaming until the process is OOM-killed
// mid-run. SECURITY: shared by both HTTP clients, the counterpart of
// maxErrorBody on the error path.
const maxResponseBody = 8 << 20 // 8 MiB

// decodeResponseBody decodes a provider's success body under maxResponseBody.
// A body that exceeds the cap is cut mid-JSON and surfaces as a decode error,
// which is exactly how the caller should treat an endpoint it cannot parse.
func decodeResponseBody(body io.Reader, into any) error {
	return json.NewDecoder(io.LimitReader(body, maxResponseBody)).Decode(into)
}

// defaultMaxAttempts is the total number of tries (1 initial + retries) for
// retryable failures (429 and 5xx).
const defaultMaxAttempts = 3

// defaultRequestTimeout bounds a single HTTP round-trip when the config does
// not set provider.request_timeout.
const defaultRequestTimeout = 120 * time.Second

// maxRetryAfter caps how long a provider's Retry-After header can stretch one
// backoff wait. Without a cap a misbehaving proxy could stall the run until
// the run timeout fires - which would then misattribute the provider problem
// to the user's budget (exit 3).
const maxRetryAfter = 60 * time.Second

// OpenAIClient talks to any OpenAI-compatible /chat/completions endpoint.
// That single wire format covers OpenAI, Ollama, vLLM, Groq, OpenRouter and
// most gateways, which is why it is the only transport in Phase 1.
type OpenAIClient struct {
	// BaseURL is the API root without a trailing slash,
	// e.g. "https://api.openai.com/v1".
	BaseURL string
	// Dialect selects the variation of the OpenAI-compatible wire format this
	// endpoint speaks: which field carries the output cap, how the reasoning
	// knob is spelled. The zero value behaves exactly like DialectOpenAI, so a
	// client constructed before dialects existed keeps its behavior.
	Dialect Dialect
	// APIKey is sent as a Bearer token when non-empty.
	APIKey string
	// HTTPClient is injectable for tests; nil means a client bounded by
	// RequestTimeout.
	HTTPClient *http.Client
	// RequestTimeout bounds a single HTTP round-trip when HTTPClient is
	// nil. Zero means defaultRequestTimeout (120s). Wired from the config's
	// provider.request_timeout so slow reasoning models are not condemned
	// to a hardcoded ceiling.
	RequestTimeout time.Duration
	// MaxAttempts overrides defaultMaxAttempts when > 0.
	MaxAttempts int
	// Sleep is injectable for tests; nil means context-aware sleeping.
	// Determinism rule (docs/engineering.md §5.4): time-dependent behavior must be
	// injectable.
	Sleep func(ctx context.Context, d time.Duration) error
}

// Wire types for the OpenAI-compatible JSON body. Kept unexported: the rest
// of the codebase only sees the neutral types in llm.go.
// Every field amele owns is a struct member (not a map entry) so the encoded
// key order is fixed by the declaration order: the wire goldens are then
// byte-stable, and a reviewer reads a diff of the request instead of a diff of
// Go map iteration.
type oaRequest struct {
	Model    string      `json:"model"`
	Messages []oaMessage `json:"messages"`
	Tools    []oaTool    `json:"tools,omitempty"`
	// ResponseFormat is a pointer so the key vanishes entirely for plain-text
	// runs: strict gateways reject unknown/null keys they do not implement.
	ResponseFormat *oaResponseFormat `json:"response_format,omitempty"`
	// MaxTokens and MaxCompletionTokens are the two spellings of the output
	// cap. Exactly one is ever set - capField picks it from the dialect - and
	// both are omitempty so a request that asked for no cap sends neither and
	// inherits the provider's own default.
	MaxTokens           int `json:"max_tokens,omitempty"`
	MaxCompletionTokens int `json:"max_completion_tokens,omitempty"`
	// Temperature and TopP are pointers because 0 is a meaningful sampling
	// value: omitempty on a float64 would silently drop `temperature: 0`, the
	// exact setting a deterministic run asks for.
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
}

// oaResponseFormat is the OpenAI json_schema response format. Only the
// json_schema variant is emitted; the older "json_object" mode is not used
// because it enforces "some JSON" rather than the caller's schema.
type oaResponseFormat struct {
	Type       string       `json:"type"`
	JSONSchema oaJSONSchema `json:"json_schema"`
}

type oaJSONSchema struct {
	Name string `json:"name"`
	// Strict is always true: a non-strict schema is advisory, which would
	// leave the caller's contract unenforced and silently shift the burden
	// onto the validate+retry layer.
	Strict bool            `json:"strict"`
	Schema json.RawMessage `json:"schema"`
}

// oaMessage is one message in both directions: what the provider returns in a
// choice and what amele sends back. The reasoning fields exist on both ends
// for exactly that reason - the payload captured from a response is the
// payload echoed on the next request.
type oaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	// ReasoningContent and ReasoningDetails are the two spellings of the
	// reasoning carrier (the json tags mirror dialect.go's fieldReasoning*
	// constants). At most one is ever set - reasoningRef picks it from the
	// dialect - and both are json.RawMessage so the payload is copied, never
	// decoded and rebuilt: providers sign (Anthropic via OpenRouter) or
	// hash-check (DeepSeek) their reasoning and reject an altered one.
	//
	// The plaintext `reasoning` summary OpenRouter returns beside the array is
	// deliberately absent: it is display sugar, it carries no signature, and
	// echoing it back is not part of any provider's contract.
	//
	// The one transformation the payload does undergo is the encoder's own
	// JSON-level escaping of <, > and & (encoding/json escapes them in every
	// string of every request amele has ever sent, content and tool arguments
	// included). It is value-preserving - the provider decodes the same string
	// it produced, which is what a signature or hash is computed over - so it
	// is left alone rather than turned off globally for this one field.
	ReasoningContent json.RawMessage `json:"reasoning_content,omitempty"`
	ReasoningDetails json.RawMessage `json:"reasoning_details,omitempty"`
	ToolCalls        []oaToolCall    `json:"tool_calls,omitempty"`
	ToolCallID       string          `json:"tool_call_id,omitempty"`
}

// reasoningRef points at the field of msg that carries reasoning for this
// dialect. Capture reads through it and echo writes through it, so the two
// directions cannot drift onto different keys (see reasoningField).
func reasoningRef(d Dialect, msg *oaMessage) *json.RawMessage {
	if reasoningField(d) == fieldReasoningDetails {
		return &msg.ReasoningDetails
	}
	return &msg.ReasoningContent
}

// carriesReasoning reports whether a wire value is a real payload rather than
// an absent one. A field the provider sent as JSON null decodes to a non-nil
// RawMessage holding "null", which would otherwise become a carrier that means
// "no reasoning" in four bytes and get echoed back as an empty field - the
// kind of unknown-shaped input the strict dialects answer with a 400. The
// neutral contract is explicit that nil, and only nil, means "no reasoning"
// (llm.Message.Reasoning).
func carriesReasoning(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) > 0 && !bytes.Equal(trimmed, []byte("null"))
}

type oaToolCall struct {
	ID       string     `json:"id"`
	Type     string     `json:"type"`
	Function oaFunction `json:"function"`
}

type oaFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type oaTool struct {
	Type     string        `json:"type"`
	Function oaFunctionDef `json:"function"`
}

type oaFunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type oaResponse struct {
	Choices []struct {
		Message      oaMessage `json:"message"`
		FinishReason string    `json:"finish_reason"`
	} `json:"choices"`
	// Usage is a pointer so "provider omitted usage entirely" is
	// distinguishable from "zero tokens" - token budgets fail closed on
	// the former (see llm.Response.UsageMissing).
	// int64, not int: a 32-bit build must be able to DECODE an absurd count
	// (json rejects an int overflow outright) so parseUsage can clamp it.
	Usage *struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
	} `json:"usage"`
}

// statusError carries the HTTP status and body snippet of a non-200 reply so
// Chat can route on them programmatically. docs/engineering.md §5.3 bans deciding
// control flow by matching error strings; this typed error is how the
// response_format fallback inspects the failure without re-parsing text.
type statusError struct {
	code    int
	snippet string
}

func (e *statusError) Error() string {
	return fmt.Sprintf("status %d: %s", e.code, e.snippet)
}

// rejectsResponseFormat reports whether this failure looks like "I do not
// support response_format".
//
// This is deliberately a heuristic: OpenAI-compatible gateways share no error
// taxonomy, so the only portable signal is a 400 whose message names the
// offending field. It is conservative in the direction that matters - an
// unrelated 400 (bad model, bad key) never mentions response_format and so
// stays a hard failure instead of being silently downgraded to a schema-less
// request.
//
// The match runs against e.snippet, which is already capped to maxErrorBody
// bytes. Accepted trade-off: a verbose proxy that only names "response_format"
// past that cut-off gets no fallback and surfaces as a hard failure instead -
// preferable to reading an unbounded body just to catch that rare case.
func (e *statusError) rejectsResponseFormat() bool {
	return e.code == http.StatusBadRequest && strings.Contains(e.snippet, "response_format")
}

// Chat implements Provider. It retries 429/5xx responses with exponential
// backoff because transient rate limits are the norm, not the exception, for
// unattended cron runs.
//
// When the request carries a ResponseFormat and the provider rejects it, Chat
// transparently repeats the call once without the field; see the fallback
// comment below.
func (c *OpenAIClient) Chat(ctx context.Context, req Request) (*Response, error) {
	wire, fields := c.toWire(req)
	body, err := encodeBody(wire, fields)
	if err != nil {
		return nil, fmt.Errorf("%w: encoding request: %v", ErrProvider, err)
	}

	// fallbackBody is the same request with response_format stripped, built
	// up-front only when a schema was actually requested. Capability is
	// rediscovered on every Chat call rather than cached on the client:
	// per-call state keeps OpenAIClient free of global mutable state
	// (docs/engineering.md §5.1), and the cost - one extra 400 round-trip - is paid
	// only by the misconfigured combination of a schema and a provider that
	// cannot honor it.
	var fallbackBody []byte
	if wire.ResponseFormat != nil {
		wire.ResponseFormat = nil
		fallbackBody, err = encodeBody(wire, fields)
		if err != nil {
			return nil, fmt.Errorf("%w: encoding fallback request: %v", ErrProvider, err)
		}
	}

	attempts := c.MaxAttempts
	if attempts <= 0 {
		attempts = defaultMaxAttempts
	}

	var lastErr error
	var retryAfter time.Duration
	// dropped remembers that the fallback fired: every response produced
	// after that point - the fallback response itself and any later retry in
	// this Chat call - was generated without native schema enforcement and
	// must say so (Response.SchemaEnforcementDropped).
	dropped := false
	for attempt := 1; attempt <= attempts; attempt++ {
		if attempt > 1 {
			// Exponential backoff: 1s, 2s, 4s... stretched (never shrunk)
			// to the provider's Retry-After when one was sent, and capped
			// by the caller's context deadline (sleep aborts when ctx is
			// done).
			delay := time.Duration(1<<(attempt-2)) * time.Second
			if retryAfter > delay {
				delay = min(retryAfter, maxRetryAfter)
			}
			if err := c.sleep(ctx, delay); err != nil {
				return nil, fmt.Errorf("%w: %v (last error: %v)", ErrProvider, err, lastErr)
			}
		}

		resp, retryable, ra, err := c.doOnce(ctx, body)
		if shouldFallbackToPlain(err, fallbackBody) {
			// Capability discovery, not a transient failure: the provider
			// will reject the field just as firmly on the next attempt, so
			// the schema-less repeat happens immediately, inside this same
			// attempt. It therefore consumes no MaxAttempts budget (which is
			// reserved for rate limits and 5xx) and no backoff sleep. Setting
			// fallbackBody to nil bounds it to exactly one extra round-trip
			// per Chat call; the stripped body is then used for any remaining
			// retries too.
			body, fallbackBody = fallbackBody, nil
			dropped = true
			resp, retryable, ra, err = c.doOnce(ctx, body)
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
// failure is worth retrying; retryAfter carries the provider's Retry-After
// wish (0 when absent).
func (c *OpenAIClient) doOnce(ctx context.Context, body []byte) (resp *Response, retryable bool, retryAfter time.Duration, err error) {
	url := strings.TrimSuffix(c.BaseURL, "/") + "/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, false, 0, fmt.Errorf("%w: building request: %v", ErrProvider, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
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
		retryable, retryAfter, err := statusFailure(httpResp)
		return nil, retryable, retryAfter, err
	}

	var wire oaResponse
	if err := decodeResponseBody(httpResp.Body, &wire); err != nil {
		return nil, false, 0, fmt.Errorf("%w: decoding response: %v", ErrProvider, err)
	}
	if len(wire.Choices) == 0 {
		return nil, false, 0, fmt.Errorf("%w: response has no choices", ErrProvider)
	}

	choice := wire.Choices[0]
	msg := Message{Role: choice.Message.Role, Content: choice.Message.Content}
	if msg.Role == "" {
		msg.Role = RoleAssistant
	}
	// CONTRACT: capturing the reasoning payload is UNCONDITIONAL dialect
	// behavior, not a consequence of the config having asked for reasoning.
	// DeepSeek thinks BY DEFAULT ("a config that never mentions reasoning
	// still receives reasoning_content and MUST echo it back in tool loops
	// (400 otherwise)" - research §"Load-bearing quirks" #2), and Kimi K3 and
	// GLM-5.3 cannot be turned off at all. A client that only captured when a
	// reasoning knob was set would break the plainest config there is.
	// The bytes are stored as they arrived; nothing here parses them.
	if raw := *reasoningRef(c.Dialect, &choice.Message); carriesReasoning(raw) {
		msg.Reasoning = raw
	}
	for _, tc := range choice.Message.ToolCalls {
		msg.ToolCalls = append(msg.ToolCalls, ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		})
	}

	resp = &Response{Message: msg, FinishReason: choice.FinishReason}
	if wire.Usage != nil {
		// CONTRACT: provider counts are sanitized here, at the parse boundary,
		// so the loop's budget arithmetic only ever sees non-negative bounded
		// values (see parseUsage).
		usage, trustworthy := parseUsage(wire.Usage.PromptTokens, wire.Usage.CompletionTokens)
		resp.Usage = usage
		resp.UsageMissing = !trustworthy
	} else {
		resp.UsageMissing = true
	}
	return resp, false, 0, nil
}

// statusFailure turns a non-200 reply into the typed provider error, reporting
// whether the failure is worth retrying and the provider's Retry-After wish.
// The response body is read here (bounded to maxErrorBody) and not by the
// caller, so the two cannot disagree about who consumed it.
func statusFailure(httpResp *http.Response) (retryable bool, retryAfter time.Duration, err error) {
	snippet, _ := io.ReadAll(io.LimitReader(httpResp.Body, maxErrorBody))
	retryable = httpResp.StatusCode == http.StatusTooManyRequests || httpResp.StatusCode >= 500
	// Double %w: the message is unchanged ("provider error: status N: …")
	// while callers keep both errors.Is(ErrProvider) and errors.As on the
	// typed status.
	statusErr := &statusError{code: httpResp.StatusCode, snippet: strings.TrimSpace(string(snippet))}
	err = fmt.Errorf("%w: %w", ErrProvider, statusErr)
	// A recognized 400 keeps its message and gains a hint at the end
	// ("… — set provider.reasoning.effort: none …"). The advice is appended
	// HERE rather than inside statusError.Error() so that the anthropic client,
	// which shares the type, keeps its own (different) signatures; and so the
	// typed error stays a pure carrier of what the wire said. Nothing else
	// changes: retryable, Retry-After and the errors.Is/As behavior are the
	// same with or without a match.
	if advice := adviceFor(statusErr); advice != "" {
		err = fmt.Errorf("%w — %s", err, advice)
	}
	return retryable, parseRetryAfter(httpResp.Header.Get("Retry-After")), err
}

// shouldFallbackToPlain reports whether a failed attempt must be repeated once
// without response_format. fallback is nil when there is nothing to strip or
// when the single permitted fallback has already been spent, so this is also
// what bounds the fallback to one extra round-trip per Chat call.
func shouldFallbackToPlain(err error, fallback []byte) bool {
	if err == nil || fallback == nil {
		return false
	}
	var se *statusError
	return errors.As(err, &se) && se.rejectsResponseFormat()
}

// parseRetryAfter reads the seconds form of a Retry-After header. The
// HTTP-date form is deliberately ignored (it would need an injected clock for
// determinism, and providers overwhelmingly send seconds); absent or
// unparseable values fall back to the exponential backoff.
func parseRetryAfter(header string) time.Duration {
	secs, err := strconv.Atoi(strings.TrimSpace(header))
	if err != nil || secs <= 0 {
		return 0
	}
	return time.Duration(secs) * time.Second
}

// toWire converts the neutral request into the OpenAI JSON shape for this
// client's dialect. It returns the struct-encoded part of the body and the
// body-root fragments that are merged after marshalling: the dialect's
// reasoning fields (whose KEY depends on the dialect, which a Go struct cannot
// express) and the caller's raw provider.params.
//
// CONTRACT: params keys cannot collide with the fields amele owns - config
// validation rejects that at exit 2 - so merging them needs no further
// defense here.
func (c *OpenAIClient) toWire(req Request) (oaRequest, map[string]json.RawMessage) {
	out := oaRequest{Model: req.Model}
	for _, m := range req.Messages {
		wm := oaMessage{Role: m.Role, Content: m.Content, ToolCallID: m.ToolCallID}
		// CONTRACT: the echo is UNCONDITIONAL too, and byte-verbatim. Every
		// dialect that returns reasoning demands it back on every subsequent
		// request of a tool loop (DeepSeek and Kimi answer a missing echo with
		// an error; GLM documents it as required-unmodified; OpenRouter
		// requires the block sequence to match "the outputs generated by the
		// model during the original request" - research §matrix "Echo-back in
		// tool loop"). The carrier is passed straight through: it is never
		// decoded, re-encoded, reordered or truncated here.
		//
		// Only assistant messages carry it. A carrier on any other role is a
		// bug upstream, and echoing it would hand a strict provider an unknown
		// field on a message shape that never has one.
		if m.Role == RoleAssistant && carriesReasoning(m.Reasoning) {
			*reasoningRef(c.Dialect, &wm) = m.Reasoning
		}
		for _, tc := range m.ToolCalls {
			wm.ToolCalls = append(wm.ToolCalls, oaToolCall{
				ID:       tc.ID,
				Type:     "function",
				Function: oaFunction{Name: tc.Name, Arguments: tc.Arguments},
			})
		}
		out.Messages = append(out.Messages, wm)
	}
	for _, t := range req.Tools {
		out.Tools = append(out.Tools, oaTool{Type: "function", Function: oaFunctionDef(t)})
	}
	if rf := req.ResponseFormat; rf != nil {
		out.ResponseFormat = &oaResponseFormat{
			Type: "json_schema",
			JSONSchema: oaJSONSchema{
				Name:   rf.Name,
				Strict: true,
				Schema: rf.Schema,
			},
		}
	}
	if req.MaxOutputTokens > 0 {
		if capField(c.Dialect) == "max_tokens" {
			out.MaxTokens = req.MaxOutputTokens
		} else {
			out.MaxCompletionTokens = req.MaxOutputTokens
		}
	}
	// Sampling is passed through on every dialect. The only target that
	// forbids it (kimi's fixed K-series values) is stopped at validate, so a
	// value that reaches here was asked for deliberately and is never dropped
	// silently by amele.
	out.Temperature = req.Temperature
	out.TopP = req.TopP

	// MapReasoning allocates a fresh map per call, so extending it with the
	// raw params below cannot leak into another request.
	var fields map[string]json.RawMessage
	if req.Reasoning != nil {
		fields = MapReasoning(c.Dialect, *req.Reasoning).Fields
	}
	for key, value := range req.Extra {
		if fields == nil {
			fields = make(map[string]json.RawMessage, len(req.Extra))
		}
		fields[key] = value
	}
	return out, fields
}

// encodeBody renders one request body: the struct-encoded fields first, in
// declaration order, then the merged fragments.
func encodeBody(wire oaRequest, fields map[string]json.RawMessage) ([]byte, error) {
	body, err := json.Marshal(wire)
	if err != nil {
		return nil, fmt.Errorf("encoding request: %w", err)
	}
	return mergeBodyFields(body, fields)
}

// mergeBodyFields appends pre-serialized fragments to the root of an already
// encoded JSON object.
//
// Hand-editing JSON is normally a smell; it is the right tool here because the
// KEY of a dialect's reasoning field and of every provider.params entry is
// data, not a Go type, and re-encoding the whole body through a map would give
// up the stable key order the goldens (and reviewable request diffs) depend
// on. The keys are merged in sorted order for that same reason - Go map
// iteration is randomized - and every value is passed through json.Compact, so
// an unparseable fragment fails the request here instead of reaching the
// provider as a malformed body it answers with an opaque 400.
func mergeBodyFields(body []byte, fields map[string]json.RawMessage) ([]byte, error) {
	if len(fields) == 0 {
		return body, nil
	}
	if len(body) < 2 || body[0] != '{' || body[len(body)-1] != '}' {
		return nil, fmt.Errorf("merging request fields: body is not a JSON object")
	}
	out := make([]byte, 0, len(body)+64)
	out = append(out, body[:len(body)-1]...)
	for _, key := range slices.Sorted(maps.Keys(fields)) {
		var compact bytes.Buffer
		if err := json.Compact(&compact, fields[key]); err != nil {
			return nil, fmt.Errorf("merging request field %q: %w", key, err)
		}
		// An object that is still empty ends with '{' and takes no separator.
		if out[len(out)-1] != '{' {
			out = append(out, ',')
		}
		encodedKey, err := json.Marshal(key)
		if err != nil {
			return nil, fmt.Errorf("merging request field %q: %w", key, err)
		}
		out = append(out, encodedKey...)
		out = append(out, ':')
		out = append(out, compact.Bytes()...)
	}
	return append(out, '}'), nil
}

func (c *OpenAIClient) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	// A fresh Client per call is cheap: it shares http.DefaultTransport, so
	// connection pooling is preserved.
	return &http.Client{Timeout: c.requestTimeout()}
}

// requestTimeout returns the effective per-request ceiling, for both the
// client construction and the timeout error message.
func (c *OpenAIClient) requestTimeout() time.Duration {
	if c.HTTPClient != nil && c.HTTPClient.Timeout > 0 {
		return c.HTTPClient.Timeout
	}
	if c.RequestTimeout > 0 {
		return c.RequestTimeout
	}
	return defaultRequestTimeout
}

func (c *OpenAIClient) sleep(ctx context.Context, d time.Duration) error {
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
