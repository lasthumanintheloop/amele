package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
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
type oaRequest struct {
	Model    string      `json:"model"`
	Messages []oaMessage `json:"messages"`
	Tools    []oaTool    `json:"tools,omitempty"`
	// ResponseFormat is a pointer so the key vanishes entirely for plain-text
	// runs: strict gateways reject unknown/null keys they do not implement.
	ResponseFormat *oaResponseFormat `json:"response_format,omitempty"`
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

type oaMessage struct {
	Role       string       `json:"role"`
	Content    string       `json:"content"`
	ToolCalls  []oaToolCall `json:"tool_calls,omitempty"`
	ToolCallID string       `json:"tool_call_id,omitempty"`
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
	wire := c.toWire(req)
	body, err := json.Marshal(wire)
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
		fallbackBody, err = json.Marshal(wire)
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
		snippet, _ := io.ReadAll(io.LimitReader(httpResp.Body, maxErrorBody))
		retryable := httpResp.StatusCode == http.StatusTooManyRequests || httpResp.StatusCode >= 500
		// Double %w: the message is unchanged ("provider error: status N: …")
		// while callers keep both errors.Is(ErrProvider) and errors.As on the
		// typed status.
		statusErr := &statusError{code: httpResp.StatusCode, snippet: strings.TrimSpace(string(snippet))}
		return nil, retryable, parseRetryAfter(httpResp.Header.Get("Retry-After")),
			fmt.Errorf("%w: %w", ErrProvider, statusErr)
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

// toWire converts the neutral request into the OpenAI JSON shape.
func (c *OpenAIClient) toWire(req Request) oaRequest {
	out := oaRequest{Model: req.Model}
	for _, m := range req.Messages {
		wm := oaMessage{Role: m.Role, Content: m.Content, ToolCallID: m.ToolCallID}
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
	return out
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
