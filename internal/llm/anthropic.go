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
	"strings"
	"time"
)

// defaultAnthropicBaseURL is the first-party Anthropic API host. The request
// path (/v1/messages) is appended by the client, so BaseURL never includes
// /v1 - unlike the OpenAI client, whose BaseURL convention carries it.
const defaultAnthropicBaseURL = "https://api.anthropic.com"

// anthropicVersion is the pinned API version header. Anthropic versions the
// wire format through this header rather than the URL; pinning it keeps the
// request shape stable regardless of server-side default changes.
const anthropicVersion = "2023-06-01"

// defaultAnthropicMaxOutput is the max_tokens sent when the config does not
// set one. Anthropic requires max_tokens on every request (there is no
// server-side default), so the client must always choose a number; 8192 is
// large enough for tool-calling agent turns without inviting runaway output.
const defaultAnthropicMaxOutput = 8192

// AnthropicClient talks to the Anthropic Messages API natively
// (POST /v1/messages). It implements Provider.
//
// The client speaks raw HTTP instead of using the official SDK on purpose:
// the single-binary constitution forbids heavyweight dependencies, only one
// small endpoint is needed, and hand-rolled wire types keep this client
// consistent with the house pattern established by OpenAIClient (typed
// statusError routing, injected Sleep, capped error snippets).
//
// Request.ResponseFormat is IGNORED: this API path has no native json_schema
// response enforcement, so no schema is sent to the provider. The
// validate+retry layer above the loop remains the enforcement for
// output.schema - and so that nothing is SILENTLY lost, every response to a
// schema-carrying request is marked with Response.SchemaEnforcementDropped,
// which callers surface as a warning.
type AnthropicClient struct {
	// BaseURL is the API root without a trailing slash and WITHOUT /v1,
	// e.g. "https://api.anthropic.com". Empty means the first-party host.
	BaseURL string
	// APIKey is sent as the x-api-key header when non-empty.
	APIKey string
	// HTTPClient is injectable for tests; nil means a client bounded by
	// RequestTimeout.
	HTTPClient *http.Client
	// RequestTimeout bounds a single HTTP round-trip when HTTPClient is
	// nil. Zero means defaultRequestTimeout (120s). Wired from the config's
	// provider.request_timeout, same semantics as OpenAIClient.
	RequestTimeout time.Duration
	// MaxAttempts overrides defaultMaxAttempts when > 0.
	MaxAttempts int
	// Sleep is injectable for tests; nil means context-aware sleeping.
	// Determinism rule (docs/engineering.md §5.4): time-dependent behavior must be
	// injectable.
	Sleep func(ctx context.Context, d time.Duration) error
	// MaxOutputTokens is the per-request max_tokens value. Anthropic
	// requires the field on every request; 0 means
	// defaultAnthropicMaxOutput.
	MaxOutputTokens int
}

// Wire types for the Anthropic Messages API JSON body. Kept unexported: the
// rest of the codebase only sees the neutral types in llm.go.
type anRequest struct {
	Model     string      `json:"model"`
	MaxTokens int         `json:"max_tokens"`
	System    string      `json:"system,omitempty"`
	Messages  []anMessage `json:"messages"`
	Tools     []anTool    `json:"tools,omitempty"`
}

type anMessage struct {
	Role    string    `json:"role"`
	Content []anBlock `json:"content"`
}

// anBlock is one content block. A single struct covers the three block types
// this client emits (text, tool_use, tool_result); omitempty keeps the
// irrelevant fields off the wire for each type.
type anBlock struct {
	Type string `json:"type"`
	// Text is set on "text" blocks.
	Text string `json:"text,omitempty"`
	// ID, Name and Input are set on "tool_use" blocks.
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
	// ToolUseID and Content are set on "tool_result" blocks.
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   string `json:"content,omitempty"`
}

type anTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type anResponse struct {
	Content []struct {
		Type  string          `json:"type"`
		Text  string          `json:"text"`
		ID    string          `json:"id"`
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	// Usage is a pointer so "provider omitted usage entirely" is
	// distinguishable from "zero tokens" - token budgets fail closed on
	// the former (see llm.Response.UsageMissing).
	// int64, not int: a 32-bit build must be able to DECODE an absurd count
	// (json rejects an int overflow outright) so parseUsage can clamp it.
	Usage *struct {
		InputTokens  int64 `json:"input_tokens"`
		OutputTokens int64 `json:"output_tokens"`
	} `json:"usage"`
}

// Chat implements Provider. It retries 429, 5xx and 529 (Anthropic's
// "overloaded" status) with exponential backoff, honoring Retry-After.
//
// The retry loop is a deliberate copy of OpenAIClient.Chat's rather than a
// shared helper: the two clients evolve independently (the OpenAI one carries
// a response_format capability-discovery fallback this one will never need),
// and extracting the ~20 shared lines would couple their futures for no
// robustness gain.
func (c *AnthropicClient) Chat(ctx context.Context, req Request) (*Response, error) {
	body, err := json.Marshal(c.toWire(req))
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
		if err == nil {
			// A schema was requested but never sent natively (see the type
			// comment), so by definition this response was produced without
			// provider-side enforcement - flag it so callers can warn.
			resp.SchemaEnforcementDropped = req.ResponseFormat != nil
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
func (c *AnthropicClient) doOnce(ctx context.Context, body []byte) (resp *Response, retryable bool, retryAfter time.Duration, err error) {
	base := c.BaseURL
	if base == "" {
		base = defaultAnthropicBaseURL
	}
	url := strings.TrimSuffix(base, "/") + "/v1/messages"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, false, 0, fmt.Errorf("%w: building request: %v", ErrProvider, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("anthropic-version", anthropicVersion)
	if c.APIKey != "" {
		// SECURITY: Anthropic authenticates via x-api-key, not a Bearer
		// token; sending the key in the wrong header would leak it to
		// intermediaries expecting no credential there.
		httpReq.Header.Set("x-api-key", c.APIKey)
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
		// 429 and every 5xx are transient; that band includes 529, the
		// non-standard "overloaded" status Anthropic documents as retryable.
		retryable := httpResp.StatusCode == http.StatusTooManyRequests || httpResp.StatusCode >= 500
		// Double %w: the message is unchanged ("provider error: status N: …")
		// while callers keep both errors.Is(ErrProvider) and errors.As on the
		// typed status (docs/engineering.md §5.3 bans string matching for control flow).
		statusErr := &statusError{code: httpResp.StatusCode, snippet: strings.TrimSpace(string(snippet))}
		return nil, retryable, parseRetryAfter(httpResp.Header.Get("Retry-After")),
			fmt.Errorf("%w: %w", ErrProvider, statusErr)
	}

	var wire anResponse
	if err := decodeResponseBody(httpResp.Body, &wire); err != nil {
		return nil, false, 0, fmt.Errorf("%w: decoding response: %v", ErrProvider, err)
	}

	msg := Message{Role: RoleAssistant}
	for _, block := range wire.Content {
		switch block.Type {
		case "text":
			// Multiple text blocks concatenate: the neutral Message carries
			// one text body, and Anthropic guarantees block order.
			msg.Content += block.Text
		case "tool_use":
			msg.ToolCalls = append(msg.ToolCalls, ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: compactJSONObject(block.Input),
			})
		}
		// Other block types (e.g. thinking) are ignored: the loop only
		// consumes text and tool calls, and dropping unknown blocks is the
		// forward-compatible reading of a versioned wire format.
	}

	resp = &Response{Message: msg, FinishReason: mapStopReason(wire.StopReason)}
	if wire.Usage != nil {
		// CONTRACT: same sanitizing boundary as the OpenAI client - the loop
		// must never accumulate a negative or unbounded provider count.
		usage, trustworthy := parseUsage(wire.Usage.InputTokens, wire.Usage.OutputTokens)
		resp.Usage = usage
		resp.UsageMissing = !trustworthy
	} else {
		resp.UsageMissing = true
	}
	return resp, false, 0, nil
}

// mapStopReason translates Anthropic stop reasons into the OpenAI-compatible
// finish reasons the loop understands. Unknown values pass through verbatim:
// the loop's badFinish path already handles unrecognized reasons defensively,
// and inventing a translation here would only hide new provider states.
func mapStopReason(stopReason string) string {
	switch stopReason {
	case "end_turn":
		return "stop"
	case "max_tokens":
		return "length"
	case "model_context_window_exceeded":
		// CONTRACT: context-window exhaustion (Claude 4.5+) is a truncation
		// like max_tokens. It must map to "length" so the loop hard-fails the
		// run; the passthrough default would land it in badFinish's
		// unknown-reason branch, which accepts non-empty content - letting a
		// truncated answer exit 0 in unattended cron runs.
		return "length"
	case "refusal":
		return "content_filter"
	case "tool_use":
		return "tool_calls"
	default:
		return stopReason
	}
}

// compactJSONObject normalizes a tool_use input into the compact JSON string
// the neutral ToolCall carries. Absent or null input becomes "{}" so tools
// always receive a parseable object.
func compactJSONObject(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return "{}"
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		// Malformed provider JSON is deferred to the tool layer, mirroring
		// the OpenAI client: parsing tool arguments is the tool's job so the
		// model can recover from its own bad output.
		return string(raw)
	}
	return buf.String()
}

// toWire converts the neutral request into the Anthropic Messages JSON shape:
// the system prompt moves to the top-level "system" field, assistant tool
// calls become tool_use content blocks, and consecutive RoleTool messages
// merge into a single user message of tool_result blocks.
func (c *AnthropicClient) toWire(req Request) anRequest {
	maxTokens := c.MaxOutputTokens
	if maxTokens <= 0 {
		maxTokens = defaultAnthropicMaxOutput
	}
	out := anRequest{Model: req.Model, MaxTokens: maxTokens}

	for _, m := range req.Messages {
		switch m.Role {
		case RoleSystem:
			// Anthropic rejects a system role inside messages; the prompt
			// belongs in the top-level "system" field. Joining defends
			// against a caller supplying several system messages.
			if out.System == "" {
				out.System = m.Content
			} else {
				out.System += "\n\n" + m.Content
			}
		case RoleTool:
			block := anBlock{Type: "tool_result", ToolUseID: m.ToolCallID, Content: m.Content}
			// CONTRACT: Anthropic requires all parallel tool results in a
			// single user turn. The loop emits one RoleTool message per
			// result sequentially, so the client merges consecutive ones
			// into the previous tool-result user message.
			if n := len(out.Messages); n > 0 && isToolResultMessage(out.Messages[n-1]) {
				out.Messages[n-1].Content = append(out.Messages[n-1].Content, block)
			} else {
				out.Messages = append(out.Messages, anMessage{Role: RoleUser, Content: []anBlock{block}})
			}
		case RoleAssistant:
			var blocks []anBlock
			if m.Content != "" {
				blocks = append(blocks, anBlock{Type: "text", Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				input := json.RawMessage(tc.Arguments)
				// Anthropic requires input to be a JSON object; the model
				// occasionally emits no arguments for zero-parameter tools.
				if len(input) == 0 {
					input = json.RawMessage("{}")
				}
				blocks = append(blocks, anBlock{Type: "tool_use", ID: tc.ID, Name: tc.Name, Input: input})
			}
			out.Messages = append(out.Messages, anMessage{Role: RoleAssistant, Content: blocks})
		default:
			out.Messages = append(out.Messages, anMessage{
				Role:    m.Role,
				Content: []anBlock{{Type: "text", Text: m.Content}},
			})
		}
	}

	for _, t := range req.Tools {
		out.Tools = append(out.Tools, anTool{Name: t.Name, Description: t.Description, InputSchema: t.Parameters})
	}
	return out
}

// isToolResultMessage reports whether msg is a user message produced by the
// tool-result merging above, i.e. one that can absorb the next tool_result
// block. Checking the first block suffices: merged messages only ever contain
// tool_result blocks.
func isToolResultMessage(msg anMessage) bool {
	return msg.Role == RoleUser && len(msg.Content) > 0 && msg.Content[0].Type == "tool_result"
}

func (c *AnthropicClient) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	// A fresh Client per call is cheap: it shares http.DefaultTransport, so
	// connection pooling is preserved.
	return &http.Client{Timeout: c.requestTimeout()}
}

// requestTimeout returns the effective per-request ceiling, for both the
// client construction and the timeout error message.
func (c *AnthropicClient) requestTimeout() time.Duration {
	if c.HTTPClient != nil && c.HTTPClient.Timeout > 0 {
		return c.HTTPClient.Timeout
	}
	if c.RequestTimeout > 0 {
		return c.RequestTimeout
	}
	return defaultRequestTimeout
}

func (c *AnthropicClient) sleep(ctx context.Context, d time.Duration) error {
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
