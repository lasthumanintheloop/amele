package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
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
// Request.ResponseFormat is sent NATIVELY as output_config.format
// (json_schema is GA on the Messages API), so a schema-carrying response is
// not flagged. An endpoint that answers 400 naming output_config - an
// Anthropic-compatible gateway that never implemented it - gets one repeat
// without the field, and only those responses carry
// Response.SchemaEnforcementDropped, which callers surface as a warning while
// the validate+retry layer above the loop enforces output.schema.
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
	// InitialBackoff is the wait before the second attempt, doubled for every
	// attempt after that. Zero means defaultInitialBackoff (1s). Wired from
	// the config's provider.retry.initial_backoff, same semantics as
	// OpenAIClient.
	InitialBackoff time.Duration
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
// Every field amele owns is a struct member (not a map entry) so the encoded
// key order is fixed by the declaration order and the wire goldens are
// byte-stable; the caller's raw provider.params are merged afterwards
// (see encodeBody).
//
// AnthropicOwnedWireFields lists these same keys for the params collision
// check - keep the two in step when a field is added here.
type anRequest struct {
	Model     string      `json:"model"`
	MaxTokens int         `json:"max_tokens"`
	System    string      `json:"system,omitempty"`
	Messages  []anMessage `json:"messages"`
	Tools     []anTool    `json:"tools,omitempty"`
	// Thinking and OutputConfig are pointers so the keys vanish entirely when
	// the config asked for nothing: this API rejects unknown AND unexpected
	// fields (research §matrix "Unknown request fields"), and a model
	// generation that does not know a thinking shape answers it with a 400.
	Thinking     *anThinking     `json:"thinking,omitempty"`
	OutputConfig *anOutputConfig `json:"output_config,omitempty"`
	// Temperature and TopP are pointers because 0 is a meaningful sampling
	// value: omitempty on a float64 would silently drop `temperature: 0`, the
	// exact setting a deterministic run asks for.
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
}

// AnthropicOwnedWireFields returns the request-body keys the native Messages
// API client writes itself - OwnedWireFields for a WIRE rather than a dialect,
// since the dialect is not consulted on that path.
//
// CONTRACT: the list mirrors anRequest's json tags, which is what makes it the
// right answer for provider.params collisions on this wire (config.
// validateParams). Note the difference from every openai-wire dialect: thinking
// and output_config ARE written here, and response_format is not.
func AnthropicOwnedWireFields() []string {
	return []string{"model", "max_tokens", "system", "messages", "tools", "thinking", "output_config", "temperature", "top_p"}
}

// anThinking is the thinking control object. Two shapes share it because they
// target two model generations: {"type":"adaptive"} (current models, with the
// depth in output_config.effort) and {"type":"enabled","budget_tokens":N} (the
// legacy shape, Haiku 4.5 and older). {"type":"disabled"} turns thinking off.
//
// CONTRACT: the shapes are NOT interchangeable - "adaptive" is a 400 on <=4.5
// and the legacy "enabled" is a 400 on 4.7+ (research §"Load-bearing quirks"
// #3). amele picks the shape from what the config asked for (a budget means
// the legacy target) and never from the model name, which churns; the
// mismatch surfaces as the API's own 400.
type anThinking struct {
	Type string `json:"type"`
	// BudgetTokens is set only by the legacy shape. omitempty keeps it off the
	// adaptive and disabled objects, which reject it.
	BudgetTokens int `json:"budget_tokens,omitempty"`
}

// anOutputConfig is the single object carrying BOTH the reasoning depth and
// the native structured-output request. One object, two independent keys: a
// request may set either or both, and both spellings are GA on the Messages
// API (research §matrix "Reasoning knob" / "response_format").
type anOutputConfig struct {
	Effort string          `json:"effort,omitempty"`
	Format *anOutputFormat `json:"format,omitempty"`
}

// anOutputFormat is the native json_schema enforcement request.
type anOutputFormat struct {
	Type   string          `json:"type"`
	Schema json.RawMessage `json:"schema"`
}

type anMessage struct {
	Role    string    `json:"role"`
	Content []anBlock `json:"content"`
	// ContentRaw, when non-nil, REPLACES Content: it is the content array
	// exactly as the provider sent it, echoed back verbatim (see MarshalJSON
	// and toWire). It never appears as a field of its own on the wire.
	ContentRaw json.RawMessage `json:"-"`
}

// MarshalJSON implements json.Marshaler.
//
// CONTRACT: this is the byte-exact echo path. When ContentRaw is set the
// message is rendered with those bytes as its content array instead of
// re-encoding the decoded blocks, because Anthropic SIGNS thinking blocks and
// rejects a modified or reordered array with a 400 (research §"Load-bearing
// quirks" #3). Passing the raw region through means nothing here can reorder,
// re-escape or drop a signature - not even a field this client does not know.
//
// The one transformation the payload undergoes is the encoder's own compaction
// and its JSON-level escaping of <, > and &: value-preserving (the provider
// decodes the same strings it produced, which is what a signature is computed
// over), and the same trade-off the OpenAI client documents on its carrier.
func (m anMessage) MarshalJSON() ([]byte, error) {
	if m.ContentRaw != nil {
		return json.Marshal(struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}{Role: m.Role, Content: m.ContentRaw})
	}
	// A defined type without anMessage's methods: marshalling it directly
	// would recurse into this function forever.
	type plain anMessage
	return json.Marshal(plain(m))
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
	// Content stays RAW at decode time and is parsed into blocks separately
	// (anContentBlocks). The raw region is what the echo contract needs: a
	// response carrying thinking blocks travels back as these exact bytes, so
	// the decoder must not be the only thing that ever sees them.
	Content    json.RawMessage `json:"content"`
	StopReason string          `json:"stop_reason"`
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

// anResponseBlock is one decoded content block of a response. Only the fields
// the loop consumes are named; a thinking block's own payload (its text,
// signature or encrypted data) is deliberately NOT among them, because it
// travels back through the raw array and nothing here may depend on its shape.
type anResponseBlock struct {
	Type  string          `json:"type"`
	Text  string          `json:"text"`
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

// The response block types this client acts on. Thinking blocks are recognized
// only to decide that the array must be carried back; they are never parsed.
const (
	blockText             = "text"
	blockToolUse          = "tool_use"
	blockThinking         = "thinking"
	blockRedactedThinking = "redacted_thinking"
)

// Chat implements Provider. It retries 429, 5xx and 529 (Anthropic's
// "overloaded" status) with exponential backoff, honoring Retry-After.
//
// The retry loop is a deliberate copy of OpenAIClient.Chat's rather than a
// shared helper: the two clients evolve independently - each carries its own
// capability-discovery fallback for the field its wire spells differently
// (response_format there, output_config here) - and extracting the ~20 shared
// lines would couple their futures for no robustness gain. What IS shared is
// the machinery both fallbacks stand on: shouldFallback, statusFailure and
// encodeBody.
func (c *AnthropicClient) Chat(ctx context.Context, req Request) (*Response, error) {
	wire, fields := c.toWire(req)
	body, err := encodeBody(wire, fields)
	if err != nil {
		return nil, fmt.Errorf("%w: encoding request: %v", ErrProvider, err)
	}

	// fallbackBody is the same request with output_config stripped, built
	// up-front only when a schema was actually requested. Capability is
	// rediscovered on every Chat call rather than cached on the client:
	// per-call state keeps the client free of global mutable state
	// (docs/engineering.md §5.1), and the cost - one extra 400 round-trip - is
	// paid only by an endpoint that cannot honor the field.
	//
	// CONTRACT: the WHOLE object goes, not just its format key. An endpoint
	// that rejects output_config rejects it just as firmly when it carries
	// only an effort, so stripping the field named in the 400 is the only
	// fallback that can succeed; a co-present effort is dropped with it, while
	// the thinking object - which still carries the on/off decision - stays.
	var fallbackBody []byte
	if wire.OutputConfig != nil && wire.OutputConfig.Format != nil {
		wire.OutputConfig = nil
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
	// dropped remembers that the fallback fired: every response produced after
	// that point - the fallback response itself and any later retry in this
	// Chat call - was generated without native schema enforcement and must say
	// so (Response.SchemaEnforcementDropped).
	dropped := false
	for attempt := 1; attempt <= attempts; attempt++ {
		if attempt > 1 {
			// Exponential backoff from InitialBackoff (1s, 2s, 4s... by
			// default), stretched to the provider's Retry-After when one was
			// sent, and bounded by the caller's context deadline (sleep aborts
			// when ctx is done).
			if err := c.sleep(ctx, backoffDelay(c.InitialBackoff, attempt, retryAfter)); err != nil {
				return nil, fmt.Errorf("%w: %v (last error: %v)", ErrProvider, err, lastErr)
			}
		}

		resp, retryable, ra, err := c.doOnce(ctx, body)
		if shouldFallback(err, fallbackBody, (*statusError).rejectsOutputConfig) {
			// Capability discovery, not a transient failure: the endpoint will
			// reject the field just as firmly on the next attempt, so the
			// stripped repeat happens immediately, inside this same attempt. It
			// therefore consumes no MaxAttempts budget (reserved for rate limits
			// and 5xx) and no backoff sleep. Setting fallbackBody to nil bounds
			// it to exactly one extra round-trip per Chat call; the stripped body
			// is then used for any remaining retries too.
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
		// Shared with the OpenAI client: same retry band (429 and every 5xx,
		// which includes 529, the non-standard "overloaded" status Anthropic
		// documents as retryable), same typed statusError, same Retry-After
		// reading - only the error-signature table differs.
		retryable, retryAfter, err := statusFailure(httpResp, anthropicErrorSignatures)
		return nil, retryable, retryAfter, err
	}

	var wire anResponse
	if err := decodeResponseBody(httpResp.Body, &wire); err != nil {
		return nil, false, 0, fmt.Errorf("%w: decoding response: %v", ErrProvider, err)
	}
	msg, err := anAssistantMessage(wire.Content)
	if err != nil {
		return nil, false, 0, fmt.Errorf("%w: decoding response: %v", ErrProvider, err)
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

// anAssistantMessage turns the raw content array of one response into the
// neutral assistant message: text blocks concatenate, tool_use blocks become
// tool calls, and the array itself becomes the reasoning carrier when it holds
// thinking blocks.
func anAssistantMessage(content json.RawMessage) (Message, error) {
	blocks, err := anContentBlocks(content)
	if err != nil {
		return Message{}, err
	}
	msg := Message{Role: RoleAssistant}
	thinking := false
	for _, block := range blocks {
		switch block.Type {
		case blockText:
			// Multiple text blocks concatenate: the neutral Message carries
			// one text body, and Anthropic guarantees block order.
			msg.Content += block.Text
		case blockToolUse:
			msg.ToolCalls = append(msg.ToolCalls, ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: compactJSONObject(block.Input),
			})
		case blockThinking, blockRedactedThinking:
			thinking = true
		}
		// Any other block type is ignored HERE while still riding along in the
		// carrier below: the loop consumes text and tool calls, and dropping
		// unknown blocks from the neutral message is the forward-compatible
		// reading of a versioned wire format.
	}
	// CONTRACT: the carrier is the ENTIRE raw content array, not the thinking
	// blocks alone. Anthropic requires the blocks back byte-exact, in the
	// original order and interleaved with the text and tool_use blocks they
	// were produced with (research §"Load-bearing quirks" #3), so the array is
	// the unit that round-trips - which also makes "thinking and
	// redacted_thinking always travel together" automatic rather than a rule
	// this code has to remember. Nothing here parses the payload.
	if thinking {
		msg.Reasoning = content
	}
	return msg, nil
}

// anContentBlocks decodes the raw content array into the blocks the loop
// consumes. An absent or null content field is not an error - it is an empty
// turn - but a content field that is not an array is: this client cannot form
// a message from it, and failing here names the response instead of producing
// a silently empty answer.
func anContentBlocks(raw json.RawMessage) ([]anResponseBlock, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, nil
	}
	var blocks []anResponseBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, fmt.Errorf("content: %w", err)
	}
	return blocks, nil
}

// rejectsOutputConfig reports whether this failure looks like "I do not
// support output_config".
//
// The Anthropic counterpart of rejectsResponseFormat, and a heuristic for the
// same reason: the Anthropic-compatible gateways (DeepSeek, GLM, Kimi) share
// no error taxonomy with the first-party API, so the only portable signal is a
// 400 whose message names the offending field. It is conservative in the
// direction that matters - an unrelated 400 (bad model, bad key, a rejected
// thinking shape) never mentions output_config and so stays a hard failure
// instead of being silently downgraded to a schema-less request.
//
// The match runs against e.snippet, already capped to maxErrorBody bytes.
func (e *statusError) rejectsOutputConfig() bool {
	return e.code == http.StatusBadRequest && strings.Contains(e.snippet, "output_config")
}

// anthropicErrorSignatures is the ordered table consulted for a non-retryable
// 400 on the Messages API wire, the counterpart of errorSignatures on the
// OpenAI wire.
//
// CONTRACT: these are STRING HEURISTICS by necessity (design doc §"Error-
// signature detection") and the fixtures in anthropic_thinking_test.go are what
// pin them. They are safe in a way a general string match is not, because they
// change NOTHING but the human-facing text: no retry, no downgrade, no request
// rewrite. A signature that stops matching (Anthropic reworded its 400) costs a
// hint, never correctness.
var anthropicErrorSignatures = []errorSignature{
	{
		// Sampling: current Claude models reject a non-default temperature
		// outright, and the older ones reject it while thinking is enabled
		// (research §matrix "temperature/top_p"). Two phrasings are in the
		// wild - "`temperature` may only be set to 1 when thinking is enabled"
		// and the "not supported" family the compatible gateways use - and the
		// field name alone is not enough: a 400 may mention temperature while
		// complaining about something else entirely.
		match: func(e *statusError) bool {
			if !strings.Contains(e.snippet, "temperature") {
				return false
			}
			return strings.Contains(e.snippet, "not supported") ||
				strings.Contains(e.snippet, "may only be set to 1")
		},
		advice: "this model rejects non-default sampling; remove provider.temperature/top_p",
	},
}

// mapAnthropicThinking translates the neutral reasoning knob into the two
// request fields Anthropic splits it across: the thinking control object and
// the effort level that belongs in output_config. It is a pure function - the
// same spec always yields the same wire fields.
//
// The vocabulary needs no rounding: Anthropic's own effort levels are
// low/medium/high/xhigh/max, which is the neutral union minus "none", and
// "none" is expressed by the thinking switch instead.
//
// CONTRACT: a BudgetTokens wins over an Effort. The two spellings target
// different model generations (legacy budget vs adaptive+effort), they cannot
// be combined in one request, and the budget is the more specific instruction
// - the same precedence the openrouter dialect applies. `amele explain`
// reports the mapping; the budget-below-max_tokens sanity check is config's
// job (exit 2), not this client's.
func mapAnthropicThinking(spec ReasoningSpec) (thinking *anThinking, effort string) {
	if spec.BudgetTokens > 0 {
		return &anThinking{Type: thinkingLegacyEnabled, BudgetTokens: spec.BudgetTokens}, ""
	}
	switch spec.Effort {
	case "":
		// The config said nothing: the provider's own default stands.
		return nil, ""
	case effortNone:
		return &anThinking{Type: thinkingOff}, ""
	default:
		return &anThinking{Type: thinkingAdaptive}, spec.Effort
	}
}

// The thinking control values. Named constants because each one is a contract
// with a model generation (see anThinking).
const (
	thinkingAdaptive      = "adaptive"
	thinkingLegacyEnabled = "enabled"
	thinkingOff           = "disabled"
)

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
// calls become tool_use content blocks, consecutive RoleTool messages merge
// into a single user message of tool_result blocks, and the reasoning,
// sampling and structured-output knobs land in their Anthropic spellings. It
// returns the struct-encoded part of the body and the body-root fragments
// merged afterwards (the caller's raw provider.params).
//
// CONTRACT: params keys cannot collide with the fields amele owns - config
// validation rejects that at exit 2 - so merging them needs no further defense
// here.
func (c *AnthropicClient) toWire(req Request) (anRequest, map[string]json.RawMessage) {
	// max_tokens is required on every Anthropic request. The per-call value is
	// the more specific instruction and wins over the client-level default, so
	// the cmd wiring can pass the config's cap through the neutral Request the
	// way every openai-wire dialect does; the constant is the last resort
	// because there is no server-side default to fall back on.
	maxTokens := req.MaxOutputTokens
	if maxTokens <= 0 {
		maxTokens = c.MaxOutputTokens
	}
	if maxTokens <= 0 {
		maxTokens = defaultAnthropicMaxOutput
	}
	out := anRequest{Model: req.Model, MaxTokens: maxTokens}

	for _, m := range req.Messages {
		out.appendMessage(m)
	}

	for _, t := range req.Tools {
		out.Tools = append(out.Tools, anTool{Name: t.Name, Description: t.Description, InputSchema: t.Parameters})
	}
	out.applyKnobs(req)
	return out, extraFields(req.Extra)
}

// appendMessage folds one neutral message into the wire request: system
// prompts hoist to the top-level field, tool results merge into the previous
// tool-result user turn, and assistant messages either echo their raw content
// array or are rebuilt from text and tool calls.
func (out *anRequest) appendMessage(m Message) {
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
		// CONTRACT: an assistant message whose carrier holds the original
		// content array is echoed VERBATIM - the array already contains the
		// text and tool_use blocks in the order the model produced them, so
		// rebuilding it here would both duplicate those blocks and break the
		// signatures Anthropic checks on the thinking blocks beside them
		// (research §"Load-bearing quirks" #3).
		//
		// The array check is the one guard: a carrier captured from another
		// wire (a reasoning_content string replayed against this client)
		// cannot be a content array, and reconstruction is a well-formed
		// request where sending it would be a guaranteed 400.
		if isJSONArray(m.Reasoning) {
			out.Messages = append(out.Messages, anMessage{Role: RoleAssistant, ContentRaw: m.Reasoning})
			return
		}
		out.Messages = append(out.Messages, anMessage{Role: RoleAssistant, Content: assistantBlocks(m)})
	default:
		out.Messages = append(out.Messages, anMessage{
			Role:    m.Role,
			Content: []anBlock{{Type: "text", Text: m.Content}},
		})
	}
}

// assistantBlocks rebuilds an assistant turn from the neutral fields: an
// optional leading text block, then one tool_use block per call. It runs only
// when there is no raw content array to echo instead.
func assistantBlocks(m Message) []anBlock {
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
	return blocks
}

// applyKnobs maps the reasoning, structured-output and sampling knobs onto
// their Anthropic spellings.
func (out *anRequest) applyKnobs(req Request) {
	var effort string
	if req.Reasoning != nil {
		out.Thinking, effort = mapAnthropicThinking(*req.Reasoning)
	}
	// output_config is ONE object carrying both the reasoning depth and the
	// native schema: they are independent keys, so a request that sets both
	// merges them here instead of sending the field twice.
	if effort != "" {
		out.OutputConfig = &anOutputConfig{Effort: effort}
	}
	if rf := req.ResponseFormat; rf != nil {
		if out.OutputConfig == nil {
			out.OutputConfig = &anOutputConfig{}
		}
		// Anthropic's format object takes the schema itself and no name (unlike
		// the OpenAI json_schema wrapper), so ResponseFormat.Name is not sent.
		out.OutputConfig.Format = &anOutputFormat{Type: "json_schema", Schema: rf.Schema}
	}
	// Sampling is passed through as given. Current Claude models answer a
	// non-default value with a 400 (research §matrix "temperature/top_p"),
	// which anthropicErrorSignatures turns into an actionable message - amele
	// never drops the value silently.
	out.Temperature = req.Temperature
	out.TopP = req.TopP
}

// extraFields copies the caller's raw provider.params into a fresh map, so the
// fragments merged into one request body can never leak into another.
func extraFields(extra map[string]json.RawMessage) map[string]json.RawMessage {
	if len(extra) == 0 {
		return nil
	}
	fields := make(map[string]json.RawMessage, len(extra))
	for key, value := range extra {
		fields[key] = value
	}
	return fields
}

// isJSONArray reports whether a carrier holds a JSON array, i.e. whether it can
// be a content array of this wire at all. nil, a JSON null and a payload from
// another provider's wire format all answer false.
func isJSONArray(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) > 0 && trimmed[0] == '['
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
