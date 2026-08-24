// Package llm defines the provider abstraction the agent loop talks to and
// the message/tool types shared across the codebase. Concrete providers
// (OpenAI-compatible HTTP in Phase 1) and the test fake live in this package
// so every consumer depends on one small surface.
package llm

import (
	"context"
	"encoding/json"
	"errors"
)

// ErrProvider is the sentinel wrapped by every transport/API failure.
// Callers map it to the frozen exit code 5 (provider error).
var ErrProvider = errors.New("provider error")

// Message roles. These mirror the OpenAI-compatible wire format because that
// format is the Phase 1 lingua franca; other providers adapt to it.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// ToolCall is a single tool invocation requested by the model. Arguments is
// the raw JSON string exactly as produced by the model; parsing is deferred
// to the tool so malformed arguments become a tool-level error the model can
// recover from.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// Message is one conversation entry. The zero value is not valid; always set
// Role.
type Message struct {
	Role string
	// Content is the text body. May be empty on assistant messages that
	// only carry tool calls.
	Content string
	// ToolCalls is set on assistant messages requesting tool use.
	ToolCalls []ToolCall
	// ToolCallID links a RoleTool message to the assistant call it answers.
	ToolCallID string
	// Reasoning is the provider's reasoning payload for this assistant
	// message, opaque to the loop. Clients fill it from their wire format (a
	// reasoning_content string, thinking blocks, a reasoning_details array)
	// and re-serialize it on the way back. Nil means "this message carries no
	// reasoning" - never an empty JSON document.
	//
	// CONTRACT: the loop and session layers never parse or modify it;
	// providers that sign their reasoning (Anthropic) or hash-check it
	// (DeepSeek) reject altered payloads, so the bytes that came off the wire
	// are the bytes that go back.
	Reasoning json.RawMessage
}

// ToolDef describes a tool offered to the model. Parameters is a JSON Schema
// document (raw, because the schema is authored as literal JSON and never
// manipulated structurally by the loop).
type ToolDef struct {
	Name        string
	Description string
	Parameters  json.RawMessage
}

// ResponseFormat requests provider-native JSON Schema enforcement for the
// answer. Providers that support it (OpenAI's response_format:json_schema and
// the gateways mirroring it) constrain decoding so the reply is valid by
// construction; providers that do not are detected at call time and the
// request is retried without it (see OpenAIClient.Chat). Either way the
// validate+retry layer above remains the safety net - the design rule is "use native
// when present, validate+retry otherwise".
type ResponseFormat struct {
	// Name identifies the schema to the provider (e.g. "amele_output").
	// OpenAI requires it and rejects empty names.
	Name string
	// Schema is the JSON Schema document, raw because it is authored as
	// literal JSON and never manipulated structurally here.
	Schema json.RawMessage
}

// ReasoningSpec is the provider-neutral reasoning knob: what the config asked
// for, before any dialect decides which wire field carries it. Each client
// maps it (effort to reasoning_effort / output_config, BudgetTokens to a
// budget-based thinking block) and reports the mapping in explain; nothing
// here is sent verbatim.
//
// The zero value means "no reasoning knob"; a Request carries *ReasoningSpec
// so that "unset" and "explicitly none" stay distinguishable - effort "none"
// is a real instruction to turn thinking off, not the absence of one.
type ReasoningSpec struct {
	// Effort is the normalized depth: none, low, medium, high, xhigh or max.
	// Dialects that lack a level round it and say so in explain; validation
	// of the value itself happens in config, not here.
	Effort string
	// BudgetTokens is an optional thinking-token budget for the targets that
	// take one (Anthropic legacy thinking, OpenRouter max_tokens reasoning).
	// Zero means "no budget given".
	BudgetTokens int
}

// Request is one chat completion call. The sampling, reasoning and cap fields
// are provider-neutral: a nil pointer or zero value means "send no such
// field", so a config that stays silent leaves the provider's own default
// untouched.
type Request struct {
	Model    string
	Messages []Message
	Tools    []ToolDef
	// ResponseFormat, when non-nil, asks the provider to constrain the reply
	// to the given JSON Schema. nil means plain text.
	ResponseFormat *ResponseFormat
	// MaxOutputTokens caps the tokens the provider may generate for this
	// call. Zero means "send no cap". The dialect picks the wire field name
	// (max_tokens vs max_completion_tokens).
	MaxOutputTokens int
	// Reasoning, when non-nil, asks for a reasoning depth. nil sends no
	// reasoning knob at all.
	Reasoning *ReasoningSpec
	// Temperature and TopP are the sampling knobs, pointers so that an
	// explicit 0 is distinguishable from "unset". Providers that reject them
	// (fixed-sampling models) are handled by the clients, not here.
	Temperature *float64
	TopP        *float64
	// Extra is the raw escape hatch (provider.params): keys merged verbatim
	// into the request body root, pre-serialized by the cmd wiring so this
	// package never re-encodes user YAML.
	//
	// CONTRACT: collisions with amele-owned wire fields are rejected at
	// validate time (exit 2), so a client may merge these keys without
	// re-checking that they cannot overwrite model, messages or tools.
	Extra map[string]json.RawMessage
}

// Usage is the token accounting reported by the provider for one call.
// CONTRACT: these provider-reported numbers are the primary budget unit of
// amele (docs/engineering.md §7) - never substitute local estimates. Every field is
// non-negative and bounded by maxTokensPerResponse: providers do not get to
// decide the arithmetic the budget runs on (see parseUsage).
type Usage struct {
	InputTokens  int
	OutputTokens int
}

// maxInt is the largest value of the platform int, the saturation point of
// every token accumulation in this package.
const maxInt = int(^uint(0) >> 1)

// maxTokensPerResponse caps a single provider-reported token count. The value
// is an order of magnitude above the largest context window on the market, so
// it cannot truncate an honest report; its job is to keep a hostile or broken
// one inside arithmetic the budget can reason about (an unbounded count could
// overflow the loop's running total into a negative number, which reads as
// "under budget" forever).
const maxTokensPerResponse = 1 << 24 // 16,777,216

// Total returns input+output tokens, saturating at maxInt rather than
// wrapping. CONTRACT: the token budget compares this against limits.max_tokens
// and a wrapped (negative) total would silently disable the budget.
func (u Usage) Total() int {
	if u.OutputTokens > maxInt-u.InputTokens {
		return maxInt
	}
	return u.InputTokens + u.OutputTokens
}

// Add returns the field-wise sum of two usages, saturating at maxInt.
// CONTRACT: accumulating usage across turns must never wrap - a negative
// running total would defeat limits.max_tokens - so callers summing per-turn
// usage use this instead of plain addition.
func (u Usage) Add(other Usage) Usage {
	return Usage{
		InputTokens:  saturatingAdd(u.InputTokens, other.InputTokens),
		OutputTokens: saturatingAdd(u.OutputTokens, other.OutputTokens),
	}
}

// saturatingAdd adds two non-negative token counts, returning maxInt instead
// of overflowing.
func saturatingAdd(a, b int) int {
	if b > maxInt-a {
		return maxInt
	}
	return a + b
}

// parseUsage normalizes the token counts one provider reported. It returns
// the sanitized usage and whether the report can be trusted; the caller marks
// an untrusted report as UsageMissing so budget enforcement fails closed.
//
// CONTRACT: counts are decoded as int64 (a 32-bit build must not fail to
// decode a large count) and land in Usage clamped to [0, maxTokensPerResponse].
//   - A negative count is impossible, not cheap: it would SHRINK the loop's
//     running total and buy the run unlimited extra spend. It is reported as
//     untrustworthy rather than silently read as 0.
//   - An absurd but positive count saturates. It is over any real budget, so
//     the run stops with a budget error instead of wrapping into "free".
func parseUsage(input, output int64) (usage Usage, trustworthy bool) {
	usage = Usage{InputTokens: clampTokens(input), OutputTokens: clampTokens(output)}
	return usage, input >= 0 && output >= 0
}

// clampTokens maps one reported count into [0, maxTokensPerResponse].
func clampTokens(n int64) int {
	if n < 0 {
		return 0
	}
	if n > maxTokensPerResponse {
		return maxTokensPerResponse
	}
	return int(n)
}

// Response is the provider's answer to one Request.
type Response struct {
	Message      Message
	Usage        Usage
	FinishReason string
	// UsageMissing is true when the provider did not report a usage object
	// at all. Callers enforcing token budgets must fail closed on it -
	// zero-because-absent must never be confused with zero-because-cheap.
	UsageMissing bool
	// SchemaEnforcementDropped is true when this response was produced
	// WITHOUT provider-native schema enforcement even though the request
	// carried a ResponseFormat: the OpenAI client sets it on every response
	// produced after its response_format fallback fired (the fallback
	// response and any later retry in the same Chat call), and the Anthropic
	// client sets it on every response to a schema-carrying request (that
	// API has no native enforcement). Callers surface it as a warning - the
	// validate+retry layer above is then the only thing enforcing
	// output.schema. Always false when the request asked for no schema.
	SchemaEnforcementDropped bool
}

// Provider is the single abstraction the agent loop depends on. Chat must be
// safe for sequential reuse; it does not need to be goroutine-safe in
// Phase 1 (the loop is strictly sequential - kept deterministic for
// replay).
type Provider interface {
	Chat(ctx context.Context, req Request) (*Response, error)
}
