package llm

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Dialect names one variation of the OpenAI-compatible wire format. The
// protocol is nominally shared, but the providers speaking it disagree on the
// details that matter to an agent: which field carries the output cap, how the
// reasoning knob is spelled, whether reasoning must be echoed back inside a
// tool loop, and which sampling parameters are accepted at all.
//
// CONTRACT: behavior keys on the DIALECT, never on the model name. Model ids
// churn within months (docs/superpowers/specs/2026-08-24-provider-dialects-research.md
// §"Load-bearing quirks"), so a model-name-keyed table would rot; model-level
// quirks are detected from provider error signatures instead.
//
// The zero value is not a dialect: use ParseDialect, which maps the empty
// config value to DialectOpenAI.
type Dialect string

// The dialects amele knows. The string values are the config file's
// vocabulary (provider.dialect) and are enumerated in the published JSON
// Schema, so they are public API.
const (
	// DialectOpenAI is the OpenAI chat/completions baseline and the meaning
	// of an omitted provider.dialect: every config written before dialects
	// existed keeps its behavior.
	DialectOpenAI Dialect = "openai"
	// DialectDeepSeek is DeepSeek's native API (api.deepseek.com).
	DialectDeepSeek Dialect = "deepseek"
	// DialectGLM is Z.ai's GLM API.
	DialectGLM Dialect = "glm"
	// DialectKimi is Moonshot's Kimi API.
	DialectKimi Dialect = "kimi"
	// DialectGroq is Groq's OpenAI-compatible endpoint.
	DialectGroq Dialect = "groq"
	// DialectOpenRouter is the OpenRouter gateway, which normalizes the
	// reasoning knob across the models it fronts.
	DialectOpenRouter Dialect = "openrouter"
)

// dialects lists every accepted value in the order ParseDialect names them,
// with the default first. One list backs both the lookup and the error
// message, so a dialect can never be accepted without being advertised.
var dialects = []Dialect{
	DialectOpenAI,
	DialectDeepSeek,
	DialectGLM,
	DialectKimi,
	DialectGroq,
	DialectOpenRouter,
}

// ParseDialect maps a config value to a Dialect. The empty string yields
// DialectOpenAI. Any other value must match a known dialect EXACTLY: matching
// is neither case-folded nor trimmed, because a value silently "corrected"
// from "OpenAI " would pick a wire mapping the author never named, and the
// consequence (a dropped reasoning field, a rejected cap) would only surface
// as a provider error much later. The returned error names the offending
// value and every accepted spelling, and the returned Dialect is the zero
// value on failure.
func ParseDialect(s string) (Dialect, error) {
	if s == "" {
		return DialectOpenAI, nil
	}
	for _, d := range dialects {
		if string(d) == s {
			return d, nil
		}
	}
	return "", fmt.Errorf("unknown dialect %q (valid: %s; omit for openai)", s, dialectValues())
}

// dialectValues renders the accepted spellings as one comma-separated list for
// ParseDialect's error, derived from the same slice the lookup walks.
func dialectValues() string {
	names := make([]string, len(dialects))
	for i, d := range dialects {
		names[i] = string(d)
	}
	return strings.Join(names, ", ")
}

// effortNone is the one effort value that means "turn thinking OFF" rather
// than "think this deeply". Dialects that carry a thinking switch map it to
// that switch instead of to a level.
const effortNone = "none"

// The thinking-object fragments the CN-native dialects (deepseek, glm) take.
// They are string constants rather than package-level json.RawMessage vars
// because a []byte var is mutable shared state (docs/engineering.md §5.1) and
// these bytes are a contract with the provider.
const (
	thinkingEnabledJSON  = `{"type":"enabled"}`
	thinkingDisabledJSON = `{"type":"disabled"}`
)

// capField returns the request-body key that carries the output-token cap for
// this dialect. The zero-value Dialect answers like DialectOpenAI, so a client
// that was never told a dialect keeps the pre-dialect behavior.
//
// CONTRACT: the split is not cosmetic. The reasoning-era OpenAI family answers
// max_tokens with a 400, and the CN natives plus OpenRouter only know
// max_tokens (research §matrix "Output cap field"). Sending the wrong name is
// either a hard failure or - worse - a silently uncapped run.
func capField(d Dialect) string {
	switch d {
	case DialectDeepSeek, DialectGLM, DialectOpenRouter:
		return "max_tokens"
	case DialectOpenAI, DialectGroq, DialectKimi:
		return "max_completion_tokens"
	default:
		// Unreachable through ParseDialect; the zero value lands here and must
		// behave like openai.
		return "max_completion_tokens"
	}
}

// ReasoningMapping is what one dialect makes of a ReasoningSpec: the request
// body fragments to merge at the body root, and the human-readable lines that
// describe the mapping.
//
// CONTRACT: this is the single source of truth for the reasoning knob. The
// wire encoder merges Fields and `amele explain` prints Notes, so what the
// report promises and what the request carries cannot drift apart.
type ReasoningMapping struct {
	// Fields maps body-root keys to pre-serialized JSON values. Nil means the
	// dialect sends no reasoning field at all.
	Fields map[string]json.RawMessage
	// Notes describes the mapping in the config's vocabulary, one line per
	// decision, including every rounding and every value that is NOT sent.
	// Order is deterministic (emission order), never map order.
	Notes []string
}

// field records one body-root fragment, allocating the map on first use so an
// empty mapping stays nil.
func (m *ReasoningMapping) field(key string, value json.RawMessage) {
	if m.Fields == nil {
		m.Fields = make(map[string]json.RawMessage, 2)
	}
	m.Fields[key] = value
}

// note appends one human-readable mapping line.
func (m *ReasoningMapping) note(format string, args ...any) {
	m.Notes = append(m.Notes, fmt.Sprintf(format, args...))
}

// MapReasoning translates the neutral reasoning knob into one dialect's wire
// fields. It is a pure function: the same spec always yields the same fields
// and the same notes, which is what lets `explain` promise before a run
// exactly what the request will carry.
//
// The effort vocabulary (none/low/medium/high/xhigh/max) is the UNION of what
// the providers accept; a dialect with a smaller vocabulary rounds UP to the
// nearest level it has and says so in Notes. Rounding up rather than down is
// deliberate: the config asked for at least this much thinking, and quietly
// buying less is the failure mode a user cannot see in the output.
//
// An empty Effort with no BudgetTokens yields an empty mapping - "the config
// said nothing, so the provider default stands". BudgetTokens is mapped only
// by the openrouter dialect; on the rest of the openai wire it is a validation
// error (config.validateReasoning), never a silent drop here.
//
// Values outside the vocabulary are passed through or rounded by the same
// rules rather than rejected: validation owns the vocabulary, and a client
// that second-guessed it would just fail later with less context.
func MapReasoning(d Dialect, spec ReasoningSpec) ReasoningMapping {
	switch d {
	case DialectDeepSeek, DialectGLM:
		return mapThinkingObjectDialect(d, spec)
	case DialectKimi:
		return mapKimi(spec)
	case DialectOpenRouter:
		return mapOpenRouter(spec)
	case DialectOpenAI, DialectGroq:
		return mapEffortPassthrough(spec)
	default:
		// The zero value behaves like openai.
		return mapEffortPassthrough(spec)
	}
}

// mapEffortPassthrough is the OpenAI baseline, shared by groq: a top-level
// reasoning_effort carrying the value verbatim. Both accept the whole
// vocabulary, "none" included, so nothing is rounded.
func mapEffortPassthrough(spec ReasoningSpec) ReasoningMapping {
	var m ReasoningMapping
	if spec.Effort == "" {
		return m
	}
	m.field("reasoning_effort", jsonString(spec.Effort))
	m.note("reasoning.effort: %s -> reasoning_effort: %s", spec.Effort, spec.Effort)
	return m
}

// mapThinkingObjectDialect covers deepseek and glm, which share a shape: a
// thinking object that switches thinking on or off, plus the level.
//
// CONTRACT: reasoning_effort goes at the BODY ROOT, not inside the thinking
// object. DeepSeek documents both placements (API reference: inside thinking;
// thinking-mode guide: top-level - research §matrix "Reasoning knob"), so one
// had to be pinned; the top-level spelling is the one GLM and Kimi also use,
// which keeps a single shape across the CN trio. An integration smoke test
// guards it, and a server-side change would surface there rather than as a
// silently ignored knob in production.
func mapThinkingObjectDialect(d Dialect, spec ReasoningSpec) ReasoningMapping {
	var m ReasoningMapping
	if spec.Effort == "" {
		return m
	}
	if spec.Effort == effortNone {
		// Thinking off: a level would be contradictory, and these APIs default
		// thinking ON, so the switch is the whole instruction.
		m.field("thinking", json.RawMessage(thinkingDisabledJSON))
		m.note(`reasoning.effort: none -> thinking: %s (%s sends no reasoning_effort with thinking off)`, thinkingDisabledJSON, d)
		return m
	}
	// The switch is sent explicitly even though these providers think by
	// default: a config that named a depth should not depend on a default
	// staying put.
	m.field("thinking", json.RawMessage(thinkingEnabledJSON))
	m.note(`reasoning.effort: %s -> thinking: %s`, spec.Effort, thinkingEnabledJSON)
	m.roundedEffort(d, spec.Effort)
	return m
}

// mapKimi maps the level for the K-series: a top-level reasoning_effort and
// never a thinking object. K3 has no thinking parameter at all, and K2.x users
// who need one set it through provider.params (research §matrix "Reasoning
// knob"), so amele emitting one would be a guess at the model behind the name.
func mapKimi(spec ReasoningSpec) ReasoningMapping {
	var m ReasoningMapping
	if spec.Effort == "" {
		return m
	}
	if spec.Effort == effortNone {
		// Unreachable through validate, which rejects kimi + none (exit 2).
		// Reported rather than mapped: the K-series thinks unconditionally, so
		// there is no field that would honor the request.
		m.note("reasoning.effort: none not sent: kimi models cannot disable thinking")
		return m
	}
	m.roundedEffort(DialectKimi, spec.Effort)
	return m
}

// mapOpenRouter maps onto the gateway's unified reasoning object. effort and
// max_tokens are mutually exclusive there (research §OpenRouter), so a spec
// carrying both sends the budget - the more specific instruction - and the
// dropped effort is reported instead of being silently ignored.
func mapOpenRouter(spec ReasoningSpec) ReasoningMapping {
	var m ReasoningMapping
	if spec.BudgetTokens > 0 {
		budget := fmt.Sprintf(`{"max_tokens":%d}`, spec.BudgetTokens)
		m.field("reasoning", json.RawMessage(budget))
		m.note("reasoning.budget_tokens: %d -> reasoning: %s", spec.BudgetTokens, budget)
		if spec.Effort != "" {
			m.note("reasoning.effort: %s not sent: openrouter takes effort or max_tokens, not both", spec.Effort)
		}
		return m
	}
	if spec.Effort == "" {
		return m
	}
	// OpenRouter accepts the full effort union and maps it per underlying
	// provider itself, so amele passes the value through unrounded.
	object := fmt.Sprintf(`{"effort":%s}`, jsonString(spec.Effort))
	m.field("reasoning", json.RawMessage(object))
	m.note("reasoning.effort: %s -> reasoning: %s", spec.Effort, object)
	return m
}

// roundedEffort emits the level for the dialects whose vocabulary is
// low/high/max (the CN trio) and notes the rounding when one happened.
func (m *ReasoningMapping) roundedEffort(d Dialect, effort string) {
	level := roundEffortLowHighMax(effort)
	m.field("reasoning_effort", jsonString(level))
	if level == effort {
		m.note("reasoning.effort: %s -> reasoning_effort: %s", effort, level)
		return
	}
	m.note("reasoning.effort: %s -> reasoning_effort: %s (%s has no %s)", effort, level, d, effort)
}

// roundEffortLowHighMax rounds the neutral vocabulary onto the low/high/max
// subset DeepSeek, GLM and Kimi share (research §"Load-bearing quirks" #5).
// Unknown values pass through untouched: the vocabulary is validated in
// config, and inventing a level for an unrecognized one would hide a typo the
// provider's own 400 names precisely.
func roundEffortLowHighMax(effort string) string {
	switch effort {
	case "medium":
		return "high"
	case "xhigh":
		return "max"
	default:
		return effort
	}
}

// jsonString renders one string as a JSON string fragment. json.Marshal cannot
// fail for a string (invalid UTF-8 is replaced, not rejected), so the error
// path is a defensive fallback rather than a reachable branch.
func jsonString(s string) json.RawMessage {
	encoded, err := json.Marshal(s)
	if err != nil {
		return json.RawMessage(`""`)
	}
	return encoded
}
