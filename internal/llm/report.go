package llm

import (
	"encoding/json"
	"net/url"
	"strings"
)

// This file holds the REPORT side of the dialect knowledge: what `amele
// explain` must be able to say about a config before a single request is made.
//
// CONTRACT: nothing here decides anything. Every function either reads the same
// table the wire encoder reads (CapField, UnknownFieldPolicy) or derives its
// sentences from the mapping function's OWN result (AnthropicReasoningNotes,
// and MapReasoning.Notes on the openai wire). A report that re-implemented a
// mapping would drift from the request it describes, which is the one failure
// mode an explain report must not have.

// UnknownFieldPolicy describes, in one phrase, what this dialect's endpoint
// does with a request field it does not recognize. It is the answer explain
// gives about provider.params: the same raw key is a hard 400 on one endpoint,
// a silent no-op on another, and a real feature toggle on a third.
//
// The phrases follow the research sweep
// (docs/superpowers/specs/2026-08-24-provider-dialects-research.md §matrix,
// row "Unknown request fields"). Where that sweep could not verify a provider's
// behavior the phrase SAYS SO and assumes the strict answer: telling an
// operator "ignored" about an endpoint that in fact 400s would send them
// hunting for a bug in amele.
func UnknownFieldPolicy(d Dialect) string {
	switch d {
	case DialectDeepSeek:
		// Undocumented but consistently observed (the ignored-penalties
		// precedent in the same API).
		return "ignored"
	case DialectOpenRouter:
		// Documented: provider-specific extras are transmitted to the
		// upstream provider.
		return "passed through"
	case DialectGLM, DialectKimi, DialectGroq:
		return "not documented; assume rejected (400)"
	case DialectOpenAI:
		return "rejected (400)"
	default:
		// Unreachable through ParseDialect; the zero value answers like
		// openai, as everywhere else in this package.
		return "rejected (400)"
	}
}

// SamplingNote returns the caveat that applies to a temperature/top_p this
// dialect will ACCEPT but not honor, or "" when the values take effect as sent.
// spec is what the config asked of the reasoning knob, because the answer can
// depend on it.
//
// Only deepseek has one today: it accepts both knobs and silently ignores them
// while thinking (research §matrix "temperature/top_p"), and thinking is ON by
// default there - so the config that says nothing about reasoning is exactly
// the config whose sampling values do nothing. The dialects that REJECT a
// sampling value instead (kimi's fixed K-series) are refused at validate, and
// the ones that merely narrow the range (glm) are reported by that range.
//
// CONTRACT: a report that showed `temperature: 0.2 -> temperature: 0.2` and
// stopped there would promise an effect the run will not have - the same
// silent-degradation failure the dialect layer exists to prevent.
func SamplingNote(d Dialect, spec ReasoningSpec) string {
	if d != DialectDeepSeek || spec.Effort == effortNone {
		return ""
	}
	return "temperature/top_p: sent but ignored by deepseek in thinking mode (thinking is on by default)"
}

// AnthropicUnknownFieldPolicy is UnknownFieldPolicy for the Anthropic Messages
// API, which is a WIRE rather than a dialect: the dialect is not consulted at
// all on that path, so the report needs its own answer. The Messages API is
// strict and answers an unrecognized field with a 400.
func AnthropicUnknownFieldPolicy() string { return "rejected (400)" }

// baseURLDialects maps a provider's API host to the dialect that speaks its
// variation of the wire. It backs a HINT and nothing else: amele never
// auto-detects a dialect from base_url (design doc §"No magic"), because a
// silently chosen dialect would reshape every request in a way the config file
// does not show. Explain prints the mismatch and the operator decides.
//
// Keys are lowercase hostnames without a port, matched EXACTLY - a suffix match
// would claim "api.deepseek.com.attacker.example" as DeepSeek.
var baseURLDialects = map[string]Dialect{
	"api.deepseek.com":  DialectDeepSeek,
	"open.bigmodel.cn":  DialectGLM,
	"api.z.ai":          DialectGLM,
	"api.moonshot.ai":   DialectKimi,
	"api.moonshot.cn":   DialectKimi,
	"api.groq.com":      DialectGroq,
	"openrouter.ai":     DialectOpenRouter,
	"api.openrouter.ai": DialectOpenRouter,
}

// DialectForBaseURL reports the dialect a known provider host speaks, and
// whether the host was known at all. api.openai.com is deliberately absent: the
// openai dialect is the default, so a "hint" about it could only ever repeat
// what the config already does.
//
// A base_url that does not parse, carries no host, or names an unknown host
// yields ("", false) - no hint, never a guess.
func DialectForBaseURL(baseURL string) (Dialect, bool) {
	host := BaseURLHost(baseURL)
	if host == "" {
		return "", false
	}
	d, ok := baseURLDialects[host]
	return d, ok
}

// BaseURLHost returns the lowercase hostname of a base URL, without its port,
// or "" when it has none (an empty or unparseable value). It exists so the
// dialect hint can NAME the host it matched while DialectForBaseURL above stays
// the only place that decides what a host means - one parse, one table.
func BaseURLHost(baseURL string) string {
	u, err := url.Parse(baseURL)
	if err != nil {
		return ""
	}
	return strings.ToLower(u.Hostname())
}

// AnthropicReasoningNotes describes what the Anthropic wire makes of the
// neutral reasoning knob, in the config's vocabulary, one line per decision.
// It is the anthropic-wire counterpart of ReasoningMapping.Notes.
//
// CONTRACT: the lines are derived from mapAnthropicThinking's OWN return value,
// never from a second reading of the spec. That is what makes the dropped
// effort of a budget+effort config reportable: the client's precedence rule
// lives in one place, and this function only puts words to its result.
func AnthropicReasoningNotes(spec ReasoningSpec) []string {
	thinking, effort := mapAnthropicThinking(spec)
	if thinking == nil {
		// The config said nothing about reasoning; the model's own default
		// stands and there is nothing to report.
		return nil
	}
	encoded, err := json.Marshal(thinking)
	if err != nil {
		// Unreachable: anThinking is two plain fields. Reporting nothing beats
		// reporting a half-rendered wire fragment.
		return nil
	}

	var m ReasoningMapping
	if thinking.BudgetTokens > 0 {
		m.note("reasoning.budget_tokens: %d -> thinking: %s", thinking.BudgetTokens, encoded)
	} else {
		m.note("reasoning.effort: %s -> thinking: %s", spec.Effort, encoded)
	}
	switch {
	case effort != "":
		m.note("reasoning.effort: %s -> output_config.effort: %s", spec.Effort, effort)
	case spec.Effort != "" && thinking.Type != thinkingOff:
		// The client dropped the effort (the budget won). Silence here would
		// be exactly the "silently dropped" failure the design forbids.
		m.note("reasoning.effort: %s not sent: the anthropic wire takes a thinking budget or an effort, not both", spec.Effort)
	}
	return m.Notes
}
