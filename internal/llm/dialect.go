package llm

import (
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
