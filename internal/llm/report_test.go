package llm

import (
	"slices"
	"strings"
	"testing"
)

// TestUnknownFieldPolicy pins the sentence `amele explain` prints about
// provider.params for each dialect. The wording is the operator's only warning
// that a raw key is either a 400 waiting to happen or a silent no-op, so it is
// asserted rather than left to prose drift.
func TestUnknownFieldPolicy(t *testing.T) {
	tests := map[Dialect]string{
		DialectOpenAI:     "rejected (400)",
		DialectDeepSeek:   "ignored",
		DialectOpenRouter: "passed through",
		DialectGLM:        "not documented; assume rejected (400)",
		DialectKimi:       "not documented; assume rejected (400)",
		DialectGroq:       "not documented; assume rejected (400)",
		Dialect(""):       "rejected (400)", // zero value = openai
	}
	for dialect, want := range tests {
		t.Run(string(dialect), func(t *testing.T) {
			if got := UnknownFieldPolicy(dialect); got != want {
				t.Errorf("UnknownFieldPolicy(%q) = %q, want %q", dialect, got, want)
			}
		})
	}
}

// TestAnthropicUnknownFieldPolicy pins the anthropic wire's own answer, which
// is not a dialect: the Messages API is strict about unrecognized fields.
func TestAnthropicUnknownFieldPolicy(t *testing.T) {
	if got := AnthropicUnknownFieldPolicy(); got != "rejected (400)" {
		t.Errorf("AnthropicUnknownFieldPolicy() = %q, want %q", got, "rejected (400)")
	}
}

// TestDialectForBaseURL pins the base_url -> dialect hint table. It is a HINT
// only: nothing here changes behavior, it just tells an operator who pointed a
// default-dialect config at api.deepseek.com that the reasoning knob they wrote
// is being spelled the OpenAI way.
func TestDialectForBaseURL(t *testing.T) {
	tests := []struct {
		name  string
		url   string
		want  Dialect
		found bool
	}{
		{"deepseek", "https://api.deepseek.com", DialectDeepSeek, true},
		{"deepseek with path", "https://api.deepseek.com/v1", DialectDeepSeek, true},
		{"deepseek anthropic-compat path", "https://api.deepseek.com/anthropic", DialectDeepSeek, true},
		{"glm bigmodel", "https://open.bigmodel.cn/api/paas/v4", DialectGLM, true},
		{"glm z.ai", "https://api.z.ai/api/paas/v4", DialectGLM, true},
		{"kimi com", "https://api.moonshot.ai/v1", DialectKimi, true},
		{"kimi cn", "https://api.moonshot.cn/v1", DialectKimi, true},
		{"groq", "https://api.groq.com/openai/v1", DialectGroq, true},
		{"openrouter", "https://openrouter.ai/api/v1", DialectOpenRouter, true},
		{"host is matched case-insensitively", "https://API.DeepSeek.com/v1", DialectDeepSeek, true},
		{"port is ignored", "https://api.groq.com:443/openai/v1", DialectGroq, true},
		{"openai itself is not a hint", "https://api.openai.com/v1", "", false},
		{"unknown host", "https://llm.internal.example/v1", "", false},
		{"a lookalike suffix is not a match", "https://api.deepseek.com.evil.test/v1", "", false},
		{"empty base_url", "", "", false},
		{"unparseable base_url", "://nope", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := DialectForBaseURL(tt.url)
			if ok != tt.found {
				t.Fatalf("DialectForBaseURL(%q) found = %v, want %v", tt.url, ok, tt.found)
			}
			if got != tt.want {
				t.Errorf("DialectForBaseURL(%q) = %q, want %q", tt.url, got, tt.want)
			}
		})
	}
}

// TestBaseURLHost pins the display half of the hint: the host the table was
// consulted with, lowercased and without its port, or "" when there is none.
func TestBaseURLHost(t *testing.T) {
	tests := map[string]string{
		"https://api.deepseek.com/v1":         "api.deepseek.com",
		"https://API.GROQ.com:443/openai/v1":  "api.groq.com",
		"https://llm.internal.example:8080/v": "llm.internal.example",
		"":                                    "",
		"://nope":                             "",
		"not-a-url":                           "",
	}
	for in, want := range tests {
		if got := BaseURLHost(in); got != want {
			t.Errorf("BaseURLHost(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestAnthropicReasoningNotes pins what explain prints for the anthropic wire.
// The notes are DERIVED from mapAnthropicThinking's own result rather than
// re-deciding the mapping, so the report cannot promise a field the client does
// not send - the dropped effort of a budget+effort config above all.
func TestAnthropicReasoningNotes(t *testing.T) {
	tests := []struct {
		name string
		spec ReasoningSpec
		want []string
	}{
		{"nothing set", ReasoningSpec{}, nil},
		{
			name: "effort maps to adaptive thinking plus output_config",
			spec: ReasoningSpec{Effort: "high"},
			want: []string{
				`reasoning.effort: high -> thinking: {"type":"adaptive"}`,
				"reasoning.effort: high -> output_config.effort: high",
			},
		},
		{
			name: "none turns thinking off and sends no effort",
			spec: ReasoningSpec{Effort: "none"},
			want: []string{`reasoning.effort: none -> thinking: {"type":"disabled"}`},
		},
		{
			name: "budget maps to the legacy thinking block",
			spec: ReasoningSpec{BudgetTokens: 8192},
			want: []string{`reasoning.budget_tokens: 8192 -> thinking: {"type":"enabled","budget_tokens":8192}`},
		},
		{
			// The drop is the point: budget wins, and an operator who wrote
			// both must read that the effort is not on the wire.
			name: "budget wins and the effort is reported as dropped",
			spec: ReasoningSpec{Effort: "max", BudgetTokens: 4096},
			want: []string{
				`reasoning.budget_tokens: 4096 -> thinking: {"type":"enabled","budget_tokens":4096}`,
				"reasoning.effort: max not sent: the anthropic wire takes a thinking budget or an effort, not both",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := AnthropicReasoningNotes(tt.spec)
			if !slices.Equal(got, tt.want) {
				t.Errorf("AnthropicReasoningNotes(%+v) =\n%s\nwant\n%s",
					tt.spec, strings.Join(got, "\n"), strings.Join(tt.want, "\n"))
			}
		})
	}
}
