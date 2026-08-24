package explain

import (
	"strings"
	"testing"

	"github.com/lasthumanintheloop/amele/internal/config"
)

// ptrFloat is the sampling-knob helper: the fields are pointers so an explicit
// 0 is distinguishable from "unset".
func ptrFloat(f float64) *float64 { return &f }

// TestProviderMappingRows pins the PROVIDER MAPPING rows: what a config's
// tuning knobs become on the wire, before a single request is made. The rows
// are the promise `explain` makes about a run, so each case asserts the exact
// lines rather than a substring of prose.
func TestProviderMappingRows(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*config.Config)
		want    []string
		notWant []string
	}{
		{
			// The compatibility case: a config with no tuning gets no mapping
			// block at all, so today's reports are unchanged.
			name:    "no tuning, no block",
			mutate:  func(*config.Config) {},
			notWant: []string{"provider mapping", "dialect:"},
		},
		{
			name: "openai dialect names the openai cap field",
			mutate: func(c *config.Config) {
				c.Provider.MaxOutputTokens = 4096
			},
			want: []string{"max_output_tokens: 4096 -> max_completion_tokens: 4096"},
		},
		{
			name: "deepseek rounds the effort and renames the cap",
			mutate: func(c *config.Config) {
				c.Provider.Dialect = "deepseek"
				c.Provider.MaxOutputTokens = 4096
				c.Provider.Reasoning = &config.ReasoningConfig{Effort: "medium"}
			},
			want: []string{
				`  dialect:         "deepseek"`,
				"max_output_tokens: 4096 -> max_tokens: 4096",
				`reasoning.effort: medium -> thinking: {"type":"enabled"}`,
				"reasoning.effort: medium -> reasoning_effort: high (deepseek has no medium)",
			},
			notWant: []string{"max_completion_tokens"},
		},
		{
			name: "sampling knobs are reported with their values",
			mutate: func(c *config.Config) {
				c.Provider.Temperature = ptrFloat(0)
				c.Provider.TopP = ptrFloat(0.9)
			},
			want: []string{
				"temperature: 0 -> temperature: 0",
				"top_p: 0.9 -> top_p: 0.9",
			},
		},
		{
			// SECURITY: a params value can be a routing credential, so the
			// report lists the KEYS and never the values.
			name: "params list keys only, with the dialect's unknown-field policy",
			mutate: func(c *config.Config) {
				c.Provider.Dialect = "deepseek"
				c.Provider.Params = map[string]any{"verbosity": "quiet-value", "clear_thinking": "secret-value"}
			},
			want: []string{
				`params keys (merged verbatim, values not shown): "clear_thinking", "verbosity"`,
				"unknown request fields: ignored",
			},
			notWant: []string{"secret-value", "quiet-value"},
		},
		{
			name: "openai rejects unknown fields",
			mutate: func(c *config.Config) {
				c.Provider.Params = map[string]any{"verbosity": "low"}
			},
			want: []string{"unknown request fields: rejected (400)"},
		},
		{
			name: "openrouter passes unknown fields through",
			mutate: func(c *config.Config) {
				c.Provider.Dialect = "openrouter"
				c.Provider.Params = map[string]any{"provider": map[string]any{"require_parameters": true}}
			},
			want: []string{"unknown request fields: passed through"},
		},
		{
			name: "openrouter maps a budget and reports the dropped effort",
			mutate: func(c *config.Config) {
				c.Provider.Dialect = "openrouter"
				c.Provider.Reasoning = &config.ReasoningConfig{Effort: "high", BudgetTokens: 8192}
			},
			want: []string{
				`reasoning.budget_tokens: 8192 -> reasoning: {"max_tokens":8192}`,
				"reasoning.effort: high not sent: openrouter takes effort or max_tokens, not both",
			},
		},
		{
			// The anthropic wire has its own mapping AND its own precedence:
			// the budget wins and the effort is dropped. An operator who wrote
			// both must read that here rather than wonder later.
			name: "anthropic budget wins and the dropped effort is reported",
			mutate: func(c *config.Config) {
				c.Provider.Type = config.ProviderTypeAnthropic
				c.Provider.MaxOutputTokens = 16384
				c.Provider.Reasoning = &config.ReasoningConfig{Effort: "max", BudgetTokens: 8192}
			},
			want: []string{
				"max_output_tokens: 16384 -> max_tokens: 16384",
				`reasoning.budget_tokens: 8192 -> thinking: {"type":"enabled","budget_tokens":8192}`,
				"reasoning.effort: max not sent: the anthropic wire takes a thinking budget or an effort, not both",
			},
		},
		{
			name: "anthropic effort maps to adaptive thinking",
			mutate: func(c *config.Config) {
				c.Provider.Type = config.ProviderTypeAnthropic
				c.Provider.Reasoning = &config.ReasoningConfig{Effort: "high"}
			},
			want: []string{
				`reasoning.effort: high -> thinking: {"type":"adaptive"}`,
				"reasoning.effort: high -> output_config.effort: high",
			},
		},
		{
			// The dialect names a variation of the OpenAI-compatible wire and
			// is documented as ignored on the anthropic one; a leftover value
			// must be reported as inert rather than silently believed.
			name: "a leftover dialect is reported as ignored on the anthropic wire",
			mutate: func(c *config.Config) {
				c.Provider.Type = config.ProviderTypeAnthropic
				c.Provider.Dialect = "kimi"
				c.Provider.Params = map[string]any{"verbosity": "low"}
			},
			want: []string{
				`dialect:         "kimi" (ignored: the anthropic wire has no dialects)`,
				"unknown request fields: rejected (400)",
			},
		},
		{
			// An empty reasoning block is what `--set provider.reasoning.effort=`
			// leaves behind, and it means the provider default - so the report
			// must not claim a reasoning field is being sent.
			name: "an empty reasoning block maps to nothing",
			mutate: func(c *config.Config) {
				c.Provider.Reasoning = &config.ReasoningConfig{}
			},
			notWant: []string{"provider mapping", "reasoning"},
		},
		{
			// A dialect that does not parse makes every dialect-dependent row
			// unanswerable. Guessing "openai" would describe a request amele
			// will never send, because the run is refused at validate.
			name: "an invalid dialect suppresses the dialect-dependent rows",
			mutate: func(c *config.Config) {
				c.Provider.Dialect = "gemini"
				c.Provider.MaxOutputTokens = 4096
				c.Provider.Reasoning = &config.ReasoningConfig{Effort: "high"}
				c.Provider.Temperature = ptrFloat(0.2)
			},
			want: []string{
				"provider.dialect is not a known dialect: the wire mapping cannot be reported (see PROBLEMS)",
				"temperature: 0.2 -> temperature: 0.2",
			},
			notWant: []string{"max_completion_tokens", "reasoning_effort"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := baseCfg()
			tt.mutate(cfg)
			got := Render(cfg, registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil)
			for _, want := range tt.want {
				if !strings.Contains(got, want) {
					t.Errorf("report is missing %q:\n%s", want, got)
				}
			}
			for _, unwanted := range tt.notWant {
				if strings.Contains(got, unwanted) {
					t.Errorf("report contains %q, which it must not:\n%s", unwanted, got)
				}
			}
		})
	}
}

// TestProviderMappingBaseURLHint pins the one place amele looks at base_url to
// say something about the dialect. It is a HINT: nothing is auto-detected (a
// silently chosen dialect would reshape every request without the file showing
// it), so the report names the host and the operator decides.
func TestProviderMappingBaseURLHint(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*config.Config)
		want    string
		notWant string
	}{
		{
			name: "known host with the default dialect",
			mutate: func(c *config.Config) {
				c.Provider.BaseURL = "https://api.deepseek.com"
			},
			want: "hint: base_url looks like api.deepseek.com; consider dialect: deepseek",
		},
		{
			name: "known host with the wrong dialect",
			mutate: func(c *config.Config) {
				c.Provider.BaseURL = "https://api.groq.com/openai/v1"
				c.Provider.Dialect = "deepseek"
			},
			want: "hint: base_url looks like api.groq.com; consider dialect: groq",
		},
		{
			name: "matching dialect draws no hint",
			mutate: func(c *config.Config) {
				c.Provider.BaseURL = "https://api.deepseek.com"
				c.Provider.Dialect = "deepseek"
			},
			notWant: "hint: base_url",
		},
		{
			name:    "unknown host draws no hint",
			mutate:  func(*config.Config) {},
			notWant: "hint: base_url",
		},
		{
			// The anthropic-compatible endpoints of the CN trio are a
			// documented setup; the dialect is not consulted on that wire, so
			// a hint there would send the operator to change a field that
			// changes nothing.
			name: "no hint on the anthropic wire",
			mutate: func(c *config.Config) {
				c.Provider.Type = config.ProviderTypeAnthropic
				c.Provider.BaseURL = "https://api.deepseek.com/anthropic"
			},
			notWant: "hint: base_url",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := baseCfg()
			tt.mutate(cfg)
			got := Render(cfg, registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil)
			if tt.want != "" && !strings.Contains(got, tt.want) {
				t.Errorf("report is missing the hint %q:\n%s", tt.want, got)
			}
			if tt.notWant != "" && strings.Contains(got, tt.notWant) {
				t.Errorf("report contains an unwanted hint %q:\n%s", tt.notWant, got)
			}
		})
	}
}

// TestProviderMappingCannotForgeRows: every mapping row embeds config text
// (a dialect, an effort, a params key), and explain reports on configs that
// FAILED validation - so a value carrying a newline must not be able to grow
// the report a row of its own.
func TestProviderMappingCannotForgeRows(t *testing.T) {
	cfg := baseCfg()
	cfg.Provider.Dialect = "deepseek\n  base_url: https://evil.example/v1"
	cfg.Provider.Reasoning = &config.ReasoningConfig{Effort: "high\n    forged: row"}
	cfg.Provider.Params = map[string]any{"k\n    forged: param": 1}

	got := Render(cfg, registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil)

	for _, forged := range []string{
		"\n  base_url: https://evil.example/v1",
		"\n    forged: row",
		"\n    forged: param",
	} {
		if strings.Contains(got, forged) {
			t.Errorf("forged row %q made it into the report:\n%s", forged, got)
		}
	}
	if n := strings.Count(got, "\n  base_url:"); n != 1 {
		t.Errorf("base_url row appears %d times, want 1:\n%s", n, got)
	}
}
