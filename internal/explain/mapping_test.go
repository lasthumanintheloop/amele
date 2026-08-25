package explain

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/llm"
	"github.com/lasthumanintheloop/amele/internal/tools"
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
			// DeepSeek accepts temperature/top_p and then IGNORES them while
			// thinking - and thinking is on by default there. Reporting the
			// values as a plain pass-through would promise an effect the run
			// will not have, which is exactly what this block exists to
			// prevent.
			name: "deepseek reports sampling as ignored in thinking mode",
			mutate: func(c *config.Config) {
				c.Provider.Dialect = "deepseek"
				c.Provider.Temperature = ptrFloat(0.2)
			},
			want: []string{
				"temperature: 0.2 -> temperature: 0.2",
				"temperature/top_p: sent but ignored by deepseek in thinking mode (thinking is on by default)",
			},
		},
		{
			// Thinking off is the one case where the values do take effect.
			name: "deepseek with thinking off reports no sampling caveat",
			mutate: func(c *config.Config) {
				c.Provider.Dialect = "deepseek"
				c.Provider.TopP = ptrFloat(0.9)
				c.Provider.Reasoning = &config.ReasoningConfig{Effort: "none"}
			},
			want:    []string{"top_p: 0.9 -> top_p: 0.9"},
			notWant: []string{"sent but ignored"},
		},
		{
			// The caveat is deepseek's, not every dialect's: glm and kimi have
			// their own (different) sampling rules, already covered elsewhere.
			name: "the sampling caveat is dialect-scoped",
			mutate: func(c *config.Config) {
				c.Provider.Dialect = "glm"
				c.Provider.Temperature = ptrFloat(0.2)
			},
			notWant: []string{"sent but ignored"},
		},
		{
			// No knob set, nothing to caveat.
			name: "no sampling knob, no caveat",
			mutate: func(c *config.Config) {
				c.Provider.Dialect = "deepseek"
				c.Provider.Reasoning = &config.ReasoningConfig{Effort: "high"}
			},
			notWant: []string{"sent but ignored"},
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
			// The gemini wire is a third family, not a dialect: its own cap
			// spelling, its own thinking object, and a sampling value Google
			// recommends against but still honors.
			name: "gemini maps the cap, rounds the effort and warns about sampling",
			mutate: func(c *config.Config) {
				c.Provider.Type = config.ProviderTypeGemini
				c.Provider.MaxOutputTokens = 4096
				c.Provider.Reasoning = &config.ReasoningConfig{Effort: "xhigh"}
				c.Provider.Temperature = ptrFloat(0.2)
				c.Provider.TopP = ptrFloat(0.9)
			},
			want: []string{
				"max_output_tokens: 4096 -> generationConfig.maxOutputTokens: 4096",
				"reasoning.effort: xhigh -> generationConfig.thinkingConfig.thinkingLevel: high (gemini has no level above high)",
				// The sampling knobs live under generationConfig here too, and
				// `top_p` is not a spelling this API accepts ANYWHERE: a report
				// that printed it would hand the operator a params key worth a
				// 400 - the very 400 the next row warns about.
				"temperature: 0.2 -> generationConfig.temperature: 0.2",
				"top_p: 0.9 -> generationConfig.topP: 0.9",
				"google recommends the default 1.0 on gemini 3 models; non-default may degrade output",
			},
			notWant: []string{
				"max_completion_tokens", "reasoning_effort", "dialect:",
				"-> temperature:", "-> top_p:",
			},
		},
		{
			// The note is about the VALUE, not about the wire: the recommended
			// temperature draws no caveat, or the report would cry wolf on every
			// gemini config that names one.
			name: "gemini stays quiet about the recommended temperature",
			mutate: func(c *config.Config) {
				c.Provider.Type = config.ProviderTypeGemini
				c.Provider.Temperature = ptrFloat(1)
			},
			want:    []string{"temperature: 1 -> generationConfig.temperature: 1"},
			notWant: []string{"google recommends"},
		},
		{
			// SECURITY + policy: the keys are listed, the values are not, and
			// this wire's answer to an unknown key is its own - protobuf-JSON is
			// stricter than "rejected (400)" alone conveys.
			name: "gemini reports the protobuf-json params policy",
			mutate: func(c *config.Config) {
				c.Provider.Type = config.ProviderTypeGemini
				c.Provider.Params = map[string]any{"labels": "secret-value"}
			},
			want: []string{
				`params keys (merged verbatim, values not shown): "labels"`,
				"unknown request fields: rejected (400) - strict protobuf JSON",
			},
			notWant: []string{"secret-value"},
		},
		{
			// A dialect is a validate ERROR on this wire (PROBLEMS says so), so
			// the row that would describe it must not appear at all: printing
			// `dialect: "kimi"` next to a gemini mapping would claim a knob the
			// request cannot carry.
			name: "a dialect draws no row on the gemini wire",
			mutate: func(c *config.Config) {
				c.Provider.Type = config.ProviderTypeGemini
				c.Provider.Dialect = "kimi"
				c.Provider.MaxOutputTokens = 4096
			},
			want:    []string{"max_output_tokens: 4096 -> generationConfig.maxOutputTokens: 4096"},
			notWant: []string{"dialect:", "kimi"},
		},
		{
			// The budget is the OTHER thinking spelling on this wire, and the
			// gemini-3 caveat on `none` has to be visible before the run.
			name: "gemini maps a thinking budget",
			mutate: func(c *config.Config) {
				c.Provider.Type = config.ProviderTypeGemini
				c.Provider.Reasoning = &config.ReasoningConfig{BudgetTokens: 8192}
			},
			want: []string{"reasoning.budget_tokens: 8192 -> generationConfig.thinkingConfig.thinkingBudget: 8192"},
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

// schemaTool is a tools.Tool whose definition carries a real parameter schema,
// which is what the gemini sanitizer rows are computed from.
type schemaTool struct {
	name   string
	schema string
}

func (s schemaTool) Def() llm.ToolDef {
	return llm.ToolDef{Name: s.name, Description: "stub", Parameters: json.RawMessage(s.schema)}
}

func (s schemaTool) Invoke(context.Context, string) (string, error) { return "", nil }

// registryWithSchemas builds a registry from name/schema pairs, in the order
// given - which is the order the report must print them in.
func registryWithSchemas(t *testing.T, tools_ ...schemaTool) *tools.Registry {
	t.Helper()
	reg := tools.NewRegistry()
	for _, tl := range tools_ {
		if err := reg.Register(tl); err != nil {
			t.Fatal(err)
		}
	}
	return reg
}

// TestGeminiToolSchemaGolden pins the sanitizer surface in full: this wire
// answers an unsupported JSON Schema keyword with a hard 400 that fails the
// WHOLE request, so amele strips them - and a strip that only `explain` can
// reveal is exactly the kind of silent change the report exists to prevent.
//
// A golden rather than substrings, per the docs/engineering.md §6 rule for a UI
// surface: the rows promise what the request will carry, so their content AND
// their order (registration order, which is the order the tools travel in) are
// pinned together. The fixture holds both outcomes - a schema that loses two
// keywords at two depths, and one that loses nothing - because "no keys
// stripped" is the row an operator reads as an all-clear.
//
// SECURITY: keys and paths only. A tool schema's `pattern`, `default` or
// `description` can carry operator text - and, from an MCP server, REMOTE text
// - so the row names the position, never the value.
func TestGeminiToolSchemaGolden(t *testing.T) {
	dirty := `{"type":"object","additionalProperties":false,
		"properties":{"path":{"type":"string","pattern":"^secret-value$"}}}`
	clean := `{"type":"object","properties":{"path":{"type":"string"}}}`

	cfg := baseCfg()
	cfg.Provider.Type = config.ProviderTypeGemini
	cfg.Provider.BaseURL = ""
	cfg.Provider.MaxOutputTokens = 4096
	cfg.Provider.Reasoning = &config.ReasoningConfig{BudgetTokens: 8192}
	cfg.Provider.Temperature = ptrFloat(0.2)
	cfg.Provider.Params = map[string]any{"labels": "value-not-shown"}
	reg := registryWithSchemas(t,
		schemaTool{name: "fs_write", schema: dirty},
		schemaTool{name: "tidy_tool", schema: clean},
	)
	got := Render(cfg, reg, nil, nil, alwaysFound, nil)

	for _, leaked := range []string{"secret-value", "value-not-shown"} {
		if strings.Contains(got, leaked) {
			t.Errorf("report leaked a VALUE (%q):\n%s", leaked, got)
		}
	}

	goldenPath := filepath.Join("testdata", "golden", "explain-gemini-tools.txt")
	if *update {
		if err := os.WriteFile(goldenPath, []byte(got), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(goldenPath) //nolint:gosec // G304: fixed testdata path.
	if err != nil {
		t.Fatalf("reading golden (run with -update to create): %v", err)
	}
	if got != string(want) {
		t.Errorf("report differs from golden.\ngot:\n%s\nwant:\n%s", got, want)
	}
}

// TestGeminiToolSchemaRowsAreGeminiOnly: the sanitizer runs on one wire, so no
// other config may grow a row because of it - the compatibility half of the
// feature.
func TestGeminiToolSchemaRowsAreGeminiOnly(t *testing.T) {
	for _, typ := range []string{config.ProviderTypeOpenAI, config.ProviderTypeAnthropic} {
		cfg := baseCfg()
		cfg.Provider.Type = typ
		reg := registryWithSchemas(t, schemaTool{name: "fs_write", schema: `{"additionalProperties":false}`})
		if got := Render(cfg, reg, nil, nil, alwaysFound, nil); strings.Contains(got, "tool schemas") {
			t.Errorf("type %q grew a sanitizer row:\n%s", typ, got)
		}
	}
}

// TestGeminiBaseURLDefaultRow: base_url is optional on this wire (the client
// knows the AI Studio host), so the placeholder must name THAT host. Printing
// the anthropic default there - which is what the row did before gemini was
// reachable - would tell an operator their run goes somewhere it does not.
func TestGeminiBaseURLDefaultRow(t *testing.T) {
	cfg := baseCfg()
	cfg.Provider.Type = config.ProviderTypeGemini
	cfg.Provider.BaseURL = ""
	got := Render(cfg, registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil)

	if !strings.Contains(got, "base_url:        (default: generativelanguage.googleapis.com)") {
		t.Errorf("report does not name the gemini default host:\n%s", got)
	}
	if strings.Contains(got, "api.anthropic.com") {
		t.Errorf("report names the anthropic host on the gemini wire:\n%s", got)
	}
}
