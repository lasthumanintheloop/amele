package llm

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// update refreshes golden files: go test ./internal/llm -update
var update = flag.Bool("update", false, "rewrite golden files")

// ptr is the one-liner that turns a literal into the pointer the neutral
// sampling fields take ("unset" vs "explicitly 0").
func ptr[T any](v T) *T { return &v }

// baseMessages is the smallest conversation that still exercises the message
// encoding, shared by every wire golden so the goldens differ only in the
// dialect-mapped fields under test.
func baseMessages() []Message {
	return []Message{
		{Role: RoleSystem, Content: "you are a log sentry"},
		{Role: RoleUser, Content: "scan today's log"},
	}
}

// wireCase is one request rendered through one dialect. The golden file holds
// the EXACT bytes that would go on the wire (plus a trailing newline so the
// file is a well-formed text file), because the whole point of the dialect
// layer is which keys and values a provider receives.
type wireCase struct {
	name    string
	golden  string
	dialect Dialect
	req     Request
}

func wireCases() []wireCase {
	return []wireCase{
		{
			// The pre-dialect baseline: a config that asks for nothing new
			// must send exactly what it sent before dialects existed.
			name:    "openai baseline",
			golden:  "openai-baseline.json",
			dialect: DialectOpenAI,
			req:     Request{Model: "gpt-5.6-sol", Messages: baseMessages()},
		},
		{
			// The zero-value client (no dialect wired) is openai behavior.
			name:   "default dialect is openai",
			golden: "openai-baseline.json",
			req:    Request{Model: "gpt-5.6-sol", Messages: baseMessages()},
		},
		{
			name:    "openai effort none with cap",
			golden:  "openai-effort-none.json",
			dialect: DialectOpenAI,
			req: Request{
				Model:           "gpt-5.6-sol",
				Messages:        baseMessages(),
				MaxOutputTokens: 4096,
				Reasoning:       &ReasoningSpec{Effort: "none"},
			},
		},
		{
			name:    "openai sampling and params",
			golden:  "openai-sampling-params.json",
			dialect: DialectOpenAI,
			req: Request{
				Model:       "gpt-5.5",
				Messages:    baseMessages(),
				Temperature: ptr(0.2),
				TopP:        ptr(0.9),
				Extra: map[string]json.RawMessage{
					"verbosity":    json.RawMessage(`"low"`),
					"service_tier": json.RawMessage(`  "flex" `),
				},
			},
		},
		{
			// Everything at once: pins the key ORDER of the struct-encoded
			// fields as well as the position of the merged fragments.
			name:    "openai full request",
			golden:  "openai-full.json",
			dialect: DialectOpenAI,
			req: Request{
				Model: "gpt-5.6-sol",
				Messages: append(baseMessages(),
					Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call_1", Name: "fs_read", Arguments: `{"path":"app.log"}`}}},
					Message{Role: RoleTool, ToolCallID: "call_1", Content: "ERROR disk full"},
				),
				Tools: []ToolDef{{
					Name:        "fs_read",
					Description: "read a file",
					Parameters:  json.RawMessage(`{"type":"object"}`),
				}},
				ResponseFormat:  &ResponseFormat{Name: "amele_output", Schema: json.RawMessage(`{"type":"object"}`)},
				MaxOutputTokens: 65536,
				Reasoning:       &ReasoningSpec{Effort: "high"},
				Temperature:     ptr(0.0),
				Extra:           map[string]json.RawMessage{"verbosity": json.RawMessage(`"low"`)},
			},
		},
		{
			// medium is not in DeepSeek's vocabulary: it rounds UP to high,
			// visibly (see the mapping notes), and the cap changes field name.
			name:    "deepseek effort medium",
			golden:  "deepseek-effort-medium.json",
			dialect: DialectDeepSeek,
			req: Request{
				Model:           "deepseek-v4-pro",
				Messages:        baseMessages(),
				MaxOutputTokens: 65536,
				Reasoning:       &ReasoningSpec{Effort: "medium"},
			},
		},
		{
			name:    "deepseek effort none disables thinking",
			golden:  "deepseek-effort-none.json",
			dialect: DialectDeepSeek,
			req: Request{
				Model:     "deepseek-v4-pro",
				Messages:  baseMessages(),
				Reasoning: &ReasoningSpec{Effort: "none"},
			},
		},
		{
			name:    "glm effort xhigh with sampling",
			golden:  "glm-effort-xhigh.json",
			dialect: DialectGLM,
			req: Request{
				Model:           "glm-5.3",
				Messages:        baseMessages(),
				MaxOutputTokens: 131072,
				Reasoning:       &ReasoningSpec{Effort: "xhigh"},
				Temperature:     ptr(0.6),
			},
		},
		{
			// Kimi takes the level top-level and never the thinking object.
			name:    "kimi effort medium",
			golden:  "kimi-effort-medium.json",
			dialect: DialectKimi,
			req: Request{
				Model:           "kimi-k3",
				Messages:        baseMessages(),
				MaxOutputTokens: 131072,
				Reasoning:       &ReasoningSpec{Effort: "medium"},
			},
		},
		{
			// Groq speaks plain OpenAI: no rounding, xhigh goes as xhigh.
			name:    "groq effort xhigh",
			golden:  "groq-effort-xhigh.json",
			dialect: DialectGroq,
			req: Request{
				Model:           "moonshotai/kimi-k3",
				Messages:        baseMessages(),
				MaxOutputTokens: 32768,
				Reasoning:       &ReasoningSpec{Effort: "xhigh"},
			},
		},
		{
			name:    "openrouter effort object",
			golden:  "openrouter-effort-high.json",
			dialect: DialectOpenRouter,
			req: Request{
				Model:           "openai/gpt-5.6-sol",
				Messages:        baseMessages(),
				MaxOutputTokens: 16384,
				Reasoning:       &ReasoningSpec{Effort: "high"},
			},
		},
		{
			// effort and max_tokens are mutually exclusive on OpenRouter: the
			// budget wins and the effort is reported as not sent.
			name:    "openrouter budget wins over effort",
			golden:  "openrouter-budget.json",
			dialect: DialectOpenRouter,
			req: Request{
				Model:     "anthropic/claude-opus-5",
				Messages:  baseMessages(),
				Reasoning: &ReasoningSpec{Effort: "high", BudgetTokens: 8192},
			},
		},
	}
}

// TestToWireGolden is the dialect contract in bytes: for each dialect and knob
// combination, exactly these keys with exactly these values leave the process.
func TestToWireGolden(t *testing.T) {
	for _, tc := range wireCases() {
		t.Run(tc.name, func(t *testing.T) {
			client := &OpenAIClient{Dialect: tc.dialect}
			wire, fields := client.toWire(tc.req)
			got, err := encodeBody(wire, fields)
			if err != nil {
				t.Fatalf("encodeBody: %v", err)
			}
			if !json.Valid(got) {
				t.Fatalf("body is not valid JSON: %s", got)
			}

			goldenPath := filepath.Join("testdata", "golden", "wire", tc.golden)
			if *update {
				if err := os.MkdirAll(filepath.Dir(goldenPath), 0o750); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(goldenPath, append(got, '\n'), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			raw, err := os.ReadFile(goldenPath) //nolint:gosec // G304: fixed testdata path.
			if err != nil {
				t.Fatalf("reading golden (run with -update to create): %v", err)
			}
			want := strings.TrimSuffix(string(raw), "\n")
			if string(got) != want {
				t.Errorf("request body differs from golden %s.\ngot:  %s\nwant: %s", tc.golden, got, want)
			}
		})
	}
}

// TestMapReasoningMatrix walks every dialect against every effort in the
// neutral vocabulary. It is the completeness check the dialect table needs:
// a new dialect or a new effort value has to be answered here explicitly.
func TestMapReasoningMatrix(t *testing.T) {
	tests := []struct {
		dialect    Dialect
		spec       ReasoningSpec
		wantFields map[string]string
		wantNotes  []string
	}{
		// openai: the full vocabulary passes through untouched.
		{DialectOpenAI, ReasoningSpec{Effort: "none"}, map[string]string{"reasoning_effort": `"none"`}, []string{"reasoning.effort: none -> reasoning_effort: none"}},
		{DialectOpenAI, ReasoningSpec{Effort: "low"}, map[string]string{"reasoning_effort": `"low"`}, []string{"reasoning.effort: low -> reasoning_effort: low"}},
		{DialectOpenAI, ReasoningSpec{Effort: "medium"}, map[string]string{"reasoning_effort": `"medium"`}, []string{"reasoning.effort: medium -> reasoning_effort: medium"}},
		{DialectOpenAI, ReasoningSpec{Effort: "high"}, map[string]string{"reasoning_effort": `"high"`}, []string{"reasoning.effort: high -> reasoning_effort: high"}},
		{DialectOpenAI, ReasoningSpec{Effort: "xhigh"}, map[string]string{"reasoning_effort": `"xhigh"`}, []string{"reasoning.effort: xhigh -> reasoning_effort: xhigh"}},
		{DialectOpenAI, ReasoningSpec{Effort: "max"}, map[string]string{"reasoning_effort": `"max"`}, []string{"reasoning.effort: max -> reasoning_effort: max"}},
		{DialectOpenAI, ReasoningSpec{}, nil, nil},

		// groq: same wire as openai, including no rounding.
		{DialectGroq, ReasoningSpec{Effort: "none"}, map[string]string{"reasoning_effort": `"none"`}, []string{"reasoning.effort: none -> reasoning_effort: none"}},
		{DialectGroq, ReasoningSpec{Effort: "medium"}, map[string]string{"reasoning_effort": `"medium"`}, []string{"reasoning.effort: medium -> reasoning_effort: medium"}},
		{DialectGroq, ReasoningSpec{Effort: "xhigh"}, map[string]string{"reasoning_effort": `"xhigh"`}, []string{"reasoning.effort: xhigh -> reasoning_effort: xhigh"}},
		{DialectGroq, ReasoningSpec{}, nil, nil},

		// deepseek: thinking object + top-level level, rounded to low/high/max.
		{DialectDeepSeek, ReasoningSpec{Effort: "none"}, map[string]string{"thinking": `{"type":"disabled"}`},
			[]string{`reasoning.effort: none -> thinking: {"type":"disabled"} (deepseek sends no reasoning_effort with thinking off)`}},
		{DialectDeepSeek, ReasoningSpec{Effort: "low"}, map[string]string{"thinking": `{"type":"enabled"}`, "reasoning_effort": `"low"`},
			[]string{`reasoning.effort: low -> thinking: {"type":"enabled"}`, "reasoning.effort: low -> reasoning_effort: low"}},
		{DialectDeepSeek, ReasoningSpec{Effort: "medium"}, map[string]string{"thinking": `{"type":"enabled"}`, "reasoning_effort": `"high"`},
			[]string{`reasoning.effort: medium -> thinking: {"type":"enabled"}`, "reasoning.effort: medium -> reasoning_effort: high (deepseek has no medium)"}},
		{DialectDeepSeek, ReasoningSpec{Effort: "high"}, map[string]string{"thinking": `{"type":"enabled"}`, "reasoning_effort": `"high"`},
			[]string{`reasoning.effort: high -> thinking: {"type":"enabled"}`, "reasoning.effort: high -> reasoning_effort: high"}},
		{DialectDeepSeek, ReasoningSpec{Effort: "xhigh"}, map[string]string{"thinking": `{"type":"enabled"}`, "reasoning_effort": `"max"`},
			[]string{`reasoning.effort: xhigh -> thinking: {"type":"enabled"}`, "reasoning.effort: xhigh -> reasoning_effort: max (deepseek has no xhigh)"}},
		{DialectDeepSeek, ReasoningSpec{Effort: "max"}, map[string]string{"thinking": `{"type":"enabled"}`, "reasoning_effort": `"max"`},
			[]string{`reasoning.effort: max -> thinking: {"type":"enabled"}`, "reasoning.effort: max -> reasoning_effort: max"}},
		{DialectDeepSeek, ReasoningSpec{}, nil, nil},

		// glm: same emission and the same rounding as deepseek.
		{DialectGLM, ReasoningSpec{Effort: "none"}, map[string]string{"thinking": `{"type":"disabled"}`},
			[]string{`reasoning.effort: none -> thinking: {"type":"disabled"} (glm sends no reasoning_effort with thinking off)`}},
		{DialectGLM, ReasoningSpec{Effort: "medium"}, map[string]string{"thinking": `{"type":"enabled"}`, "reasoning_effort": `"high"`},
			[]string{`reasoning.effort: medium -> thinking: {"type":"enabled"}`, "reasoning.effort: medium -> reasoning_effort: high (glm has no medium)"}},
		{DialectGLM, ReasoningSpec{Effort: "xhigh"}, map[string]string{"thinking": `{"type":"enabled"}`, "reasoning_effort": `"max"`},
			[]string{`reasoning.effort: xhigh -> thinking: {"type":"enabled"}`, "reasoning.effort: xhigh -> reasoning_effort: max (glm has no xhigh)"}},
		{DialectGLM, ReasoningSpec{}, nil, nil},

		// kimi: level only, never a thinking object; none is unreachable
		// through validate and is reported as not sent rather than guessed at.
		{DialectKimi, ReasoningSpec{Effort: "none"}, nil,
			[]string{"reasoning.effort: none not sent: kimi models cannot disable thinking"}},
		{DialectKimi, ReasoningSpec{Effort: "low"}, map[string]string{"reasoning_effort": `"low"`},
			[]string{"reasoning.effort: low -> reasoning_effort: low"}},
		{DialectKimi, ReasoningSpec{Effort: "medium"}, map[string]string{"reasoning_effort": `"high"`},
			[]string{"reasoning.effort: medium -> reasoning_effort: high (kimi has no medium)"}},
		{DialectKimi, ReasoningSpec{Effort: "high"}, map[string]string{"reasoning_effort": `"high"`},
			[]string{"reasoning.effort: high -> reasoning_effort: high"}},
		{DialectKimi, ReasoningSpec{Effort: "xhigh"}, map[string]string{"reasoning_effort": `"max"`},
			[]string{"reasoning.effort: xhigh -> reasoning_effort: max (kimi has no xhigh)"}},
		{DialectKimi, ReasoningSpec{Effort: "max"}, map[string]string{"reasoning_effort": `"max"`},
			[]string{"reasoning.effort: max -> reasoning_effort: max"}},
		{DialectKimi, ReasoningSpec{}, nil, nil},

		// openrouter: one unified object, effort passed through untouched.
		{DialectOpenRouter, ReasoningSpec{Effort: "none"}, map[string]string{"reasoning": `{"effort":"none"}`},
			[]string{`reasoning.effort: none -> reasoning: {"effort":"none"}`}},
		{DialectOpenRouter, ReasoningSpec{Effort: "medium"}, map[string]string{"reasoning": `{"effort":"medium"}`},
			[]string{`reasoning.effort: medium -> reasoning: {"effort":"medium"}`}},
		{DialectOpenRouter, ReasoningSpec{Effort: "xhigh"}, map[string]string{"reasoning": `{"effort":"xhigh"}`},
			[]string{`reasoning.effort: xhigh -> reasoning: {"effort":"xhigh"}`}},
		{DialectOpenRouter, ReasoningSpec{BudgetTokens: 8192}, map[string]string{"reasoning": `{"max_tokens":8192}`},
			[]string{`reasoning.budget_tokens: 8192 -> reasoning: {"max_tokens":8192}`}},
		{DialectOpenRouter, ReasoningSpec{Effort: "high", BudgetTokens: 8192}, map[string]string{"reasoning": `{"max_tokens":8192}`},
			[]string{
				`reasoning.budget_tokens: 8192 -> reasoning: {"max_tokens":8192}`,
				"reasoning.effort: high not sent: openrouter takes effort or max_tokens, not both",
			}},
		{DialectOpenRouter, ReasoningSpec{}, nil, nil},
	}

	for _, tt := range tests {
		name := string(tt.dialect) + "/" + tt.spec.Effort
		if tt.spec.Effort == "" {
			name = string(tt.dialect) + "/unset"
		}
		if tt.spec.BudgetTokens > 0 {
			name += "+budget"
		}
		t.Run(name, func(t *testing.T) {
			got := MapReasoning(tt.dialect, tt.spec)
			if len(got.Fields) != len(tt.wantFields) {
				t.Fatalf("fields = %v, want %v", got.Fields, tt.wantFields)
			}
			for key, want := range tt.wantFields {
				value, ok := got.Fields[key]
				if !ok {
					t.Fatalf("field %q missing from %v", key, got.Fields)
				}
				if string(value) != want {
					t.Errorf("field %q = %s, want %s", key, value, want)
				}
			}
			if len(got.Notes) != len(tt.wantNotes) {
				t.Fatalf("notes = %q, want %q", got.Notes, tt.wantNotes)
			}
			for i, want := range tt.wantNotes {
				if got.Notes[i] != want {
					t.Errorf("note %d = %q, want %q", i, got.Notes[i], want)
				}
			}
		})
	}
}

// TestCapFieldPerDialect pins the split the research doc records: the
// reasoning-era OpenAI family answers max_tokens with a 400, while the CN
// natives and OpenRouter only know max_tokens.
func TestCapFieldPerDialect(t *testing.T) {
	tests := map[Dialect]string{
		DialectOpenAI:     "max_completion_tokens",
		DialectGroq:       "max_completion_tokens",
		DialectKimi:       "max_completion_tokens",
		DialectDeepSeek:   "max_tokens",
		DialectGLM:        "max_tokens",
		DialectOpenRouter: "max_tokens",
		Dialect(""):       "max_completion_tokens", // zero value = openai
	}
	for dialect, want := range tests {
		t.Run(string(dialect), func(t *testing.T) {
			if got := CapField(dialect); got != want {
				t.Errorf("CapField(%q) = %q, want %q", dialect, got, want)
			}
			client := &OpenAIClient{Dialect: dialect}
			wire, _ := client.toWire(Request{Model: "m", Messages: baseMessages(), MaxOutputTokens: 4096})
			body, err := json.Marshal(wire)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(body), `"`+want+`":4096`) {
				t.Errorf("body %s does not carry %q", body, want)
			}
			other := "max_tokens"
			if want == other {
				other = "max_completion_tokens"
			}
			if strings.Contains(string(body), `"`+other+`"`) {
				t.Errorf("body %s carries the wrong cap field %q", body, other)
			}
		})
	}
}

// TestNoCapWithoutRequest is the "silence stays silence" rule: a config that
// sets no cap must leave the provider default alone, on every dialect.
func TestNoCapWithoutRequest(t *testing.T) {
	for _, dialect := range dialects {
		client := &OpenAIClient{Dialect: dialect}
		wire, _ := client.toWire(Request{Model: "m", Messages: baseMessages()})
		body, err := json.Marshal(wire)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(body), "max_tokens") || strings.Contains(string(body), "max_completion_tokens") {
			t.Errorf("dialect %q sent a cap that was never asked for: %s", dialect, body)
		}
	}
}

// TestSamplingSentOnEveryDialect covers the "pass through everywhere" rule:
// the only dialect that forbids sampling (kimi) is stopped at validate, so the
// client itself never drops the fields. An explicit zero must survive too.
func TestSamplingSentOnEveryDialect(t *testing.T) {
	for _, dialect := range dialects {
		client := &OpenAIClient{Dialect: dialect}
		wire, _ := client.toWire(Request{
			Model:       "m",
			Messages:    baseMessages(),
			Temperature: ptr(0.0),
			TopP:        ptr(1.0),
		})
		body, err := json.Marshal(wire)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(body), `"temperature":0`) {
			t.Errorf("dialect %q dropped an explicit temperature 0: %s", dialect, body)
		}
		if !strings.Contains(string(body), `"top_p":1`) {
			t.Errorf("dialect %q dropped top_p: %s", dialect, body)
		}
	}
}

// TestNoReasoningSpecSendsNoKnob pins the difference between "unset" and
// "explicitly none": a nil ReasoningSpec must not put any reasoning key on the
// wire, not even for the dialects whose providers think by default.
func TestNoReasoningSpecSendsNoKnob(t *testing.T) {
	for _, dialect := range dialects {
		client := &OpenAIClient{Dialect: dialect}
		_, fields := client.toWire(Request{Model: "m", Messages: baseMessages()})
		if len(fields) != 0 {
			t.Errorf("dialect %q sent %v for a request with no reasoning knob", dialect, fields)
		}
	}
}

// TestMergeBodyFields tests the post-marshal merge on its own: it is the one
// place where amele hand-writes JSON, so its edge cases are pinned directly
// rather than only through the goldens.
func TestMergeBodyFields(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		fields  map[string]json.RawMessage
		want    string
		wantErr bool
	}{
		{
			name:   "no fields is a passthrough",
			body:   `{"model":"m"}`,
			fields: nil,
			want:   `{"model":"m"}`,
		},
		{
			// Sorted keys, so the bytes are reproducible across runs (Go map
			// iteration order is randomized) and the goldens are stable.
			name: "keys are merged in sorted order",
			body: `{"model":"m"}`,
			fields: map[string]json.RawMessage{
				"zeta":  json.RawMessage(`1`),
				"alpha": json.RawMessage(`{"a":true}`),
				"mid":   json.RawMessage(`"x"`),
			},
			want: `{"model":"m","alpha":{"a":true},"mid":"x","zeta":1}`,
		},
		{
			name:   "empty object gets no leading comma",
			body:   `{}`,
			fields: map[string]json.RawMessage{"a": json.RawMessage(`1`)},
			want:   `{"a":1}`,
		},
		{
			// Whatever the caller pre-serialized is compacted, so indentation
			// from the YAML->JSON conversion cannot leak into the wire bytes.
			name:   "values are compacted",
			body:   `{"model":"m"}`,
			fields: map[string]json.RawMessage{"a": json.RawMessage("{\n  \"b\": 1\n}")},
			want:   `{"model":"m","a":{"b":1}}`,
		},
		{
			name:   "keys are escaped",
			body:   `{"model":"m"}`,
			fields: map[string]json.RawMessage{`we"ird`: json.RawMessage(`1`)},
			want:   `{"model":"m","we\"ird":1}`,
		},
		{
			name:    "invalid value is an error, not a corrupt body",
			body:    `{"model":"m"}`,
			fields:  map[string]json.RawMessage{"a": json.RawMessage(`{oops`)},
			wantErr: true,
		},
		{
			name:    "empty value is an error",
			body:    `{"model":"m"}`,
			fields:  map[string]json.RawMessage{"a": nil},
			wantErr: true,
		},
		{
			name:    "non-object body is refused",
			body:    `[1,2]`,
			fields:  map[string]json.RawMessage{"a": json.RawMessage(`1`)},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := mergeBodyFields([]byte(tt.body), tt.fields)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected an error, got %s", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("mergeBodyFields: %v", err)
			}
			if string(got) != tt.want {
				t.Errorf("merged = %s, want %s", got, tt.want)
			}
			if !json.Valid(got) {
				t.Errorf("merged body is not valid JSON: %s", got)
			}
		})
	}
}

// TestEncodeBodyRejectsBadExtra makes sure a broken params fragment fails the
// request loudly instead of shipping a malformed body the provider answers
// with an opaque 400.
func TestEncodeBodyRejectsBadExtra(t *testing.T) {
	client := &OpenAIClient{}
	wire, fields := client.toWire(Request{
		Model:    "m",
		Messages: baseMessages(),
		Extra:    map[string]json.RawMessage{"bad": json.RawMessage(`{`)},
	})
	if _, err := encodeBody(wire, fields); err == nil {
		t.Fatal("expected an error for an unparseable params value")
	}
}
