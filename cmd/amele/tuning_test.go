package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/llm"
)

// ptrFloat is the local pointer helper for the sampling knobs, which are
// pointers precisely so an explicit 0 is not "unset".
func ptrFloat(f float64) *float64 { return &f }

// TestProviderTuning pins the translation from the config surface to the
// neutral request knobs the loop forwards: the ONE place where YAML params
// become JSON, and where an empty reasoning block becomes no knob at all.
func TestProviderTuning(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*config.Config)
		want    func() loopTuningWant
		wantErr bool
	}{
		{
			name:   "nothing set sends nothing",
			mutate: func(*config.Config) {},
			want:   func() loopTuningWant { return loopTuningWant{} },
		},
		{
			name: "every knob is carried",
			mutate: func(c *config.Config) {
				c.Provider.MaxOutputTokens = 65536
				c.Provider.Reasoning = &config.ReasoningConfig{Effort: "high", BudgetTokens: 8192}
				c.Provider.Temperature = ptrFloat(0.2)
				c.Provider.TopP = ptrFloat(0.9)
				c.Provider.Params = map[string]any{"verbosity": "low"}
			},
			want: func() loopTuningWant {
				return loopTuningWant{
					maxOutputTokens: 65536,
					reasoning:       &llm.ReasoningSpec{Effort: "high", BudgetTokens: 8192},
					temperature:     ptrFloat(0.2),
					topP:            ptrFloat(0.9),
					extra:           map[string]json.RawMessage{"verbosity": json.RawMessage(`"low"`)},
				}
			},
		},
		{
			// The exact shape `--set provider.reasoning.effort=` leaves behind
			// on a config whose YAML had a reasoning block: present but empty.
			// It must mean what an absent block means - no knob on the wire -
			// rather than an "effort: ''" the client has to interpret.
			name: "an empty reasoning block is no reasoning knob",
			mutate: func(c *config.Config) {
				c.Provider.Reasoning = &config.ReasoningConfig{}
			},
			want: func() loopTuningWant { return loopTuningWant{} },
		},
		{
			name: "a zero temperature is still a setting",
			mutate: func(c *config.Config) {
				c.Provider.Temperature = ptrFloat(0)
			},
			want: func() loopTuningWant { return loopTuningWant{temperature: ptrFloat(0)} },
		},
		{
			name: "nested params become nested JSON",
			mutate: func(c *config.Config) {
				c.Provider.Params = map[string]any{
					"provider": map[string]any{"require_parameters": true},
				}
			},
			want: func() loopTuningWant {
				return loopTuningWant{
					extra: map[string]json.RawMessage{"provider": json.RawMessage(`{"require_parameters":true}`)},
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.Config{Model: "m"}
			tt.mutate(cfg)

			got, err := providerTuning(cfg)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("providerTuning = %+v, want an error", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("providerTuning: %v", err)
			}
			want := tt.want()
			if got.MaxOutputTokens != want.maxOutputTokens {
				t.Errorf("MaxOutputTokens = %d, want %d", got.MaxOutputTokens, want.maxOutputTokens)
			}
			if !reflect.DeepEqual(got.Reasoning, want.reasoning) {
				t.Errorf("Reasoning = %+v, want %+v", got.Reasoning, want.reasoning)
			}
			if !reflect.DeepEqual(got.Temperature, want.temperature) || !reflect.DeepEqual(got.TopP, want.topP) {
				t.Errorf("sampling = %v/%v, want %v/%v", got.Temperature, got.TopP, want.temperature, want.topP)
			}
			if !reflect.DeepEqual(got.Extra, want.extra) {
				t.Errorf("Extra = %v, want %v", got.Extra, want.extra)
			}
		})
	}
}

// loopTuningWant is the expectation side of TestProviderTuning, spelled out
// field by field so a case says what the request will carry.
type loopTuningWant struct {
	maxOutputTokens int
	reasoning       *llm.ReasoningSpec
	temperature     *float64
	topP            *float64
	extra           map[string]json.RawMessage
}

// TestBuildProviderDialect pins that the config's dialect reaches the client
// that speaks it. Nothing downstream can recover a dialect that is dropped
// here: the wire mapping is chosen per request from this field.
func TestBuildProviderDialect(t *testing.T) {
	cfg := &config.Config{Model: "m", Provider: config.ProviderConfig{
		BaseURL: "https://api.deepseek.com", Dialect: "deepseek",
	}}
	provider, err := buildProvider(cfg)
	if err != nil {
		t.Fatalf("buildProvider: %v", err)
	}
	client, ok := provider.(*llm.OpenAIClient)
	if !ok {
		t.Fatalf("provider is %T, want *llm.OpenAIClient", provider)
	}
	if client.Dialect != llm.DialectDeepSeek {
		t.Errorf("Dialect = %q, want %q", client.Dialect, llm.DialectDeepSeek)
	}
}

// TestBuildProviderRejectsUnknownDialect: validate cannot be bypassed, but the
// wiring must not paper over an unparseable dialect either - a client silently
// falling back to the openai mapping would reshape every request.
func TestBuildProviderRejectsUnknownDialect(t *testing.T) {
	cfg := &config.Config{Model: "m", Provider: config.ProviderConfig{
		BaseURL: "https://example.test/v1", Dialect: "gemini",
	}}
	if _, err := buildProvider(cfg); err == nil {
		t.Fatal("an unknown dialect must be an error, not a silent openai fallback")
	}
}

// rawCapturingServer records the RAW request body of every provider call as a
// map of top-level keys, which is what a wire assertion needs: a decoded struct
// cannot tell "the field was absent" from "the field was zero", and absence is
// exactly what the untuned cases assert.
func rawCapturingServer(t *testing.T, bodies ...string) (*httptest.Server, *[]map[string]json.RawMessage) {
	t.Helper()
	var (
		call int
		reqs []map[string]json.RawMessage
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got := map[string]json.RawMessage{}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		reqs = append(reqs, got)
		if call >= len(bodies) {
			t.Errorf("unexpected extra provider call #%d", call+1)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte(bodies[call]))
		call++
	}))
	t.Cleanup(srv.Close)
	return srv, &reqs
}

// TestE2ERunSendsTunedRequest is the end-to-end proof of the wiring: a YAML
// file naming a dialect and every tuning knob produces a request body carrying
// that dialect's spellings - the deepseek thinking object, its rounded effort,
// its max_tokens cap - plus the raw params key, on a real (local) HTTP round
// trip.
func TestE2ERunSendsTunedRequest(t *testing.T) {
	srv, reqs := rawCapturingServer(t, textBody("done"))
	cfgPath := writeTunedConfig(t, srv.URL, `
  dialect: deepseek
  max_output_tokens: 4096
  reasoning:
    effort: medium
  temperature: 0.2
  top_p: 0.9
  params:
    verbosity: low
`)

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "task"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if len(*reqs) != 1 {
		t.Fatalf("provider calls: %d want 1", len(*reqs))
	}
	body := (*reqs)[0]
	want := map[string]string{
		// medium rounds UP to high: deepseek has no medium level.
		"reasoning_effort": `"high"`,
		"thinking":         `{"type":"enabled"}`,
		"max_tokens":       "4096",
		"temperature":      "0.2",
		"top_p":            "0.9",
		"verbosity":        `"low"`,
	}
	for key, value := range want {
		if got := string(body[key]); got != value {
			t.Errorf("request body %q = %q, want %q", key, got, value)
		}
	}
	// The cap field is dialect-chosen: sending the OpenAI spelling to DeepSeek
	// would be an uncapped run at best.
	if _, ok := body["max_completion_tokens"]; ok {
		t.Errorf("deepseek request carries max_completion_tokens: %v", body)
	}
}

// TestE2ERunClearedReasoningSendsNoKnob: `--set provider.reasoning.effort=`
// clears the depth the YAML asked for, and a cleared depth must leave NO
// reasoning field on the wire - not an empty one for the provider to interpret.
func TestE2ERunClearedReasoningSendsNoKnob(t *testing.T) {
	srv, reqs := rawCapturingServer(t, textBody("done"))
	cfgPath := writeTunedConfig(t, srv.URL, `
  dialect: deepseek
  reasoning:
    effort: high
`)

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "--set", "provider.reasoning.effort=", "task"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	body := (*reqs)[0]
	for _, key := range []string{"reasoning_effort", "thinking", "reasoning"} {
		if _, ok := body[key]; ok {
			t.Errorf("cleared reasoning still sent %q: %v", key, body)
		}
	}
}

// TestE2ERunUntunedRequestIsUnchanged: a config that names no tuning at all
// sends exactly what it sent before dialects existed. This is the compatibility
// guarantee of the whole slice.
func TestE2ERunUntunedRequestIsUnchanged(t *testing.T) {
	srv, reqs := rawCapturingServer(t, textBody("done"))
	cfgPath := writeTunedConfig(t, srv.URL, "")

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "task"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	body := (*reqs)[0]
	for _, key := range []string{
		"reasoning_effort", "thinking", "reasoning", "temperature", "top_p",
		"max_tokens", "max_completion_tokens",
	} {
		if _, ok := body[key]; ok {
			t.Errorf("untuned request carries %q: %v", key, body)
		}
	}
	if got := string(body["model"]); got != `"test-model"` {
		t.Errorf("model = %s", got)
	}
}

// writeTunedConfig writes a minimal runnable config whose provider block is
// extended with the given YAML fragment (already indented two spaces).
func writeTunedConfig(t *testing.T, baseURL, providerExtra string) string {
	t.Helper()
	dir := t.TempDir()
	yaml := fmt.Sprintf(`
model: test-model
provider:
  base_url: %s/v1
  api_key: ${TEST_KEY}%s
system_prompt: "You are a test agent."
`, baseURL, providerExtra)
	path := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestExplainProviderMappingGolden pins the whole report for the two wires
// that map the tuning knobs differently: an openai-wire dialect that renames
// the cap and rounds the effort, and the anthropic wire where a thinking budget
// beats an effort. Goldens rather than substrings, per the docs/engineering.md
// §6 rule for a UI surface: the mapping rows are the promise explain makes
// about a run, so they are pinned in their entirety and in their order.
func TestExplainProviderMappingGolden(t *testing.T) {
	cases := []struct {
		name   string
		golden string
		yaml   string
	}{
		{
			// base_url deliberately does NOT match the dialect: the report
			// must carry the hint row too.
			name:   "openai wire with a dialect",
			golden: "explain-dialect.txt",
			yaml: `model: golden-model
provider:
  base_url: https://api.groq.com/openai/v1
  api_key: ${TEST_KEY}
  dialect: deepseek
  max_output_tokens: 4096
  reasoning:
    effort: medium
  temperature: 0.2
  top_p: 0.9
  params:
    verbosity: low
    provider:
      require_parameters: true
`,
		},
		{
			name:   "anthropic wire with a thinking budget",
			golden: "explain-anthropic-tuning.txt",
			yaml: `model: golden-model
provider:
  type: anthropic
  api_key: ${TEST_KEY}
  dialect: kimi
  max_output_tokens: 16384
  reasoning:
    effort: max
    budget_tokens: 8192
  temperature: 0.5
  params:
    top_k: 40
`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			cfgPath := filepath.Join(dir, "agent.yaml")
			if err := os.WriteFile(cfgPath, []byte(tc.yaml), 0o600); err != nil {
				t.Fatal(err)
			}

			code, stdout, stderr := execCLI(t, []string{"explain", cfgPath}, "")
			if code != ExitOK {
				t.Fatalf("exit %d, stderr: %s", code, stderr)
			}
			got := strings.ReplaceAll(stdout, dir, "<TMP>")

			goldenPath := filepath.Join("testdata", "golden", tc.golden)
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
				t.Errorf("explain output differs from golden.\ngot:\n%s\nwant:\n%s", got, want)
			}
		})
	}
}

// TestExplainProviderMappingRedactsNothingFromParams is the SECURITY assertion
// on the report: provider.params values may embed a credential (a routing token
// under a provider-specific key), so explain prints the KEYS only.
func TestExplainProviderMappingHidesParamValues(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "agent.yaml")
	yaml := `
model: test-model
provider:
  base_url: https://api.example.com/v1
  api_key: ${TEST_KEY}
  params:
    routing_token: super-secret-routing-value
`
	if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	code, stdout, stderr := execCLI(t, []string{"explain", path}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if !strings.Contains(stdout, "routing_token") {
		t.Errorf("report does not list the params key:\n%s", stdout)
	}
	if strings.Contains(stdout, "super-secret-routing-value") {
		t.Errorf("report leaked a params VALUE:\n%s", stdout)
	}
}
