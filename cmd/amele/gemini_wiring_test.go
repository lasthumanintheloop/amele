package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/llm"
)

// geminiTextBody is a minimal successful generateContent response: one model
// candidate carrying text, plus the usage block the loop's budget reads.
func geminiTextBody(text string) string {
	b, _ := json.Marshal(text)
	return fmt.Sprintf(`{
		"candidates": [{"content": {"role": "model", "parts": [{"text": %s}]}, "finishReason": "STOP"}],
		"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}
	}`, b)
}

// geminiCall is one recorded request: enough of it to prove the wiring reached
// the native endpoint rather than the OpenAI-compatible one.
type geminiCall struct {
	path   string
	apiKey string
	body   map[string]json.RawMessage
}

// geminiCapturingServer records every generateContent call and answers with the
// given bodies in order.
func geminiCapturingServer(t *testing.T, bodies ...string) (*httptest.Server, *[]geminiCall) {
	t.Helper()
	var (
		call  int
		calls []geminiCall
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body := map[string]json.RawMessage{}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decoding request body: %v", err)
		}
		calls = append(calls, geminiCall{path: r.URL.Path, apiKey: r.Header.Get("x-goog-api-key"), body: body})
		if call >= len(bodies) {
			t.Errorf("unexpected extra provider call #%d", call+1)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte(bodies[call]))
		call++
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

// writeGeminiConfig writes a runnable `type: gemini` config whose provider block
// is extended with the given already-indented YAML fragment.
func writeGeminiConfig(t *testing.T, baseURL, providerExtra, extra string) string {
	t.Helper()
	dir := t.TempDir()
	yaml := fmt.Sprintf(`
model: gemini-3-pro
provider:
  type: gemini
  base_url: %s
  api_key: ${TEST_KEY}%s
system_prompt: "You are a test agent."%s
`, baseURL, providerExtra, extra)
	path := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestBuildProviderSelectsGemini pins the third branch of the type-to-client
// mapping. Nothing downstream can recover a wrong choice here: the OpenAI
// client would POST chat/completions to the Gemini host and fail every run.
func TestBuildProviderSelectsGemini(t *testing.T) {
	pc := config.ProviderConfig{
		Type:            config.ProviderTypeGemini,
		BaseURL:         "https://gemini.example.com",
		APIKey:          "k",
		RequestTimeout:  config.Duration(5 * time.Second),
		MaxOutputTokens: 256,
	}

	provider, err := buildProvider(&config.Config{Model: "m", Provider: pc})
	if err != nil {
		t.Fatalf("buildProvider: %v", err)
	}
	client, ok := provider.(*llm.GeminiClient)
	if !ok {
		t.Fatalf("provider is %T, want *llm.GeminiClient", provider)
	}
	if client.BaseURL != pc.BaseURL || client.APIKey != pc.APIKey || client.RequestTimeout != 5*time.Second {
		t.Errorf("gemini fields not wired: %+v", client)
	}
}

// TestBuildProviderGeminiWiresRetry: the retry block reaches the third client
// too, and a config without one leaves it on the llm package's own defaults -
// the zero values - rather than a number cmd invented.
func TestBuildProviderGeminiWiresRetry(t *testing.T) {
	base := config.ProviderConfig{Type: config.ProviderTypeGemini, APIKey: "k"}

	provider, err := buildProvider(&config.Config{Provider: base})
	if err != nil {
		t.Fatalf("buildProvider: %v", err)
	}
	if c := provider.(*llm.GeminiClient); c.MaxAttempts != 0 || c.InitialBackoff != 0 {
		t.Errorf("gemini retry knobs: got %d/%v, want the zero values", c.MaxAttempts, c.InitialBackoff)
	}

	tuned := base
	tuned.Retry = &config.RetryConfig{MaxAttempts: 5, InitialBackoff: config.Duration(250 * time.Millisecond)}
	provider, err = buildProvider(&config.Config{Provider: tuned})
	if err != nil {
		t.Fatalf("buildProvider: %v", err)
	}
	if c := provider.(*llm.GeminiClient); c.MaxAttempts != 5 || c.InitialBackoff != 250*time.Millisecond {
		t.Errorf("gemini retry knobs not wired: %d/%v", c.MaxAttempts, c.InitialBackoff)
	}
}

// TestGeminiWithoutAPIKeyIsExit2: slice 1 speaks only the AI Studio key, so a
// keyless config must fail at validate - naming Vertex, the auth path an
// operator who wrote one was probably reaching for - instead of buying a 401
// from an unattended run.
func TestGeminiWithoutAPIKeyIsExit2(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "agent.yaml")
	yaml := `
model: gemini-3-pro
provider:
  type: gemini
system_prompt: "You are a test agent."
`
	if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, cmd := range []string{"validate", "run"} {
		args := []string{cmd, path}
		if cmd == "run" {
			args = append(args, "task")
		}
		code, _, stderr := execCLI(t, args, "")
		if code != ExitConfigError {
			t.Errorf("%s: exit %d, want %d (stderr: %s)", cmd, code, ExitConfigError, stderr)
		}
		if !strings.Contains(stderr, "gemini needs api_key (Vertex support lands with the vertex block)") {
			t.Errorf("%s: stderr does not name the missing key: %s", cmd, stderr)
		}
	}
}

// TestE2EGeminiRunSendsTunedRequest is the end-to-end proof that `type: gemini`
// reaches the native wire with every tuning knob mapped: the versioned
// generateContent path, the key header, the generationConfig spellings and the
// raw params key merged at the body ROOT.
func TestE2EGeminiRunSendsTunedRequest(t *testing.T) {
	srv, calls := geminiCapturingServer(t, geminiTextBody("done"))
	cfgPath := writeGeminiConfig(t, srv.URL, `
  max_output_tokens: 4096
  reasoning:
    effort: medium
  temperature: 0.2
  top_p: 0.9
  params:
    labels:
      team: ops
`, "")

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "task"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "done\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if len(*calls) != 1 {
		t.Fatalf("provider calls: %d want 1", len(*calls))
	}
	got := (*calls)[0]
	if got.path != "/v1beta/models/gemini-3-pro:generateContent" {
		t.Errorf("request path = %q", got.path)
	}
	if got.apiKey != "sk-test-secret-key" {
		t.Errorf("x-goog-api-key = %q", got.apiKey)
	}
	// The escape hatch merges at the ROOT on this wire, not into
	// generationConfig: the owned-field list is what keeps the two apart.
	if v := string(got.body["labels"]); v != `{"team":"ops"}` {
		t.Errorf("params key not merged at the body root: %q", v)
	}
	var gen map[string]json.RawMessage
	if err := json.Unmarshal(got.body["generationConfig"], &gen); err != nil {
		t.Fatalf("generationConfig: %v (body %v)", err, got.body)
	}
	want := map[string]string{
		"maxOutputTokens": "4096",
		"temperature":     "0.2",
		"topP":            "0.9",
		// medium is a level this wire HAS, so it travels verbatim.
		"thinkingConfig": `{"thinkingLevel":"medium"}`,
	}
	for key, value := range want {
		if v := string(gen[key]); v != value {
			t.Errorf("generationConfig[%q] = %q, want %q", key, v, value)
		}
	}
	// The openai-wire spellings must be nowhere in sight.
	for _, key := range []string{"messages", "max_completion_tokens", "reasoning_effort", "temperature"} {
		if _, ok := got.body[key]; ok {
			t.Errorf("gemini request carries the openai spelling %q: %v", key, got.body)
		}
	}
}

// TestE2EGeminiRunWarnsAboutSanitizedToolSchemas: this wire answers an
// unsupported JSON Schema keyword with a hard 400, so amele strips them - and
// the strip is announced. CONTRACT (design doc §"Gemini-specific mechanics" 1):
// nothing is silently dropped, so the run says which key left which tool.
func TestE2EGeminiRunWarnsAboutSanitizedToolSchemas(t *testing.T) {
	srv, calls := geminiCapturingServer(t, geminiTextBody("done"))
	cfgPath := writeGeminiConfig(t, srv.URL, "", "\ntools:\n  fs: true\n")

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "task"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if !strings.Contains(stderr, `"fs_read": "additionalProperties"`) {
		t.Errorf("stderr does not name the stripped keyword: %s", stderr)
	}
	// The request itself must carry the sanitized schema: an unstripped one is
	// a 400 for the whole request, not just for that tool.
	if body := string((*calls)[0].body["tools"]); strings.Contains(body, "additionalProperties") {
		t.Errorf("tool declaration still carries additionalProperties: %s", body)
	}
}

// TestE2EGeminiQuietRunDropsTheSanitizerWarning: -q is "errors only" for cron
// (docs/contracts/cli.md), and a sanitized schema is a warning.
func TestE2EGeminiQuietRunDropsTheSanitizerWarning(t *testing.T) {
	srv, _ := geminiCapturingServer(t, geminiTextBody("done"))
	cfgPath := writeGeminiConfig(t, srv.URL, "", "\ntools:\n  fs: true\n")

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "-q", "task"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if strings.Contains(stderr, "sanitized") {
		t.Errorf("-q still printed the sanitizer warning: %s", stderr)
	}
}

// TestE2ENonGeminiRunHasNoSanitizerWarning is the compatibility half: the
// sanitizer is a gemini-wire concern, so no other config may grow a line
// because of it.
func TestE2ENonGeminiRunHasNoSanitizerWarning(t *testing.T) {
	srv, _ := rawCapturingServer(t, textBody("done"))
	cfgPath := writeTunedConfig(t, srv.URL, "\ntools:\n  fs: true\n")

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "task"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if strings.Contains(stderr, "sanitized") {
		t.Errorf("an openai-wire run printed a sanitizer warning: %s", stderr)
	}
}
