package main

import (
	"bytes"
	"context"
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
	"github.com/lasthumanintheloop/amele/internal/session"
	"github.com/lasthumanintheloop/amele/internal/tools"
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

	provider, err := buildProvider(&config.Config{Model: "m", Provider: pc}, nil)
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

	provider, err := buildProvider(&config.Config{Provider: base}, nil)
	if err != nil {
		t.Fatalf("buildProvider: %v", err)
	}
	if c := provider.(*llm.GeminiClient); c.MaxAttempts != 0 || c.InitialBackoff != 0 {
		t.Errorf("gemini retry knobs: got %d/%v, want the zero values", c.MaxAttempts, c.InitialBackoff)
	}

	tuned := base
	tuned.Retry = &config.RetryConfig{MaxAttempts: 5, InitialBackoff: config.Duration(250 * time.Millisecond)}
	provider, err = buildProvider(&config.Config{Provider: tuned}, nil)
	if err != nil {
		t.Fatalf("buildProvider: %v", err)
	}
	if c := provider.(*llm.GeminiClient); c.MaxAttempts != 5 || c.InitialBackoff != 250*time.Millisecond {
		t.Errorf("gemini retry knobs not wired: %d/%v", c.MaxAttempts, c.InitialBackoff)
	}
}

// TestGeminiWithoutAPIKeyIsExit2: this wire has two credentials and a config
// must name one of them, so a config naming neither fails at validate - listing
// both paths - instead of buying a 401 from an unattended run.
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
		if !strings.Contains(stderr, "gemini needs api_key (AI Studio) or a vertex block (Vertex AI)") {
			t.Errorf("%s: stderr does not name the missing key: %s", cmd, stderr)
		}
	}
}

// TestVertexConfigReachesTheVertexEndpoint is the wiring proof: a vertex block
// in the YAML becomes the client's target, so the request is addressed by
// project and location instead of to the AI Studio host.
//
// It stops at the client rather than running the agent because the CREDENTIAL
// is the next slice: a run would fail at the token source, which is the honest
// intermediate state (the request never leaves for the wrong endpoint) but says
// nothing about the endpoint itself.
func TestVertexConfigReachesTheVertexEndpoint(t *testing.T) {
	cfg := &config.Config{Provider: config.ProviderConfig{
		Type: config.ProviderTypeGemini,
		//nolint:gosec // G101: a service-account key PATH, not a credential.
		Vertex: &config.VertexConfig{Project: "my-project", Location: "europe-west4", Credentials: "/etc/amele/sa.json"},
	}}
	provider, err := buildProvider(cfg, nil)
	if err != nil {
		t.Fatalf("buildProvider: %v", err)
	}
	client, ok := provider.(*llm.GeminiClient)
	if !ok {
		t.Fatalf("buildProvider returned %T, want *llm.GeminiClient", provider)
	}
	if client.Vertex == nil {
		t.Fatal("the vertex block did not reach the client")
	}
	if got := *client.Vertex; got != (llm.VertexTarget{Project: "my-project", Location: "europe-west4"}) {
		t.Errorf("vertex target = %+v", got)
	}

	// A config without the block keeps the AI Studio backend.
	plain := &config.Config{Provider: config.ProviderConfig{Type: config.ProviderTypeGemini, APIKey: "k"}}
	provider, err = buildProvider(plain, nil)
	if err != nil {
		t.Fatalf("buildProvider: %v", err)
	}
	if c := provider.(*llm.GeminiClient); c.Vertex != nil {
		t.Errorf("a keyed gemini config became a vertex client: %+v", c.Vertex)
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

// sanitizeStubTool is a minimal tools.Tool whose parameters schema always
// carries one gemini-unsupported keyword ("additionalProperties" is not in
// geminiSchemaKeywords), so each registration contributes exactly one
// stripped tool:key pair to warnSanitizedToolSchemas's line - enough to build
// a warning of a known size without standing up a real MCP toolset.
type sanitizeStubTool struct{ name string }

func (s sanitizeStubTool) Def() llm.ToolDef {
	return llm.ToolDef{Name: s.name, Parameters: json.RawMessage(`{"additionalProperties":false}`)}
}

func (s sanitizeStubTool) Invoke(context.Context, string) (string, error) { return "", nil }

// sanitizeStubRegistry builds a registry of n stub tools, each stripping
// exactly one key, so the sanitizer warning lists exactly n tool:key pairs
// before any cap is applied.
func sanitizeStubRegistry(t *testing.T, n int) *tools.Registry {
	t.Helper()
	reg := tools.NewRegistry()
	for i := 0; i < n; i++ {
		if err := reg.Register(sanitizeStubTool{name: fmt.Sprintf("tool%02d", i)}); err != nil {
			t.Fatalf("registering stub tool %d: %v", i, err)
		}
	}
	return reg
}

// TestSanitizerWarningCapBoundary pins issue #20's cap across the three
// cases that matter: under the cap (unaffected), exactly AT the cap (the
// boundary a `>` vs `>=` off-by-one in warnSanitizedToolSchemas would flip
// silently, since 8 is both "the last kept pair" and "the count that would
// trigger a suffix"), and over the cap (truncated with a count pointing at
// `amele explain`).
func TestSanitizerWarningCapBoundary(t *testing.T) {
	cases := []struct {
		name       string
		n          int  // stripped tool:key pairs the stub registry produces
		wantSuffix bool // whether an "and N more" suffix must appear
		wantPairs  int  // pairs (tool00..tool(wantPairs-1)) that must be listed
	}{
		{name: "under cap: 5 pairs, no suffix", n: 5, wantSuffix: false, wantPairs: 5},
		{name: "exactly at cap: 8 pairs, no suffix", n: 8, wantSuffix: false, wantPairs: 8},
		{name: "over cap: 12 pairs collapse to 8 plus a count", n: 12, wantSuffix: true, wantPairs: 8},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reg := sanitizeStubRegistry(t, tc.n)
			cfg := &config.Config{Provider: config.ProviderConfig{Type: config.ProviderTypeGemini}}

			var buf bytes.Buffer
			warnSanitizedToolSchemas(cfg, reg, &buf, false, session.NewSecretSet(nil))
			got := buf.String()

			if tc.wantSuffix {
				want := fmt.Sprintf("and %d more (run `amele explain` for the full list)", tc.n-maxSanitizedWarnEntries)
				if !strings.Contains(got, want) {
					t.Errorf("warning not capped: want substring %q in %q", want, got)
				}
			} else if strings.Contains(got, "more (run") {
				t.Errorf("warning carries a cap suffix it should not (n=%d): %q", tc.n, got)
			}
			for i := 0; i < tc.wantPairs; i++ {
				want := fmt.Sprintf(`"tool%02d"`, i)
				if !strings.Contains(got, want) {
					t.Errorf("warning dropped %q it should have kept: %q", want, got)
				}
			}
			if tc.n > maxSanitizedWarnEntries {
				if absent := fmt.Sprintf(`"tool%02d"`, maxSanitizedWarnEntries); strings.Contains(got, absent) {
					t.Errorf("warning lists more than the cap: %q", got)
				}
			}
			if n := strings.Count(got, "\n"); n > 1 {
				t.Errorf("warning must stay one line, got %d newlines: %q", n, got)
			}
		})
	}
}

// TestVertexTokenSourceIsWiredAndItsTokenRedacted is the credential half of the
// vertex wiring, proved end to end against the run's own secret registry.
//
// SECURITY: the token a run mints is not in the config, so it cannot be part of
// the redactor's starting list. The whole point of handing the token source the
// LIVE SecretSet is that the moment a token exists, every sink already scrubs
// it. This drives a real ADC resolution (the gcloud user-credential shape, so
// no key generation is needed) against a local token endpoint and then asks the
// registry what it now knows.
func TestVertexTokenSourceIsWiredAndItsTokenRedacted(t *testing.T) {
	const token = "ya29.wiring-test-token" //nolint:gosec // G101: a fixture value minted by the local server below.
	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"access_token":"` + token + `","token_type":"Bearer","expires_in":3600}`))
	}))
	defer srv.Close()

	// The gcloud application-default shape: the ADC leg that also asks for the
	// quota-project header.
	adc := filepath.Join(t.TempDir(), "application_default_credentials.json")
	body := `{"type":"authorized_user","client_id":"fake-id","client_secret":"fake-secret",` +
		`"refresh_token":"fake-refresh","token_uri":"` + srv.URL + `/token"}`
	if err := os.WriteFile(adc, []byte(body), 0o600); err != nil {
		t.Fatalf("writing the ADC fixture: %v", err)
	}
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", adc)

	cfg := &config.Config{Provider: config.ProviderConfig{
		Type:   config.ProviderTypeGemini,
		Vertex: &config.VertexConfig{Project: "my-project", Location: "europe-west4"},
	}}
	secrets := session.NewSecretSet(agentSecrets(cfg))

	provider, err := buildProvider(cfg, secrets.Add)
	if err != nil {
		t.Fatalf("buildProvider: %v", err)
	}
	client, ok := provider.(*llm.GeminiClient)
	if !ok {
		t.Fatalf("buildProvider returned %T, want *llm.GeminiClient", provider)
	}
	if client.TokenSource == nil {
		t.Fatal("a vertex config produced no token source")
	}

	got, err := client.TokenSource.Token(t.Context())
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if got != token || hits != 1 {
		t.Fatalf("token = %q after %d exchanges", got, hits)
	}
	// The registry is LIVE: a value it had never seen at startup is scrubbed
	// the moment the token source registers it.
	line := "provider error: status 401 with Authorization: Bearer " + token
	if redacted := secrets.Redact(line); strings.Contains(redacted, token) {
		t.Errorf("the vertex token survived redaction: %q", redacted)
	} else if !strings.Contains(redacted, "[REDACTED]") {
		t.Errorf("redacted line does not carry the marker: %q", redacted)
	}

	// The gcloud leg is the one that needs the quota header, and the project
	// it names is the one from the vertex block.
	quota, ok := client.TokenSource.(llm.GeminiQuotaProject)
	if !ok {
		t.Fatal("the wired token source cannot report a quota project")
	}
	if p := quota.QuotaProject(); p != "my-project" {
		t.Errorf("QuotaProject = %q, want %q", p, "my-project")
	}
}

// TestAIStudioConfigGetsNoTokenSource: the credential seam stays nil on the
// keyed backend. A non-nil source there would be a bearer token offered to an
// endpoint that authenticates with x-goog-api-key.
func TestAIStudioConfigGetsNoTokenSource(t *testing.T) {
	cfg := &config.Config{Provider: config.ProviderConfig{Type: config.ProviderTypeGemini, APIKey: "k"}}
	provider, err := buildProvider(cfg, nil)
	if err != nil {
		t.Fatalf("buildProvider: %v", err)
	}
	if ts := provider.(*llm.GeminiClient).TokenSource; ts != nil {
		t.Errorf("an AI Studio config carries a token source: %#v", ts)
	}
}

// TestVertexCredentialsFileReachesTheTokenSource: provider.vertex.credentials
// is the knob that bypasses the ADC search, so it has to arrive at the source
// that would otherwise perform it.
func TestVertexCredentialsFileReachesTheTokenSource(t *testing.T) {
	// A service-account key PATH, not a credential.
	const path = "/etc/amele/sa.json"
	cfg := &config.Config{Provider: config.ProviderConfig{
		Type:   config.ProviderTypeGemini,
		Vertex: &config.VertexConfig{Project: "p", Location: "us-central1", Credentials: path},
	}}
	provider, err := buildProvider(cfg, nil)
	if err != nil {
		t.Fatalf("buildProvider: %v", err)
	}
	source, ok := provider.(*llm.GeminiClient).TokenSource.(*llm.GoogleTokenSource)
	if !ok {
		t.Fatalf("token source is %T, want *llm.GoogleTokenSource", provider.(*llm.GeminiClient).TokenSource)
	}
	if source.CredentialsFile != path {
		t.Errorf("CredentialsFile = %q, want %q", source.CredentialsFile, path)
	}
	if source.Project != "p" {
		t.Errorf("Project = %q, want %q", source.Project, "p")
	}
}
