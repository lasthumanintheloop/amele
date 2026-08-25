package config

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/lasthumanintheloop/amele/internal/llm"
)

// envMap builds a LookupEnv from a plain map, keeping tests hermetic (no
// process environment involved).
func envMap(m map[string]string) LookupEnv {
	return func(key string) (string, bool) {
		v, ok := m[key]
		return v, ok
	}
}

// writeConfig writes YAML content into a temp dir and returns the file path.
func writeConfig(t *testing.T, dir, content string) string {
	t.Helper()
	path := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// minimalYAML is a valid baseline config used across tests.
const minimalYAML = `
model: test-model
provider:
  base_url: https://api.example.com/v1
  api_key: ${API_KEY}
`

func TestLoadValid(t *testing.T) {
	dir := t.TempDir()
	path := writeConfig(t, dir, minimalYAML+`
system_prompt: "You are helpful."
tools:
  fs: true
  subprocess:
    - name: send_mail
      description: Send an email via msmtp
      command: ["msmtp", "-t"]
      timeout: 30s
limits:
  max_turns: 5
  max_tokens: 1000
  timeout: 2m
session_dir: sessions
`)

	cfg, err := Load(path, envMap(map[string]string{"API_KEY": "sk-test"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}

	if cfg.Provider.APIKey != "sk-test" {
		t.Errorf("api_key interpolation: got %q", cfg.Provider.APIKey)
	}
	if cfg.Limits.Timeout.Std() != 2*time.Minute {
		t.Errorf("timeout: got %v", cfg.Limits.Timeout.Std())
	}
	if cfg.Tools.Subprocess[0].Timeout.Std() != 30*time.Second {
		t.Errorf("tool timeout: got %v", cfg.Tools.Subprocess[0].Timeout.Std())
	}
	// Relative paths must resolve against the config file directory so a
	// cron job can run the agent from any CWD.
	if cfg.SessionDir != filepath.Join(dir, "sessions") {
		t.Errorf("session_dir: got %q", cfg.SessionDir)
	}
	if cfg.Workspace != dir {
		t.Errorf("workspace default: got %q want config dir %q", cfg.Workspace, dir)
	}
}

func TestLoadDefaults(t *testing.T) {
	dir := t.TempDir()
	path := writeConfig(t, dir, minimalYAML)

	cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Limits.MaxTurns != 20 {
		t.Errorf("max_turns default: got %d want 20", cfg.Limits.MaxTurns)
	}
	if cfg.Limits.MaxTokens != 0 {
		t.Errorf("max_tokens default: got %d want 0 (disabled)", cfg.Limits.MaxTokens)
	}
	if cfg.SessionDir != "" {
		t.Errorf("session_dir default: got %q want disabled", cfg.SessionDir)
	}
}

// TestLoadRelativeConfigPathYieldsAbsolutePaths guards the pack use case:
// `amele run ./pack/agent.yaml` must resolve workspace/session_dir to
// absolute paths, so they stay correct no matter what cwd a tool later
// runs in (subprocess cmd.Dir differs from the caller's cwd).
func TestLoadRelativeConfigPathYieldsAbsolutePaths(t *testing.T) {
	dir := t.TempDir()
	sub := filepath.Join(dir, "pack")
	if err := os.Mkdir(sub, 0o750); err != nil {
		t.Fatal(err)
	}
	yaml := "model: m\nprovider:\n  base_url: https://x/v1\nsession_dir: sessions\n"
	if err := os.WriteFile(filepath.Join(sub, "agent.yaml"), []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(dir) // makes "pack/agent.yaml" a RELATIVE config path

	cfg, err := Load(filepath.Join("pack", "agent.yaml"), func(string) (string, bool) { return "", false })
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !filepath.IsAbs(cfg.Workspace) {
		t.Errorf("Workspace = %q, want absolute", cfg.Workspace)
	}
	if !filepath.IsAbs(cfg.SessionDir) {
		t.Errorf("SessionDir = %q, want absolute", cfg.SessionDir)
	}
}

// TestLoadLock pins the run-lock switch. It is opt-in and defaults to false on
// purpose: locking every config would break the supported pattern of running
// one YAML concurrently with different tasks (the LLM-judge recipe), so the
// default must stay "no lock" and only a cron user writes lock: true.
func TestLoadLock(t *testing.T) {
	tests := []struct {
		name string
		yaml string
		want bool
	}{
		{name: "absent defaults to false", yaml: minimalYAML, want: false},
		{name: "explicit false", yaml: minimalYAML + "lock: false\n", want: false},
		{name: "explicit true", yaml: minimalYAML + "lock: true\n", want: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			path := writeConfig(t, t.TempDir(), tc.yaml)
			cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k"}))
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			if cfg.Lock != tc.want {
				t.Errorf("lock = %v, want %v", cfg.Lock, tc.want)
			}
			// A bool needs no validation of its own; what matters is that it
			// never turns an otherwise-valid config into a config error.
			if err := cfg.Validate(); err != nil {
				t.Errorf("Validate: %v", err)
			}
		})
	}
}

func TestLoadErrors(t *testing.T) {
	tests := []struct {
		name    string
		yaml    string
		env     map[string]string
		wantSub string
	}{
		{
			name:    "undefined env var",
			yaml:    minimalYAML,
			env:     nil,
			wantSub: "API_KEY",
		},
		{
			name:    "unknown key rejected",
			yaml:    "model: m\nprovider:\n  base_url: https://x/v1\nmax_token: 5\n",
			env:     map[string]string{},
			wantSub: "max_token",
		},
		{
			name:    "invalid duration",
			yaml:    "model: m\nprovider:\n  base_url: https://x/v1\nlimits:\n  timeout: banana\n",
			env:     map[string]string{},
			wantSub: "duration",
		},
		{
			// A structurally wrong duration (a typo'd list) must be an
			// error, not a silent zero - zero disables the timeout kill
			// switch, which a cron user would never notice.
			name:    "non-scalar duration",
			yaml:    "model: m\nprovider:\n  base_url: https://x/v1\nlimits:\n  timeout: [5m]\n",
			env:     map[string]string{},
			wantSub: "duration",
		},
		{
			name:    "missing prompt file",
			yaml:    "model: m\nprovider:\n  base_url: https://x/v1\nsystem_prompt_file: nope.md\n",
			env:     map[string]string{},
			wantSub: "system_prompt_file",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := writeConfig(t, t.TempDir(), tt.yaml)
			_, err := Load(path, envMap(tt.env))
			if err == nil {
				t.Fatal("expected error")
			}
			if !errors.Is(err, ErrInvalid) {
				t.Errorf("error should wrap ErrInvalid: %v", err)
			}
			if !strings.Contains(err.Error(), tt.wantSub) {
				t.Errorf("error %q does not mention %q", err, tt.wantSub)
			}
		})
	}
}

// TestPromptExclusivityIsAViolation is the regression test for the
// short-circuit: setting both prompt forms used to abort the load, so the
// remaining violations in the same file stayed invisible until the author
// fixed this one and re-ran - the exact "one error per run" loop Violations
// exists to prevent. The rule now travels with every other violation, and the
// load no longer touches the prompt file: the config cannot run either way,
// and reading it would reinstate the short-circuit whenever the named file is
// missing (as it is here).
func TestPromptExclusivityIsAViolation(t *testing.T) {
	yaml := "model: m\nprovider:\n  base_url: https://x/v1\n" +
		"system_prompt: inline\nsystem_prompt_file: absent.md\n" +
		"limits:\n  max_turns: -1\n"
	path := writeConfig(t, t.TempDir(), yaml)

	cfg, err := Load(path, envMap(nil))
	if err != nil {
		t.Fatalf("Load must succeed so validation can report every violation: %v", err)
	}
	if cfg.SystemPrompt != "inline" {
		t.Errorf("system_prompt = %q, want the inline text left untouched", cfg.SystemPrompt)
	}

	got := cfg.Violations()
	for _, want := range []string{
		"system_prompt and system_prompt_file are mutually exclusive",
		"limits.max_turns must not be negative",
	} {
		if !slices.Contains(got, want) {
			t.Errorf("violations %q missing %q", got, want)
		}
	}

	err = cfg.Validate()
	if !errors.Is(err, ErrInvalid) {
		t.Fatalf("Validate must wrap ErrInvalid, got %v", err)
	}
	if !strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("Validate error %q does not mention the exclusivity rule", err)
	}
}

func TestInterpolateEscape(t *testing.T) {
	// "$$" must yield a literal "$" so prompts can talk about ${} syntax
	// without triggering interpolation.
	got, refs := interpolateString("price is $$5 and $${HOME} stays", envMap(nil), false)
	if len(refs) != 0 {
		t.Fatalf("escaped references must not count as references: %v", refs)
	}
	want := "price is $5 and ${HOME} stays"
	if got != want {
		t.Errorf("got %q want %q", got, want)
	}
}

// TestProviderRequestTimeout: the per-request HTTP ceiling is configurable -
// slow reasoning models legitimately exceed the 120s default.
func TestProviderRequestTimeout(t *testing.T) {
	yaml := `
model: m
provider:
  base_url: https://api.example.com/v1
  api_key: ${API_KEY}
  request_timeout: 5m
`
	path := writeConfig(t, t.TempDir(), yaml)
	cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Provider.RequestTimeout.Std() != 5*time.Minute {
		t.Errorf("request_timeout: got %v want 5m", cfg.Provider.RequestTimeout.Std())
	}

	cfg.Provider.RequestTimeout = Duration(-time.Second)
	cfg.Workspace = t.TempDir()
	if err := cfg.Validate(); err == nil || !strings.Contains(err.Error(), "request_timeout") {
		t.Errorf("negative request_timeout must fail validation: %v", err)
	}
}

// TestInterpolationIgnoresComments: a ${VAR} inside a YAML comment is not a
// reference - commenting a line out must fully disable it, not turn it into
// an "undefined environment variable" load failure.
func TestInterpolationIgnoresComments(t *testing.T) {
	yaml := minimalYAML + "# example: api_key: ${NOT_DEFINED_EXAMPLE}\n"
	path := writeConfig(t, t.TempDir(), yaml)
	if _, err := Load(path, envMap(map[string]string{"API_KEY": "k"})); err != nil {
		t.Fatalf("commented-out ${VAR} must not require the variable: %v", err)
	}
}

// TestInterpolationYAMLMetacharacters: substituted environment values must be
// treated as data, never as YAML syntax - real-world passwords contain
// newlines, colons and quotes.
func TestInterpolationYAMLMetacharacters(t *testing.T) {
	tests := []struct {
		name  string
		value string
	}{
		{"newline", "line1\nline2"},
		{"colon space", "hunter: 2"},
		{"quotes", `he said "hi" and 'bye'`},
		{"leading dash", "- not a list"},
		{"hash", "pass #word"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			yaml := minimalYAML + "system_prompt: use ${DB_PASS} here\n"
			path := writeConfig(t, t.TempDir(), yaml)
			cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k", "DB_PASS": tt.value}))
			if err != nil {
				t.Fatalf("value %q must not break YAML parsing: %v", tt.value, err)
			}
			want := "use " + tt.value + " here"
			if cfg.SystemPrompt != want {
				t.Errorf("system_prompt: got %q want %q", cfg.SystemPrompt, want)
			}
		})
	}
}

// TestLiteralAPIKeyRejected: committing a real key to YAML must fail at load
// time (docs/engineering.md §5.5) - interpolated references and empty keys stay valid.
func TestLiteralAPIKeyRejected(t *testing.T) {
	yaml := "model: m\nprovider:\n  base_url: https://x/v1\n  api_key: sk-live-verysecret\n"
	path := writeConfig(t, t.TempDir(), yaml)
	_, err := Load(path, envMap(map[string]string{}))
	if err == nil || !errors.Is(err, ErrInvalid) {
		t.Fatalf("literal api_key must be rejected: %v", err)
	}
	if !strings.Contains(err.Error(), "literal secrets") {
		t.Errorf("error should explain the rule: %v", err)
	}

	// A key that is mostly literal with a token ${VAR} appended is still a
	// committed secret; the whole value must be built from references.
	mixed := writeConfig(t, t.TempDir(), "model: m\nprovider:\n  base_url: https://x/v1\n  api_key: sk-live-verysecret-${SUFFIX}\n")
	_, err = Load(mixed, envMap(map[string]string{"SUFFIX": "x"}))
	if err == nil || !errors.Is(err, ErrInvalid) {
		t.Fatalf("mixed literal api_key must be rejected: %v", err)
	}

	// Empty api_key stays valid (local endpoints like Ollama).
	noKey := writeConfig(t, t.TempDir(), "model: m\nprovider:\n  base_url: https://x/v1\n")
	if _, err := Load(noKey, envMap(nil)); err != nil {
		t.Errorf("missing api_key must load: %v", err)
	}
}

// TestInterpolationSkipsMappingKeys pins the fix for the literal-key smuggling
// bypass: interpolating a mapping KEY would let `"${KEY}": sk-literal` become
// `api_key: sk-literal` AFTER rejectLiteralAPIKey has probed the raw file,
// walking a plaintext credential straight past the ban (docs/engineering.md §5.5).
func TestInterpolationSkipsMappingKeys(t *testing.T) {
	t.Run("key interpolation cannot smuggle a literal api_key", func(t *testing.T) {
		yaml := "model: m\nprovider:\n  base_url: https://x/v1\n  \"${KEY}\": sk-live-verysecret\n"
		path := writeConfig(t, t.TempDir(), yaml)
		cfg, err := Load(path, envMap(map[string]string{"KEY": "api_key"}))
		if err == nil {
			t.Fatalf("literal key smuggled through: api_key = %q", cfg.Provider.APIKey)
		}
		if !errors.Is(err, ErrInvalid) {
			t.Errorf("error should wrap ErrInvalid: %v", err)
		}
		// The key stays literal, so the strict decoder reports it as unknown.
		if !strings.Contains(err.Error(), "${KEY}") {
			t.Errorf("error should name the offending key: %v", err)
		}
	})

	t.Run("values still interpolate everywhere", func(t *testing.T) {
		dir := t.TempDir()
		yaml := minimalYAML + `system_prompt: "hello ${WHO}"
workspace: ${SUB}
tools:
  subprocess:
    - name: greet
      description: greet
      command: ["echo", "${WHO}"]
`
		if err := os.Mkdir(filepath.Join(dir, "sub"), 0o750); err != nil {
			t.Fatal(err)
		}
		path := writeConfig(t, dir, yaml)
		cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k", "WHO": "world", "SUB": "sub"}))
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		if cfg.Provider.APIKey != "k" {
			t.Errorf("api_key: got %q want %q", cfg.Provider.APIKey, "k")
		}
		if cfg.SystemPrompt != "hello world" {
			t.Errorf("system_prompt: got %q", cfg.SystemPrompt)
		}
		if cfg.Workspace != filepath.Join(dir, "sub") {
			t.Errorf("workspace: got %q", cfg.Workspace)
		}
		if got := cfg.Tools.Subprocess[0].Command[1]; got != "world" {
			t.Errorf("command arg: got %q", got)
		}
	})
}

// TestLoadRejectsMultipleDocuments: a config split over two `---` documents
// used to have everything after the first document silently dropped, which
// quietly discarded budgets and a `permissions.default: deny` block and left
// the dangerous allow-everything default in force. It is a config error now.
func TestLoadRejectsMultipleDocuments(t *testing.T) {
	// A trailing "---" is refused too: it is still a second document, and the
	// rule is easier to explain (and to trust) without an exception.
	for _, tc := range []struct{ name, yaml string }{
		{"second document with content", minimalYAML + "---\npermissions:\n  default: deny\n"},
		{"empty second document", minimalYAML + "---\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := writeConfig(t, t.TempDir(), tc.yaml)
			_, err := Load(path, envMap(map[string]string{"API_KEY": "k"}))
			if err == nil {
				t.Fatal("a second YAML document must be rejected, not ignored")
			}
			if !errors.Is(err, ErrInvalid) {
				t.Errorf("error should wrap ErrInvalid: %v", err)
			}
			if !strings.Contains(err.Error(), "multiple YAML documents") {
				t.Errorf("error should name the problem: %v", err)
			}
		})
	}

	// A leading "---" opens a single document and must keep loading.
	t.Run("leading document marker is fine", func(t *testing.T) {
		path := writeConfig(t, t.TempDir(), "---"+minimalYAML)
		if _, err := Load(path, envMap(map[string]string{"API_KEY": "k"})); err != nil {
			t.Errorf("single document with a leading ---: %v", err)
		}
	})
}

// TestInterpolatedSecrets: every substituted env value is exposed for
// session-log redaction, deduplicated, without empties.
func TestInterpolatedSecrets(t *testing.T) {
	yaml := minimalYAML + "system_prompt: \"use ${DB_PASSWORD} and ${DB_PASSWORD} and ${EMPTY}\"\n"
	path := writeConfig(t, t.TempDir(), yaml)
	cfg, err := Load(path, envMap(map[string]string{
		"API_KEY": "sk-key", "DB_PASSWORD": "hunter2", "EMPTY": "",
	}))
	if err != nil {
		t.Fatal(err)
	}
	secrets := cfg.InterpolatedSecrets()
	want := map[string]bool{"sk-key": true, "hunter2": true}
	if len(secrets) != len(want) {
		t.Fatalf("secrets: %v", secrets)
	}
	for _, s := range secrets {
		if !want[s] {
			t.Errorf("unexpected secret %q", s)
		}
	}
}

func TestLoadCollectsEnvReferences(t *testing.T) {
	dir := t.TempDir()
	yaml := "model: m\nprovider:\n  base_url: https://x/v1\n  api_key: ${KEY_A}\nsystem_prompt: uses ${KEY_B} and ${KEY_A} again\n"
	path := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	env := func(name string) (string, bool) { return "v-" + name, true }
	cfg, err := Load(path, env)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	want := []string{"KEY_A", "KEY_B"} // first-appearance order, deduplicated
	if got := cfg.EnvReferenced(); !slices.Equal(got, want) {
		t.Errorf("EnvReferenced() = %v, want %v", got, want)
	}
	if got := cfg.EnvMissing(); len(got) != 0 {
		t.Errorf("EnvMissing() = %v, want empty", got)
	}
}

// TestEnvBindings pins the per-variable view explain needs to decide what it
// may display: name, substituted value, missing flag, and whether the
// reference fed provider.api_key. The api_key flag cannot be recovered after
// the fact (the field is never printed, but the same value may appear again in
// an argv), so it is recorded during interpolation.
func TestEnvBindings(t *testing.T) {
	yaml := `model: ${MODEL}
provider:
  base_url: https://x/v1
  api_key: ${OPENROUTER}
system_prompt: "model ${MODEL}, key ${OPENROUTER}, gone ${ABSENT}"
`
	path := writeConfig(t, t.TempDir(), yaml)
	env := envMap(map[string]string{"MODEL": "gpt-4o-mini", "OPENROUTER": "sk-live-xyz"})

	cfg, err := LoadTolerant(path, env)
	if err != nil {
		t.Fatalf("LoadTolerant: %v", err)
	}
	want := []EnvBinding{
		{Name: "MODEL", Value: "gpt-4o-mini"},
		{Name: "OPENROUTER", Value: "sk-live-xyz", APIKey: true},
		{Name: "ABSENT", Missing: true},
	}
	if got := cfg.EnvBindings(); !slices.Equal(got, want) {
		t.Errorf("EnvBindings() = %+v, want %+v", got, want)
	}
	// The derived views must stay consistent with the bindings they come from.
	if got := cfg.EnvReferenced(); !slices.Equal(got, []string{"MODEL", "OPENROUTER", "ABSENT"}) {
		t.Errorf("EnvReferenced() = %v", got)
	}
	if got := cfg.EnvMissing(); !slices.Equal(got, []string{"ABSENT"}) {
		t.Errorf("EnvMissing() = %v", got)
	}
	if got := cfg.InterpolatedSecrets(); !slices.Equal(got, []string{"gpt-4o-mini", "sk-live-xyz"}) {
		t.Errorf("InterpolatedSecrets() = %v", got)
	}
}

// TestEnvBindingsAPIKeyOnlyForTheAPIKeyField: the api_key marker is a field
// path, not a name heuristic - a variable used elsewhere must not inherit it,
// or explain would redact values it is now expected to show.
func TestEnvBindingsAPIKeyOnlyForTheAPIKeyField(t *testing.T) {
	yaml := `model: m
provider:
  base_url: ${BASE_URL}
  api_key: ${API_KEY}
`
	path := writeConfig(t, t.TempDir(), yaml)
	cfg, err := Load(path, envMap(map[string]string{"BASE_URL": "https://x/v1", "API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	for _, b := range cfg.EnvBindings() {
		if want := b.Name == "API_KEY"; b.APIKey != want {
			t.Errorf("binding %q: APIKey = %v, want %v", b.Name, b.APIKey, want)
		}
	}
}

// TestViolations: Validate's message is one blob, but explain reports problems
// as individual lines, so the violations are available un-joined too. Both
// views must agree - Validate is defined in terms of Violations.
func TestViolations(t *testing.T) {
	dir := t.TempDir()
	path := writeConfig(t, dir, "provider:\n  base_url: https://x/v1\nworkspace: /nonexistent-workspace-xyz\n")
	cfg, err := Load(path, envMap(nil))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got := cfg.Violations()
	if len(got) < 2 {
		t.Fatalf("Violations() = %v, want the model and workspace violations", got)
	}
	joined := strings.Join(got, "\n")
	if !strings.Contains(joined, "model is required") || !strings.Contains(joined, "is not accessible") {
		t.Errorf("Violations() = %v", got)
	}
	err = cfg.Validate()
	if err == nil {
		t.Fatal("Validate() = nil, want an error")
	}
	for _, v := range got {
		if !strings.Contains(err.Error(), v) {
			t.Errorf("Validate() message %q does not carry violation %q", err, v)
		}
	}
	// A clean config reports no violations at all.
	ok := writeConfig(t, t.TempDir(), minimalYAML)
	cleanCfg, err := Load(ok, envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if v := cleanCfg.Violations(); len(v) != 0 {
		t.Errorf("Violations() on a clean config = %v, want none", v)
	}
}

func TestLoadTolerantRecordsMissingEnv(t *testing.T) {
	dir := t.TempDir()
	yaml := "model: m\nprovider:\n  base_url: https://x/v1\n  api_key: ${MISSING_KEY}\n"
	path := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	none := func(string) (string, bool) { return "", false }

	// Plain Load: unchanged contract - missing env is a load error.
	if _, err := Load(path, none); err == nil {
		t.Fatal("Load should fail on undefined environment variables")
	}

	cfg, err := LoadTolerant(path, none)
	if err != nil {
		t.Fatalf("LoadTolerant: %v", err)
	}
	if got := cfg.EnvMissing(); !slices.Equal(got, []string{"MISSING_KEY"}) {
		t.Errorf("EnvMissing() = %v, want [MISSING_KEY]", got)
	}
	if got := cfg.Provider.APIKey; got != "" {
		t.Errorf("missing var should substitute empty, got %q", got)
	}
}

func TestSystemPromptFile(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "prompt.md"), []byte("from file"), 0o600); err != nil {
		t.Fatal(err)
	}
	path := writeConfig(t, dir, minimalYAML+"system_prompt_file: prompt.md\n")

	cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.SystemPrompt != "from file" {
		t.Errorf("system prompt: got %q", cfg.SystemPrompt)
	}
}

// TestLoadTolerantFailsOnUnreadablePromptFileWithoutMissingEnv pins the exit-2
// regression: LoadTolerant's tolerance of a system_prompt_file read error
// exists only to let the requirements report render when the file's path
// itself came from a missing ${VAR}. A config with no ${VAR} references at
// all that simply names a nonexistent prompt file is a real config error and
// must fail LoadTolerant exactly as Load does - an unreadable file is the one
// thing `explain` still refuses to report on, and a silent skip here would
// leave it unmentioned by every command.
func TestLoadTolerantFailsOnUnreadablePromptFileWithoutMissingEnv(t *testing.T) {
	dir := t.TempDir()
	path := writeConfig(t, dir, minimalYAML+"system_prompt_file: missing.md\n")
	env := envMap(map[string]string{"API_KEY": "k"})

	if _, err := LoadTolerant(path, env); err == nil {
		t.Fatal("LoadTolerant should fail: no missing env vars, so the unreadable prompt file is a real config error")
	}
}

// TestLoadTolerantSkipsPromptFileErrorWhenEnvMissing pins the surviving half
// of the tolerant branch: when the load also has missing ${VAR}s, LoadTolerant
// must still succeed even though the prompt file can't be read - explain's
// report (the answer to "what do I set?") needs a *Config to render, and the
// missing variables are already reported as problems, so failing the load
// here would only lose the report.
func TestLoadTolerantSkipsPromptFileErrorWhenEnvMissing(t *testing.T) {
	dir := t.TempDir()
	yaml := "model: m\nprovider:\n  base_url: https://x/v1\n  api_key: ${MISSING_KEY}\nsystem_prompt_file: missing.md\n"
	path := writeConfig(t, dir, yaml)
	none := func(string) (string, bool) { return "", false }

	cfg, err := LoadTolerant(path, none)
	if err != nil {
		t.Fatalf("LoadTolerant: %v", err)
	}
	if got := cfg.EnvMissing(); !slices.Equal(got, []string{"MISSING_KEY"}) {
		t.Errorf("EnvMissing() = %v, want [MISSING_KEY]", got)
	}
}

// TestOutputSchema: an inline output.schema mapping parses into
// Config.Output.Schema and SchemaJSON() re-encodes it as JSON Schema text
// suitable for compilation (Task 5, cmd layer).
func TestOutputSchema(t *testing.T) {
	yaml := minimalYAML + `
output:
  schema:
    type: object
    required: ["verdict"]
    properties:
      verdict:
        type: string
        enum: ["pass", "fail", 1]
  max_schema_retries: 3
`
	path := writeConfig(t, t.TempDir(), yaml)
	cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Output.MaxSchemaRetries != 3 {
		t.Errorf("max_schema_retries: got %d want 3", cfg.Output.MaxSchemaRetries)
	}

	raw, err := cfg.SchemaJSON()
	if err != nil {
		t.Fatalf("SchemaJSON: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("SchemaJSON produced invalid JSON: %v\n%s", err, raw)
	}
	if got["type"] != "object" {
		t.Errorf("type: got %v", got["type"])
	}
	props, ok := got["properties"].(map[string]any)
	if !ok {
		t.Fatalf("properties: got %v", got["properties"])
	}
	verdict, ok := props["verdict"].(map[string]any)
	if !ok {
		t.Fatalf("properties.verdict: got %v", props["verdict"])
	}
	enum, ok := verdict["enum"].([]any)
	if !ok || len(enum) != 3 {
		t.Fatalf("properties.verdict.enum: got %v", verdict["enum"])
	}
}

// TestOutputConfigDefaults: without an output block, the schema stays nil and
// max_schema_retries stays 0 - Load does not apply the "default 2" retry
// count, since that default belongs to cmd (Task 5), not config.
func TestOutputConfigDefaults(t *testing.T) {
	path := writeConfig(t, t.TempDir(), minimalYAML)
	cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Output.Schema != nil {
		t.Errorf("output.schema default: got %v want nil", cfg.Output.Schema)
	}
	if cfg.Output.MaxSchemaRetries != 0 {
		t.Errorf("max_schema_retries default: got %d want 0", cfg.Output.MaxSchemaRetries)
	}

	raw, err := cfg.SchemaJSON()
	if err != nil || raw != nil {
		t.Errorf("SchemaJSON with no schema: got (%s, %v) want (nil, nil)", raw, err)
	}
}

// TestSchemaJSONEmptyMap: an explicitly empty schema mapping is equivalent to
// no schema at all - both mean "no constraint," so both must marshal to nil.
func TestSchemaJSONEmptyMap(t *testing.T) {
	cfg := &Config{Output: OutputConfig{Schema: map[string]any{}}}
	raw, err := cfg.SchemaJSON()
	if err != nil || raw != nil {
		t.Errorf("SchemaJSON with empty map: got (%s, %v) want (nil, nil)", raw, err)
	}
}

// TestPermissionsLoad verifies the permissions block parses into the typed
// policy map and that an absent block stays zero (Phase 1 parity: allow).
func TestPermissionsLoad(t *testing.T) {
	yaml := minimalYAML + `
permissions:
  default: deny
  tools:
    fs_read: allow
    fs_write: ask
    send_mail: deny
`
	path := writeConfig(t, t.TempDir(), yaml)
	cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if cfg.Permissions.Default != PolicyDeny {
		t.Errorf("default: got %q", cfg.Permissions.Default)
	}
	want := map[string]Policy{"fs_read": PolicyAllow, "fs_write": PolicyAsk, "send_mail": PolicyDeny}
	for name, policy := range want {
		if got := cfg.Permissions.Tools[name]; got != policy {
			t.Errorf("tools[%s]: got %q want %q", name, got, policy)
		}
	}

	bare := writeConfig(t, t.TempDir(), minimalYAML)
	cfg, err = Load(bare, envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Permissions.Default != "" || cfg.Permissions.Tools != nil {
		t.Errorf("absent permissions block should stay zero: %+v", cfg.Permissions)
	}
}

// TestPermissionsUnknownToolNameAllowed pins the deliberate looseness: a
// policy may name a tool this config does not declare.
func TestPermissionsUnknownToolNameAllowed(t *testing.T) {
	yaml := minimalYAML + `
permissions:
  tools:
    not_a_declared_tool: deny
`
	path := writeConfig(t, t.TempDir(), yaml)
	cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if err := cfg.Validate(); err != nil {
		t.Errorf("naming an undeclared tool must not be an error: %v", err)
	}
}

// TestPermissionsGlobKeyAccepted pins that a glob key validates like any
// other: `permissions.tools` keys are patterns (perm.policyFor), and MCP
// configs govern a whole server with one `<server>__*` line.
func TestPermissionsGlobKeyAccepted(t *testing.T) {
	yaml := minimalYAML + `
permissions:
  default: allow
  tools:
    "github__*": ask
    "*_delete*": deny
`
	path := writeConfig(t, t.TempDir(), yaml)
	cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("a glob key must validate: %v", err)
	}
	if got := cfg.Permissions.Tools["github__*"]; got != PolicyAsk {
		t.Errorf("tools[github__*] = %q, want ask", got)
	}
	if got := cfg.Permissions.Tools["*_delete*"]; got != PolicyDeny {
		t.Errorf("tools[*_delete*] = %q, want deny", got)
	}
}

// TestPermissionsGlobKeyInvalidPolicy: a glob key buys no leniency on the
// policy value itself.
func TestPermissionsGlobKeyInvalidPolicy(t *testing.T) {
	yaml := minimalYAML + `
permissions:
  tools:
    "github__*": maybe
`
	path := writeConfig(t, t.TempDir(), yaml)
	cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	err = cfg.Validate()
	if err == nil || !strings.Contains(err.Error(), "permissions.tools.github__*") {
		t.Errorf("want an error naming the glob key, got %v", err)
	}
}

func TestValidateErrors(t *testing.T) {
	dir := t.TempDir()

	tests := []struct {
		name    string
		mutate  func(*Config)
		wantSub string
	}{
		{"missing model", func(c *Config) { c.Model = "" }, "model is required"},
		{"missing base_url", func(c *Config) { c.Provider.BaseURL = "" }, "base_url is required"},
		{"relative base_url", func(c *Config) { c.Provider.BaseURL = "not-a-url" }, "not a valid absolute URL"},
		{
			// The message must list the valid values so a typo is fixable
			// without reading the docs.
			"invalid provider type",
			func(c *Config) { c.Provider.Type = "bedrock" },
			"openai, anthropic, gemini",
		},
		{
			"missing base_url with explicit openai type",
			func(c *Config) { c.Provider.Type = ProviderTypeOpenAI; c.Provider.BaseURL = "" },
			"base_url is required",
		},
		{
			"invalid base_url with anthropic type",
			func(c *Config) { c.Provider.Type = ProviderTypeAnthropic; c.Provider.BaseURL = "not-a-url" },
			"not a valid absolute URL",
		},
		{
			// The native client appends /v1/messages itself; a base_url that
			// already ends in /v1 (the OpenAI-compat convention) would 404 on
			// /v1/v1/messages at runtime, so it is caught at validation time.
			"anthropic base_url ending in /v1",
			func(c *Config) {
				c.Provider.Type = ProviderTypeAnthropic
				c.Provider.BaseURL = "https://gw.example.com/v1"
			},
			"/v1/messages",
		},
		{
			"anthropic base_url ending in /v1/ (trailing slash)",
			func(c *Config) {
				c.Provider.Type = ProviderTypeAnthropic
				c.Provider.BaseURL = "https://gw.example.com/v1/"
			},
			"/v1/messages",
		},
		{
			// Same class of mistake on the gemini wire, and the same silent
			// 404: the client appends /v1beta/models/{model}:generateContent
			// itself, so a base_url that already carries the version segment
			// would send it twice. The proxy answers 404 at run time, which
			// reads as a broken endpoint rather than as the config error it is.
			"gemini base_url ending in /v1beta",
			func(c *Config) {
				c.Provider.Type = ProviderTypeGemini
				c.Provider.BaseURL = "https://gw.example.com/v1beta"
			},
			"the client appends",
		},
		{
			"gemini base_url ending in /v1beta/ (trailing slash)",
			func(c *Config) {
				c.Provider.Type = ProviderTypeGemini
				c.Provider.BaseURL = "https://gw.example.com/v1beta/"
			},
			"the client appends",
		},
		{
			// The OpenAI-compat habit reaches this wire too: /v1 is what an
			// operator moving a config from a gateway leaves behind.
			"gemini base_url ending in /v1",
			func(c *Config) {
				c.Provider.Type = ProviderTypeGemini
				c.Provider.BaseURL = "https://gw.example.com/v1"
			},
			"the client appends",
		},
		{
			// CONTRACT: a config that passes validate must not fail
			// configuration at run time. A non-http(s) scheme reaches the HTTP
			// client and surfaces as a provider error (exit 5) instead of the
			// config error (exit 2) it really is.
			"non-http scheme",
			func(c *Config) { c.Provider.BaseURL = "ftp://gw.example.com" },
			"http or https",
		},
		{
			"misspelled https scheme",
			func(c *Config) { c.Provider.BaseURL = "htps://gw.example.com" },
			"http or https",
		},
		{
			// The client appends request paths to base_url, so a query or
			// fragment would end up in the middle of the request URL.
			"base_url with query",
			func(c *Config) { c.Provider.BaseURL = "http://gw.example.com/?q=1" },
			"query string or fragment",
		},
		{
			"base_url with fragment",
			func(c *Config) { c.Provider.BaseURL = "http://gw.example.com/#frag" },
			"query string or fragment",
		},
		{
			"negative max_output_tokens",
			func(c *Config) { c.Provider.MaxOutputTokens = -1 },
			"max_output_tokens",
		},
		{"negative turns", func(c *Config) { c.Limits.MaxTurns = -1 }, "max_turns"},
		{"negative tokens", func(c *Config) { c.Limits.MaxTokens = -1 }, "max_tokens"},
		{"bad workspace", func(c *Config) { c.Workspace = filepath.Join(dir, "missing") }, "not accessible"},
		{
			"bad tool name",
			func(c *Config) {
				c.Tools.Subprocess = []SubprocessTool{{Name: "bad name!", Description: "d", Command: []string{"true"}}}
			},
			"must match",
		},
		{
			"duplicate tool",
			func(c *Config) {
				tool := SubprocessTool{Name: "dup", Description: "d", Command: []string{"true"}}
				c.Tools.Subprocess = []SubprocessTool{tool, tool}
			},
			"duplicate",
		},
		{
			"reserved tool name",
			func(c *Config) {
				c.Tools.Subprocess = []SubprocessTool{{Name: "fs_read", Description: "d", Command: []string{"true"}}}
			},
			"reserved",
		},
		{
			// "shell" is the builtin shell tool's name: a subprocess tool must
			// not be able to shadow it (and thereby quietly replace a tool the
			// permission profile names).
			"reserved shell name",
			func(c *Config) {
				c.Tools.Subprocess = []SubprocessTool{{Name: "shell", Description: "d", Command: []string{"true"}}}
			},
			"reserved",
		},
		{
			"negative shell timeout",
			func(c *Config) { c.Tools.Shell = ShellConfig{Enabled: true, Timeout: -1} },
			"tools.shell.timeout",
		},
		{
			// An empty allowlist entry can only be a YAML slip ("- " with
			// nothing after it) and would silently allow nothing.
			"empty shell env entry",
			func(c *Config) { c.Tools.Shell = ShellConfig{Enabled: true, Env: []string{""}} },
			"tools.shell.env",
		},
		{
			"duplicate shell env name",
			func(c *Config) { c.Tools.Shell = ShellConfig{Enabled: true, Env: []string{"HOME", "HOME"}} },
			"duplicate env name",
		},
		{
			// "PATH=/usr/bin" in the allowlist means someone thought they were
			// assigning a value. The list carries names only; catch the slip.
			"shell env entry with equals sign",
			func(c *Config) { c.Tools.Shell = ShellConfig{Enabled: true, Env: []string{"PATH=/usr/bin"}} },
			"must be a variable name",
		},
		{
			"subprocess env entry with equals sign",
			func(c *Config) {
				c.Tools.Subprocess = []SubprocessTool{{Name: "t", Description: "d", Command: []string{"true"}, Env: []string{"KEY=val"}}}
			},
			"must be a variable name",
		},
		{
			"empty subprocess env entry",
			func(c *Config) {
				c.Tools.Subprocess = []SubprocessTool{{Name: "t", Description: "d", Command: []string{"true"}, Env: []string{""}}}
			},
			"tools.subprocess[0].env",
		},
		{
			"duplicate subprocess env name",
			func(c *Config) {
				c.Tools.Subprocess = []SubprocessTool{{Name: "t", Description: "d", Command: []string{"true"}, Env: []string{"A", "A"}}}
			},
			"duplicate env name",
		},
		{
			"empty command",
			func(c *Config) {
				c.Tools.Subprocess = []SubprocessTool{{Name: "t", Description: "d"}}
			},
			"command",
		},
		{
			"empty command[0]",
			func(c *Config) {
				c.Tools.Subprocess = []SubprocessTool{{
					Name:        "t",
					Description: "d",
					Command:     []string{""},
				}}
			},
			"command[0] must not be empty",
		},
		{
			"negative max_schema_retries",
			func(c *Config) { c.Output.MaxSchemaRetries = -1 },
			"max_schema_retries",
		},
		{
			"invalid default policy",
			func(c *Config) { c.Permissions.Default = "auto_approve" },
			"permissions.default",
		},
		{
			"invalid tool policy",
			func(c *Config) { c.Permissions.Tools = map[string]Policy{"fs_write": "maybe"} },
			"permissions.tools.fs_write",
		},
		{
			// An empty value ("fs_write:" with nothing after it) is a typo,
			// not "unset": per-tool entries exist to override the default.
			"empty tool policy",
			func(c *Config) { c.Permissions.Tools = map[string]Policy{"fs_write": ""} },
			"permissions.tools.fs_write",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{
				Model:     "m",
				Provider:  ProviderConfig{BaseURL: "https://api.example.com/v1"},
				Workspace: dir,
				Limits:    Limits{MaxTurns: 20},
			}
			tt.mutate(cfg)
			err := cfg.Validate()
			if err == nil {
				t.Fatal("expected error")
			}
			if !errors.Is(err, ErrInvalid) {
				t.Errorf("error should wrap ErrInvalid: %v", err)
			}
			if !strings.Contains(err.Error(), tt.wantSub) {
				t.Errorf("error %q does not mention %q", err, tt.wantSub)
			}
		})
	}
}

// TestProviderBaseURLAccepted keeps the scheme/query tightening from rejecting
// the shapes real gateways use: plain hosts, ports, paths and a trailing
// slash all stay valid.
func TestProviderBaseURLAccepted(t *testing.T) {
	dir := t.TempDir()
	for _, base := range []string{
		"http://host",
		"https://host/v1",
		"http://localhost:11434/v1",
		"https://gw.example.com/openai/v1/",
		"HTTPS://gw.example.com/v1",
	} {
		t.Run(base, func(t *testing.T) {
			cfg := &Config{
				Model:     "m",
				Provider:  ProviderConfig{BaseURL: base},
				Workspace: dir,
				Limits:    Limits{MaxTurns: 20},
			}
			if err := cfg.Validate(); err != nil {
				t.Errorf("base_url %q must validate: %v", base, err)
			}
		})
	}
}

// TestProviderTypeAnthropic pins the provider.type / provider.max_output_tokens
// schema: with type anthropic both fields round-trip and base_url may be
// omitted (the native client falls back to the fixed official endpoint).
func TestProviderTypeAnthropic(t *testing.T) {
	dir := t.TempDir()
	path := writeConfig(t, dir, `
model: claude-x
provider:
  type: anthropic
  api_key: ${API_KEY}
  max_output_tokens: 4096
`)
	cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Provider.Type != ProviderTypeAnthropic {
		t.Errorf("type: got %q want %q", cfg.Provider.Type, ProviderTypeAnthropic)
	}
	if cfg.Provider.MaxOutputTokens != 4096 {
		t.Errorf("max_output_tokens: got %d want 4096", cfg.Provider.MaxOutputTokens)
	}
	if err := cfg.Validate(); err != nil {
		t.Errorf("anthropic without base_url must validate: %v", err)
	}

	// An explicit gateway URL (that does not end in /v1) stays valid: proxies
	// exposing the Anthropic wire format are a supported deployment.
	cfg.Provider.BaseURL = "https://proxy.example.com"
	if err := cfg.Validate(); err != nil {
		t.Errorf("anthropic with explicit base_url must validate: %v", err)
	}
}

// TestProviderTypeGemini pins the third wire value: `type: gemini` round-trips,
// base_url is optional (the client knows the fixed AI Studio host, exactly as
// on the anthropic path), and an explicit host override stays valid - proxies
// in front of the Gemini API are a supported deployment.
func TestProviderTypeGemini(t *testing.T) {
	dir := t.TempDir()
	path := writeConfig(t, dir, `
model: gemini-3-pro
provider:
  type: gemini
  api_key: ${API_KEY}
  max_output_tokens: 4096
`)
	cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Provider.Type != ProviderTypeGemini {
		t.Errorf("type: got %q want %q", cfg.Provider.Type, ProviderTypeGemini)
	}
	if cfg.Provider.MaxOutputTokens != 4096 {
		t.Errorf("max_output_tokens: got %d want 4096", cfg.Provider.MaxOutputTokens)
	}
	if err := cfg.Validate(); err != nil {
		t.Errorf("gemini without base_url must validate: %v", err)
	}

	cfg.Provider.BaseURL = "https://proxy.example.com"
	if err := cfg.Validate(); err != nil {
		t.Errorf("gemini with explicit base_url must validate: %v", err)
	}
}

// TestGeminiWireRejectsDialect pins the message, not just the refusal: the
// dialect names a variation of the OPENAI wire, so on the gemini wire its value
// is never parsed - whatever it spells, the fix is to delete the line. A config
// that also misspells it must therefore hear about the wire, not about the
// dialect vocabulary, which would send the operator to correct a key that has
// to go.
func TestGeminiWireRejectsDialect(t *testing.T) {
	for _, dialect := range []string{"kimi", "not-a-dialect"} {
		t.Run(dialect, func(t *testing.T) {
			cfg := tuningBase(t.TempDir())
			cfg.Provider.Type = ProviderTypeGemini
			cfg.Provider.BaseURL = ""
			cfg.Provider.Dialect = dialect

			err := cfg.Validate()
			if err == nil {
				t.Fatal("a dialect on the gemini wire must be refused")
			}
			if !strings.Contains(err.Error(), "remove it for type gemini") {
				t.Errorf("error %q does not tell the operator to remove the dialect", err)
			}
			if strings.Contains(err.Error(), "unknown dialect") {
				t.Errorf("error %q blames the dialect vocabulary; on this wire the key itself is wrong", err)
			}
		})
	}
}

// TestGeminiRequiresAPIKey pins slice 1's one auth path: the AI Studio key.
// The Gemini API has a second one - Vertex, with Google credentials - and it is
// not built yet, so a `type: gemini` config with no key would otherwise reach
// the wire as an unauthenticated request and come back a 401 from a run nobody
// was watching. The message names the successor explicitly: an operator who
// wanted Vertex must learn that here, not from a 401.
//
// The rule is deliberately NOT in the JSON Schema: it is a cross-field relation
// (and Task 7 relaxes it to "api_key or vertex"), which the schema copies would
// have to re-encode identically in three places.
func TestGeminiRequiresAPIKey(t *testing.T) {
	base := func() *Config {
		cfg := tuningBase(t.TempDir())
		cfg.Provider.Type = ProviderTypeGemini
		cfg.Provider.BaseURL = ""
		return cfg
	}

	cfg := base()
	err := cfg.Validate()
	if err == nil {
		t.Fatal("type gemini without api_key must be a config error")
	}
	const want = "gemini needs api_key (Vertex support lands with the vertex block)"
	if !strings.Contains(err.Error(), want) {
		t.Errorf("error %q does not carry %q", err, want)
	}

	withKey := base()
	withKey.Provider.APIKey = "k"
	if err := withKey.Validate(); err != nil {
		t.Errorf("type gemini with an api_key must validate: %v", err)
	}

	// The rule is scoped to the gemini wire: a keyless openai-compatible config
	// is a local Ollama server, and a keyless anthropic one is already refused
	// by the endpoint itself rather than by amele.
	for _, typ := range []string{"", ProviderTypeOpenAI, ProviderTypeAnthropic} {
		other := tuningBase(t.TempDir())
		other.Provider.Type = typ
		if typ == ProviderTypeAnthropic {
			other.Provider.BaseURL = ""
		}
		if err := other.Validate(); err != nil {
			t.Errorf("type %q without api_key must still validate: %v", typ, err)
		}
	}
}

// TestProviderTypeOpenAIParity: "" and "openai" are the same OpenAI-compatible
// path - both accept a valid base_url and both keep requiring one.
func TestProviderTypeOpenAIParity(t *testing.T) {
	for _, typ := range []string{"", ProviderTypeOpenAI} {
		t.Run("type "+strconv.Quote(typ), func(t *testing.T) {
			cfg := &Config{
				Model:     "m",
				Provider:  ProviderConfig{Type: typ, BaseURL: "https://api.example.com/v1"},
				Workspace: t.TempDir(),
				Limits:    Limits{MaxTurns: 20},
			}
			if err := cfg.Validate(); err != nil {
				t.Errorf("valid openai-compatible config rejected: %v", err)
			}
		})
	}
}

// TestValidateReportsAllErrors verifies violations are joined rather than
// returned one-by-one: headless users should not fix cron configs by
// replaying failures.
func TestValidateReportsAllErrors(t *testing.T) {
	cfg := &Config{Workspace: t.TempDir()}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected error")
	}
	for _, sub := range []string{"model is required", "base_url is required"} {
		if !strings.Contains(err.Error(), sub) {
			t.Errorf("joined error missing %q: %v", sub, err)
		}
	}
}

// TestLoadShellBlock pins the tools.shell schema: absent block stays disabled
// (default-off is a security property, docs/engineering.md Phase 2), and every field
// round-trips.
func TestLoadShellBlock(t *testing.T) {
	dir := t.TempDir()
	path := writeConfig(t, dir, minimalYAML+`
tools:
  shell:
    enabled: true
    allow: ["git *", "ls*"]
    deny: ["rm *"]
    timeout: 30s
`)
	cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	sh := cfg.Tools.Shell
	if !sh.Enabled {
		t.Error("enabled: got false")
	}
	if len(sh.Allow) != 2 || sh.Allow[0] != "git *" || sh.Allow[1] != "ls*" {
		t.Errorf("allow: %+v", sh.Allow)
	}
	if len(sh.Deny) != 1 || sh.Deny[0] != "rm *" {
		t.Errorf("deny: %+v", sh.Deny)
	}
	if sh.Timeout.Std() != 30*time.Second {
		t.Errorf("timeout: %v", sh.Timeout.Std())
	}

	// SECURITY: an absent block must never mean "enabled".
	bare := writeConfig(t, t.TempDir(), minimalYAML)
	cfg, err = Load(bare, envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Tools.Shell.Enabled {
		t.Error("absent tools.shell block must leave the shell tool disabled")
	}
}

// TestLoadResolvesPathLikeCommands: the pack contract - a tool shipped
// inside the pack folder ("./tools/x.sh") must run no matter where the
// workspace points, while bare names keep PATH lookup semantics.
func TestLoadResolvesPathLikeCommands(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want func(dir string) string
	}{
		{"dot-slash relative", "./tools/scan.sh", func(d string) string { return filepath.Join(d, "tools", "scan.sh") }},
		{"bare relative with separator", "tools/scan.sh", func(d string) string { return filepath.Join(d, "tools", "scan.sh") }},
		{"bare name untouched", "msmtp", func(string) string { return "msmtp" }},
		{"absolute untouched", "/usr/bin/msmtp", func(string) string { return "/usr/bin/msmtp" }},
		{
			"workspace independence",
			"./tools/scan.sh",
			func(d string) string { return filepath.Join(d, "tools", "scan.sh") },
		},
	}
	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			configDir := t.TempDir()
			yaml := "model: m\nprovider:\n  base_url: https://x/v1\ntools:\n  subprocess:\n    - name: t\n      description: d\n      command: [" + strconv.Quote(tc.in) + "]\n"

			// For the workspace-independence case (index 4), add a separate workspace.
			if i == 4 {
				workspaceDir := t.TempDir()
				yaml += "workspace: " + strconv.Quote(workspaceDir) + "\n"
			}

			path := filepath.Join(configDir, "agent.yaml")
			if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
				t.Fatal(err)
			}
			cfg, err := Load(path, func(string) (string, bool) { return "", false })
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			if got := cfg.Tools.Subprocess[0].Command[0]; got != tc.want(configDir) {
				t.Errorf("Command[0] = %q, want %q", got, tc.want(configDir))
			}
		})
	}
}

// ptrFloat is the literal-to-pointer helper the sampling tests need:
// temperature and top_p are *float64 so that "unset" and "0" stay
// distinguishable (0 is a legal temperature and the usual choice for a judge
// agent).
func ptrFloat(v float64) *float64 { return &v }

// hasFloat reports whether p is set and points at want. It keeps the sampling
// assertions to one comparison each: the two questions ("is it set?" and "is it
// right?") have the same answer for a test that expects a value.
func hasFloat(p *float64, want float64) bool { return p != nil && *p == want }

// tuningBase is a valid config carrying no tuning at all - the starting point
// for every provider-tuning case, so a case's mutation is the only reason it
// can fail.
func tuningBase(dir string) *Config {
	return &Config{
		Model:     "m",
		Provider:  ProviderConfig{BaseURL: "https://api.example.com/v1"},
		Workspace: dir,
		Limits:    Limits{MaxTurns: 20},
	}
}

// TestLoadProviderTuning pins the YAML surface: dialect, reasoning, sampling
// and the raw params escape hatch all round-trip from the file into the
// struct, and a config that sets all of them validates.
func TestLoadProviderTuning(t *testing.T) {
	dir := t.TempDir()
	path := writeConfig(t, dir, `
model: test-model
provider:
  base_url: https://openrouter.ai/api/v1
  api_key: ${API_KEY}
  dialect: openrouter
  max_output_tokens: 65536
  reasoning:
    effort: high
    budget_tokens: 8192
  temperature: 0.2
  top_p: 0.9
  params:
    verbosity: low
    provider:
      require_parameters: true
`)
	cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Provider.Dialect != "openrouter" {
		t.Errorf("dialect = %q, want openrouter", cfg.Provider.Dialect)
	}
	if cfg.Provider.MaxOutputTokens != 65536 {
		t.Errorf("max_output_tokens = %d", cfg.Provider.MaxOutputTokens)
	}
	if cfg.Provider.Reasoning == nil {
		t.Fatal("reasoning block was dropped")
	}
	if cfg.Provider.Reasoning.Effort != "high" || cfg.Provider.Reasoning.BudgetTokens != 8192 {
		t.Errorf("reasoning = %+v", *cfg.Provider.Reasoning)
	}
	if !hasFloat(cfg.Provider.Temperature, 0.2) {
		t.Errorf("temperature = %v", cfg.Provider.Temperature)
	}
	if !hasFloat(cfg.Provider.TopP, 0.9) {
		t.Errorf("top_p = %v", cfg.Provider.TopP)
	}
	if got := cfg.Provider.Params["verbosity"]; got != "low" {
		t.Errorf("params.verbosity = %v", got)
	}
	if _, ok := cfg.Provider.Params["provider"].(map[string]any); !ok {
		t.Errorf("params.provider = %#v, want a nested mapping preserved verbatim", cfg.Provider.Params["provider"])
	}
	cfg.Workspace = dir
	if err := cfg.Validate(); err != nil {
		t.Errorf("a fully tuned config must validate: %v", err)
	}
}

// TestLoadWithoutProviderTuning is the compatibility half of the surface: a
// file that mentions none of the tuning keys leaves every one of them absent.
// That is what keeps "the provider decides" distinguishable from "the config
// asked for the zero value", and what makes the whole addition invisible to
// configs written before it existed.
func TestLoadWithoutProviderTuning(t *testing.T) {
	cfg, err := Load(writeConfig(t, t.TempDir(), minimalYAML), envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Provider.Dialect != "" || cfg.Provider.Reasoning != nil ||
		cfg.Provider.Temperature != nil || cfg.Provider.TopP != nil || cfg.Provider.Params != nil {
		t.Errorf("a config without a tuning block gained values: %+v", cfg.Provider)
	}
}

// TestValidateProviderTuning is the rule-by-rule table for the tuning surface.
// Every case is a rule an operator can only learn about at exit 2, so each
// message must name the field and say what to do instead.
func TestValidateProviderTuning(t *testing.T) {
	dir := t.TempDir()

	tests := []struct {
		name    string
		mutate  func(*Config)
		wantSub string // "" means the config must validate
	}{
		// Rule 1: the dialect must parse, and the message must list the
		// alternatives so a typo is fixable without the docs.
		{"unknown dialect", func(c *Config) { c.Provider.Dialect = "gemini" }, "openrouter"},
		{"dialect is case sensitive", func(c *Config) { c.Provider.Dialect = "DeepSeek" }, "provider.dialect"},
		{"known dialect", func(c *Config) { c.Provider.Dialect = "deepseek" }, ""},
		{"omitted dialect", func(c *Config) { c.Provider.Dialect = "" }, ""},

		// Rule 2: the effort vocabulary is the union of what the providers
		// accept; anything else would be dropped or 400 at run time.
		{"unknown effort", func(c *Config) { c.Provider.Reasoning = &ReasoningConfig{Effort: "insane"} }, "provider.reasoning.effort"},
		{"effort none", func(c *Config) { c.Provider.Reasoning = &ReasoningConfig{Effort: "none"} }, ""},
		{"effort xhigh", func(c *Config) { c.Provider.Reasoning = &ReasoningConfig{Effort: "xhigh"} }, ""},
		{"empty effort", func(c *Config) { c.Provider.Reasoning = &ReasoningConfig{} }, ""},

		// Rule 3: budget_tokens is only mapped on the anthropic wire or by
		// the openrouter gateway; anywhere else it would be silently dropped.
		{
			"negative budget_tokens",
			func(c *Config) { c.Provider.Reasoning = &ReasoningConfig{BudgetTokens: -1} },
			"provider.reasoning.budget_tokens must not be negative",
		},
		{
			"budget_tokens on the openai dialect",
			func(c *Config) { c.Provider.Reasoning = &ReasoningConfig{BudgetTokens: 8192} },
			"budget_tokens is only mapped for the anthropic or gemini wire, or the openrouter dialect",
		},
		{
			"budget_tokens on the deepseek dialect",
			func(c *Config) {
				c.Provider.Dialect = "deepseek"
				c.Provider.Reasoning = &ReasoningConfig{BudgetTokens: 8192}
			},
			"budget_tokens is only mapped for the anthropic or gemini wire, or the openrouter dialect",
		},
		{
			"budget_tokens on the openrouter dialect",
			func(c *Config) {
				c.Provider.Dialect = "openrouter"
				c.Provider.Reasoning = &ReasoningConfig{BudgetTokens: 8192}
			},
			"",
		},
		{
			"budget_tokens on the anthropic wire",
			func(c *Config) {
				c.Provider.Type = ProviderTypeAnthropic
				c.Provider.BaseURL = ""
				// Below the cap the client sends when max_output_tokens is
				// unset - see the implicit-cap cases under Rule 9.
				c.Provider.Reasoning = &ReasoningConfig{BudgetTokens: 4096}
			},
			"",
		},

		// Rule 3b: the gemini wire carries a thinkingBudget of its own
		// (thinkingConfig.thinkingBudget on the 2.5-era models), so a budget
		// alone is a mapped, legal request there - and the LEVEL and the count
		// are alternatives: the API 400s when both fields arrive.
		{
			"budget_tokens on the gemini wire",
			func(c *Config) {
				c.Provider.Type = ProviderTypeGemini
				c.Provider.BaseURL = ""
				c.Provider.APIKey = "k"
				c.Provider.Reasoning = &ReasoningConfig{BudgetTokens: 8192}
			},
			"",
		},
		{
			"effort alone on the gemini wire",
			func(c *Config) {
				c.Provider.Type = ProviderTypeGemini
				c.Provider.BaseURL = ""
				c.Provider.APIKey = "k"
				c.Provider.Reasoning = &ReasoningConfig{Effort: "high"}
			},
			"",
		},
		{
			"effort and budget_tokens together on the gemini wire",
			func(c *Config) {
				c.Provider.Type = ProviderTypeGemini
				c.Provider.BaseURL = ""
				c.Provider.APIKey = "k"
				c.Provider.Reasoning = &ReasoningConfig{Effort: "high", BudgetTokens: 8192}
			},
			"gemini accepts thinkingLevel or thinkingBudget, not both",
		},
		{
			// The pair is a relation between two gemini-wire fields, so an
			// illegal dialect (itself reported) must not swallow it: validate
			// owes one pass, every violation.
			"effort and budget_tokens together survive a dialect on the gemini wire",
			func(c *Config) {
				c.Provider.Type = ProviderTypeGemini
				c.Provider.BaseURL = ""
				c.Provider.APIKey = "k"
				c.Provider.Dialect = "kimi"
				c.Provider.Reasoning = &ReasoningConfig{Effort: "high", BudgetTokens: 8192}
			},
			"gemini accepts thinkingLevel or thinkingBudget, not both",
		},
		{
			// The same pair is legal on the openrouter gateway, which converts
			// the budget itself: this rule is about the gemini wire only.
			"effort and budget_tokens together is not the openrouter rule",
			func(c *Config) {
				c.Provider.Dialect = "openrouter"
				c.Provider.Reasoning = &ReasoningConfig{Effort: "high", BudgetTokens: 8192}
			},
			"",
		},
		{
			// Deliberately NOT a rule: "none" maps to thinkingBudget 0 on the
			// 2.5-era models and is refused by the Gemini 3 generation, and
			// validate cannot know which generation a model string names. It
			// stays a runtime error with a mapped message (spec §Mapping),
			// because refusing it here would reject a combination that works.
			"effort none on the gemini wire",
			func(c *Config) {
				c.Provider.Type = ProviderTypeGemini
				c.Provider.BaseURL = ""
				c.Provider.APIKey = "k"
				c.Provider.Reasoning = &ReasoningConfig{Effort: "none"}
			},
			"",
		},
		{
			"dialect on the gemini wire",
			func(c *Config) {
				c.Provider.Type = ProviderTypeGemini
				c.Provider.BaseURL = ""
				c.Provider.APIKey = "k"
				c.Provider.Dialect = "deepseek"
			},
			"provider.dialect: \"deepseek\" applies to the openai wire; remove it for type gemini",
		},
		{
			"gemini without a dialect",
			func(c *Config) {
				c.Provider.Type = ProviderTypeGemini
				c.Provider.BaseURL = ""
				c.Provider.APIKey = "k"
			},
			"",
		},

		// Rule 4: the K-series fixes its sampling values, so anything else is
		// a 400 - caught here instead of on the first cron run.
		{
			"kimi with temperature",
			func(c *Config) {
				c.Provider.Dialect = "kimi"
				c.Provider.Temperature = ptrFloat(0.2)
			},
			"kimi K-series models fix sampling; remove temperature/top_p",
		},
		{
			"kimi with top_p",
			func(c *Config) {
				c.Provider.Dialect = "kimi"
				c.Provider.TopP = ptrFloat(0.9)
			},
			"kimi K-series models fix sampling; remove temperature/top_p",
		},
		{"kimi without sampling", func(c *Config) { c.Provider.Dialect = "kimi" }, ""},
		{
			// The dialect describes an openai-wire variation and is documented
			// as ignored when type is anthropic (config.schema.json), so a
			// leftover dialect must not refuse a config for a provider it is
			// not talking to.
			"kimi dialect is inert on the anthropic wire",
			func(c *Config) {
				c.Provider.Type = ProviderTypeAnthropic
				c.Provider.BaseURL = ""
				c.Provider.Dialect = "kimi"
				c.Provider.Temperature = ptrFloat(0.5)
				c.Provider.TopP = ptrFloat(0.9)
			},
			"",
		},

		// Rule 5: Kimi's thinking models cannot be switched off.
		{
			"kimi with effort none",
			func(c *Config) {
				c.Provider.Dialect = "kimi"
				c.Provider.Reasoning = &ReasoningConfig{Effort: "none"}
			},
			"kimi models cannot disable thinking",
		},
		{
			"kimi with effort high",
			func(c *Config) {
				c.Provider.Dialect = "kimi"
				c.Provider.Reasoning = &ReasoningConfig{Effort: "high"}
			},
			"",
		},
		{
			// Same reason as the sampling case above: on the anthropic wire the
			// dialect is not consulted, and "none" is a legal thinking setting
			// there.
			"kimi dialect does not block effort none on the anthropic wire",
			func(c *Config) {
				c.Provider.Type = ProviderTypeAnthropic
				c.Provider.BaseURL = ""
				c.Provider.Dialect = "kimi"
				c.Provider.Reasoning = &ReasoningConfig{Effort: "none"}
			},
			"",
		},

		// Rule 6: params is a raw merge into the request body, so a key amele
		// sets itself would either be overwritten or overwrite a contract.
		{
			"params collides with an owned field",
			func(c *Config) { c.Provider.Params = map[string]any{"temperature": 0.5} },
			`provider.params key "temperature"`,
		},
		{
			"params collides with messages",
			func(c *Config) { c.Provider.Params = map[string]any{"messages": []any{}} },
			`provider.params key "messages"`,
		},
		{
			// Owned where it is actually emitted: the deepseek mapper writes a
			// thinking object, so params must not also write one.
			"params collides with thinking on deepseek",
			func(c *Config) {
				c.Provider.Dialect = "deepseek"
				c.Provider.Params = map[string]any{"thinking": map[string]any{"type": "enabled"}}
			},
			`provider.params key "thinking"`,
		},
		{
			// ... and free where it is not: kimi emits no thinking object, so
			// the K2.x controls stay reachable through the escape hatch.
			"params thinking is allowed on kimi",
			func(c *Config) {
				c.Provider.Dialect = "kimi"
				c.Provider.Params = map[string]any{"thinking": map[string]any{"type": "enabled", "keep": true}}
			},
			"",
		},
		{
			"params with a provider-specific key",
			func(c *Config) { c.Provider.Params = map[string]any{"verbosity": "low", "clear_thinking": false} },
			"",
		},
		{
			// The gemini wire owns a different set of body keys than the openai
			// one, in both spellings the API accepts.
			"params collides with a gemini owned field",
			func(c *Config) {
				c.Provider.Type = ProviderTypeGemini
				c.Provider.BaseURL = ""
				c.Provider.APIKey = "k"
				c.Provider.Params = map[string]any{"generationConfig": map[string]any{"topK": 5}}
			},
			`provider.params key "generationConfig"`,
		},
		{
			"params collides with a gemini owned field in snake_case",
			func(c *Config) {
				c.Provider.Type = ProviderTypeGemini
				c.Provider.BaseURL = ""
				c.Provider.APIKey = "k"
				c.Provider.Params = map[string]any{"safety_settings": []any{}}
			},
			`provider.params key "safety_settings"`,
		},
		{
			// ... and an openai-wire key amele never writes on this wire stays
			// reachable, exactly as thinking does on kimi.
			"params messages is allowed on the gemini wire",
			func(c *Config) {
				c.Provider.Type = ProviderTypeGemini
				c.Provider.BaseURL = ""
				c.Provider.APIKey = "k"
				c.Provider.Params = map[string]any{"messages": []any{}}
			},
			"",
		},
		{
			// The reserved keys are refused on EVERY target: amele's transport
			// reads a single JSON body and its loop owns the tool protocol.
			"params stream is refused on the gemini wire",
			func(c *Config) {
				c.Provider.Type = ProviderTypeGemini
				c.Provider.BaseURL = ""
				c.Provider.APIKey = "k"
				c.Provider.Params = map[string]any{"stream": true}
			},
			`provider.params key "stream"`,
		},

		// Rule 7: ranges. They are total for the dialect, so they belong in
		// validate rather than in a runtime error message.
		{"temperature too high", func(c *Config) { c.Provider.Temperature = ptrFloat(2.5) }, "provider.temperature"},
		{"temperature negative", func(c *Config) { c.Provider.Temperature = ptrFloat(-0.1) }, "provider.temperature"},
		{"temperature at the ceiling", func(c *Config) { c.Provider.Temperature = ptrFloat(2) }, ""},
		{"temperature zero", func(c *Config) { c.Provider.Temperature = ptrFloat(0) }, ""},
		{
			"glm narrows temperature",
			func(c *Config) {
				c.Provider.Dialect = "glm"
				c.Provider.Temperature = ptrFloat(1.5)
			},
			"provider.temperature",
		},
		{
			"glm temperature at its ceiling",
			func(c *Config) {
				c.Provider.Dialect = "glm"
				c.Provider.Temperature = ptrFloat(1)
			},
			"",
		},
		{
			// The Anthropic Messages API documents temperature as 0..1 for
			// every model it serves - a WIRE-total limit, so it belongs here
			// exactly as the glm dialect's does.
			"anthropic wire narrows temperature",
			func(c *Config) {
				c.Provider.Type = ProviderTypeAnthropic
				c.Provider.BaseURL = ""
				c.Provider.Temperature = ptrFloat(1.5)
			},
			"provider.temperature",
		},
		{
			"anthropic temperature at its ceiling",
			func(c *Config) {
				c.Provider.Type = ProviderTypeAnthropic
				c.Provider.BaseURL = ""
				c.Provider.Temperature = ptrFloat(1)
			},
			"",
		},
		{
			// The gemini wire documents temperature 0..2 like the OpenAI
			// baseline, so it keeps the baseline ceiling: Google's advice to
			// leave Gemini 3 at 1.0 is a recommendation per model generation,
			// not a total limit of the wire, and explain reports it as a note.
			"gemini wire keeps the baseline temperature ceiling",
			func(c *Config) {
				c.Provider.Type = ProviderTypeGemini
				c.Provider.BaseURL = ""
				c.Provider.APIKey = "k"
				c.Provider.Temperature = ptrFloat(1.5)
			},
			"",
		},
		{
			"gemini wire still refuses a temperature above the baseline",
			func(c *Config) {
				c.Provider.Type = ProviderTypeGemini
				c.Provider.BaseURL = ""
				c.Provider.APIKey = "k"
				c.Provider.Temperature = ptrFloat(2.5)
			},
			"provider.temperature",
		},
		{
			"glm dialect keeps the anthropic ceiling on the anthropic wire",
			func(c *Config) {
				c.Provider.Type = ProviderTypeAnthropic
				c.Provider.BaseURL = ""
				c.Provider.Dialect = "glm"
				c.Provider.Temperature = ptrFloat(0.8)
			},
			"",
		},
		// NaN is reachable from both directions (YAML ".nan", --set "NaN")
		// and every comparison against it is false, so a range check written
		// the natural way would let it through to a request body JSON cannot
		// even encode.
		{"temperature NaN", func(c *Config) { c.Provider.Temperature = ptrFloat(math.NaN()) }, "provider.temperature"},
		{"top_p NaN", func(c *Config) { c.Provider.TopP = ptrFloat(math.NaN()) }, "provider.top_p"},
		{"temperature infinity", func(c *Config) { c.Provider.Temperature = ptrFloat(math.Inf(1)) }, "provider.temperature"},
		{"top_p above one", func(c *Config) { c.Provider.TopP = ptrFloat(1.1) }, "provider.top_p"},
		{"top_p zero", func(c *Config) { c.Provider.TopP = ptrFloat(0) }, "provider.top_p"},
		{"top_p at one", func(c *Config) { c.Provider.TopP = ptrFloat(1) }, ""},

		// Rule 8: params is serialized to JSON verbatim, so a value JSON
		// cannot express must fail here rather than at the first request.
		{
			"params value is not JSON-serializable",
			func(c *Config) { c.Provider.Params = map[string]any{"x": map[any]any{1: "a"}} },
			"provider.params",
		},
		{
			"params NaN is not JSON-serializable",
			func(c *Config) { c.Provider.Params = map[string]any{"x": math.NaN()} },
			"provider.params",
		},

		// Rule 9: on the anthropic wire the thinking budget is drawn FROM the
		// output cap, so a budget that meets or exceeds the cap leaves no room
		// for an answer. The API answers it with a 400, which would arrive as
		// an exit-5 provider error in the middle of a cron run; the numbers are
		// both in the file, so the mistake is knowable at validate time.
		{
			"budget_tokens at the anthropic output cap",
			func(c *Config) {
				c.Provider.Type = ProviderTypeAnthropic
				c.Provider.BaseURL = ""
				c.Provider.MaxOutputTokens = 8192
				c.Provider.Reasoning = &ReasoningConfig{BudgetTokens: 8192}
			},
			"reasoning.budget_tokens must be below provider.max_output_tokens",
		},
		{
			"budget_tokens above the anthropic output cap",
			func(c *Config) {
				c.Provider.Type = ProviderTypeAnthropic
				c.Provider.BaseURL = ""
				c.Provider.MaxOutputTokens = 4096
				c.Provider.Reasoning = &ReasoningConfig{BudgetTokens: 8192}
			},
			"reasoning.budget_tokens must be below provider.max_output_tokens",
		},
		{
			"budget_tokens below the anthropic output cap",
			func(c *Config) {
				c.Provider.Type = ProviderTypeAnthropic
				c.Provider.BaseURL = ""
				c.Provider.MaxOutputTokens = 16384
				c.Provider.Reasoning = &ReasoningConfig{BudgetTokens: 8192}
			},
			"",
		},
		{
			// An unset cap is NOT an absent one on this wire: max_tokens is
			// required, so the client sends its own default and the API
			// measures the budget against that number. Same relation, same
			// 400, and the config still contains everything needed to know it.
			"budget_tokens at the implicit anthropic cap",
			func(c *Config) {
				c.Provider.Type = ProviderTypeAnthropic
				c.Provider.BaseURL = ""
				c.Provider.Reasoning = &ReasoningConfig{BudgetTokens: llm.DefaultAnthropicMaxOutput}
			},
			"reasoning.budget_tokens must be below provider.max_output_tokens",
		},
		{
			"budget_tokens below the implicit anthropic cap",
			func(c *Config) {
				c.Provider.Type = ProviderTypeAnthropic
				c.Provider.BaseURL = ""
				c.Provider.Reasoning = &ReasoningConfig{BudgetTokens: llm.DefaultAnthropicMaxOutput - 1}
			},
			"",
		},
		{
			// The openrouter gateway carves the budget out of max_tokens
			// itself, so the same pair is legal there; this rule is about the
			// anthropic wire only.
			"budget_tokens at the cap is not the openrouter rule",
			func(c *Config) {
				c.Provider.Dialect = "openrouter"
				c.Provider.MaxOutputTokens = 8192
				c.Provider.Reasoning = &ReasoningConfig{BudgetTokens: 8192}
			},
			"",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := tuningBase(dir)
			tt.mutate(cfg)
			err := cfg.Validate()
			if tt.wantSub == "" {
				if err != nil {
					t.Fatalf("config must validate, got: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatal("expected a violation")
			}
			if !errors.Is(err, ErrInvalid) {
				t.Errorf("error should wrap ErrInvalid: %v", err)
			}
			if !strings.Contains(err.Error(), tt.wantSub) {
				t.Errorf("error %q does not mention %q", err, tt.wantSub)
			}
		})
	}
}

// TestValidateProviderRetry is the rule table for the retry policy. The bounds
// exist because both knobs multiply: a large max_attempts with a large
// initial_backoff turns one rate-limited turn into a wait longer than most cron
// windows, and the operator would see it as a hung run, not as a setting.
func TestValidateProviderRetry(t *testing.T) {
	dir := t.TempDir()

	tests := []struct {
		name    string
		retry   *RetryConfig
		wantSub string // "" means the config must validate
	}{
		{"no retry block", nil, ""},
		{"empty retry block means defaults", &RetryConfig{}, ""},

		// Zero is "omitted" for both fields: the struct cannot tell an absent
		// key from a written 0, and every other budget in this file spells
		// "use the default" that way (limits.timeout, request_timeout).
		{"explicit zero attempts is the default", &RetryConfig{MaxAttempts: 0}, ""},
		{"explicit zero backoff is the default", &RetryConfig{InitialBackoff: 0}, ""},

		{"one attempt disables retrying", &RetryConfig{MaxAttempts: 1}, ""},
		{"ten attempts is the ceiling", &RetryConfig{MaxAttempts: 10}, ""},
		{"eleven attempts", &RetryConfig{MaxAttempts: 11}, "provider.retry.max_attempts"},
		{"negative attempts", &RetryConfig{MaxAttempts: -1}, "provider.retry.max_attempts"},

		{"backoff at the floor", &RetryConfig{InitialBackoff: Duration(100 * time.Millisecond)}, ""},
		{"backoff at the ceiling", &RetryConfig{InitialBackoff: Duration(60 * time.Second)}, ""},
		{"backoff below the floor", &RetryConfig{InitialBackoff: Duration(50 * time.Millisecond)}, "provider.retry.initial_backoff"},
		{"backoff above the ceiling", &RetryConfig{InitialBackoff: Duration(90 * time.Second)}, "provider.retry.initial_backoff"},
		{"negative backoff", &RetryConfig{InitialBackoff: Duration(-time.Second)}, "provider.retry.initial_backoff"},

		{"both knobs set", &RetryConfig{MaxAttempts: 5, InitialBackoff: Duration(2 * time.Second)}, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := tuningBase(dir)
			cfg.Provider.Retry = tt.retry
			err := cfg.Validate()
			if tt.wantSub == "" {
				if err != nil {
					t.Fatalf("config must validate, got: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatal("expected a violation")
			}
			if !errors.Is(err, ErrInvalid) {
				t.Errorf("error should wrap ErrInvalid: %v", err)
			}
			if !strings.Contains(err.Error(), tt.wantSub) {
				t.Errorf("error %q does not mention %q", err, tt.wantSub)
			}
		})
	}
}

// TestValidateProviderRetryMessagesAreActionable pins the two messages
// themselves: an out-of-range value is only fixable if the error states the
// accepted range and the default the operator gets by deleting the key.
func TestValidateProviderRetryMessagesAreActionable(t *testing.T) {
	cfg := tuningBase(t.TempDir())
	cfg.Provider.Retry = &RetryConfig{MaxAttempts: 99, InitialBackoff: Duration(5 * time.Millisecond)}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected violations")
	}
	for _, want := range []string{
		"provider.retry.max_attempts", "1 and 10", "default 3",
		"provider.retry.initial_backoff", "100ms and 60s", "default 1s",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

// TestLoadProviderRetry pins the YAML surface: the block round-trips from the
// file into the struct, durations included, and a config that omits it keeps a
// nil block - "the client decides" must stay distinguishable from "the file
// asked for zero".
func TestLoadProviderRetry(t *testing.T) {
	dir := t.TempDir()
	path := writeConfig(t, dir, `
model: test-model
provider:
  base_url: https://api.example.com/v1
  api_key: ${API_KEY}
  retry:
    max_attempts: 5
    initial_backoff: 250ms
`)
	cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Provider.Retry == nil {
		t.Fatal("retry block was dropped")
	}
	if cfg.Provider.Retry.MaxAttempts != 5 {
		t.Errorf("max_attempts = %d, want 5", cfg.Provider.Retry.MaxAttempts)
	}
	if got := cfg.Provider.Retry.InitialBackoff.Std(); got != 250*time.Millisecond {
		t.Errorf("initial_backoff = %v, want 250ms", got)
	}

	plain, err := Load(writeConfig(t, dir, minimalYAML), envMap(map[string]string{"API_KEY": "k"}))
	if err != nil {
		t.Fatalf("Load minimal: %v", err)
	}
	if plain.Provider.Retry != nil {
		t.Errorf("a config without a retry block gained one: %+v", plain.Provider.Retry)
	}
}

// TestValidateSamplingWireBeatsDialect pins WHICH rule answers when both could
// narrow the temperature range. The wire wins: with type anthropic the dialect
// is documented as ignored (config.schema.json), so an operator who left a
// dialect behind while switching wires must be told about the wire they are
// actually talking to, not about a provider that is out of the picture.
func TestValidateSamplingWireBeatsDialect(t *testing.T) {
	cfg := tuningBase(t.TempDir())
	cfg.Provider.Type = ProviderTypeAnthropic
	cfg.Provider.BaseURL = ""
	cfg.Provider.Dialect = "glm"
	cfg.Provider.Temperature = ptrFloat(1.5)

	err := cfg.Validate()
	if err == nil {
		t.Fatal("temperature 1.5 must be refused on the anthropic wire")
	}
	if !strings.Contains(err.Error(), "anthropic wire") {
		t.Errorf("error %q does not name the wire that set the ceiling", err)
	}
	if strings.Contains(err.Error(), "glm dialect") {
		t.Errorf("error %q blames the dialect, which the anthropic wire ignores", err)
	}
}

// TestValidateParamsOwnedKeysAreDialectScoped walks the owned-field set per
// TARGET. Each name the active dialect/wire writes into the request body itself
// must be refused in params, or the escape hatch would silently overwrite a
// contract (the tool definitions, the response format, the budget cap) - and
// each name that target never writes must be ACCEPTED.
//
// The second half is the regression: the check used to refuse the union of
// every wire's spellings, so `params: {thinking: ...}` was a config error on
// kimi although the kimi mapper emits no thinking object at all. That left the
// K2.x thinking controls unreachable from any config, blocked by a collision
// with a field nothing was going to send.
func TestValidateParamsOwnedKeysAreDialectScoped(t *testing.T) {
	dir := t.TempDir()
	anthropicWire := func(c *Config) {
		c.Provider.Type = ProviderTypeAnthropic
		c.Provider.BaseURL = ""
	}
	dialect := func(name string) func(*Config) {
		return func(c *Config) { c.Provider.Dialect = name }
	}
	// Keys every openai-wire dialect writes, whatever else it maps.
	shared := []string{"model", "messages", "tools", "response_format", "temperature", "top_p"}
	tests := []struct {
		name string
		// apply selects the target.
		apply func(*Config)
		// refused must be a config error; allowed must validate.
		refused []string
		allowed []string
	}{
		{
			name:    "openai dialect",
			apply:   func(*Config) {},
			refused: append(shared, "max_completion_tokens", "reasoning_effort", "stream", "tool_choice"),
			allowed: []string{"max_tokens", "thinking", "reasoning", "output_config", "system"},
		},
		{
			name:    "deepseek dialect",
			apply:   dialect("deepseek"),
			refused: append(shared, "max_tokens", "reasoning_effort", "thinking"),
			allowed: []string{"max_completion_tokens", "reasoning", "output_config", "system"},
		},
		{
			// The fix: kimi emits no thinking object, so params may carry the
			// K2.x controls.
			name:    "kimi dialect",
			apply:   dialect("kimi"),
			refused: append(shared, "max_completion_tokens", "reasoning_effort"),
			allowed: []string{"thinking", "reasoning", "max_tokens"},
		},
		{
			name:    "openrouter dialect",
			apply:   dialect("openrouter"),
			refused: append(shared, "max_tokens", "reasoning"),
			allowed: []string{"reasoning_effort", "thinking", "max_completion_tokens"},
		},
		{
			name:    "anthropic wire",
			apply:   anthropicWire,
			refused: []string{"model", "messages", "tools", "temperature", "top_p", "max_tokens", "thinking", "output_config", "system", "stream", "tool_choice"},
			allowed: []string{"response_format", "reasoning", "reasoning_effort", "max_completion_tokens"},
		},
	}

	for _, tt := range tests {
		for _, key := range tt.refused {
			t.Run(tt.name+"/refused/"+key, func(t *testing.T) {
				cfg := tuningBase(dir)
				tt.apply(cfg)
				cfg.Provider.Params = map[string]any{key: "x"}
				err := cfg.Validate()
				if err == nil {
					t.Fatalf("params key %q was accepted", key)
				}
				if !strings.Contains(err.Error(), key) {
					t.Errorf("error %q does not name the colliding key %q", err, key)
				}
			})
		}
		for _, key := range tt.allowed {
			t.Run(tt.name+"/allowed/"+key, func(t *testing.T) {
				cfg := tuningBase(dir)
				tt.apply(cfg)
				cfg.Provider.Params = map[string]any{key: "x"}
				if err := cfg.Validate(); err != nil {
					t.Fatalf("params key %q is not written by this target but was refused: %v", key, err)
				}
			})
		}
	}
}

// TestValidateParamsSkipsCollisionsOnUnknownDialect: which keys a DIALECT owns
// is a dialect question, so an unparseable dialect makes it unanswerable. The
// dialect itself is reported (once); guessing a collision on top would send the
// operator to the wrong line, exactly as the other dialect-dependent rules
// already decided.
//
// The reserved keys are the exception and are reported in the same pass: they
// are refused on EVERY target - amele's own machinery cannot survive them - so
// the dialect has no say, and withholding them would cost the operator a second
// validate round for a violation that was already answerable.
func TestValidateParamsSkipsCollisionsOnUnknownDialect(t *testing.T) {
	cfg := tuningBase(t.TempDir())
	cfg.Provider.Dialect = "kimi-k3"
	cfg.Provider.Params = map[string]any{"messages": "x"}

	joined := strings.Join(cfg.Violations(), "\n")
	if !strings.Contains(joined, "provider.dialect") {
		t.Fatalf("violations do not report the dialect:\n%s", joined)
	}
	if strings.Contains(joined, "provider.params key") {
		t.Errorf("dialect-owned collision reported against a dialect that did not parse:\n%s", joined)
	}
	// Both reserved keys surface in the same pass as the dialect violation.
	cfg.Provider.Params = map[string]any{"stream": true, "tool_choice": "required"}
	joined = strings.Join(cfg.Violations(), "\n")
	for _, key := range []string{"stream", "tool_choice"} {
		if !strings.Contains(joined, "provider.params key "+strconv.Quote(key)) {
			t.Errorf("reserved key %q was not reported on an unparseable dialect:\n%s", key, joined)
		}
	}
	// The dialect-independent params rule still fires in the same pass.
	cfg.Provider.Params = map[string]any{"bad": math.NaN()}
	if joined := strings.Join(cfg.Violations(), "\n"); !strings.Contains(joined, "not JSON-serializable") {
		t.Errorf("the dialect-independent params check was skipped too:\n%s", joined)
	}
}

// TestValidateDialectViolationSuppressesDialectRules: an unparseable dialect
// makes every dialect-dependent rule unanswerable, so validate reports the
// dialect once instead of piling on rules the operator cannot act on until it
// is fixed. The dialect-independent rules still fire in the same pass.
func TestValidateDialectViolationSuppressesDialectRules(t *testing.T) {
	cfg := tuningBase(t.TempDir())
	cfg.Provider.Dialect = "kimi-k3"
	cfg.Provider.Temperature = ptrFloat(0.2)
	cfg.Provider.Reasoning = &ReasoningConfig{Effort: "bogus", BudgetTokens: 4096}

	msgs := cfg.Violations()
	joined := strings.Join(msgs, "\n")
	if !strings.Contains(joined, "provider.dialect") {
		t.Fatalf("violations do not report the dialect:\n%s", joined)
	}
	if strings.Contains(joined, "kimi K-series models fix sampling") ||
		strings.Contains(joined, "budget_tokens is only mapped") {
		t.Errorf("dialect-dependent rules fired on an unparseable dialect:\n%s", joined)
	}
	if !strings.Contains(joined, "provider.reasoning.effort") {
		t.Errorf("the dialect-independent effort rule was skipped:\n%s", joined)
	}
}

// TestValidateBudgetCapReportedOnUnknownDialect: the budget-fits-cap rule is
// dialect-INDEPENDENT - it is a relation between two anthropic-wire fields, and
// `type: anthropic` ignores the dialect entirely - so an unparseable dialect
// must not hide it. It used to sit below the dialect early return, which cost
// the operator a second validate round for a violation that was answerable in
// the first: the Messages API would have 400'd on it either way.
func TestValidateBudgetCapReportedOnUnknownDialect(t *testing.T) {
	cfg := tuningBase(t.TempDir())
	cfg.Provider.Type = ProviderTypeAnthropic
	cfg.Provider.BaseURL = ""
	// A leftover dialect that does not parse, alongside a budget with no cap to
	// fit into: two independent mistakes, one pass.
	cfg.Provider.Dialect = "kimi-k3"
	cfg.Provider.Reasoning = &ReasoningConfig{BudgetTokens: llm.DefaultAnthropicMaxOutput}

	joined := strings.Join(cfg.Violations(), "\n")
	if !strings.Contains(joined, "provider.dialect") {
		t.Errorf("violations do not report the dialect:\n%s", joined)
	}
	if !strings.Contains(joined, "reasoning.budget_tokens must be below provider.max_output_tokens") {
		t.Errorf("the dialect-independent budget/cap rule was hidden by the dialect:\n%s", joined)
	}
	// The dialect-DEPENDENT rules stay suppressed, unchanged.
	if strings.Contains(joined, "budget_tokens is only mapped") {
		t.Errorf("a dialect-dependent rule fired on an unparseable dialect:\n%s", joined)
	}
}

// TestToolsParallelDefault: an omitted tools.parallel must mean "on". The
// default is the whole point of the field - configs written before v0.2 get
// concurrent tool calls without being edited, and only an explicit `false`
// takes it back.
func TestToolsParallelDefault(t *testing.T) {
	cases := []struct {
		name     string
		fragment string
		want     bool
	}{
		{name: "omitted", fragment: "", want: true},
		{name: "explicit true", fragment: "tools:\n  parallel: true\n", want: true},
		{name: "explicit false", fragment: "tools:\n  parallel: false\n", want: false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := writeConfig(t, t.TempDir(), minimalYAML+tc.fragment)
			cfg, err := Load(path, envMap(map[string]string{"API_KEY": "k"}))
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			if err := cfg.Validate(); err != nil {
				t.Fatalf("Validate: %v", err)
			}
			if got := cfg.Tools.IsParallel(); got != tc.want {
				t.Errorf("IsParallel() = %v, want %v", got, tc.want)
			}
		})
	}
}
