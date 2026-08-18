package config

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
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
			func(c *Config) { c.Provider.Type = "gemini" },
			"openai, anthropic",
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
