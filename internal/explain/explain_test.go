package explain

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/llm"
	"github.com/lasthumanintheloop/amele/internal/session"
	"github.com/lasthumanintheloop/amele/internal/tools"
)

// alwaysFound is an ExecProbe stub for tests that exercise sections other than
// requirements: it keeps Render's executable probe off the host PATH so the
// suite stays hermetic regardless of what happens to be installed.
func alwaysFound(string) error { return nil }

// ptrBool returns a pointer to b, for the tri-state config fields (tools.parallel)
// where nil, true and false are three different answers.
func ptrBool(b bool) *bool { return &b }

// stubTool is the minimal tools.Tool for registry construction: explain only
// reads names, so Def carries a name and Invoke is never called.
type stubTool struct{ name string }

func (s stubTool) Def() llm.ToolDef { return llm.ToolDef{Name: s.name, Description: "stub"} }

func (s stubTool) Invoke(context.Context, string) (string, error) { return "", nil }

// registryWith builds a registry containing stub tools with the given names.
func registryWith(t *testing.T, names ...string) *tools.Registry {
	t.Helper()
	reg := tools.NewRegistry()
	for _, n := range names {
		if err := reg.Register(stubTool{name: n}); err != nil {
			t.Fatal(err)
		}
	}
	return reg
}

// baseCfg is a fully-specified config that produces zero warnings, so each
// table case below flips exactly one dimension.
func baseCfg() *config.Config {
	return &config.Config{
		Model: "test-model",
		Provider: config.ProviderConfig{
			Type:    config.ProviderTypeOpenAI,
			BaseURL: "https://api.example.com/v1",
			APIKey:  "sk-super-secret-value",
		},
		Workspace:  "/ws",
		Tools:      config.ToolsConfig{FS: true},
		Limits:     config.Limits{MaxTurns: 20, MaxTokens: 50000},
		SessionDir: "/ws/sessions",
	}
}

// fsBuiltins is the registry contents a case gets when it does not name its
// own tools - the fs builtins baseCfg enables.
var fsBuiltins = []string{"fs_read", "fs_write", "fs_list"}

func TestRender(t *testing.T) {
	cases := []struct {
		name string
		cfg  func() *config.Config
		// regTools names the tools the registry holds; nil means fsBuiltins.
		regTools []string
		want     []string
		notWant  []string
	}{
		{
			name: "openai defaults",
			cfg:  baseCfg,
			want: []string{
				"MODEL & PROVIDER\n  model:           \"test-model\"\n  provider type:   \"openai\"\n  base_url:        \"https://api.example.com/v1\"\n  request_timeout: default (120s)\n",
				"fs builtins (fs_read, fs_write, fs_list): enabled",
				"shell: disabled",
				"subprocess tools: (none)",
				"BUDGETS\n  max_turns:  20\n  max_tokens: 50000\n  timeout:    none\n",
				"OUTPUT\n  schema: none (final answer is unconstrained text)\n",
				"session_dir: \"/ws/sessions\"",
				"WARNINGS\n  (none)\n",
			},
			notWant: []string{"sk-super-secret-value", "api_key"},
		},
		{
			name: "anthropic default base url",
			cfg: func() *config.Config {
				c := baseCfg()
				c.Provider = config.ProviderConfig{
					Type:            config.ProviderTypeAnthropic,
					APIKey:          "sk-super-secret-value",
					MaxOutputTokens: 1024,
				}
				return c
			},
			want: []string{
				"provider type:   \"anthropic\"",
				"base_url:        (default: api.anthropic.com)",
				"max_output_tokens: 1024",
			},
			notWant: []string{"sk-super-secret-value"},
		},
		{
			name: "unknown permission entry warns",
			cfg: func() *config.Config {
				c := baseCfg()
				c.Permissions = config.Permissions{
					Default: config.PolicyAsk,
					Tools: map[string]config.Policy{
						"zz_ghost": config.PolicyDeny,
						"aa_ghost": config.PolicyDeny,
						"fs_write": config.PolicyAsk,
					},
				}
				return c
			},
			want: []string{
				"default policy: \"ask\"",
				// Per-tool entries and warnings are sorted by tool name, so
				// aa_ghost precedes zz_ghost deterministically.
				"    \"aa_ghost\": \"deny\"\n    \"fs_write\": \"ask\"\n    \"zz_ghost\": \"deny\"\n",
				"without a TTY, every \"ask\" policy is auto-denied",
				"  - permission entry \"aa_ghost\" matches no tool - typo?\n  - permission entry \"zz_ghost\" matches no tool - typo?\n",
			},
		},
		{
			name: "glob permission entry matching a tool does not warn",
			cfg: func() *config.Config {
				c := baseCfg()
				c.Permissions = config.Permissions{
					Default: config.PolicyAsk,
					Tools: map[string]config.Policy{
						"github__*": config.PolicyAsk,
						"gitlab__*": config.PolicyDeny,
					},
				}
				return c
			},
			regTools: []string{"github__create_issue"},
			want: []string{
				// Only the glob that truly matches nothing is a typo suspect.
				"  - permission entry \"gitlab__*\" matches no tool - typo?\n",
			},
			notWant: []string{"permission entry \"github__*\" matches no tool"},
		},
		{
			name: "shell enabled without patterns warns",
			cfg: func() *config.Config {
				c := baseCfg()
				c.Tools.Shell = config.ShellConfig{Enabled: true}
				return c
			},
			regTools: []string{"shell"},
			want: []string{
				"shell: ENABLED",
				"allow patterns: (none)",
				"deny patterns:  (none)",
				"tools.shell is enabled with no allow or deny patterns",
			},
		},
		{
			name: "shell patterns listed",
			cfg: func() *config.Config {
				c := baseCfg()
				c.Tools.Shell = config.ShellConfig{
					Enabled: true,
					Allow:   []string{"git *", "ls *"},
					Deny:    []string{"git push*"},
				}
				return c
			},
			regTools: []string{"shell"},
			want: []string{
				"allow patterns: \"git *\", \"ls *\"",
				"deny patterns:  \"git push*\"",
			},
			notWant: []string{"tools.shell is enabled with no allow or deny patterns"},
		},
		{
			name: "unbounded tokens and no session warn",
			cfg: func() *config.Config {
				c := baseCfg()
				c.Limits.MaxTokens = 0
				c.SessionDir = ""
				return c
			},
			want: []string{
				"max_tokens: UNBOUNDED (no token budget)",
				"session_dir: none (no audit log)",
				"limits.max_tokens is 0: no token budget bounds this run",
				"session_dir is not set: no session log (audit trail) will be written",
			},
		},
		{
			name: "schema present with default retries",
			cfg: func() *config.Config {
				c := baseCfg()
				c.Output.Schema = map[string]any{"type": "object"}
				return c
			},
			want: []string{
				"schema: present (stdout carries only schema-valid JSON)",
				"max_schema_retries: 2 (default)",
			},
		},
		{
			name: "schema present with explicit retries",
			cfg: func() *config.Config {
				c := baseCfg()
				c.Output.Schema = map[string]any{"type": "object"}
				c.Output.MaxSchemaRetries = 5
				return c
			},
			want: []string{"max_schema_retries: 5\n"},
			// Scoped to this row: other rows legitimately name their own
			// defaults (the retry policy, the tool-call parallelism), so a bare
			// "(default)" would no longer be about output.max_schema_retries.
			notWant: []string{"max_schema_retries: 2 (default)"},
		},
		{
			// The dry run must answer "can this config run twice at once?",
			// because the answer decides whether a cron line is safe.
			name: "run lock disabled by default",
			cfg:  baseCfg,
			want: []string{"CONCURRENCY\n  lock: disabled (concurrent runs of this config are allowed)\n"},
		},
		{
			name: "run lock enabled",
			cfg: func() *config.Config {
				c := baseCfg()
				c.Lock = true
				return c
			},
			want:    []string{"CONCURRENCY\n  lock: enabled (a run started while another holds <config>.lock exits 7)\n"},
			notWant: []string{"lock: disabled"},
		},
		{
			// The second concurrency question: two tool calls inside ONE turn.
			// An omitted tools.parallel means true, which is exactly the fact
			// an operator cannot read off the YAML file.
			name: "parallel tool calls by default",
			cfg:  baseCfg,
			want: []string{"  tool calls in a turn: parallel (default)\n"},
		},
		{
			name: "parallel tool calls turned off",
			cfg: func() *config.Config {
				c := baseCfg()
				c.Tools.Parallel = ptrBool(false)
				return c
			},
			want:    []string{"  tool calls in a turn: sequential (tools.parallel: false)\n"},
			notWant: []string{"parallel (default)"},
		},
		{
			// An explicit true is not a default, and saying so would misreport
			// where the value came from.
			name: "parallel tool calls asked for explicitly",
			cfg: func() *config.Config {
				c := baseCfg()
				c.Tools.Parallel = ptrBool(true)
				return c
			},
			want:    []string{"  tool calls in a turn: parallel (tools.parallel: true)\n"},
			notWant: []string{"parallel (default)"},
		},
		{
			// max_attempts: 0 is spelled "omitted" in RetryConfig, so an
			// operator who typed the 0 on purpose has no other way to learn
			// that the client still tries three times.
			name: "retry policy defaults are named",
			cfg:  baseCfg,
			want: []string{"  retry:           3 attempts (default), 1s initial backoff (default)\n"},
		},
		{
			name: "retry policy configured",
			cfg: func() *config.Config {
				c := baseCfg()
				c.Provider.Retry = &config.RetryConfig{
					MaxAttempts:    5,
					InitialBackoff: config.Duration(2 * time.Second),
				}
				return c
			},
			want:    []string{"  retry:           5 attempts, 2s initial backoff\n"},
			notWant: []string{"attempts (default)", "backoff (default)"},
		},
		{
			// Half-configured must not read as fully defaulted: the annotation
			// belongs to each number, not to the row.
			name: "retry policy half configured",
			cfg: func() *config.Config {
				c := baseCfg()
				c.Provider.Retry = &config.RetryConfig{MaxAttempts: 7}
				return c
			},
			want: []string{"  retry:           7 attempts, 1s initial backoff (default)\n"},
		},
		{
			// Reachable since explain reports on invalid configs too; the
			// field must read as unset rather than as trailing whitespace.
			name: "missing model reads as unset",
			cfg: func() *config.Config {
				c := baseCfg()
				c.Model = ""
				return c
			},
			want:    []string{"  model:           (unset)\n"},
			notWant: []string{"  model:           \n"},
		},
		{
			name: "subprocess tools listed with argv and timeouts",
			cfg: func() *config.Config {
				c := baseCfg()
				c.Tools.Subprocess = []config.SubprocessTool{
					{Name: "grep_logs", Description: "d", Command: []string{"grep", "-i", "error"}, Timeout: config.Duration(30_000_000_000)},
					{Name: "notify", Description: "d", Command: []string{"notify-send"}, AllowArgs: true},
				}
				return c
			},
			regTools: []string{"fs_read", "fs_write", "fs_list", "grep_logs", "notify"},
			want: []string{
				"- \"grep_logs\": exec [\"grep\" \"-i\" \"error\"] (timeout 30s, model-supplied args: no)",
				"- \"notify\": exec [\"notify-send\"] (timeout default (60s), model-supplied args: YES)",
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			names := tc.regTools
			if names == nil {
				names = fsBuiltins
			}
			got := Render(tc.cfg(), registryWith(t, names...), nil, nil, alwaysFound, nil)
			for _, want := range tc.want {
				if !strings.Contains(got, want) {
					t.Errorf("report missing %q\nfull report:\n%s", want, got)
				}
			}
			for _, notWant := range tc.notWant {
				if strings.Contains(got, notWant) {
					t.Errorf("report must not contain %q\nfull report:\n%s", notWant, got)
				}
			}
		})
	}
}

// TestRenderMarksOverrides: with --set in play, the report must say which
// numbers came from the command line rather than from the file under review.
// Without the marker the dry run would be actively misleading - an operator
// auditing a cron line would read a value the YAML does not contain and
// attribute it to the YAML.
func TestRenderMarksOverrides(t *testing.T) {
	cfg := baseCfg()
	cfg.Model = "override-model"
	cfg.Workspace = "/tmp/elsewhere"
	cfg.SessionDir = ""
	cfg.Lock = true
	cfg.Limits = config.Limits{MaxTurns: 3, MaxTokens: 99, Timeout: config.Duration(90_000_000_000)}
	cfg.Output.Schema = map[string]any{"type": "object"}
	cfg.Output.MaxSchemaRetries = 4
	// lock is deliberately absent: it left the --set allowlist on 2026-08-12,
	// so no report line can carry its marker any more (config.SettableKeys).
	pairs := []string{
		"model=override-model", "workspace=/tmp/elsewhere", "session_dir=",
		"limits.max_turns=3", "limits.max_tokens=99", "limits.timeout=90s",
		"output.max_schema_retries=4", "prompt=summarize {{input}}",
		"system_prompt_file=/tmp/p.txt",
	}

	got := Render(cfg, registryWith(t, fsBuiltins...), pairs, nil, alwaysFound, nil)

	// Every override is echoed verbatim at the top - including the two
	// (prompt, system_prompt_file) that no report line below can carry.
	want := []string{
		"OVERRIDES\n",
		"--set model=\"override-model\"",
		"--set prompt=\"summarize {{input}}\"",
		"--set system_prompt_file=\"/tmp/p.txt\"",
		"--set session_dir=\"\"",
		// ... and each affected line is marked where it is read.
		"model:           \"override-model\" (overridden via --set)",
		"workspace: \"/tmp/elsewhere\" (overridden via --set)",
		"max_turns:  3 (overridden via --set)",
		"max_tokens: 99 (overridden via --set)",
		"timeout:    1m30s (overridden via --set)",
		"max_schema_retries: 4 (overridden via --set)",
		"session_dir: none (no audit log) (overridden via --set)",
	}
	for _, w := range want {
		if !strings.Contains(got, w) {
			t.Errorf("report missing %q\nfull report:\n%s", w, got)
		}
	}
}

// TestRenderWithoutOverridesHasNoSection: the section and the markers appear
// only when something was overridden, so the report of a plain
// `amele explain agent.yaml` is unchanged by this feature.
func TestRenderWithoutOverridesHasNoSection(t *testing.T) {
	got := Render(baseCfg(), registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil)
	for _, notWant := range []string{"OVERRIDES", "overridden via --set"} {
		if strings.Contains(got, notWant) {
			t.Errorf("report mentions %q with no overrides:\n%s", notWant, got)
		}
	}
}

// TestRenderIgnoresMalformedOverride: Render is given the pairs that
// ApplyOverrides already accepted, so a pair with no "=" cannot occur - but a
// report is not the place to panic if one ever does.
func TestRenderIgnoresMalformedOverride(t *testing.T) {
	got := Render(baseCfg(), registryWith(t, fsBuiltins...), []string{"model"}, nil, alwaysFound, nil)
	if strings.Contains(got, "overridden via --set") {
		t.Errorf("malformed pair marked a line:\n%s", got)
	}
}

// TestRenderNeverPrintsSecrets pins the SECURITY contract on its own: the API
// key and interpolated values have no line in the report at all.
func TestRenderNeverPrintsSecrets(t *testing.T) {
	cfg := baseCfg()
	cfg.Provider.APIKey = "sk-live-DO-NOT-PRINT"
	got := Render(cfg, registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil)
	if strings.Contains(got, "sk-live-DO-NOT-PRINT") {
		t.Fatalf("API key leaked into report:\n%s", got)
	}
	if strings.Contains(strings.ToLower(got), "api_key") {
		t.Fatalf("report mentions api_key field (must be omitted entirely):\n%s", got)
	}
}

// TestRenderRedactsInterpolatedSecrets is the regression test for the review
// finding: subprocess argv vectors and shell patterns are printed verbatim,
// and ${ENV_VAR} interpolation can put a secret inside either. The config
// MUST come from config.Load - a struct literal has an empty interpolated
// list, which is exactly why the original tests could not catch the leak.
func TestRenderRedactsInterpolatedSecrets(t *testing.T) {
	const secret = "tok-super-secret-value"
	yaml := `model: test-model
provider:
  base_url: https://api.example.com/v1
  api_key: ${SECRET_TOKEN}
tools:
  shell:
    enabled: true
    deny: ["curl * ${SECRET_TOKEN} *"]
  subprocess:
    - name: upload
      description: uploads a report
      command: ["curl", "-H", "Authorization: Bearer ${SECRET_TOKEN}"]
`
	dir := t.TempDir()
	path := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Load(path, func(key string) (string, bool) {
		if key == "SECRET_TOKEN" {
			return secret, true
		}
		return "", false
	})
	if err != nil {
		t.Fatal(err)
	}

	got := Render(cfg, registryWith(t, "shell", "upload"), nil, nil, alwaysFound, nil)
	if strings.Contains(got, secret) {
		t.Fatalf("interpolated secret leaked into report:\n%s", got)
	}
	// The redaction marker must appear where the secret was, in BOTH surfaces:
	// the subprocess argv and the shell deny pattern.
	for _, want := range []string{"Authorization: Bearer [REDACTED]", "curl * [REDACTED] *"} {
		if !strings.Contains(got, want) {
			t.Errorf("report missing redacted form %q\nfull report:\n%s", want, got)
		}
	}
}

// TestRedactSecretsOverlapping mirrors internal/session's regression: the
// report redactor replaced values in registration order, so a short secret
// that prefixes a longer one ("sk-" before "sk-secret-value") left the longer
// one's tail in the report.
func TestRedactSecretsOverlapping(t *testing.T) {
	const long = "sk-secret-value"
	for _, secrets := range [][]string{{"sk-", long}, {long, "sk-"}} {
		got := redactSecrets("key: "+long, secrets)
		if strings.Contains(got, "secret-value") {
			t.Errorf("secrets %v: leaked %q", secrets, got)
		}
		if want := "key: [REDACTED]"; got != want {
			t.Errorf("secrets %v: got %q, want %q", secrets, got, want)
		}
	}
}

// TestRenderRedactsQuoteEscapedSecrets is the regression test for the review
// finding: the report Go-quotes config values (field, the %q argv rows), so a
// credential carrying a quote or a backslash reaches the assembled text in its
// ESCAPED spelling and no longer equals the raw value the redactor was given.
// The reviewer reproduced the leak with exactly this value.
func TestRenderRedactsQuoteEscapedSecrets(t *testing.T) {
	const secret = `sk-"quoted"\part` //nolint:gosec // G101: deliberately fake credential, the test proves it gets redacted
	yaml := `model: test-model
provider:
  base_url: https://api.example.com/v1
  api_key: ${SEC_TOKEN}
tools:
  subprocess:
    - name: upload
      description: uploads a report
      command: ["curl", "-H", "Authorization: Bearer ${SEC_TOKEN}"]
`
	dir := t.TempDir()
	path := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Load(path, func(key string) (string, bool) {
		if key == "SEC_TOKEN" {
			return secret, true
		}
		return "", false
	})
	if err != nil {
		t.Fatal(err)
	}

	got := Render(cfg, registryWith(t, "upload"), nil, nil, alwaysFound, nil)
	// The escaped spelling is the one that actually appears in the report; the
	// raw one is checked too so the test keeps meaning if quoting ever moves.
	for _, leak := range []string{secret, `sk-\"quoted\"\\part`} {
		if strings.Contains(got, leak) {
			t.Errorf("credential leaked as %q into report:\n%s", leak, got)
		}
	}
	if !strings.Contains(got, redactedMarker) {
		t.Errorf("report carries no redaction marker at all:\n%s", got)
	}
}

// TestRedactSecretsOrdersStripBeforeReplace pins the strip-then-redact
// ordering. Redacting first left two holes: report text that carries the
// secret with a terminal-control rune wedged into it survives the by-value
// match and is then stripped back into the plain credential, and a secret
// whose own value carries such a rune stops matching once the report is
// stripped. Both are covered by matching against stripped text.
func TestRedactSecretsOrdersStripBeforeReplace(t *testing.T) {
	tests := []struct {
		name   string
		report string
		secret string
	}{
		{"control rune wedged into the report text", "key: sk-\x1bsecret-value", "sk-secret-value"},
		{"control rune inside the secret value", "key: sk-\x1bsecret-value", "sk-\x1bsecret-value"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := redactSecrets(tc.report, []string{tc.secret})
			if want := "key: " + redactedMarker; got != want {
				t.Errorf("got %q, want %q", got, want)
			}
		})
	}
}

// TestRedactSecretsIgnoresTooShortValues: by-value redaction is a blind
// substring replace, so a one-character value turns every occurrence of that
// character into "[REDACTED]" and destroys the report ("e[REDACTED]its 7").
// Such a value is not a credential worth protecting; the variable NAME must
// still be reported, because the requirements checklist is how an operator
// learns what to set.
func TestRedactSecretsIgnoresTooShortValues(t *testing.T) {
	const report = "explain exits 7\n"
	if got := redactSecrets(report, []string{"x"}); got != report {
		t.Errorf("short value mangled the report: %q", got)
	}

	yaml := `model: test-model
provider:
  base_url: https://api.example.com/v1
  api_key: ${SHORT_KEY}
`
	dir := t.TempDir()
	path := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Load(path, func(key string) (string, bool) {
		if key == "SHORT_KEY" {
			return "x", true
		}
		return "", false
	})
	if err != nil {
		t.Fatal(err)
	}
	out := requirementsReport(cfg, alwaysFound)
	if !strings.Contains(out, "SHORT_KEY") || !strings.Contains(out, "set") {
		t.Errorf("requirements dropped the short-valued variable:\n%s", out)
	}
	if strings.Contains(out, redactedMarker) {
		t.Errorf("short value still drove a replacement:\n%s", out)
	}
}

// TestRequirementsSection: the "YAML is the manifest" payoff - explain
// derives required env vars from ${VAR} references and required host
// executables from subprocess command[0], with found/missing marks.
func TestRequirementsSection(t *testing.T) {
	dir := t.TempDir()
	yaml := `model: m
provider:
  base_url: https://x/v1
  api_key: ${SET_VAR}
tools:
  subprocess:
    - name: present
      description: d
      command: ["present-tool"]
    - name: absent
      description: d
      command: ["/opt/nonexistent/tool"]
`
	path := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	// The value must not collide with the report's own vocabulary: SET_VAR
	// feeds api_key, so it is redacted by value, and a one-letter value would
	// eat every occurrence of that letter in the section headings.
	env := func(key string) (string, bool) {
		if key == "SET_VAR" {
			return "zzz-test-value-zzz", true
		}
		return "", false
	}
	cfg, err := config.Load(path, env)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	probe := func(name string) error {
		if name == "present-tool" {
			return nil
		}
		return errors.New("not found")
	}
	out := requirementsReport(cfg, probe)
	for _, want := range []string{
		"REQUIREMENTS",
		"SET_VAR", // env var name listed
		"present-tool", "found",
		"MISSING", // the absolute tool reported missing
	} {
		if !strings.Contains(out, want) {
			t.Errorf("requirements output missing %q:\n%s", want, out)
		}
	}
}

// TestSectionHeadersAreUpperCase pins the report's one structural convention:
// a line at column 0 opens a section and is upper-case, while everything a
// section says is indented. `requirements:` was the lone lower-case header, so
// it read as a stray subsection between TOOLS and PERMISSIONS instead of a
// section of its own. Checking the assembled report (not one section) is what
// makes the rule hold for sections added later.
func TestSectionHeadersAreUpperCase(t *testing.T) {
	cfg := baseCfg()
	cfg.Tools.Subprocess = []config.SubprocessTool{
		{Name: "notify", Description: "d", Command: []string{"true"}, Env: []string{"MAILTO"}},
	}
	cfg.Permissions = config.Permissions{Default: config.PolicyAsk}
	cfg.Lock = true

	report := Render(cfg, registryWith(t, "notify"), []string{"model=m"},
		[]string{"model is required (set it in the YAML or via --model)"}, alwaysFound, nil)

	var headers []string
	for _, line := range strings.Split(report, "\n") {
		if line == "" || line[0] == ' ' {
			continue
		}
		headers = append(headers, line)
		// The header may carry a parenthetical (PROBLEMS does), so only the
		// leading word is checked - that is the part readers scan for.
		word, _, _ := strings.Cut(line, " ")
		if word != strings.ToUpper(word) {
			t.Errorf("section header %q is not upper-case", line)
		}
	}
	// A guard on the guard: a report that somehow lost its headers would pass
	// the loop above vacuously.
	if len(headers) < 8 {
		t.Fatalf("expected the full report's sections, got headers %q", headers)
	}
	if !slices.Contains(headers, "REQUIREMENTS") {
		t.Errorf("headers %q do not include REQUIREMENTS", headers)
	}
}

// TestRequirementsSectionMissingEnv: LoadTolerant records undefined ${VAR}
// names; the requirements block must mark them missing (✗) rather than
// silently omitting them, so a new user knows exactly what to set.
func TestRequirementsSectionMissingEnv(t *testing.T) {
	dir := t.TempDir()
	yaml := "model: m\nprovider:\n  base_url: https://x/v1\n  api_key: ${UNSET_VAR}\n"
	path := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	none := func(string) (string, bool) { return "", false }
	cfg, err := config.LoadTolerant(path, none)
	if err != nil {
		t.Fatalf("LoadTolerant: %v", err)
	}

	out := requirementsReport(cfg, nil)
	if !strings.Contains(out, "UNSET_VAR") {
		t.Errorf("requirements output missing UNSET_VAR:\n%s", out)
	}
	if !strings.Contains(out, "✗") {
		t.Errorf("requirements output missing missing-marker ✗:\n%s", out)
	}
}

// TestRenderRequirementsRedactsInterpolatedValues is the regression test for
// the review finding: the requirements block returned its builder's output
// directly, skipping the redaction pass Render applies at its own return - so
// a config loaded via LoadTolerant with one DEFINED secret ${VAR} substituted
// into a subprocess command[0] and one UNDEFINED ${VAR} elsewhere would leak
// the defined value's substituted text straight to stdout.
func TestRenderRequirementsRedactsInterpolatedValues(t *testing.T) {
	const secret = "super-secret-command-value"
	dir := t.TempDir()
	yaml := `model: m
provider:
  base_url: https://x/v1
  api_key: ${MISSING_VAR}
tools:
  subprocess:
    - name: t
      description: d
      command: ["${TOOL_SECRET}"]
`
	path := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	env := func(key string) (string, bool) {
		if key == "TOOL_SECRET" {
			return secret, true
		}
		return "", false // MISSING_VAR stays undefined
	}
	cfg, err := config.LoadTolerant(path, env)
	if err != nil {
		t.Fatalf("LoadTolerant: %v", err)
	}

	out := requirementsReport(cfg, func(string) error { return nil })
	if strings.Contains(out, secret) {
		t.Fatalf("interpolated value leaked into requirements output:\n%s", out)
	}
	if !strings.Contains(out, redactedMarker) {
		t.Errorf("requirements output missing redaction marker:\n%s", out)
	}
}

// TestRequirementsSectionOmission pins the three omission rules from the
// binding: no env refs and no subprocess tools omits the whole section, and
// each subsection is independently omitted when its own list is empty - a
// config with neither must read exactly like it did before this feature
// existed, and a config with only one kind of requirement must not print a
// misleading empty subsection for the other.
func TestRequirementsSectionOmission(t *testing.T) {
	found := func(string) error { return nil }

	t.Run("no env refs, no subprocess tools: section omitted entirely", func(t *testing.T) {
		out := requirementsReport(baseCfg(), found)
		if out != "" {
			t.Errorf("expected empty requirements output, got:\n%s", out)
		}
		full := Render(baseCfg(), registryWith(t, fsBuiltins...), nil, nil, found, nil)
		if strings.Contains(full, "REQUIREMENTS") {
			t.Errorf("full report must not carry a REQUIREMENTS section with nothing to report:\n%s", full)
		}
	})

	t.Run("env refs, no subprocess tools: env present, executables omitted", func(t *testing.T) {
		dir := t.TempDir()
		yaml := "model: m\nprovider:\n  base_url: https://x/v1\n  api_key: ${ONLY_VAR}\n"
		path := filepath.Join(dir, "agent.yaml")
		if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
			t.Fatal(err)
		}
		// The substituted value must not collide with report vocabulary (a
		// bare "v" would land inside "env:" itself and get redacted along
		// with it), so use a value distinctive enough to only ever match
		// its own substitution site.
		cfg, err := config.Load(path, func(string) (string, bool) { return "zzz-test-value-zzz", true })
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		out := requirementsReport(cfg, found)
		if !strings.Contains(out, "env:") || !strings.Contains(out, "ONLY_VAR") {
			t.Errorf("expected an env: subsection naming ONLY_VAR:\n%s", out)
		}
		if strings.Contains(out, "executables:") {
			t.Errorf("no subprocess tools: executables: subsection must be omitted:\n%s", out)
		}
	})

	t.Run("subprocess tools, no env refs: executables present, env omitted", func(t *testing.T) {
		cfg := baseCfg()
		cfg.Tools.Subprocess = []config.SubprocessTool{
			{Name: "t", Description: "d", Command: []string{"present-tool"}},
		}
		out := requirementsReport(cfg, found)
		if !strings.Contains(out, "executables:") || !strings.Contains(out, "present-tool") {
			t.Errorf("expected an executables: subsection naming present-tool:\n%s", out)
		}
		if strings.Contains(out, "env:") {
			t.Errorf("no env refs: env: subsection must be omitted:\n%s", out)
		}
	})
}

// TestRequirementsSectionQuotesExecutables: command[0] in an untrusted pack
// can carry terminal control bytes (OSC/ESC sequences); the requirements
// report must print it Go-quoted - as the TOOLS section already does - so
// `amele explain pack/`, the recommended pre-run audit, cannot be used to
// drive the operator's terminal or forge checklist rows.
func TestRequirementsSectionQuotesExecutables(t *testing.T) {
	cfg := baseCfg()
	cfg.Tools.Subprocess = []config.SubprocessTool{{
		Name:        "evil",
		Description: "d",
		Command:     []string{"\x1b]52;c;ZXZpbA==\x07./tool"},
	}}
	out := requirementsReport(cfg, alwaysFound)
	if strings.ContainsRune(out, '\x1b') {
		t.Errorf("raw ESC byte leaked into requirements output:\n%q", out)
	}
	if !strings.Contains(out, `\x1b`) {
		t.Errorf("executable is not Go-quoted in requirements output:\n%q", out)
	}
}

// writeCfg writes YAML to a temp dir and loads it tolerantly with env, the
// shape every display-policy test needs: struct literals carry no env
// bindings, so only a real load can exercise redaction decisions.
func writeCfg(t *testing.T, yaml string, env map[string]string) *config.Config {
	t.Helper()
	path := filepath.Join(t.TempDir(), "agent.yaml")
	if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.LoadTolerant(path, func(key string) (string, bool) {
		v, ok := env[key]
		return v, ok
	})
	if err != nil {
		t.Fatalf("LoadTolerant: %v", err)
	}
	return cfg
}

// TestSecretVarName is the display rule's core table: a variable NAME decides
// whether its value may be shown. Over-matching ("MONKEY" contains "key") is
// deliberate - the rule fails closed.
func TestSecretVarName(t *testing.T) {
	cases := []struct {
		name string
		want bool
	}{
		{"MODEL", false},
		{"BASE_URL", false},
		{"TZ", false},
		{"LOG_DIR", false},
		{"API_KEY", true},
		{"key", true},
		{"GH_TOKEN", true},
		{"MY_SECRET", true},
		{"DB_PASSWORD", true},
		{"passwd", true},
		{"AWS_CREDENTIALS", true},
		{"MONKEY", true}, // over-matches on purpose: fail closed
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := secretVarName(tc.name); got != tc.want {
				t.Errorf("secretVarName(%q) = %v, want %v", tc.name, got, tc.want)
			}
		})
	}
}

// TestRenderShowsNonSecretInterpolatedValues pins the display rule end to end:
// a pack that parameterises model/base_url/TZ through the environment must be
// pre-flightable - the operator has to SEE which model the cron line will buy
// - while credentials stay redacted, whether they are named like one
// (${GH_TOKEN}) or merely used as one (${OPENROUTER} feeding api_key).
func TestRenderShowsNonSecretInterpolatedValues(t *testing.T) {
	yaml := `model: ${MODEL}
provider:
  base_url: ${BASE_URL}
  api_key: ${OPENROUTER}
workspace: /
tools:
  shell:
    enabled: true
    deny: ["curl * ${GH_TOKEN} *"]
  subprocess:
    - name: upload
      description: d
      command: ["curl", "-H", "Authorization: Bearer ${OPENROUTER}", "--tz", "${TZ}"]
`
	cfg := writeCfg(t, yaml, map[string]string{
		"MODEL":      "gpt-4o-mini",
		"BASE_URL":   "https://api.example.com/v1",
		"OPENROUTER": "sk-live-xyz",
		"GH_TOKEN":   "ghp-abc",
		"TZ":         "Europe/Istanbul",
	})

	got := Render(cfg, registryWith(t, "shell", "upload"), nil, nil, alwaysFound, nil)

	for _, want := range []string{
		"model:           \"gpt-4o-mini\"",
		"base_url:        \"https://api.example.com/v1\"",
		"Europe/Istanbul",
		"Authorization: Bearer [REDACTED]",
		"curl * [REDACTED] *",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("report missing %q\nfull report:\n%s", want, got)
		}
	}
	for _, leak := range []string{"sk-live-xyz", "ghp-abc"} {
		if strings.Contains(got, leak) {
			t.Fatalf("secret %q leaked into report:\n%s", leak, got)
		}
	}
}

// TestRequirementsMarksMissingEnvLoudly: `explain` no longer refuses a config
// with unset variables, so the requirements block is the only thing telling
// the operator the run cannot work yet - the mark must be unmissable.
func TestRequirementsMarksMissingEnvLoudly(t *testing.T) {
	cfg := writeCfg(t, "model: m\nprovider:\n  base_url: https://x/v1\n  api_key: ${PACK_KEY}\n", nil)

	out := requirementsReport(cfg, alwaysFound)
	if !strings.Contains(out, "PACK_KEY") || !strings.Contains(out, "✗ MISSING") {
		t.Errorf("requirements block does not mark PACK_KEY as MISSING:\n%s", out)
	}
}

// TestRequirementsListsToolEnvAllowlists: the env allowlist is a capability
// grant (which of amele's own variables a tool's process can read), so the
// pre-run audit must show it. Names only - the values are the point of the
// allowlist.
func TestRequirementsListsToolEnvAllowlists(t *testing.T) {
	cfg := baseCfg()
	cfg.Tools.Shell = config.ShellConfig{Enabled: true, Env: []string{"TZ", "GH_TOKEN"}}
	cfg.Tools.Subprocess = []config.SubprocessTool{
		{Name: "upload", Description: "d", Command: []string{"curl"}, Env: []string{"API_BASE"}},
		{Name: "plain", Description: "d", Command: []string{"date"}},
	}

	out := requirementsReport(cfg, alwaysFound)
	for _, want := range []string{
		"env allowlists",
		"shell",
		`"TZ", "GH_TOKEN"`,
		"upload",
		`"API_BASE"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("requirements block missing %q:\n%s", want, out)
		}
	}
	// A tool without an allowlist has nothing to list; inventing a row for it
	// would read as "this tool sees nothing", the opposite of the truth.
	if strings.Contains(out, "plain") {
		t.Errorf("tool without an env allowlist must not get a row:\n%s", out)
	}
}

// TestRequirementsOmitsEnvAllowlistsWhenNoneDeclared keeps the section's
// "silence when there is nothing to say" rule.
func TestRequirementsOmitsEnvAllowlistsWhenNoneDeclared(t *testing.T) {
	cfg := baseCfg()
	cfg.Tools.Subprocess = []config.SubprocessTool{{Name: "t", Description: "d", Command: []string{"date"}}}
	if out := requirementsReport(cfg, alwaysFound); strings.Contains(out, "env allowlists") {
		t.Errorf("no allowlists declared, subsection must be omitted:\n%s", out)
	}
}

// TestRenderProblemsSection: explain reports, run gates - so a config that
// cannot run is still fully described, with its problems stated at the top
// where they cannot be missed.
func TestRenderProblemsSection(t *testing.T) {
	problems := []string{
		"workspace \"/nope\" is not accessible",
		"undefined environment variable(s): PACK_KEY",
	}
	got := Render(baseCfg(), registryWith(t, fsBuiltins...), nil, problems, alwaysFound, nil)

	if !strings.HasPrefix(got, "PROBLEMS") {
		t.Errorf("PROBLEMS must open the report:\n%s", got)
	}
	for _, p := range problems {
		if !strings.Contains(got, "  - "+p+"\n") {
			t.Errorf("report missing problem %q:\n%s", p, got)
		}
	}
	// The rest of the report is still there: that is the whole point.
	for _, section := range []string{"MODEL & PROVIDER", "TOOLS", "BUDGETS", "WARNINGS"} {
		if !strings.Contains(got, section) {
			t.Errorf("report missing section %q:\n%s", section, got)
		}
	}
	if clean := Render(baseCfg(), registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil); strings.Contains(clean, "PROBLEMS") {
		t.Errorf("a clean config must not grow a PROBLEMS section:\n%s", clean)
	}
}

// TestRenderWithoutRegistry: a config whose tool registry could not be built
// is exactly the config an operator most needs described, so Render must
// accept a nil registry - and must not then accuse every permission entry of
// being a typo, since it has nothing to check them against.
func TestRenderWithoutRegistry(t *testing.T) {
	cfg := baseCfg()
	cfg.Permissions = config.Permissions{Tools: map[string]config.Policy{"fs_write": config.PolicyAsk}}

	got := Render(cfg, nil, nil, []string{"initializing fs tools: no such directory"}, alwaysFound, nil)
	if !strings.Contains(got, "WARNINGS") {
		t.Errorf("report truncated without a registry:\n%s", got)
	}
	if strings.Contains(got, "matches no tool") {
		t.Errorf("no registry means no typo verdict:\n%s", got)
	}
}

// TestRenderStripsTerminalControls: since explain became a report, it also
// renders configs Validate rejected - including a subprocess tool whose name
// is invalid precisely because it carries an escape sequence. That name is
// printed unquoted in TOOLS and PERMISSIONS, so `amele explain evil-pack/`,
// the recommended audit of an untrusted pack, must not hand the pack the
// operator's terminal.
func TestRenderStripsTerminalControls(t *testing.T) {
	const spoof = "ok\x1b[2K\rfs_read"
	cfg := baseCfg()
	cfg.Tools.Subprocess = []config.SubprocessTool{{
		Name:        spoof,
		Description: "d",
		Command:     []string{"date"},
		Env:         []string{"TZ"},
	}}
	cfg.Permissions = config.Permissions{Tools: map[string]config.Policy{spoof: config.PolicyAllow}}

	got := Render(cfg, registryWith(t, fsBuiltins...), nil, []string{"a problem\u009b2K"}, alwaysFound, nil)
	for _, r := range []rune{'\x1b', '\r', '\u009b', '\u202e'} {
		if strings.ContainsRune(got, r) {
			t.Errorf("control rune %q survived into the report:\n%q", r, got)
		}
	}
}

// TestRenderCannotForgeRowsWithNewlines is the regression test for the review
// finding: a tool name or permission key containing a NEWLINE forges whole
// rows in the report. Both fields violate their validation rules when they
// carry one - but explain reports on configs Validate rejected, so the rules
// no longer stand between an untrusted pack and this renderer, and
// stripTerminalControls deliberately keeps "\n" (the report is line-oriented)
// so it cannot catch this. The fixture below invented a "shell: disabled"
// line for a config whose shell is ENABLED, and a second subprocess tool that
// does not exist.
func TestRenderCannotForgeRowsWithNewlines(t *testing.T) {
	const forgedTool = "harmless\n  shell: disabled\n  subprocess tools:\n    - realname"
	const forgedPerm = "ok\n    fs_write: allow"
	cfg := baseCfg()
	cfg.Tools.Shell = config.ShellConfig{Enabled: true, Allow: []string{"date*"}}
	cfg.Tools.Subprocess = []config.SubprocessTool{{Name: forgedTool, Description: "d", Command: []string{"date"}}}
	cfg.Permissions = config.Permissions{Tools: map[string]config.Policy{forgedPerm: config.PolicyDeny}}

	got := Render(cfg, registryWith(t, "shell"), nil, nil, alwaysFound, nil)

	// Every forged row is a line the config does not contain.
	for _, forged := range []string{
		"\n  shell: disabled",
		"\n  subprocess tools:\n    - realname",
		"\n    fs_write: allow",
	} {
		if strings.Contains(got, forged) {
			t.Errorf("forged row %q made it into the report:\n%s", forged, got)
		}
	}
	// ... because both fields are rendered escaped, on one line each.
	if !strings.Contains(got, `harmless\n`) {
		t.Errorf("tool name is not rendered escaped:\n%s", got)
	}
	if !strings.Contains(got, `ok\n`) {
		t.Errorf("permission key is not rendered escaped:\n%s", got)
	}
}

// TestRenderCannotForgeRowsWithConfigValues is the sibling of
// TestRenderCannotForgeRowsWithNewlines for the rest of the class: quoting the
// two named sites (tool name, permission key) left every other
// config-sourced STRING rendered raw, and a newline in any of them forges
// rows just as well. `model: "m\n  base_url: https://evil.example/v1"` invented
// a base_url row; a newline-bearing workspace invented a duplicate fs-builtins
// row; a crafted problem invented a second problem. None of these fields
// survives validation with a newline in it, which is exactly why explain -
// the one command that renders configs Validate rejected - must not trust
// them.
func TestRenderCannotForgeRowsWithConfigValues(t *testing.T) {
	cfg := baseCfg()
	cfg.Model = "m\n  base_url: https://evil.example/v1"
	cfg.Provider.Type = "openai\n  max_output_tokens: 1"
	cfg.Provider.BaseURL = "https://x/v1\n  request_timeout: 999s"
	cfg.Workspace = "/ws\n  fs builtins (fs_read, fs_write, fs_list): disabled"
	cfg.SessionDir = "/s\n  lock: enabled (a run started while another holds <config>.lock exits 7)"
	cfg.Permissions = config.Permissions{
		Default: config.Policy("allow\n  per-tool overrides: (none)"),
		Tools:   map[string]config.Policy{"fs_read": config.Policy("ask\n    fs_write: allow")},
	}
	problems := []string{"cannot run\n  - forged problem"}

	got := Render(cfg, registryWith(t, fsBuiltins...), nil, problems, alwaysFound, nil)

	for _, forged := range []string{
		"\n  base_url: https://evil.example/v1",
		"\n  max_output_tokens: 1",
		"\n  request_timeout: 999s",
		"\n  fs builtins (fs_read, fs_write, fs_list): disabled",
		"\n  lock: enabled (a run started while another holds <config>.lock exits 7)",
		"\n  per-tool overrides: (none)",
		"\n    fs_write: allow",
		"\n  - forged problem",
	} {
		if strings.Contains(got, forged) {
			t.Errorf("forged row %q made it into the report:\n%s", forged, got)
		}
	}
	// Each labelled row appears exactly once: no config value can grow the
	// report a second one.
	for _, label := range []string{"\n  model:", "\n  base_url:", "\n  workspace:", "\n  session_dir:", "\n  default policy:"} {
		if n := strings.Count(got, label); n != 1 {
			t.Errorf("row %q appears %d times, want 1:\n%s", label, n, got)
		}
	}
	// The PROBLEMS block stays one bullet long.
	if n := strings.Count(got, "\n  - "); n != 1 {
		t.Errorf("PROBLEMS bullet count = %d, want 1:\n%s", n, got)
	}
}

// ptrInt returns a pointer to n, for limits.max_logged_field where nil (use
// the default), 0 (unbounded) and n are three different answers.
func ptrInt(n int) *int { return &n }

// TestRenderMaxLoggedField: the clip bound is the one settable key whose three
// states cannot be read off the YAML - an omitted key means 8192, and a 0
// means the opposite of what a budget line usually means (no bound at all).
// That is precisely what a dry run exists to say out loud.
func TestRenderMaxLoggedField(t *testing.T) {
	cases := []struct {
		name    string
		value   *int
		want    string
		notWant string
	}{
		{
			name: "unset names the default",
			want: "  max_logged_field: 8192 (default)\n",
		},
		{
			name:    "explicit zero is unbounded",
			value:   ptrInt(0),
			want:    "  max_logged_field: UNBOUNDED (every logged field written whole)\n",
			notWant: "8192",
		},
		{
			name:    "custom bound is stated bare",
			value:   ptrInt(256),
			want:    "  max_logged_field: 256\n",
			notWant: "(default)",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := baseCfg()
			cfg.Limits.MaxLoggedField = tc.value
			got := Render(cfg, registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil)
			if !strings.Contains(got, tc.want) {
				t.Errorf("report missing %q:\n%s", tc.want, got)
			}
			if tc.notWant != "" && strings.Contains(got, "max_logged_field: "+tc.notWant) {
				t.Errorf("report still says %q:\n%s", tc.notWant, got)
			}
		})
	}
}

// TestExplainDefaultMatchesSessionDefault pins the number this package prints
// for an omitted limits.max_logged_field to the bound the writer actually
// applies. The constant is duplicated (session does not export it), so the two
// are held together by behavior instead: a writer opened with zero options
// clips at exactly the number the report names. A drift here would make the
// dry run lie about the log it is describing.
func TestExplainDefaultMatchesSessionDefault(t *testing.T) {
	w, err := session.New(t.TempDir(), session.Options{})
	if err != nil {
		t.Fatal(err)
	}
	w.RunStart("m", strings.Repeat("t", defaultMaxLoggedField*2))

	data, err := os.ReadFile(w.Path())
	if err != nil {
		t.Fatal(err)
	}
	var e struct {
		Task string `json:"task"`
	}
	if err := json.Unmarshal([]byte(strings.SplitN(strings.TrimSpace(string(data)), "\n", 2)[0]), &e); err != nil {
		t.Fatal(err)
	}
	if got := len(strings.TrimSuffix(e.Task, "...[clipped]")); got != defaultMaxLoggedField {
		t.Errorf("writer clips at %d bytes, explain reports %d", got, defaultMaxLoggedField)
	}
}

// TestRenderMaxLoggedFieldOverrideMarked: the key is on the --set allowlist,
// so its line must carry the same provenance marker every other settable
// budget line carries.
func TestRenderMaxLoggedFieldOverrideMarked(t *testing.T) {
	cfg := baseCfg()
	cfg.Limits.MaxLoggedField = ptrInt(64)
	got := Render(cfg, registryWith(t, fsBuiltins...), []string{"limits.max_logged_field=64"}, nil, alwaysFound, nil)
	if want := "  max_logged_field: 64 (overridden via --set)\n"; !strings.Contains(got, want) {
		t.Errorf("report missing %q:\n%s", want, got)
	}
}

// TestRenderMaxToolResultBytes pins both directions of the guard's BUDGETS
// row. It belongs under BUDGETS rather than beside max_logged_field because it
// bounds what the RUN spends - bytes the model reads, and therefore context and
// tokens - not what lands on disk. Both directions are printed because the
// default is not "no cap": a reviewer pre-flighting someone else's pack must be
// able to read the built-in caps off the report without opening the source.
func TestRenderMaxToolResultBytes(t *testing.T) {
	const defaultRow = "  max_tool_result_bytes: built-in per-tool caps " +
		"(fs_read 256 KiB, fs_list/subprocess/shell 64 KiB per stream, mcp 64 KiB)\n"

	cases := []struct {
		name string
		cap  *int
		want string
	}{
		{"omitted names the built-in caps", nil, defaultRow},
		{
			"a set cap names its reach",
			ptrInt(8192),
			"  max_tool_result_bytes: 8192 (every tool family, and the framed result)\n",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := baseCfg()
			cfg.Limits.MaxToolResultBytes = tc.cap
			got := Render(cfg, registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil)
			if !strings.Contains(got, tc.want) {
				t.Errorf("report missing %q:\n%s", tc.want, got)
			}
		})
	}
}

// TestRenderMaxToolResultBytesOverrideMarked: the key is on the --set
// allowlist, so its line must carry the same provenance marker every other
// settable budget line carries - the report's job is to say where each number
// came from, and a cap raised on the cron line is exactly the number a
// reviewer would otherwise attribute to the audited YAML.
func TestRenderMaxToolResultBytesOverrideMarked(t *testing.T) {
	cfg := baseCfg()
	cfg.Limits.MaxToolResultBytes = ptrInt(2048)
	got := Render(cfg, registryWith(t, fsBuiltins...), []string{"limits.max_tool_result_bytes=2048"}, nil, alwaysFound, nil)
	want := "  max_tool_result_bytes: 2048 (every tool family, and the framed result) (overridden via --set)\n"
	if !strings.Contains(got, want) {
		t.Errorf("report missing %q:\n%s", want, got)
	}
}

// TestRenderPromptCacheRow pins the one row that answers "will this run be
// billed for the whole prompt every turn". All three answers are here because
// they are three DIFFERENT facts, not one fact with a value: the anthropic wire
// is configured (and says where the breakpoints go), the same wire switched off
// names the key that did it, and every other wire caches without asking - a
// row that said "off" there would be a lie about the endpoint.
//
// Unit-tested rather than golden-only for the direction the fixtures cannot
// show: no golden config sets the key, so the "disabled" row would otherwise
// never be rendered by the suite at all.
func TestRenderPromptCacheRow(t *testing.T) {
	const anthropicOn = "  prompt cache:    anthropic cache_control on tools, system and the last message (up to 3 breakpoints)\n"
	const anthropicOff = "  prompt cache:    disabled (provider.prompt_cache: false)\n"
	const automatic = "  prompt cache:    automatic on this wire (reported in the session log when the endpoint says so)\n"

	cases := []struct {
		name    string
		mutate  func(*config.Config)
		want    string
		notWant []string
	}{
		{
			name: "anthropic with the key omitted is on",
			mutate: func(c *config.Config) {
				c.Provider.Type = config.ProviderTypeAnthropic
				c.Provider.BaseURL = ""
			},
			want:    anthropicOn,
			notWant: []string{anthropicOff, automatic},
		},
		{
			name: "anthropic with the key true is the same row",
			mutate: func(c *config.Config) {
				c.Provider.Type = config.ProviderTypeAnthropic
				c.Provider.BaseURL = ""
				c.Provider.PromptCache = ptrBool(true)
			},
			want:    anthropicOn,
			notWant: []string{anthropicOff, automatic},
		},
		{
			name: "anthropic with the key false names the key",
			mutate: func(c *config.Config) {
				c.Provider.Type = config.ProviderTypeAnthropic
				c.Provider.BaseURL = ""
				c.Provider.PromptCache = ptrBool(false)
			},
			want:    anthropicOff,
			notWant: []string{anthropicOn, automatic},
		},
		{
			name:    "the openai wire caches on its own",
			mutate:  func(*config.Config) {},
			want:    automatic,
			notWant: []string{anthropicOn, anthropicOff},
		},
		{
			name: "so does the gemini wire",
			mutate: func(c *config.Config) {
				c.Provider.Type = config.ProviderTypeGemini
				c.Provider.BaseURL = ""
			},
			want:    automatic,
			notWant: []string{anthropicOn, anthropicOff},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := baseCfg()
			tc.mutate(cfg)
			got := Render(cfg, registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil)
			if !strings.Contains(got, tc.want) {
				t.Errorf("report missing %q:\n%s", tc.want, got)
			}
			for _, notWant := range tc.notWant {
				if strings.Contains(got, notWant) {
					t.Errorf("report unexpectedly contains %q:\n%s", notWant, got)
				}
			}
		})
	}
}

// TestRenderDataGovernanceKeys: log_reasoning and print_session_path are the
// two keys --set cannot reach, which makes this report the only place an
// operator pre-flighting someone else's pack can learn that the run writes the
// model's scratchpad to disk. Both directions are pinned: enabled prints a row
// (and, for log_reasoning, a warning), disabled prints nothing at all.
//
// No golden fixture enables either key, so the rows are tested here rather
// than by widening eleven goldens that would then only prove the default.
func TestRenderDataGovernanceKeys(t *testing.T) {
	const reasoningRow = "  log_reasoning: enabled"
	const pathRow = "  print_session_path: enabled"
	const reasoningWarn = "  - log_reasoning is enabled: the session log will contain the model's reasoning"

	cases := []struct {
		name    string
		mutate  func(*config.Config)
		want    []string
		notWant []string
	}{
		{
			name:    "off by default: no rows, no warning",
			mutate:  func(*config.Config) {},
			notWant: []string{reasoningRow, pathRow, "log_reasoning", "print_session_path"},
		},
		{
			name:   "log_reasoning says what it means and warns",
			mutate: func(c *config.Config) { c.LogReasoning = true },
			want: []string{
				reasoningRow + " (the provider's reasoning payload - the model's own " +
					"scratchpad - is written into the log)\n",
				reasoningWarn + "; redaction matches secret values and cannot catch a paraphrased one\n",
			},
			notWant: []string{pathRow},
		},
		{
			name:    "print_session_path is reported, and warns about nothing",
			mutate:  func(c *config.Config) { c.PrintSessionPath = true },
			want:    []string{pathRow + " (the run names its session file on stderr)\n"},
			notWant: []string{reasoningRow, reasoningWarn},
		},
		{
			// Both keys are inert without a log: cmd/amele consults them on
			// the writer it opens, and there is none. A row here would promise
			// something the run does not do - and the warning would cry wolf.
			name: "inert without a session log",
			mutate: func(c *config.Config) {
				c.SessionDir = ""
				c.LogReasoning = true
				c.PrintSessionPath = true
			},
			notWant: []string{reasoningRow, pathRow, reasoningWarn},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := baseCfg()
			tc.mutate(cfg)
			got := Render(cfg, registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil)
			for _, want := range tc.want {
				if !strings.Contains(got, want) {
					t.Errorf("report missing %q:\n%s", want, got)
				}
			}
			for _, notWant := range tc.notWant {
				if strings.Contains(got, notWant) {
					t.Errorf("report unexpectedly contains %q:\n%s", notWant, got)
				}
			}
		})
	}
}
