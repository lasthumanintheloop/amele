package config

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"
)

// overrideBase is the loaded-and-defaulted config every override case starts
// from: the shape config.Load hands over, so a test overrides the same values
// the CLI does.
func overrideBase() *Config {
	return &Config{
		Model:      "config-model",
		Workspace:  "/from-config",
		SessionDir: "/from-config/sessions",
		Limits:     Limits{MaxTurns: 20},
	}
}

// TestApplyOverridesAllowedKeys walks the allowlist: every settable key must
// round-trip from the command line into the field it names. A key that
// silently does nothing would be the worst possible failure - the run would
// proceed with the config's value while the operator believes otherwise.
// The four provider tuning keys live in TestApplyOverridesProviderTuningKeys,
// which covers the same ground for them.
func TestApplyOverridesAllowedKeys(t *testing.T) {
	baseDir := t.TempDir()

	tests := []struct {
		name  string
		pair  string
		check func(t *testing.T, c *Config)
	}{
		{"model", "model=gpt-4o-mini", func(t *testing.T, c *Config) {
			if c.Model != "gpt-4o-mini" {
				t.Errorf("model = %q", c.Model)
			}
		}},
		{"prompt", "prompt={{args}} over {{input}}", func(t *testing.T, c *Config) {
			if c.Prompt != "{{args}} over {{input}}" {
				t.Errorf("prompt = %q", c.Prompt)
			}
		}},
		{"workspace relative resolves against baseDir", "workspace=ws", func(t *testing.T, c *Config) {
			if want := filepath.Join(baseDir, "ws"); c.Workspace != want {
				t.Errorf("workspace = %q, want %q", c.Workspace, want)
			}
		}},
		{"workspace absolute kept", "workspace=/abs/ws", func(t *testing.T, c *Config) {
			if c.Workspace != "/abs/ws" {
				t.Errorf("workspace = %q", c.Workspace)
			}
		}},
		{"session_dir relative resolves against baseDir", "session_dir=logs", func(t *testing.T, c *Config) {
			if want := filepath.Join(baseDir, "logs"); c.SessionDir != want {
				t.Errorf("session_dir = %q, want %q", c.SessionDir, want)
			}
		}},
		{"session_dir empty disables logging", "session_dir=", func(t *testing.T, c *Config) {
			if c.SessionDir != "" {
				t.Errorf("session_dir = %q, want empty (logging disabled)", c.SessionDir)
			}
		}},
		{"limits.max_turns", "limits.max_turns=7", func(t *testing.T, c *Config) {
			if c.Limits.MaxTurns != 7 {
				t.Errorf("max_turns = %d", c.Limits.MaxTurns)
			}
		}},
		{"limits.max_tokens", "limits.max_tokens=12345", func(t *testing.T, c *Config) {
			if c.Limits.MaxTokens != 12345 {
				t.Errorf("max_tokens = %d", c.Limits.MaxTokens)
			}
		}},
		{"limits.timeout", "limits.timeout=90s", func(t *testing.T, c *Config) {
			if c.Limits.Timeout.Std() != 90*time.Second {
				t.Errorf("timeout = %s", c.Limits.Timeout.Std())
			}
		}},
		{"output.max_schema_retries", "output.max_schema_retries=5", func(t *testing.T, c *Config) {
			if c.Output.MaxSchemaRetries != 5 {
				t.Errorf("max_schema_retries = %d", c.Output.MaxSchemaRetries)
			}
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := overrideBase()
			if err := ApplyOverrides(cfg, []string{tt.pair}, baseDir); err != nil {
				t.Fatalf("ApplyOverrides(%q): %v", tt.pair, err)
			}
			tt.check(t, cfg)
		})
	}
}

// TestSettableKeysCoversAllowlist pins the exported list against the keys
// ApplyOverrides actually accepts: the list is what the error message and the
// documentation advertise, so a key added to one and not the other would send
// operators to a flag that does not exist.
func TestSettableKeysCoversAllowlist(t *testing.T) {
	want := []string{
		"limits.max_tokens", "limits.max_turns", "limits.timeout", "model",
		"output.max_schema_retries", "prompt", "provider.max_output_tokens",
		"provider.reasoning.effort", "provider.temperature", "provider.top_p",
		"session_dir", "system_prompt_file", "workspace",
	}
	got := SettableKeys()
	if len(got) != len(want) {
		t.Fatalf("SettableKeys() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("SettableKeys() = %v, want %v (sorted)", got, want)
		}
	}
	// The returned slice must be a copy: a caller mutating it must not be able
	// to widen or narrow the allowlist for the rest of the process.
	got[0] = "tools.fs"
	if SettableKeys()[0] != want[0] {
		t.Error("SettableKeys() exposes its backing array")
	}
}

// TestApplyOverridesRejectsExcludedKeys is the SECURITY test: the fields that
// grant capability (tools, permissions) and the provider's IDENTITY (type,
// base_url, api_key - where the run's credentials go) are deliberately NOT
// settable, so the YAML stays the audited grant of authority
// (docs/threat-model.md §2). The provider TUNING keys settable since
// 2026-08-24 do not weaken that: they change what a run spends, not what it
// may do. A typo'd key must fail just as loudly.
func TestApplyOverridesRejectsExcludedKeys(t *testing.T) {
	for _, key := range []string{
		"tools.fs", "tools.shell.enabled", "permissions.default", "permissions.tools.fs_write",
		"provider.api_key", "provider.base_url", "provider.type", "system_prompt",
		"output.schema", "limits", "max_turns", "lock", "",
		// The tuning surface is settable only where the allowlist says so:
		// the dialect reshapes every request, params writes arbitrary keys
		// into the body, and budget_tokens is legal on two targets only.
		"provider.dialect", "provider.params", "provider.reasoning",
		"provider.reasoning.budget_tokens",
	} {
		t.Run(key, func(t *testing.T) {
			cfg := overrideBase()
			err := ApplyOverrides(cfg, []string{key + "=value"}, t.TempDir())
			if err == nil {
				t.Fatalf("ApplyOverrides(%q) succeeded, want a rejection", key)
			}
			if !errors.Is(err, ErrInvalid) {
				t.Errorf("error does not wrap ErrInvalid: %v", err)
			}
			want := "cannot override \"" + key + "\" from the command line; settable keys: " +
				strings.Join(SettableKeys(), ", ")
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error = %q, want it to contain %q", err, want)
			}
		})
	}
}

// TestApplyOverridesRejectsLock is the regression for the one allowlist entry
// that was withdrawn after it shipped (2026-08-12).
//
// SECURITY: `lock` was the only settable key that could WEAKEN a run.
// `--set lock=false` disarmed the single-flight guard an audited YAML had
// armed - from the cron line, invisibly to anyone reading the file or its
// `amele explain` report - while every other settable key only retunes a run
// whose authority is already fixed. docs/threat-model.md §2 promises the
// invocation cannot widen what the YAML granted, so the key left the list.
// `lock: true` written in YAML is untouched: only the CLI override is gone.
func TestApplyOverridesRejectsLock(t *testing.T) {
	for _, pair := range []string{"lock=false", "lock=true", "lock="} {
		t.Run(pair, func(t *testing.T) {
			cfg := overrideBase()
			cfg.Lock = true
			err := ApplyOverrides(cfg, []string{pair}, t.TempDir())
			if err == nil {
				t.Fatalf("ApplyOverrides(%q) succeeded, want the same rejection every non-settable key gets", pair)
			}
			if !errors.Is(err, ErrInvalid) {
				t.Errorf("error does not wrap ErrInvalid: %v", err)
			}
			if !strings.Contains(err.Error(), `cannot override "lock"`) {
				t.Errorf("error = %q, want the excluded-key refusal naming lock", err)
			}
			if !cfg.Lock {
				t.Error("the config's own lock: true was changed by a rejected override")
			}
		})
	}
	if slices.Contains(SettableKeys(), "lock") {
		t.Error("SettableKeys() still advertises lock")
	}
}

// TestApplyOverridesMalformedPair pins the key=value shape. A bare key is
// almost always a shell quoting slip; guessing what it meant would be worse
// than saying so.
func TestApplyOverridesMalformedPair(t *testing.T) {
	cfg := overrideBase()
	err := ApplyOverrides(cfg, []string{"model"}, t.TempDir())
	if err == nil || !errors.Is(err, ErrInvalid) {
		t.Fatalf("err = %v, want an ErrInvalid rejection", err)
	}
	if !strings.Contains(err.Error(), "key=value") {
		t.Errorf("error = %q, want it to name the key=value form", err)
	}
	if cfg.Model != "config-model" {
		t.Errorf("model changed despite the error: %q", cfg.Model)
	}
}

// TestApplyOverridesSplitsOnFirstEquals: the value may itself contain "=" -
// a prompt template or a duration expression must survive intact.
func TestApplyOverridesSplitsOnFirstEquals(t *testing.T) {
	cfg := overrideBase()
	if err := ApplyOverrides(cfg, []string{"prompt=a=b=c"}, t.TempDir()); err != nil {
		t.Fatal(err)
	}
	if cfg.Prompt != "a=b=c" {
		t.Errorf("prompt = %q, want %q", cfg.Prompt, "a=b=c")
	}
}

// TestApplyOverridesBadScalars: a value the field cannot hold is a usage
// error that names the key, so an operator editing a cron line knows which
// --set to fix.
func TestApplyOverridesBadScalars(t *testing.T) {
	for _, tt := range []struct{ name, pair, wantIn string }{
		{"int", "limits.max_turns=many", "limits.max_turns"},
		{"int with unit", "limits.max_tokens=10k", "limits.max_tokens"},
		{"retries", "output.max_schema_retries=x", "output.max_schema_retries"},
		{"duration", "limits.timeout=5 minutes", "limits.timeout"},
		{"empty duration", "limits.timeout=", "limits.timeout"},
		{"empty int", "limits.max_turns=", "limits.max_turns"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := ApplyOverrides(overrideBase(), []string{tt.pair}, t.TempDir())
			if err == nil || !errors.Is(err, ErrInvalid) {
				t.Fatalf("err = %v, want an ErrInvalid rejection", err)
			}
			if !strings.Contains(err.Error(), tt.wantIn) {
				t.Errorf("error = %q, want it to name %q", err, tt.wantIn)
			}
		})
	}
}

// TestApplyOverridesRejectsEmptyPathValues: an empty workspace or prompt file
// names nothing readable. Both would fail later anyway - with a stat error
// about the empty string - so they are refused here, where the message can
// still say which --set caused it. (session_dir is the deliberate exception:
// empty there means "no session log", a real setting.)
func TestApplyOverridesRejectsEmptyPathValues(t *testing.T) {
	for _, key := range []string{"workspace", "system_prompt_file"} {
		t.Run(key, func(t *testing.T) {
			err := ApplyOverrides(overrideBase(), []string{key + "="}, t.TempDir())
			if err == nil || !errors.Is(err, ErrInvalid) {
				t.Fatalf("err = %v, want an ErrInvalid rejection", err)
			}
			if !strings.Contains(err.Error(), key) {
				t.Errorf("error = %q, want it to name %q", err, key)
			}
		})
	}
}

// TestApplyOverridesSystemPromptFileReReads: config.Load already read the
// config's own prompt file, so an override that only swapped the path would
// leave the OLD prompt in place - the model would be told one thing while the
// operator read another. The file must be re-read here.
func TestApplyOverridesSystemPromptFileReReads(t *testing.T) {
	baseDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(baseDir, "prompt.txt"), []byte("you are the override"), 0o600); err != nil {
		t.Fatal(err)
	}

	cfg := overrideBase()
	cfg.SystemPrompt = "you are the config"
	if err := ApplyOverrides(cfg, []string{"system_prompt_file=prompt.txt"}, baseDir); err != nil {
		t.Fatal(err)
	}
	if cfg.SystemPrompt != "you are the override" {
		t.Errorf("system_prompt = %q, want the overridden file's content", cfg.SystemPrompt)
	}
	if want := filepath.Join(baseDir, "prompt.txt"); cfg.SystemPromptFile != want {
		t.Errorf("system_prompt_file = %q, want the resolved path %q", cfg.SystemPromptFile, want)
	}
}

// TestApplyOverridesSystemPromptFileClearsConflict: the exclusivity rule
// exists because a config that sets both prompt forms leaves the effective
// prompt ambiguous. `--set system_prompt_file` removes the ambiguity by naming
// the winner explicitly, so it must clear the violation the same way
// `--set model=` clears "model is required" - otherwise a parametrized
// invocation could never repair a pack that shipped both fields.
func TestApplyOverridesSystemPromptFileClearsConflict(t *testing.T) {
	baseDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(baseDir, "prompt.txt"), []byte("you are the override"), 0o600); err != nil {
		t.Fatal(err)
	}

	cfg := overrideBase()
	cfg.SystemPrompt = "you are the config"
	cfg.SystemPromptFile = "other.md"
	cfg.promptConflict = true
	if err := ApplyOverrides(cfg, []string{"system_prompt_file=prompt.txt"}, baseDir); err != nil {
		t.Fatal(err)
	}
	for _, v := range cfg.Violations() {
		if strings.Contains(v, "mutually exclusive") {
			t.Errorf("violations still report the resolved conflict: %q", cfg.Violations())
		}
	}
}

// TestApplyOverridesSystemPromptFileMissing: an unreadable prompt file is a
// config error reported before anything runs, not an empty system prompt.
func TestApplyOverridesSystemPromptFileMissing(t *testing.T) {
	cfg := overrideBase()
	cfg.SystemPrompt = "you are the config"
	err := ApplyOverrides(cfg, []string{"system_prompt_file=nope.txt"}, t.TempDir())
	if err == nil || !errors.Is(err, ErrInvalid) {
		t.Fatalf("err = %v, want an ErrInvalid rejection", err)
	}
	if !strings.Contains(err.Error(), "system_prompt_file") {
		t.Errorf("error = %q, want it to name system_prompt_file", err)
	}
	if cfg.SystemPrompt != "you are the config" {
		t.Errorf("system_prompt changed despite the error: %q", cfg.SystemPrompt)
	}
}

// TestApplyOverridesLaterWins pins the documented duplicate rule: the last
// occurrence on the command line is the effective one, so a wrapper script can
// append an override that beats the one it inherited.
func TestApplyOverridesLaterWins(t *testing.T) {
	cfg := overrideBase()
	pairs := []string{"model=first", "limits.max_turns=1", "model=second", "limits.max_turns=9"}
	if err := ApplyOverrides(cfg, pairs, t.TempDir()); err != nil {
		t.Fatal(err)
	}
	if cfg.Model != "second" || cfg.Limits.MaxTurns != 9 {
		t.Errorf("model=%q max_turns=%d, want second/9", cfg.Model, cfg.Limits.MaxTurns)
	}
}

// TestApplyOverridesNoPairs: the no-override path must leave the config
// byte-identical, including the fields the resolver would otherwise touch.
func TestApplyOverridesNoPairs(t *testing.T) {
	cfg := overrideBase()
	if err := ApplyOverrides(cfg, nil, t.TempDir()); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(cfg, overrideBase()) {
		t.Errorf("config changed with no overrides: %+v", cfg)
	}
}

// TestSplitOverride pins the parser explain shares with ApplyOverrides, so the
// report can never mark a different key than the one that was applied.
func TestSplitOverride(t *testing.T) {
	for _, tt := range []struct {
		pair, key, value string
		ok               bool
	}{
		{"model=x", "model", "x", true},
		{"prompt=a=b", "prompt", "a=b", true},
		{"session_dir=", "session_dir", "", true},
		{"=value", "", "value", true},
		{"model", "", "", false},
		{"", "", "", false},
	} {
		t.Run(tt.pair, func(t *testing.T) {
			key, value, ok := SplitOverride(tt.pair)
			if key != tt.key || value != tt.value || ok != tt.ok {
				t.Errorf("SplitOverride(%q) = (%q, %q, %v), want (%q, %q, %v)",
					tt.pair, key, value, ok, tt.key, tt.value, tt.ok)
			}
		})
	}
}

// TestApplyOverridesProviderTuningKeys is TestApplyOverridesAllowedKeys for the
// four provider tuning keys: each must land in the field it names, and a
// sampling value must land as a SET pointer even when it is zero.
func TestApplyOverridesProviderTuningKeys(t *testing.T) {
	tests := []struct {
		name  string
		pair  string
		check func(c *Config) bool
	}{
		{"max_output_tokens", "provider.max_output_tokens=65536", func(c *Config) bool {
			return c.Provider.MaxOutputTokens == 65536
		}},
		{"reasoning.effort", "provider.reasoning.effort=high", func(c *Config) bool {
			return c.Provider.Reasoning != nil && c.Provider.Reasoning.Effort == "high"
		}},
		{"temperature", "provider.temperature=0.2", func(c *Config) bool {
			return hasFloat(c.Provider.Temperature, 0.2)
		}},
		// 0 is the deterministic setting, not "unset" - the pointer field
		// exists precisely so the two stay distinguishable.
		{"temperature zero is a value", "provider.temperature=0", func(c *Config) bool {
			return hasFloat(c.Provider.Temperature, 0)
		}},
		{"top_p", "provider.top_p=0.9", func(c *Config) bool {
			return hasFloat(c.Provider.TopP, 0.9)
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := overrideBase()
			if err := ApplyOverrides(cfg, []string{tt.pair}, t.TempDir()); err != nil {
				t.Fatalf("ApplyOverrides(%q): %v", tt.pair, err)
			}
			if !tt.check(cfg) {
				t.Errorf("--set %s did not land in the provider block: %+v", tt.pair, cfg.Provider)
			}
		})
	}
}

// TestApplyOverridesProviderTuning covers what the four tuning keys do beyond
// landing in a field: they retune a run whose authority is already fixed, so
// they must be as predictable as the budget keys - a bad number is exit 2, and
// an override never invents a reasoning block the operator did not ask for.
func TestApplyOverridesProviderTuning(t *testing.T) {
	t.Run("effort keeps an existing budget", func(t *testing.T) {
		cfg := overrideBase()
		cfg.Provider.Reasoning = &ReasoningConfig{Effort: "low", BudgetTokens: 4096}
		if err := ApplyOverrides(cfg, []string{"provider.reasoning.effort=max"}, t.TempDir()); err != nil {
			t.Fatal(err)
		}
		if cfg.Provider.Reasoning.Effort != "max" || cfg.Provider.Reasoning.BudgetTokens != 4096 {
			t.Errorf("reasoning = %+v, want the effort replaced and the budget untouched", *cfg.Provider.Reasoning)
		}
	})

	// An empty value is how an operator drops back to the provider default,
	// exactly as an absent block does - so with no block to clear there is
	// nothing to create either.
	t.Run("empty effort clears without creating a block", func(t *testing.T) {
		cfg := overrideBase()
		if err := ApplyOverrides(cfg, []string{"provider.reasoning.effort="}, t.TempDir()); err != nil {
			t.Fatal(err)
		}
		if cfg.Provider.Reasoning != nil {
			t.Errorf("reasoning = %+v, want nil", *cfg.Provider.Reasoning)
		}

		cfg.Provider.Reasoning = &ReasoningConfig{Effort: "high"}
		if err := ApplyOverrides(cfg, []string{"provider.reasoning.effort="}, t.TempDir()); err != nil {
			t.Fatal(err)
		}
		if cfg.Provider.Reasoning.Effort != "" {
			t.Errorf("effort = %q, want it cleared", cfg.Provider.Reasoning.Effort)
		}
	})

	// The value is not judged here - Validate owns the vocabulary, once, for
	// YAML and --set alike.
	t.Run("an unknown effort survives to validation", func(t *testing.T) {
		cfg := overrideBase()
		if err := ApplyOverrides(cfg, []string{"provider.reasoning.effort=insane"}, t.TempDir()); err != nil {
			t.Fatalf("ApplyOverrides rejected a value Validate owns: %v", err)
		}
		if cfg.Provider.Reasoning.Effort != "insane" {
			t.Errorf("effort = %q", cfg.Provider.Reasoning.Effort)
		}
	})

	for _, tt := range []struct {
		name string
		pair string
	}{
		{"temperature is not a number", "provider.temperature=warm"},
		{"top_p is not a number", "provider.top_p="},
		{"max_output_tokens is not an integer", "provider.max_output_tokens=lots"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg := overrideBase()
			err := ApplyOverrides(cfg, []string{tt.pair}, t.TempDir())
			if err == nil {
				t.Fatal("expected a rejection")
			}
			if !errors.Is(err, ErrInvalid) {
				t.Errorf("error does not wrap ErrInvalid: %v", err)
			}
			key, _, _ := SplitOverride(tt.pair)
			if !strings.Contains(err.Error(), key) {
				t.Errorf("error %q does not name the key %q", err, key)
			}
		})
	}
}
