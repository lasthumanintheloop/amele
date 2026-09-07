package explain

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/lasthumanintheloop/amele/internal/config"
)

// fallbackCfg is a config whose primary target emits a mapping block (the
// max_output_tokens row) and which carries the two entry shapes the row has to
// tell apart: a native wire with no base_url of its own, and an openai dialect
// that names its endpoint.
func fallbackCfg() *config.Config {
	cfg := baseCfg()
	cfg.Provider.MaxOutputTokens = 4096
	cfg.Provider.Fallback = []config.FallbackTarget{
		{
			Model:          "claude-opus-5",
			ProviderConfig: config.ProviderConfig{Type: config.ProviderTypeAnthropic},
		},
		{
			Model: "deepseek-v4",
			ProviderConfig: config.ProviderConfig{
				Dialect: "deepseek",
				BaseURL: "https://api.deepseek.com/v1",
			},
		},
	}
	return cfg
}

// TestFallbackRows pins the rows themselves AND their position. Both matter:
// the chain is an order, so a reader has to see it as one list, immediately
// under the primary target it backs up and inside the same section - a row that
// drifted above the mapping block would read as a property of the primary.
func TestFallbackRows(t *testing.T) {
	got := Render(fallbackCfg(), registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil)

	// The mapping row before, the section's blank line and the next header
	// after: this block pins the rows between the two things they sit between.
	want := "    max_output_tokens: 4096 -> max_completion_tokens: 4096\n" +
		"  fallback 1:      \"claude-opus-5\" via anthropic (default: api.anthropic.com)\n" +
		"  fallback 2:      \"deepseek-v4\" via openai/deepseek https://api.deepseek.com/v1\n" +
		"\nTOOLS\n"
	if !strings.Contains(got, want) {
		t.Errorf("report missing the fallback block\nwant:\n%s\nfull report:\n%s", want, got)
	}
}

// TestFallbackRowsAbsentWithoutAChain is the compatibility half: every config
// written before fallback existed must render byte-identically, which is also
// what keeps the other goldens unchanged.
func TestFallbackRowsAbsentWithoutAChain(t *testing.T) {
	got := Render(baseCfg(), registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil)
	if strings.Contains(got, "fallback") {
		t.Errorf("a config with no fallback grew a fallback row:\n%s", got)
	}
}

// TestFallbackHostPlaceholders pins the third piece of the row for the states
// an operator gets wrong: the gemini backends default to two different hosts,
// and the openai wire has NO default at all - it must say so rather than
// borrow another wire's host, because that would tell an operator the failover
// goes somewhere it cannot.
func TestFallbackHostPlaceholders(t *testing.T) {
	tests := []struct {
		name  string
		model string
		entry config.ProviderConfig
		want  string
	}{
		{
			name:  "openai wire without a base_url has no default",
			model: "backup",
			entry: config.ProviderConfig{Type: config.ProviderTypeOpenAI},
			want:  "  fallback 1:      \"backup\" via openai (unset)\n",
		},
		{
			name:  "gemini ai studio",
			model: "backup",
			entry: config.ProviderConfig{Type: config.ProviderTypeGemini},
			want:  "  fallback 1:      \"backup\" via gemini (default: generativelanguage.googleapis.com)\n",
		},
		{
			name:  "gemini vertex is addressed by location",
			model: "backup",
			entry: config.ProviderConfig{
				Type:   config.ProviderTypeGemini,
				Vertex: &config.VertexConfig{Project: "p", Location: "europe-west4"},
			},
			want: "  fallback 1:      \"backup\" via gemini/vertex " +
				"(default: europe-west4-aiplatform.googleapis.com)\n",
		},
		{
			// Validate refuses an entry with no model; explain reports on such
			// configs anyway, and a blank there would read as a rendering bug.
			name:  "an unset model still reports its target",
			model: "",
			entry: config.ProviderConfig{Type: config.ProviderTypeAnthropic},
			want:  "  fallback 1:      (unset) via anthropic (default: api.anthropic.com)\n",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := baseCfg()
			cfg.Provider.Fallback = []config.FallbackTarget{{Model: tc.model, ProviderConfig: tc.entry}}
			got := Render(cfg, registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil)
			if !strings.Contains(got, tc.want) {
				t.Errorf("report missing %q\nfull report:\n%s", tc.want, got)
			}
		})
	}
}

// TestFallbackRowCannotForgeARow is the security regression. explain reports on
// configs Validate REJECTED, so every piece of a fallback row is attacker-shaped
// text: the model, the base_url (printed bare, unlike the quoted model) and the
// dialect that composes the identity. A newline in any of them would invent a
// report line an operator reads as amele's own words.
func TestFallbackRowCannotForgeARow(t *testing.T) {
	forged := "  fallback 2:      \"ghost\" via anthropic https://evil.example.com/v1"
	tests := []struct {
		name  string
		entry config.FallbackTarget
	}{
		{
			name: "newline in the model",
			entry: config.FallbackTarget{
				Model:          "backup\n" + forged,
				ProviderConfig: config.ProviderConfig{Type: config.ProviderTypeAnthropic},
			},
		},
		{
			name: "newline in the base_url",
			entry: config.FallbackTarget{
				Model: "backup",
				ProviderConfig: config.ProviderConfig{
					Type:    config.ProviderTypeAnthropic,
					BaseURL: "https://backup.example.com\n" + forged,
				},
			},
		},
		{
			name: "newline in the dialect",
			entry: config.FallbackTarget{
				Model: "backup",
				ProviderConfig: config.ProviderConfig{
					BaseURL: "https://backup.example.com/v1",
					Dialect: "deepseek\n" + forged,
				},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := baseCfg()
			cfg.Provider.Fallback = []config.FallbackTarget{tc.entry}
			got := Render(cfg, registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil)
			// Counted per LINE, not per substring: the escaped payload still
			// contains the row prefix as text, and that is the point - it must
			// stay text inside the one row the single entry earns.
			rows := 0
			for _, line := range strings.Split(got, "\n") {
				if strings.HasPrefix(line, "  fallback ") {
					rows++
				}
				if strings.HasPrefix(line, "  fallback 2:") {
					t.Errorf("a newline forged a second row: %q\nfull report:\n%s", line, got)
				}
			}
			if rows != 1 {
				t.Errorf("want exactly 1 fallback row, got %d\nfull report:\n%s", rows, got)
			}
		})
	}
}

// fallbackEnvYAML references a variable ONLY from inside a fallback entry: the
// requirements checklist is what tells an operator the failover cannot
// authenticate, and a chain whose credential is invisible until the primary
// dies is the worst possible time to discover it.
const fallbackEnvYAML = `model: golden-model
provider:
  base_url: https://api.example.com/v1
  api_key: ${TEST_KEY}
  fallback:
    - model: claude-opus-5
      type: anthropic
      api_key: ${FALLBACK_KEY}
limits:
  max_turns: 10
  max_tokens: 1000
`

// loadYAML loads text the way the CLI would, from a file in a temporary
// directory, so the config carries its interpolation bindings - a struct
// literal would have none and the env assertions would pass vacuously. It
// returns the directory the golden normalizes away.
func loadYAML(t *testing.T, text string, env config.LookupEnv) (*config.Config, string) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "agent.yaml")
	// The suppression below is for a path built from t.TempDir() and a fixed
	// name; the analyzer flags it only because the CONTENT arrives from a
	// fixture file, which is not what the path is built from.
	if err := os.WriteFile(path, []byte(text), 0o600); err != nil { //nolint:gosec // G703: temp dir + constant name.
		t.Fatal(err)
	}
	cfg, err := config.LoadTolerant(path, env)
	if err != nil {
		t.Fatal(err)
	}
	return cfg, dir
}

// TestFallbackEnvInRequirements pins that a fallback entry's ${VAR} reaches the
// REQUIREMENTS checklist with its state, in both directions.
func TestFallbackEnvInRequirements(t *testing.T) {
	tests := []struct {
		name string
		env  config.LookupEnv
		want string
	}{
		{
			name: "set",
			env: func(key string) (string, bool) {
				return "value-for-" + key, true
			},
			want: "    FALLBACK_KEY    ✓ set\n",
		},
		{
			name: "missing",
			env: func(key string) (string, bool) {
				if key == "TEST_KEY" {
					return "sk-primary", true
				}
				return "", false
			},
			want: "    FALLBACK_KEY    ✗ MISSING\n",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg, _ := loadYAML(t, fallbackEnvYAML, tc.env)
			got := requirementsReport(cfg, alwaysFound)
			if !strings.Contains(got, tc.want) {
				t.Errorf("requirements missing %q\ngot:\n%s", tc.want, got)
			}
		})
	}
}

// TestRenderFallbackGolden is the golden docs/engineering.md §6 requires for a
// UI surface: the whole report for a two-entry chain, so the rows are pinned
// together with the REQUIREMENTS lines the entries' credentials produce.
func TestRenderFallbackGolden(t *testing.T) {
	text, err := os.ReadFile(filepath.Join("testdata", "explain-fallback.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	// PRIMARY_KEY resolves, FALLBACK_KEY does not: the golden then shows both
	// marks, and the missing one is the state a chain is actually deployed in.
	env := func(key string) (string, bool) {
		if key == "PRIMARY_KEY" {
			return "sk-primary-key-value", true
		}
		return "", false
	}
	cfg, dir := loadYAML(t, string(text), env)
	got := strings.ReplaceAll(
		Render(cfg, registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil), dir, "<TMP>")

	goldenPath := filepath.Join("testdata", "golden", "explain-fallback.txt")
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
