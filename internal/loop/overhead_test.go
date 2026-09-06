package loop

import (
	"encoding/json"
	"testing"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/tools"
)

// harnessTokenBudget is the CI-enforced ceiling for the harness's own prompt
// overhead (docs/engineering.md §8): the total size of all builtin tool definitions.
// The estimate is chars/4 - crude but stable, and it only needs to catch
// definitions bloating over time, not be exact.
// Subprocess and MCP tool definitions are deliberately NOT counted here: they
// are the operator's own budget, chosen per config, and `amele explain` reports
// their size (with a warning past ~4000 tokens) - this ceiling governs only what
// the harness itself injects into every context, whatever the config says.
const harnessTokenBudget = 1500

// TestBuiltinToolDefinitionBudget guards the "measured minimalism" promise:
// every builtin tool definition the harness injects into the model's context
// must fit the token budget. This covers every builtin the registry can
// wire up, including the shell tool - a config with Enabled:true is only
// used to construct it here, since NewShell itself does not gate on
// cfg.Enabled (the registry does); its definition text is identical whether
// or not the tool is actually turned on for a given run.
func TestBuiltinToolDefinitionBudget(t *testing.T) {
	fs, err := tools.NewFSTools(t.TempDir(), tools.FSOptions{})
	if err != nil {
		t.Fatal(err)
	}

	shell, err := tools.NewShell(config.ShellConfig{Enabled: true}, t.TempDir(), tools.ShellOptions{})
	if err != nil {
		t.Fatal(err)
	}

	all := append([]tools.Tool{}, fs...)
	all = append(all, shell)

	totalChars := 0
	for _, tool := range all {
		def := tool.Def()
		// Compact the schema first: pretty-printing whitespace is not sent
		// to the provider as-is, so it should not count against the budget.
		var compact json.RawMessage
		buf, err := json.Marshal(json.RawMessage(def.Parameters))
		if err != nil {
			t.Fatalf("tool %s has invalid parameter JSON: %v", def.Name, err)
		}
		compact = buf
		totalChars += len(def.Name) + len(def.Description) + len(compact)
	}

	estTokens := totalChars / 4
	t.Logf("builtin tool definitions: %d chars ≈ %d tokens (budget %d)", totalChars, estTokens, harnessTokenBudget)
	if estTokens > harnessTokenBudget {
		t.Errorf("builtin tool definitions exceed the harness token budget: ~%d > %d", estTokens, harnessTokenBudget)
	}
}
