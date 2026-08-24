package llm

import (
	"slices"
	"strings"
	"testing"
)

// TestParseDialect pins the whole dialect vocabulary and the two rules that
// make it safe in a config file: an omitted value means the OpenAI dialect (so
// every config written before dialects existed keeps its behavior), and
// anything else must match a known name EXACTLY. Case folding and trimming are
// deliberately absent - a dialect silently "corrected" from "OpenAI " would
// pick the wire mapping for a provider the author did not name.
func TestParseDialect(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		want    Dialect
		wantErr bool
	}{
		{"empty means openai", "", DialectOpenAI, false},
		{"openai", "openai", DialectOpenAI, false},
		{"deepseek", "deepseek", DialectDeepSeek, false},
		{"glm", "glm", DialectGLM, false},
		{"kimi", "kimi", DialectKimi, false},
		{"groq", "groq", DialectGroq, false},
		{"openrouter", "openrouter", DialectOpenRouter, false},
		{"mixed case is not folded", "OpenAI", "", true},
		{"upper case is not folded", "DEEPSEEK", "", true},
		{"surrounding space is not trimmed", "deepseek ", "", true},
		{"leading space is not trimmed", " glm", "", true},
		{"unknown provider", "gemini", "", true},
		{"wire type is not a dialect", "anthropic", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseDialect(tt.in)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("ParseDialect(%q) = %q, want an error", tt.in, got)
				}
				// The message is the whole fix instruction: a typo must be
				// repairable without opening the docs, so it names the input
				// and every accepted spelling.
				if !strings.Contains(err.Error(), tt.in) {
					t.Errorf("error %q does not quote the offending value %q", err, tt.in)
				}
				for _, valid := range []string{"openai", "deepseek", "glm", "kimi", "groq", "openrouter"} {
					if !strings.Contains(err.Error(), valid) {
						t.Errorf("error %q does not list the valid value %q", err, valid)
					}
				}
				if got != "" {
					t.Errorf("ParseDialect(%q) = %q on error, want the zero Dialect", tt.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("ParseDialect(%q): %v", tt.in, err)
			}
			if got != tt.want {
				t.Errorf("ParseDialect(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

// TestDialectConstantValues pins the wire spellings of the constants
// themselves: they are the config file's vocabulary (provider.dialect) and the
// published JSON Schema enumerates them, so renaming one is a contract change
// rather than a refactor.
func TestDialectConstantValues(t *testing.T) {
	want := map[Dialect]string{
		DialectOpenAI:     "openai",
		DialectDeepSeek:   "deepseek",
		DialectGLM:        "glm",
		DialectKimi:       "kimi",
		DialectGroq:       "groq",
		DialectOpenRouter: "openrouter",
	}
	for d, s := range want {
		if string(d) != s {
			t.Errorf("dialect constant = %q, want %q", string(d), s)
		}
	}
}

// TestOwnedWireFieldsCoverEveryMappedKey is the anti-drift check behind
// OwnedWireFields: whatever key MapReasoning can emit for a dialect must be in
// that dialect's owned set, or provider.params would be allowed to carry a key
// the very next request overwrites. It walks the whole effort vocabulary plus a
// budget, so a new mapping that invents a key fails here rather than in
// production.
func TestOwnedWireFieldsCoverEveryMappedKey(t *testing.T) {
	// The union config validation accepts (config.effortValues); duplicated
	// here on purpose - llm must not import config, and a value that exists on
	// only one side is exactly what this test should catch.
	efforts := []string{"", "none", "low", "medium", "high", "xhigh", "max"}
	for _, d := range dialects {
		owned := OwnedWireFields(d)
		for _, effort := range efforts {
			for _, budget := range []int{0, 8192} {
				mapped := MapReasoning(d, ReasoningSpec{Effort: effort, BudgetTokens: budget})
				for key := range mapped.Fields {
					if !slices.Contains(owned, key) {
						t.Errorf("dialect %q maps effort %q/budget %d onto key %q, which OwnedWireFields does not claim: %v",
							d, effort, budget, key, owned)
					}
				}
			}
		}
	}
}

// TestOwnedWireFieldsAreDialectScoped pins the two answers the params escape
// hatch depends on: the cap spelling follows CapField, and a dialect that emits
// no thinking object does not claim the key (the kimi case that used to block
// the K2.x controls).
func TestOwnedWireFieldsAreDialectScoped(t *testing.T) {
	tests := []struct {
		dialect Dialect
		want    []string
		absent  []string
	}{
		{DialectOpenAI, []string{"max_completion_tokens", "reasoning_effort"}, []string{"max_tokens", "thinking", "reasoning"}},
		{DialectDeepSeek, []string{"max_tokens", "thinking", "reasoning_effort"}, []string{"max_completion_tokens", "reasoning"}},
		{DialectGLM, []string{"max_tokens", "thinking", "reasoning_effort"}, []string{"max_completion_tokens", "reasoning"}},
		{DialectKimi, []string{"max_completion_tokens", "reasoning_effort"}, []string{"thinking", "reasoning"}},
		{DialectGroq, []string{"max_completion_tokens", "reasoning_effort"}, []string{"thinking", "reasoning"}},
		{DialectOpenRouter, []string{"max_tokens", "reasoning"}, []string{"max_completion_tokens", "thinking"}},
	}
	for _, tt := range tests {
		t.Run(string(tt.dialect), func(t *testing.T) {
			owned := OwnedWireFields(tt.dialect)
			// Every dialect writes these, whatever else it maps.
			for _, key := range []string{"model", "messages", "tools", "response_format", "temperature", "top_p"} {
				if !slices.Contains(owned, key) {
					t.Errorf("owned set is missing the shared key %q: %v", key, owned)
				}
			}
			for _, key := range tt.want {
				if !slices.Contains(owned, key) {
					t.Errorf("owned set is missing %q: %v", key, owned)
				}
			}
			for _, key := range tt.absent {
				if slices.Contains(owned, key) {
					t.Errorf("owned set claims %q, which this dialect never writes: %v", key, owned)
				}
			}
		})
	}
}
