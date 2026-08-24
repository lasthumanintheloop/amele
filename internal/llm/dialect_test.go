package llm

import (
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
