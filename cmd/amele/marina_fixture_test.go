package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestMarinaFixtureValidatesForEveryTarget keeps the committed acceptance
// fixture honest without a network: testdata/marina/style-guide.yaml is
// parametrized over the provider, so every dialect the integration test points
// it at must still be a legal configuration.
//
// It exists because the fixture's own test (TestIntegrationMarina) is behind
// the integration tag and needs API keys, so CI never runs it. A validation
// rule that grows a new dialect-dependent case - kimi's fixed sampling was one
// - would otherwise break the release-gate run instead of the pull request.
func TestMarinaFixtureValidatesForEveryTarget(t *testing.T) {
	cfgPath, err := filepath.Abs(filepath.Join("..", "..", "testdata", "marina", "style-guide.yaml"))
	if err != nil {
		t.Fatal(err)
	}

	targets := []struct {
		dialect string
		baseURL string
		model   string
	}{
		{"groq", "https://api.groq.com/openai/v1", "openai/gpt-oss-20b"},
		{"deepseek", "https://api.deepseek.com", "deepseek-v4-flash"},
	}
	for _, target := range targets {
		t.Run(target.dialect, func(t *testing.T) {
			fixture := map[string]string{
				"MARINA_MODEL":    target.model,
				"MARINA_BASE_URL": target.baseURL,
				"MARINA_DIALECT":  target.dialect,
				"MARINA_API_KEY":  "test-key-not-a-real-credential",
			}
			lookup := func(name string) (string, bool) {
				if value, ok := fixture[name]; ok {
					return value, true
				}
				return os.LookupEnv(name)
			}

			var stdout, stderr bytes.Buffer
			code := run(context.Background(), []string{"validate", cfgPath},
				strings.NewReader(""), &stdout, &stderr, lookup)
			if code != ExitOK {
				t.Fatalf("exit %d\nstdout: %s\nstderr: %s", code, stdout.String(), stderr.String())
			}
		})
	}
}
