//go:build integration

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestIntegrationMarina is the acceptance test for the provider dialect layer:
// the committed fixture testdata/marina/style-guide.yaml run end to end against
// a real endpoint, once per dialect the environment supplies a key for.
//
// It is the only test that can prove the whole slice, because every part of it
// is a live-provider fact:
//
//   - the reasoning knob reaches the field the dialect actually reads;
//   - a dialect that RETURNS reasoning gets it echoed back inside the tool loop
//     (DeepSeek answers a missing echo with a 400 - research §"Load-bearing
//     quirks" #2), which no in-process fake can check;
//   - max_output_tokens leaves room for the thinking as well as the answer, so
//     no turn ends with finish_reason "length" (the truncation this fixture is
//     named after);
//   - output.schema survives all of it: stdout is one valid JSON document.
//
// Release-gate material, not CI-default (docs/engineering.md §6). Each target
// skips itself when its key is absent, so a partial environment runs the part
// it can:
//
//	GROQ_API_KEY=gsk_... DEEPSEEK_API_KEY=sk-... \
//	go test -tags integration ./cmd/amele -run TestIntegrationMarina -v
//
// The model and endpoint of each target can be overridden - model ids churn
// within months (research §"Load-bearing quirks" #4) - with
// AMELE_TEST_<TARGET>_MODEL and AMELE_TEST_<TARGET>_BASE_URL.
func TestIntegrationMarina(t *testing.T) {
	targets := []struct {
		name    string
		keyEnv  string
		dialect string
		model   string
		baseURL string
	}{
		{
			// Groq: reasoning_effort straight through, max_completion_tokens as
			// the cap, and no echo-back requirement - the simple end of the
			// dialect table.
			name:    "groq",
			keyEnv:  "GROQ_API_KEY",
			dialect: "groq",
			model:   "openai/gpt-oss-20b",
			baseURL: "https://api.groq.com/openai/v1",
		},
		{
			// DeepSeek native: thinking object plus reasoning_effort, max_tokens
			// as the cap, reasoning_content that MUST come back on every request
			// of the tool loop, and no native json_schema - so this target also
			// exercises the schema fallback (validate + retry).
			name:    "deepseek",
			keyEnv:  "DEEPSEEK_API_KEY",
			dialect: "deepseek",
			model:   "deepseek-v4-flash",
			baseURL: "https://api.deepseek.com",
		},
	}

	cfgPath, err := filepath.Abs(filepath.Join("..", "..", "testdata", "marina", "style-guide.yaml"))
	if err != nil {
		t.Fatal(err)
	}

	for _, target := range targets {
		t.Run(target.name, func(t *testing.T) {
			key := os.Getenv(target.keyEnv)
			if key == "" {
				t.Skipf("%s not set", target.keyEnv)
			}
			model := envOr("AMELE_TEST_"+strings.ToUpper(target.name)+"_MODEL", target.model)
			baseURL := envOr("AMELE_TEST_"+strings.ToUpper(target.name)+"_BASE_URL", target.baseURL)

			// The fixture is parametrized (docs/providers.md); the test supplies
			// the four variables through the injected environment rather than
			// through the process environment, so the same file serves every
			// dialect and nothing leaks between subtests.
			fixture := map[string]string{
				"MARINA_MODEL":    model,
				"MARINA_BASE_URL": baseURL,
				"MARINA_DIALECT":  target.dialect,
				"MARINA_API_KEY":  key,
			}
			lookup := func(name string) (string, bool) {
				if value, ok := fixture[name]; ok {
					return value, true
				}
				return os.LookupEnv(name)
			}

			// The session goes to a temp directory: an integration run must not
			// write into the checked-in fixture.
			sessionDir := t.TempDir()
			var stdout, stderr bytes.Buffer
			code := run(context.Background(),
				[]string{"run", "--set", "session_dir=" + sessionDir, cfgPath, "check violating.md"},
				strings.NewReader(""), &stdout, &stderr, lookup)
			if code != ExitOK {
				t.Fatalf("exit %d\nstderr: %s", code, stderr.String())
			}
			t.Logf("verdict: %s", strings.TrimSpace(stdout.String()))
			t.Logf("summary: %s", strings.TrimSpace(stderr.String()))

			// CONTRACT: with output.schema set, stdout is exactly one valid JSON
			// document and nothing else (docs/contracts/cli.md).
			var report struct {
				Verdict    string `json:"verdict"`
				Violations []struct {
					Rule   string `json:"rule"`
					Quote  string `json:"quote"`
					Reason string `json:"reason"`
				} `json:"violations"`
			}
			if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
				t.Fatalf("stdout is not a single JSON document: %v\nstdout: %s", err, stdout.String())
			}
			// The input breaks all three rules blatantly (an exclamation mark,
			// "We are", "revolutionary"), so a verdict of "pass" means the run
			// did not do the job - not that the model has taste.
			if report.Verdict != "fail" {
				t.Errorf("expected verdict \"fail\" for a document breaking all three rules, got %q", report.Verdict)
			}
			if len(report.Violations) == 0 {
				t.Fatal("verdict without a single violation")
			}
			for i, v := range report.Violations {
				// The schema REQUIRES a quote and a reason per violation; empty
				// strings would satisfy the type and not the intent.
				if strings.TrimSpace(v.Quote) == "" {
					t.Errorf("violations[%d] (%s): empty quote", i, v.Rule)
				}
				if strings.TrimSpace(v.Reason) == "" {
					t.Errorf("violations[%d] (%s): empty reason", i, v.Rule)
				}
			}

			assertMarinaSession(t, sessionDir, key)
		})
	}
}

// envOr returns the environment value of name, or fallback when it is unset or
// empty.
func envOr(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

// assertMarinaSession checks the run's session log for the two failures a
// schema-valid stdout cannot rule out: a turn the provider cut short, and a
// leaked API key.
//
// A `length` finish reason is the exact bug the fixture is named after - the
// reasoning ate the output cap and the answer was truncated. It can hide behind
// a successful run whenever a later turn repairs the damage, so it is asserted
// on the log rather than on stdout.
func assertMarinaSession(t *testing.T, sessionDir, key string) {
	t.Helper()
	sessions, _ := filepath.Glob(filepath.Join(sessionDir, "*.jsonl"))
	if len(sessions) != 1 {
		t.Fatalf("expected one session file, got %v", sessions)
	}
	data, err := os.ReadFile(sessions[0])
	if err != nil {
		t.Fatal(err)
	}
	// SECURITY: the interpolated key must never reach the log (redaction is by
	// value, so it cannot depend on the variable's name).
	if strings.Contains(string(data), key) {
		t.Error("API key leaked into session log")
	}

	turns := 0
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var event struct {
			Type           string `json:"type"`
			Turn           int    `json:"turn"`
			FinishReason   string `json:"finish_reason"`
			ReasoningBytes int    `json:"reasoning_bytes"`
		}
		if err := json.Unmarshal([]byte(line), &event); err != nil {
			t.Fatalf("session line is not JSON: %v\nline: %s", err, line)
		}
		if event.Type != "llm_response" {
			continue
		}
		turns++
		if event.FinishReason == "length" {
			t.Errorf("turn %d was truncated (finish_reason %q): provider.max_output_tokens left no room for the reasoning plus the answer",
				event.Turn, event.FinishReason)
		}
		// Not an assertion: a dialect that returns no reasoning is legitimate
		// (Groq's models differ), and the value is the cheapest evidence that
		// the carrier is doing its job when one does.
		t.Logf("turn %d: finish_reason %q, reasoning_bytes %d", event.Turn, event.FinishReason, event.ReasoningBytes)
	}
	if turns == 0 {
		t.Error("session log has no llm_response event")
	}
}
