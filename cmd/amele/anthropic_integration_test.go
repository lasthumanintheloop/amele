//go:build integration

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestIntegrationAnthropic exercises the full binary path against the real
// Anthropic Messages API: native provider selection (provider.type),
// tool calling round-trip, and structured output via the validate+retry
// fallback (Anthropic has no native response_format, so this is the only
// schema path that runs there). Release-gate material, not CI-default
// (docs/engineering.md §6): run with
//
//	AMELE_TEST_ANTHROPIC_API_KEY=sk-ant-... \
//	go test -tags integration ./cmd/amele -run TestIntegrationAnthropic -v
//
// Optional overrides: AMELE_TEST_ANTHROPIC_MODEL (default claude-haiku-4-5),
// AMELE_TEST_ANTHROPIC_BASE_URL (default https://api.anthropic.com).
func TestIntegrationAnthropic(t *testing.T) {
	if os.Getenv("AMELE_TEST_ANTHROPIC_API_KEY") == "" {
		t.Skip("AMELE_TEST_ANTHROPIC_API_KEY not set")
	}
	model := os.Getenv("AMELE_TEST_ANTHROPIC_MODEL")
	if model == "" {
		// Cheapest current model: a smoke test should cost cents, not dollars.
		model = "claude-haiku-4-5"
	}
	baseURL := os.Getenv("AMELE_TEST_ANTHROPIC_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.anthropic.com"
	}

	dir := t.TempDir()
	logPath := filepath.Join(dir, "app.log")
	logContent := "2026-08-10 03:12:44 ERROR payment-service: connection refused to db-primary:5432\n" +
		"2026-08-10 03:12:45 INFO payment-service: retrying in 5s\n"
	if err := os.WriteFile(logPath, []byte(logContent), 0o600); err != nil {
		t.Fatal(err)
	}

	yaml := fmt.Sprintf(`
model: %s
provider:
  type: anthropic
  base_url: %s
  api_key: ${AMELE_TEST_ANTHROPIC_API_KEY}
system_prompt: |
  You are a log triage agent. Read app.log with fs_read and report the most
  severe problem you find.
tools:
  fs: true
output:
  schema:
    type: object
    properties:
      severity:
        type: string
        enum: [error, warning, info]
      summary:
        type: string
    required: [severity, summary]
    additionalProperties: false
limits:
  max_turns: 6
  max_tokens: 50000
  timeout: 2m
session_dir: sessions
`, model, baseURL)
	cfgPath := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(cfgPath, []byte(yaml), 0o600); err != nil { //nolint:gosec // G703: cfgPath is t.TempDir() + a constant name, not tainted input.
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := run(context.Background(), []string{"run", cfgPath, "triage the logs"},
		strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != ExitOK {
		t.Fatalf("exit %d\nstderr: %s", code, stderr.String())
	}
	t.Logf("agent answer: %s", stdout.String())
	t.Logf("summary: %s", strings.TrimSpace(stderr.String()))

	// CONTRACT: with output.schema set, stdout must be exactly one valid JSON
	// document matching the schema - nothing else (exit-code contract's
	// structured-output companion, docs/contracts/cli.md).
	var out struct {
		Severity string `json:"severity"`
		Summary  string `json:"summary"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &out); err != nil {
		t.Fatalf("stdout is not a single JSON document: %v\nstdout: %s", err, stdout.String())
	}
	if out.Severity != "error" {
		t.Errorf("expected severity \"error\" for a connection-refused log, got %q (summary: %q)", out.Severity, out.Summary)
	}
	if out.Summary == "" {
		t.Error("summary must not be empty")
	}

	sessions, _ := filepath.Glob(filepath.Join(dir, "sessions", "*.jsonl"))
	if len(sessions) != 1 {
		t.Fatalf("expected one session file, got %v", sessions)
	}
	// The session log must never contain the raw API key (SECURITY: session
	// redaction covers interpolated secrets).
	data, err := os.ReadFile(sessions[0])
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), os.Getenv("AMELE_TEST_ANTHROPIC_API_KEY")) {
		t.Error("API key leaked into session log")
	}
}
