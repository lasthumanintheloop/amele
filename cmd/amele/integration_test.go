//go:build integration

package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestIntegrationRealEndpoint exercises the full binary path against a real
// OpenAI-compatible endpoint. It is release-gate material, not CI-default
// (docs/engineering.md §6): run with
//
//	AMELE_TEST_BASE_URL=https://api.openai.com/v1 \
//	AMELE_TEST_MODEL=gpt-4.1-mini \
//	AMELE_TEST_API_KEY=sk-... \
//	go test -tags integration ./cmd/amele -run TestIntegrationRealEndpoint -v
func TestIntegrationRealEndpoint(t *testing.T) {
	baseURL := os.Getenv("AMELE_TEST_BASE_URL")
	model := os.Getenv("AMELE_TEST_MODEL")
	if baseURL == "" || model == "" {
		t.Skip("AMELE_TEST_BASE_URL / AMELE_TEST_MODEL not set")
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
  base_url: %s
  api_key: ${AMELE_TEST_API_KEY}
system_prompt: |
  You are a log triage agent. Read app.log with fs_read and answer with a
  one-line summary of the most severe problem you find.
tools:
  fs: true
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

	sessions, _ := filepath.Glob(filepath.Join(dir, "sessions", "*.jsonl"))
	if len(sessions) != 1 {
		t.Errorf("expected one session file, got %v", sessions)
	}
}
