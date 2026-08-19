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

// The in-process tests cover the protocol against a fake server; these two
// exercise the same code against a REAL MCP server, which is where the things
// a fake cannot fake live: a binary that must be on the host, a process that
// must be reaped, a remote endpoint that must accept the headers amele sends.
// They are release-gate material, not CI-default (docs/engineering.md §6), and
// they skip themselves when the environment does not name a server.

// TestIntegrationMCPStdio runs a real `amele run` against a real stdio MCP
// server, so the child process, its minimal environment and the process-group
// cleanup are all exercised end to end. The server command is split on spaces
// (argv, never a shell string):
//
//	AMELE_MCP_STDIO_CMD="/usr/local/bin/mcp-server-filesystem /tmp/notes" \
//	AMELE_TEST_BASE_URL=https://api.openai.com/v1 \
//	AMELE_TEST_MODEL=gpt-4.1-mini \
//	AMELE_TEST_API_KEY=sk-... \
//	go test -tags integration ./cmd/amele -run TestIntegrationMCPStdio -v
//
// Without the provider variables the test still runs, as an `explain`: that
// alone proves the server spawns, handshakes and lists tools, and it costs no
// tokens.
func TestIntegrationMCPStdio(t *testing.T) {
	cmdline := os.Getenv("AMELE_MCP_STDIO_CMD")
	if cmdline == "" {
		t.Skip("AMELE_MCP_STDIO_CMD not set")
	}
	argv := strings.Fields(cmdline)

	var quoted []string
	for _, arg := range argv {
		quoted = append(quoted, fmt.Sprintf("%q", arg))
	}

	baseURL := os.Getenv("AMELE_TEST_BASE_URL")
	model := os.Getenv("AMELE_TEST_MODEL")
	provider := "model: fake\nprovider:\n  base_url: https://example.invalid/v1\n  api_key: ${AMELE_MCP_UNUSED_KEY}\n"
	if baseURL != "" && model != "" {
		provider = fmt.Sprintf("model: %s\nprovider:\n  base_url: %s\n  api_key: ${AMELE_TEST_API_KEY}\n", model, baseURL)
	}

	dir := t.TempDir()
	yaml := provider + fmt.Sprintf(`system_prompt: |
  Use the MCP tools you were given to answer. Do not guess.
tools:
  fs: false
mcp:
  servers:
    - name: files
      transport:
        type: stdio
        command: [%s]
      call_timeout: 30s
      required: true
limits:
  max_turns: 6
  max_tokens: 50000
  timeout: 2m
session_dir: sessions
`, strings.Join(quoted, ", "))
	cfgPath := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(cfgPath, []byte(yaml), 0o600); err != nil { //nolint:gosec // G703: cfgPath is t.TempDir() + a constant name, not tainted input.
		t.Fatal(err)
	}

	// `explain` is the cheap half: it connects, lists and disconnects. A
	// required server that failed is reported in the body, never in the exit
	// code, so the output has to be inspected rather than the status.
	var stdout, stderr bytes.Buffer
	if code := run(context.Background(), []string{"explain", cfgPath},
		strings.NewReader(""), &stdout, &stderr, os.LookupEnv); code != ExitOK {
		t.Fatalf("explain: exit %d\nstderr: %s", code, stderr.String())
	}
	report := stdout.String()
	t.Logf("explain:\n%s", report)
	if !strings.Contains(report, "connected") {
		t.Fatalf("explain did not report a connected server:\n%s", report)
	}

	if baseURL == "" || model == "" {
		t.Skip("AMELE_TEST_BASE_URL / AMELE_TEST_MODEL not set: stopping after explain")
	}

	stdout.Reset()
	stderr.Reset()
	code := run(context.Background(), []string{"run", cfgPath, "list the files you can see and name one of them"},
		strings.NewReader(""), &stdout, &stderr, os.LookupEnv)
	if code != ExitOK {
		t.Fatalf("run: exit %d\nstderr: %s", code, stderr.String())
	}
	t.Logf("agent answer: %s", stdout.String())

	sessions, _ := filepath.Glob(filepath.Join(dir, "sessions", "*.jsonl"))
	if len(sessions) != 1 {
		t.Fatalf("expected one session file, got %v", sessions)
	}
	events, err := os.ReadFile(sessions[0])
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(events), `"type":"mcp_connect"`) ||
		!strings.Contains(string(events), `"type":"mcp_tools_listed"`) {
		t.Errorf("session log is missing the MCP events:\n%s", events)
	}
}

// TestIntegrationMCPHTTP points `explain` at a real Streamable HTTP MCP server,
// which is the only way to prove the header injection, the https rule and the
// session lifecycle against something that was not written by this repository:
//
//	AMELE_MCP_HTTP_URL=https://api.example.com/mcp/ \
//	AMELE_MCP_HTTP_TOKEN=... \
//	go test -tags integration ./cmd/amele -run TestIntegrationMCPHTTP -v
//
// The token is optional: a server that needs no credential is configured
// without the Authorization header rather than with an empty one, because an
// empty ${VAR} is a config error.
func TestIntegrationMCPHTTP(t *testing.T) {
	url := os.Getenv("AMELE_MCP_HTTP_URL")
	if url == "" {
		t.Skip("AMELE_MCP_HTTP_URL not set")
	}

	headers := ""
	if os.Getenv("AMELE_MCP_HTTP_TOKEN") != "" {
		headers = "        headers:\n          Authorization: \"Bearer ${AMELE_MCP_HTTP_TOKEN}\"\n"
	}

	dir := t.TempDir()
	yaml := fmt.Sprintf(`model: fake
provider:
  base_url: https://example.invalid/v1
  api_key: ${AMELE_MCP_HTTP_URL}
system_prompt: "unused: this config is only ever explained."
tools:
  fs: false
mcp:
  servers:
    - name: remote
      transport:
        type: http
        url: %s
%s      required: true
`, url, headers)
	cfgPath := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(cfgPath, []byte(yaml), 0o600); err != nil { //nolint:gosec // G703: cfgPath is t.TempDir() + a constant name, not tainted input.
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	if code := run(context.Background(), []string{"explain", cfgPath},
		strings.NewReader(""), &stdout, &stderr, os.LookupEnv); code != ExitOK {
		t.Fatalf("explain: exit %d\nstderr: %s", code, stderr.String())
	}
	report := stdout.String()
	t.Logf("explain:\n%s", report)
	if !strings.Contains(report, "connected") {
		t.Fatalf("explain did not report a connected server:\n%s", report)
	}
	// SECURITY: whatever else the report says, it must never echo the bearer
	// token - the whole point of the ${ENV}-only rule for sensitive headers.
	if token := os.Getenv("AMELE_MCP_HTTP_TOKEN"); token != "" && strings.Contains(report, token) {
		t.Error("explain leaked the Authorization token into its report")
	}
}

// TestIntegrationMCPOAuth points `amele mcp status` and, optionally,
// `amele explain` at a real OAuth-protected Streamable HTTP MCP server. This
// is the one place slice 1's fake authorization server (Task 6) cannot stand
// in: a real discovery document, a real token endpoint and a real client-id
// metadata fetch.
//
// It NEVER opens a browser, in CI or anywhere else: `amele mcp login` is not
// invoked by this test. Logging in is a manual precondition the operator
// performs once, out of band, with:
//
//	amele mcp login <config> <server>
//
// before running this test with AMELE_MCP_OAUTH_ASSUME_LOGIN=1. Without that
// variable the test only proves the read-only half - `status` never talks to
// the authorization server, never refreshes and never opens anything, so it
// is safe to run with no credential stored at all.
//
//	AMELE_MCP_OAUTH_URL=https://mcp.example.com/mcp \
//	go test -tags integration ./cmd/amele -run TestIntegrationMCPOAuth -v
//
//	# after a one-time manual `amele mcp login`:
//	AMELE_MCP_OAUTH_URL=https://mcp.example.com/mcp \
//	AMELE_MCP_OAUTH_ASSUME_LOGIN=1 \
//	go test -tags integration ./cmd/amele -run TestIntegrationMCPOAuth -v
func TestIntegrationMCPOAuth(t *testing.T) {
	url := os.Getenv("AMELE_MCP_OAUTH_URL")
	if url == "" {
		t.Skip("AMELE_MCP_OAUTH_URL not set")
	}

	dir := t.TempDir()
	yaml := fmt.Sprintf(`model: fake
provider:
  base_url: https://example.invalid/v1
  api_key: ${AMELE_MCP_OAUTH_URL}
system_prompt: "unused: this config is only ever explained or statused."
tools:
  fs: false
mcp:
  servers:
    - name: oauth-server
      transport:
        type: http
        url: %s
      auth:
        type: oauth
      required: true
`, url)
	cfgPath := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(cfgPath, []byte(yaml), 0o600); err != nil { //nolint:gosec // G703: cfgPath is t.TempDir() + a constant name, not tainted input.
		t.Fatal(err)
	}

	// `status` is the read-only half of the CLI contract (docs/mcp.md, "The
	// three commands"): it never dials the authorization server, so this call
	// is safe to make with or without a stored credential and proves nothing
	// more than "the credential store and the config are both readable."
	var stdout, stderr bytes.Buffer
	if code := run(context.Background(), []string{"mcp", "status", cfgPath},
		strings.NewReader(""), &stdout, &stderr, os.LookupEnv); code != ExitOK {
		t.Fatalf("mcp status: exit %d\nstderr: %s", code, stderr.String())
	}
	status := stdout.String()
	t.Logf("mcp status:\n%s", status)
	if !strings.Contains(status, "oauth-server") {
		t.Fatalf("mcp status did not report the configured server:\n%s", status)
	}

	if os.Getenv("AMELE_MCP_OAUTH_ASSUME_LOGIN") != "1" {
		t.Skip("AMELE_MCP_OAUTH_ASSUME_LOGIN not set to 1: stopping after status " +
			"(run 'amele mcp login' by hand first, then rerun with the flag set)")
	}

	// `explain` connects for real (docs/mcp.md, "amele explain connects for
	// real") and, with a stored credential, exercises silent refresh at
	// connect time - the thing a fake authorization server can only simulate.
	stdout.Reset()
	stderr.Reset()
	if code := run(context.Background(), []string{"explain", cfgPath},
		strings.NewReader(""), &stdout, &stderr, os.LookupEnv); code != ExitOK {
		t.Fatalf("explain: exit %d\nstderr: %s", code, stderr.String())
	}
	report := stdout.String()
	t.Logf("explain:\n%s", report)
	if !strings.Contains(report, "connected") {
		t.Fatalf("explain did not report a connected server:\n%s", report)
	}
}
