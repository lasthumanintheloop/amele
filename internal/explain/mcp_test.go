package explain

import (
	"flag"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/lasthumanintheloop/amele/internal/config"
)

// update rewrites the golden files instead of comparing against them. It is a
// deliberate, explicit act (docs/engineering.md §6): the report is a UI
// surface, and a diff a reviewer never sees is a contract change nobody
// approved.
var update = flag.Bool("update", false, "rewrite golden files")

// mcpYAML is the fixture behind the MCP tests: one http server whose
// Authorization header is composed from an ${ENV} credential, and one stdio
// server with a command and an env allowlist. It exercises every input the new
// section, the requirements rows and the redaction rule read.
const mcpYAML = `model: golden-model
provider:
  base_url: https://api.example.com/v1
  api_key: ${TEST_KEY}
limits:
  max_turns: 10
  max_tokens: 1000
mcp:
  servers:
    - name: github
      transport:
        type: http
        url: https://mcp.example.com/v1/mcp?session=abc
        headers:
          Authorization: Bearer ${GH_TOKEN}
    - name: files
      transport:
        type: stdio
        command: ["/opt/mcp/files-server", "--root", "/srv"]
        env: ["TZ", "HOME"]
`

// mcpTestEnv resolves the fixture's two variables. GH_TOKEN's value is the
// secret the redaction assertions hunt for.
func mcpTestEnv(key string) (string, bool) {
	switch key {
	case "TEST_KEY":
		return "sk-provider-key-value", true
	case "GH_TOKEN":
		return "ghp-token-value", true
	}
	return "", false
}

// loadMCPConfig loads mcpYAML the way the CLI would, so the config carries its
// interpolation bindings - a struct literal would have none, and the redaction
// assertions below would pass vacuously.
// It returns the config and the directory paths resolve against, which the
// golden normalizes away.
func loadMCPConfig(t *testing.T) (*config.Config, string) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(path, []byte(mcpYAML), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.LoadTolerant(path, mcpTestEnv)
	if err != nil {
		t.Fatal(err)
	}
	return cfg, dir
}

// mcpTestReports is the connect outcome the golden pins: one server that
// answered (three tools kept - one destructive, one whose name was rewritten -
// and one excluded by the filter), and one that did not. The failed server's
// error text quotes the composed Authorization header back at us, which is
// exactly what a real 401 body does and what the redaction must catch.
func mcpTestReports() []MCPServerReport {
	return []MCPServerReport{
		{
			Name: "github", Transport: "http", Target: "https://mcp.example.com/v1/mcp",
			Connected: true, DurationMS: 12, ProtocolVersion: "2025-06-18",
			ServerName: "gh", ServerVersion: "1.0",
			Auth:       "oauth",
			AuthStatus: "token valid until 2026-08-19T12:00:00Z, refresh: yes",
			Tools: []MCPToolReport{
				{Name: "github__create_issue", Kept: true, Bytes: 500, Hint: "destructive"},
				{Name: "github__list_issues", Kept: true, Bytes: 400, Hint: "read-only"},
				{Name: "github__a_b_1f2e3d4c", Original: "a.b", Normalized: true, Kept: true, Bytes: 334},
				{Name: "repo_delete", Reason: "excluded"},
			},
			TotalBytes: 1234, EstTokens: EstTokens(1234),
		},
		{
			Name: "files", Transport: "stdio", Target: "/opt/mcp/files-server",
			DurationMS: 30, ErrorClass: "auth",
			Auth:       "oauth",
			AuthStatus: "no token - run 'amele mcp login agent.yaml files'",
			Error:      "401 Unauthorized (sent Authorization: Bearer ghp-token-value)",
		},
	}
}

// TestRenderMCPGolden pins the whole report for a config with MCP servers: the
// MCP SERVERS block, the stdio command in the executables checklist and the
// server's env allowlist row. It is the golden docs/engineering.md §6 requires
// for a UI surface.
func TestRenderMCPGolden(t *testing.T) {
	cfg, dir := loadMCPConfig(t)
	// workspace defaults to the config's own (temporary) directory, so the
	// report is normalized the way cmd/amele's explain goldens are.
	got := strings.ReplaceAll(
		Render(cfg, registryWith(t, "fs_read"), nil, nil, alwaysFound, mcpTestReports()), dir, "<TMP>")

	goldenPath := filepath.Join("testdata", "golden", "explain-mcp.txt")
	if *update {
		if err := os.MkdirAll(filepath.Dir(goldenPath), 0o750); err != nil {
			t.Fatal(err)
		}
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

// TestRenderMCPRedactsHeaderSecret is the security regression: a server's
// error text is REMOTE input and a 401 body routinely echoes the credential it
// rejected. The composed header ("Bearer <token>") must be redacted as a
// whole, not just the ${ENV} value inside it.
func TestRenderMCPRedactsHeaderSecret(t *testing.T) {
	cfg, _ := loadMCPConfig(t)
	got := Render(cfg, registryWith(t, "fs_read"), nil, nil, alwaysFound, mcpTestReports())
	for _, secret := range []string{"ghp-token-value", "Bearer ghp-token-value"} {
		if strings.Contains(got, secret) {
			t.Errorf("secret %q leaked into report:\n%s", secret, got)
		}
	}
	if !strings.Contains(got, "[REDACTED]") {
		t.Errorf("report has no redaction marker where the header was:\n%s", got)
	}
}

// TestRenderWithoutMCPReportsHasNoSection keeps the feature silent for the
// configs that do not use it: a report with no servers must be byte-identical
// to the pre-feature one.
func TestRenderWithoutMCPReportsHasNoSection(t *testing.T) {
	got := Render(baseCfg(), registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil)
	if strings.Contains(got, "MCP SERVERS") {
		t.Errorf("MCP section rendered for a config without servers:\n%s", got)
	}
}

// TestMCPTokenWarning pins the cost warning: definitions ride along on every
// request, so a large toolset earns a line in WARNINGS. The threshold is
// judged on the TOTAL across servers, and a toolset under it stays silent.
func TestMCPTokenWarning(t *testing.T) {
	const wantLine = "mcp definitions ≈ 4500 tokens; consider tools.include to trim"
	reports := []MCPServerReport{
		{Name: "a", Transport: "http", Target: "https://a.example.com/mcp", Connected: true, EstTokens: 2500},
		{Name: "b", Transport: "http", Target: "https://b.example.com/mcp", Connected: true, EstTokens: 2000},
	}
	got := Render(baseCfg(), registryWith(t, fsBuiltins...), nil, nil, alwaysFound, reports)
	if !strings.Contains(got, wantLine) {
		t.Errorf("report missing warning %q:\n%s", wantLine, got)
	}

	// At the threshold exactly: no warning (the bound is "more than").
	reports[0].EstTokens, reports[1].EstTokens = mcpTokenWarnThreshold, 0
	quiet := Render(baseCfg(), registryWith(t, fsBuiltins...), nil, nil, alwaysFound, reports)
	if strings.Contains(quiet, "mcp definitions") {
		t.Errorf("warning raised at the threshold, want silence:\n%s", quiet)
	}
}

// TestMCPRequirements pins what an operator must provision for the servers:
// the stdio command[0] joins the executables checklist (a missing `npx` in a
// cron PATH is the classic unattended failure) and the server's env allowlist
// gets a row of its own, labelled so it cannot be mistaken for a tool.
func TestMCPRequirements(t *testing.T) {
	cfg, _ := loadMCPConfig(t)
	missing := func(string) error { return os.ErrNotExist }
	got := requirementsReport(cfg, missing)
	for _, want := range []string{
		`"/opt/mcp/files-server"   ✗ MISSING`,
		`"mcp:files"     "TZ", "HOME"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("requirements missing %q:\n%s", want, got)
		}
	}
}

// TestMCPSectionQuotesUntrustedStrings is the row-forgery regression. A tool
// name, a server's self-reported name and an error message all originate on
// the far side of the connection; a newline in any of them would invent report
// rows, and the report is line-oriented so stripping newlines is not an
// option.
func TestMCPSectionQuotesUntrustedStrings(t *testing.T) {
	reports := []MCPServerReport{{
		Name: "evil", Transport: "http", Target: "https://evil.example.com/mcp",
		Connected: true, DurationMS: 1, ProtocolVersion: "2025-06-18",
		ServerName: "srv\nWARNINGS", ServerVersion: "1.0",
		Tools: []MCPToolReport{
			{Name: "evil__ok\n    fake__row", Kept: true, Bytes: 10},
		},
		TotalBytes: 10, EstTokens: EstTokens(10),
	}}
	got := Render(baseCfg(), registryWith(t, fsBuiltins...), nil, nil, alwaysFound, reports)
	for _, line := range strings.Split(got, "\n") {
		if line == "fake__row" || strings.HasPrefix(line, "    fake__row") {
			t.Errorf("a tool name forged a report row:\n%s", got)
		}
	}
	if strings.Count(got, "WARNINGS\n") != 1 {
		t.Errorf("a server name forged a section header:\n%s", got)
	}
}

// TestMCPFailedServerHasNoDefinitionsLine keeps the arithmetic honest: a
// server that never answered contributed no tools, and printing "0 tools, 0
// bytes" for it would read as "this server offers nothing" rather than "this
// server was not reached".
func TestMCPFailedServerHasNoDefinitionsLine(t *testing.T) {
	reports := []MCPServerReport{{
		Name: "down", Transport: "stdio", Target: "/opt/mcp/down",
		ErrorClass: "spawn", Error: "exec: \"down\": executable file not found in $PATH",
	}}
	got := Render(baseCfg(), registryWith(t, fsBuiltins...), nil, nil, alwaysFound, reports)
	if strings.Contains(got, "definitions:") {
		t.Errorf("failed server got a definitions line:\n%s", got)
	}
	if !strings.Contains(got, "✗ FAILED (spawn):") {
		t.Errorf("failed server not marked:\n%s", got)
	}
}

// TestMCPAuthRow pins the credential line: it names the mechanism and the
// state of the stored token, it appears for a failed server too (a connect
// that was refused is where the reader most needs to know whether a token is
// even stored), and it is omitted entirely for a server with no auth block.
func TestMCPAuthRow(t *testing.T) {
	cfg, _ := loadMCPConfig(t)
	got := Render(cfg, registryWith(t, "fs_read"), nil, nil, alwaysFound, mcpTestReports())
	for _, want := range []string{
		"    auth: oauth (token valid until 2026-08-19T12:00:00Z, refresh: yes)\n",
		"    auth: oauth (no token - run 'amele mcp login agent.yaml files')\n",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("report does not carry %q:\n%s", want, got)
		}
	}

	plain := mcpTestReports()
	plain[0].Auth, plain[0].AuthStatus = "", ""
	plain[1].Auth, plain[1].AuthStatus = "", ""
	if out := Render(cfg, registryWith(t, "fs_read"), nil, nil, alwaysFound, plain); strings.Contains(out, "auth:") {
		t.Errorf("a server without an auth block still got a credential line:\n%s", out)
	}
}

// TestMCPAuthRowIsFlattened is the forgery regression: the status text is
// assembled from a record whose issuer and scopes an authorization server
// chose, so a newline in it must not be able to write a report row of its own.
func TestMCPAuthRowIsFlattened(t *testing.T) {
	cfg, _ := loadMCPConfig(t)
	reports := mcpTestReports()
	reports[0].AuthStatus = "ok\n    \"evil\" http \"x\": ✓ connected"
	got := Render(cfg, registryWith(t, "fs_read"), nil, nil, alwaysFound, reports)
	if strings.Contains(got, "\n    \"evil\"") {
		t.Errorf("the auth status forged a row:\n%s", got)
	}
}
