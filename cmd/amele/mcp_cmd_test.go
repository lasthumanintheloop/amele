package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/mcp"
	"github.com/lasthumanintheloop/amele/internal/oauthtoken"
)

// mcpCLI is execCLI with a state directory of its own: the token store lives
// under XDG_STATE_HOME, so every `amele mcp` test gets an isolated one and
// nothing reaches the developer's real credentials.
func mcpCLI(t *testing.T, stateDir string, args []string, stdin string) (code int, stdout, stderr string) {
	t.Helper()
	var out, errBuf bytes.Buffer
	lookup := func(key string) (string, bool) {
		if key == "XDG_STATE_HOME" {
			return stateDir, true
		}
		return env(t)(key)
	}
	code = run(context.Background(), args, strings.NewReader(stdin), &out, &errBuf, lookup)
	return code, out.String(), errBuf.String()
}

// mcpTestConfig writes a config with one OAuth server ("github") and one
// plain stdio server ("plain"), plus the state directory the store uses.
func mcpTestConfig(t *testing.T) (cfgPath, stateDir string) {
	t.Helper()
	dir := t.TempDir()
	yaml := `
model: test-model
provider:
  base_url: http://127.0.0.1:1/v1
  api_key: ${TEST_KEY}
mcp:
  servers:
    - name: github
      transport: {type: http, url: https://mcp.example/mcp}
      auth: {type: oauth}
    - name: plain
      transport: {type: stdio, command: ["/bin/true"]}
`
	cfgPath = filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(cfgPath, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	return cfgPath, t.TempDir()
}

// seedCredential files one credential for the test config's github server.
func seedCredential(t *testing.T, stateDir string, mutate func(*oauthtoken.Record)) (*oauthtoken.Store, oauthtoken.Key) {
	t.Helper()
	resource, err := oauthtoken.CanonicalResource("https://mcp.example/mcp")
	if err != nil {
		t.Fatal(err)
	}
	store := oauthtoken.NewStore(filepath.Join(stateDir, "amele", "mcp"), time.Now)
	//nolint:gosec // G101: the fixture credential IS the test input; it exists to prove it never reaches the report.
	rec := &oauthtoken.Record{
		Version:       oauthtoken.Version,
		Issuer:        "https://as.example",
		Resource:      resource,
		ClientID:      "client-a",
		TokenEndpoint: "https://as.example/token",
		AccessToken:   "seeded-access-1",
		RefreshToken:  "seeded-refresh-1",
		ExpiresAt:     time.Now().Add(time.Hour),
		Scopes:        []string{"repo"},
	}
	if mutate != nil {
		mutate(rec)
	}
	key := oauthtoken.Key{Issuer: rec.Issuer, Resource: rec.Resource, ClientID: rec.ClientID}
	if err := store.Save(key, rec); err != nil {
		t.Fatalf("seeding the store: %v", err)
	}
	return store, key
}

// fakeTTY makes the TTY seam report an interactive terminal for one test.
func fakeTTY(t *testing.T, interactive bool) {
	t.Helper()
	prev := stdinIsTerminal
	stdinIsTerminal = func(io.Reader) bool { return interactive }
	t.Cleanup(func() { stdinIsTerminal = prev })
}

// TestMCPLoginRequiresTTY pins the gate: a login opens a browser and waits for
// a human, so a piped or cron stdin is refused outright rather than being
// treated as a refusal the way the permission system treats EOF.
func TestMCPLoginRequiresTTY(t *testing.T) {
	cfgPath, stateDir := mcpTestConfig(t)

	code, stdout, stderr := mcpCLI(t, stateDir, []string{"mcp", "login", cfgPath}, "")
	if code != ExitConfigError {
		t.Fatalf("exit %d, want %d; stderr: %s", code, ExitConfigError, stderr)
	}
	if !strings.Contains(stderr, "mcp login needs an interactive terminal") {
		t.Errorf("stderr = %q", stderr)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want nothing", stdout)
	}
}

// TestMCPLoginUnknownServer: a name no server answers to is a config error,
// reported before the TTY gate so the operator hears the real mistake.
func TestMCPLoginUnknownServer(t *testing.T) {
	cfgPath, stateDir := mcpTestConfig(t)

	code, _, stderr := mcpCLI(t, stateDir, []string{"mcp", "login", cfgPath, "nope"}, "")
	if code != ExitConfigError {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if !strings.Contains(stderr, "no such mcp server") {
		t.Errorf("stderr = %q", stderr)
	}
}

// TestMCPLoginServerWithoutAuth: naming a server that declares no oauth block
// is a config error too - there is nothing to log into.
func TestMCPLoginServerWithoutAuth(t *testing.T) {
	cfgPath, stateDir := mcpTestConfig(t)

	code, _, stderr := mcpCLI(t, stateDir, []string{"mcp", "login", cfgPath, "plain"}, "")
	if code != ExitConfigError {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if !strings.Contains(stderr, `server "plain" has no auth block`) {
		t.Errorf("stderr = %q", stderr)
	}
}

// TestMCPLoginSequential pins the two properties the CLI owns around the flow
// itself: every OAuth server is logged into IN CONFIG ORDER (three servers
// must not open three browsers at once), and each success prints one stderr
// line naming the server and the expiry.
func TestMCPLoginSequential(t *testing.T) {
	dir := t.TempDir()
	yaml := `
model: test-model
provider:
  base_url: http://127.0.0.1:1/v1
  api_key: ${TEST_KEY}
mcp:
  servers:
    - name: alpha
      transport: {type: http, url: https://alpha.example/mcp}
      auth: {type: oauth}
    - name: beta
      transport: {type: http, url: https://beta.example/mcp}
      auth: {type: oauth}
`
	cfgPath := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(cfgPath, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	fakeTTY(t, true)
	var order []string
	prev := loginToServer
	loginToServer = func(_ context.Context, cfg config.MCPServer, _ mcp.Deps, opts mcp.LoginOptions) (*oauthtoken.Record, error) {
		order = append(order, cfg.Name)
		if opts.Confirm != nil && !opts.Confirm("https://as.example", cfg.Transport.URL) {
			return nil, fmt.Errorf("declined")
		}
		return &oauthtoken.Record{ExpiresAt: time.Unix(1_800_000_000, 0).UTC()}, nil
	}
	t.Cleanup(func() { loginToServer = prev })

	code, stdout, stderr := mcpCLI(t, t.TempDir(), []string{"mcp", "login", cfgPath}, "y\ny\n")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if strings.Join(order, ",") != "alpha,beta" {
		t.Errorf("login order = %v, want config order", order)
	}
	for _, want := range []string{"mcp login ok: alpha", "mcp login ok: beta", "expires 2027-01-15"} {
		if !strings.Contains(stderr, want) {
			t.Errorf("stderr lacks %q: %s", want, stderr)
		}
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want nothing (login talks on stderr)", stdout)
	}
}

// TestMCPLoginDeclinedIsNotSuccess: answering anything but yes aborts, and the
// exit code says the task did not happen.
func TestMCPLoginDeclinedIsNotSuccess(t *testing.T) {
	cfgPath, stateDir := mcpTestConfig(t)
	fakeTTY(t, true)
	prev := loginToServer
	loginToServer = func(_ context.Context, cfg config.MCPServer, _ mcp.Deps, opts mcp.LoginOptions) (*oauthtoken.Record, error) {
		if opts.Confirm != nil && !opts.Confirm("https://as.example", cfg.Transport.URL) {
			return nil, fmt.Errorf("declined")
		}
		return &oauthtoken.Record{}, nil
	}
	t.Cleanup(func() { loginToServer = prev })

	code, _, stderr := mcpCLI(t, stateDir, []string{"mcp", "login", cfgPath}, "n\n")
	if code != ExitTaskFailed {
		t.Fatalf("exit %d, want %d; stderr: %s", code, ExitTaskFailed, stderr)
	}
	if !strings.Contains(stderr, "declined") {
		t.Errorf("stderr = %q", stderr)
	}
}

// TestMCPStatusNoToken: the report contract - a row on stdout, nothing on
// stderr, exit 0 even though there is no credential at all.
func TestMCPStatusNoToken(t *testing.T) {
	cfgPath, stateDir := mcpTestConfig(t)

	code, stdout, stderr := mcpCLI(t, stateDir, []string{"mcp", "status", cfgPath}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stderr != "" {
		t.Errorf("stderr = %q, want nothing (status is a report)", stderr)
	}
	if !strings.Contains(stdout, "github") || !strings.Contains(stdout, "no token") {
		t.Errorf("stdout = %q", stdout)
	}
	// A server without auth has nothing to report and must not appear.
	if strings.Contains(stdout, "plain") {
		t.Errorf("stdout names a non-oauth server: %q", stdout)
	}
}

// TestMCPStatusNeverPrintsToken is the SECURITY property of the report: it
// says whether a credential exists and when it expires, never what it is.
func TestMCPStatusNeverPrintsToken(t *testing.T) {
	cfgPath, stateDir := mcpTestConfig(t)
	seedCredential(t, stateDir, nil)

	code, stdout, stderr := mcpCLI(t, stateDir, []string{"mcp", "status", cfgPath}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	for _, secret := range []string{"seeded-access-1", "seeded-refresh-1"} {
		if strings.Contains(stdout+stderr, secret) {
			t.Errorf("output leaked %q: %s", secret, stdout)
		}
	}
	for _, want := range []string{"github", "ok", "refresh=yes", "scopes=repo", "issuer=https://as.example"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("stdout lacks %q: %s", want, stdout)
		}
	}
}

// TestMCPStatusExpired: an elapsed expiry reads as expired, and status still
// exits 0 - it reports, it does not gate.
func TestMCPStatusExpired(t *testing.T) {
	cfgPath, stateDir := mcpTestConfig(t)
	seedCredential(t, stateDir, func(r *oauthtoken.Record) {
		r.ExpiresAt = time.Now().Add(-time.Hour)
		r.RefreshToken = ""
	})

	code, stdout, stderr := mcpCLI(t, stateDir, []string{"mcp", "status", cfgPath}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if !strings.Contains(stdout, "expired") || !strings.Contains(stdout, "refresh=no") {
		t.Errorf("stdout = %q", stdout)
	}
}

// TestMCPLogoutDeletesLocally: with no revocation endpoint recorded there is
// nobody to tell, so the credential is simply forgotten - and the line says so.
func TestMCPLogoutDeletesLocally(t *testing.T) {
	cfgPath, stateDir := mcpTestConfig(t)
	store, key := seedCredential(t, stateDir, nil)

	code, stdout, stderr := mcpCLI(t, stateDir, []string{"mcp", "logout", cfgPath}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if _, err := store.Load(key); err == nil {
		t.Error("the credential is still on disk")
	}
	if !strings.Contains(stderr, "mcp logout: github (local only)") {
		t.Errorf("stderr = %q", stderr)
	}
	if stdout != "" {
		t.Errorf("stdout = %q, want nothing", stdout)
	}
}

// TestMCPLogoutRevokesWhenAdvertised: a record that carries the RFC 7009
// endpoint is handed back to the authorization server before it is deleted.
func TestMCPLogoutRevokesWhenAdvertised(t *testing.T) {
	cfgPath, stateDir := mcpTestConfig(t)
	store, key := seedCredential(t, stateDir, func(r *oauthtoken.Record) {
		r.RevocationEndpoint = "https://as.example/revoke"
	})
	var revoked []*oauthtoken.Record
	prev := revokeCredential
	revokeCredential = func(_ context.Context, rec *oauthtoken.Record) error {
		revoked = append(revoked, rec)
		return nil
	}
	t.Cleanup(func() { revokeCredential = prev })

	code, _, stderr := mcpCLI(t, stateDir, []string{"mcp", "logout", cfgPath}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if len(revoked) != 1 || revoked[0].RevocationEndpoint != "https://as.example/revoke" {
		t.Fatalf("revoke calls = %+v", revoked)
	}
	if revoked[0].RefreshToken != "seeded-refresh-1" {
		t.Errorf("revoked record = %+v, want the stored credential", revoked[0])
	}
	if !strings.Contains(stderr, "mcp logout: github (revoked)") {
		t.Errorf("stderr = %q", stderr)
	}
	if _, err := store.Load(key); err == nil {
		t.Error("the credential is still on disk")
	}
}

// TestMCPLogoutRevokeFailureStillDeletes: revocation is best effort. A server
// that refuses or is unreachable must not leave the credential on disk - the
// operator asked to be rid of it.
func TestMCPLogoutRevokeFailureStillDeletes(t *testing.T) {
	cfgPath, stateDir := mcpTestConfig(t)
	store, key := seedCredential(t, stateDir, func(r *oauthtoken.Record) {
		r.RevocationEndpoint = "https://as.example/revoke"
	})
	prev := revokeCredential
	revokeCredential = func(context.Context, *oauthtoken.Record) error {
		return fmt.Errorf("revocation endpoint returned 503 Service Unavailable")
	}
	t.Cleanup(func() { revokeCredential = prev })

	code, _, stderr := mcpCLI(t, stateDir, []string{"mcp", "logout", cfgPath}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if _, err := store.Load(key); err == nil {
		t.Error("the credential is still on disk")
	}
	if !strings.Contains(stderr, "warning") || !strings.Contains(stderr, "503") {
		t.Errorf("stderr lacks the revoke warning: %q", stderr)
	}
	if !strings.Contains(stderr, "mcp logout: github (local only)") {
		t.Errorf("stderr = %q", stderr)
	}
}

// TestMCPLogoutOneServer: naming a server logs out of that one only.
func TestMCPLogoutOneServer(t *testing.T) {
	cfgPath, stateDir := mcpTestConfig(t)
	store, key := seedCredential(t, stateDir, nil)

	code, _, stderr := mcpCLI(t, stateDir, []string{"mcp", "logout", cfgPath, "github"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if _, err := store.Load(key); err == nil {
		t.Error("the credential is still on disk")
	}
}

// TestMCPLogoutWithoutToken is a no-op that says so, and still exits 0:
// logging out of something already logged out is not an error.
func TestMCPLogoutWithoutToken(t *testing.T) {
	cfgPath, stateDir := mcpTestConfig(t)

	code, _, stderr := mcpCLI(t, stateDir, []string{"mcp", "logout", cfgPath}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if !strings.Contains(stderr, "mcp logout: github (no token)") {
		t.Errorf("stderr = %q", stderr)
	}
}

// TestMCPUnknownSubcommand and the arity cases: every malformed invocation is
// exit 2 with the usage line on stderr.
func TestMCPUnknownSubcommand(t *testing.T) {
	cfgPath, stateDir := mcpTestConfig(t)
	cases := [][]string{
		{"mcp"},
		{"mcp", "nope", cfgPath},
		{"mcp", "status"},
		{"mcp", "status", cfgPath, "extra"},
		{"mcp", "login", cfgPath, "github", "extra"},
	}
	for _, args := range cases {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			code, stdout, stderr := mcpCLI(t, stateDir, args, "")
			if code != ExitConfigError {
				t.Fatalf("exit %d, want %d", code, ExitConfigError)
			}
			if !strings.Contains(stderr, "usage: amele mcp") {
				t.Errorf("stderr = %q", stderr)
			}
			if stdout != "" {
				t.Errorf("stdout = %q", stdout)
			}
		})
	}
}

// TestMCPBadConfigIsExit2 keeps `mcp` in line with every other command: a
// config that will not load is exit 2, whichever subcommand asked.
func TestMCPBadConfigIsExit2(t *testing.T) {
	for _, sub := range []string{"login", "status", "logout"} {
		t.Run(sub, func(t *testing.T) {
			code, _, stderr := mcpCLI(t, t.TempDir(), []string{"mcp", sub, "/nope/agent.yaml"}, "")
			if code != ExitConfigError {
				t.Fatalf("exit %d, stderr: %s", code, stderr)
			}
		})
	}
}

// TestMCPNoOAuthServers: a config with nothing to log into is not an error,
// it is a report with nothing in it.
func TestMCPNoOAuthServers(t *testing.T) {
	srv := scriptedServer(t)
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	for _, sub := range []string{"status", "logout"} {
		code, stdout, stderr := mcpCLI(t, t.TempDir(), []string{"mcp", sub, cfgPath}, "")
		if code != ExitOK {
			t.Fatalf("%s: exit %d, stderr: %s", sub, code, stderr)
		}
		if sub == "status" && !strings.Contains(stdout, "no mcp server") {
			t.Errorf("status said nothing about an empty report: %q", stdout)
		}
	}
	code, _, stderr := mcpCLI(t, t.TempDir(), []string{"mcp", "login", cfgPath}, "")
	if code != ExitConfigError {
		t.Fatalf("login: exit %d, want %d; stderr: %s", code, ExitConfigError, stderr)
	}
}

// TestMCPHelp: the subcommand family owns a help page like every other
// command, and -h reaches it.
func TestMCPHelp(t *testing.T) {
	for _, args := range [][]string{{"help", "mcp"}, {"mcp", "-h"}, {"mcp", "--help"}} {
		code, stdout, stderr := execCLI(t, args, "")
		if code != ExitOK {
			t.Fatalf("%v: exit %d, stderr: %s", args, code, stderr)
		}
		if !strings.Contains(stdout, "amele mcp login") {
			t.Errorf("%v: stdout = %q", args, stdout)
		}
	}
}
