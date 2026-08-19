package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/mcp"
	"github.com/lasthumanintheloop/amele/internal/session"
)

// The compiled MCP test server is shared by every test in this file: `go build`
// is by far the slowest thing here, and the binary is stateless.
var (
	mcpBuildDir  string
	mcpBuildOnce sync.Once
	mcpBinPath   string
	mcpBuildErr  error
)

// TestMain builds nothing eagerly; it only removes the directory the MCP test
// server was compiled into, since a package-level temp dir outlives t.Cleanup.
func TestMain(m *testing.M) {
	code := m.Run()
	if mcpBuildDir != "" {
		_ = os.RemoveAll(mcpBuildDir)
	}
	os.Exit(code)
}

// buildMCPTestServer compiles internal/mcp/testdata/mcptestserver once and
// returns the binary's path. It is the same server the internal/mcp tests use
// (echo/env/sleep/big/fail/structured), reached here by relative path because
// testdata is deliberately never linked into the amele binary.
func buildMCPTestServer(t *testing.T) string {
	t.Helper()
	mcpBuildOnce.Do(func() {
		dir, err := os.MkdirTemp("", "amele-cmd-mcptestserver")
		if err != nil {
			mcpBuildErr = err
			return
		}
		mcpBuildDir = dir
		bin := filepath.Join(dir, "mcptestserver")
		out, err := exec.Command("go", "build", "-o", bin, //nolint:gosec // bin is a path this test just created
			"../../internal/mcp/testdata/mcptestserver").CombinedOutput()
		if err != nil {
			mcpBuildErr = fmt.Errorf("go build: %w: %s", err, out)
			return
		}
		mcpBinPath = bin
	})
	if mcpBuildErr != nil {
		t.Fatalf("building mcp test server: %v", mcpBuildErr)
	}
	return mcpBinPath
}

// stdioServerYAML renders an `mcp:` block for one stdio server.
func stdioServerYAML(name, command, extra string) string {
	return fmt.Sprintf(`mcp:
  servers:
    - name: %s
      transport:
        type: stdio
        command: [%q]
%s`, name, command, extra)
}

// assertListedTool fails unless the single mcp_tools_listed event advertises
// the named model-facing tool.
func assertListedTool(t *testing.T, events []session.Event, name string) {
	t.Helper()
	listed := eventsOfType(events, "mcp_tools_listed")
	if len(listed) != 1 {
		t.Fatalf("mcp_tools_listed events: %d", len(listed))
	}
	for _, tool := range listed[0].Tools {
		if tool.Name == name {
			return
		}
	}
	t.Errorf("%s not listed: %+v", name, listed[0].Tools)
}

// assertBefore fails unless every event of type early precedes every event of
// type late.
func assertBefore(t *testing.T, events []session.Event, early, late string) {
	t.Helper()
	lastEarly, firstLate := -1, -1
	for i, e := range events {
		switch e.Type {
		case early:
			lastEarly = i
		case late:
			if firstLate < 0 {
				firstLate = i
			}
		}
	}
	if lastEarly < 0 || firstLate < 0 {
		t.Fatalf("expected both %s and %s in the log, got %d/%d", early, late, lastEarly, firstLate)
	}
	if lastEarly > firstLate {
		t.Errorf("last %s is at index %d, after the first %s at %d", early, lastEarly, late, firstLate)
	}
}

// eventsOfType returns every logged event of one type.
func eventsOfType(events []session.Event, typ string) []session.Event {
	var out []session.Event
	for _, e := range events {
		if e.Type == typ {
			out = append(out, e)
		}
	}
	return out
}

// TestRunMCPStdioToolCall is the happy path: a stdio server's tool is
// registered under its prefixed name, the model calls it, and the session log
// carries the whole MCP lifecycle between run_start and run_end.
func TestRunMCPStdioToolCall(t *testing.T) {
	bin := buildMCPTestServer(t)
	srv := scriptedServer(t,
		toolCallBody("files__echo", `{"text":"hi"}`),
		textBody("done"),
	)
	cfgPath, dir := writeTestConfig(t, srv.URL,
		"session_dir: sessions\n"+stdioServerYAML("files", bin, ""))

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "echo hi"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "done\n" {
		t.Errorf("stdout: %q", stdout)
	}

	events := readSessionEvents(t, dir)
	if events[0].Type != "run_start" {
		t.Fatalf("first event is %q, want run_start", events[0].Type)
	}
	connects := eventsOfType(events, "mcp_connect")
	if len(connects) != 1 || connects[0].OK == nil || !*connects[0].OK {
		t.Fatalf("mcp_connect: %+v", connects)
	}
	assertListedTool(t, events, "files__echo")
	// CONTRACT (docs/contracts/jsonl-events.md, Ordering): the connects and the
	// tool listings happen before the first llm_response of the run - the model
	// must never have been asked anything with a half-built toolset.
	assertBefore(t, events, "mcp_connect", "llm_response")
	assertBefore(t, events, "mcp_tools_listed", "llm_response")
	results := eventsOfType(events, "tool_result")
	if len(results) != 1 || results[0].Outcome != session.OutcomeOK {
		t.Fatalf("tool_result: %+v", results)
	}
	disconnects := eventsOfType(events, "mcp_disconnect")
	if len(disconnects) != 1 || disconnects[0].Reason != "run_end" {
		t.Fatalf("mcp_disconnect: %+v", disconnects)
	}
	last := events[len(events)-1]
	if last.Type != "run_end" {
		t.Fatalf("last event is %q, want run_end", last.Type)
	}
}

// TestRunMCPRequiredUnavailableExit8 pins the fail-fast contract: a required
// server that cannot be started aborts the run with exit 8, and the session
// file still ends with a truthful run_end.
func TestRunMCPRequiredUnavailableExit8(t *testing.T) {
	srv := scriptedServer(t)
	cfgPath, dir := writeTestConfig(t, srv.URL,
		"session_dir: sessions\n"+stdioServerYAML("files", "/nonexistent/amele-mcp-server", ""))

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "task"}, "")
	if code != ExitMCPUnavailable {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "" {
		t.Errorf("stdout: %q", stdout)
	}
	if !strings.Contains(stderr, `mcp server "files"`) || !strings.Contains(stderr, "spawn") {
		t.Errorf("stderr does not name the server and the failure class: %q", stderr)
	}

	events := readSessionEvents(t, dir)
	connects := eventsOfType(events, "mcp_connect")
	if len(connects) != 1 || connects[0].OK == nil || *connects[0].OK {
		t.Fatalf("mcp_connect: %+v", connects)
	}
	if connects[0].ErrorClass != "spawn" {
		t.Errorf("error_class: %q", connects[0].ErrorClass)
	}
	last := events[len(events)-1]
	if last.Type != "run_end" || last.ExitCode == nil || *last.ExitCode != ExitMCPUnavailable {
		t.Fatalf("run_end: %+v", last)
	}
	// The missing REQUIRED dependency is an MCP failure like any other: an
	// exit-8 run_end saying "0 mcp errors" would contradict its own exit code.
	if last.MCPErrors != 1 {
		t.Errorf("run_end.mcp_errors = %d, want 1", last.MCPErrors)
	}
}

// TestRunMCPOptionalUnavailableContinues pins the opt-out: `required: false`
// degrades the run instead of ending it, but the loss is on stderr and counted
// in run_end.mcp_errors.
func TestRunMCPOptionalUnavailableContinues(t *testing.T) {
	srv := scriptedServer(t, textBody("answered without the server"))
	cfgPath, dir := writeTestConfig(t, srv.URL,
		"session_dir: sessions\n"+
			stdioServerYAML("files", "/nonexistent/amele-mcp-server", "      required: false\n"))

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "task"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "answered without the server\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if !strings.Contains(stderr, "warning: mcp server") {
		t.Errorf("no degradation warning: %q", stderr)
	}

	events := readSessionEvents(t, dir)
	last := events[len(events)-1]
	if last.Type != "run_end" || last.MCPErrors != 1 {
		t.Fatalf("run_end: %+v", last)
	}
}

// TestRunMCPOptionalUnavailableQuiet checks that -q drops the warning: a quiet
// cron run stays silent while it is healthy, and the count is still logged.
func TestRunMCPOptionalUnavailableQuiet(t *testing.T) {
	srv := scriptedServer(t, textBody("ok"))
	cfgPath, _ := writeTestConfig(t, srv.URL,
		stdioServerYAML("files", "/nonexistent/amele-mcp-server", "      required: false\n"))

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "-q", "task"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stderr != "" {
		t.Errorf("quiet run wrote to stderr: %q", stderr)
	}
}

// TestRunMCPNameCollisionExit2 pins the toolset contract: a server whose
// prefixed tool name is already taken is a config error (exit 2), not a
// transient failure - retrying could never fix it.
func TestRunMCPNameCollisionExit2(t *testing.T) {
	bin := buildMCPTestServer(t)
	srv := scriptedServer(t)
	// The subprocess block continues the `tools:` mapping writeTestConfig
	// already opened, so the config has exactly one of each top-level key.
	cfgPath, _ := writeTestConfig(t, srv.URL, `  subprocess:
    - name: files__echo
      description: collides with the MCP tool
      command: ["/bin/true"]
`+stdioServerYAML("files", bin, ""))

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "task"}, "")
	if code != ExitConfigError {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if !strings.Contains(stderr, "files__echo") {
		t.Errorf("stderr does not name the colliding tool: %q", stderr)
	}
}

// TestRunMCPPermissionsGlobAsk checks that MCP tools obey the permission
// profile like any other tool: a glob covering the server's prefix with `ask`
// and no TTY auto-denies, and the run continues.
func TestRunMCPPermissionsGlobAsk(t *testing.T) {
	bin := buildMCPTestServer(t)
	srv := scriptedServer(t,
		toolCallBody("files__echo", `{"text":"hi"}`),
		textBody("denied, moving on"),
	)
	cfgPath, dir := writeTestConfig(t, srv.URL, `session_dir: sessions
permissions:
  tools:
    "files__*": ask
`+stdioServerYAML("files", bin, ""))

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "task"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "denied, moving on\n" {
		t.Errorf("stdout: %q", stdout)
	}
	events := readSessionEvents(t, dir)
	results := eventsOfType(events, "tool_result")
	if len(results) != 1 || results[0].Outcome != session.OutcomeDeniedNoTTY {
		t.Fatalf("tool_result: %+v", results)
	}
}

// TestRunMCPHTTPHeaderRedacted pins the secret contract for HTTP servers: an
// endpoint that rejects the credential ends the run with exit 8, and the token
// itself appears in no output channel.
func TestRunMCPHTTPHeaderRedacted(t *testing.T) {
	mcpSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	}))
	t.Cleanup(mcpSrv.Close)

	srv := scriptedServer(t)
	// DB_PASSWORD, not TEST_KEY: the API key is redacted by a list that
	// predates MCP, so a header built from it could not prove anything about
	// cfg.MCPHeaderSecrets.
	cfgPath, dir := writeTestConfig(t, srv.URL, fmt.Sprintf(`session_dir: sessions
mcp:
  servers:
    - name: remote
      transport:
        type: http
        url: %s
        headers:
          Authorization: "Bearer ${DB_PASSWORD}"
`, mcpSrv.URL))

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "task"}, "")
	if code != ExitMCPUnavailable {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	// Both halves of the promise: the environment value itself, and the
	// COMPOSED header - the string cfg.MCPHeaderSecrets exists to register.
	leaks := []string{testInterpolatedValue, "Bearer " + testInterpolatedValue}
	for _, leak := range leaks {
		if strings.Contains(stderr, leak) {
			t.Errorf("stderr leaked %q: %q", leak, stderr)
		}
	}
	files, err := filepath.Glob(filepath.Join(dir, "sessions", "*.jsonl"))
	if err != nil || len(files) != 1 {
		t.Fatalf("session files: %v, %v", files, err)
	}
	data, err := os.ReadFile(files[0])
	if err != nil {
		t.Fatal(err)
	}
	for _, leak := range leaks {
		if strings.Contains(string(data), leak) {
			t.Errorf("session log leaked %q:\n%s", leak, data)
		}
	}
}

// TestChatConnectsMCP checks that `chat` gets the same MCP wiring as `run`,
// including the disconnect before the session's run_end.
func TestChatConnectsMCP(t *testing.T) {
	bin := buildMCPTestServer(t)
	srv := scriptedServer(t)
	cfgPath, dir := writeTestConfig(t, srv.URL,
		"session_dir: sessions\n"+stdioServerYAML("files", bin, ""))

	code, _, stderr := execCLI(t, []string{"chat", cfgPath}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	events := readSessionEvents(t, dir)
	if events[0].Type != "run_start" {
		t.Fatalf("first event is %q, want run_start", events[0].Type)
	}
	connects := eventsOfType(events, "mcp_connect")
	if len(connects) != 1 || connects[0].OK == nil || !*connects[0].OK {
		t.Fatalf("mcp_connect: %+v", connects)
	}
	disconnects := eventsOfType(events, "mcp_disconnect")
	if len(disconnects) != 1 || disconnects[0].Reason != "run_end" {
		t.Fatalf("mcp_disconnect: %+v", disconnects)
	}
	if last := events[len(events)-1]; last.Type != "run_end" {
		t.Fatalf("last event is %q, want run_end", last.Type)
	}
}

// TestRunMCPInterruptedDuringConnect pins the signal path through the connect
// window: a run cancelled while its servers are coming up is an interrupted
// run (exit 1), not a missing dependency (exit 8), and the session log still
// ends with run_end.
func TestRunMCPInterruptedDuringConnect(t *testing.T) {
	bin := buildMCPTestServer(t)
	srv := scriptedServer(t)
	cfgPath, dir := writeTestConfig(t, srv.URL,
		"session_dir: sessions\n"+stdioServerYAML("files", bin, ""))

	// Cancelled up front: with a connect that can only fail, this is the same
	// observable state as a SIGTERM landing mid-handshake, and it needs no
	// polling to be deterministic.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	var stdout, stderr bytes.Buffer
	code := run(ctx, []string{"run", cfgPath, "task"}, strings.NewReader(""), &stdout, &stderr, env(t))
	if code != ExitTaskFailed {
		t.Fatalf("exit %d, want %d; stderr: %s", code, ExitTaskFailed, stderr.String())
	}
	if !strings.Contains(stderr.String(), "run interrupted") {
		t.Errorf("stderr must say the run was interrupted: %q", stderr.String())
	}
	events := readSessionEvents(t, dir)
	if last := events[len(events)-1]; last.Type != "run_end" {
		t.Fatalf("last event is %q, want run_end", last.Type)
	}
}

// TestMCPStderrRelay pins the three properties of the stdio stderr relay:
// secrets are redacted, terminal control bytes are stripped, and the whole
// relay is bounded with the drop announced.
func TestMCPStderrRelay(t *testing.T) {
	var mu sync.Mutex
	var out bytes.Buffer
	relay := &mcpStderr{mu: &mu, w: &out, prefix: "amele: mcp files: ",
		redact: session.Redactor([]string{"sk-secret"}), budget: 100}

	// Split across writes and lacking a trailing newline: the relay must
	// reassemble the line and flush the remainder at shutdown.
	_, _ = relay.Write([]byte("starting with sk-se"))
	_, _ = relay.Write([]byte("cret\nforged\x1b[2K\rquestion\nlast"))
	relay.flush()

	got := out.String()
	if strings.Contains(got, "sk-secret") {
		t.Errorf("relay leaked the secret: %q", got)
	}
	if strings.Contains(got, "\x1b") {
		t.Errorf("relay passed an escape sequence through: %q", got)
	}
	if !strings.Contains(got, "amele: mcp files: last") {
		t.Errorf("relay dropped the unterminated last line: %q", got)
	}

	// Past the budget every further line is dropped, once and audibly.
	for range 10 {
		_, _ = relay.Write([]byte("noise noise noise noise noise\n"))
	}
	if n := strings.Count(out.String(), "(further output dropped)"); n != 1 {
		t.Errorf("drop announced %d times, want 1:\n%s", n, out.String())
	}
}

// TestAnnotationMap checks that an unstated hint stays absent while an
// explicit false is recorded: they are different facts to an operator.
func TestAnnotationMap(t *testing.T) {
	if got := annotationMap(mcp.Annotations{}); got != nil {
		t.Errorf("no annotations should map to nil, got %v", got)
	}
	no, yes := false, true
	got := annotationMap(mcp.Annotations{ReadOnly: &no, Destructive: &yes})
	want := map[string]bool{"readOnly": false, "destructive": true}
	if !maps.Equal(got, want) {
		t.Errorf("annotationMap = %v, want %v", got, want)
	}
}

// TestAgentSecretsIncludesComposedMCPHeader pins the reason
// cfg.MCPHeaderSecrets() is part of agentSecrets: the header is a COMPOSED
// value ("Bearer " + the environment's), and only registering that whole string
// keeps a redactor honest that does not happen to match on substrings.
//
// It is asserted white-box because it cannot be observed from the outside:
// session.Redactor replaces substrings, so the environment value alone (which
// InterpolatedSecrets has always carried) already blanks the composed string in
// every current sink. See the fix report for the mutation experiment.
func TestAgentSecretsIncludesComposedMCPHeader(t *testing.T) {
	srv := scriptedServer(t)
	cfgPath, _ := writeTestConfig(t, srv.URL, `mcp:
  servers:
    - name: remote
      transport:
        type: http
        url: https://example.invalid/mcp
        headers:
          Authorization: "Bearer ${DB_PASSWORD}"
`)
	cfg, err := config.Load(cfgPath, env(t))
	if err != nil {
		t.Fatalf("loading config: %v", err)
	}
	want := "Bearer " + testInterpolatedValue
	if !slices.Contains(agentSecrets(cfg), want) {
		t.Errorf("agentSecrets does not register the composed header %q: %q", want, agentSecrets(cfg))
	}
}

// TestRunMCPInterruptedOptionalNotCounted pins the other half of the signal
// path: an interruption fails every connect at once, and those failures belong
// to the interruption, not to the servers. An operator reading the log of a
// SIGTERMed cron job must not find "mcp_errors: N" for a healthy fleet.
func TestRunMCPInterruptedOptionalNotCounted(t *testing.T) {
	bin := buildMCPTestServer(t)
	srv := scriptedServer(t)
	cfgPath, dir := writeTestConfig(t, srv.URL,
		"session_dir: sessions\n"+stdioServerYAML("files", bin, "      required: false\n"))

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	var stdout, stderr bytes.Buffer
	code := run(ctx, []string{"run", cfgPath, "task"}, strings.NewReader(""), &stdout, &stderr, env(t))
	if code != ExitTaskFailed {
		t.Fatalf("exit %d, want %d; stderr: %s", code, ExitTaskFailed, stderr.String())
	}
	if strings.Contains(stderr.String(), "warning: mcp server") {
		t.Errorf("interruption reported as a degraded server: %q", stderr.String())
	}
	events := readSessionEvents(t, dir)
	last := events[len(events)-1]
	if last.Type != "run_end" {
		t.Fatalf("last event is %q, want run_end", last.Type)
	}
	if last.MCPErrors != 0 {
		t.Errorf("run_end.mcp_errors = %d, want 0 (the interruption caused the failure)", last.MCPErrors)
	}
}

// TestExplainMCPReportsAndExitsZero is the CONTRACT test for the report: an
// `explain` whose config names a live server and a REQUIRED one that cannot be
// started still exits 0, describes both, and lists the live server's tools -
// the same failure that aborts `amele run` with exit 8.
func TestExplainMCPReportsAndExitsZero(t *testing.T) {
	bin := buildMCPTestServer(t)
	cfgPath, dir := writeTestConfig(t, "https://api.example.com", fmt.Sprintf(`session_dir: sessions
mcp:
  servers:
    - name: files
      transport:
        type: stdio
        command: [%q]
    - name: dead
      transport:
        type: stdio
        command: ["/nonexistent/amele-mcp-server"]
`, bin))

	code, stdout, stderr := execCLI(t, []string{"explain", cfgPath}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	// CONTRACT (docs/contracts/cli.md): stderr stays empty whenever the report
	// was printed - a server's stderr must not leak into it.
	if stderr != "" {
		t.Errorf("stderr: %q", stderr)
	}
	for _, want := range []string{
		"MCP SERVERS",
		`"files"        stdio`,
		"✓ connected",
		`"files__echo"`,
		"definitions:",
		`"dead"         stdio`,
		"✗ FAILED (spawn)",
	} {
		if !strings.Contains(stdout, want) {
			t.Errorf("report missing %q:\n%s", want, stdout)
		}
	}
	// explain writes no audit trail: session_dir is set, and dialling servers
	// must not have turned the report into a logged run.
	if _, err := os.Stat(filepath.Join(dir, "sessions")); !os.IsNotExist(err) {
		t.Errorf("explain created a session directory (err=%v)", err)
	}
}

// TestExplainMCPRequirements pins the checklist half of the feature: a stdio
// server's command[0] is an executable the host must provide, and its env
// allowlist is a capability grant - both belong in REQUIREMENTS, where the
// operator reads what to provision before the first cron run.
func TestExplainMCPRequirements(t *testing.T) {
	cfgPath, _ := writeTestConfig(t, "https://api.example.com", `mcp:
  servers:
    - name: files
      transport:
        type: stdio
        command: ["/nonexistent/amele-mcp-server"]
        env: ["TZ"]
`)
	code, stdout, stderr := execCLI(t, []string{"explain", cfgPath}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	for _, want := range []string{
		`"/nonexistent/amele-mcp-server" ✗ MISSING`,
		`"mcp:files"     "TZ"`,
	} {
		if !strings.Contains(stdout, want) {
			t.Errorf("report missing %q:\n%s", want, stdout)
		}
	}
}

// TestConnectFailedRedactsWarning pins the redaction of the degradation
// warning: a connect error may quote a header value the config interpolated,
// and the warning lands in cron mail.
func TestConnectFailedRedactsWarning(t *testing.T) {
	var buf bytes.Buffer
	set := &mcpSet{redact: session.Redactor([]string{"s3cret-token"})}
	err := &mcp.ConnectError{Server: "github", Class: mcp.ClassAuth,
		Err: errors.New("401 Unauthorized: bad token s3cret-token")}
	cfg := config.MCPServer{Name: "github", Required: new(bool)} // required: false
	set.connectFailed(cfg, err, false, false, &buf, func(error) {})

	out := buf.String()
	if strings.Contains(out, "s3cret-token") {
		t.Errorf("warning leaked the secret: %q", out)
	}
	if !strings.Contains(out, "[REDACTED]") || !strings.Contains(out, "auth") {
		t.Errorf("warning lost its content: %q", out)
	}
	if set.failed != 1 {
		t.Errorf("failed = %d, want 1", set.failed)
	}
}

// TestConnectFailedCountsRequired pins fix: a required server's connect
// failure is counted toward mcp_errors (unless the run was interrupted, which
// is nobody's unavailability).
func TestConnectFailedCountsRequired(t *testing.T) {
	err := &mcp.ConnectError{Server: "files", Class: mcp.ClassSpawn, Err: errors.New("no such file")}
	for _, tc := range []struct {
		name        string
		interrupted bool
		want        int
	}{
		{"required failure counts", false, 1},
		{"interrupted does not", true, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			set := &mcpSet{}
			var fatal error
			set.connectFailed(config.MCPServer{Name: "files"}, err, true, tc.interrupted, io.Discard, func(e error) { fatal = e })
			if fatal == nil {
				t.Fatal("required failure did not fail the run")
			}
			if set.failed != tc.want {
				t.Errorf("failed = %d, want %d", set.failed, tc.want)
			}
		})
	}
}

// TestMCPTargetStripsUserinfo: explain's target column must not print a
// credential embedded in the URL, even on a config Validate rejected.
func TestMCPTargetStripsUserinfo(t *testing.T) {
	//nolint:gosec // G101: the fake credential IS the test input
	s := config.MCPServer{Transport: config.MCPTransport{
		Type: config.MCPTransportHTTP, URL: "https://user:hunter2@example.com/mcp?key=abc#frag"}}
	got := mcpTarget(s)
	if strings.Contains(got, "hunter2") || strings.Contains(got, "abc") {
		t.Errorf("target leaked credentials: %q", got)
	}
	if got != "https://example.com/mcp" {
		t.Errorf("target = %q, want https://example.com/mcp", got)
	}
}

// TestConnectMCPSharesLiveSecretSet: the MCP set must redact through the
// caller's registry, not a snapshot of it. A token minted after the connect
// phase (OAuth) has to be scrubbed from the set's operator-facing lines - the
// degradation warning and the stderr relays - without rewiring anything.
func TestConnectMCPSharesLiveSecretSet(t *testing.T) {
	secrets := session.NewSecretSet(nil)
	noEnv := func(string) (string, bool) { return "", false }
	set, err := connectMCP(context.Background(), &config.Config{}, nil, nil,
		io.Discard, noEnv, true, "test", secrets)
	if err != nil {
		t.Fatal(err)
	}
	if set.secrets != secrets {
		t.Errorf("connectMCP built its own secret set; the run must share exactly one")
	}
	secrets.Add("late-token")
	if got := set.scrub("server said late-token"); strings.Contains(got, "late-token") {
		t.Errorf("mid-run secret leaked through the MCP set: %q", got)
	}
}

// TestExplainMCPHasSecretSet: `explain` is its own invocation with no session
// log to share with, but it still owns exactly one registry so an OAuth flow
// started for the report has somewhere to register its token.
func TestExplainMCPHasSecretSet(t *testing.T) {
	_, _, set := explainMCP(context.Background(), &config.Config{}, nil,
		func(string) (string, bool) { return "", false }, "test")
	if set.secrets == nil {
		t.Fatal("explainMCP left the set without a secret registry")
	}
	set.secrets.Add("late-token")
	if got := set.scrub("banner late-token"); strings.Contains(got, "late-token") {
		t.Errorf("mid-report secret leaked: %q", got)
	}
}

func TestDedupRegistrarPassesEachValueOnce(t *testing.T) {
	var mu sync.Mutex
	var got []string
	register := dedupRegistrar(func(values ...string) {
		mu.Lock()
		defer mu.Unlock()
		got = append(got, values...)
	})

	var wg sync.WaitGroup
	for range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			register("at1", "rt1", "")
		}()
	}
	wg.Wait()
	register("at2", "rt1")

	mu.Lock()
	defer mu.Unlock()
	want := []string{"at1", "rt1", "at2"}
	if len(got) != len(want) {
		t.Fatalf("registered %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("registered %v, want %v", got, want)
		}
	}
}
