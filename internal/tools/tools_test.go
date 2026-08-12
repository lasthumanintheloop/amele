package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
)

// newFS builds the fs tools over a fresh temp workspace and returns both.
func newFS(t *testing.T, opts FSOptions) (string, map[string]Tool) {
	t.Helper()
	dir := t.TempDir()
	list, err := NewFSTools(dir, opts)
	if err != nil {
		t.Fatal(err)
	}
	byName := map[string]Tool{}
	for _, tool := range list {
		byName[tool.Def().Name] = tool
	}
	return dir, byName
}

func TestRegistry(t *testing.T) {
	r := NewRegistry()
	_, fs := newFS(t, FSOptions{})
	if err := r.Register(fs["fs_read"]); err != nil {
		t.Fatal(err)
	}
	if err := r.Register(fs["fs_read"]); err == nil {
		t.Error("duplicate registration must fail")
	}
	if _, ok := r.Get("fs_read"); !ok {
		t.Error("Get should find registered tool")
	}
	if _, ok := r.Get("nope"); ok {
		t.Error("Get should miss unknown tool")
	}
	if defs := r.Defs(); len(defs) != 1 || defs[0].Name != "fs_read" {
		t.Errorf("Defs: %+v", defs)
	}
}

func TestFSReadWriteList(t *testing.T) {
	ctx := context.Background()
	_, fs := newFS(t, FSOptions{})

	// write → read round trip
	out, err := fs["fs_write"].Invoke(ctx, `{"path": "notes/hello.txt", "content": "merhaba"}`)
	if err != nil {
		t.Fatalf("fs_write: %v", err)
	}
	if !strings.Contains(out, "7 bytes") {
		t.Errorf("write result: %q", out)
	}
	got, err := fs["fs_read"].Invoke(ctx, `{"path": "notes/hello.txt"}`)
	if err != nil || got != "merhaba" {
		t.Errorf("fs_read: %q, %v", got, err)
	}

	// list root: file entries plain, directories with trailing slash
	listing, err := fs["fs_list"].Invoke(ctx, `{}`)
	if err != nil || listing != "notes/" {
		t.Errorf("fs_list: %q, %v", listing, err)
	}
	// empty args string must behave like "{}"
	if _, err := fs["fs_list"].Invoke(ctx, ""); err != nil {
		t.Errorf("fs_list empty args: %v", err)
	}
}

func TestFSReadTruncation(t *testing.T) {
	ctx := context.Background()
	dir, fs := newFS(t, FSOptions{MaxReadBytes: 10})
	if err := os.WriteFile(filepath.Join(dir, "big.txt"), []byte(strings.Repeat("a", 100)), 0o600); err != nil {
		t.Fatal(err)
	}
	out, err := fs["fs_read"].Invoke(ctx, `{"path": "big.txt"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(out, truncationMarker) || !strings.HasPrefix(out, strings.Repeat("a", 10)) {
		t.Errorf("truncation: %q", out)
	}
}

// TestFSListBounded: fs_read and subprocess output are capped, and fs_list
// must be too - a directory with tens of thousands of rotated logs would
// otherwise send megabytes to the model in one tool result (review finding
// P1-5).
func TestFSListBounded(t *testing.T) {
	dir, fs := newFS(t, FSOptions{})
	// ~1000 entries x ~80 bytes ≈ 80KB of listing, comfortably over the cap.
	for i := 0; i < 1000; i++ {
		name := fmt.Sprintf("%04d-%s.log", i, strings.Repeat("x", 70))
		if err := os.WriteFile(filepath.Join(dir, name), nil, 0o600); err != nil {
			t.Fatal(err)
		}
	}

	out, err := fs["fs_list"].Invoke(context.Background(), `{}`)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) > maxOutputBytes+1024 {
		t.Errorf("fs_list output not bounded: %d bytes", len(out))
	}
	if !strings.Contains(out, "truncated") {
		t.Error("truncated listing must carry a marker the model can react to")
	}
	// The marker must say how much is missing so the model can narrow down.
	if !strings.Contains(out, "1000") {
		t.Errorf("marker should state the total entry count, got tail: %q", out[len(out)-120:])
	}
}

// TestSandboxEscapes is the security test for the path sandbox: every escape
// technique must be rejected for both read and write.
func TestSandboxEscapes(t *testing.T) {
	ctx := context.Background()
	dir, fs := newFS(t, FSOptions{})

	// A symlink inside the workspace pointing outside it.
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "secret.txt"), []byte("secret"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(dir, "link")); err != nil {
		t.Fatal(err)
	}

	escapes := []string{
		"../etc/passwd",
		"a/../../etc/passwd",
		"/etc/passwd",
		"link/secret.txt",   // read through the symlink
		"link/new-file.txt", // write through the symlink
	}
	for _, path := range escapes {
		args, _ := json.Marshal(map[string]string{"path": path})
		if _, err := fs["fs_read"].Invoke(ctx, string(args)); err == nil {
			t.Errorf("fs_read must reject %q", path)
		}
		wargs, _ := json.Marshal(map[string]string{"path": path, "content": "x"})
		if _, err := fs["fs_write"].Invoke(ctx, string(wargs)); err == nil {
			t.Errorf("fs_write must reject %q", path)
		}
	}
}

func TestFSBadArgs(t *testing.T) {
	ctx := context.Background()
	_, fs := newFS(t, FSOptions{})
	// Unknown fields are rejected so a model typo cannot be silently ignored.
	if _, err := fs["fs_read"].Invoke(ctx, `{"file": "x"}`); err == nil {
		t.Error("unknown arg field must be rejected")
	}
	if _, err := fs["fs_read"].Invoke(ctx, `not-json`); err == nil {
		t.Error("malformed JSON must be rejected")
	}
	if _, err := fs["fs_read"].Invoke(ctx, `{"path": "missing.txt"}`); err == nil {
		t.Error("missing file must error")
	}
}

func subprocessTool(t *testing.T, def config.SubprocessTool) (*Subprocess, string) {
	t.Helper()
	dir := t.TempDir()
	return NewSubprocess(def, dir), dir
}

func TestSubprocessEcho(t *testing.T) {
	tool, _ := subprocessTool(t, config.SubprocessTool{
		Name: "echo", Description: "d", Command: []string{"echo", "hello"},
	})
	out, err := tool.Invoke(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(out) != "hello" {
		t.Errorf("out: %q", out)
	}
}

func TestSubprocessStdin(t *testing.T) {
	tool, _ := subprocessTool(t, config.SubprocessTool{
		Name: "cat", Description: "d", Command: []string{"cat"},
	})
	out, err := tool.Invoke(context.Background(), `{"stdin": "piped"}`)
	if err != nil || out != "piped" {
		t.Errorf("out: %q, %v", out, err)
	}
}

func TestSubprocessExitCodeReportedToModel(t *testing.T) {
	tool, _ := subprocessTool(t, config.SubprocessTool{
		Name: "fail", Description: "d", Command: []string{"sh", "-c", "echo out; echo err >&2; exit 3"},
	})
	out, err := tool.Invoke(context.Background(), "")
	// A failing command is task information, not a harness error.
	if err != nil {
		t.Fatalf("non-zero exit must not be a Go error: %v", err)
	}
	for _, sub := range []string{"exit status 3", "out", "err"} {
		if !strings.Contains(out, sub) {
			t.Errorf("result %q missing %q", out, sub)
		}
	}
}

func TestSubprocessArgsPolicy(t *testing.T) {
	ctx := context.Background()

	// SECURITY: extra args denied by default.
	denied, _ := subprocessTool(t, config.SubprocessTool{
		Name: "echo", Description: "d", Command: []string{"echo"},
	})
	if _, err := denied.Invoke(ctx, `{"args": ["injected"]}`); err == nil {
		t.Error("extra args must be rejected without allow_args")
	}
	// The advertised schema must not even mention args when disabled.
	if strings.Contains(string(denied.Def().Parameters), `"args"`) {
		t.Error("schema should hide args when allow_args is false")
	}

	allowed, _ := subprocessTool(t, config.SubprocessTool{
		Name: "echo", Description: "d", Command: []string{"echo"}, AllowArgs: true,
	})
	out, err := allowed.Invoke(ctx, `{"args": ["a b", "c"]}`)
	if err != nil {
		t.Fatal(err)
	}
	// Arguments pass as discrete argv elements - "a b" stays one argument.
	if strings.TrimSpace(out) != "a b c" {
		t.Errorf("out: %q", out)
	}
}

func TestSubprocessTimeout(t *testing.T) {
	tool, _ := subprocessTool(t, config.SubprocessTool{
		Name: "sleepy", Description: "d",
		Command: []string{"sleep", "5"},
		Timeout: config.Duration(50 * time.Millisecond),
	})
	start := time.Now()
	out, err := tool.Invoke(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "timed out") {
		t.Errorf("result: %q", out)
	}
	if time.Since(start) > 3*time.Second {
		t.Error("timeout did not fire promptly")
	}
}

// TestSubprocessRunCancellationAttribution: when the RUN's deadline (or a
// Ctrl-C) kills the child, the result must not blame the tool's own timeout -
// "command timed out after 60s" after 3 seconds of run time would send the
// operator debugging the wrong knob (review finding P2-4).
func TestSubprocessRunCancellationAttribution(t *testing.T) {
	tool, _ := subprocessTool(t, config.SubprocessTool{
		Name: "sleepy", Description: "d",
		Command: []string{"sleep", "5"},
		Timeout: config.Duration(10 * time.Second), // tool budget NOT the cause
	})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	out, err := tool.Invoke(ctx, "")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out, "timed out after 10s") {
		t.Errorf("run cancellation misattributed to the tool timeout: %q", out)
	}
	if !strings.Contains(out, "run") {
		t.Errorf("result should attribute the abort to the run: %q", out)
	}
}

// TestSubprocessGrandchildDoesNotHang is the regression test for the live
// nested-timeout finding (B2): a tool that spawns a grandchild in its OWN
// process group used to hang Invoke past the deadline, because the orphaned
// grandchild kept the captured stdout pipe open. cmd.WaitDelay must bound
// the wait so the timeout is always honored.
//
// `setsid sleep 30` starts sleep in a brand-new session/process group that
// the tool's group-kill cannot reach, and it inherits the stdout pipe -
// exactly the shape of a nested amele spawning its own subprocess.
func TestSubprocessGrandchildDoesNotHang(t *testing.T) {
	if _, err := exec.LookPath("setsid"); err != nil {
		t.Skip("setsid not available")
	}
	tool, _ := subprocessTool(t, config.SubprocessTool{
		Name: "spawner", Description: "d",
		Command: []string{"sh", "-c", "setsid sleep 30 & echo spawned; exit 0"},
		Timeout: config.Duration(300 * time.Millisecond),
	})

	start := time.Now()
	// The tool's own command exits immediately; the point is that the
	// orphaned grandchild holding the pipe must not make Invoke block.
	if _, err := tool.Invoke(context.Background(), ""); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	// Without WaitDelay this returns only after the 30s sleep; with it,
	// Invoke returns within the WaitDelay budget (3s) plus slack.
	if elapsed := time.Since(start); elapsed > 8*time.Second {
		t.Errorf("Invoke hung on an orphaned grandchild: took %v", elapsed)
	}
}

func TestSubprocessOutputTruncation(t *testing.T) {
	tool, _ := subprocessTool(t, config.SubprocessTool{
		Name: "spam", Description: "d",
		Command: []string{"sh", "-c", "yes x | head -c 200000"},
	})
	out, err := tool.Invoke(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	if len(out) > maxOutputBytes+len(truncationMarker) {
		t.Errorf("output not bounded: %d bytes", len(out))
	}
	if !strings.HasSuffix(out, truncationMarker) {
		t.Error("missing truncation marker")
	}
}

func TestSubprocessMissingExecutable(t *testing.T) {
	tool, _ := subprocessTool(t, config.SubprocessTool{
		Name: "ghost", Description: "d", Command: []string{"definitely-not-a-real-binary-xyz"},
	})
	// Spawn failure is a harness error (the YAML is wrong), not model info.
	if _, err := tool.Invoke(context.Background(), ""); err == nil {
		t.Error("missing executable must be a Go error")
	}
}

func TestSubprocessRunsInWorkspace(t *testing.T) {
	tool, dir := subprocessTool(t, config.SubprocessTool{
		Name: "pwd", Description: "d", Command: []string{"pwd"},
	})
	out, err := tool.Invoke(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	resolved, _ := filepath.EvalSymlinks(dir)
	if got := strings.TrimSpace(out); got != resolved && got != dir {
		t.Errorf("cwd: %q want %q", got, dir)
	}
}

// --- builtin shell tool ---------------------------------------------------

// shellTool builds the builtin shell tool over a fresh temp workspace.
func shellTool(t *testing.T, cfg config.ShellConfig) (*Shell, string) {
	t.Helper()
	dir := t.TempDir()
	sh, err := NewShell(cfg, dir)
	if err != nil {
		t.Fatalf("NewShell: %v", err)
	}
	return sh, dir
}

func TestShellRunsCommand(t *testing.T) {
	sh, _ := shellTool(t, config.ShellConfig{Enabled: true})
	out, err := sh.Invoke(context.Background(), `{"command": "echo hi"}`)
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(out) != "hi" {
		t.Errorf("out: %q", out)
	}
	if sh.Def().Name != "shell" {
		t.Errorf("tool name: %q", sh.Def().Name)
	}
}

// The tool runs a shell line, so shell metacharacters must work - that is the
// whole difference from a subprocess tool's fixed argv.
func TestShellUsesShellSemantics(t *testing.T) {
	sh, dir := shellTool(t, config.ShellConfig{Enabled: true})
	out, err := sh.Invoke(context.Background(), `{"command": "echo one; echo two | tr a-z A-Z"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "one") || !strings.Contains(out, "TWO") {
		t.Errorf("out: %q", out)
	}
	// cwd is the workspace.
	out, err = sh.Invoke(context.Background(), `{"command": "pwd"}`)
	if err != nil {
		t.Fatal(err)
	}
	resolved, _ := filepath.EvalSymlinks(dir)
	if got := strings.TrimSpace(out); got != resolved && got != dir {
		t.Errorf("cwd: %q want %q", got, dir)
	}
}

func TestShellDenyRejects(t *testing.T) {
	sh, _ := shellTool(t, config.ShellConfig{Enabled: true, Deny: []string{"rm *"}})
	out, err := sh.Invoke(context.Background(), `{"command": "rm -rf /tmp/x"}`)
	// A rejection is model-facing information, not a harness failure.
	if err != nil {
		t.Fatalf("rejection must not be a Go error: %v", err)
	}
	if !strings.HasPrefix(out, "command rejected by shell policy:") {
		t.Errorf("result: %q", out)
	}
}

// Deny wins over allow: an explicitly allowed prefix must not resurrect a
// denied command.
func TestShellDenyBeatsAllow(t *testing.T) {
	sh, _ := shellTool(t, config.ShellConfig{
		Enabled: true,
		Allow:   []string{"git *"},
		Deny:    []string{"git push*"},
	})
	out, err := sh.Invoke(context.Background(), `{"command": "git push origin main"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(out, "command rejected by shell policy:") {
		t.Errorf("result: %q", out)
	}
}

func TestShellAllowList(t *testing.T) {
	sh, _ := shellTool(t, config.ShellConfig{Enabled: true, Allow: []string{"echo *"}})

	// Not in the allow list → rejected.
	out, err := sh.Invoke(context.Background(), `{"command": "cat /etc/passwd"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(out, "command rejected by shell policy:") {
		t.Errorf("unlisted command should be rejected: %q", out)
	}

	// In the allow list → runs.
	out, err = sh.Invoke(context.Background(), `{"command": "echo ok"}`)
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(out) != "ok" {
		t.Errorf("allowed command should run: %q", out)
	}
}

// An empty allow list means "anything the deny list does not catch".
func TestShellEmptyAllowRunsEverything(t *testing.T) {
	sh, _ := shellTool(t, config.ShellConfig{Enabled: true})
	out, err := sh.Invoke(context.Background(), `{"command": "echo anything"}`)
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(out) != "anything" {
		t.Errorf("out: %q", out)
	}
}

func TestGlobMatch(t *testing.T) {
	tests := []struct {
		pattern string
		command string
		want    bool
	}{
		// literal
		{"ls", "ls", true},
		{"ls", "ls -l", false},
		{"", "", true},
		{"", "ls", false},
		// prefix star
		{"*status", "git status", true},
		{"*status", "git status --short", false},
		// suffix star
		{"git *", "git status", true},
		{"git *", "gitk", false},
		{"ls*", "ls", true}, // * matches the empty substring
		{"ls*", "ls -la", true},
		// middle star
		{"git * --short", "git status --short", true},
		{"git * --short", "git status", false},
		// multiple stars
		{"*rm *", "sudo rm -rf /", true},
		{"*a*b*c*", "xxaxxbxxcxx", true},
		{"*a*b*c*", "xxaxxcxxbxx", false},
		// bare star matches everything, including the empty string
		{"*", "anything at all", true},
		{"*", "", true},
		// case sensitivity
		{"git *", "GIT status", false},
		{"GIT *", "git status", false},
		// no other metacharacters: ? and [] are literal
		{"ls ?", "ls x", false},
		{"ls ?", "ls ?", true},
		{"ls [ab]", "ls a", false},
		// overlapping literal runs must not be double-counted
		{"a*a", "a", false},
		{"a*a", "aa", true},
	}
	for _, tt := range tests {
		if got := globMatch(tt.pattern, tt.command); got != tt.want {
			t.Errorf("globMatch(%q, %q) = %v want %v", tt.pattern, tt.command, got, tt.want)
		}
	}
}

func TestShellTimeout(t *testing.T) {
	sh, _ := shellTool(t, config.ShellConfig{
		Enabled: true,
		Timeout: config.Duration(50 * time.Millisecond),
	})
	start := time.Now()
	out, err := sh.Invoke(context.Background(), `{"command": "sleep 5"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "timed out") {
		t.Errorf("result: %q", out)
	}
	if time.Since(start) > 3*time.Second {
		t.Error("timeout did not fire promptly")
	}
}

func TestShellExitCodeReportedToModel(t *testing.T) {
	sh, _ := shellTool(t, config.ShellConfig{Enabled: true})
	out, err := sh.Invoke(context.Background(), `{"command": "echo out; echo err >&2; exit 3"}`)
	if err != nil {
		t.Fatalf("non-zero exit must not be a Go error: %v", err)
	}
	for _, sub := range []string{"exit status 3", "out", "err"} {
		if !strings.Contains(out, sub) {
			t.Errorf("result %q missing %q", out, sub)
		}
	}
}

func TestShellBadArgs(t *testing.T) {
	sh, _ := shellTool(t, config.ShellConfig{Enabled: true})
	ctx := context.Background()
	if _, err := sh.Invoke(ctx, `{"cmd": "echo hi"}`); err == nil {
		t.Error("unknown arg field must be rejected")
	}
	if _, err := sh.Invoke(ctx, `not-json`); err == nil {
		t.Error("malformed JSON must be rejected")
	}
	if _, err := sh.Invoke(ctx, `{"command": ""}`); err == nil {
		t.Error("empty command must be rejected")
	}
	if _, err := sh.Invoke(ctx, ``); err == nil {
		t.Error("missing arguments must be rejected")
	}
}

func TestShellOutputTruncation(t *testing.T) {
	sh, _ := shellTool(t, config.ShellConfig{Enabled: true})
	out, err := sh.Invoke(context.Background(), `{"command": "yes x | head -c 200000"}`)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) > maxOutputBytes+len(truncationMarker) {
		t.Errorf("output not bounded: %d bytes", len(out))
	}
	if !strings.HasSuffix(out, truncationMarker) {
		t.Error("missing truncation marker")
	}
}

// The shell tool shares the subprocess run core, so a run-level cancellation
// must be attributed to the run, not to the tool's own timeout.
func TestShellRunCancellationAttribution(t *testing.T) {
	sh, _ := shellTool(t, config.ShellConfig{
		Enabled: true,
		Timeout: config.Duration(10 * time.Second),
	})
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	out, err := sh.Invoke(ctx, `{"command": "sleep 5"}`)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out, "timed out after 10s") {
		t.Errorf("run cancellation misattributed to the tool timeout: %q", out)
	}
	if !strings.Contains(out, "run") {
		t.Errorf("result should attribute the abort to the run: %q", out)
	}
}

// TestShellPolicyNormalizesCommand covers the bypasses the accident-prevention
// layer must not have: leading whitespace, a tab, and - the common one, because
// well-behaved models emit multi-line commands routinely - a denied command
// hiding on the second line.
func TestShellPolicyNormalizesCommand(t *testing.T) {
	rejected := func(t *testing.T, sh *Shell, command string) bool {
		t.Helper()
		raw, err := json.Marshal(shellArgs{Command: command})
		if err != nil {
			t.Fatal(err)
		}
		out, err := sh.Invoke(context.Background(), string(raw))
		if err != nil {
			t.Fatalf("Invoke(%q): %v", command, err)
		}
		return strings.HasPrefix(out, "command rejected by shell policy:")
	}

	t.Run("deny survives whitespace and extra lines", func(t *testing.T) {
		sh, _ := shellTool(t, config.ShellConfig{Enabled: true, Deny: []string{"rm *"}})
		for _, command := range []string{
			" rm -rf x",            // leading space
			"\trm -rf x",           // leading tab
			"rm -rf x ",            // trailing space
			"cd build\nrm -rf .",   // denied command on the second line
			"cd build\n  rm -rf .", // ...indented, as a model would write it
			"echo a\n\nrm -rf x\n", // blank lines around it
		} {
			if !rejected(t, sh, command) {
				t.Errorf("deny bypassed by %q", command)
			}
		}
	})

	t.Run("allow applies to every line", func(t *testing.T) {
		sh, _ := shellTool(t, config.ShellConfig{Enabled: true, Allow: []string{"echo *", "ls*"}})

		// Every line allowed → runs.
		out, err := sh.Invoke(context.Background(), `{"command": "  echo one\n\nls\n"}`)
		if err != nil {
			t.Fatal(err)
		}
		if strings.HasPrefix(out, "command rejected by shell policy:") {
			t.Errorf("all-allowed multi-line command was rejected: %q", out)
		}
		if !strings.Contains(out, "one") {
			t.Errorf("command did not run: %q", out)
		}

		// One line outside the allow list → the whole command is rejected.
		if !rejected(t, sh, "echo one\ncurl http://evil.example") {
			t.Error("allow list bypassed by a second, unlisted line")
		}
	})
}

// TestSubprocessEnvAllowlist: with env set, the child process is built from
// the allowlist - a secret in amele's environment that is not listed never
// reaches a model-driven command, while listed names and the PATH/HOME/LANG
// base survive. A listed-but-unset name is silently skipped, not an error.
func TestSubprocessEnvAllowlist(t *testing.T) {
	t.Setenv("AMELE_TEST_SECRET", "topsecret-value")
	t.Setenv("AMELE_TEST_KEEP", "kept-value")

	tool, _ := subprocessTool(t, config.SubprocessTool{
		Name: "envprobe", Description: "d",
		// ${VAR-<unset>} (no colon) distinguishes unset from empty.
		Command: []string{"sh", "-c", `echo "SECRET=${AMELE_TEST_SECRET-<unset>} KEEP=${AMELE_TEST_KEEP-<unset>} PATH=${PATH:+<set>}"`},
		Env:     []string{"AMELE_TEST_KEEP", "AMELE_TEST_NEVER_SET"},
	})
	out, err := tool.Invoke(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out, "topsecret-value") {
		t.Errorf("unlisted secret leaked into allowlisted child: %q", out)
	}
	for _, want := range []string{"SECRET=<unset>", "KEEP=kept-value", "PATH=<set>"} {
		if !strings.Contains(out, want) {
			t.Errorf("out %q missing %q", out, want)
		}
	}
}

// TestSubprocessEnvInheritedWhenUnset pins backward compatibility: without an
// env allowlist the child inherits amele's entire environment, exactly as
// every config written before the field existed relies on.
func TestSubprocessEnvInheritedWhenUnset(t *testing.T) {
	t.Setenv("AMELE_TEST_SECRET", "topsecret-value")
	tool, _ := subprocessTool(t, config.SubprocessTool{
		Name: "envprobe", Description: "d",
		Command: []string{"sh", "-c", `echo "S=${AMELE_TEST_SECRET-<unset>}"`},
	})
	out, err := tool.Invoke(context.Background(), "")
	if err != nil || !strings.Contains(out, "S=topsecret-value") {
		t.Errorf("full inheritance broken without an allowlist: %q, %v", out, err)
	}
}

// TestShellEnvAllowlist: the builtin shell honors tools.shell.env the same
// way - printenv on an unlisted secret prints nothing, while a listed
// variable and PATH (always passed) remain visible.
func TestShellEnvAllowlist(t *testing.T) {
	t.Setenv("AMELE_TEST_SECRET", "topsecret-value")
	t.Setenv("AMELE_TEST_KEEP", "kept-value")
	sh, _ := shellTool(t, config.ShellConfig{Enabled: true, Env: []string{"AMELE_TEST_KEEP"}})

	out, err := sh.Invoke(context.Background(), `{"command": "printenv AMELE_TEST_SECRET"}`)
	if err != nil {
		t.Fatal(err)
	}
	// printenv exits 1 for an unset name; the point is the VALUE is absent.
	if strings.Contains(out, "topsecret-value") {
		t.Errorf("unlisted secret visible to allowlisted shell: %q", out)
	}

	out, err = sh.Invoke(context.Background(), `{"command": "printenv AMELE_TEST_KEEP; printenv PATH"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "kept-value") {
		t.Errorf("listed variable missing in allowlisted shell: %q", out)
	}
	if !strings.Contains(out, "/") {
		t.Errorf("PATH not passed to allowlisted shell: %q", out)
	}
}

// TestShellEnvInheritedWhenUnset: an absent tools.shell.env keeps full
// inheritance for the shell too.
func TestShellEnvInheritedWhenUnset(t *testing.T) {
	t.Setenv("AMELE_TEST_SECRET", "topsecret-value")
	sh, _ := shellTool(t, config.ShellConfig{Enabled: true})
	out, err := sh.Invoke(context.Background(), `{"command": "printenv AMELE_TEST_SECRET"}`)
	if err != nil || !strings.Contains(out, "topsecret-value") {
		t.Errorf("full inheritance broken without an allowlist: %q, %v", out, err)
	}
}

// TestDecodeArgsRejectsMalformed is a regression test: the decoder read ONE
// JSON value and stopped, so `{"path":"x"} rm -rf /` ran as if the trailing
// text were not there, and a bare `null` decoded into a zero struct - which
// made `fs_list null` list the workspace root instead of failing.
func TestDecodeArgsRejectsMalformed(t *testing.T) {
	dir, fs := newFS(t, FSOptions{})
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("hi"), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	bad := []struct {
		name string
		args string
	}{
		{"trailing garbage", `{"path":"a.txt"} garbage`},
		{"second json value", `{"path":"a.txt"} {"path":"a.txt"}`},
		{"literal null", `null`},
		{"null with whitespace", "  null\n"},
	}
	for _, tt := range bad {
		t.Run(tt.name, func(t *testing.T) {
			if out, err := fs["fs_read"].Invoke(ctx, tt.args); err == nil {
				t.Errorf("fs_read accepted %q: %q", tt.args, out)
			}
			if out, err := fs["fs_list"].Invoke(ctx, tt.args); err == nil {
				t.Errorf("fs_list accepted %q: %q", tt.args, out)
			}
		})
	}

	// The valid forms must keep working: a well-formed object, an empty
	// object, and the empty string the loop passes for argument-less calls.
	if got, err := fs["fs_read"].Invoke(ctx, `{"path":"a.txt"}`); err != nil || got != "hi" {
		t.Errorf("valid args broke: %q, %v", got, err)
	}
	if _, err := fs["fs_list"].Invoke(ctx, `{}`); err != nil {
		t.Errorf("empty object rejected: %v", err)
	}
	if _, err := fs["fs_list"].Invoke(ctx, ""); err != nil {
		t.Errorf("empty args rejected: %v", err)
	}
}

// TestSubprocessTruncationBoundary is a regression test for an off-by-one:
// truncation was inferred from "buffer is full", so output of EXACTLY
// maxOutputBytes was marked truncated even though nothing was dropped, and
// truncated stderr carried no marker at all - the model saw a cut error
// message as if it were complete.
func TestSubprocessTruncationBoundary(t *testing.T) {
	// yes|head produces a deterministic byte count without a temp file.
	spam := func(bytes int, stream string) []string {
		return []string{"sh", "-c", fmt.Sprintf("yes x | head -c %d %s", bytes, stream)}
	}
	tests := []struct {
		name          string
		command       []string
		wantTruncated bool
	}{
		{"exactly max", spam(maxOutputBytes, ""), false},
		{"one over max", spam(maxOutputBytes+1, ""), true},
		{"well under max", spam(10, ""), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tool, _ := subprocessTool(t, config.SubprocessTool{
				Name: "spam", Description: "d", Command: tt.command,
			})
			out, err := tool.Invoke(context.Background(), "")
			if err != nil {
				t.Fatal(err)
			}
			if got := strings.HasSuffix(out, truncationMarker); got != tt.wantTruncated {
				t.Errorf("truncation marker: got %v, want %v (output %d bytes)",
					got, tt.wantTruncated, len(out))
			}
		})
	}

	// Truncated stderr must be marked too: it reaches the model on the
	// non-zero-exit path, where a silently cut message is misleading.
	tool, _ := subprocessTool(t, config.SubprocessTool{
		Name: "noisy", Description: "d",
		Command: []string{"sh", "-c", fmt.Sprintf("yes x | head -c %d >&2; exit 1", maxOutputBytes+1000)},
	})
	out, err := tool.Invoke(context.Background(), "")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "stderr:") {
		t.Fatalf("stderr not reported: %q", out[:min(len(out), 200)])
	}
	if !strings.Contains(out, truncationMarker) {
		t.Error("truncated stderr carries no truncation marker")
	}
	if len(out) > 2*maxOutputBytes+1024 {
		t.Errorf("combined output not bounded: %d bytes", len(out))
	}
}

// TestInvokeOutcomeClassifiesEndings pins the out-of-band answer to "did this
// tool call work?" for every ending that reports itself as ordinary result
// text with a nil error. Without it, a consumer that can only see (text, err)
// - `amele run -v` - has to call all of them "ok" (live-test finding C-2).
//
// The text stays what the model reads; only the classification is asserted
// here, plus the invariant that Invoke and InvokeOutcome return the same text.
func TestInvokeOutcomeClassifiesEndings(t *testing.T) {
	tests := []struct {
		name string
		call func(t *testing.T) (string, Outcome, error)
		want Outcome
	}{
		{
			name: "shell success",
			call: func(t *testing.T) (string, Outcome, error) {
				sh, _ := shellTool(t, config.ShellConfig{Enabled: true})
				return sh.InvokeOutcome(context.Background(), `{"command": "echo hi"}`)
			},
			want: Outcome{Kind: OutcomeOK},
		},
		{
			name: "shell policy rejection",
			call: func(t *testing.T) (string, Outcome, error) {
				sh, _ := shellTool(t, config.ShellConfig{Enabled: true, Deny: []string{"rm *"}})
				return sh.InvokeOutcome(context.Background(), `{"command": "rm -rf /tmp/x"}`)
			},
			want: Outcome{Kind: OutcomeRejected},
		},
		{
			name: "shell non-zero exit",
			call: func(t *testing.T) (string, Outcome, error) {
				sh, _ := shellTool(t, config.ShellConfig{Enabled: true})
				return sh.InvokeOutcome(context.Background(), `{"command": "exit 3"}`)
			},
			want: Outcome{Kind: OutcomeExit, ExitCode: 3},
		},
		{
			name: "shell tool timeout",
			call: func(t *testing.T) (string, Outcome, error) {
				sh, _ := shellTool(t, config.ShellConfig{
					Enabled: true, Timeout: config.Duration(50 * time.Millisecond),
				})
				return sh.InvokeOutcome(context.Background(), `{"command": "sleep 5"}`)
			},
			want: Outcome{Kind: OutcomeTimedOut},
		},
		{
			name: "subprocess success",
			call: func(t *testing.T) (string, Outcome, error) {
				tool, _ := subprocessTool(t, config.SubprocessTool{
					Name: "echo", Description: "d", Command: []string{"echo", "hello"},
				})
				return tool.InvokeOutcome(context.Background(), "")
			},
			want: Outcome{Kind: OutcomeOK},
		},
		{
			name: "subprocess non-zero exit",
			call: func(t *testing.T) (string, Outcome, error) {
				tool, _ := subprocessTool(t, config.SubprocessTool{
					Name: "failing", Description: "d", Command: []string{"sh", "-c", "exit 7"},
				})
				return tool.InvokeOutcome(context.Background(), "")
			},
			want: Outcome{Kind: OutcomeExit, ExitCode: 7},
		},
		{
			name: "subprocess tool timeout",
			call: func(t *testing.T) (string, Outcome, error) {
				tool, _ := subprocessTool(t, config.SubprocessTool{
					Name: "sleepy", Description: "d", Command: []string{"sleep", "5"},
					Timeout: config.Duration(50 * time.Millisecond),
				})
				return tool.InvokeOutcome(context.Background(), "")
			},
			want: Outcome{Kind: OutcomeTimedOut},
		},
		{
			// The run's own deadline, not the tool's: the outcome must
			// distinguish them for the same reason the result text does.
			name: "run cancellation is not a tool timeout",
			call: func(t *testing.T) (string, Outcome, error) {
				tool, _ := subprocessTool(t, config.SubprocessTool{
					Name: "sleepy", Description: "d", Command: []string{"sleep", "5"},
					Timeout: config.Duration(10 * time.Second),
				})
				ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
				defer cancel()
				return tool.InvokeOutcome(ctx, "")
			},
			want: Outcome{Kind: OutcomeAborted},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out, got, err := tt.call(t)
			if err != nil {
				t.Fatalf("these endings are result text, not Go errors: %v", err)
			}
			if got != tt.want {
				t.Errorf("outcome = %+v (%q), want %+v; result text: %q", got, got, tt.want, out)
			}
		})
	}
}

// TestInvokeMatchesInvokeOutcomeText: Invoke is the narrow view of the same
// call, so it must never disagree with the richer one about what the model is
// shown.
func TestInvokeMatchesInvokeOutcomeText(t *testing.T) {
	sh, _ := shellTool(t, config.ShellConfig{Enabled: true, Deny: []string{"rm *"}})
	const args = `{"command": "rm -rf /tmp/x"}`

	plain, err := sh.Invoke(context.Background(), args)
	if err != nil {
		t.Fatal(err)
	}
	rich, outcome, err := sh.InvokeOutcome(context.Background(), args)
	if err != nil {
		t.Fatal(err)
	}
	if plain != rich {
		t.Errorf("Invoke text %q != InvokeOutcome text %q", plain, rich)
	}
	if outcome.Kind != OutcomeRejected {
		t.Errorf("outcome = %+v", outcome)
	}
}

// TestOutcomeString pins the operator-facing wording every consumer embeds
// verbatim, including the exit status the number carries.
func TestOutcomeString(t *testing.T) {
	tests := []struct {
		outcome Outcome
		want    string
	}{
		{Outcome{}, "ok"},
		{Outcome{Kind: OutcomeRejected}, "rejected"},
		{Outcome{Kind: OutcomeTimedOut}, "timed out"},
		{Outcome{Kind: OutcomeAborted}, "aborted"},
		{Outcome{Kind: OutcomeExit, ExitCode: 3}, "exit 3"},
		// An unknown kind must not masquerade as success.
		{Outcome{Kind: OutcomeKind(99)}, "outcome 99"},
	}
	for _, tt := range tests {
		if got := tt.outcome.String(); got != tt.want {
			t.Errorf("Outcome%+v.String() = %q, want %q", tt.outcome, got, tt.want)
		}
	}
}
