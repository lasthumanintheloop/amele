package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/lasthumanintheloop/amele/internal/session"
)

// syncBuffer is a bytes.Buffer safe for the one pattern the chat interruption
// test needs: the run writes stderr from its own goroutine while the test polls
// it for the prompt. Without the lock that poll is a data race, and `go test
// -race` is a gate (docs/engineering.md §6).
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// blockingServer is the provider stand-in for the interruption tests: it
// answers nothing and holds every request open for as long as the test runs.
// The returned channel is closed as soon as the first request lands, which is
// the only synchronisation point the tests need - once amele is blocked inside
// the provider call, cancelling is guaranteed to hit a run in flight rather
// than a run that already finished.
//
// The handler waits on a test-owned channel rather than on r.Context(): an
// HTTP/1.1 server only learns that the client hung up through a background
// read that is not necessarily active while a handler is running, so relying
// on the request context to unblock it would hang Close (observed: httptest
// "blocked in Close after 5 seconds"). The release channel is closed by a
// cleanup registered AFTER the server's, and cleanups run LIFO, so every
// handler is unblocked before Close waits for it.
func blockingServer(t *testing.T) (*httptest.Server, <-chan struct{}) {
	t.Helper()
	hit := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		once.Do(func() { close(hit) }) // only the first request opens the gate
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) })
	return srv, hit
}

// sessionEvents decodes the single session file written under dir/sessions.
func sessionEvents(t *testing.T, dir string) []session.Event {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(dir, "sessions", "*.jsonl"))
	if err != nil || len(files) != 1 {
		t.Fatalf("session files: %v, %v", files, err)
	}
	data, err := os.ReadFile(files[0])
	if err != nil {
		t.Fatal(err)
	}
	var events []session.Event
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var e session.Event
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Fatalf("decoding session line %q: %v", line, err)
		}
		events = append(events, e)
	}
	return events
}

// findEvent returns the first event of the given type.
func findEvent(events []session.Event, typ string) (session.Event, bool) {
	for _, e := range events {
		if e.Type == typ {
			return e, true
		}
	}
	return session.Event{}, false
}

// TestRunCanceledMidRunWritesRunEnd pins the SIGINT/SIGTERM contract at the
// level where it is actually implemented: main() turns both signals into a
// cancellation of the run context (signal.NotifyContext), so cancelling that
// context IS the signal path. CONTRACT (docs/contracts/cli.md "Signals"):
// exit 1, a truthful run_end in the session log, and the closing summary still
// on stderr - an interrupted cron run must leave the same evidence a failed one
// does, not a truncated log.
func TestRunCanceledMidRunWritesRunEnd(t *testing.T) {
	srv, hit := blockingServer(t)
	cfgPath, dir := writeTestConfig(t, srv.URL, "session_dir: sessions\n")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-hit
		cancel()
	}()

	var stdout, stderr bytes.Buffer
	code := run(ctx, []string{"run", cfgPath, "task"}, strings.NewReader(""), &stdout, &stderr, env(t))

	assertInterruptedRun(t, code, stdout.String(), stderr.String(), dir)
}

// assertInterruptedRun checks the whole observable contract of an interrupted
// `amele run`, whichever way the interruption was delivered: exit 1, no stdout,
// the interruption named on stderr, the summary line, and a truthful run_end in
// the session log under dir. Shared by the in-process and the cross-process
// tests so both pin exactly the same promise.
func assertInterruptedRun(t *testing.T, code int, stdout, stderr, dir string) {
	t.Helper()
	if code != ExitTaskFailed {
		t.Fatalf("exit %d, want %d (interruption is a task failure, not a budget kill); stderr: %s",
			code, ExitTaskFailed, stderr)
	}
	// CONTRACT: an interrupted run produces no answer, so nothing reaches a pipe.
	if stdout != "" {
		t.Errorf("stdout must stay empty on an interrupted run: %q", stdout)
	}
	if !strings.Contains(stderr, "run interrupted") {
		t.Errorf("stderr must say the run was interrupted: %q", stderr)
	}
	// The summary line is what a human reads in cron mail; it must survive.
	// "1 turn": the interrupted round-trip still counts as a turn.
	if !strings.Contains(stderr, "✗ 1 turn,") {
		t.Errorf("stderr must carry the closing summary: %q", stderr)
	}

	events := sessionEvents(t, dir)
	end, ok := findEvent(events, "run_end")
	if !ok {
		t.Fatalf("session log has no run_end event: %+v", events)
	}
	if end.Status != "error" {
		t.Errorf("run_end status = %q, want %q", end.Status, "error")
	}
	if end.ExitCode == nil || *end.ExitCode != ExitTaskFailed {
		t.Errorf("run_end exit_code = %v, want %d", end.ExitCode, ExitTaskFailed)
	}
	if end.Turns != 1 {
		t.Errorf("run_end turns = %d, want 1 (the interrupted round-trip still counts)", end.Turns)
	}
}

// TestRunInterruptedDuringStdinReadIsReported pins the same contract for the
// window BEFORE the first provider call: a run whose piped task input is still
// being read when the interruption arrives is a run that already started, so it
// must end like every other interrupted run - run_end in the session log, the
// interruption named on stderr, the summary unless -q, and the exit code the
// cause maps to. It used to return straight out of the stdin read
// ("reading stdin: context canceled", exit 1 with no session ending at all),
// which left a cron job's session file without a truthful end event.
//
// The context is ended up front rather than mid-read: with a reader that never
// delivers, that is the same observable state as a signal landing while the
// read blocks, and it needs no polling to be deterministic.
func TestRunInterruptedDuringStdinReadIsReported(t *testing.T) {
	tests := []struct {
		name     string
		extra    string
		ctx      func(t *testing.T) context.Context
		wantCode int
		wantErr  string
	}{
		{
			name:  "canceled",
			extra: "session_dir: sessions\n",
			ctx: func(t *testing.T) context.Context {
				t.Helper()
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				return ctx
			},
			wantCode: ExitTaskFailed,
			wantErr:  "run interrupted",
		},
		{
			// limits.timeout is a configured budget, so it keeps exit 3 - but it
			// must leave the same evidence behind.
			name:     "timeout",
			extra:    "session_dir: sessions\nlimits:\n  timeout: 50ms\n",
			ctx:      func(*testing.T) context.Context { return context.Background() },
			wantCode: ExitBudgetExceeded,
			wantErr:  "budget exceeded",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := scriptedServer(t) // the provider must never be reached
			cfgPath, dir := writeTestConfig(t, srv.URL, tt.extra)

			// Neither task text nor a prompt template: stdin IS the task, so it
			// is read - and this reader never delivers.
			done := make(chan struct{})
			t.Cleanup(func() { close(done) })

			var stdout, stderr bytes.Buffer
			code := run(tt.ctx(t), []string{"run", cfgPath}, hangingReader{done}, &stdout, &stderr, env(t))

			if code != tt.wantCode {
				t.Fatalf("exit %d, want %d; stderr: %s", code, tt.wantCode, stderr.String())
			}
			if stdout.String() != "" {
				t.Errorf("stdout must stay empty: %q", stdout.String())
			}
			if !strings.Contains(stderr.String(), tt.wantErr) {
				t.Errorf("stderr must name the cause (%q): %q", tt.wantErr, stderr.String())
			}
			if !strings.Contains(stderr.String(), "✗") {
				t.Errorf("stderr must carry the closing summary: %q", stderr.String())
			}
			end, ok := findEvent(sessionEvents(t, dir), "run_end")
			if !ok {
				t.Fatal("session log has no run_end event")
			}
			if end.Status != "error" {
				t.Errorf("run_end status = %q, want %q", end.Status, "error")
			}
			if end.ExitCode == nil || *end.ExitCode != tt.wantCode {
				t.Errorf("run_end exit_code = %v, want %d", end.ExitCode, tt.wantCode)
			}
		})
	}
}

// TestRunCanceledMidRunQuiet pins what -q may and may not silence when the run
// is interrupted: the summary goes, the error and the session log stay. A cron
// job running -q must still learn that its run died and must still be able to
// audit it afterwards.
func TestRunCanceledMidRunQuiet(t *testing.T) {
	srv, hit := blockingServer(t)
	cfgPath, dir := writeTestConfig(t, srv.URL, "session_dir: sessions\n")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-hit
		cancel()
	}()

	var stdout, stderr bytes.Buffer
	code := run(ctx, []string{"run", cfgPath, "-q", "task"}, strings.NewReader(""), &stdout, &stderr, env(t))

	if code != ExitTaskFailed {
		t.Fatalf("exit %d, want %d; stderr: %s", code, ExitTaskFailed, stderr.String())
	}
	if strings.Contains(stderr.String(), "✗") {
		t.Errorf("-q must suppress the summary line: %q", stderr.String())
	}
	if !strings.Contains(stderr.String(), "run interrupted") {
		t.Errorf("-q must not suppress the error: %q", stderr.String())
	}
	if _, ok := findEvent(sessionEvents(t, dir), "run_end"); !ok {
		t.Error("-q must not affect session logging: run_end missing")
	}
}

// TestChatCanceledAtPromptWritesRunEnd documents chat's side of the same
// contract. Ctrl-C at the `> ` prompt cannot be observed by the blocked stdin
// read, so the REPL races the read against the context (readAsync) and ends the
// session itself: exit 1, run_end, summary.
func TestChatCanceledAtPromptWritesRunEnd(t *testing.T) {
	// No scripted bodies: the interrupt lands at the prompt, before any
	// provider call, which is exactly what a Ctrl-C usually interrupts.
	srv := scriptedServer(t)
	cfgPath, dir := writeTestConfig(t, srv.URL, "session_dir: sessions\n")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// A pipe that never delivers a line: the REPL sits at the prompt exactly
	// as it does in front of a human.
	pr, pw, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = pw.Close(); _ = pr.Close() }()

	done := make(chan int, 1)
	var stdout, stderr syncBuffer
	go func() {
		done <- run(ctx, []string{"chat", cfgPath}, pr, &stdout, &stderr, env(t))
	}()

	// The prompt is written before the read blocks; waiting for it keeps the
	// cancellation on the far side of "chat actually started".
	waitFor(t, func() bool { return strings.Contains(stderr.String(), chatPrompt) })
	cancel()

	select {
	case code := <-done:
		if code != ExitTaskFailed {
			t.Fatalf("exit %d, want %d; stderr: %s", code, ExitTaskFailed, stderr.String())
		}
	case <-time.After(10 * time.Second):
		t.Fatal("chat did not exit after the context was canceled")
	}

	if !strings.Contains(stderr.String(), "chat interrupted") {
		t.Errorf("stderr must say the chat was interrupted: %q", stderr.String())
	}
	if !strings.Contains(stderr.String(), "✗") {
		t.Errorf("stderr must carry the closing summary: %q", stderr.String())
	}
	end, ok := findEvent(sessionEvents(t, dir), "run_end")
	if !ok {
		t.Fatal("session log has no run_end event")
	}
	if end.ExitCode == nil || *end.ExitCode != ExitTaskFailed {
		t.Errorf("run_end exit_code = %v, want %d", end.ExitCode, ExitTaskFailed)
	}
}

// buildAmele compiles the real binary into a temp dir and returns its path.
// The in-process tests above exercise run(); only a real process can exercise
// main()'s signal.NotifyContext wiring and the exit status a shell observes,
// which is what the "Signals" contract actually promises.
//
// It is hermetic: a local compile of this module, no network, no toolchain
// downloads (the test binary running it was itself just compiled by the same
// toolchain). It is skipped only when there is no `go` to build with.
func buildAmele(t *testing.T) string {
	t.Helper()
	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("no go toolchain in PATH: cannot build the binary for the signal test")
	}
	bin := filepath.Join(t.TempDir(), "amele-signal-test")
	//nolint:gosec // G204: goBin is the toolchain that is already compiling this test; the rest is constant.
	cmd := exec.Command(goBin, "build", "-o", bin, ".")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("building amele: %v\n%s", err, out)
	}
	return bin
}

// TestSignalsInterruptRealProcess is the cross-process half of the signal
// contract: a real amele process, a real SIGINT/SIGTERM, and the exit status a
// shell would see.
//
// CONTRACT (docs/contracts/cli.md "Signals"): amele CATCHES both signals, so
// the process exits normally with code 1 - a script sees 1, never the 128+N a
// shell reports for a tool killed by an uncaught signal - and the session log
// still ends with run_end.
//
// Hermetic and race-free by construction: the signal is sent only after the
// fake provider has seen the request, so the process is provably inside the
// run (and past signal.NotifyContext) when it arrives - there is no window in
// which the default signal action could kill it instead.
func TestSignalsInterruptRealProcess(t *testing.T) {
	bin := buildAmele(t)

	for _, tc := range []struct {
		name string
		sig  os.Signal
	}{
		{"SIGINT", syscall.SIGINT},
		{"SIGTERM", syscall.SIGTERM},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv, hit := blockingServer(t)
			cfgPath, dir := writeTestConfig(t, srv.URL, "session_dir: sessions\n")

			code, stdout, stderr := signalAmele(t, bin, cfgPath, hit, tc.sig)
			// A -1 exit code means the process was killed BY the signal, i.e.
			// amele failed to catch it and a shell would report 128+N.
			assertInterruptedRun(t, code, stdout, stderr, dir)
		})
	}
}

// signalAmele starts `bin run cfgPath task`, waits until the fake provider has
// the request in hand, sends sig, and returns the process's exit code and
// output. Waiting for the provider is what makes the test deterministic: the
// signal provably arrives while the run is in flight, hence after
// signal.NotifyContext registered its handlers.
func signalAmele(t *testing.T, bin, cfgPath string, hit <-chan struct{}, sig os.Signal) (code int, stdout, stderr string) {
	t.Helper()
	var outBuf, errBuf syncBuffer
	//nolint:gosec // G204: bin is this test's own freshly built binary; the rest is constant.
	cmd := exec.Command(bin, "run", cfgPath, "task")
	// The child resolves ${TEST_KEY} through the real environment - it has no
	// injected LookupEnv the way the in-process tests do.
	cmd.Env = append(os.Environ(), "TEST_KEY=sk-test-secret-key")
	cmd.Stdout = &outBuf
	cmd.Stderr = &errBuf
	if err := cmd.Start(); err != nil {
		t.Fatalf("starting amele: %v", err)
	}
	// A failed expectation below must not leak a live process.
	defer func() { _ = cmd.Process.Kill() }()

	select {
	case <-hit:
	case <-time.After(30 * time.Second):
		t.Fatalf("amele never reached the provider; stderr: %s", errBuf.String())
	}
	if err := cmd.Process.Signal(sig); err != nil {
		t.Fatalf("signalling amele: %v", err)
	}

	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatalf("amele did not exit after %v; stderr: %s", sig, errBuf.String())
	}
	return cmd.ProcessState.ExitCode(), outBuf.String(), errBuf.String()
}

// waitFor polls cond until it holds, failing the test if it never does. Used
// where the observable event is a write to a buffer rather than a channel.
func waitFor(t *testing.T, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("timed out waiting for the expected condition")
}
