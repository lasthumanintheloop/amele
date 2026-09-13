//go:build unix

package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

// blockingFIFO creates a FIFO with no writer: opening it for reading blocks
// until one appears, which is the hang issue #29 is about (a --resume path or
// a config path on a hung mount behaves the same way). The cleanup opens the
// writer side non-blocking and closes it, so the read amele abandoned gets its
// EOF and its goroutine exits before the test process does.
func blockingFIFO(t *testing.T, name string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := syscall.Mkfifo(path, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if w, err := os.OpenFile(path, os.O_WRONLY|syscall.O_NONBLOCK, 0); err == nil { //nolint:gosec // G304: test-owned path.
			_ = w.Close()
		}
	})
	return path
}

// TestRunInterruptedDuringFileRead (issue #29): every operator-supplied file
// the binary opens is read through the run's context, so a path that blocks
// ends with the run instead of outliving limits.timeout or a SIGTERM.
func TestRunInterruptedDuringFileRead(t *testing.T) {
	t.Run("--resume log honors limits.timeout", func(t *testing.T) {
		srv := scriptedServer(t) // the provider must never be reached
		cfgPath, dir := writeResumeConfig(t, srv.URL, "limits:\n  timeout: 100ms\n")
		fifo := blockingFIFO(t, "hang.jsonl")

		start := time.Now()
		code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "--resume", fifo, "go on"}, "")
		if elapsed := time.Since(start); elapsed > 5*time.Second {
			t.Fatalf("the run outlived its timeout by %s", elapsed)
		}
		// CONTRACT: the timeout is a configured budget, so exit 3 - and the
		// session log gets its run_end like every other interrupted read.
		if code != ExitBudgetExceeded {
			t.Fatalf("exit %d, want %d; stderr: %s", code, ExitBudgetExceeded, stderr)
		}
		if stdout != "" {
			t.Errorf("stdout must stay empty: %q", stdout)
		}
		if !strings.Contains(stderr, "budget exceeded") {
			t.Errorf("stderr must name the cause: %q", stderr)
		}
		end, ok := findEvent(sessionEvents(t, dir), "run_end")
		if !ok || end.ExitCode == nil || *end.ExitCode != ExitBudgetExceeded {
			t.Errorf("run_end = %+v, want exit_code %d", end, ExitBudgetExceeded)
		}
	})

	t.Run("--resume log honors a signal", func(t *testing.T) {
		srv := scriptedServer(t)
		cfgPath, dir := writeResumeConfig(t, srv.URL, "")
		fifo := blockingFIFO(t, "hang.jsonl")
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()

		var stdout, stderr bytes.Buffer
		code := run(ctx, []string{"run", cfgPath, "--resume", fifo, "go on"}, strings.NewReader(""), &stdout, &stderr, env(t))
		// A WithTimeout context reports DeadlineExceeded, which the run reads
		// as its budget; the point here is that it ended at all, with the
		// ordinary evidence behind it.
		if code != ExitBudgetExceeded {
			t.Fatalf("exit %d, want %d; stderr: %s", code, ExitBudgetExceeded, stderr.String())
		}
		if _, ok := findEvent(sessionEvents(t, dir), "run_end"); !ok {
			t.Error("session log has no run_end event")
		}
	})

}

// TestRunInterruptedBeforeLoad (issue #29) is the pre-session half: a signal
// during the config or prompt-file read ends the read and exits 1, with the
// cause named and nothing to audit because nothing existed yet.
func TestRunInterruptedBeforeLoad(t *testing.T) {
	t.Run("config file honors a signal", func(t *testing.T) {
		fifo := blockingFIFO(t, "agent.yaml")
		ctx, cancel := context.WithCancel(context.Background())
		go func() {
			time.Sleep(100 * time.Millisecond)
			cancel()
		}()
		defer cancel()

		var stdout, stderr bytes.Buffer
		start := time.Now()
		code := run(ctx, []string{"run", fifo, "task"}, strings.NewReader(""), &stdout, &stderr, env(t))
		if elapsed := time.Since(start); elapsed > 5*time.Second {
			t.Fatalf("the run outlived the signal by %s", elapsed)
		}
		// CONTRACT (docs/contracts/cli.md, Signals): an interrupted run is exit
		// 1, even when the interruption came before the config was read. No
		// session exists yet, so there is nothing to audit - but the cause is
		// named.
		if code != ExitTaskFailed {
			t.Fatalf("exit %d, want %d; stderr: %s", code, ExitTaskFailed, stderr.String())
		}
		if !strings.Contains(stderr.String(), "run interrupted: context canceled") {
			t.Errorf("stderr must name the cause: %q", stderr.String())
		}
		if stdout.String() != "" {
			t.Errorf("stdout must stay empty: %q", stdout.String())
		}
	})

	t.Run("system_prompt_file honors a signal", func(t *testing.T) {
		fifo := blockingFIFO(t, "prompt.txt")
		dir := t.TempDir()
		cfgPath := filepath.Join(dir, "agent.yaml")
		yaml := "model: m\nprovider:\n  base_url: http://127.0.0.1:1/v1\n  api_key: ${TEST_KEY}\nsystem_prompt_file: " + fifo + "\n"
		if err := os.WriteFile(cfgPath, []byte(yaml), 0o600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		go func() {
			time.Sleep(100 * time.Millisecond)
			cancel()
		}()
		defer cancel()

		var stdout, stderr bytes.Buffer
		code := run(ctx, []string{"run", cfgPath, "task"}, strings.NewReader(""), &stdout, &stderr, env(t))
		if code != ExitTaskFailed {
			t.Fatalf("exit %d, want %d; stderr: %s", code, ExitTaskFailed, stderr.String())
		}
		if !strings.Contains(stderr.String(), "run interrupted: context canceled") {
			t.Errorf("stderr must name the cause: %q", stderr.String())
		}
	})
}
