//go:build unix

package mcp

import (
	"bufio"
	"context"
	"errors"
	"io"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// TestStdioKillsProcessGroup is the reason this package does not use
// sdk.CommandTransport: killing the direct child leaves its children running.
func TestStdioKillsProcessGroup(t *testing.T) {
	bin := buildTestServer(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	pr, pw := io.Pipe()
	defer func() { _ = pr.Close() }()

	tr, kill, err := newStdioTransport(ctx, []string{bin, "-spawn-child"}, t.TempDir(), nil, lookupPATH(), pw)
	if err != nil {
		t.Fatalf("newStdioTransport: %v", err)
	}
	sess, err := sdk.NewClient(&sdk.Implementation{Name: "amele-test", Version: "0"}, nil).Connect(ctx, tr, nil)
	if err != nil {
		kill()
		t.Fatalf("connect: %v", err)
	}

	pid := readChildPID(t, pr)
	if err := syscall.Kill(pid, 0); err != nil {
		t.Fatalf("grandchild %d is not running before kill: %v", pid, err)
	}

	_ = sess.Close()
	kill()

	deadline := time.Now().Add(6 * time.Second)
	for {
		err := syscall.Kill(pid, 0)
		if errors.Is(err, syscall.ESRCH) {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("grandchild %d still alive 6s after kill (err=%v)", pid, err)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// readChildPID reads the "child=<pid>" line the test server writes to stderr.
func readChildPID(t *testing.T, r io.Reader) int {
	t.Helper()
	type result struct {
		pid int
		err error
	}
	ch := make(chan result, 1)
	go func() {
		sc := bufio.NewScanner(r)
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if rest, ok := strings.CutPrefix(line, "child="); ok {
				pid, err := strconv.Atoi(rest)
				ch <- result{pid, err}
				return
			}
		}
		ch <- result{0, io.EOF}
	}()
	select {
	case res := <-ch:
		if res.err != nil {
			t.Fatalf("reading child pid: %v", res.err)
		}
		return res.pid
	case <-time.After(10 * time.Second):
		t.Fatal("timed out waiting for the child pid on stderr")
		return 0
	}
}
