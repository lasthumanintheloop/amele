//go:build unix

package ctxfile

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// blockingFIFO creates a FIFO nobody writes to: opening it for reading blocks
// until a writer appears, which is the shape of the hang issue #29 is about.
// The cleanup opens the writer side (non-blocking, so it cannot itself hang
// when no reader is left) and closes it, which lets the abandoned read return
// with EOF and its goroutine exit before the test process does.
func blockingFIFO(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "hang.fifo")
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

// TestReadFileStopsWhenTheContextDoes is the issue #29 scenario: the read
// blocks forever on its own, and the context is what ends it.
func TestReadFileStopsWhenTheContextDoes(t *testing.T) {
	path := blockingFIFO(t)
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := ReadFile(ctx, path)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context.DeadlineExceeded", err)
	}
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("the read outlived its context by %s", elapsed)
	}
}
