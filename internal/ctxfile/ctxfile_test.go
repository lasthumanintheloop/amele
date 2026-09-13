package ctxfile

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestReadFileReadsAWholeFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "f.txt")
	if err := os.WriteFile(path, []byte("hello\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := ReadFile(context.Background(), path)
	if err != nil || string(got) != "hello\n" {
		t.Fatalf("ReadFile = %q, %v", got, err)
	}
}

// The file error is os.ReadFile's own, so a caller's "reading X: ..." phrasing
// and its errors.Is(err, fs.ErrNotExist) checks keep working.
func TestReadFileReturnsTheOSError(t *testing.T) {
	_, err := ReadFile(context.Background(), filepath.Join(t.TempDir(), "missing"))
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("err = %v, want a not-exist error", err)
	}
}

func TestReadFileRefusesADoneContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	path := filepath.Join(t.TempDir(), "f.txt")
	if err := os.WriteFile(path, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadFile(ctx, path); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

func TestStat(t *testing.T) {
	path := filepath.Join(t.TempDir(), "f.txt")
	if err := os.WriteFile(path, []byte("hello"), 0o600); err != nil {
		t.Fatal(err)
	}
	info, err := Stat(context.Background(), path)
	if err != nil || info.Size() != 5 {
		t.Fatalf("Stat = %v, %v", info, err)
	}
	if _, err := Stat(context.Background(), filepath.Join(t.TempDir(), "missing")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("err = %v, want a not-exist error", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := Stat(ctx, path); !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}
