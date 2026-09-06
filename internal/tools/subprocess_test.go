package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/lasthumanintheloop/amele/internal/config"
)

// TestSubprocessRunsScriptOutsideWorkspace: cmd.Dir is the workspace, but an
// absolute Command[0] (as produced by config path resolution) must execute a
// script that lives outside it.
func TestSubprocessRunsScriptOutsideWorkspace(t *testing.T) {
	packDir := t.TempDir()
	workspace := t.TempDir() // deliberately NOT where the script lives
	script := filepath.Join(packDir, "hello.sh")
	if err := os.WriteFile(script, []byte("#!/bin/sh\necho pack-hello\n"), 0o700); err != nil { //nolint:gosec // G306: the test needs an executable script; 0700 is the point
		t.Fatal(err)
	}
	tool := NewSubprocess(config.SubprocessTool{
		Name:        "hello",
		Description: "d",
		Command:     []string{script},
	}, workspace, SubprocessOptions{})
	out, err := tool.Invoke(context.Background(), "")
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if !strings.Contains(out, "pack-hello") {
		t.Errorf("output %q does not contain pack-hello", out)
	}
}
