//go:build windows

package tools

import "os/exec"

// setupProcessGroup is a no-op on Windows: there is no POSIX process group
// to target, and exec.CommandContext's default kill covers the direct child.
// Full job-object based tree termination is tracked for the Windows support
// milestone.
func setupProcessGroup(_ *exec.Cmd) {}
