//go:build windows

package mcp

import "os/exec"

// setupProcessGroup is a no-op on Windows: there is no process group to join,
// and job objects are out of scope for v0.2 (stdio servers on Windows get
// best-effort termination of the direct child only).
func setupProcessGroup(cmd *exec.Cmd) {
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		return cmd.Process.Kill()
	}
}

// killGroup terminates the server process. Grandchildren are not tracked on
// this platform; see setupProcessGroup.
func killGroup(cmd *exec.Cmd) {
	if cmd.Process == nil {
		return
	}
	_ = cmd.Process.Kill()
	waitForExit(cmd, func() { _ = cmd.Process.Kill() })
}
