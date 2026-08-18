//go:build unix

package mcp

import (
	"os/exec"
	"syscall"
)

// setupProcessGroup puts the MCP server into its own process group and makes
// context cancellation terminate that whole group.
//
// This duplicates internal/tools/subprocess_unix.go on purpose: internal/mcp
// stays self-contained rather than depending on another package's unexported
// helpers, and the two may diverge (an MCP server is long-lived, a tool is
// not).
// SECURITY: without a group, a server that spawns helpers leaves them running
// past the run's budgets.
func setupProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		// Negative PID targets the group created by Setpgid. SIGTERM first so
		// a server can clean up its own subtree; cmd.WaitDelay force-kills.
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
	}
}

// killGroup terminates the server's whole process group: SIGTERM, then SIGKILL
// if it has not exited within stdioGrace. It reaps the process, so it must not
// be paired with a separate cmd.Wait.
func killGroup(cmd *exec.Cmd) {
	if cmd.Process == nil {
		return
	}
	pgid := -cmd.Process.Pid
	_ = syscall.Kill(pgid, syscall.SIGTERM)
	waitForExit(cmd, func() { _ = syscall.Kill(pgid, syscall.SIGKILL) })
}
