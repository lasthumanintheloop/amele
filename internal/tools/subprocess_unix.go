//go:build unix

package tools

import (
	"os/exec"
	"syscall"
)

// setupProcessGroup puts the child into its own process group and arranges
// for the whole group to be terminated on context cancellation.
// SECURITY: without this, a timed-out `sh -c "a | b"` tool leaves
// grandchildren running past the run's budgets.
func setupProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		// SIGTERM (not SIGKILL) to the whole group: a graceful signal lets
		// a child that manages its own children - a nested `amele` tool
		// spawning its own process group - catch it and clean up its
		// subtree before exiting. Children that ignore SIGTERM are still
		// force-killed when cmd.WaitDelay elapses. Negative PID targets the
		// process group created by Setpgid.
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGTERM)
	}
}
