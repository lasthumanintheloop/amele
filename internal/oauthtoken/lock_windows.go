//go:build windows

package oauthtoken

import (
	"fmt"
	"os"

	"golang.org/x/sys/windows"
)

// lockRegionLen is the byte range locked on Windows. The lock file is empty, so
// the range is nominal: LockFileEx locks byte ranges rather than whole files and
// a range may extend past end-of-file, so every process locking the same first
// byte gets the mutual exclusion flock(2) gives for free elsewhere.
const lockRegionLen = 1

// lockExclusive blocks until it holds an exclusive lock on f.
//
// LOCKFILE_EXCLUSIVE_LOCK without LOCKFILE_FAIL_IMMEDIATELY is the blocking
// form; like flock(2) it has no deadline, so the caller races it against the
// context on a separate goroutine. Windows file locks are mandatory and are
// released when the handle closes, which also covers a crashed process.
func lockExclusive(f *os.File) error {
	ol := new(windows.Overlapped)
	if err := windows.LockFileEx(windows.Handle(f.Fd()), windows.LOCKFILE_EXCLUSIVE_LOCK, 0, lockRegionLen, 0, ol); err != nil {
		return fmt.Errorf("locking %s: %w", f.Name(), err)
	}
	return nil
}

// unlockFile releases the lock held on f.
func unlockFile(f *os.File) error {
	ol := new(windows.Overlapped)
	if err := windows.UnlockFileEx(windows.Handle(f.Fd()), 0, lockRegionLen, 0, ol); err != nil {
		return fmt.Errorf("unlocking %s: %w", f.Name(), err)
	}
	return nil
}
