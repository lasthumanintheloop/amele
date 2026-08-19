//go:build unix

package oauthtoken

import (
	"errors"
	"fmt"
	"os"

	"golang.org/x/sys/unix"
)

// lockExclusive blocks until it holds an exclusive lock on f.
//
// flock(2) is used rather than fcntl(2) record locks because its ownership is
// the open file description: closing one unrelated descriptor for the same
// file, which fcntl locks treat as "drop everything", cannot silently release
// this lock. There is no deadline variant, which is why the caller runs this on
// its own goroutine and races it against the context.
func lockExclusive(f *os.File) error {
	for {
		err := unix.Flock(int(f.Fd()), unix.LOCK_EX)
		// A signal (Go's own preemption signals included) can interrupt the
		// blocking call; that is not a failure to lock, so retry.
		if errors.Is(err, unix.EINTR) {
			continue
		}
		if err != nil {
			return fmt.Errorf("locking %s: %w", f.Name(), err)
		}
		return nil
	}
}

// unlockFile releases the lock held on f. Closing f would release it too, but
// the explicit call keeps the release ordered before the close so a caller that
// reuses the descriptor still behaves.
func unlockFile(f *os.File) error {
	if err := unix.Flock(int(f.Fd()), unix.LOCK_UN); err != nil {
		return fmt.Errorf("unlocking %s: %w", f.Name(), err)
	}
	return nil
}
