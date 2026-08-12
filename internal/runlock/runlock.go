// Package runlock makes a run single-flight: it takes a non-blocking advisory
// lock on a file so a second copy of the same agent cannot start while the
// first one is still working.
//
// It exists for the canonical headless deployment. A crontab line fires
// `amele run config.yaml` every N minutes; when one run hangs past the
// interval, the next one starts anyway and the two interleave - duplicate
// emails, a workspace edited from two sides. With `lock: true` the second run
// finds the lock held and exits immediately (exit code 7) instead of racing.
//
// CONTRACT: the lock is EXCLUSIVE and NON-BLOCKING. Acquire never waits: it
// either takes the lock or reports ErrHeld right away, because a cron job that
// queues up behind its predecessor eventually queues up a hundred of them.
//
// SECURITY: the lock is ADVISORY. It only excludes processes that cooperate by
// locking the same file - i.e. other amele runs. It is not a permission system
// and it does not protect the workspace from anything else on the machine.
// The lock file is created 0600 so a foreign user cannot pre-create it.
package runlock

import (
	"errors"
	"fmt"
	"os"
	"sync"
)

// ErrHeld is returned by Acquire when another process already holds the lock.
// It is a sentinel so callers can tell contention (a normal, expected outcome
// of overlapping cron runs) from a broken lock path, which is a config error.
var ErrHeld = errors.New("run lock is held by another process")

// Acquire takes the exclusive advisory lock on the file at path, creating it
// with mode 0600 if needed, and returns the function that releases it.
//
// It never blocks. If another process holds the lock, Acquire returns a nil
// release function and an error matching ErrHeld; any other failure (an
// unusable path, a filesystem without lock support, a platform with no flock)
// is returned wrapped and does NOT match ErrHeld.
//
// The returned release function is idempotent and safe to call from any
// goroutine: the second and later calls do nothing, so a caller may both
// `defer release()` and release early without risk of dropping a lock that a
// later Acquire has taken in the meantime. Acquire itself is safe for
// concurrent use.
//
// The lock file is deliberately NOT deleted on release. Unlinking it would
// race: another process may already have the same inode open and locked, and
// removing the name lets a third process create a fresh file and lock that
// one instead - two "holders" of a lock that no longer refers to one object.
// A stale empty file costs nothing; a broken mutual exclusion costs a double
// cron run, which is exactly what this package exists to prevent.
func Acquire(path string) (release func(), err error) {
	// O_RDWR (not O_RDONLY): some platforms require write access on the
	// descriptor an exclusive lock is taken through. Nothing is ever written
	// to the file - its content is irrelevant, only the lock on it matters.
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0o600) //nolint:gosec // G304: the lock path is derived from the user's own config path, which is the point.
	if err != nil {
		return nil, fmt.Errorf("opening lock file %s: %w", path, err)
	}
	if err := tryLock(f); err != nil {
		_ = f.Close()
		if errors.Is(err, ErrHeld) {
			return nil, err
		}
		return nil, fmt.Errorf("locking %s: %w", path, err)
	}

	var once sync.Once
	return func() {
		once.Do(func() {
			// Unlock explicitly before closing. Closing the descriptor would
			// release the lock on its own, but an explicit unlock states the
			// intent and keeps the ordering obvious if a future change ever
			// hands the descriptor somewhere else.
			_ = unlock(f)
			_ = f.Close()
		})
	}, nil
}
