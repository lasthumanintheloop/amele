package runlock_test

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/lasthumanintheloop/amele/internal/runlock"
)

// lockPath returns a fresh lock path inside a per-test temp directory.
func lockPath(t *testing.T) string {
	t.Helper()
	return filepath.Join(t.TempDir(), "agent.yaml.lock")
}

// TestAcquireAndRelease pins the happy path: the lock file is created with
// owner-only permissions, and releasing it lets the next caller in.
func TestAcquireAndRelease(t *testing.T) {
	path := lockPath(t)

	release, err := runlock.Acquire(path)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("lock file was not created: %v", err)
	}
	// SECURITY: the lock file sits next to the config, which may live in a
	// shared directory; 0600 keeps a foreign user from creating it first and
	// pinning the lock (or from holding it open).
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("lock file mode = %o, want 600", perm)
	}

	release()

	// The second Acquire proves release() really dropped the lock rather than
	// only closing a duplicate descriptor.
	release2, err := runlock.Acquire(path)
	if err != nil {
		t.Fatalf("Acquire after release: %v", err)
	}
	release2()
}

// TestAcquireTwiceReportsHeld pins the whole point of the package: while one
// holder is alive, every other Acquire fails immediately with ErrHeld instead
// of blocking. Two Acquire calls in one process are a faithful stand-in for two
// processes - each call opens its own file description, which is the unit flock
// locks.
func TestAcquireTwiceReportsHeld(t *testing.T) {
	path := lockPath(t)

	release, err := runlock.Acquire(path)
	if err != nil {
		t.Fatalf("first Acquire: %v", err)
	}
	defer release()

	release2, err := runlock.Acquire(path)
	if !errors.Is(err, runlock.ErrHeld) {
		t.Fatalf("second Acquire err = %v, want ErrHeld", err)
	}
	if release2 != nil {
		t.Error("a failed Acquire must not return a release function")
	}
}

// TestReleaseIsIdempotent pins that a double release is safe: callers wire it
// with defer and may also call it on an early return path, and a second call
// must never unlock a lock a later Acquire has since taken.
func TestReleaseIsIdempotent(t *testing.T) {
	path := lockPath(t)

	release, err := runlock.Acquire(path)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	release()

	held, err := runlock.Acquire(path)
	if err != nil {
		t.Fatalf("Acquire after release: %v", err)
	}
	defer held()

	release() // second call on the stale holder: must be a no-op

	if _, err := runlock.Acquire(path); !errors.Is(err, runlock.ErrHeld) {
		t.Fatalf("a repeated release dropped someone else's lock: err = %v, want ErrHeld", err)
	}
}

// TestLockFileSurvivesRelease pins the documented no-unlink contract: the file
// stays on disk after release, because deleting it would race a concurrent
// process that already holds the same (now unlinked) inode open.
func TestLockFileSurvivesRelease(t *testing.T) {
	path := lockPath(t)

	release, err := runlock.Acquire(path)
	if err != nil {
		t.Fatalf("Acquire: %v", err)
	}
	release()

	if _, err := os.Stat(path); err != nil {
		t.Errorf("lock file must survive release: %v", err)
	}
}

// TestAcquireOpenErrorIsNotHeld pins the error taxonomy: an unusable lock path
// is a configuration problem, not contention, so callers can map it to exit 2
// instead of reporting a phantom concurrent run.
func TestAcquireOpenErrorIsNotHeld(t *testing.T) {
	tests := []struct {
		name string
		path func(t *testing.T) string
	}{
		{
			name: "missing directory",
			path: func(t *testing.T) string {
				return filepath.Join(t.TempDir(), "nope", "agent.yaml.lock")
			},
		},
		{
			name: "path is a directory",
			path: func(t *testing.T) string {
				p := filepath.Join(t.TempDir(), "agent.yaml.lock")
				if err := os.Mkdir(p, 0o750); err != nil {
					t.Fatal(err)
				}
				return p
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			release, err := runlock.Acquire(tc.path(t))
			if err == nil {
				release()
				t.Fatal("Acquire succeeded on an unusable lock path")
			}
			if errors.Is(err, runlock.ErrHeld) {
				t.Errorf("err = %v, must not be ErrHeld", err)
			}
		})
	}
}
