//go:build !unix

package runlock

import (
	"errors"
	"os"
)

// errUnsupported is returned on platforms with no flock. It is deliberately
// NOT ErrHeld: the caller must report "this platform cannot lock", not invent
// a concurrent run that does not exist.
var errUnsupported = errors.New("run lock is not supported on this platform")

// tryLock refuses on every non-unix platform.
//
// Windows can lock a byte range with LockFileEx, but that call lives in
// golang.org/x/sys/windows (the standard syscall package does not export it),
// and a lock implementation no CI job ever executes is worse than an honest
// refusal: a wrong flag combination there would silently permit exactly the
// overlapping runs this package exists to prevent. Failing loudly keeps
// `lock: true` a promise instead of a maybe; it becomes a real lock on this
// platform the day the project ships and tests a Windows build.
func tryLock(_ *os.File) error { return errUnsupported }

// unlock is unreachable: nothing on this platform ever acquires the lock.
func unlock(_ *os.File) error { return nil }
