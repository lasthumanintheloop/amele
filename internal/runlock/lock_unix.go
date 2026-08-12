//go:build unix

package runlock

import (
	"errors"
	"os"
	"syscall"
)

// tryLock takes the exclusive flock on f without blocking, translating the
// "someone else has it" errno into ErrHeld.
//
// flock is used rather than fcntl/POSIX record locks because its ownership
// unit is the open file description: the lock belongs to this descriptor and
// dies with the process, so a killed run (SIGKILL, OOM, power loss) never
// leaves a lock nobody can clear. POSIX locks are per process-and-inode and
// are silently dropped by an unrelated close of the same file, which would
// make the guard quietly unreliable.
func tryLock(f *os.File) error {
	err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
	// Linux reports EAGAIN and the BSDs EWOULDBLOCK; they are the same value
	// on Linux, and checking both keeps the meaning explicit.
	if errors.Is(err, syscall.EWOULDBLOCK) || errors.Is(err, syscall.EAGAIN) {
		return ErrHeld
	}
	return err
}

// unlock drops the flock held through f.
func unlock(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
}
