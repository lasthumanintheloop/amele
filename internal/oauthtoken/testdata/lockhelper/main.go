//go:build unix

// Command lockhelper takes an exclusive flock(2) on the file named by its first
// argument, prints "held", holds the lock for the duration named by its second
// argument and exits.
//
// It exists because the cross-process guarantee of Store.WithLock cannot be
// shown from inside one process by construction: a purely in-memory mutex would
// satisfy every in-process test. It lives under testdata so the go tool leaves
// it out of ./... builds, and it deliberately reimplements the two syscalls
// instead of importing the (unexported) package helpers.
package main

import (
	"fmt"
	"os"
	"time"

	"golang.org/x/sys/unix"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: lockhelper <lockfile> <duration>")
		os.Exit(2)
	}
	d, err := time.ParseDuration(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, "bad duration:", err)
		os.Exit(2)
	}
	f, err := os.OpenFile(os.Args[1], os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		fmt.Fprintln(os.Stderr, "open:", err)
		os.Exit(1)
	}
	defer func() { _ = f.Close() }()
	if err := unix.Flock(int(f.Fd()), unix.LOCK_EX); err != nil {
		fmt.Fprintln(os.Stderr, "flock:", err)
		os.Exit(1)
	}
	fmt.Println("held")
	os.Stdout.Sync() //nolint:errcheck // best effort: the parent reads the handshake or times out.
	time.Sleep(d)
	_ = unix.Flock(int(f.Fd()), unix.LOCK_UN)
}
