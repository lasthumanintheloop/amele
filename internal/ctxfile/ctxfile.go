// Package ctxfile reads operator-supplied files without ignoring the run's
// context. The binary opens a handful of paths the operator names - the config
// file, a system_prompt_file, the --resume session log - and a plain
// os.ReadFile of any of them cannot observe limits.timeout or a SIGTERM: a
// FIFO with no writer, or a hung network mount, blocks the process past every
// deadline it armed (issue #29). ReadFile is the one entry point, and the
// whole package is the answer to "how does a file read stop when the run
// does?".
package ctxfile

import (
	"context"
	"os"
)

// ReadFile returns the whole file at path, or the context's error as soon as
// the context ends - whichever comes first. A context that is already done
// reads nothing. The file error, when there is one, is os.ReadFile's own,
// unwrapped, so callers keep their existing phrasing around it.
func ReadFile(ctx context.Context, path string) ([]byte, error) {
	return await(ctx, func() ([]byte, error) {
		return os.ReadFile(path) //nolint:gosec // G304: reading the operator-named file is this package's purpose.
	})
}

// Stat is os.Stat under the same rule as ReadFile: the context's error as
// soon as the context ends. A metadata lookup on a hung network mount blocks
// exactly like a read does, and the CLI stats every config argument before it
// reads one (the directory shortcut).
func Stat(ctx context.Context, path string) (os.FileInfo, error) {
	return await(ctx, func() (os.FileInfo, error) {
		return os.Stat(path)
	})
}

// await runs one blocking file operation and returns its result, or the
// context's error if the context ends first.
//
// The operation runs in a goroutine that is ABANDONED when the context wins,
// which is the one place this binary departs from "every goroutine has a
// closing path": a blocked open(2), read(2) or stat(2) cannot be interrupted
// portably (no descriptor exists yet to close, and Windows has no signal to
// send), so the choice is between abandoning the goroutine and abandoning the
// deadline. The goroutine holds no lock and blocks nobody - its result channel
// is buffered - and it exits the moment the operating system lets the call
// return: when a writer opens the FIFO, when the mount answers, or when the
// process ends. Every caller in this binary is on its way to exit once its
// context is cancelled, and a run touches a handful of operator files, so the
// leak is bounded by that handful and lives at most as long as the process.
// The tests release their FIFOs in cleanup so the test binary, which does not
// exit per test, sees the same bound.
func await[T any](ctx context.Context, op func() (T, error)) (T, error) {
	var zero T
	if err := ctx.Err(); err != nil {
		return zero, err
	}
	type result struct {
		value T
		err   error
	}
	done := make(chan result, 1)
	go func() {
		v, err := op()
		done <- result{v, err}
	}()
	select {
	case r := <-done:
		return r.value, r.err
	case <-ctx.Done():
		return zero, ctx.Err()
	}
}
