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
//
// The read runs in a goroutine that is abandoned when the context wins. That
// goroutine holds no lock and blocks nobody: its result channel is buffered,
// so it exits the moment the operating system lets the read return - when a
// writer finally opens the FIFO, when the mount answers, or when the process
// ends, which for every caller in this binary is what a cancelled context
// leads to anyway. That is the deliberate trade: a blocked read cannot be
// interrupted portably, and a goroutine waiting on it costs a run nothing.
func ReadFile(ctx context.Context, path string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	type result struct {
		data []byte
		err  error
	}
	done := make(chan result, 1)
	go func() {
		data, err := os.ReadFile(path) //nolint:gosec // G304: reading the operator-named file is this package's purpose.
		done <- result{data, err}
	}()
	select {
	case r := <-done:
		return r.data, r.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}
