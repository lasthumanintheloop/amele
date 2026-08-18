package mcp

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"sync"
	"time"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

const (
	// MaxMessageBytes caps a single JSON-RPC message on both transports. It
	// matches the pre-decode cap the provider client uses (internal/llm):
	// nothing an unreviewed peer sends is decoded before its size is known.
	// SECURITY: the cap is enforced while reading, so an endless "message"
	// costs bounded memory instead of the machine's RAM.
	MaxMessageBytes = 8 << 20
	// stdioGrace is how long a stdio server gets to exit after SIGTERM before
	// the group is SIGKILLed. It mirrors the subprocess tool's WaitDelay: long
	// enough for a server to flush and close its own children, short enough
	// that a run's shutdown stays snappy.
	stdioGrace = 5 * time.Second
	// maxStderrBytes bounds what a server's stderr can cost. Diagnostics are
	// worth keeping; a server that loops printing is not.
	maxStderrBytes = 16 << 10
	// msgTooLargeText is the exact wording every size-cap failure carries. It is
	// a constant because it is load-bearing: IsMessageTooLarge matches on it
	// when the error chain has been severed (see that function).
	msgTooLargeText = "message exceeds the size cap"
)

// errMessageTooLarge is returned by both transports when a peer's message
// exceeds MaxMessageBytes. It is a sentinel so callers can match on it instead
// of on wording - but see IsMessageTooLarge for the case where the chain does
// not survive.
var errMessageTooLarge = errors.New(msgTooLargeText)

// IsMessageTooLarge reports whether err was caused by a peer exceeding
// MaxMessageBytes on either transport.
//
// It falls back to a substring match on purpose: on the Streamable HTTP path
// the SDK formats read failures with %v (mcp/streamable.go), which severs the
// %w chain, so errors.Is alone would miss every HTTP cap failure. The wording
// is pinned by msgTooLargeText, which both the stdio reader and the HTTP body
// wrapper embed.
func IsMessageTooLarge(err error) bool {
	if err == nil {
		return false
	}
	return errors.Is(err, errMessageTooLarge) || strings.Contains(err.Error(), msgTooLargeText)
}

// newStdioTransport starts argv in dir as its own process group with a minimal
// environment and returns an SDK transport over the child's stdio plus a kill
// function that terminates the whole group.
//
// The returned kill function closes the child's stdin, signals the whole
// process group and reaps the process, so no goroutine and no grandchild
// outlives the run. Callers must call it even when the session closed cleanly.
// It is idempotent and safe to call concurrently (guarded by a sync.Once), and
// it blocks until the child is reaped: worst case about stdioGrace for the
// explicit SIGTERM-then-SIGKILL sequence plus cmd.WaitDelay for a
// cancellation-driven kill already in flight, so roughly 10s.
//
// SECURITY: the child never inherits amele's environment (see childEnv) and
// never shares amele's process group (see setupProcessGroup).
func newStdioTransport(ctx context.Context, argv []string, dir string, allow []string, env func(string) (string, bool), stderr io.Writer) (sdk.Transport, func(), error) {
	if len(argv) == 0 {
		return nil, nil, errors.New("empty command")
	}
	// argv comes from the operator's YAML, never from the model: the same trust
	// boundary the subprocess tool uses.
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...) //nolint:gosec // argv is operator config, not model input (docs/threat-model.md)
	cmd.Dir = dir
	cmd.Env = childEnv(allow, env)
	cmd.Stderr = &cappedWriter{w: stderr, max: maxStderrBytes}
	setupProcessGroup(cmd)
	// WaitDelay bounds the ctx-cancellation path (cmd.Cancel sends SIGTERM to
	// the group); the explicit kill path below has its own grace timer.
	cmd.WaitDelay = stdioGrace

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("stdin pipe for %q: %w", argv[0], err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("stdout pipe for %q: %w", argv[0], err)
	}
	if err := cmd.Start(); err != nil {
		return nil, nil, fmt.Errorf("starting %q: %w", argv[0], err)
	}

	var killOnce sync.Once
	// sync.Once, not a bare closure: Connect's error path and the run's shutdown
	// may both call kill, and two concurrent cmd.Wait calls are a data race.
	kill := func() {
		killOnce.Do(func() {
			// Closing stdin is the polite shutdown an MCP server expects; the
			// signals below are the guarantee that it happens anyway.
			_ = stdin.Close()
			killGroup(cmd)
		})
	}
	return &sdk.IOTransport{
		Reader: &lineCappedReader{r: stdout, max: MaxMessageBytes},
		Writer: stdin,
	}, kill, nil
}

// childEnv builds the environment of a stdio MCP server: PATH, HOME and LANG
// plus the operator's allowlist, in that order, deduplicated, skipping
// variables the lookup does not define.
//
// SECURITY: this is deliberately stricter than the subprocess tool, which
// inherits amele's environment. An MCP server is third-party code addressed by
// name in a config file; it must not be able to read the provider API key just
// because amele can.
func childEnv(allow []string, env func(string) (string, bool)) []string {
	names := append([]string{"PATH", "HOME", "LANG"}, allow...)
	seen := make(map[string]bool, len(names))
	out := make([]string, 0, len(names))
	for _, name := range names {
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		if v, ok := env(name); ok {
			out = append(out, name+"="+v)
		}
	}
	return out
}

// lineCappedReader passes a child's stdout through unchanged until a single
// newline-delimited message exceeds max, at which point every further read
// fails with errMessageTooLarge.
//
// The SDK frames stdio messages by newline, so "bytes since the last \n" is
// exactly one message; failing the reader fails the connection, which is the
// outcome we want (the peer is either broken or hostile). The terminating
// newline counts toward the cap, so the payload budget is max-1 bytes - an
// off-by-one nobody can notice at 8 MiB.
type lineCappedReader struct {
	r     io.ReadCloser
	max   int
	since int
	err   error
}

// Read implements io.Reader.
func (r *lineCappedReader) Read(p []byte) (int, error) {
	if r.err != nil {
		return 0, r.err
	}
	n, err := r.r.Read(p)
	// Account per line, not per chunk: a chunk that both finishes a line and
	// starts the next one must charge the finished line its full length before
	// the counter resets, or an oversized message whose last chunk happens to
	// carry the newline slips through.
	for chunk := p[:n]; len(chunk) > 0; {
		i := bytes.IndexByte(chunk, '\n')
		if i < 0 {
			r.since += len(chunk)
			break
		}
		r.since += i + 1
		if r.since > r.max {
			break
		}
		r.since = 0
		chunk = chunk[i+1:]
	}
	if r.since > r.max {
		r.err = fmt.Errorf("reading from MCP server: %w (%d bytes in one message, cap %d)", errMessageTooLarge, r.since, r.max)
		return 0, r.err
	}
	return n, err
}

// Close implements io.Closer.
func (r *lineCappedReader) Close() error { return r.r.Close() }

// cappedWriter forwards at most max bytes to w and silently drops the rest. It
// wraps the operator's stderr sink, which is responsible for redaction.
//
// cappedWriter itself is not safe for concurrent use, and it does not
// serialize writes to w: when one sink is shared by several servers, that sink
// must be safe for concurrent use (os/exec writes each child's stderr from its
// own goroutine).
type cappedWriter struct {
	w       io.Writer
	max     int
	written int
}

// Write implements io.Writer. It never reports a short write: dropping a
// chatty server's output must not look like an I/O failure to os/exec.
func (c *cappedWriter) Write(p []byte) (int, error) {
	room := c.max - c.written
	if room <= 0 {
		return len(p), nil
	}
	out := p
	if len(out) > room {
		out = out[:room]
	}
	n, err := c.w.Write(out)
	c.written += n
	if err != nil {
		return n, err
	}
	return len(p), nil
}

// waitForExit reaps cmd, calling force once if it has not exited within
// stdioGrace. The goroutine it starts is owned by this call and always exits:
// force is a signal the process cannot ignore, so cmd.Wait returns.
func waitForExit(cmd *exec.Cmd, force func()) {
	done := make(chan struct{})
	go func() {
		_ = cmd.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(stdioGrace):
		force()
		<-done
	}
}
