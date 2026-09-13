package llm

// This file holds the server-sent-events reader the three streaming paths
// share. It reads the transport shape only - `event:` and `data:` lines,
// blank-line dispatch - and leaves every payload to the wire that asked for
// it.

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"strings"
)

// sseEvent is one dispatched event: the event name (empty when the stream
// sends none, as the OpenAI wire does) and the data lines joined with "\n".
type sseEvent struct {
	event string
	data  string
}

// readSSE reads a text/event-stream body and calls fn once per event, in
// order, until the body ends or fn returns an error (which readSSE returns
// as-is, so a wire can stop the read on its own terminator or error event).
//
// The body is bounded by maxResponseBody like every other success body: a
// stream that never ends is cut there and surfaces as ErrProvider, the same
// way an oversized JSON body surfaces as a decode error. Only the three
// things the SSE format promises are interpreted - `event:`, `data:` (several
// per event, joined with a newline) and the blank line that dispatches;
// comment lines (`:`) and every other field (`id:`, `retry:`) are skipped.
func readSSE(body io.Reader, fn func(sseEvent) error) error {
	limited := &limitedReader{r: io.LimitReader(body, maxResponseBody)}
	r := bufio.NewReader(limited)
	var ev sseEvent
	var data []string
	pending := false
	dispatch := func() error {
		if !pending {
			return nil
		}
		ev.data = strings.Join(data, "\n")
		err := fn(ev)
		ev, data, pending = sseEvent{}, nil, false
		return err
	}
	for {
		line, err := r.ReadString('\n')
		if line != "" {
			line = strings.TrimRight(line, "\r\n")
			switch {
			case line == "":
				if err := dispatch(); err != nil {
					return err
				}
			case strings.HasPrefix(line, ":"):
				// A comment; keep-alives are sent this way.
			case strings.HasPrefix(line, "event:"):
				ev.event = strings.TrimPrefix(strings.TrimPrefix(line, "event:"), " ")
				pending = true
			case strings.HasPrefix(line, "data:"):
				data = append(data, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
				pending = true
			}
		}
		if err != nil {
			if !errors.Is(err, io.EOF) {
				return fmt.Errorf("%w: reading stream: %v", ErrProvider, err)
			}
			if limited.n >= maxResponseBody {
				return fmt.Errorf("%w: stream exceeded %d bytes", ErrProvider, maxResponseBody)
			}
			// A stream that ends without a final blank line still delivers
			// its last event, as the format allows.
			return dispatch()
		}
	}
}

// limitedReader counts what passed through so a cut at the bound can be told
// from an ordinary end of stream.
type limitedReader struct {
	r io.Reader
	n int64
}

func (l *limitedReader) Read(p []byte) (int, error) {
	n, err := l.r.Read(p)
	l.n += int64(n)
	return n, err
}
