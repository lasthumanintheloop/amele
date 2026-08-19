package mcp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"github.com/modelcontextprotocol/go-sdk/auth"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// maxRedirects is how many redirects one request may FOLLOW before amele gives
// up: the len(via) check below refuses the hop that would be redirect number
// maxRedirects+1. Any chain longer than this is a misconfiguration, not a
// feature.
const maxRedirects = 3

// noDeadlineTimeout bounds any request that arrives with no deadline of its
// own. With the standalone SSE stream disabled, every request the SDK sends on
// this transport carries a context deadline (initialize and tools/list run
// under the connect window, tools/call under the call timeout) EXCEPT the
// session-ending DELETE, which the SDK issues from Close without a context.
// Left unbounded, a server that accepts that DELETE and never answers pins a
// goroutine and a connection for the rest of the run - the mid-run reconnect
// path tears down the old session exactly this way. 10 s is generous for a
// request whose answer nobody reads.
const noDeadlineTimeout = 10 * time.Second

// newHTTPTransport builds a Streamable HTTP transport for one remote MCP
// server: the operator's headers on every request, no standalone SSE stream,
// no SDK-level reconnects, same-origin redirects only and a response body cap.
//
// handler, when non-nil, makes the transport an OAuth client: the SDK asks it
// for a token before every request and calls it again on a 401/403. nil keeps
// slice 1's behaviour, where the only credentials are the operator's static
// headers.
//
// It returns the transport plus a release function that drops the client's
// idle connections - the HTTP analogue of the stdio kill: a fleet worker that
// closed its session must not keep a warm socket to the server for the rest
// of its life. An error is returned only when rawURL cannot be parsed or is
// not http(s); no network I/O happens here.
func newHTTPTransport(rawURL string, headers map[string]string, handler auth.OAuthHandler) (*sdk.StreamableClientTransport, func(), error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, nil, fmt.Errorf("parsing url %q: %w", rawURL, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" || u.Host == "" {
		return nil, nil, fmt.Errorf("url %q: want an absolute http(s) url", rawURL)
	}

	// Comma-ok, never a bare assertion: a library must not panic because some
	// other package replaced http.DefaultTransport (CLAUDE.md 5.3). The
	// fallback keeps proxy support, which is the setting operators actually
	// depend on behind a corporate egress.
	base, ok := http.DefaultTransport.(*http.Transport)
	if !ok {
		base = &http.Transport{Proxy: http.ProxyFromEnvironment}
	} else {
		base = base.Clone()
	}
	client := &http.Client{
		// No client-level timeout: an MCP call's deadline is the run's ctx,
		// and a streaming response may legitimately outlive any fixed bound.
		// The deadlineRoundTripper below covers the one request that carries
		// no context deadline at all.
		Timeout: 0,
		Transport: &deadlineRoundTripper{
			timeout: noDeadlineTimeout,
			next: &headerRoundTripper{
				next:    &cappedBodyRoundTripper{next: base, max: MaxMessageBytes},
				headers: headers,
			},
		},
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			// SECURITY: the operator's credentials travel in headers; a
			// redirect to another origin would hand them to a third party.
			if req.URL.Scheme != u.Scheme || req.URL.Host != u.Host {
				return fmt.Errorf("redirect to %s://%s refused: not the configured origin", req.URL.Scheme, req.URL.Host)
			}
			if len(via) >= maxRedirects {
				return errors.New("too many redirects")
			}
			return nil
		},
	}
	return &sdk.StreamableClientTransport{
		Endpoint:   rawURL,
		HTTPClient: client,
		// amele owns reconnection (one attempt, at a moment it chooses); an
		// SDK retry loop underneath would double the traffic and hide failures.
		MaxRetries: -1,
		// The toolset is frozen at Connect, so there is nothing a
		// server-initiated stream could tell us that we act on - and a fleet of
		// cron workers must not each hold an idle connection open.
		DisableStandaloneSSE: true,
		OAuthHandler:         handler,
	}, base.CloseIdleConnections, nil
}

// deadlineRoundTripper attaches noDeadlineTimeout to any request whose context
// has no deadline (see the constant for why only the session DELETE qualifies).
// Requests that already carry one pass through untouched.
type deadlineRoundTripper struct {
	next    http.RoundTripper
	timeout time.Duration
}

// RoundTrip implements http.RoundTripper.
func (d *deadlineRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if _, ok := req.Context().Deadline(); ok {
		return d.next.RoundTrip(req)
	}
	ctx, cancel := context.WithCancel(req.Context())
	timer := time.AfterFunc(d.timeout, cancel)
	resp, err := d.next.RoundTrip(req.WithContext(ctx))
	if err != nil {
		timer.Stop()
		cancel()
		return nil, err
	}
	// The body outlives RoundTrip; the context must outlive the body. Close
	// releases both, and the timer keeps the deadline armed for a peer that
	// answers the headers and then stalls the body.
	resp.Body = &cancelOnCloseBody{ReadCloser: resp.Body, stop: func() { timer.Stop(); cancel() }}
	return resp, nil
}

// cancelOnCloseBody cancels its request's context when the body is closed.
type cancelOnCloseBody struct {
	io.ReadCloser
	stop func()
}

// Close implements io.Closer.
func (b *cancelOnCloseBody) Close() error {
	b.stop()
	return b.ReadCloser.Close()
}

// headerRoundTripper adds the operator's static headers to every request.
// Headers that the transport manages itself (Content-Type, Accept,
// Mcp-Session-Id, ...) are rejected by config validation, so Set is safe here.
type headerRoundTripper struct {
	next    http.RoundTripper
	headers map[string]string
}

// RoundTrip implements http.RoundTripper.
func (h *headerRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(h.headers) == 0 {
		return h.next.RoundTrip(req)
	}
	// RoundTrip must not modify the request it is given.
	clone := req.Clone(req.Context())
	for k, v := range h.headers {
		clone.Header.Set(k, v)
	}
	return h.next.RoundTrip(clone)
}

// cappedBodyRoundTripper wraps every response body in a reader that fails past
// max bytes.
// SECURITY: the cap is applied before any decoding, so a server cannot spend
// amele's memory by answering a small request with an endless body.
//
// The cap is per response, not per JSON-RPC message: a text/event-stream POST
// response is billed cumulatively across its events. That is accepted - the
// standalone SSE stream is disabled and a POST response carries the answer to
// one call, for which 8 MiB is generous.
type cappedBodyRoundTripper struct {
	next http.RoundTripper
	max  int64
}

// RoundTrip implements http.RoundTripper.
func (c *cappedBodyRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := c.next.RoundTrip(req)
	if err != nil || resp.Body == nil {
		return resp, err
	}
	// remaining is max+1 so that a body of exactly max bytes still reaches
	// EOF, while the read that delivers byte max+1 - proof the peer exceeded
	// the cap - fails on the spot (see limitedBody.Read).
	resp.Body = &limitedBody{ReadCloser: resp.Body, remaining: c.max + 1, max: c.max}
	return resp, nil
}

// limitedBody is a response body that returns errMessageTooLarge once the peer
// has sent more than max bytes.
type limitedBody struct {
	io.ReadCloser
	remaining int64
	max       int64
}

// Read implements io.Reader.
func (b *limitedBody) Read(p []byte) (int, error) {
	if b.remaining <= 0 {
		return 0, fmt.Errorf("reading from MCP server: %w (cap %d bytes)", errMessageTooLarge, b.max)
	}
	if int64(len(p)) > b.remaining {
		p = p[:b.remaining]
	}
	n, err := b.ReadCloser.Read(p)
	b.remaining -= int64(n)
	if b.remaining <= 0 {
		// The peer has now delivered max+1 bytes: over the cap, with THIS
		// read, not the next one - a body of exactly max+1 bytes followed by
		// EOF would otherwise be accepted whole.
		return n, fmt.Errorf("reading from MCP server: %w (cap %d bytes)", errMessageTooLarge, b.max)
	}
	return n, err
}
