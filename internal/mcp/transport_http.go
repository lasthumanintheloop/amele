package mcp

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// maxRedirects is how many hops a request may take before amele gives up. Any
// redirect chain longer than this is a misconfiguration, not a feature.
const maxRedirects = 3

// newHTTPTransport builds a Streamable HTTP transport for one remote MCP
// server: the operator's headers on every request, no standalone SSE stream,
// no SDK-level reconnects, same-origin redirects only and a response body cap.
//
// It returns an error only when rawURL cannot be parsed or is not http(s); no
// network I/O happens here.
func newHTTPTransport(rawURL string, headers map[string]string) (*sdk.StreamableClientTransport, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("parsing url %q: %w", rawURL, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" || u.Host == "" {
		return nil, fmt.Errorf("url %q: want an absolute http(s) url", rawURL)
	}

	base := http.DefaultTransport.(*http.Transport).Clone()
	client := &http.Client{
		// No client-level timeout: an MCP call's deadline is the run's ctx,
		// and a streaming response may legitimately outlive any fixed bound.
		Timeout: 0,
		Transport: &headerRoundTripper{
			next:    &cappedBodyRoundTripper{next: base, max: MaxMessageBytes},
			headers: headers,
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
	}, nil
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
	// remaining is max+1 so that a body of exactly max bytes still reaches EOF.
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
	return n, err
}
