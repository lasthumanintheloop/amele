package mcp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync"
	"time"
)

// Timings and shape of the loopback redirect endpoint. They are constants
// rather than options for the same reason the connection budgets are: an
// operator who could stretch them would only be extending how long a browser
// window can hold a terminal hostage.
const (
	// callbackPath is the ONLY path the listener answers. A redirect URI with
	// a path is what RFC 8252 §7.3 expects, and it keeps a stray request to
	// "/" (a bookmarked port, a favicon probe) from even reaching the
	// parameter checks.
	callbackPath = "/callback"
	// loginCallbackTimeout caps how long Login waits for the browser to come
	// back. Five minutes is long enough for a consent screen with a password
	// manager and an MFA prompt, and short enough that a forgotten tab does
	// not pin a terminal - or an open port - for an afternoon.
	loginCallbackTimeout = 5 * time.Minute
	// loopbackReadHeaderTimeout bounds a client that opens the port and then
	// dribbles a request line. The browser is on this machine; anything
	// slower than this is not a browser.
	loopbackReadHeaderTimeout = 10 * time.Second
	// loopbackShutdownTimeout bounds the wait for in-flight callback handlers
	// when Login is done. It only has to cover writing a one-line page.
	loopbackShutdownTimeout = 5 * time.Second
)

// errCallbackTimeout reports that the browser never came back. It is a
// sentinel so a caller (today: Login's error text) can tell "the human walked
// away" apart from "the authorization server said no".
var errCallbackTimeout = errors.New("timed out waiting for the authorization callback")

// callbackResult is one delivered authorization response.
//
// err carries an authorization server REFUSAL (an `error` parameter on the
// redirect); the transport-level failures are returned by wait itself.
type callbackResult struct {
	code, state, iss string
	err              error
}

// loopbackListener is the redirect endpoint of the authorization code flow: a
// short-lived HTTP server on a kernel-chosen 127.0.0.1 port.
//
// CONTRACT: "single shot" means ONE VALID callback, not one request. A request
// that is not structurally a callback - wrong method, wrong path, missing
// code/state - is answered and IGNORED, and the listener keeps listening. A
// favicon probe, a browser prefetch or a hostile GET from another process on
// the box must not be able to make a login fail; the state comparison the SDK
// performs afterwards is what makes a forged callback useless.
type loopbackListener struct {
	ln  net.Listener
	srv *http.Server
	// result is buffered so a handler never blocks on a caller that has
	// already given up.
	result chan callbackResult
	// once makes delivery single-shot: the first valid callback wins and
	// every later one is answered but dropped.
	once sync.Once
	// serveDone is closed when Serve has returned, so close can promise that
	// no goroutine of this listener outlives it.
	serveDone chan struct{}
}

// newLoopbackListener binds 127.0.0.1 on a kernel-chosen port and starts
// serving. The caller MUST call close.
//
// SECURITY: the bind is to the loopback interface explicitly, never to
// 0.0.0.0: the authorization code is a bearer credential for the length of one
// exchange, and a listener on a routable interface would offer it to the local
// network.
func newLoopbackListener() (*loopbackListener, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, fmt.Errorf("opening the loopback callback listener: %w", err)
	}
	l := &loopbackListener{
		ln:        ln,
		result:    make(chan callbackResult, 1),
		serveDone: make(chan struct{}),
	}
	mux := http.NewServeMux()
	mux.HandleFunc(callbackPath, l.serve)
	l.srv = &http.Server{Handler: mux, ReadHeaderTimeout: loopbackReadHeaderTimeout}
	go func() {
		defer close(l.serveDone)
		_ = l.srv.Serve(ln)
	}()
	return l, nil
}

// addr is the listener's host:port.
func (l *loopbackListener) addr() string { return l.ln.Addr().String() }

// URL is the redirect URI to register with the authorization server.
func (l *loopbackListener) URL() string { return "http://" + l.addr() + callbackPath }

// serve answers one request on the callback path.
func (l *loopbackListener) serve(w http.ResponseWriter, r *http.Request) {
	// SECURITY: no-referrer keeps the authorization code out of the Referer
	// header of anything the success page could ever link to, and no-store
	// keeps the full callback URL out of a shared browser cache.
	w.Header().Set("Referrer-Policy", "no-referrer")
	w.Header().Set("Cache-Control", "no-store")
	if r.Method != http.MethodGet {
		http.Error(w, "expected GET", http.StatusMethodNotAllowed)
		return
	}
	q := r.URL.Query()
	code, state, authErr := q.Get("code"), q.Get("state"), q.Get("error")
	switch {
	case state != "" && authErr != "":
		// A refusal is a structurally valid callback: it carries the state the
		// authorization request went out with. The error CODE is deliberately
		// the only thing kept - error_description is server-controlled text
		// that ends up on an operator's terminal.
		l.deliver(callbackResult{err: fmt.Errorf("the authorization server refused the request (%s)", safeParam(authErr))})
		writeCallbackPage(w, "amele: authorization was refused. You can close this tab.")
	case code != "" && state != "":
		l.deliver(callbackResult{code: code, state: state, iss: q.Get("iss")})
		writeCallbackPage(w, "amele: authorization received. You can close this tab.")
	default:
		// Not a callback at all. Answered, logged nowhere, and the listener
		// stays up.
		http.Error(w, "not an oauth callback", http.StatusBadRequest)
	}
}

// writeCallbackPage sends the plain-text page the human sees. It is plain text
// on purpose: no markup, no scripts and nothing echoed from the request, so
// there is no surface for the query string to become content.
func writeCallbackPage(w http.ResponseWriter, msg string) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, msg+"\n")
}

// safeParam renders an authorization server-supplied parameter for a terminal.
//
// SECURITY: the value is untrusted input from a redirect (threat model S2/S9).
// Everything outside printable ASCII is dropped so an escape sequence cannot
// repaint the operator's screen, and the result is clipped.
func safeParam(s string) string {
	const maxParam = 64
	out := make([]rune, 0, maxParam)
	for _, r := range s {
		if r < 0x20 || r > 0x7e {
			continue
		}
		if len(out) == maxParam {
			break
		}
		out = append(out, r)
	}
	if len(out) == 0 {
		return "unspecified"
	}
	return string(out)
}

// deliver hands the first valid callback to whoever is waiting.
func (l *loopbackListener) deliver(res callbackResult) {
	l.once.Do(func() { l.result <- res })
}

// wait blocks until a valid callback arrives, ctx is done, or the five-minute
// cap elapses. A refusal from the authorization server is returned as an
// error, not as a result.
func (l *loopbackListener) wait(ctx context.Context) (callbackResult, error) {
	timer := time.NewTimer(loginCallbackTimeout)
	defer timer.Stop()
	select {
	case res := <-l.result:
		if res.err != nil {
			return callbackResult{}, res.err
		}
		return res, nil
	case <-ctx.Done():
		return callbackResult{}, ctx.Err()
	case <-timer.C:
		return callbackResult{}, errCallbackTimeout
	}
}

// close stops the listener and waits for its goroutines.
//
// The graceful shutdown is not politeness: the browser is mid-request when the
// code is delivered, and dropping the connection would leave the human staring
// at a connection-reset page after a login that actually succeeded.
func (l *loopbackListener) close() {
	ctx, cancel := context.WithTimeout(context.Background(), loopbackShutdownTimeout)
	defer cancel()
	if err := l.srv.Shutdown(ctx); err != nil {
		_ = l.srv.Close()
	}
	<-l.serveDone
}
