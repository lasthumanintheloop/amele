package mcp

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/tools"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// discardWriteCloser swallows writes; the far end of a server that never was.
type discardWriteCloser struct{}

func (discardWriteCloser) Write(p []byte) (int, error) { return len(p), nil }
func (discardWriteCloser) Close() error                { return nil }

// useConnectWindow shrinks the shared connect window for one test.
func useConnectWindow(t *testing.T, d time.Duration) {
	t.Helper()
	prev := connectWindow
	connectWindow = d
	t.Cleanup(func() { connectWindow = prev })
}

// TestClassifyAuthBounded pins the two halves of the auth-classification fix:
// bare status digits only count when they stand alone, and a typed network
// error wins over auth-looking text riding inside it.
func TestClassifyAuthBounded(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want ErrorClass
	}{
		{"status with name", errors.New("initialize: 401 Unauthorized"), ClassAuth},
		{"delimited bare status", errors.New("http status 403"), ClassAuth},
		{"digits inside a port", errors.New("dial tcp 127.0.0.1:4013: no route"), ClassProtocol},
		{"digits inside a count", errors.New("read 4030 bytes then EOF mid-frame"), ClassProtocol},
		{"typed network wins over auth text", fmt.Errorf("server said 401: %w", syscall.ECONNRESET), ClassNetwork},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := classify(tc.err); got != tc.want {
				t.Errorf("classify(%v) = %v, want %v", tc.err, got, tc.want)
			}
		})
	}
}

// TestConnectSharedWindow: both attempts (and the pause between them) share
// ONE ConnectTimeout window, so a hanging server costs a run roughly the
// window - not attempts x window.
func TestConnectSharedWindow(t *testing.T) {
	const window = 300 * time.Millisecond
	useConnectWindow(t, window)
	// A transport nobody serves: the initialize request vanishes into Discard
	// and the response never comes, exactly like a remote that accepts the
	// connection and never answers. The pipe is closed by the kill func so no
	// goroutine outlives the test.
	useTransport(t, func(context.Context, config.MCPServer, Deps) (sdk.Transport, func(), error) {
		pr, pw := io.Pipe()
		return &sdk.IOTransport{Reader: pr, Writer: discardWriteCloser{}},
			func() { _ = pr.Close(); _ = pw.Close() }, nil
	})

	start := time.Now()
	_, err := Connect(context.Background(), testCfg("s", config.MCPToolFilter{}), testDeps(nil))
	elapsed := time.Since(start)
	if err == nil {
		t.Fatal("Connect succeeded against a hanging factory")
	}
	var ce *ConnectError
	if !errors.As(err, &ce) || ce.Class != ClassTimeout {
		t.Errorf("err = %v, want ConnectError with class timeout", err)
	}
	// Generous bound: anything close to 2x the window means each attempt got
	// its own timer again.
	if elapsed > window+window/2 {
		t.Errorf("Connect took %v, want about one %v window", elapsed, window)
	}
}

// TestReconnectAbortedNotCounted: a reconnect cut short because the RUN ended
// is the run stopping, not the server failing - it must not inflate Errors(),
// and the call classifies as aborted instead of surfacing a dispatch error.
func TestReconnectAbortedNotCounted(t *testing.T) {
	fc := &fakeConn{defs: callToolset()}
	srv := connectFake(t, fc, testCfg("s", config.MCPToolFilter{}), testDeps(nil))

	// Lose the session: the next call reports the loss and counts ONE error.
	fc.drop()
	text, out, err := toolNamed(t, srv, "s__echo").InvokeOutcome(context.Background(), `{"text":"x"}`)
	if err != nil || out.Kind != tools.OutcomeIndeterminate {
		t.Fatalf("lost call = (%q, %v, %v), want indeterminate", text, out.Kind, err)
	}
	if got := srv.Errors(); got != 1 {
		t.Fatalf("Errors() = %d after the loss, want 1", got)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // the run is already over when the call arrives
	text, out, err = toolNamed(t, srv, "s__echo").InvokeOutcome(ctx, `{"text":"x"}`)
	if err != nil || out.Kind != tools.OutcomeAborted || text != abortedText {
		t.Errorf("cancelled reconnect = (%q, %v, %v), want (%q, aborted, nil)", text, out.Kind, err, abortedText)
	}
	if got := srv.Errors(); got != 1 {
		t.Errorf("Errors() = %d after a cancelled reconnect, want still 1", got)
	}
}

// TestReconnectTimeoutNotCounted: a reconnect that ran out of the CALL's own
// budget classifies as the timeout it is, and is not counted as a server
// failure either.
func TestReconnectTimeoutNotCounted(t *testing.T) {
	deps := testDeps(nil)
	// Near-full jitter: the reconnect pause (~900ms) outlives the 50ms call
	// budget below, so the call deterministically times out mid-reconnect.
	deps.Rand = func() float64 { return 0.9 }
	cfg := testCfg("s", config.MCPToolFilter{})
	cfg.CallTimeout = config.Duration(50 * time.Millisecond)
	fc := &fakeConn{defs: callToolset()}
	srv := connectFake(t, fc, cfg, deps)

	fc.drop()
	if _, out, _ := toolNamed(t, srv, "s__echo").InvokeOutcome(context.Background(), `{"text":"x"}`); out.Kind != tools.OutcomeIndeterminate {
		t.Fatalf("outcome = %v, want indeterminate", out.Kind)
	}
	text, out, err := toolNamed(t, srv, "s__echo").InvokeOutcome(context.Background(), `{"text":"x"}`)
	if err != nil || out.Kind != tools.OutcomeTimedOut || !strings.Contains(text, "timed out") {
		t.Errorf("timed-out reconnect = (%q, %v, %v), want a timeout outcome", text, out.Kind, err)
	}
	if got := srv.Errors(); got != 1 {
		t.Errorf("Errors() = %d after a timed-out reconnect, want still 1", got)
	}
}

// TestFailedReconnectEmitsConnectFailed: a reconnect that genuinely failed
// must land in the session log as mcp_connect{ok:false}, or the operator sees
// a disconnect followed by nothing while every later call errors.
func TestFailedReconnectEmitsConnectFailed(t *testing.T) {
	obs := &recObserver{}
	fc := &fakeConn{defs: callToolset()}
	srv := connectFake(t, fc, testCfg("s", config.MCPToolFilter{}), testDeps(obs))

	fc.drop()
	if _, out, _ := toolNamed(t, srv, "s__echo").InvokeOutcome(context.Background(), `{"text":"x"}`); out.Kind != tools.OutcomeIndeterminate {
		t.Fatalf("outcome = %v, want indeterminate", out.Kind)
	}
	fc.setFail(errors.New("still down"))
	if _, _, err := toolNamed(t, srv, "s__echo").InvokeOutcome(context.Background(), `{"text":"x"}`); err == nil {
		t.Fatal("reconnect against a failing factory succeeded")
	}
	snap := obs.snapshot()
	if len(snap.failed) != 1 || snap.failed[0].server != "s" {
		t.Errorf("ConnectFailed events = %+v, want exactly one for server s", snap.failed)
	}
	if got := srv.Errors(); got != 2 {
		t.Errorf("Errors() = %d, want 2 (one lost response, one failed reconnect)", got)
	}
}

// discoverSeq adapts a slice of tools to the iterator discover consumes.
func discoverSeq(defs []*sdk.Tool) iter.Seq2[*sdk.Tool, error] {
	return func(yield func(*sdk.Tool, error) bool) {
		for _, d := range defs {
			if !yield(d, nil) {
				return
			}
		}
	}
}

// TestDiscoveryChargesOutputSchema: outputSchema bytes count toward the
// definition caps like every other part of the definition.
func TestDiscoveryChargesOutputSchema(t *testing.T) {
	big := map[string]any{
		"type":        "object",
		"description": strings.Repeat("x", MaxDefinitionBytes),
	}
	s := &Server{cfg: testCfg("s", config.MCPToolFilter{})}
	err := s.discover(discoverSeq([]*sdk.Tool{
		{Name: "bloated", InputSchema: map[string]any{"type": "object"}, OutputSchema: big},
		{Name: "fine", InputSchema: map[string]any{"type": "object"}},
	}))
	if err != nil {
		t.Fatalf("discover: %v", err)
	}
	if len(s.tools) != 1 || s.listed[0].Original != "fine" {
		t.Fatalf("kept %d tools, want only 'fine'", len(s.tools))
	}
	if len(s.skipped) != 1 || s.skipped[0].Reason != "definition too large" {
		t.Errorf("skipped = %+v, want bloated/definition too large", s.skipped)
	}
	if s.totalBytes <= MaxDefinitionBytes {
		t.Errorf("totalBytes = %d, want the output schema charged (> %d)", s.totalBytes, MaxDefinitionBytes)
	}
}

// TestDiscoverySkipsNonObjectInputSchema: a tool whose parameters are not an
// object schema would be rejected downstream by every provider; it is skipped
// at discovery with a reason instead.
func TestDiscoverySkipsNonObjectInputSchema(t *testing.T) {
	s := &Server{cfg: testCfg("s", config.MCPToolFilter{})}
	err := s.discover(discoverSeq([]*sdk.Tool{
		{Name: "arrayed", InputSchema: map[string]any{"type": "array"}},
		{Name: "boolean", InputSchema: true},
		{Name: "fine", InputSchema: map[string]any{"type": "object"}},
	}))
	if err != nil {
		t.Fatalf("discover: %v", err)
	}
	if len(s.tools) != 1 || s.listed[0].Original != "fine" {
		t.Fatalf("kept %d tools, want only 'fine'", len(s.tools))
	}
	for _, sk := range s.skipped {
		if sk.Reason != "input schema not an object" {
			t.Errorf("skip %q reason = %q, want input schema not an object", sk.Name, sk.Reason)
		}
	}
	if len(s.skipped) != 2 {
		t.Errorf("skipped %d tools, want 2", len(s.skipped))
	}
}

// TestIsObjectSchema pins the root-shape rules.
func TestIsObjectSchema(t *testing.T) {
	cases := []struct {
		in   string
		want bool
	}{
		{`{"type":"object"}`, true},
		{`{"properties":{}}`, true}, // no type: conventionally an object
		{`{"type":"array"}`, false},
		{`true`, false},
		{`[]`, false},
	}
	for _, tc := range cases {
		if got := isObjectSchema([]byte(tc.in)); got != tc.want {
			t.Errorf("isObjectSchema(%s) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

// TestLimitedBodyOffByOne: a body of exactly max bytes reads to EOF; a body of
// max+1 bytes fails on the read that delivers the extra byte, not one read
// later (where an EOF would have masked it).
func TestLimitedBodyOffByOne(t *testing.T) {
	const max = 16
	read := func(size int) error {
		b := &limitedBody{
			ReadCloser: io.NopCloser(bytes.NewReader(bytes.Repeat([]byte{'a'}, size))),
			remaining:  max + 1, max: max,
		}
		_, err := io.ReadAll(b)
		return err
	}
	if err := read(max); err != nil {
		t.Errorf("body of exactly max bytes: %v, want nil", err)
	}
	if err := read(max + 1); !IsMessageTooLarge(err) {
		t.Errorf("body of max+1 bytes: %v, want the size-cap error", err)
	}
}

// TestDeadlineRoundTripper: a request without a context deadline gets one; a
// request that brought its own passes through untouched.
func TestDeadlineRoundTripper(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/slow" {
			time.Sleep(300 * time.Millisecond)
		}
		_, _ = w.Write([]byte("ok"))
	}))
	defer ts.Close()
	client := &http.Client{Transport: &deadlineRoundTripper{next: http.DefaultTransport, timeout: 50 * time.Millisecond}}

	req, _ := http.NewRequest(http.MethodDelete, ts.URL+"/slow", nil) //nolint:noctx // the missing context IS the case under test
	if resp, err := client.Do(req); err == nil {
		_ = resp.Body.Close()
		t.Error("deadline-less request against a stalled server succeeded, want a timeout")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	req, _ = http.NewRequestWithContext(ctx, http.MethodGet, ts.URL+"/", nil)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("request with its own deadline: %v", err)
	}
	body, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if string(body) != "ok" {
		t.Errorf("body = %q, want ok", body)
	}
}

// syncBuffer is a concurrency-safe bytes.Buffer: os/exec writes the child's
// stderr from its own goroutine.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) Len() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Len()
}

// TestStdioStderrUncapped: the transport relays the child's stderr unwrapped -
// bounding (and announcing the drop) is the caller's relay's job, and the old
// silent 16 KiB inner cap turned that announcement into a lie.
func TestStdioStderrUncapped(t *testing.T) {
	bin := buildTestServer(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	const banner = 20 << 10 // comfortably past the removed 16 KiB cap
	var buf syncBuffer
	sess := connectStdio(t, ctx, []string{bin, "-stderr-bytes", "20480"}, nil, func(string) (string, bool) { return "", false }, &buf)
	_ = sess.Close()

	// The child wrote the banner before serving, so by the time initialize
	// completed it has been relayed in full.
	if got := buf.Len(); got < banner {
		t.Errorf("captured %d stderr bytes, want at least %d (inner cap resurrected?)", got, banner)
	}
}

// TestClosedPipeIsConnectionLoss: the two errors a transport torn down under a
// call can produce mean the same thing. Which one surfaces is a race, and the
// drop-based tests above flaked on it (a tool_error instead of the
// indeterminate outcome the contract promises for a possibly-delivered
// request).
func TestClosedPipeIsConnectionLoss(t *testing.T) {
	for _, err := range []error{io.ErrClosedPipe, fs.ErrClosed} {
		if !isConnectionLoss(fmt.Errorf(`calling "tools/call": %w`, err)) {
			t.Errorf("isConnectionLoss(%v) = false, want true", err)
		}
	}
}
