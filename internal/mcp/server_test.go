package mcp

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"net"
	"net/url"
	"os/exec"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/tools"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// discardLogger silences the SDK's server-side logger; these tests deliberately
// feed it invalid tool names ("b.c"), which it reports at error level.
func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// objectSchema is the input schema every fake tool publishes. The SDK's
// Server.AddTool panics without one.
func objectSchema() map[string]any { return map[string]any{"type": "object"} }

// fakeTool is one tool of the in-memory fake server.
type fakeTool struct {
	def     *sdk.Tool
	handler sdk.ToolHandler
}

// textTool builds a tool that answers with a fixed string.
func textTool(name, desc, reply string) fakeTool {
	return fakeTool{
		def: &sdk.Tool{Name: name, Description: desc, InputSchema: objectSchema()},
		handler: func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: reply}}}, nil
		},
	}
}

// fakeConn is the transport seam: it counts factory calls, can be made to fail,
// and can block so a test can line two callers up behind one attempt.
type fakeConn struct {
	mu      sync.Mutex
	calls   int
	failErr error
	defs    []fakeTool

	entered chan struct{} // signalled (non-blocking) on every factory call
	release chan struct{} // if non-nil, the factory waits for it

	session *sdk.ServerSession // the server side of the newest connection
	kills   atomic.Int64       // how many times a returned kill func ran
	killed  atomic.Int64       // index of the newest connection whose kill ran
}

// currentSession reports the server side of the newest connection.
func (f *fakeConn) currentSession() *sdk.ServerSession {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.session
}

// drop closes the server side of the current connection, which is how these
// tests simulate a server that died under a live session.
func (f *fakeConn) drop() {
	f.mu.Lock()
	ss := f.session
	f.mu.Unlock()
	if ss != nil {
		_ = ss.Close()
	}
}

// count reports how many times the factory ran.
func (f *fakeConn) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

// setFail makes every later factory call fail with err.
func (f *fakeConn) setFail(err error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.failErr = err
}

// factory returns a transportFactory serving f.defs over an in-memory pipe.
func (f *fakeConn) factory() transportFactory {
	return func(ctx context.Context, _ config.MCPServer, _ Deps) (sdk.Transport, func(), error) {
		f.mu.Lock()
		f.calls++
		err := f.failErr
		entered, release := f.entered, f.release
		defs := f.defs
		f.mu.Unlock()
		if entered != nil {
			select {
			case entered <- struct{}{}:
			default:
			}
		}
		if release != nil {
			select {
			case <-release:
			case <-ctx.Done():
				return nil, nil, ctx.Err()
			}
		}
		if err != nil {
			return nil, nil, err
		}
		srv := sdk.NewServer(&sdk.Implementation{Name: "fake", Version: "1.2.3"}, &sdk.ServerOptions{Logger: discardLogger()})
		for _, ft := range defs {
			srv.AddTool(ft.def, ft.handler)
		}
		clientT, serverT := sdk.NewInMemoryTransports()
		ss, err := srv.Connect(context.WithoutCancel(ctx), serverT, nil)
		if err != nil {
			return nil, nil, err
		}
		f.mu.Lock()
		f.session = ss
		idx := int64(f.calls)
		f.mu.Unlock()
		return clientT, func() {
			f.kills.Add(1)
			// Record the newest connection released, so a test can tell "the
			// old transport was torn down" from "the new one was too".
			for prev := f.killed.Load(); idx > prev && !f.killed.CompareAndSwap(prev, idx); prev = f.killed.Load() {
			}
			_ = ss.Close()
		}, nil
	}
}

// useTransport installs a factory for the duration of one test.
func useTransport(t *testing.T, f transportFactory) {
	t.Helper()
	prev := newTransport
	newTransport = f
	t.Cleanup(func() { newTransport = prev })
}

// recObserver records every observer callback.
type recObserver struct {
	mu           sync.Mutex
	connected    []ConnectInfo
	failed       []failedRec
	listed       []listedRec
	disconnected []string
}

type failedRec struct {
	server, transport string
	class             ErrorClass
	err               error
}

type listedRec struct {
	server  string
	tools   []ListedTool
	bytes   int
	skipped []SkippedTool
}

func (r *recObserver) Connected(i ConnectInfo) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.connected = append(r.connected, i)
}

func (r *recObserver) ConnectFailed(server, transport string, class ErrorClass, err error, _ time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.failed = append(r.failed, failedRec{server, transport, class, err})
}

func (r *recObserver) ToolsListed(server string, tools []ListedTool, totalBytes int, skipped []SkippedTool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.listed = append(r.listed, listedRec{server, tools, totalBytes, skipped})
}

func (r *recObserver) Disconnected(server, reason string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.disconnected = append(r.disconnected, server+":"+reason)
}

// snapshot copies the recorded state so assertions never race the observer.
func (r *recObserver) snapshot() recObserver {
	r.mu.Lock()
	defer r.mu.Unlock()
	return recObserver{
		connected:    append([]ConnectInfo(nil), r.connected...),
		failed:       append([]failedRec(nil), r.failed...),
		listed:       append([]listedRec(nil), r.listed...),
		disconnected: append([]string(nil), r.disconnected...),
	}
}

// testDeps builds the injected dependencies every test uses: a real clock, a
// zero jitter (so the tests never wait on randomness) and an empty environment.
func testDeps(obs Observer) Deps {
	return Deps{
		Clock:    time.Now,
		Rand:     func() float64 { return 0 },
		Env:      func(string) (string, bool) { return "", false },
		Observer: obs,
		Stderr:   io.Discard,
		Version:  "test",
	}
}

// testCfg builds a stdio server config; the transport seam ignores the command.
func testCfg(name string, filter config.MCPToolFilter) config.MCPServer {
	return config.MCPServer{
		Name:      name,
		Transport: config.MCPTransport{Type: config.MCPTransportStdio, Command: []string{"unused"}},
		Tools:     filter,
	}
}

// defHash recomputes the definition hash independently of the implementation.
func defHash(name, desc, schema string) string {
	sum := sha256.Sum256([]byte(name + "\n" + desc + "\n" + schema))
	return hex.EncodeToString(sum[:])
}

// connectThreeTools connects to a fake server publishing a, b.c and d, with d
// excluded by the filter.
func connectThreeTools(t *testing.T, obs Observer) *Server {
	t.Helper()
	fc := &fakeConn{defs: []fakeTool{
		textTool("a", "tool a", "A"),
		textTool("b.c", "tool b.c", "B"),
		textTool("d", "tool d", "D"),
	}}
	useTransport(t, fc.factory())
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	srv, err := Connect(ctx, testCfg("s", config.MCPToolFilter{Exclude: []string{"d"}}), testDeps(obs))
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() { _ = srv.Close(context.Background()) })
	return srv
}

func TestConnectListsAndNames(t *testing.T) {
	srv := connectThreeTools(t, nil)

	got := srv.Tools()
	if len(got) != 2 {
		t.Fatalf("Tools() = %d tools, want 2", len(got))
	}
	if n := got[0].Def().Name; n != "s__a" {
		t.Errorf("tool[0] = %q, want s__a", n)
	}
	want1 := EffectiveName("s", "b.c").Effective
	if n := got[1].Def().Name; n != want1 {
		t.Errorf("tool[1] = %q, want %q", n, want1)
	}
	if !strings.HasPrefix(want1, "s__b_c_") {
		t.Errorf("normalized name %q lost its shape", want1)
	}
	if p := string(got[0].Def().Parameters); p != `{"type":"object"}` {
		t.Errorf("parameters = %s, want the object schema", p)
	}
	skipped := srv.Skipped()
	if len(skipped) != 1 || skipped[0].Name != "d" || skipped[0].Reason != "excluded" {
		t.Errorf("Skipped() = %+v, want d/excluded", skipped)
	}
}

func TestConnectObserverEvents(t *testing.T) {
	obs := &recObserver{}
	srv := connectThreeTools(t, obs)
	_ = srv

	snap := obs.snapshot()
	if len(snap.connected) != 1 {
		t.Fatalf("Connected calls = %d, want 1", len(snap.connected))
	}
	info := snap.connected[0]
	if info.ToolCount != 2 || info.Server != "s" || info.ServerName != "fake" || info.ServerVersion != "1.2.3" {
		t.Errorf("ConnectInfo = %+v", info)
	}
	if info.ProtocolVersion == "" || info.Transport != config.MCPTransportStdio {
		t.Errorf("ConnectInfo = %+v", info)
	}
	if len(snap.listed) != 1 {
		t.Fatalf("ToolsListed calls = %d, want 1", len(snap.listed))
	}
	lr := snap.listed[0]
	if len(lr.tools) != 2 || len(lr.skipped) != 1 {
		t.Fatalf("ToolsListed = %d tools, %d skipped; want 2 and 1", len(lr.tools), len(lr.skipped))
	}
	if lr.tools[0].SHA256 != defHash("a", "tool a", `{"type":"object"}`) {
		t.Errorf("hash = %q, want the sha256 of the definition", lr.tools[0].SHA256)
	}
	if lr.tools[0].Normalized || !lr.tools[1].Normalized {
		t.Errorf("Normalized flags = %v/%v, want false/true", lr.tools[0].Normalized, lr.tools[1].Normalized)
	}
	if lr.bytes <= 0 {
		t.Errorf("totalBytes = %d, want > 0", lr.bytes)
	}
}

// iterOf turns a fixed list of results into the iterator discover consumes.
func iterOf(items []*sdk.Tool, err error) func(func(*sdk.Tool, error) bool) {
	return func(yield func(*sdk.Tool, error) bool) {
		for _, it := range items {
			if !yield(it, nil) {
				return
			}
		}
		if err != nil {
			yield(nil, err)
		}
	}
}

// newTestServer builds a Server value without connecting, for discover tests.
func newTestServer(t *testing.T, cfg config.MCPServer, deps Deps) *Server {
	t.Helper()
	d, err := deps.withDefaults()
	if err != nil {
		t.Fatalf("withDefaults: %v", err)
	}
	return &Server{cfg: cfg, deps: d}
}

func TestDiscoverPaginatesAndRejectsDuplicates(t *testing.T) {
	t.Run("every page is kept", func(t *testing.T) {
		s := newTestServer(t, testCfg("s", config.MCPToolFilter{}), testDeps(nil))
		items := []*sdk.Tool{
			{Name: "one", InputSchema: objectSchema()},
			{Name: "two", InputSchema: objectSchema()},
			{Name: "three", InputSchema: objectSchema()},
		}
		if err := s.discover(iterOf(items, nil)); err != nil {
			t.Fatalf("discover: %v", err)
		}
		if len(s.Tools()) != 3 {
			t.Fatalf("Tools() = %d, want 3", len(s.Tools()))
		}
	})

	t.Run("a repeated original name is a protocol error", func(t *testing.T) {
		s := newTestServer(t, testCfg("s", config.MCPToolFilter{}), testDeps(nil))
		items := []*sdk.Tool{
			{Name: "one", InputSchema: objectSchema()},
			{Name: "one", InputSchema: objectSchema()},
		}
		err := s.discover(iterOf(items, nil))
		if err == nil || !strings.Contains(err.Error(), "duplicate") {
			t.Fatalf("discover error = %v, want a duplicate-name error", err)
		}
		if classify(err) != ClassProtocol {
			t.Errorf("classify = %q, want protocol", classify(err))
		}
	})

	t.Run("a listing error is returned", func(t *testing.T) {
		s := newTestServer(t, testCfg("s", config.MCPToolFilter{}), testDeps(nil))
		err := s.discover(iterOf(nil, errors.New("boom")))
		if err == nil || !strings.Contains(err.Error(), "boom") {
			t.Fatalf("discover error = %v, want boom", err)
		}
	})
}

func TestDiscoverCaps(t *testing.T) {
	t.Run("too many tools", func(t *testing.T) {
		s := newTestServer(t, testCfg("s", config.MCPToolFilter{}), testDeps(nil))
		items := make([]*sdk.Tool, 0, MaxToolsPerServer+1)
		for i := range MaxToolsPerServer + 1 {
			items = append(items, &sdk.Tool{Name: fmt.Sprintf("t%d", i), InputSchema: objectSchema()})
		}
		err := s.discover(iterOf(items, nil))
		if err == nil || !strings.Contains(err.Error(), "too many tools") {
			t.Fatalf("discover error = %v, want a tool-count cap error", err)
		}
	})

	t.Run("one oversized definition is skipped", func(t *testing.T) {
		s := newTestServer(t, testCfg("s", config.MCPToolFilter{}), testDeps(nil))
		items := []*sdk.Tool{
			{Name: "big", Description: strings.Repeat("x", MaxDefinitionBytes+1), InputSchema: objectSchema()},
			{Name: "small", InputSchema: objectSchema()},
		}
		if err := s.discover(iterOf(items, nil)); err != nil {
			t.Fatalf("discover: %v", err)
		}
		if len(s.Tools()) != 1 {
			t.Errorf("Tools() = %d, want only the small tool", len(s.Tools()))
		}
		sk := s.Skipped()
		if len(sk) != 1 || sk[0].Name != "big" || sk[0].Reason != "definition too large" {
			t.Errorf("Skipped() = %+v, want big/definition too large", sk)
		}
	})

	t.Run("total discovery bytes are capped", func(t *testing.T) {
		s := newTestServer(t, testCfg("s", config.MCPToolFilter{}), testDeps(nil))
		var items []*sdk.Tool
		for i := range MaxDiscoveryBytes/MaxDefinitionBytes + 2 {
			items = append(items, &sdk.Tool{
				Name:        fmt.Sprintf("t%d", i),
				Description: strings.Repeat("x", MaxDefinitionBytes-64),
				InputSchema: objectSchema(),
			})
		}
		err := s.discover(iterOf(items, nil))
		if err == nil || !strings.Contains(err.Error(), "tool definitions exceed") {
			t.Fatalf("discover error = %v, want the discovery byte cap", err)
		}
	})

	t.Run("an uncompilable output schema skips the tool", func(t *testing.T) {
		s := newTestServer(t, testCfg("s", config.MCPToolFilter{}), testDeps(nil))
		items := []*sdk.Tool{
			{Name: "bad", InputSchema: objectSchema(), OutputSchema: map[string]any{"type": 42}},
		}
		if err := s.discover(iterOf(items, nil)); err != nil {
			t.Fatalf("discover: %v", err)
		}
		sk := s.Skipped()
		if len(sk) != 1 || sk[0].Reason != "invalid output schema" {
			t.Errorf("Skipped() = %+v, want bad/invalid output schema", sk)
		}
	})
}

// TestDiscoverPassesMaxResultBytesToTools proves the option travels: it is set
// once on Deps and every Tool built at discovery must render its results with
// it. Asserted end to end through a real call, because a field copied into the
// struct but never handed to RenderResult would still pass a field-only check.
func TestDiscoverPassesMaxResultBytesToTools(t *testing.T) {
	deps := testDeps(nil)
	deps.MaxResultBytes = 8
	fc := &fakeConn{defs: []fakeTool{textTool("long", "a chatty tool", strings.Repeat("x", 20))}}
	srv := connectFake(t, fc, testCfg("s", config.MCPToolFilter{}), deps)

	text, out, err := toolNamed(t, srv, "s__long").InvokeOutcome(context.Background(), "")
	if err != nil {
		t.Fatalf("InvokeOutcome: %v", err)
	}
	if want := "xxxxxxxx" + tools.TruncationMarker; text != want {
		t.Errorf("text = %q, want %q", text, want)
	}
	if !out.Truncated {
		t.Error("Truncated = false, want true")
	}
}

func TestDiscoverNameCollision(t *testing.T) {
	t.Run("against an existing registry name", func(t *testing.T) {
		deps := testDeps(nil)
		deps.ExistingNames = map[string]bool{"s__a": true}
		s := newTestServer(t, testCfg("s", config.MCPToolFilter{}), deps)
		err := s.discover(iterOf([]*sdk.Tool{{Name: "a", InputSchema: objectSchema()}}, nil))
		if !errors.Is(err, ErrToolset) {
			t.Fatalf("discover error = %v, want ErrToolset", err)
		}
	})

	t.Run("connect reports it without retrying", func(t *testing.T) {
		fc := &fakeConn{defs: []fakeTool{textTool("a", "tool a", "A")}}
		useTransport(t, fc.factory())
		deps := testDeps(&recObserver{})
		deps.ExistingNames = map[string]bool{"s__a": true}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		_, err := Connect(ctx, testCfg("s", config.MCPToolFilter{}), deps)
		if !errors.Is(err, ErrToolset) {
			t.Fatalf("Connect error = %v, want ErrToolset", err)
		}
		if errors.Is(err, ErrUnavailable) {
			t.Error("a static toolset error must not read as unavailable")
		}
		if fc.count() != 1 {
			t.Errorf("factory calls = %d, want 1 (retrying cannot help)", fc.count())
		}
	})
}

func TestConnectRetriesWithJitter(t *testing.T) {
	t.Run("a transient failure is retried", func(t *testing.T) {
		fc := &fakeConn{defs: []fakeTool{textTool("a", "tool a", "A")}}
		fc.setFail(errors.New("transient"))
		useTransport(t, func(ctx context.Context, cfg config.MCPServer, deps Deps) (sdk.Transport, func(), error) {
			if fc.count() == 0 {
				fc.mu.Lock()
				fc.calls++
				fc.mu.Unlock()
				return nil, nil, errors.New("transient")
			}
			fc.setFail(nil)
			return fc.factory()(ctx, cfg, deps)
		})
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		srv, err := Connect(ctx, testCfg("s", config.MCPToolFilter{}), testDeps(&recObserver{}))
		if err != nil {
			t.Fatalf("Connect: %v", err)
		}
		_ = srv.Close(ctx)
		if fc.count() != connectAttempts {
			t.Errorf("factory calls = %d, want %d", fc.count(), connectAttempts)
		}
	})

	t.Run("a permanent failure stops after connectAttempts", func(t *testing.T) {
		fc := &fakeConn{}
		fc.setFail(errors.New("nope"))
		useTransport(t, fc.factory())
		obs := &recObserver{}
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		_, err := Connect(ctx, testCfg("s", config.MCPToolFilter{}), testDeps(obs))
		if !errors.Is(err, ErrUnavailable) {
			t.Fatalf("Connect error = %v, want ErrUnavailable", err)
		}
		var ce *ConnectError
		if !errors.As(err, &ce) || ce.Server != "s" {
			t.Fatalf("Connect error = %v, want a *ConnectError for s", err)
		}
		if fc.count() != connectAttempts {
			t.Errorf("factory calls = %d, want %d", fc.count(), connectAttempts)
		}
		snap := obs.snapshot()
		if len(snap.failed) != 1 || snap.failed[0].server != "s" {
			t.Fatalf("ConnectFailed = %+v", snap.failed)
		}
	})
}

func TestConnectTimeoutClass(t *testing.T) {
	useTransport(t, func(ctx context.Context, _ config.MCPServer, _ Deps) (sdk.Transport, func(), error) {
		<-ctx.Done()
		return nil, nil, ctx.Err()
	})
	obs := &recObserver{}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := Connect(ctx, testCfg("s", config.MCPToolFilter{}), testDeps(obs))
	var ce *ConnectError
	if !errors.As(err, &ce) {
		t.Fatalf("Connect error = %v, want *ConnectError", err)
	}
	if ce.Class != ClassTimeout {
		t.Errorf("class = %q, want timeout", ce.Class)
	}
	snap := obs.snapshot()
	if len(snap.failed) != 1 || snap.failed[0].class != ClassTimeout {
		t.Errorf("ConnectFailed = %+v", snap.failed)
	}
}

func TestConnectRequiresClockAndRand(t *testing.T) {
	cases := map[string]Deps{
		"no clock": {Rand: func() float64 { return 0 }, Env: func(string) (string, bool) { return "", false }},
		"no rand":  {Clock: time.Now, Env: func(string) (string, bool) { return "", false }},
		"no env":   {Clock: time.Now, Rand: func() float64 { return 0 }},
	}
	for name, deps := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := Connect(context.Background(), testCfg("s", config.MCPToolFilter{}), deps); err == nil {
				t.Fatal("Connect succeeded with incomplete Deps")
			}
		})
	}
}

func TestErrorClassification(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want ErrorClass
	}{
		{"deadline", context.DeadlineExceeded, ClassTimeout},
		{"wrapped deadline", fmt.Errorf("connecting: %w", context.DeadlineExceeded), ClassTimeout},
		{"exec error", &exec.Error{Name: "nope", Err: exec.ErrNotFound}, ClassSpawn},
		{"missing file", fmt.Errorf("starting %q: %w", "x", fs.ErrNotExist), ClassSpawn},
		{"permission", fmt.Errorf("starting %q: %w", "x", fs.ErrPermission), ClassSpawn},
		{"unauthorized text", errors.New(`http status 401 Unauthorized`), ClassAuth},
		{"forbidden text", errors.New(`Forbidden`), ClassAuth},
		{"op error", &net.OpError{Op: "dial", Err: errors.New("connection refused")}, ClassNetwork},
		{"url error", &url.Error{Op: "Post", URL: "http://x", Err: errors.New("connection refused")}, ClassNetwork},
		{"eof", io.EOF, ClassNetwork},
		{"message too large", fmt.Errorf("reading: %w", errMessageTooLarge), ClassProtocol},
		{"json-rpc", errors.New("decoding response: invalid character"), ClassProtocol},
		{"nil", nil, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := classify(tc.err); got != tc.want {
				t.Errorf("classify(%v) = %q, want %q", tc.err, got, tc.want)
			}
		})
	}
}

func TestCloseIsIdempotentAndEmitsDisconnect(t *testing.T) {
	fc := &fakeConn{defs: []fakeTool{textTool("a", "tool a", "A")}}
	useTransport(t, fc.factory())
	obs := &recObserver{}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	srv, err := Connect(ctx, testCfg("s", config.MCPToolFilter{}), testDeps(obs))
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	if err := srv.Close(ctx); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := srv.Close(ctx); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	snap := obs.snapshot()
	if len(snap.disconnected) != 1 || snap.disconnected[0] != "s:run_end" {
		t.Errorf("Disconnected = %v, want one s:run_end", snap.disconnected)
	}
	if _, err := srv.Tools()[0].Invoke(ctx, "{}"); err == nil {
		t.Error("invoking after Close succeeded, want an error")
	}
}

func TestNopObserverAndAccessors(t *testing.T) {
	var o Observer = NopObserver{}
	o.Connected(ConnectInfo{})
	o.ConnectFailed("s", "stdio", ClassNetwork, errors.New("x"), time.Second)
	o.ToolsListed("s", nil, 0, nil)
	o.Disconnected("s", "run_end")

	fc := &fakeConn{defs: []fakeTool{textTool("a", "tool a", "A")}}
	useTransport(t, fc.factory())
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	deps := testDeps(nil) // nil observer must be accepted
	srv, err := Connect(ctx, testCfg("s", config.MCPToolFilter{}), deps)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = srv.Close(ctx) }()
	if srv.Name() != "s" {
		t.Errorf("Name() = %q", srv.Name())
	}
	if srv.Errors() != 0 {
		t.Errorf("Errors() = %d, want 0", srv.Errors())
	}
	if len(srv.Listed()) != 1 || srv.Listed()[0].Original != "a" {
		t.Errorf("Listed() = %+v", srv.Listed())
	}
	if srv.Info().SessionFP == "" && srv.Info().ServerName != "fake" {
		t.Errorf("Info() = %+v", srv.Info())
	}
	// Tools() hands out a copy: mutating it must not disturb the server.
	got := srv.Tools()
	got[0] = nil
	if srv.Tools()[0] == nil {
		t.Error("Tools() exposed its backing array")
	}
}

// ConnectError must read as both its cause and the shared sentinel so cmd can
// map it to exit 8 while a log line still shows the real failure.
func TestConnectErrorUnwrap(t *testing.T) {
	cause := errors.New("dial tcp: refused")
	err := error(&ConnectError{Server: "s", Class: ClassNetwork, Err: cause})
	if !errors.Is(err, cause) || !errors.Is(err, ErrUnavailable) {
		t.Fatalf("ConnectError does not unwrap to both: %v", err)
	}
	if !strings.Contains(err.Error(), "s") || !strings.Contains(err.Error(), "refused") {
		t.Errorf("Error() = %q", err.Error())
	}
}

// flakyLister serves the tools list one page at a time and fails the nth
// tools/list request, which is how a server that dies while the client is
// paginating looks from amele's side.
func flakyLister(failOn int) sdk.Middleware {
	var seen atomic.Int64
	return func(next sdk.MethodHandler) sdk.MethodHandler {
		return func(ctx context.Context, method string, req sdk.Request) (sdk.Result, error) {
			if method == "tools/list" && seen.Add(1) == int64(failOn) {
				return nil, errors.New("listing died mid-page")
			}
			return next(ctx, method, req)
		}
	}
}

// pagedFactory serves defs one tool per page; the first connection fails its
// second tools/list, every later one is healthy.
func pagedFactory(t *testing.T, defs []fakeTool, calls *atomic.Int64) transportFactory {
	t.Helper()
	return func(ctx context.Context, _ config.MCPServer, _ Deps) (sdk.Transport, func(), error) {
		first := calls.Add(1) == 1
		srv := sdk.NewServer(&sdk.Implementation{Name: "fake", Version: "1.2.3"},
			&sdk.ServerOptions{Logger: discardLogger(), PageSize: 1})
		for _, ft := range defs {
			srv.AddTool(ft.def, ft.handler)
		}
		if first {
			srv.AddReceivingMiddleware(flakyLister(2))
		}
		clientT, serverT := sdk.NewInMemoryTransports()
		ss, err := srv.Connect(context.WithoutCancel(ctx), serverT, nil)
		if err != nil {
			return nil, nil, err
		}
		return clientT, func() { _ = ss.Close() }, nil
	}
}

// A retry must not append a second copy of a partially discovered toolset:
// discovery starts from empty on every attempt.
func TestConnectRetryDoesNotDuplicateTools(t *testing.T) {
	var calls atomic.Int64
	useTransport(t, pagedFactory(t, []fakeTool{
		textTool("a", "tool a", "A"),
		textTool("b", "tool b", "B"),
		textTool("c", "tool c", "C"),
	}, &calls))
	obs := &recObserver{}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	srv, err := Connect(ctx, testCfg("s", config.MCPToolFilter{}), testDeps(obs))
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = srv.Close(ctx) }()
	if calls.Load() != 2 {
		t.Fatalf("factory calls = %d, want 2 (the first listing died mid-page)", calls.Load())
	}

	var names []string
	for _, tl := range srv.Tools() {
		names = append(names, tl.Def().Name)
	}
	want := []string{"s__a", "s__b", "s__c"}
	if !slices.Equal(names, want) {
		t.Errorf("Tools() = %v, want exactly %v", names, want)
	}
	if len(srv.Listed()) != len(want) {
		t.Errorf("Listed() = %d entries, want %d", len(srv.Listed()), len(want))
	}

	snap := obs.snapshot()
	if len(snap.listed) != 1 {
		t.Fatalf("ToolsListed calls = %d, want 1", len(snap.listed))
	}
	if len(snap.listed[0].tools) != len(want) {
		t.Errorf("ToolsListed carried %d tools, want %d", len(snap.listed[0].tools), len(want))
	}
	if snap.connected[0].ToolCount != len(want) {
		t.Errorf("ConnectInfo.ToolCount = %d, want %d", snap.connected[0].ToolCount, len(want))
	}
}

// A reconnect that finishes after Close must release what it built instead of
// installing a session nobody will ever close (and, for stdio, a child process
// that would outlive the run).
func TestCloseDuringReconnectReleasesNewSession(t *testing.T) {
	fc := &fakeConn{defs: callToolset()}
	srv := connectFake(t, fc, testCfg("s", config.MCPToolFilter{}), testDeps(&recObserver{}))
	ctx := context.Background()

	loseSession(t, srv, fc)

	entered := make(chan struct{}, 1)
	release := make(chan struct{})
	fc.mu.Lock()
	fc.entered, fc.release = entered, release
	killsBefore := fc.kills.Load()
	fc.mu.Unlock()

	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _, _ = toolNamed(t, srv, "s__echo").InvokeOutcome(ctx, `{"text":"x"}`)
	}()

	<-entered // the reconnect is inside the factory
	if err := srv.Close(ctx); err != nil {
		t.Fatalf("Close: %v", err)
	}
	close(release)
	<-done

	// Two releases: the dead transport at the start of the reconnect, and the
	// fresh one that Close refused to adopt.
	if got := fc.kills.Load() - killsBefore; got != 2 {
		t.Errorf("transports released = %d, want 2", got)
	}
	if got, want := fc.killed.Load(), int64(fc.count()); got != want {
		t.Errorf("newest released connection = %d, want %d (the session built after Close must not leak)", got, want)
	}
	if fc.currentSession() == nil {
		t.Error("the factory never built the racing session")
	}
	if _, err := srv.session(ctx); !errors.Is(err, errClosed) {
		t.Errorf("session() = %v, want the closed error", err)
	}
}
