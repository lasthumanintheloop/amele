package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/tools"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Tool must satisfy both the registry interface and the loop's richer optional
// one; a compile-time assertion is cheaper than discovering it at run time.
var (
	_ tools.Tool = (*Tool)(nil)
	_ interface {
		InvokeOutcome(ctx context.Context, rawArgs string) (string, tools.Outcome, error)
	} = (*Tool)(nil)
)

// errNoServer is the failure the reconnect tests inject into the factory.
var errNoServer = errors.New("no server")

// rawArgs decodes a handler's arguments into m, ignoring absent arguments.
func rawArgs(req *sdk.CallToolRequest, m any) {
	if len(req.Params.Arguments) == 0 {
		return
	}
	_ = json.Unmarshal(req.Params.Arguments, m)
}

// callToolset is the fake server used by the invoke tests.
func callToolset() []fakeTool {
	return []fakeTool{
		{
			def: &sdk.Tool{Name: "echo", Description: "echo", InputSchema: objectSchema()},
			handler: func(_ context.Context, req *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
				var in struct {
					Text string `json:"text"`
				}
				rawArgs(req, &in)
				return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: in.Text}}}, nil
			},
		},
		{
			def: &sdk.Tool{Name: "fail", Description: "fail", InputSchema: objectSchema()},
			handler: func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
				return &sdk.CallToolResult{IsError: true, Content: []sdk.Content{&sdk.TextContent{Text: "boom"}}}, nil
			},
		},
		{
			def: &sdk.Tool{Name: "sleep", Description: "sleep", InputSchema: objectSchema()},
			handler: func(ctx context.Context, req *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
				var in struct {
					Ms int `json:"ms"`
				}
				rawArgs(req, &in)
				select {
				case <-time.After(time.Duration(in.Ms) * time.Millisecond):
				case <-ctx.Done():
				}
				return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: "slept"}}}, nil
			},
		},
	}
}

// connectFake connects to an in-memory fake server built from defs.
func connectFake(t *testing.T, fc *fakeConn, cfg config.MCPServer, deps Deps) *Server {
	t.Helper()
	useTransport(t, fc.factory())
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	srv, err := Connect(ctx, cfg, deps)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() { _ = srv.Close(context.Background()) })
	return srv
}

// toolNamed finds one tool by its model-facing name.
func toolNamed(t *testing.T, srv *Server, name string) *Tool {
	t.Helper()
	for _, tl := range srv.Tools() {
		if tl.Def().Name == name {
			mt, ok := tl.(*Tool)
			if !ok {
				t.Fatalf("tool %q is %T, want *Tool", name, tl)
			}
			return mt
		}
	}
	t.Fatalf("tool %q not found", name)
	return nil
}

func TestInvokeText(t *testing.T) {
	srv := connectFake(t, &fakeConn{defs: callToolset()}, testCfg("s", config.MCPToolFilter{}), testDeps(nil))
	ctx := context.Background()

	text, out, err := toolNamed(t, srv, "s__echo").InvokeOutcome(ctx, `{"text":"hi"}`)
	if err != nil {
		t.Fatalf("InvokeOutcome: %v", err)
	}
	if text != "hi" || out.Kind != tools.OutcomeOK {
		t.Errorf("got (%q, %v), want (hi, ok)", text, out.Kind)
	}
	// Invoke is the same call through the plain registry interface.
	plain, err := toolNamed(t, srv, "s__echo").Invoke(ctx, `{"text":"again"}`)
	if err != nil || plain != "again" {
		t.Errorf("Invoke = (%q, %v)", plain, err)
	}
}

func TestInvokeIsError(t *testing.T) {
	srv := connectFake(t, &fakeConn{defs: callToolset()}, testCfg("s", config.MCPToolFilter{}), testDeps(nil))

	text, out, err := toolNamed(t, srv, "s__fail").InvokeOutcome(context.Background(), "")
	if err != nil {
		t.Fatalf("InvokeOutcome returned a Go error: %v", err)
	}
	if text != "error: boom" || out.Kind != tools.OutcomeToolError {
		t.Errorf("got (%q, %v), want (error: boom, tool error)", text, out.Kind)
	}
}

func TestInvokeBadArgsJSON(t *testing.T) {
	srv := connectFake(t, &fakeConn{defs: callToolset()}, testCfg("s", config.MCPToolFilter{}), testDeps(nil))

	text, out, err := toolNamed(t, srv, "s__echo").InvokeOutcome(context.Background(), "{")
	if err != nil {
		t.Fatalf("InvokeOutcome returned a Go error: %v", err)
	}
	if !strings.HasPrefix(text, "error: invalid JSON arguments") || out.Kind != tools.OutcomeToolError {
		t.Errorf("got (%q, %v), want an invalid-JSON tool error", text, out.Kind)
	}
}

func TestInvokeUnknownToolIsToolError(t *testing.T) {
	srv := connectFake(t, &fakeConn{defs: callToolset()}, testCfg("s", config.MCPToolFilter{}), testDeps(nil))
	tl := toolNamed(t, srv, "s__echo")
	tl.original = "no-such-tool" // the server dropped the tool after discovery

	text, out, err := tl.InvokeOutcome(context.Background(), "{}")
	if err != nil {
		t.Fatalf("InvokeOutcome returned a Go error: %v", err)
	}
	if !strings.HasPrefix(text, "error: ") || out.Kind != tools.OutcomeToolError {
		t.Errorf("got (%q, %v), want a tool error", text, out.Kind)
	}
	if srv.Errors() != 0 {
		t.Errorf("Errors() = %d, want 0 (a JSON-RPC error is not a transport failure)", srv.Errors())
	}
}

func TestInvokeCallTimeout(t *testing.T) {
	cfg := testCfg("s", config.MCPToolFilter{})
	cfg.CallTimeout = config.Duration(50 * time.Millisecond)
	srv := connectFake(t, &fakeConn{defs: callToolset()}, cfg, testDeps(nil))

	text, out, err := toolNamed(t, srv, "s__sleep").InvokeOutcome(context.Background(), `{"ms":5000}`)
	if err != nil {
		t.Fatalf("InvokeOutcome returned a Go error: %v", err)
	}
	if text != "error: tool call timed out after 50ms" || out.Kind != tools.OutcomeTimedOut {
		t.Errorf("got (%q, %v), want the timeout wording", text, out.Kind)
	}
}

func TestInvokeAbortedRun(t *testing.T) {
	srv := connectFake(t, &fakeConn{defs: callToolset()}, testCfg("s", config.MCPToolFilter{}), testDeps(nil))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	text, out, err := toolNamed(t, srv, "s__echo").InvokeOutcome(ctx, `{"text":"hi"}`)
	if err != nil {
		t.Fatalf("InvokeOutcome returned a Go error: %v", err)
	}
	if text != "error: run aborted" || out.Kind != tools.OutcomeAborted {
		t.Errorf("got (%q, %v), want the aborted wording", text, out.Kind)
	}
}

// loseSession drops the server side of the connection and calls until amele
// notices. The first call after a drop may still succeed - a closed pipe is
// only discovered on a write - so the loop, not a single call, is what makes
// "the session died" deterministic.
func loseSession(t *testing.T, srv *Server, fc *fakeConn) (string, tools.Outcome) {
	t.Helper()
	fc.drop()
	for range 5 {
		text, out, err := toolNamed(t, srv, "s__echo").InvokeOutcome(context.Background(), `{"text":"probe"}`)
		if err != nil {
			t.Fatalf("probe call: %v", err)
		}
		if out.Kind == tools.OutcomeIndeterminate {
			return text, out
		}
	}
	t.Fatal("the client never noticed the dropped connection")
	return "", tools.Outcome{}
}

func TestInvokeLostResponseIsIndeterminate(t *testing.T) {
	fc := &fakeConn{defs: callToolset()}
	srv := connectFake(t, fc, testCfg("s", config.MCPToolFilter{}), testDeps(nil))

	// The server dies between calls: the request leaves amele and no answer
	// ever comes back, which is exactly the indeterminate case.
	text, _ := loseSession(t, srv, fc)
	if !strings.HasPrefix(text, "error: response lost; the action may or may not have happened") {
		t.Errorf("text = %q", text)
	}
	if srv.Errors() != 1 {
		t.Errorf("Errors() = %d, want 1", srv.Errors())
	}
}

func TestReconnectAfterLostResponse(t *testing.T) {
	fc := &fakeConn{defs: callToolset()}
	srv := connectFake(t, fc, testCfg("s", config.MCPToolFilter{}), testDeps(&recObserver{}))
	ctx := context.Background()

	loseSession(t, srv, fc)
	// The next call reconnects (a fresh in-memory server) and succeeds; the
	// toolset is frozen, so no rediscovery happens.
	before := len(srv.Listed())
	text, out, err := toolNamed(t, srv, "s__echo").InvokeOutcome(ctx, `{"text":"back"}`)
	if err != nil {
		t.Fatalf("echo after reconnect: %v", err)
	}
	if text != "back" || out.Kind != tools.OutcomeOK {
		t.Errorf("got (%q, %v), want (back, ok)", text, out.Kind)
	}
	if fc.count() != 2 {
		t.Errorf("factory calls = %d, want 2 (initial + one reconnect)", fc.count())
	}
	if len(srv.Listed()) != before {
		t.Errorf("Listed() changed across a reconnect: %d -> %d", before, len(srv.Listed()))
	}
}

func TestReconnectSingleflightOnce(t *testing.T) {
	fc := &fakeConn{defs: callToolset()}
	srv := connectFake(t, fc, testCfg("s", config.MCPToolFilter{}), testDeps(&recObserver{}))
	ctx := context.Background()

	loseSession(t, srv, fc)

	// From here the factory blocks until the test releases it, so the second
	// caller is guaranteed to arrive while the first attempt is in flight.
	entered := make(chan struct{}, 1)
	release := make(chan struct{})
	fc.mu.Lock()
	fc.entered, fc.release = entered, release
	fc.failErr = errNoServer
	callsBefore := fc.calls
	fc.mu.Unlock()

	var wg sync.WaitGroup
	errs := make([]error, 2)
	texts := make([]string, 2)
	outs := make([]tools.Outcome, 2)
	for i := range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			texts[i], outs[i], errs[i] = toolNamed(t, srv, "s__echo").InvokeOutcome(ctx, `{"text":"x"}`)
		}()
	}
	// One caller is now inside the factory and the other is queued behind it;
	// only then is the attempt allowed to finish, so "one factory call" is a
	// property of the code and not of the scheduler.
	<-entered
	time.Sleep(200 * time.Millisecond)
	close(release)
	wg.Wait()

	if got := fc.count() - callsBefore; got != 1 {
		t.Errorf("factory calls = %d, want 1 (singleflight)", got)
	}
	for i := range 2 {
		if errs[i] == nil {
			t.Errorf("call %d: err = nil, want a reconnect error (text %q)", i, texts[i])
			continue
		}
		if !strings.Contains(errs[i].Error(), "reconnect") || !strings.Contains(errs[i].Error(), `"s"`) {
			t.Errorf("call %d: err = %v, want a named reconnect error", i, errs[i])
		}
		if outs[i].Kind != tools.OutcomeOK {
			t.Errorf("call %d: outcome = %v, want the zero outcome (nothing was sent)", i, outs[i].Kind)
		}
	}
	if srv.Errors() < 2 {
		t.Errorf("Errors() = %d, want the lost call plus the failed reconnect", srv.Errors())
	}
}

// connectAnnotated connects to a fake server whose tools carry annotations:
// "d" claims both destructive and read-only, "r" is read-only and idempotent
// in a closed world, "p" publishes no annotations at all.
func connectAnnotated(t *testing.T) *Server {
	t.Helper()
	yes, no := true, false
	defs := []fakeTool{
		{def: &sdk.Tool{Name: "d", InputSchema: objectSchema(), Annotations: &sdk.ToolAnnotations{DestructiveHint: &yes, ReadOnlyHint: true}}},
		{def: &sdk.Tool{Name: "r", InputSchema: objectSchema(), Annotations: &sdk.ToolAnnotations{ReadOnlyHint: true, IdempotentHint: true, OpenWorldHint: &no}}},
		{def: &sdk.Tool{Name: "p", InputSchema: objectSchema()}},
	}
	for i := range defs {
		defs[i].handler = func(context.Context, *sdk.CallToolRequest) (*sdk.CallToolResult, error) {
			return &sdk.CallToolResult{}, nil
		}
	}
	return connectFake(t, &fakeConn{defs: defs}, testCfg("s", config.MCPToolFilter{}), testDeps(nil))
}

func TestHint(t *testing.T) {
	srv := connectAnnotated(t)
	cases := map[string]string{
		"s__d": "server marks this tool destructive", // destructive wins
		"s__r": "server marks this tool read-only",
		"s__p": "",
	}
	for name, want := range cases {
		if got := toolNamed(t, srv, name).Hint(); got != want {
			t.Errorf("%s.Hint() = %q, want %q", name, got, want)
		}
	}
}

// wantBool compares one annotation field against an expected tri-state.
func wantBool(t *testing.T, field string, got, want *bool) {
	t.Helper()
	switch {
	case want == nil && got != nil:
		t.Errorf("%s = %v, want nil (the server did not say)", field, *got)
	case want != nil && got == nil:
		t.Errorf("%s = nil, want a set %v", field, *want)
	case want != nil && *got != *want:
		t.Errorf("%s = %v, want %v", field, *got, *want)
	}
}

func TestAnnotationsMapping(t *testing.T) {
	srv := connectAnnotated(t)
	yes, no := true, false

	// A server that sends no annotations object leaves every field nil; one
	// that sends an object always sets the two fields the wire format spells
	// as plain bools.
	plain := toolNamed(t, srv, "s__p").Annotations()
	wantBool(t, "p.read only", plain.ReadOnly, nil)
	wantBool(t, "p.destructive", plain.Destructive, nil)
	wantBool(t, "p.open world", plain.OpenWorld, nil)
	wantBool(t, "p.idempotent", plain.Idempotent, nil)

	a := toolNamed(t, srv, "s__r").Annotations()
	wantBool(t, "r.read only", a.ReadOnly, &yes)
	wantBool(t, "r.idempotent", a.Idempotent, &yes)
	wantBool(t, "r.open world", a.OpenWorld, &no)
	wantBool(t, "r.destructive", a.Destructive, nil)

	for _, lt := range srv.Listed() {
		if lt.Original == "r" {
			wantBool(t, "listed r.read only", lt.Annotations.ReadOnly, &yes)
		}
	}
}
