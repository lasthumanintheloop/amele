package mcp

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// testBuildDir holds the compiled test server for the whole package run; it is
// built at most once (go build is the slowest thing in this file) and removed
// by TestMain.
var (
	testBuildDir   string
	testServerOnce sync.Once
	testServerPath string
	testServerErr  error
)

func TestMain(m *testing.M) {
	code := m.Run()
	if testBuildDir != "" {
		_ = os.RemoveAll(testBuildDir)
	}
	os.Exit(code)
}

// buildTestServer compiles testdata/mcptestserver once and returns its path.
func buildTestServer(t *testing.T) string {
	t.Helper()
	testServerOnce.Do(func() {
		dir, err := os.MkdirTemp("", "amele-mcptestserver")
		if err != nil {
			testServerErr = err
			return
		}
		testBuildDir = dir
		bin := filepath.Join(dir, "mcptestserver")
		out, err := exec.Command("go", "build", "-o", bin, "./testdata/mcptestserver").CombinedOutput() //nolint:gosec // bin is a path this test just created
		if err != nil {
			testServerErr = fmt.Errorf("go build: %w: %s", err, out)
			return
		}
		testServerPath = bin
	})
	if testServerErr != nil {
		t.Fatalf("building test server: %v", testServerErr)
	}
	return testServerPath
}

// lookup builds an env func over a fixed map.
func lookup(m map[string]string) func(string) (string, bool) {
	return func(k string) (string, bool) {
		v, ok := m[k]
		return v, ok
	}
}

// connectStdio starts the test server over stdio and returns a live session.
func connectStdio(t *testing.T, ctx context.Context, argv []string, allow []string, env func(string) (string, bool), stderr io.Writer) *sdk.ClientSession {
	t.Helper()
	tr, kill, err := newStdioTransport(ctx, argv, t.TempDir(), allow, env, stderr)
	if err != nil {
		t.Fatalf("newStdioTransport: %v", err)
	}
	t.Cleanup(kill)
	sess, err := sdk.NewClient(&sdk.Implementation{Name: "amele-test", Version: "0"}, nil).Connect(ctx, tr, nil)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	t.Cleanup(func() { _ = sess.Close() })
	return sess
}

// callText calls a tool and returns its first text block.
func callText(t *testing.T, ctx context.Context, sess *sdk.ClientSession, name string, args map[string]any) string {
	t.Helper()
	res, err := sess.CallTool(ctx, &sdk.CallToolParams{Name: name, Arguments: args})
	if err != nil {
		t.Fatalf("CallTool(%s): %v", name, err)
	}
	if len(res.Content) == 0 {
		t.Fatalf("CallTool(%s): no content", name)
	}
	tc, ok := res.Content[0].(*sdk.TextContent)
	if !ok {
		t.Fatalf("CallTool(%s): content is %T, want text", name, res.Content[0])
	}
	return tc.Text
}

// lookupPATH is the smallest env a spawned server needs: the PATH of the test
// process (so it can find "sleep") and nothing else.
func lookupPATH() func(string) (string, bool) {
	return lookup(map[string]string{"PATH": os.Getenv("PATH")})
}

func TestStdioMinimalEnv(t *testing.T) {
	bin := buildTestServer(t)
	base := map[string]string{"SECRET": "1", "HOME": "/h", "PATH": os.Getenv("PATH")}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	t.Run("secret is not inherited", func(t *testing.T) {
		sess := connectStdio(t, ctx, []string{bin}, nil, lookup(base), io.Discard)
		if got := callText(t, ctx, sess, "env", map[string]any{"name": "SECRET"}); got != "<unset>" {
			t.Errorf("SECRET = %q, want <unset>", got)
		}
		if got := callText(t, ctx, sess, "env", map[string]any{"name": "HOME"}); got != "/h" {
			t.Errorf("HOME = %q, want /h", got)
		}
	})

	t.Run("allowlisted var passes through", func(t *testing.T) {
		sess := connectStdio(t, ctx, []string{bin}, []string{"SECRET"}, lookup(base), io.Discard)
		if got := callText(t, ctx, sess, "env", map[string]any{"name": "SECRET"}); got != "1" {
			t.Errorf("SECRET = %q, want 1", got)
		}
	})
}

func TestStdioSpawnError(t *testing.T) {
	_, _, err := newStdioTransport(context.Background(), []string{"/nonexistent/amele-mcp"}, t.TempDir(), nil, os.LookupEnv, io.Discard)
	if err == nil {
		t.Fatal("newStdioTransport succeeded for a missing binary")
	}
}

func TestStdioMessageCap(t *testing.T) {
	bin := buildTestServer(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	sess := connectStdio(t, ctx, []string{bin}, nil, os.LookupEnv, io.Discard)

	_, err := sess.CallTool(ctx, &sdk.CallToolParams{Name: "big", Arguments: map[string]any{"n": MaxMessageBytes + 1}})
	if err == nil {
		t.Fatal("CallTool(big) succeeded, want the message cap to break the connection")
	}
	if !IsMessageTooLarge(err) {
		t.Fatalf("error = %v, want the size cap", err)
	}
}

// recordingHandler counts GET requests and remembers the last Authorization
// header, so the HTTP tests can assert on what actually went over the wire.
type recordingHandler struct {
	next  http.Handler
	gets  atomic.Int64
	auth  atomic.Value // string
	calls atomic.Int64
}

func (h *recordingHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.calls.Add(1)
	if r.Method == http.MethodGet {
		h.gets.Add(1)
	}
	h.auth.Store(r.Header.Get("Authorization"))
	h.next.ServeHTTP(w, r)
}

// startHTTPServer runs an in-process MCP server over Streamable HTTP.
func startHTTPServer(t *testing.T) (*httptest.Server, *recordingHandler) {
	t.Helper()
	srv := sdk.NewServer(&sdk.Implementation{Name: "amele-http-test", Version: "0"}, nil)
	sdk.AddTool(srv, &sdk.Tool{Name: "echo", Description: "echo"},
		func(_ context.Context, _ *sdk.CallToolRequest, in struct {
			Text string `json:"text"`
		}) (*sdk.CallToolResult, any, error) {
			return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: in.Text}}}, nil, nil
		})
	handler := sdk.NewStreamableHTTPHandler(func(*http.Request) *sdk.Server { return srv }, nil)
	rec := &recordingHandler{next: handler}
	ts := httptest.NewServer(rec)
	t.Cleanup(ts.Close)
	return ts, rec
}

func TestHTTPHeadersAndNoSSE(t *testing.T) {
	ts, rec := startHTTPServer(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	tr, err := newHTTPTransport(ts.URL, map[string]string{"Authorization": "Bearer t0ken"})
	if err != nil {
		t.Fatalf("newHTTPTransport: %v", err)
	}
	sess, err := sdk.NewClient(&sdk.Implementation{Name: "amele-test", Version: "0"}, nil).Connect(ctx, tr, nil)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = sess.Close() }()

	if _, err := sess.ListTools(ctx, nil); err != nil {
		t.Fatalf("ListTools: %v", err)
	}
	if got := callText(t, ctx, sess, "echo", map[string]any{"text": "hi"}); got != "hi" {
		t.Errorf("echo = %q, want hi", got)
	}
	if got, _ := rec.auth.Load().(string); got != "Bearer t0ken" {
		t.Errorf("Authorization = %q, want Bearer t0ken", got)
	}
	if n := rec.gets.Load(); n != 0 {
		t.Errorf("GET requests = %d, want 0 (standalone SSE must be disabled)", n)
	}
}

func TestHTTPBodyCap(t *testing.T) {
	// The handler streams forever: if the cap did not exist, the client would
	// read until it ran out of memory. Flushing keeps bytes moving so the
	// failure is the cap and not a stalled connection.
	served := make(chan struct{})
	var servedOnce sync.Once
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Once, not a bare close: the SDK may retry the POST, and closing a
		// closed channel panics inside the test server's goroutine.
		defer servedOnce.Do(func() { close(served) })
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Error("ResponseWriter does not flush")
			return
		}
		chunk := bytes.Repeat([]byte("x"), 64<<10)
		for {
			select {
			case <-r.Context().Done():
				return
			default:
			}
			if _, err := w.Write(chunk); err != nil {
				return
			}
			flusher.Flush()
		}
	}))
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	tr, err := newHTTPTransport(ts.URL, nil)
	if err != nil {
		t.Fatalf("newHTTPTransport: %v", err)
	}

	start := time.Now()
	sess, err := sdk.NewClient(&sdk.Implementation{Name: "amele-test", Version: "0"}, nil).Connect(ctx, tr, nil)
	elapsed := time.Since(start)
	if err == nil {
		_ = sess.Close()
		t.Fatal("connect succeeded against an endless body, want the cap to fail it")
	}
	// The cap, not a decode error further down: without this assertion the test
	// would pass even with no cap at all.
	if !IsMessageTooLarge(err) {
		t.Fatalf("error = %v, want the size cap", err)
	}
	// Bounded: reading 8 MiB and giving up, not chasing an endless stream.
	if elapsed > 10*time.Second {
		t.Errorf("connect took %v, want the cap to end it quickly", elapsed)
	}
	select {
	case <-served:
	case <-time.After(10 * time.Second):
		t.Error("handler still streaming after the client gave up")
	}
}

func TestIsMessageTooLarge(t *testing.T) {
	if IsMessageTooLarge(nil) {
		t.Error("IsMessageTooLarge(nil) = true")
	}
	if IsMessageTooLarge(errors.New("something else")) {
		t.Error("IsMessageTooLarge matched an unrelated error")
	}
	if !IsMessageTooLarge(fmt.Errorf("wrapped: %w", errMessageTooLarge)) {
		t.Error("IsMessageTooLarge missed a wrapped sentinel")
	}
	// The SDK severs the chain with %v; the text must still be recognised.
	if !IsMessageTooLarge(errors.New("read: " + msgTooLargeText + " (cap 8388608 bytes)")) {
		t.Error("IsMessageTooLarge missed a flattened error")
	}
}

func TestHTTPRedirectOtherOriginRejected(t *testing.T) {
	other, _ := startHTTPServer(t)
	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, other.URL, http.StatusFound)
	}))
	defer redirector.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	tr, err := newHTTPTransport(redirector.URL, map[string]string{"Authorization": "Bearer t0ken"})
	if err != nil {
		t.Fatalf("newHTTPTransport: %v", err)
	}
	sess, err := sdk.NewClient(&sdk.Implementation{Name: "amele-test", Version: "0"}, nil).Connect(ctx, tr, nil)
	if err == nil {
		_ = sess.Close()
		t.Fatal("connect followed a cross-origin redirect, want refusal")
	}
	if !strings.Contains(err.Error(), "refused") {
		t.Errorf("error = %v, want a refused-redirect error", err)
	}
}

func TestHTTP401ClassifiedAuth(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "nope", http.StatusUnauthorized)
	}))
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	tr, err := newHTTPTransport(ts.URL, nil)
	if err != nil {
		t.Fatalf("newHTTPTransport: %v", err)
	}
	sess, err := sdk.NewClient(&sdk.Implementation{Name: "amele-test", Version: "0"}, nil).Connect(ctx, tr, nil)
	if err == nil {
		_ = sess.Close()
		t.Fatal("connect succeeded against a 401")
	}
	// The SDK renders the status as text ("Unauthorized"), not as the number;
	// Task 9's classifier must match on that spelling.
	if !strings.Contains(err.Error(), "Unauthorized") && !strings.Contains(err.Error(), "401") {
		t.Errorf("error = %v, want it to name the 401", err)
	}
}

func TestNewHTTPTransportRejectsBadURL(t *testing.T) {
	if _, err := newHTTPTransport("://nope", nil); err == nil {
		t.Fatal("newHTTPTransport accepted an unparseable URL")
	}
}

func TestChildEnvMinimal(t *testing.T) {
	env := lookup(map[string]string{"PATH": "/bin", "HOME": "/h", "LANG": "C", "SECRET": "s", "X": "1"})
	got := childEnv(nil, env)
	want := []string{"PATH=/bin", "HOME=/h", "LANG=C"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Errorf("childEnv(nil) = %v, want %v", got, want)
	}
	got = childEnv([]string{"X", "PATH", "MISSING"}, env)
	want = []string{"PATH=/bin", "HOME=/h", "LANG=C", "X=1"}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Errorf("childEnv(allow) = %v, want %v", got, want)
	}
}

func TestLineCappedReader(t *testing.T) {
	// Regression: a chunk that carries both the tail of an oversized line and
	// its newline used to reset the counter before the line was charged.
	tests := []struct {
		name    string
		line    int
		max     int
		wantErr bool
	}{
		{"under the cap", 10, 16, false},
		{"exactly the cap", 16, 16, false},
		{"over the cap", 17, 16, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			data := append(bytes.Repeat([]byte("y"), tc.line-1), '\n')
			r := &lineCappedReader{r: io.NopCloser(bytes.NewReader(data)), max: tc.max}
			_, err := io.Copy(io.Discard, r)
			if tc.wantErr {
				if !errors.Is(err, errMessageTooLarge) {
					t.Fatalf("err = %v, want errMessageTooLarge", err)
				}
				// The failure is sticky: the connection is over.
				if _, err := r.Read(make([]byte, 1)); !errors.Is(err, errMessageTooLarge) {
					t.Errorf("second Read err = %v, want errMessageTooLarge", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestCappedWriter(t *testing.T) {
	var buf bytes.Buffer
	w := &cappedWriter{w: &buf, max: 5}
	n, err := w.Write([]byte("abcdefgh"))
	if err != nil || n != 8 {
		t.Fatalf("Write = %d, %v; want 8, nil (a dropped tail is not a short write)", n, err)
	}
	if n, err := w.Write([]byte("ijk")); err != nil || n != 3 {
		t.Fatalf("second Write = %d, %v; want 3, nil", n, err)
	}
	if got := buf.String(); got != "abcde" {
		t.Errorf("captured %q, want abcde", got)
	}
}
