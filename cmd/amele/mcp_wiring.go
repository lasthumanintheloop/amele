package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/mcp"
	"github.com/lasthumanintheloop/amele/internal/session"
	"github.com/lasthumanintheloop/amele/internal/tools"
)

// maxMCPStderrBytes bounds how much of ONE stdio server's stderr this process
// relays per run. A server that logs a line per request must not be able to
// fill a cron mail spool or a journald ring buffer through amele; the cap is
// generous enough for a real startup banner and a stack trace, and the drop is
// announced so nobody debugs against silently truncated output.
const maxMCPStderrBytes = 64 << 10

// mcpSet is the run's live MCP connections plus what cmd needs from them: the
// ask-prompt hints of the tools that were registered, and the count of
// MCP-attributable failures that goes into run_end.mcp_errors.
type mcpSet struct {
	// servers are the connected servers, in config order.
	servers []*mcp.Server
	// hints maps a model-facing tool name to the phrase the approval question
	// shows beside it (see (*mcp.Tool).Hint).
	hints map[string]string
	// failed counts servers that were declared `required: false` and could not
	// be reached - the run went on without them, but the loss is a fact the
	// log must carry.
	failed int
	// relays are the per-server stderr writers, kept so their last partial
	// line can be flushed at shutdown.
	relays []*mcpStderr
}

// close ends every session, in config order, and flushes the servers' stderr.
//
// CONTRACT (docs/contracts/jsonl-events.md): this emits the mcp_disconnect
// events, so it must run before run_end is written - on the success path, on
// every error path, and on the signal path alike. Callers pass a context that
// is NOT the (possibly already cancelled) run context, so an orderly close
// still gets its grace period after a SIGTERM.
func (m *mcpSet) close(ctx context.Context) {
	if m == nil {
		return
	}
	for _, s := range m.servers {
		_ = s.Close(ctx) // Close is idempotent and reports only what it logged
	}
	for _, r := range m.relays {
		r.flush()
	}
}

// errors reports how many MCP-attributable failures the run saw: the optional
// servers that never came up, plus every mid-run failure the live servers
// counted (a lost session, a failed reconnect, a call that could not be
// dispatched).
func (m *mcpSet) errors() int {
	if m == nil {
		return 0
	}
	n := m.failed
	for _, s := range m.servers {
		n += s.Errors()
	}
	return n
}

// mcpUnavailableError is the operator-facing rendering of a required server's
// connect failure: `mcp server "github": auth: 401 Unauthorized`.
//
// It exists because *mcp.ConnectError phrases the same facts for a library
// caller ("mcp server %q unavailable (%s): %v"), and this is the line that
// lands in a cron mail. It unwraps to the ConnectError, so errors.Is still
// finds mcp.ErrUnavailable and exitCodeFor still returns 8.
type mcpUnavailableError struct {
	err *mcp.ConnectError
}

// Error implements error.
func (e *mcpUnavailableError) Error() string {
	return fmt.Sprintf("mcp server %q: %s: %s", e.err.Server, e.err.Class, mcpCause(e.err))
}

// mcpCause renders a ConnectError's cause without repeating the server name
// the caller is about to print: internal/mcp prefixes most of its causes with
// it, and `mcp server "files": spawn: mcp server "files": starting ...` reads
// like two failures instead of one.
func mcpCause(ce *mcp.ConnectError) string {
	return strings.TrimPrefix(ce.Err.Error(), fmt.Sprintf("mcp server %q: ", ce.Server))
}

// Unwrap exposes the ConnectError (and through it mcp.ErrUnavailable).
func (e *mcpUnavailableError) Unwrap() error { return e.err }

// connectMCP brings up every declared MCP server and registers its tools.
//
// Connecting is done in PARALLEL - a run with three servers must not pay three
// spawn-and-handshake round trips in sequence - but registration is strictly
// sequential in config order, so the tool list the model sees does not depend
// on which server answered first.
//
// Failure policy (docs/contracts/exit-codes.md): a `required` server that
// cannot be reached returns an error wrapping mcp.ErrUnavailable (exit 8); an
// optional one only warns and is counted. A toolset error - a tool name that
// is already taken - is fatal for required and optional servers alike
// (mcp.ErrToolset, exit 2): it is a mistake in the config, and no retry, no
// degradation and no other run can make it work.
//
// w may be nil (no session_dir, or `explain` inspecting a config), in which
// case the events are simply discarded.
func connectMCP(ctx context.Context, cfg *config.Config, reg *tools.Registry, w *session.Writer,
	stderr io.Writer, env config.LookupEnv, quiet bool, version string) (*mcpSet, error) {
	set := &mcpSet{hints: map[string]string{}}
	servers := cfg.MCP.Servers
	if len(servers) == 0 {
		return set, nil
	}

	// Snapshot of the names already taken by builtins and subprocess tools.
	// Collisions BETWEEN servers cannot be caught here (discovery runs in
	// parallel, so no server knows what its neighbours will publish); they are
	// caught at Register below, which is why registration is sequential.
	existing := make(map[string]bool, len(reg.Names()))
	for _, name := range reg.Names() {
		existing[name] = true
	}

	// One observer and one stderr mutex for the whole set: session.Writer and
	// the process's stderr are both single-threaded resources, and every
	// server writes to them from its own goroutine.
	observer := &sessionObserver{w: w}
	redact := session.Redactor(agentSecrets(cfg))
	var stderrMu sync.Mutex

	type result struct {
		srv *mcp.Server
		err error
	}
	results := make([]result, len(servers))
	var wg sync.WaitGroup
	for i, s := range servers {
		relay := &mcpStderr{mu: &stderrMu, w: stderr, redact: redact,
			prefix: fmt.Sprintf("amele: mcp %s: ", s.Name), budget: maxMCPStderrBytes}
		set.relays = append(set.relays, relay)
		wg.Add(1)
		go func(i int, s config.MCPServer) {
			defer wg.Done()
			// cmd is the composition root: it is the one place allowed to
			// name the real clock and the real random source, which every
			// package below takes injected (docs/engineering.md §5.4).
			results[i].srv, results[i].err = mcp.Connect(ctx, s, mcp.Deps{
				Clock:         time.Now,
				Rand:          rand.Float64,
				Env:           env,
				Observer:      observer,
				Stderr:        relay,
				Workspace:     cfg.Workspace,
				ExistingNames: existing,
				Version:       version,
			})
		}(i, s)
	}
	wg.Wait()

	var fatal error
	fail := func(err error) {
		if fatal == nil {
			fatal = err
		}
	}
	// An interruption fails every connect at once. Those failures are the
	// run ending, not the servers being unhealthy, so they are not counted or
	// warned about - an operator must not read "3 mcp errors" off a fleet that
	// was simply told to stop.
	interrupted := ctx.Err() != nil
	for i, s := range servers {
		r := results[i]
		if r.err != nil {
			set.connectFailed(s, r.err, quiet || interrupted, interrupted, stderr, fail)
			continue
		}
		// Appended before registration so a half-registered server is still
		// closed by mcpSet.close: an abandoned child process outliving the run
		// is exactly what the process-group kill exists to prevent.
		set.servers = append(set.servers, r.srv)
		set.register(s.Name, r.srv, reg, fail)
	}
	return set, fatal
}

// connectFailed applies the required/optional policy to one failed connect.
// interrupted says the run's context ended, in which case an optional server's
// failure is not the server's fault and is neither counted nor reported.
func (m *mcpSet) connectFailed(s config.MCPServer, err error, quiet, interrupted bool, stderr io.Writer, fail func(error)) {
	if errors.Is(err, mcp.ErrToolset) {
		fail(fmt.Errorf("mcp server %q: %w", s.Name, err))
		return
	}
	var ce *mcp.ConnectError
	if s.IsRequired() {
		if errors.As(err, &ce) {
			fail(&mcpUnavailableError{err: ce})
			return
		}
		fail(fmt.Errorf("mcp server %q: %w", s.Name, err))
		return
	}
	if interrupted {
		return
	}
	// Opted out of fail-fast: the run continues with fewer tools, but the
	// operator hears about it once and run_end.mcp_errors carries the count.
	m.failed++
	if quiet {
		return
	}
	// The class and the cause are read off the ConnectError rather than
	// printing its own sentence, which would repeat "mcp server %q
	// unavailable" inside a line that already says it.
	if errors.As(err, &ce) {
		_, _ = fmt.Fprintf(stderr, "amele: warning: mcp server %q unavailable (%s): %s\n", s.Name, ce.Class, mcpCause(ce))
		return
	}
	_, _ = fmt.Fprintf(stderr, "amele: warning: mcp server %q unavailable: %v\n", s.Name, err)
}

// register adds one server's tools to the registry and collects their hints.
func (m *mcpSet) register(name string, srv *mcp.Server, reg *tools.Registry, fail func(error)) {
	for _, t := range srv.Tools() {
		if err := reg.Register(t); err != nil {
			// A name already taken by a builtin, a subprocess tool or an
			// earlier server. CONTRACT: exit 2 - the config is wrong.
			fail(fmt.Errorf("mcp server %q: %w: %v", name, mcp.ErrToolset, err))
			return
		}
		if mt, ok := t.(*mcp.Tool); ok {
			if hint := mt.Hint(); hint != "" {
				m.hints[t.Def().Name] = hint
			}
		}
	}
}

// sessionObserver writes the MCP connection lifecycle into the session log.
//
// It is the adapter that keeps internal/mcp free of any logging dependency.
// The mutex is not optional: session.Writer is a plain appender with no
// locking of its own, and connectMCP drives every server from its own
// goroutine.
type sessionObserver struct {
	mu sync.Mutex
	w  *session.Writer
}

// Connected implements mcp.Observer.
func (o *sessionObserver) Connected(info mcp.ConnectInfo) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.w.MCPConnect(session.MCPConnect{
		Server: info.Server, Transport: info.Transport, OK: true,
		DurationMS:      info.Duration.Milliseconds(),
		ProtocolVersion: info.ProtocolVersion,
		ServerName:      info.ServerName, ServerVersion: info.ServerVersion,
		SessionFP: info.SessionFP, ToolCount: info.ToolCount,
	})
}

// ConnectFailed implements mcp.Observer.
func (o *sessionObserver) ConnectFailed(server, transport string, class mcp.ErrorClass, err error, d time.Duration) {
	o.mu.Lock()
	defer o.mu.Unlock()
	// The Writer redacts and clips the message; a server's error text is
	// remote input and may quote a header this run interpolated.
	o.w.MCPConnect(session.MCPConnect{
		Server: server, Transport: transport, OK: false,
		ErrorClass: string(class), Error: err.Error(),
		DurationMS: d.Milliseconds(),
	})
}

// ToolsListed implements mcp.Observer.
func (o *sessionObserver) ToolsListed(server string, listed []mcp.ListedTool, totalBytes int, skipped []mcp.SkippedTool) {
	o.mu.Lock()
	defer o.mu.Unlock()
	e := session.MCPToolsListed{Server: server, TotalBytes: totalBytes}
	for _, l := range listed {
		t := session.MCPToolListed{
			Name: l.Name, SHA256: l.SHA256, Bytes: l.Bytes,
			Annotations: annotationMap(l.Annotations),
		}
		// Only when the name was rewritten: for a plain join the original is
		// the suffix of Name and repeating it would be noise.
		if l.Normalized {
			t.OriginalName = l.Original
		}
		e.Tools = append(e.Tools, t)
	}
	for _, s := range skipped {
		e.Skipped = append(e.Skipped, session.MCPSkippedTool{Name: s.Name, Reason: s.Reason})
	}
	o.w.MCPToolsListed(e)
}

// Disconnected implements mcp.Observer.
func (o *sessionObserver) Disconnected(server, reason string) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.w.MCPDisconnect(session.MCPDisconnect{Server: server, Reason: reason})
}

// annotationMap flattens the MCP tool hints for the log. A nil pointer means
// the server said nothing, and the key stays absent - an explicit `false` and
// "unstated" are different facts to an operator auditing a toolset.
func annotationMap(a mcp.Annotations) map[string]bool {
	m := map[string]bool{}
	for key, value := range map[string]*bool{
		"readOnly": a.ReadOnly, "destructive": a.Destructive,
		"openWorld": a.OpenWorld, "idempotent": a.Idempotent,
	} {
		if value != nil {
			m[key] = *value
		}
	}
	if len(m) == 0 {
		return nil
	}
	return m
}

// mcpStderr relays one stdio server's stderr to this process's stderr.
//
// SECURITY, three properties in one writer:
//
//   - Redaction. A server started with an ${ENV} credential in its environment
//     routinely echoes it into a startup banner or an error. The same
//     session.Redactor the session log and the -v feed use runs here, so
//     adding an output channel did not add a leak channel.
//   - Terminal safety. Server output is remote text; it goes through
//     safeForTerminal for the same reason the approval question does - an
//     escape sequence could otherwise repaint the operator's screen.
//   - Boundedness. maxMCPStderrBytes caps the whole relay, and the drop is
//     announced rather than silent.
//
// It is line-oriented (a write is buffered until a newline) so the prefix
// names the server exactly once per line, and it takes a SHARED mutex because
// every server in the run writes to the same underlying stderr.
type mcpStderr struct {
	mu     *sync.Mutex
	w      io.Writer
	prefix string
	redact func(string) string

	buf     []byte
	budget  int
	dropped bool
}

// Write implements io.Writer. It never reports an error: a broken stderr must
// not take down a working MCP session.
func (s *mcpStderr) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.buf = append(s.buf, p...)
	for {
		i := bytes.IndexByte(s.buf, '\n')
		if i < 0 {
			// A server that never emits a newline must not grow this buffer
			// without bound; flush what would be one screen line anyway.
			if len(s.buf) >= maxProgressLine {
				s.emit(string(s.buf))
				s.buf = s.buf[:0]
			}
			return len(p), nil
		}
		s.emit(string(s.buf[:i]))
		s.buf = s.buf[i+1:]
	}
}

// flush writes the last partial line, if any. Called once at shutdown.
func (s *mcpStderr) flush() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.buf) > 0 {
		s.emit(string(s.buf))
		s.buf = s.buf[:0]
	}
}

// emit renders one line. The caller holds the mutex.
func (s *mcpStderr) emit(line string) {
	line = strings.TrimSuffix(line, "\r")
	if s.budget <= 0 {
		if !s.dropped {
			s.dropped = true
			_, _ = fmt.Fprintf(s.w, "%s(further output dropped)\n", s.prefix)
		}
		return
	}
	s.budget -= len(line)
	_, _ = fmt.Fprintf(s.w, "%s%s\n", s.prefix, safeForTerminal(s.redact(line), maxProgressLine))
}
