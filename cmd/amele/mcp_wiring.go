package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/explain"
	"github.com/lasthumanintheloop/amele/internal/mcp"
	"github.com/lasthumanintheloop/amele/internal/oauthtoken"
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
	// failed counts servers that could not be reached, required and optional
	// alike: an optional one the run went on without, and a required one that
	// is about to end the run with exit 8 - either way run_end.mcp_errors must
	// carry the loss, or a log line saying "0 mcp errors" would sit next to an
	// exit code that says the opposite.
	failed int
	// secrets is the invocation's shared secret registry, kept so an MCP
	// server that mints a credential (an OAuth access token) can register it
	// on the SAME set every sink of this run already redacts through.
	secrets *session.SecretSet
	// redact is the run's secret scrubber, applied to every operator-facing
	// line this set prints: a connect error may quote a header value the
	// config interpolated. It is secrets.Redact, so it sees values added
	// after the set was built.
	redact func(string) string
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

// dialResult is one server's connect outcome: exactly one of the fields is
// set.
type dialResult struct {
	srv *mcp.Server
	err error
}

// dialMCP connects every declared server IN PARALLEL and returns the outcomes
// in config order, appending one stderr relay per server to set.
//
// It is the phase connectMCP and explainMCP share: both must dial the same
// way (same deps, same parallelism, same stderr relays), and only what they do
// with a failure differs - a run applies the required/optional policy, a report
// prints it. Keeping the dial in one function is what makes "explain sees what
// run would see" a property of the code rather than of two copies staying in
// step.
func dialMCP(ctx context.Context, cfg *config.Config, set *mcpSet, observer mcp.Observer,
	redact func(string) string, stderrMu *sync.Mutex, stderr io.Writer,
	env config.LookupEnv, existing map[string]bool, version string) []dialResult {
	servers := cfg.MCP.Servers
	results := make([]dialResult, len(servers))
	// cmd is the composition root, so the OAuth token store is built HERE -
	// one store for the whole invocation, over the XDG state directory the
	// `amele mcp login` commands write. internal/mcp never learns where
	// credentials live; it is handed a store or nothing.
	store := newTokenStore(env)
	// SecretSet has no de-duplication, and every connect attempt (and every
	// mid-run reconnect) builds a fresh OAuth handler that reads the same
	// stored tokens. Without this guard a flapping server would grow the
	// redactor's list one copy per attempt, for no added protection.
	register := dedupRegistrar(set.secrets.Add)
	var wg sync.WaitGroup
	for i, s := range servers {
		relay := &mcpStderr{mu: stderrMu, w: stderr, redact: redact,
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
				TokenStore:    store,
				// A token refreshed mid-run must be scrubbed from every sink
				// this invocation already redacts through, which is why the
				// registry is the run's live SecretSet rather than a snapshot.
				RegisterSecret: register,
			})
		}(i, s)
	}
	wg.Wait()
	return results
}

// dedupRegistrar wraps a secret registrar so each distinct value is passed on
// at most once. Empty values are dropped; the returned function is safe for
// concurrent use, because servers are dialled in parallel.
func dedupRegistrar(add func(...string)) func(...string) {
	var (
		mu   sync.Mutex
		seen = map[string]bool{}
	)
	return func(values ...string) {
		mu.Lock()
		fresh := make([]string, 0, len(values))
		for _, v := range values {
			if v == "" || seen[v] {
				continue
			}
			seen[v] = true
			fresh = append(fresh, v)
		}
		mu.Unlock()
		if len(fresh) > 0 {
			add(fresh...)
		}
	}
}

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
	stderr io.Writer, env config.LookupEnv, quiet bool, version string,
	secrets *session.SecretSet) (*mcpSet, error) {
	set := &mcpSet{hints: map[string]string{}, secrets: secrets, redact: secrets.Redact}
	servers := cfg.MCP.Servers
	if len(servers) == 0 {
		return set, nil
	}

	// Collisions BETWEEN servers cannot be caught from this snapshot (discovery
	// runs in parallel, so no server knows what its neighbours will publish);
	// they are caught at Register below, which is why registration is
	// sequential.
	existing := existingToolNames(reg)

	// One observer and one stderr mutex for the whole set: session.Writer and
	// the process's stderr are both single-threaded resources, and every
	// server writes to them from its own goroutine.
	observer := &sessionObserver{w: w, auth: mcpAuthKinds(cfg)}
	// The registry comes from the caller: `run` and `chat` build exactly one
	// per invocation and hand the same set to every other sink (see
	// runSecrets), so nothing here may build a second one.
	redact := set.redact
	var stderrMu sync.Mutex

	results := dialMCP(ctx, cfg, set, observer, redact, &stderrMu, stderr, env, existing, version)

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
		// Counted like an optional failure (unless the run was interrupted,
		// which is nobody's unavailability): the exit-8 run's run_end must
		// say how many declared dependencies were missing.
		if !interrupted {
			m.failed++
		}
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
	// unavailable" inside a line that already says it. SECURITY: the cause is
	// remote error text and may quote an interpolated header, so it goes
	// through the run's redactor like every other operator-facing line.
	if errors.As(err, &ce) {
		_, _ = fmt.Fprintf(stderr, "amele: warning: mcp server %q unavailable (%s): %s\n", s.Name, ce.Class, m.scrub(mcpCause(ce)))
		return
	}
	_, _ = fmt.Fprintf(stderr, "amele: warning: mcp server %q unavailable: %s\n", s.Name, m.scrub(err.Error()))
}

// scrub applies the set's redactor when it has one. Every set built by
// connectMCP or explainMCP has one; only a zero value in a test does not, and
// that passes text through.
func (m *mcpSet) scrub(text string) string {
	if m == nil || m.redact == nil {
		return text
	}
	return m.redact(text)
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
	// auth maps a server name to the credential mechanism its config declared
	// ("oauth"), so a FAILED connect can carry it too: ConnectFailed has no
	// ConnectInfo to read it off, and an oauth server that never came up is
	// exactly the connect a reader wants the mechanism on. A nil map is fine -
	// a lookup then yields "", which is what a server without auth reports.
	auth map[string]string
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
		SessionFP: info.SessionFP, ToolCount: info.ToolCount, Auth: info.Auth,
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
		DurationMS: d.Milliseconds(), Auth: o.auth[server],
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

// mcpAuthKinds is the server-name -> credential-mechanism map the observer
// needs for the connects that failed. It is read-only once built, so the
// observer's mutex does not have to cover it.
func mcpAuthKinds(cfg *config.Config) map[string]string {
	m := make(map[string]string, len(cfg.MCP.Servers))
	for _, s := range cfg.MCP.Servers {
		if s.Auth != nil {
			m[s.Name] = s.Auth.Type
		}
	}
	return m
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
//     routinely echoes it into a startup banner or an error. The same live
//     secret registry the session log and the -v feed use runs here, so
//     adding an output channel did not add a leak channel - including for a
//     credential that only came into existence mid-run.
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

// explainMCP dials every declared MCP server for `amele explain` and turns the
// outcomes into report rows.
//
// CONTRACT (docs/contracts/cli.md): explain reports, run gates. It connects for
// real - the toolset is the one part of an amele config that lives on the far
// side of a connection, so a dry run that did not ask for it would leave the
// largest surface unreviewed - but NO connect failure changes the exit code,
// not even for a `required: true` server that would abort `amele run` with
// exit 8. Failures become rows with Connected false.
//
// The tools of the servers that did answer ARE registered, so the WARNINGS
// section can tell a `permissions.tools` entry naming an MCP tool from a typo.
// A registration collision is the one MCP outcome that is a CONFIG error
// rather than an environmental one, so it is returned as a problem line for
// the PROBLEMS block instead of being buried in the server's row.
//
// The caller must call close on the returned set (with a context that is not
// the report's, so an orderly shutdown still gets its grace period): the stdio
// servers are live child processes.
//
// Nothing here writes to stderr: the command's contract is a report on stdout
// and an empty stderr.
func explainMCP(ctx context.Context, cfg *config.Config, cfgPath string, reg *tools.Registry,
	env config.LookupEnv, version string) ([]explain.MCPServerReport, []string, *mcpSet) {
	// `explain` is its own invocation with no other sink to share with, so it
	// builds the one registry it needs here.
	secrets := runSecrets(cfg)
	set := &mcpSet{hints: map[string]string{}, secrets: secrets, redact: secrets.Redact}
	if len(cfg.MCP.Servers) == 0 {
		return nil, nil, set
	}
	existing := existingToolNames(reg)
	// A nil session.Writer discards every event: explain writes no session
	// log, and its connects must not land in the one a `run` is keeping.
	observer := &sessionObserver{w: nil, auth: mcpAuthKinds(cfg)}
	redact := set.redact
	var stderrMu sync.Mutex
	// CONTRACT (docs/contracts/cli.md): `explain` writes the report to stdout
	// and NOTHING to stderr when it printed one. A stdio server's startup
	// banner is therefore discarded rather than relayed - what matters about a
	// server that failed to start is already in its connect error, which the
	// report prints.
	results := dialMCP(ctx, cfg, set, observer, redact, &stderrMu, io.Discard, env, existing, version)

	var problems []string
	reports := make([]explain.MCPServerReport, 0, len(results))
	// The credential rows are read AFTER the dial, so they describe the store
	// as the connect left it: an expired token the run-mode path just refreshed
	// (and rotated) must not be reported with the expiry it had a second ago.
	store := newTokenStore(env)
	for i, s := range cfg.MCP.Servers {
		r := results[i]
		if r.err != nil {
			row := failedServerReport(s, r.err)
			row.Auth, row.AuthStatus = mcpAuthReport(s, store, cfgPath)
			reports = append(reports, row)
			continue
		}
		// Appended before registration so a half-registered server is still
		// closed by the caller: an abandoned child process outliving the
		// report is exactly what the process-group kill exists to prevent.
		set.servers = append(set.servers, r.srv)
		set.register(s.Name, r.srv, reg, func(err error) {
			problems = append(problems, err.Error())
		})
		row := serverReport(s, r.srv)
		row.Auth, row.AuthStatus = mcpAuthReport(s, store, cfgPath)
		reports = append(reports, row)
	}
	return reports, problems, set
}

// mcpAuthReport summarises one server's credential for the explain report:
// the mechanism the config declared, plus what is stored for it.
//
// SECURITY: FACTS ABOUT the credential only - the expiry, whether a refresh
// token exists, how many records are filed. No token, and no issuer either:
// `amele mcp status` is the place for the full picture, and this line is meant
// to be safe in a pasted report.
//
// It reads the store through serverCredentials, the same lookup `status`,
// `logout` and the run path use, so the report cannot disagree with the
// runner about which credential a server would pick up.
func mcpAuthReport(s config.MCPServer, store *oauthtoken.Store, cfgPath string) (kind, status string) {
	if s.Auth == nil {
		return "", ""
	}
	kind = s.Auth.Type
	matches, err := serverCredentials(store, s)
	if len(matches) == 0 {
		if err != nil {
			// An unreadable store is not a failed report: explain reports.
			return kind, "credential state unknown: " + err.Error()
		}
		return kind, fmt.Sprintf("no token - run 'amele mcp login %s %s'", cfgPath, s.Name)
	}
	rec := freshestCredential(matches)
	// The raw expiry, with no refresh margin: this states what is on disk, and
	// the margin is a decision about when to refresh, not a fact about the
	// credential (the same rule `amele mcp status` follows).
	verb := "token valid until"
	if !rec.ExpiresAt.After(store.Now()) {
		verb = "token expired at"
	}
	status = fmt.Sprintf("%s %s, refresh: %s", verb,
		rec.ExpiresAt.UTC().Format(mcpTimeFormat), yesNo(rec.RefreshToken != ""))
	// More than one record can be filed for one server (a pre-registered client
	// and a discovered one). The row summarises the one a run would most likely
	// still be able to use, and says the others exist rather than hiding them.
	if extra := len(matches) - 1; extra > 0 {
		status += fmt.Sprintf(", +%d more stored", extra)
	}
	return kind, status
}

// freshestCredential picks the record with the latest expiry, the one most
// likely to carry a run without a refresh. matches must be non-empty.
func freshestCredential(matches []oauthtoken.Entry) *oauthtoken.Record {
	best := matches[0].Record
	for _, m := range matches[1:] {
		if m.Record.ExpiresAt.After(best.ExpiresAt) {
			best = m.Record
		}
	}
	return best
}

// failedServerReport renders a connect failure as a report row. The class and
// the cause are read off a *mcp.ConnectError when there is one, so the row
// says "auth: 401 Unauthorized" rather than repeating the "mcp server %q
// unavailable" preamble the row's own name column already carries.
func failedServerReport(s config.MCPServer, err error) explain.MCPServerReport {
	r := explain.MCPServerReport{
		Name: s.Name, Transport: s.Transport.Type, Target: mcpTarget(s),
		Error: err.Error(),
	}
	var ce *mcp.ConnectError
	if errors.As(err, &ce) {
		r.ErrorClass = string(ce.Class)
		r.Error = mcpCause(ce)
	}
	return r
}

// serverReport renders a live server as a report row: the handshake facts, the
// tools the model would receive, then the ones that were left out.
func serverReport(s config.MCPServer, srv *mcp.Server) explain.MCPServerReport {
	info := srv.Info()
	r := explain.MCPServerReport{
		Name: s.Name, Transport: s.Transport.Type, Target: mcpTarget(s),
		Connected: true, DurationMS: info.Duration.Milliseconds(),
		ProtocolVersion: info.ProtocolVersion,
		ServerName:      info.ServerName, ServerVersion: info.ServerVersion,
	}
	for _, l := range srv.Listed() {
		r.TotalBytes += l.Bytes
		r.Tools = append(r.Tools, explain.MCPToolReport{
			Name: l.Name, Original: l.Original, Normalized: l.Normalized,
			Bytes: l.Bytes, Hint: annotationHint(l.Annotations), Kept: true,
		})
	}
	for _, sk := range srv.Skipped() {
		r.Tools = append(r.Tools, explain.MCPToolReport{Name: sk.Name, Reason: sk.Reason})
	}
	r.EstTokens = explain.EstTokens(r.TotalBytes)
	return r
}

// annotationHint is the report's short spelling of a tool's annotations, in
// the same precedence (*mcp.Tool).Hint uses for the approval prompt: the
// warning first, the reassurance second. An unstated annotation yields "".
func annotationHint(a mcp.Annotations) string {
	switch {
	case a.Destructive != nil && *a.Destructive:
		return "destructive"
	case a.ReadOnly != nil && *a.ReadOnly:
		return "read-only"
	}
	return ""
}

// mcpTarget names what a server connection actually dials, for the report.
//
// SECURITY: the http form drops the URL's QUERY STRING. Some hosted MCP
// endpoints carry the credential there (?key=...), and explain is the report
// an operator pastes into a ticket - the redactor cannot help, because a query
// token need not have come from an ${ENV} the config interpolated.
func mcpTarget(s config.MCPServer) string {
	if s.Transport.Type == config.MCPTransportStdio {
		if cmd := s.Transport.Command; len(cmd) > 0 {
			return cmd[0]
		}
		return ""
	}
	raw := s.Transport.URL
	u, err := url.Parse(raw)
	if err != nil {
		// Unparseable: Validate rejects it, but explain reports on configs
		// Validate rejected. Showing nothing hides which endpoint was meant,
		// so the raw value is shown - it is Go-quoted on output like every
		// other config-sourced string.
		return raw
	}
	u.RawQuery, u.Fragment = "", ""
	// Userinfo is rejected by Validate, but explain reports on configs
	// Validate rejected - and a credential is a credential wherever it sits.
	u.User = nil
	return u.String()
}

// existingToolNames snapshots the tool names already registered (builtins and
// subprocess tools), as the set an MCP server's discovered names are made
// unique against.
func existingToolNames(reg *tools.Registry) map[string]bool {
	names := reg.Names()
	existing := make(map[string]bool, len(names))
	for _, name := range names {
		existing[name] = true
	}
	return existing
}

// newTokenStore builds the OAuth credential store over the XDG state
// directory. cmd is the composition root, so this is the ONE place that knows
// where credentials live; internal/mcp is handed a store or nothing.
func newTokenStore(env config.LookupEnv) *oauthtoken.Store {
	return oauthtoken.NewStore(oauthtoken.DefaultDir(env), time.Now)
}

// loginDeps composes the dependencies an interactive login runs with. It is
// shared by `amele mcp login` and the pre-connect phase of `run`/`chat` so the
// two flows cannot drift into logging in under different clocks, different
// randomness or - worst - a different secret registry.
func loginDeps(env config.LookupEnv, store *oauthtoken.Store, secrets *session.SecretSet) mcp.Deps {
	return mcp.Deps{
		// cmd is the composition root: the one place allowed to name the real
		// clock and the real random source (docs/engineering.md §5.4).
		Clock:          time.Now,
		Rand:           rand.Float64,
		Env:            env,
		TokenStore:     store,
		RegisterSecret: secrets.Add,
	}
}

// hasOAuthServer reports whether this config declares any server the OAuth
// phase would have to consider. It exists so a config without one pays
// nothing: no store is opened and no state directory is touched.
func hasOAuthServer(cfg *config.Config) bool {
	for _, s := range cfg.MCP.Servers {
		if s.Auth != nil && s.Auth.Type == config.MCPAuthOAuth {
			return true
		}
	}
	return false
}

// mcpCredentialGate is the pre-connect OAuth phase of `run` and `chat`.
//
// CONTRACT (docs/contracts/cli.md, spec §3.1): the phase sits between the run
// lock and the `limits.timeout` deadline - a human walking to a browser must
// not spend the run's budget - and it precedes run_start, so the credential
// question is settled before a turn is ever attempted.
//
// A refusal is still audited: the caller (reportGateFailure, and the mirror of
// it in `chat`) writes run_start plus run_end with exit 8 and mcp_errors 1, so
// a run that never obtained a credential leaves exactly the evidence a failed
// connect would have left.
//
// It opens the store only when a server actually declares oauth.
func mcpCredentialGate(ctx context.Context, cfg *config.Config, cfgPath string, lines *lineReader,
	stderr io.Writer, env config.LookupEnv, secrets *session.SecretSet, quiet bool) error {
	if !hasOAuthServer(cfg) {
		return nil
	}
	return ensureMCPCredentials(ctx, cfg, cfgPath, lines, stderr, env, newTokenStore(env), secrets, quiet)
}

// ensureMCPCredentials runs BEFORE the parallel connect: for every OAuth
// server without a usable credential it either logs in interactively (real
// TTY + the operator said yes) or applies the required/optional policy.
//
// It runs SEQUENTIALLY on purpose - three servers must not open three browser
// windows at once, and the operator must be able to answer one question at a
// time on the terminal the run shares with the permission prompt.
//
// Returns an error wrapping mcp.ErrUnavailable (exit 8) for a REQUIRED server
// the operator declined, or that has no terminal to ask on; the text names the
// runnable `amele mcp login <config> <server>`. An optional server only gets a
// warning here and is NOT counted: the count belongs to connectFailed, which
// is where the connection actually fails, and counting it twice would report
// two losses for one missing credential.
//
// A credential is "usable" when it is unexpired OR carries a refresh token: a
// stale-but-refreshable record is refreshed silently at connect time, and
// stopping to ask for a browser because an access token aged out would make
// every long-lived cron agent interactive.
func ensureMCPCredentials(ctx context.Context, cfg *config.Config, cfgPath string, lines *lineReader,
	stderr io.Writer, env config.LookupEnv, store *oauthtoken.Store, secrets *session.SecretSet, quiet bool) error {
	// SECURITY: the login flow mints tokens and quotes what it wrote, so every
	// line internal/mcp emits here goes through the run's live redactor - the
	// same one the session log and the progress feed use.
	out := &redactingWriter{w: stderr, redact: secrets.Redact}
	deps := loginDeps(env, store, secrets)
	for _, s := range cfg.MCP.Servers {
		if s.Auth == nil || s.Auth.Type != config.MCPAuthOAuth {
			continue
		}
		if hasUsableCredential(store, s) {
			continue
		}
		if err := ctx.Err(); err != nil {
			// A SIGTERM between two servers is an interrupted run (exit 1),
			// not a missing dependency: nobody may be paged about a server
			// that was never asked.
			return interruptedError(err)
		}
		// The question is asked on a REAL terminal only. The permission
		// system's "EOF is a refusal" tolerance is deliberately not reused:
		// a cron job with /dev/null on stdin must hear that a login needs a
		// human rather than wait for a browser nobody will see.
		if lines.IsTerminal() && askMCPLogin(s, lines, stderr) {
			// Confirm asks a SECOND time, with the discovered issuer: the
			// question above can only name the resource (the issuer is
			// unknown until discovery), and consenting to a browser is not
			// the same as consenting to hand this identity to that issuer.
			rec, err := loginToServer(ctx, s, deps, mcp.LoginOptions{
				Stderr:  out,
				Confirm: mcpConfirm(s.Name, lines, stderr),
			})
			if err == nil {
				_, _ = fmt.Fprintf(stderr, "mcp login ok: %s (expires %s)\n", s.Name, rec.ExpiresAt.UTC().Format(mcpTimeFormat))
				continue
			}
			if ctx.Err() != nil {
				return interruptedError(ctx.Err())
			}
			// Reported, then judged by the same policy as "no credential at
			// all": a login that did not complete leaves the run exactly as
			// unequipped as one that was never attempted.
			_, _ = fmt.Fprintf(stderr, "amele: %s\n", secrets.Redact(err.Error()))
		}
		if s.IsRequired() {
			return fmt.Errorf("mcp server %q: no oauth credential: %w (run 'amele mcp login %s %s' first)",
				s.Name, mcp.ErrUnavailable, cfgPath, s.Name)
		}
		if !quiet {
			_, _ = fmt.Fprintf(stderr, "amele: warning: mcp server %q has no oauth credential; run 'amele mcp login %s %s' (continuing without it)\n",
				s.Name, cfgPath, s.Name)
		}
	}
	return nil
}

// hasUsableCredential reports whether the store already holds something this
// run can connect with. Unreadable files in the token directory are ignored
// here: they are reported by the connect that follows, and a stray file must
// not turn a healthy credential into a browser prompt.
func hasUsableCredential(store *oauthtoken.Store, s config.MCPServer) bool {
	matches, _ := serverCredentials(store, s)
	now := store.Now()
	for _, m := range matches {
		// No refresh margin: the margin decides WHEN a run refreshes, not
		// whether the credential is worth having. A record that is merely
		// close to expiry is refreshed silently at connect time.
		if m.Record.Fresh(now, 0) || m.Record.RefreshToken != "" {
			return true
		}
	}
	return false
}

// askMCPLogin is the pre-discovery question. It names the resource, which is
// all the config knows - the issuer only becomes visible after discovery, and
// mcpConfirm shows it before the browser opens.
//
// SECURITY: only an explicit y/yes proceeds; a blank line, an unrecognized
// word or EOF declines, the same rule the tool approval prompt follows.
func askMCPLogin(s config.MCPServer, lines *lineReader, stderr io.Writer) bool {
	_, _ = fmt.Fprintf(stderr, "amele: mcp server %q needs a login (issuer unknown until discovery; resource %s). Open a browser now? [y/N] ",
		safeForTerminal(s.Name, maxToolName), safeForTerminal(s.Transport.URL, maxMCPReportField))
	return readConfirmation(lines, stderr)
}
