package mcp

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"iter"
	"net"
	"net/url"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/llm"
	"github.com/lasthumanintheloop/amele/internal/oauthtoken"
	"github.com/lasthumanintheloop/amele/internal/schema"
	"github.com/lasthumanintheloop/amele/internal/tools"
	"github.com/modelcontextprotocol/go-sdk/auth"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Budgets and timings of one server connection. They are constants rather than
// config keys on purpose: they bound what an unreviewed peer can cost, and an
// operator who could raise them would be removing his own safety net.
const (
	// ConnectTimeout bounds the WHOLE of Connect - every attempt, the jittered
	// pause between them, initialize and discovery together share this one
	// window (docs/mcp.md promises "30 s, covering at most two jittered
	// attempts").
	ConnectTimeout = 30 * time.Second
	// connectAttempts is how many times Connect tries before giving up. Two:
	// enough to ride out a server that was still starting, few enough that a
	// broken config fails a cron run fast.
	connectAttempts = 2
	// MaxToolsPerServer caps how many tools one server may publish. Past this
	// the connection is a protocol error, not a truncation: a toolset that
	// large is a bug or an attack, and silently exposing half of it would make
	// the model's view of the run depend on iteration order.
	MaxToolsPerServer = 512
	// MaxDiscoveryBytes caps the total size of all definitions one server may
	// send. It is charged before any tool is exposed.
	MaxDiscoveryBytes = 4 << 20
	// MaxDefinitionBytes caps ONE tool definition. Over this the tool is
	// skipped rather than the server rejected: one bloated tool must not cost
	// the run its other tools.
	MaxDefinitionBytes = 32 << 10
	// retryGap is the base pause between connect attempts; the jitter drawn
	// from Deps.Rand adds up to the same amount again.
	retryGap = time.Second
	// reconnectJitter bounds the pause before a mid-run reconnect.
	reconnectJitter = time.Second
	// closeGrace bounds the orderly session close at shutdown.
	closeGrace = 2 * time.Second
	// clientName is what amele calls itself at initialize.
	clientName = "amele"
	// defaultVersion is the version reported when the caller supplies none.
	defaultVersion = "dev"
	// fpLen is how many hex characters of the session-id hash are kept.
	fpLen = 8
)

// Sentinels for the two ways an MCP server can fail a run. They are matched
// with errors.Is by cmd, which maps them onto the frozen exit codes.
var (
	// ErrUnavailable means the server could not be reached. CONTRACT: cmd maps
	// it to exit 8 when the server is required.
	ErrUnavailable = errors.New("mcp server unavailable")
	// ErrToolset means the server answered but its toolset cannot be used as
	// declared (a name collision with an existing tool). CONTRACT: cmd maps it
	// to exit 2 - it is a configuration error, and retrying cannot fix it.
	ErrToolset = errors.New("mcp toolset invalid")
	// errClosed is returned by a call that arrives after Close.
	errClosed = errors.New("server closed")
)

// ConnectError is a failure to establish a session, carrying the class an
// operator needs to act on.
//
// It unwraps to BOTH its cause and ErrUnavailable, so a caller can match the
// sentinel for the exit code while a log line still shows the real error.
type ConnectError struct {
	// Server is the configured server name.
	Server string
	// Class is the coarse reason (see ErrorClass).
	Class ErrorClass
	// Err is the underlying failure.
	Err error
}

// Error implements error.
func (e *ConnectError) Error() string {
	return fmt.Sprintf("mcp server %q unavailable (%s): %v", e.Server, e.Class, e.Err)
}

// Unwrap returns both the cause and ErrUnavailable (Go 1.20 multi-unwrap).
func (e *ConnectError) Unwrap() []error { return []error{e.Err, ErrUnavailable} }

// Deps are the injected dependencies of one server connection. Nothing in this
// package reads the clock, the environment or a random source directly
// (docs/engineering.md §5.4), so every one of them arrives here.
type Deps struct {
	// Clock supplies the current time; used only to measure durations.
	// REQUIRED.
	Clock func() time.Time
	// Rand returns a value in [0,1) used as connect/reconnect jitter.
	// REQUIRED.
	Rand func() float64
	// Env looks up an environment variable for a stdio server's allowlist.
	// REQUIRED.
	Env func(string) (string, bool)
	// Observer receives the connection lifecycle; nil means NopObserver.
	Observer Observer
	// Stderr is where a stdio server's stderr goes, unwrapped and uncapped.
	// The caller is responsible for redaction, for BOUNDING what a chatty
	// server may write, and for making it safe for concurrent use; nil means
	// io.Discard.
	Stderr io.Writer
	// Workspace is the working directory of a stdio server.
	Workspace string
	// ExistingNames are the model-facing tool names already taken by builtins,
	// subprocess tools and earlier servers. A collision is fatal (ErrToolset).
	ExistingNames map[string]bool
	// Version is amele's version, reported to the server at initialize. Empty
	// means "dev".
	Version string
	// TokenStore holds the OAuth credentials. REQUIRED when any server
	// declares `auth`; ignored otherwise, so a config without OAuth needs no
	// state directory at all.
	TokenStore *oauthtoken.Store
	// RegisterSecret publishes a value the run's redactor must scrub. It is
	// how a token refreshed MID-RUN still gets kept out of the session log,
	// which a redactor frozen at startup could not do. nil is allowed.
	RegisterSecret func(...string)
	// authFailed is set by Connect, not by the caller: it lets the OAuth
	// handler tell its Server that this run's authorization is dead. It is
	// unexported precisely because it is not a dependency cmd may choose.
	authFailed func(error)
}

// withDefaults fills the optional fields and rejects a Deps missing a required
// one. The required ones are reported as an error rather than defaulted: a
// library that silently substitutes time.Now defeats the determinism rule, and
// a library that panics takes the run down for a programmer's typo.
func (d Deps) withDefaults() (Deps, error) {
	if d.Clock == nil || d.Rand == nil || d.Env == nil {
		return d, errors.New("mcp: Deps.Clock, Deps.Rand and Deps.Env are required")
	}
	if d.Observer == nil {
		d.Observer = NopObserver{}
	}
	if d.Stderr == nil {
		d.Stderr = io.Discard
	}
	if d.Version == "" {
		d.Version = defaultVersion
	}
	return d, nil
}

// transportFactory builds the transport for one connection attempt. It returns
// the transport and a kill function that releases whatever the transport owns
// (a child process for stdio, idle connections for HTTP).
type transportFactory func(ctx context.Context, cfg config.MCPServer, deps Deps) (sdk.Transport, func(), error)

// newTransport is the seam the tests replace with an in-memory pair. It is a
// package variable rather than a Deps field because it is not a dependency an
// operator or cmd may choose: production has exactly one implementation.
var newTransport transportFactory = defaultTransport

// connectWindow is ConnectTimeout behind a variable, so a test of the window
// semantics does not have to wait thirty real seconds. Production never
// changes it.
var connectWindow = ConnectTimeout

// reconnectWaiting, when non-nil, is called by session() just before a caller
// starts waiting for another caller's reconnect. It exists ONLY so the
// singleflight test can know deterministically that the second caller has
// queued (a sleep there would be a scheduler bet); production never sets it.
var reconnectWaiting func()

// defaultTransport builds the real transport for cfg.
//
// ctx bounds the LIFETIME of what the transport owns (the child process), not
// the connect attempt: a stdio server started with the attempt's 30 s context
// would be killed 30 s into the run.
func defaultTransport(ctx context.Context, cfg config.MCPServer, deps Deps) (sdk.Transport, func(), error) {
	switch cfg.Transport.Type {
	case config.MCPTransportStdio:
		return newStdioTransport(ctx, cfg.Transport.Command, deps.Workspace, cfg.Transport.Env, deps.Env, deps.Stderr)
	case config.MCPTransportHTTP:
		// A typed nil would satisfy the interface and make the SDK ask a nil
		// handler for tokens, so the handler is only ever assigned when there
		// is a real one.
		var handler auth.OAuthHandler
		if cfg.Auth != nil {
			h, err := newRunOAuthHandler(cfg, deps)
			if err != nil {
				return nil, nil, fmt.Errorf("mcp server %q: %w", cfg.Name, err)
			}
			h.onDead = deps.authFailed
			handler = h
		}
		return newHTTPTransport(cfg.Transport.URL, cfg.Transport.Headers, handler)
	default:
		// Unreachable through a validated config; a library must still not
		// dereference its way into a panic.
		return nil, nil, fmt.Errorf("unknown transport type %q", cfg.Transport.Type)
	}
}

// Server is one live MCP connection and the frozen set of tools it contributed.
//
// CONTRACT: the toolset is discovered exactly once, at Connect. tools/list is
// never called again - not on a list_changed notification (amele does not
// subscribe) and not after a reconnect - so the names the model was given at
// turn 1 still mean the same thing at turn 20.
//
// A Server is safe for concurrent use: the loop may run several tool calls at
// once, and whichever of them notices a dead session performs the single
// reconnect the others wait for.
type Server struct {
	cfg  config.MCPServer
	deps Deps
	// runCtx bounds the lifetime of a spawned server. Storing a context in a
	// struct is normally wrong; here it is the point: a reconnect happens on a
	// tool call's context, and a child process started with THAT context would
	// die when the call returns.
	runCtx context.Context //nolint:containedctx // process lifetime, not request scope

	// toolset, frozen at Connect and read without the lock afterwards.
	tools      []tools.Tool
	listed     []ListedTool
	skipped    []SkippedTool
	totalBytes int

	mu           sync.Mutex
	sess         *sdk.ClientSession
	kill         func()
	dead         bool
	closed       bool
	reconnecting chan struct{}
	reconnectErr error
	errCount     int
	authDead     error
	info         ConnectInfo
}

// Connect establishes one MCP session and discovers its tools.
//
// It makes at most connectAttempts attempts, with a jittered pause in between;
// attempts and pause together share ONE ConnectTimeout window, so a required
// server that is down delays a run by at most ~30 s, not by attempts × 30 s.
// On success the Observer has seen Connected
// and ToolsListed; on failure it has seen ConnectFailed and the returned error
// is a *ConnectError wrapping ErrUnavailable - except for a toolset error
// (a name collision), which is returned bare because retrying cannot help.
//
// ctx bounds the whole call AND the lifetime of a stdio server's process: the
// caller must pass the run's context, not a per-call one.
func Connect(ctx context.Context, cfg config.MCPServer, deps Deps) (*Server, error) {
	deps, err := deps.withDefaults()
	if err != nil {
		return nil, err
	}
	// Checked here rather than defaulted: a server that declares OAuth in a
	// process with no token store is a wiring mistake, and inventing a store
	// (or a directory) on its behalf would write credentials somewhere nobody
	// chose. Like the required Deps above it is an error, never a panic.
	if cfg.Auth != nil && deps.TokenStore == nil {
		return nil, fmt.Errorf("mcp server %q: %w", cfg.Name, errNoTokenStore)
	}
	start := deps.Clock()
	s := &Server{cfg: cfg, deps: deps, runCtx: ctx}

	// One window over everything: attempts must not each get a fresh 30 s, or
	// a hanging server would cost a run twice the documented bound.
	wctx, cancel := context.WithTimeout(ctx, connectWindow)
	defer cancel()

	var lastErr error
	for attempt := range connectAttempts {
		if attempt > 0 {
			// Jittered pause so a fleet restarting on the same cron tick does
			// not hammer one server in lockstep.
			if err := sleepCtx(wctx, retryGap+time.Duration(deps.Rand()*float64(retryGap))); err != nil {
				// The window (or the run) ended during the pause. The failure
				// worth reporting is the attempt that drove us here, when
				// there was one.
				if lastErr == nil {
					lastErr = err
				}
				break
			}
		}
		err := s.connectOnce(ctx, wctx, true)
		if err == nil {
			deps.Observer.Connected(s.setDuration(deps.Clock().Sub(start)))
			deps.Observer.ToolsListed(cfg.Name, s.Listed(), s.totalBytes, s.Skipped())
			return s, nil
		}
		lastErr = err
		if errors.Is(err, ErrToolset) {
			break // static problem: another attempt would fail identically
		}
	}

	class := classify(lastErr)
	deps.Observer.ConnectFailed(cfg.Name, cfg.Transport.Type, class, lastErr, deps.Clock().Sub(start))
	if errors.Is(lastErr, ErrToolset) {
		return nil, lastErr
	}
	return nil, &ConnectError{Server: cfg.Name, Class: class, Err: lastErr}
}

// connectOnce performs one attempt: build the transport, initialize, and (only
// at Connect, discoverTools) discover the toolset. procCtx bounds the
// transport's lifetime, ctx bounds this attempt.
//
// On any failure it releases everything it created, so a failed attempt leaves
// no process and no goroutine behind.
func (s *Server) connectOnce(procCtx, ctx context.Context, discoverTools bool) error {
	// The transport's OAuth handler must be able to condemn THIS server when
	// the authorization server refuses the credential, so the callback is
	// attached to the copy of Deps the factory sees.
	deps := s.deps
	deps.authFailed = s.markAuthDead
	transport, kill, err := newTransport(procCtx, s.cfg, deps)
	if err != nil {
		return fmt.Errorf("mcp server %q: %w", s.cfg.Name, err)
	}
	if kill == nil {
		kill = func() {}
	}
	client := sdk.NewClient(&sdk.Implementation{Name: clientName, Version: s.deps.Version}, clientOptions())
	sess, err := client.Connect(ctx, transport, nil)
	if err != nil {
		kill()
		return fmt.Errorf("mcp server %q: initialize: %w", s.cfg.Name, err)
	}
	if discoverTools {
		if err := s.discover(sess.Tools(ctx, nil)); err != nil {
			closeSession(ctx, sess)
			kill()
			return err
		}
	}
	if !s.adopt(sess, kill) {
		// Close won the race: release what this attempt built. WithoutCancel so
		// the orderly close still gets its grace period even when the caller's
		// context is already done.
		closeSession(context.WithoutCancel(ctx), sess)
		kill()
		return errClosed
	}
	return nil
}

// clientOptions are the client capabilities amele advertises: none.
//
// An empty ClientCapabilities disables the roots advertisement, and the
// multi-round-trip middleware is disabled so a server asking for interactive
// input gets a tool_error instead of a client that tries to answer for a human
// who is not there (amele is headless).
func clientOptions() *sdk.ClientOptions {
	return &sdk.ClientOptions{
		Capabilities:   &sdk.ClientCapabilities{},
		MultiRoundTrip: &sdk.MultiRoundTripOptions{Disabled: true},
	}
}

// adopt installs a fresh session together with its transport-release function
// and records what initialize reported. Both are published under one lock, so
// no caller can ever see a new session paired with the previous kill.
//
// It reports false when the server was closed while this connection was being
// built - a reconnect racing shutdown. The caller must then tear the new
// session down: installing it would leave a session nobody ever closes and, for
// stdio, a child process alive until the run context is cancelled.
func (s *Server) adopt(sess *sdk.ClientSession, kill func()) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return false
	}
	s.sess = sess
	s.kill = kill
	s.dead = false
	s.info = ConnectInfo{
		Server:    s.cfg.Name,
		Transport: s.cfg.Transport.Type,
		SessionFP: fingerprint(sess.ID()),
		ToolCount: len(s.tools),
	}
	if init := sess.InitializeResult(); init != nil {
		s.info.ProtocolVersion = init.ProtocolVersion
		if init.ServerInfo != nil {
			s.info.ServerName = init.ServerInfo.Name
			s.info.ServerVersion = init.ServerInfo.Version
		}
		// SECURITY: init.Instructions is untrusted prompt content written by
		// the server operator. It is deliberately never forwarded to the model
		// and never logged (docs/threat-model.md S9).
	}
	return true
}

// setDuration records how long the connect took and returns the finished
// ConnectInfo.
func (s *Server) setDuration(d time.Duration) ConnectInfo {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.info.Duration = d
	return s.info
}

// discoverState carries the running totals of one discovery pass.
type discoverState struct {
	seen       map[string]bool
	effective  map[string]bool
	count      int
	totalBytes int
}

// discover consumes the tools/list iterator (the SDK follows cursors) and
// builds the frozen toolset.
//
// Every definition is charged against the size caps BEFORE it becomes a tool.
// A tool the operator filtered out, one whose definition is too large and one
// whose output schema will not compile are skipped with a reason; a duplicate
// name, too many tools and too many bytes are protocol errors, because they
// make the whole listing untrustworthy rather than one entry unusable.
func (s *Server) discover(seq iter.Seq2[*sdk.Tool, error]) error {
	// Start from empty on every pass. Connect retries after a failure, and a
	// listing that died halfway through the pages has already appended part of
	// a toolset; without this reset the second attempt would hand the model
	// every surviving tool twice (the per-pass seen/effective maps cannot catch
	// it - they are fresh too).
	s.tools, s.listed, s.skipped, s.totalBytes = nil, nil, nil, 0
	st := &discoverState{seen: map[string]bool{}, effective: map[string]bool{}}
	for t, err := range seq {
		if err != nil {
			return fmt.Errorf("mcp server %q: listing tools: %w", s.cfg.Name, err)
		}
		if t == nil {
			continue
		}
		if err := s.acceptTool(t, st); err != nil {
			return err
		}
	}
	s.totalBytes = st.totalBytes
	return nil
}

// acceptTool applies the caps, the filter and the naming rules to one tool. It
// returns an error only for a failure that condemns the whole server.
func (s *Server) acceptTool(t *sdk.Tool, st *discoverState) error {
	if st.seen[t.Name] {
		return fmt.Errorf("mcp server %q: duplicate tool name %q", s.cfg.Name, t.Name)
	}
	st.seen[t.Name] = true
	st.count++
	if st.count > MaxToolsPerServer {
		return fmt.Errorf("mcp server %q: too many tools (cap %d)", s.cfg.Name, MaxToolsPerServer)
	}

	inputSchema, err := marshalSchema(t.InputSchema)
	if err != nil {
		return fmt.Errorf("mcp server %q: tool %q: input schema: %w", s.cfg.Name, t.Name, err)
	}
	// The output schema is marshalled here, before the size accounting, so its
	// bytes count toward the discovery caps like every other part of the
	// definition - a server must not get a free 4 MiB channel just because the
	// bytes arrived under a different key. A value that will not even marshal
	// has no measurable size; it costs the tool, not the budget.
	var outputRaw []byte
	if t.OutputSchema != nil {
		if outputRaw, err = json.Marshal(t.OutputSchema); err != nil {
			s.skip(t.Name, "invalid output schema")
			return nil
		}
	}
	size := len(t.Name) + len(t.Description) + len(inputSchema) + len(outputRaw)
	st.totalBytes += size
	if st.totalBytes > MaxDiscoveryBytes {
		return fmt.Errorf("mcp server %q: tool definitions exceed %d bytes", s.cfg.Name, MaxDiscoveryBytes)
	}
	if size > MaxDefinitionBytes {
		s.skip(t.Name, "definition too large")
		return nil
	}
	if keep, reason := Keep(t.Name, s.cfg.Tools.Include, s.cfg.Tools.Exclude); !keep {
		s.skip(t.Name, reason)
		return nil
	}
	// Providers require a tool's parameters to be an OBJECT schema; a boolean
	// or array-rooted schema is valid JSON Schema but would be rejected (or
	// silently mangled) downstream, so the tool is skipped here with a reason
	// the operator can see instead of failing at the first provider call.
	if !isObjectSchema(inputSchema) {
		s.skip(t.Name, "input schema not an object")
		return nil
	}

	named := EffectiveName(s.cfg.Name, t.Name)
	if s.deps.ExistingNames[named.Effective] || st.effective[named.Effective] {
		return fmt.Errorf("%w: mcp server %q: tool name %q collides with an existing tool", ErrToolset, s.cfg.Name, named.Effective)
	}
	st.effective[named.Effective] = true

	validator, err := outputValidator(outputRaw)
	if err != nil {
		s.skip(t.Name, "invalid output schema")
		return nil
	}

	annotations := copyAnnotations(t.Annotations)
	s.tools = append(s.tools, &Tool{
		srv:      s,
		original: t.Name,
		def: llm.ToolDef{
			Name:        named.Effective,
			Description: t.Description,
			Parameters:  inputSchema,
		},
		outputSchema: validator,
		annotations:  annotations,
	})
	s.listed = append(s.listed, ListedTool{
		Name:        named.Effective,
		Original:    t.Name,
		Normalized:  named.Normalized,
		SHA256:      definitionHash(t.Name, t.Description, inputSchema),
		Bytes:       size,
		Annotations: annotations,
	})
	return nil
}

// skip records one tool amele chose not to expose.
func (s *Server) skip(name, reason string) {
	s.skipped = append(s.skipped, SkippedTool{Name: name, Reason: reason})
}

// marshalSchema renders a schema the server sent as compact JSON. A tool
// without an input schema gets the empty object schema: a provider rejects a
// tool definition with no parameters object.
func marshalSchema(v any) (json.RawMessage, error) {
	if v == nil {
		return json.RawMessage(`{"type":"object"}`), nil
	}
	encoded, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	if string(encoded) == "null" {
		return json.RawMessage(`{"type":"object"}`), nil
	}
	return encoded, nil
}

// outputValidator compiles a tool's already-marshalled output schema once, at
// discovery, so that RenderResult can check every result without paying for a
// compile per call. A schema that will not compile is the tool's problem, not
// the server's: the caller skips that one tool.
func outputValidator(encoded []byte) (*schema.Validator, error) {
	if encoded == nil {
		return nil, nil //nolint:nilnil // "no schema" is a legitimate, non-error result
	}
	return schema.Compile(encoded)
}

// isObjectSchema reports whether a marshalled JSON Schema describes an object -
// the only shape a provider accepts as a tool's parameters. A boolean schema
// (`true`) and a root that declares a non-object type both fail; a root with
// no `type` passes, because "properties without type" conventionally means an
// object and rejecting it would drop real-world tools.
func isObjectSchema(encoded json.RawMessage) bool {
	var root map[string]json.RawMessage
	if json.Unmarshal(encoded, &root) != nil {
		return false
	}
	if t, ok := root["type"]; ok {
		var name string
		if json.Unmarshal(t, &name) == nil && name != "object" {
			return false
		}
	}
	return true
}

// copyAnnotations maps the SDK's annotation shape onto amele's.
//
// The SDK models readOnlyHint and idempotentHint as plain bools (their JSON
// default is false) and the other two as pointers. Copying the plain ones as
// always-set pointers keeps the distinction amele cares about honest for the
// fields where the wire format still has one, and never invents a hint for the
// fields where it does not.
func copyAnnotations(a *sdk.ToolAnnotations) Annotations {
	if a == nil {
		return Annotations{}
	}
	readOnly, idempotent := a.ReadOnlyHint, a.IdempotentHint
	return Annotations{
		ReadOnly:    &readOnly,
		Idempotent:  &idempotent,
		Destructive: a.DestructiveHint,
		OpenWorld:   a.OpenWorldHint,
	}
}

// definitionHash fingerprints one tool definition so an operator can see that a
// server changed a tool between runs.
func definitionHash(name, description string, inputSchema []byte) string {
	h := sha256.New()
	h.Write([]byte(name + "\n" + description + "\n"))
	h.Write(inputSchema)
	return hex.EncodeToString(h.Sum(nil))
}

// fingerprint shortens a session id to something loggable.
//
// SECURITY: the raw Mcp-Session-Id is a capability for the Streamable HTTP
// session and is never written anywhere; the hash prefix is enough to tell two
// sessions apart in a log.
func fingerprint(id string) string {
	if id == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(id))
	return hex.EncodeToString(sum[:])[:fpLen]
}

// Name returns the configured server name.
func (s *Server) Name() string { return s.cfg.Name }

// Info returns the current session's connection details.
func (s *Server) Info() ConnectInfo {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.info
}

// Tools returns the frozen toolset, in the order the server listed it. The
// slice is a copy; the tools themselves are shared and safe for concurrent use.
func (s *Server) Tools() []tools.Tool {
	return append([]tools.Tool(nil), s.tools...)
}

// Listed returns what discovery accepted, for `explain` and the session log.
func (s *Server) Listed() []ListedTool {
	return append([]ListedTool(nil), s.listed...)
}

// Skipped returns what discovery refused, with reasons.
func (s *Server) Skipped() []SkippedTool {
	return append([]SkippedTool(nil), s.skipped...)
}

// Errors reports how many transport-level failures this server has cost the
// run: lost responses plus failed reconnects. A tool that merely reported its
// own failure is not counted - that is the tool working.
func (s *Server) Errors() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.errCount
}

// countError increments the transport failure counter.
func (s *Server) countError() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.errCount++
}

// markAuthDead records that this run's authorization for the server is over:
// the authorization server refused the credential and no refresh can fix it
// before someone runs `amele mcp login`.
//
// CONTRACT (spec §5): the verdict is CACHED for the rest of the run - every
// further tool call fails without a round trip - and counted ONCE, so
// run_end.mcp_errors reports one dead server rather than one error per call.
func (s *Server) markAuthDead(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.authDead != nil {
		return
	}
	s.authDead = err
	s.errCount++
}

// authDeadErr returns the cached authorization verdict, or nil.
func (s *Server) authDeadErr() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.authDead
}

// markDead records that sess is unusable, so the next call reconnects. The
// pointer comparison matters: a call that lost its response on the OLD session
// must not condemn a session another goroutine has already replaced.
func (s *Server) markDead(sess *sdk.ClientSession) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.sess == sess {
		s.dead = true
	}
}

// session returns a live session, reconnecting once if the previous one died.
//
// Reconnection is single-flight: the first caller performs it while the others
// wait for its result, so a turn with eight parallel tool calls spawns one new
// server process, not eight.
func (s *Server) session(ctx context.Context) (*sdk.ClientSession, error) {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, errClosed
	}
	if !s.dead {
		sess := s.sess
		s.mu.Unlock()
		return sess, nil
	}
	if ch := s.reconnecting; ch != nil {
		s.mu.Unlock()
		if reconnectWaiting != nil {
			reconnectWaiting()
		}
		select {
		case <-ch:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		return s.sessionAfterReconnect()
	}
	ch := make(chan struct{})
	s.reconnecting = ch
	s.mu.Unlock()

	err := s.reconnect(ctx)

	s.mu.Lock()
	s.reconnecting = nil
	s.reconnectErr = err
	sess := s.sess
	s.mu.Unlock()
	// Closing after the state is published: a waiter must never observe the
	// old error with the new session, or the other way round.
	close(ch)

	if err != nil {
		return nil, fmt.Errorf("reconnect: %w", err)
	}
	return sess, nil
}

// sessionAfterReconnect reads the result of the attempt this caller waited for.
// It deliberately does not start an attempt of its own: one lost session buys
// one reconnect, whatever the number of callers.
func (s *Server) sessionAfterReconnect() (*sdk.ClientSession, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	switch {
	case s.reconnectErr != nil:
		return nil, fmt.Errorf("reconnect: %w", s.reconnectErr)
	case s.closed:
		return nil, errClosed
	case s.dead:
		return nil, fmt.Errorf("reconnect: %w", ErrUnavailable)
	default:
		return s.sess, nil
	}
}

// reconnect replaces a dead session with a fresh one.
//
// CONTRACT: it does NOT rediscover. The toolset stays exactly as it was at
// Connect - the existing Tool values are reused and only the session is
// swapped - so a server that came back with a different tool list cannot
// change what the model was told mid-run. A call to a tool the new server no
// longer has fails as an ordinary tool error.
func (s *Server) reconnect(ctx context.Context) error {
	s.deps.Observer.Disconnected(s.cfg.Name, "reconnect")
	s.releaseCurrent(ctx)

	// A short jittered pause: a server that just died is often being restarted
	// by something else, and a fleet must not all retry on the same tick.
	if err := sleepCtx(ctx, time.Duration(s.deps.Rand()*float64(reconnectJitter))); err != nil {
		// The CALLER's context ended (a SIGTERM, or the call's own budget):
		// that is the run stopping, not the server failing, and counting it
		// would inflate run_end.mcp_errors on every interrupted fleet.
		return err
	}

	start := s.deps.Clock()
	actx, cancel := context.WithTimeout(ctx, ConnectTimeout)
	defer cancel()
	if err := s.connectOnce(s.runCtx, actx, false); err != nil {
		// A reconnect that lost the race with Close, or one cut short by the
		// caller's own context ending, is the run shutting down - not a
		// failure of the server. Only a genuine failure is counted, and only
		// a genuine failure earns a mcp_connect{ok:false} event, so the log
		// shows WHY the tools kept failing after the disconnect.
		if errors.Is(err, errClosed) || ctx.Err() != nil {
			return err
		}
		s.countError()
		s.deps.Observer.ConnectFailed(s.cfg.Name, s.cfg.Transport.Type, classify(err), err, s.deps.Clock().Sub(start))
		return err
	}
	s.deps.Observer.Connected(s.setDuration(s.deps.Clock().Sub(start)))
	return nil
}

// releaseCurrent tears down the dead session and its transport.
func (s *Server) releaseCurrent(ctx context.Context) {
	s.mu.Lock()
	sess, kill := s.sess, s.kill
	s.sess, s.kill = nil, nil
	s.mu.Unlock()
	closeSession(ctx, sess)
	if kill != nil {
		kill()
	}
}

// Close ends the session and releases the transport. It is idempotent, and
// safe to call on a server whose session already died.
//
// The session close is bounded by closeGrace (or ctx, whichever is sooner)
// because the SDK's Close takes no context: it runs on its own goroutine while
// this one waits. That goroutine holds nothing but the session and ends when
// the close returns - which the transport kill below forces for stdio; on HTTP
// a stuck DELETE can outlive the wait, and the process is exiting anyway.
func (s *Server) Close(ctx context.Context) error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	sess, kill := s.sess, s.kill
	s.sess, s.kill = nil, nil
	s.mu.Unlock()

	closeSession(ctx, sess)
	if kill != nil {
		kill()
	}
	s.deps.Observer.Disconnected(s.cfg.Name, "run_end")
	return nil
}

// closeSession closes sess, waiting at most closeGrace.
func closeSession(ctx context.Context, sess *sdk.ClientSession) {
	if sess == nil {
		return
	}
	done := make(chan struct{})
	go func() {
		_ = sess.Close()
		close(done)
	}()
	timer := time.NewTimer(closeGrace)
	defer timer.Stop()
	select {
	case <-done:
	case <-timer.C:
	case <-ctx.Done():
	}
}

// sleepCtx waits for d unless ctx ends first, in which case it returns the
// context's error. A non-positive d does not wait at all.
func sleepCtx(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// classify names the coarse reason a connection failed, in a fixed order:
// deadline first (it can wrap anything), then the failures that identify
// themselves through the error chain, and only then the text-matched auth
// statuses - a typed network error whose message happens to contain "401"
// (an address, a byte count) must classify as the network failure it is.
// protocol is the catch-all.
func classify(err error) ErrorClass {
	if err == nil {
		return ""
	}
	if errors.Is(err, context.DeadlineExceeded) || isNetTimeout(err) {
		return ClassTimeout
	}
	if IsMessageTooLarge(err) {
		return ClassProtocol
	}
	if isSpawnError(err) {
		return ClassSpawn
	}
	if errors.Is(err, errTransientAuth) {
		// An authorization server that is down is an availability problem, not
		// a rejected credential: reporting it as `auth` would send an operator
		// hunting for a login he does not need.
		return ClassNetwork
	}
	if isNetworkError(err) {
		return ClassNetwork
	}
	if errors.Is(err, errAuthDenied) || isAuthError(err) {
		return ClassAuth
	}
	return ClassProtocol
}

// isNetTimeout reports an i/o timeout that does not travel as a context error.
func isNetTimeout(err error) bool {
	var ne net.Error
	return errors.As(err, &ne) && ne.Timeout()
}

// isSpawnError reports a child process that could not be started. fs.ErrNotExist
// and fs.ErrPermission cover the platform errno values through errors.Is, which
// keeps this free of build tags.
func isSpawnError(err error) bool {
	return errors.Is(err, exec.ErrNotFound) ||
		errors.Is(err, fs.ErrNotExist) ||
		errors.Is(err, fs.ErrPermission) ||
		errors.As(err, new(*exec.Error))
}

// authStatusRe matches a 401/403 that stands alone as a status code rather
// than as digits inside something longer (a port, a byte count, a request id).
var authStatusRe = regexp.MustCompile(`(?:^|[^0-9])(?:401|403)(?:[^0-9]|$)`)

// isAuthError reports credentials the server refused.
//
// It matches on TEXT because it has to: the SDK renders an HTTP failure status
// into the error message rather than a typed error, so there is nothing else to
// match on. The status names come first; the bare numbers are a fallback and
// must be delimited, so "port 4013" or "403 bytes read" in some other failure
// cannot turn it into an auth verdict (classify also asks the typed checks
// first for the same reason).
func isAuthError(err error) bool {
	msg := err.Error()
	return strings.Contains(msg, "Unauthorized") || strings.Contains(msg, "Forbidden") ||
		authStatusRe.MatchString(msg)
}

// isNetworkError reports a failure below the protocol.
func isNetworkError(err error) bool {
	var (
		ne net.Error
		oe *net.OpError
		ue *url.Error
	)
	return errors.As(err, &ne) || errors.As(err, &oe) || errors.As(err, &ue) ||
		errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) ||
		errors.Is(err, net.ErrClosed) || errors.Is(err, syscall.ECONNREFUSED) ||
		errors.Is(err, syscall.ECONNRESET) || errors.Is(err, syscall.EPIPE)
}

// isConnectionLoss reports an error that means the request may have reached the
// server but the answer did not: the transport died under the call.
//
// ErrSessionMissing (an HTTP 404 on a session the server forgot) belongs here:
// the next call must re-initialize rather than keep using a dead session id.
func isConnectionLoss(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, sdk.ErrConnectionClosed) || errors.Is(err, sdk.ErrSessionMissing) {
		return true
	}
	return isNetworkError(err) || IsMessageTooLarge(err)
}
