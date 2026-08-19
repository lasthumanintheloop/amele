package mcp

import "time"

// ErrorClass names WHY a connection to an MCP server failed, in the coarse
// terms an operator can act on: a command that will not start is a different
// problem from a token the server rejected.
//
// It is a small closed vocabulary on purpose: it is written into progress
// output and (through cmd) into the session log, so its values are read by
// humans and by scripts. classify assigns it; nothing else invents a value.
type ErrorClass string

// The error classes. Every connect failure gets exactly one; ClassProtocol is
// the catch-all, so an unrecognised failure is never silently dropped.
const (
	// ClassSpawn means the stdio server could not be started at all (missing
	// binary, not executable, bad working directory).
	ClassSpawn ErrorClass = "spawn"
	// ClassNetwork means the endpoint could not be reached or the connection
	// broke below the protocol (refused, reset, DNS, TLS).
	ClassNetwork ErrorClass = "network"
	// ClassAuth means the server answered and refused the credentials (401,
	// 403).
	ClassAuth ErrorClass = "auth"
	// ClassProtocol means the peer spoke, but not something amele could use:
	// a malformed message, an unsupported initialize, a duplicate tool name, a
	// message past the size cap.
	ClassProtocol ErrorClass = "protocol"
	// ClassTimeout means the attempt ran out of time (ConnectTimeout or the
	// run's own deadline).
	ClassTimeout ErrorClass = "timeout"
)

// ConnectInfo describes one established session. It is what an operator sees
// after a successful connect and what `explain` reprints; every field is either
// amele's own or a value the server reported at initialize.
type ConnectInfo struct {
	// Server is the name from the config, the prefix of every tool name.
	Server string
	// Transport is "stdio" or "http".
	Transport string
	// Duration is how long the connect took, including retries.
	Duration time.Duration
	// ProtocolVersion is the MCP version the two sides agreed on.
	ProtocolVersion string
	// ServerName and ServerVersion are what the server called itself. They are
	// UNTRUSTED display strings: amele never dispatches on them.
	ServerName, ServerVersion string
	// SessionFP is a short fingerprint of the session id (the first 8 hex
	// characters of its sha256), or "" when the transport has no session id.
	// SECURITY: the raw id is a bearer-ish value for Streamable HTTP and is
	// never logged; the fingerprint is enough to correlate two log lines.
	SessionFP string
	// ToolCount is how many tools the server contributed after filtering.
	ToolCount int
	// Auth is the credential mechanism the session authenticated with
	// ("oauth"), or "" when the server needed none - a static header counts as
	// none, since nothing about it is amele's to manage.
	//
	// SECURITY: the mechanism only. The token, the issuer and the expiry are
	// deliberately absent: this value is copied into the session log, and a
	// connect record must never be a place a credential can leak from.
	Auth string
}

// ListedTool is one tool amele accepted from a server, in the form an operator
// needs to audit it: which name the model sees, which name goes on the wire,
// and a hash of the definition so a changed tool is visible between runs.
type ListedTool struct {
	// Name is the model-facing name ("<server>__<tool>").
	Name string
	// Original is the server-side name sent back in tools/call.
	Original string
	// Normalized reports that Name is a rewrite, not a plain join.
	Normalized bool
	// SHA256 is the hex sha256 of the definition as published:
	// original name + "\n" + description + "\n" + compact input schema.
	SHA256 string
	// Bytes is the size of that same definition, the value the discovery caps
	// are charged against.
	Bytes int
	// Annotations are the server's hints, copied verbatim; nil fields mean the
	// server said nothing. SECURITY: they are advisory only - a permission
	// ruling never depends on them (docs/threat-model.md S9).
	Annotations Annotations
}

// Annotations are the MCP tool hints amele keeps. Each field is a pointer so
// that "the server did not say" (nil) stays distinguishable from an explicit
// false, which is the whole point of a hint an operator may act on.
type Annotations struct {
	// ReadOnly says the tool does not modify its environment.
	ReadOnly *bool
	// Destructive says the tool may perform destructive updates.
	Destructive *bool
	// OpenWorld says the tool talks to an open world of external entities.
	OpenWorld *bool
	// Idempotent says repeating the call with the same arguments is a no-op.
	Idempotent *bool
}

// SkippedTool is one tool a server published that amele did not expose, with
// the short reason why. Reasons are a closed set: "not included", "excluded",
// "definition too large", "invalid output schema".
//
// A skip is never fatal - a filter or a broken schema on one tool must not
// cost the run the other twenty - so the reason has to be visible instead.
type SkippedTool struct {
	// Name is the server-side name.
	Name string
	// Reason is the loggable phrase.
	Reason string
}

// Observer receives the connection lifecycle of one run's MCP servers. It is
// how internal/mcp stays free of any logging or progress dependency: cmd
// implements it over the session log and the -v progress line.
//
// Implementations must be safe for concurrent use: a reconnect happens on the
// goroutine of whichever tool call noticed the loss.
type Observer interface {
	// Connected reports a session that is ready to serve calls.
	Connected(ConnectInfo)
	// ConnectFailed reports that all attempts to reach a server failed.
	ConnectFailed(server, transport string, class ErrorClass, err error, d time.Duration)
	// ToolsListed reports the outcome of discovery, once per successful
	// Connect. totalBytes is the size of every definition the server sent,
	// including the skipped ones.
	ToolsListed(server string, tools []ListedTool, totalBytes int, skipped []SkippedTool)
	// Disconnected reports a session going away. reason is "run_end" for the
	// orderly shutdown and "reconnect" when a lost session is being replaced.
	Disconnected(server, reason string)
}

// NopObserver is the Observer used when a caller supplies none. It exists so
// the Server code never has to check for nil.
type NopObserver struct{}

// Connected implements Observer.
func (NopObserver) Connected(ConnectInfo) {}

// ConnectFailed implements Observer.
func (NopObserver) ConnectFailed(string, string, ErrorClass, error, time.Duration) {}

// ToolsListed implements Observer.
func (NopObserver) ToolsListed(string, []ListedTool, int, []SkippedTool) {}

// Disconnected implements Observer.
func (NopObserver) Disconnected(string, string) {}
