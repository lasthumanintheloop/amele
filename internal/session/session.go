// Package session records every run as a single append-only JSONL file.
//
// One format serves three purposes at once (docs/contracts/jsonl-events.md): it is the
// observability log ("what did the agent do at 03:00?"), the future resume
// source, and the future replay input. Events are therefore written in
// strict chronological order and never mutated.
package session

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// SchemaVersion identifies the JSONL event schema. CONTRACT: the event
// schema is public API (docs/engineering.md §7); bump this on breaking changes and
// document the migration in docs/contracts/.
const SchemaVersion = 1

// Clock supplies time. Injected so golden-file tests produce byte-identical
// logs (docs/engineering.md §5.4 determinism rule).
type Clock func() time.Time

// Event is one JSONL line. Fields are omitted when empty so each event type
// stays compact and greppable.
type Event struct {
	V    int       `json:"v"`
	Type string    `json:"type"`
	TS   time.Time `json:"ts"`

	// run_start
	Model string `json:"model,omitempty"`
	Task  string `json:"task,omitempty"`

	// llm_response. Content is the assistant's text (clipped); ToolCallIDs
	// are the IDs of the tool calls requested in the same message. Together
	// with the tool_call/tool_result events these make the log a complete
	// conversation record - the replay/resume source the package promises.
	Turn         int      `json:"turn,omitempty"`
	Content      string   `json:"content,omitempty"`
	ToolCallIDs  []string `json:"tool_call_ids,omitempty"`
	InputTokens  int      `json:"input_tokens,omitempty"`
	OutputTokens int      `json:"output_tokens,omitempty"`
	FinishReason string   `json:"finish_reason,omitempty"`
	// ReasoningBytes is the size of the provider's reasoning payload for this
	// turn (v1.4, additive; absent means none).
	//
	// SECURITY: the size, never the content. Reasoning is the model's
	// unfiltered scratchpad - it can restate a credential in words the value
	// redactor never sees - and replay does not need it (the fake provider
	// scripts responses). What an operator asking "why did this turn cost
	// that much?" needs is exactly this number.
	ReasoningBytes int `json:"reasoning_bytes,omitempty"`

	// tool_call / tool_result. CallID links both back to the requesting
	// llm_response entry in ToolCallIDs.
	CallID string `json:"tool_call_id,omitempty"`
	Tool   string `json:"tool,omitempty"`
	Args   string `json:"args,omitempty"`
	Result string `json:"result,omitempty"`
	IsErr  bool   `json:"is_error,omitempty"`
	// Outcome and ResultBytes are the tool_result observability fields added
	// in v0.1.0 (additive within schema v1, see docs/contracts/jsonl-events.md).
	// ResultBytes is the FULL size of the result text before clipping, so a
	// reader can see how much the 8 KiB log clip dropped from what the model
	// actually read. The tool's exit status reuses ExitCode below: it is the
	// same fact under the same JSON name, and two struct fields cannot share
	// one tag.
	Outcome     ToolOutcome `json:"outcome,omitempty"`
	ResultBytes int         `json:"result_bytes,omitempty"`

	// MCP events (v1.2). Server names one server; the rest is per event type.
	// OK is a pointer for the same reason ExitCode is: `ok:false` is the
	// interesting half of a connect attempt and omitempty would delete it.
	Server          string `json:"server,omitempty"`
	Transport       string `json:"transport,omitempty"`
	OK              *bool  `json:"ok,omitempty"`
	ErrorClass      string `json:"error_class,omitempty"`
	Error           string `json:"error,omitempty"`
	ProtocolVersion string `json:"protocol_version,omitempty"`
	ServerName      string `json:"server_name,omitempty"`
	ServerVersion   string `json:"server_version,omitempty"`
	SessionFP       string `json:"session_fp,omitempty"`
	ToolCount       int    `json:"tool_count,omitempty"`
	// Auth names the credential mechanism the connect used ("oauth"), and is
	// absent when the server needed none (v1.3, additive). SECURITY: it is the
	// MECHANISM, never the credential - no token, issuer or expiry is logged.
	Auth       string           `json:"auth,omitempty"`
	Tools      []MCPToolListed  `json:"tools,omitempty"`
	TotalBytes int              `json:"total_bytes,omitempty"`
	Skipped    []MCPSkippedTool `json:"skipped,omitempty"`
	Reason     string           `json:"reason,omitempty"`
	MCPErrors  int              `json:"mcp_errors,omitempty"`

	// run_end
	Status      string `json:"status,omitempty"`
	ExitCode    *int   `json:"exit_code,omitempty"`
	Turns       int    `json:"turns,omitempty"`
	ToolCalls   int    `json:"tool_calls,omitempty"`
	TotalTokens int    `json:"total_tokens,omitempty"`
	DurationMS  int64  `json:"duration_ms,omitempty"`
}

// maxLoggedField bounds how much of args/results is persisted per event so a
// single huge tool result cannot balloon the session file.
const maxLoggedField = 8 * 1024

// Writer appends events to one run's JSONL file. A nil *Writer is valid and
// discards everything, so callers do not branch on "is logging enabled".
type Writer struct {
	w      io.WriteCloser
	clock  Clock
	redact func(string) string
	path   string
	// mu serializes emit and the mcpErrors accesses, making a *Writer safe
	// for concurrent use. Today's loop is sequential, but the MCP connect
	// phase already drives one writer from several goroutines (through cmd's
	// observer), and parallel tool calls (roadmap #2) will do the same for
	// tool events - the lock belongs here, not in every caller.
	mu sync.Mutex
	// mcpErrors counts MCP-attributable failures over the whole run; it is
	// reported once, on run_end, so an operator grepping a single line can see
	// that a degraded MCP server was in play. SetMCPErrors is the only writer.
	mcpErrors int
}

// Options configures New.
type Options struct {
	// Clock defaults to time.Now (UTC).
	Clock Clock
	// Secrets lists values (e.g. the API key) that must never appear in
	// the log. SECURITY: redaction is by value because secrets reach the
	// log through arbitrary channels (tool output, model echoes).
	//
	// It is a snapshot: values added to the run afterwards are not covered.
	// Ignored when SecretSource is set.
	Secrets []string
	// SecretSource is the run's live secret registry. When set, the Writer
	// redacts through it on every write, so a secret registered after New
	// (an OAuth token minted mid-run) is scrubbed from that moment on.
	// SECURITY: prefer this over Secrets whenever the run can mint a
	// credential; the two exist together only so callers with a fixed list
	// need not build a set.
	SecretSource *SecretSet
}

// New creates the session file inside dir. The filename embeds the UTC
// timestamp so files sort chronologically with plain ls.
func New(dir string, opts Options) (*Writer, error) {
	clock := opts.Clock
	if clock == nil {
		clock = time.Now
	}
	// 0o750/0o600: session logs can carry sensitive tool output; keep them
	// out of reach of other local users.
	if err := os.MkdirAll(dir, 0o750); err != nil {
		return nil, fmt.Errorf("creating session dir: %w", err)
	}
	// The filename embeds the PID next to the timestamp so two cron runs
	// starting the same second get distinct files.
	name := fmt.Sprintf("run-%s-%d.jsonl", clock().UTC().Format("20060102T150405Z"), os.Getpid())
	path := filepath.Join(dir, name)
	// O_EXCL: even a same-second same-PID collision must not interleave two
	// runs into one file; failing loudly beats corrupting the log.
	//
	// CONTRACT: O_APPEND makes "append-only" (docs/contracts/jsonl-events.md) a kernel guarantee
	// instead of a property of this process being the only writer. Every write
	// is positioned at the current end of file by the kernel, so nothing this
	// writer emits can land on top of bytes it did not write - whoever else
	// grew the file (a rotator, an operator, a log shipper). It costs nothing:
	// the writer never seeks, so the flag only removes a way to be wrong.
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL|os.O_APPEND, 0o600) //nolint:gosec // G304: the path is built from the user's own session_dir config.
	if err != nil {
		return nil, fmt.Errorf("creating session file: %w", err)
	}
	// The live registry wins; a caller that passed only a fixed list gets an
	// equivalent one-off set, so there is a single redaction path below.
	secrets := opts.SecretSource
	if secrets == nil {
		secrets = NewSecretSet(opts.Secrets)
	}
	return &Writer{w: f, clock: clock, redact: secrets.Redact, path: path}, nil
}

// Path returns the session file location ("" for a nil writer).
func (w *Writer) Path() string {
	if w == nil {
		return ""
	}
	return w.path
}

// SecretSet is a concurrent, add-only set of secret values shared by every
// sink of one run (session log, -v progress feed, MCP stderr relay, error
// lines).
//
// SECURITY: it exists because redaction used to be frozen when the run
// started. OAuth mints an access token in the middle of a run, and a sink
// holding a snapshot taken before that would print the token verbatim. One
// live set, handed to every sink at startup, means a value registered at any
// later moment is scrubbed everywhere from the moment it exists.
//
// It is add-only on purpose: a credential that has been rotated away is still
// a credential in the transcript above, so nothing ever leaves the set.
// A nil *SecretSet is valid and redacts nothing, so callers with no secrets do
// not branch.
type SecretSet struct {
	// mu guards secrets. RWMutex because Redact is the hot, concurrent
	// operation (several MCP relays plus the loop) and Add is rare.
	mu sync.RWMutex
	// secrets holds the non-empty values, kept sorted longest-first (see
	// Redact for why the order is load-bearing).
	secrets []string
}

// NewSecretSet returns a set seeded with initial. Empty strings in initial are
// ignored.
func NewSecretSet(initial []string) *SecretSet {
	s := &SecretSet{}
	s.Add(initial...)
	return s
}

// Add registers more secrets. Empty values are ignored (replacing "" would
// corrupt every event), duplicates are harmless, and the set is re-sorted so
// the longest-first guarantee holds for values added later too. Safe for
// concurrent use; a nil receiver is a no-op.
func (s *SecretSet) Add(values ...string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, v := range values {
		if v != "" {
			s.secrets = append(s.secrets, v)
		}
	}
	// Stable so equal-length secrets keep a deterministic (registration)
	// order - golden files must not depend on sort tie-breaking.
	slices.SortStableFunc(s.secrets, func(a, b string) int { return len(b) - len(a) })
}

// Redact replaces every registered secret in text with "[REDACTED]". Safe for
// concurrent use with Add; a nil receiver returns text unchanged.
//
// SECURITY: every registered non-empty secret is redacted unconditionally - a
// short DB password is still a credential, and a noisy log is strictly better
// than a leaked secret.
// SECURITY: replacement runs longest-secret-first. Registration order must not
// decide how much leaks: in raw order, a short secret that is a substring of a
// longer one ("sk-" registered before "sk-secret-value") consumes its prefix
// and the longer secret's tail survives into the log.
func (s *SecretSet) Redact(text string) string {
	if s == nil {
		return text
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	for _, secret := range s.secrets {
		text = strings.ReplaceAll(text, secret, "[REDACTED]")
	}
	return text
}

// Redactor builds the value-replacement function this package applies to
// every event it writes: each registered secret becomes "[REDACTED]". The
// returned function closes over a frozen set and is safe for concurrent use.
//
// It is exported because the session log is not the only sink a run's text
// reaches - `amele run -v` streams the same model-chosen tool arguments to
// stderr, which a cron job persists just as durably. SECURITY: any such sink
// must redact through THIS definition, so there is one meaning of redaction
// rather than a second one that drifts.
//
// It is kept for callers whose secret list cannot change after startup; a
// caller that must see secrets registered later (OAuth tokens) holds a
// *SecretSet and calls its Redact method instead. Both paths run the exact
// same code - this is NewSecretSet(secrets).Redact.
func Redactor(secrets []string) func(string) string {
	return NewSecretSet(secrets).Redact
}

// emit writes one event line. Write errors are deliberately swallowed after
// the file is open: a full disk must degrade observability, never abort a
// running agent.
func (w *Writer) emit(e Event) {
	if w == nil {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	e.V = SchemaVersion
	e.TS = w.clock().UTC()
	line, err := json.Marshal(e)
	if err != nil {
		return
	}
	_, _ = w.w.Write(append(line, '\n'))
}

// clip bounds and redacts a free-text field before logging.
func (w *Writer) clip(text string) string {
	if w == nil {
		return ""
	}
	text = w.redact(text)
	if len(text) > maxLoggedField {
		cut := maxLoggedField
		// back up to a rune boundary, torn runes become U+FFFD in json.Marshal
		for cut > 0 && !utf8.RuneStart(text[cut]) {
			cut--
		}
		// 8192 + 12 baytlık marker = 8204; sözleşme dokümanında da böyle geçiyor,
		// cap'i marker'ı içerecek şekilde değiştirme
		return text[:cut] + "...[clipped]"
	}
	return text
}

// RunStart records the beginning of a run.
func (w *Writer) RunStart(model, task string) {
	w.emit(Event{Type: "run_start", Model: model, Task: w.clip(task)})
}

// LLMResponse records one provider round-trip: the assistant's text (clipped
// and redacted), the IDs of any tool calls it requested, the token accounting,
// and the SIZE of the turn's reasoning payload.
//
// reasoningBytes is a length, not a payload: the caller passes
// len(message.Reasoning) and the reasoning itself is never written (see
// Event.ReasoningBytes). Zero means the turn carried none.
func (w *Writer) LLMResponse(turn int, content string, toolCallIDs []string, inputTokens, outputTokens int, finishReason string, reasoningBytes int) {
	w.emit(Event{
		Type: "llm_response", Turn: turn,
		Content: w.clip(content), ToolCallIDs: toolCallIDs,
		InputTokens: inputTokens, OutputTokens: outputTokens,
		FinishReason: finishReason, ReasoningBytes: reasoningBytes,
	})
}

// ToolCall records a tool invocation request from the model. callID links the
// event to the requesting llm_response.
func (w *Writer) ToolCall(callID, tool, args string) {
	w.emit(Event{Type: "tool_call", CallID: callID, Tool: tool, Args: w.clip(args)})
}

// ToolOutcome classifies how one tool call ended, for the operator reading the
// log rather than for the model reading the result text.
//
// CONTRACT: these values are the published `outcome` enum of the tool_result
// event (docs/contracts/jsonl-events.md). They are matched literally by log
// consumers, so a spelling change is a breaking schema change; adding a value
// is additive, and a consumer must treat an unknown one as "something else
// happened".
//
// It exists because `is_error` cannot answer the operator's question. is_error
// is frozen at "the harness failed to dispatch this call", which deliberately
// says nothing about a tool that ran and failed: a non-zero exit is ordinary
// task information (`grep` exits 1 when it finds nothing), so it is content the
// model reacts to, not an error. The outcome is the second, out-of-band answer
// - what happened - carried beside the text instead of inside it.
type ToolOutcome string

// The tool_result outcome values.
//
// Seven of the first eight are the enum as specified; `aborted` is the
// addition,
// and it is here because the runner can genuinely end a tool call that way
// (the RUN was cancelled or hit its overall timeout under a still-running
// command) and folding it into `timeout` would blame the tool's own budget for
// a Ctrl-C, which is exactly the confusion this field exists to remove.
// `tool_error` and `indeterminate` were added in v1.2 for MCP tool calls
// (docs/contracts/jsonl-events.md): a tool that answered "I failed", and a
// call whose answer never came back.
const (
	// OutcomeOK means the tool did its job.
	OutcomeOK ToolOutcome = "ok"
	// OutcomeTimeout means the tool's own timeout killed the command.
	OutcomeTimeout ToolOutcome = "timeout"
	// OutcomeNonzeroExit means the command ran to completion and exited
	// non-zero; the status is in ExitCode. Not an error: see ToolOutcome.
	OutcomeNonzeroExit ToolOutcome = "nonzero_exit"
	// OutcomeAborted means the run ended under the command - its overall
	// timeout or a SIGINT/SIGTERM - not the tool's own budget.
	OutcomeAborted ToolOutcome = "aborted"
	// OutcomeDeniedPolicy means an operator policy refused the call before it
	// ran: the permission profile's `deny`, or the shell tool's allow/deny
	// patterns.
	OutcomeDeniedPolicy ToolOutcome = "denied_policy"
	// OutcomeDeniedNoTTY means an "ask" policy auto-denied the call because no
	// terminal was attached to ask a human on (the headless fail-safe).
	OutcomeDeniedNoTTY ToolOutcome = "denied_no_tty"
	// OutcomeAskRefused means a human was asked and said no.
	OutcomeAskRefused ToolOutcome = "ask_refused"
	// OutcomeError means the harness could not dispatch the call at all -
	// unknown tool, unusable arguments, a failed approval check. This is the
	// situation is_error marks.
	OutcomeError ToolOutcome = "error"
	// OutcomeToolError means an MCP tool ran and reported its own failure
	// (`isError` in the MCP tool result). Not a harness error: the failure is
	// the tool's answer, and the model reads it as content.
	OutcomeToolError ToolOutcome = "tool_error"
	// OutcomeIndeterminate means the response was lost after the request was
	// sent - the side effect may or may not have happened. amele never
	// retries such a call; that decision belongs to a human reading the log.
	OutcomeIndeterminate ToolOutcome = "indeterminate"
)

// ToolResult is one completed tool call as the log records it.
//
// It is a struct rather than a parameter list because the fields are optional
// and easy to transpose: three strings and a bool in a row is a call site where
// swapping two arguments compiles and lies.
type ToolResult struct {
	// CallID links the event to its tool_call (and to the requesting
	// llm_response).
	CallID string
	// Tool is the tool name as the model wrote it - it may not exist.
	Tool string
	// Result is the text handed back to the model, before clipping and
	// redaction.
	Result string
	// IsErr marks a harness dispatch failure. CONTRACT: frozen meaning - a
	// tool that ran and failed is NOT an error here; see Outcome.
	IsErr bool
	// Outcome classifies the ending for the operator.
	Outcome ToolOutcome
	// ExitCode is the process exit status when a subprocess/shell tool ran and
	// its status is known; nil otherwise. A pointer because "not applicable"
	// and "exited 0" are different statements.
	ExitCode *int
}

// ToolResult records a tool's outcome.
//
// CONTRACT: this is the one place the tool_result event is serialized. The
// result size is measured HERE, before clip() shortens (and redaction rewrites)
// the text, so result_bytes is the size of what the model actually read - the
// whole point of the field.
func (w *Writer) ToolResult(r ToolResult) {
	w.emit(Event{
		Type: "tool_result", CallID: r.CallID, Tool: r.Tool,
		Result: w.clip(r.Result), IsErr: r.IsErr,
		Outcome: r.Outcome, ResultBytes: len(r.Result), ExitCode: r.ExitCode,
	})
}

// MCPToolListed is one tool an MCP server advertised, as the log records it.
//
// The SHA-256 and byte size exist so an operator can prove after the fact
// WHICH definition the model was shown - an MCP server can change a tool's
// description between runs, and the harness token budget is spent on those
// bytes.
type MCPToolListed struct {
	// Name is the tool name as amele exposed it to the model (server-prefixed
	// and, when needed, normalized to the provider's allowed character set).
	Name string `json:"name"`
	// OriginalName is the server's own name, present only when Name differs
	// from it because normalization rewrote it.
	OriginalName string `json:"original_name,omitempty"`
	// SHA256 is the hex digest of the tool definition amele sent to the model.
	SHA256 string `json:"sha256"`
	// Bytes is the size of that definition.
	Bytes int `json:"bytes"`
	// Annotations carries the MCP tool hints amele understood
	// (readOnly, destructive, openWorld, idempotent); only present keys are
	// written, so an absent key means "the server said nothing", not "false".
	Annotations map[string]bool `json:"annotations,omitempty"`
}

// MCPSkippedTool is one advertised tool amele did not expose to the model.
type MCPSkippedTool struct {
	// Name is the server's name for the tool.
	Name string `json:"name"`
	// Reason is why it was dropped (e.g. `excluded`, `not included`,
	// `definition too large`; a name conflict is never a skip - it is fatal).
	Reason string `json:"reason"`
}

// MCPConnect is one connection attempt to one MCP server, successful or not.
type MCPConnect struct {
	// Server is the server's name from the config.
	Server string
	// Transport is how it was reached (`stdio`, `http`).
	Transport string
	// OK reports whether the handshake completed.
	OK bool
	// ErrorClass groups the failure for aggregation (one of `spawn`, `network`,
	// `auth`, `protocol`, `timeout`); empty on success.
	ErrorClass string
	// Error is the human-readable failure text; clipped and redacted before
	// it is written.
	Error string
	// DurationMS is how long the attempt took.
	DurationMS int64
	// ProtocolVersion is the MCP protocol version agreed on.
	ProtocolVersion string
	// ServerName and ServerVersion are the server's self-reported identity.
	ServerName    string
	ServerVersion string
	// SessionFP is a short fingerprint of the session id. SECURITY: the raw
	// id is a bearer credential for the session and is never logged.
	SessionFP string
	// ToolCount is how many tools the server advertised.
	ToolCount int
	// Auth is the credential mechanism the attempt used (`oauth`), or empty
	// when the server needed none. SECURITY: the mechanism only - the token
	// itself never reaches the log.
	Auth string
}

// MCPToolsListed is the tool inventory amele took from one MCP server.
type MCPToolsListed struct {
	// Server is the server's name from the config.
	Server string
	// Tools are the definitions actually exposed to the model.
	Tools []MCPToolListed
	// TotalBytes is the summed size of those definitions - the token budget
	// this server cost.
	TotalBytes int
	// Skipped lists advertised tools that were not exposed, with the reason.
	Skipped []MCPSkippedTool
}

// MCPDisconnect records a server connection ending.
type MCPDisconnect struct {
	// Server is the server's name from the config.
	Server string
	// Reason is `run_end`, `reconnect` or `error`.
	Reason string
}

// MCPConnect records one connection attempt to an MCP server. A nil *Writer
// discards it, like every other method here.
func (w *Writer) MCPConnect(e MCPConnect) {
	ok := e.OK
	w.emit(Event{
		Type: "mcp_connect", Server: e.Server, Transport: e.Transport, OK: &ok,
		ErrorClass: e.ErrorClass, Error: w.clip(e.Error), DurationMS: e.DurationMS,
		ProtocolVersion: e.ProtocolVersion, ServerName: e.ServerName,
		ServerVersion: e.ServerVersion, SessionFP: e.SessionFP, ToolCount: e.ToolCount,
		Auth: e.Auth,
	})
}

// MCPToolsListed records the tool inventory taken from one MCP server.
func (w *Writer) MCPToolsListed(e MCPToolsListed) {
	w.emit(Event{
		Type: "mcp_tools_listed", Server: e.Server, Tools: e.Tools,
		TotalBytes: e.TotalBytes, Skipped: e.Skipped,
	})
}

// MCPDisconnect records an MCP server connection ending.
func (w *Writer) MCPDisconnect(e MCPDisconnect) {
	w.emit(Event{Type: "mcp_disconnect", Server: e.Server, Reason: w.clip(e.Reason)})
}

// SetMCPErrors records how many MCP-attributable failures the run saw. The
// count is written once, as run_end.mcp_errors, and only when it is non-zero:
// an absent field means 0. Callers set the running total; a nil *Writer
// ignores it.
func (w *Writer) SetMCPErrors(n int) {
	if w == nil {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.mcpErrors = n
}

// RunEnd records the final status and totals, then closes the file.
func (w *Writer) RunEnd(status string, exitCode int, turns, toolCalls, totalTokens int, duration time.Duration) {
	if w == nil {
		return
	}
	// The counter is read under the lock, then released before emit takes it
	// again: mu is not reentrant.
	w.mu.Lock()
	mcpErrors := w.mcpErrors
	w.mu.Unlock()
	w.emit(Event{
		Type: "run_end", Status: status, ExitCode: &exitCode,
		Turns: turns, ToolCalls: toolCalls, TotalTokens: totalTokens,
		DurationMS: duration.Milliseconds(), MCPErrors: mcpErrors,
	})
	_ = w.w.Close()
}

// Summary renders the single-line run summary printed to stderr after every
// run: `✓ 8 turns, 3 tool calls, 41k tokens, 34.2s`. The turn and tool-call
// nouns are singular for a count of exactly 1 (`✓ 1 turn, 1 tool call, ...`);
// tokens and seconds are units, which stay as they are at any count.
func Summary(ok bool, turns, toolCalls, totalTokens int, duration time.Duration) string {
	mark := "✓"
	if !ok {
		mark = "✗"
	}
	return fmt.Sprintf("%s %d %s, %d %s, %s tokens, %.1fs",
		mark, turns, plural(turns, "turn"), toolCalls, plural(toolCalls, "tool call"),
		formatTokens(totalTokens), duration.Seconds())
}

// plural returns noun as-is for a count of exactly 1 and the "+s" form
// otherwise, including for 0 ("0 turns" is how English counts nothing).
// Only the two nouns Summary counts go through it, so the naive "+s" rule
// cannot meet an irregular word.
func plural(n int, noun string) string {
	if n == 1 {
		return noun
	}
	return noun + "s"
}

// formatTokens renders token counts the way humans read them (41k, 1.2M).
func formatTokens(n int) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	case n >= 1_000:
		return fmt.Sprintf("%.1fk", float64(n)/1_000)
	default:
		return fmt.Sprintf("%d", n)
	}
}
