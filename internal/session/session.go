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
	// CacheReadTokens and CacheWriteTokens are the prompt-cache share of the
	// turn's input (v1.7, additive; absent means none). Both are a SUBSET of
	// InputTokens, never an addition to it - a reader that sums them into the
	// total double-charges the run, and so would a budget.
	//
	// The pair is shared with run_end, which carries CacheReadTokens ONLY: a
	// write count is a per-turn fact about where a cache breakpoint landed,
	// while "how much of this run came back from cache" is the cumulative
	// number an operator reads against the bill.
	CacheReadTokens  int `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
	// ReasoningBytes is the size of the provider's reasoning payload for this
	// turn (v1.4, additive; absent means none). It is recorded whether or not
	// the content itself is logged, so "why did this turn cost that much?"
	// always has an answer.
	//
	// SECURITY: the size only, BY DEFAULT. Reasoning is the model's unfiltered
	// scratchpad - it can restate a credential in words the value redactor
	// never sees - so the content is written only when the operator opts in
	// with log_reasoning: true (Options.LogReasoning), and then it goes into
	// Reasoning below through the same redact+clip path as every other
	// free-text field.
	ReasoningBytes int `json:"reasoning_bytes,omitempty"`
	// Reasoning is the provider's reasoning payload for the turn, as a
	// string (clipped + redacted like every other free-text field). Present
	// only when the writer was opened with LogReasoning (log_reasoning:
	// true, JSONL v1.5); absent otherwise - absence is the pre-v1.5 and
	// default-config shape.
	Reasoning string `json:"reasoning,omitempty"`

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

	// Truncated marks a tool_result whose text was cut to the tool-result byte
	// cap before the model saw it (v1.6, additive). Omitted when false.
	Truncated bool `json:"truncated,omitempty"`

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

// maxLoggedField is the DEFAULT bound on how much of args/results is persisted
// per event, so a single huge tool result cannot balloon the session file. It
// applies whenever Options.MaxLoggedField is zero, which is every caller that
// does not configure limits.max_logged_field.
const maxLoggedField = 8 * 1024

// Writer appends events to one run's JSONL file. A nil *Writer is valid and
// discards everything, so callers do not branch on "is logging enabled".
type Writer struct {
	w      io.WriteCloser
	clock  Clock
	redact func(string) string
	path   string
	// maxField is the resolved per-field byte bound: Options.MaxLoggedField
	// when non-zero, the maxLoggedField default otherwise. Negative means no
	// bound at all (redaction still runs; see clip).
	maxField int
	// logReasoning mirrors Options.LogReasoning. SECURITY: when false the
	// reasoning payload handed to LLMResponse is dropped rather than written.
	logReasoning bool
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
	// MaxLoggedField overrides how many bytes of each free-text field are
	// persisted per event. Zero means the package default (8192), so every
	// existing caller keeps today's behavior; negative disables the bound
	// entirely (limits.max_logged_field: 0 in the config). The ...[clipped]
	// marker is appended on top of the bound, and redaction runs regardless.
	MaxLoggedField int
	// LogReasoning opts the provider's reasoning payload into the log
	// (log_reasoning: true). SECURITY: default off. Reasoning is the model's
	// unfiltered scratchpad - it can restate a credential in words the value
	// redactor never sees - so the content is persisted only on this
	// explicit opt-in, and even then it passes through the same redact+clip
	// path as every other field. Event.ReasoningBytes carries the size
	// either way.
	LogReasoning bool
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
	// Zero is "unset", not "unbounded": a caller that never heard of
	// MaxLoggedField must keep the historical 8 KiB bound. Unbounded is
	// spelled with a negative value so it has to be asked for.
	maxField := opts.MaxLoggedField
	if maxField == 0 {
		maxField = maxLoggedField
	}
	return &Writer{
		w: f, clock: clock, redact: secrets.Redact, path: path,
		maxField: maxField, logReasoning: opts.LogReasoning,
	}, nil
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
//
// SECURITY: each value is registered in every spelling a sink can carry it in,
// not only the literal one (see secretVariants).
func (s *SecretSet) Add(values ...string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, v := range values {
		s.secrets = append(s.secrets, secretVariants(v)...)
	}
	// Stable so equal-length secrets keep a deterministic (registration)
	// order - golden files must not depend on sort tie-breaking.
	slices.SortStableFunc(s.secrets, func(a, b string) int { return len(b) - len(a) })
}

// secretVariants returns every spelling of one secret the sinks can carry: the
// literal value, the interiors of its JSON string encodings, and the interiors
// of THOSE (a value quoted into a document that is itself quoted into another).
// An empty value yields nothing - replacing "" would corrupt every event.
//
// SECURITY: the JSON spellings are not cosmetic. Redaction is a literal
// substring search, and a secret does not only travel as itself: a provider's
// 400 body quotes the offending value back inside the server's own JSON, where
// a quote or a backslash arrives escaped (`a"b` as `a\"b`). Registering only
// the raw bytes let that copy through into the error message and from there
// into the log. Params values interpolated from ${ENV} do reach request bodies,
// so this is a live path, not a hypothetical one.
//
// Two JSON forms are registered because encoders disagree about HTML escaping:
// Go's own rewrites <, > and & into their \u escapes while most other servers
// emit them verbatim, and which encoder produced the body amele is redacting is
// not something amele gets to choose. Forms that coincide are skipped, so a
// secret made of ordinary token characters still registers exactly once.
//
// SECURITY: two levels of encoding are registered, not one. A gateway that
// wraps the upstream response - already a JSON document - inside its OWN JSON
// error body escapes the escapes, so `a"b` reaches the log as `a\\\"b` and
// shares no bytes with either one-level form. The \u spellings are registered
// for the same reason: an encoder that prefers unicode escapes writes the same
// quote as \u0022, which no other variant contains.
//
// This is best-effort defense in depth over the representations real gateways
// emit, not a semantic JSON redactor: it enumerates spellings rather than
// parsing the text it is about to redact, so an encoding nobody has been seen
// to produce (three levels of wrapping, a value split across two fields) is out
// of reach by construction. The enumeration is what makes it cheap and
// order-independent enough to run on every event.
func secretVariants(v string) []string {
	if v == "" {
		return nil
	}
	variants := []string{v}
	add := func(s string) {
		if s != "" && !slices.Contains(variants, s) {
			variants = append(variants, s)
		}
	}
	// One level: the shape the value has spliced into a JSON document.
	level1 := encodedSpellings(v)
	for _, s := range level1 {
		add(s)
	}
	// Two levels: that document quoted inside another one. Deeper nesting is
	// deliberately not pursued - each level multiplies the variant count while
	// the observed shapes stop at two.
	for _, s := range level1 {
		for _, ss := range encodedSpellings(s) {
			add(ss)
		}
	}
	return variants
}

// encodedSpellings returns the JSON-string-interior spellings of v that DIFFER
// from v: the two escapeHTML modes Go's encoder offers, plus the \uXXXX form an
// encoder that prefers unicode escapes emits. Spellings that coincide with v or
// with each other are dropped, which is why a secret made of ordinary token
// characters yields none at all and still registers exactly once.
func encodedSpellings(v string) []string {
	out := make([]string, 0, 3)
	appendNew := func(s string) {
		if s != "" && s != v && !slices.Contains(out, s) {
			out = append(out, s)
		}
	}
	for _, escapeHTML := range []bool{false, true} {
		appendNew(jsonStringInterior(v, escapeHTML))
	}
	appendNew(unicodeEscaper.Replace(v))
	return out
}

// unicodeEscaper spells the characters a JSON encoder MAY escape as \uXXXX the
// way an encoder that prefers that form writes them. Go's encoder uses the
// short escapes for the quote and the backslash and the \u form for the HTML
// trio; other encoders (PHP's json_encode without JSON_UNESCAPED_*, several
// gateway front ends) use \u throughout, and that spelling shares no bytes with
// the short one.
//
// Single-pass by construction: strings.Replacer never rescans what it wrote, so
// the backslashes it introduces are not escaped again. Control characters are
// left to the encoder-produced spellings above - a secret containing a newline
// is not a shape any provider has been seen to emit.
//
// Package-level because it is a compiled table with no mutable state: a
// Replacer is immutable once built and safe for concurrent use, so this is a
// constant that the language cannot spell as one.
var unicodeEscaper = strings.NewReplacer(
	`"`, `\u0022`,
	`\`, `\u005c`,
	`<`, `\u003c`,
	`>`, `\u003e`,
	`&`, `\u0026`,
)

// jsonStringInterior renders v as a JSON string and returns it without the
// surrounding quotes - the shape a value has when a server has spliced it into
// a JSON document. It returns "" if the encoding fails, which json cannot do
// for a string (invalid UTF-8 is replaced, not rejected); the branch is a
// defensive fallback rather than a reachable one.
func jsonStringInterior(v string, escapeHTML bool) string {
	var buf strings.Builder
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(escapeHTML)
	if err := enc.Encode(v); err != nil {
		return ""
	}
	// Encode appends a newline; the quotes are the encoder's, not the body's.
	encoded := strings.TrimSuffix(buf.String(), "\n")
	if len(encoded) < 2 {
		return ""
	}
	return encoded[1 : len(encoded)-1]
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
//
// SECURITY: redaction runs FIRST and unconditionally - before the bound is
// even consulted - so disabling the bound (w.maxField < 0) never means
// disabling redaction.
func (w *Writer) clip(text string) string {
	if w == nil {
		return ""
	}
	text = w.redact(text)
	if w.maxField < 0 || len(text) <= w.maxField {
		return text
	}
	cut := w.maxField
	// back up to a rune boundary, torn runes become U+FFFD in json.Marshal
	for cut > 0 && !utf8.RuneStart(text[cut]) {
		cut--
	}
	// 8192 + the 12-byte marker = 8204, exactly as the contract doc states;
	// do not change the cap to absorb the marker.
	return text[:cut] + "...[clipped]"
}

// RunStart records the beginning of a run.
func (w *Writer) RunStart(model, task string) {
	w.emit(Event{Type: "run_start", Model: model, Task: w.clip(task)})
}

// LLMResponse is one model turn as the log records it.
//
// It is a struct rather than a parameter list for the same reason ToolResult
// is: the positional form took seven arguments, four of them int and two of
// those adjacent (inputTokens, outputTokens), so a transposed call site
// compiles and then lies in the log (issue #15).
type LLMResponse struct {
	// Turn is the 1-based turn number, offset by the caller's TurnBase so a
	// multi-call session keeps counting up.
	Turn int
	// Content is the assistant text, before clipping and redaction.
	Content string
	// ToolCallIDs are the IDs of any tool calls the turn requested.
	ToolCallIDs []string
	// InputTokens and OutputTokens are the provider-reported accounting.
	InputTokens  int
	OutputTokens int
	// FinishReason is the provider's stop reason, verbatim.
	FinishReason string
	// CacheReadTokens and CacheWriteTokens are the prompt-cache share of this
	// turn's InputTokens (see Event.CacheReadTokens): tokens served from the
	// provider's cache, and tokens the provider charged to write a new cache
	// entry. Both are inside InputTokens, so neither is added to any total.
	CacheReadTokens  int
	CacheWriteTokens int
	// ReasoningBytes is the SIZE of the turn's reasoning payload - a length,
	// not a payload: the caller passes len(message.Reasoning) and the size is
	// recorded whether or not the content is (see Event.ReasoningBytes). Zero
	// means the turn carried none.
	ReasoningBytes int
	// Reasoning is the turn's raw reasoning payload as text, before clipping
	// and redaction. SECURITY: the writer DROPS it unless it was opened with
	// Options.LogReasoning; passing it is always safe.
	Reasoning string
}

// LLMResponse records a model turn: content (clipped and redacted), the IDs of
// any tool calls it requested, the token accounting (including the turn's
// prompt-cache share, when the provider reported one), and the size of the
// turn's reasoning payload. The reasoning content itself is written only when
// the writer was opened with LogReasoning, through the same redact+clip path.
func (w *Writer) LLMResponse(r LLMResponse) {
	e := Event{
		Type: "llm_response", Turn: r.Turn,
		Content: w.clip(r.Content), ToolCallIDs: r.ToolCallIDs,
		InputTokens: r.InputTokens, OutputTokens: r.OutputTokens,
		FinishReason: r.FinishReason, ReasoningBytes: r.ReasoningBytes,
		CacheReadTokens: r.CacheReadTokens, CacheWriteTokens: r.CacheWriteTokens,
	}
	// SECURITY: the opt-in is checked here, so the default path cannot write a
	// reasoning byte no matter what the caller passes.
	if w != nil && w.logReasoning {
		e.Reasoning = w.clip(r.Reasoning)
	}
	w.emit(e)
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
	// Truncated marks that the text was cut to the cap; the marker in Result
	// says so to the model, this says so to the reader.
	Truncated bool
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
		Truncated: r.Truncated,
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
//
// cacheReadTokens is the run's cumulative prompt-cache read count. It is a
// SUBSET of totalTokens (which is already input+output), so it is reported
// beside the total, never added to it; zero writes no key at all.
func (w *Writer) RunEnd(status string, exitCode int, turns, toolCalls, totalTokens, cacheReadTokens int, duration time.Duration) {
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
		CacheReadTokens: cacheReadTokens,
		DurationMS:      duration.Milliseconds(), MCPErrors: mcpErrors,
	})
	_ = w.w.Close()
}

// Summary renders the single-line run summary printed to stderr after every
// run: `✓ 8 turns, 3 tool calls, 41k tokens, 34.2s`. The turn and tool-call
// nouns are singular for a count of exactly 1 (`✓ 1 turn, 1 tool call, ...`);
// tokens and seconds are units, which stay as they are at any count.
//
// cachedTokens is the share of totalTokens that came back from a prompt cache.
// When it is positive the token figure gains a parenthetical -
// `41.0k tokens (28.0k cached)` - and when it is zero the line is byte-for-byte
// the one every pre-v0.3 run printed: an operator grepping the old shape, and
// a provider that reports no cache counts at all, must see no change.
func Summary(ok bool, turns, toolCalls, totalTokens, cachedTokens int, duration time.Duration) string {
	mark := "✓"
	if !ok {
		mark = "✗"
	}
	cached := ""
	if cachedTokens > 0 {
		cached = fmt.Sprintf(" (%s cached)", formatTokens(cachedTokens))
	}
	return fmt.Sprintf("%s %d %s, %d %s, %s tokens%s, %.1fs",
		mark, turns, plural(turns, "turn"), toolCalls, plural(toolCalls, "tool call"),
		formatTokens(totalTokens), cached, duration.Seconds())
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
