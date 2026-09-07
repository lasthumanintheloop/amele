// Package resume rebuilds a conversation from a session log so a run can
// continue where an earlier one stopped. It reads the JSONL the session
// package writes (docs/contracts/jsonl-events.md) and produces the neutral
// message history the loop consumes; it never talks to a provider or a tool.
//
// # Reconstruction
//
// The log is decoded into session.Event values and walked in order, which is
// the order the run happened in (the contract's ordering guarantees):
//
//   - run_start opens the replay: its task becomes the first user message and
//     its model/provider are reported for the caller to compare against the
//     current config.
//   - llm_response opens a turn: a new assistant message carrying the turn's
//     text, and the turn's tool_call_ids as the calls it is expected to make.
//   - tool_call fills one of those expected calls in - name and raw argument
//     string - onto the open assistant message.
//   - tool_result appends the tool message that answers one call.
//   - closing a turn (the next llm_response, or the end of the file) gives
//     every requested-and-dispatched call that never got a result a synthetic
//     tool message (PendingResultMessage) and lists it in Replay.Pending, so
//     the history a provider sees is complete without anything being re-run.
//   - every other event (mcp_*, provider_fallback, run_end) says nothing about
//     what the model saw and is ignored here; provider_fallback is read in a
//     pre-scan, for carriers only.
//
// A call the turn requested but never dispatched - no tool_call event, because
// the run died between the two writes - is DROPPED from the assistant message
// instead of being invented: its name and arguments were never written, so
// there is nothing faithful to send, and a call that never left the harness has
// no result to stand in for either. It is therefore not Pending; the model is
// free to ask again.
//
// # Fidelity
//
// The log is a record, not a transcript store: free-text fields are clipped to
// limits.max_logged_field bytes. A conversation rebuilt from clipped text would
// be a DIFFERENT conversation than the one the model had, so any field this
// package would consume that ends in session.ClipMarker fails the read with
// ErrClipped rather than being resumed from silently. Nothing is guessed or
// repaired: a log that is not a faithful record is refused.
//
// SECURITY: a session log is operator-owned input, and this package treats it
// as data only - it decodes JSON, never executes anything, and re-runs no tool.
// The tool results it puts back into the history are the same model-facing text
// they were when they were produced, so resuming shows the model nothing it was
// not already shown. Secret values were replaced by "[REDACTED]" on the way
// into the log; that is what the model reads back and it is NOT a fidelity
// gate - the redacted text is the text of the run being continued.
package resume

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/lasthumanintheloop/amele/internal/llm"
	"github.com/lasthumanintheloop/amele/internal/session"
)

// Replay is what a log yields for continuation.
type Replay struct {
	// Task is run_start.task, the instruction the old run was given. It is the
	// first user message of Messages.
	Task string
	// Model is run_start.model of the old run. It is reported, never applied:
	// the resumed run uses the current config's model.
	Model string
	// Provider is run_start.provider, the backend identity of the old run. It
	// is empty in logs written before JSONL v1.8, which is why an empty
	// Options.Provider never matches it (see Options).
	Provider string
	// LastTurn is the highest llm_response turn number in the log. It is
	// reported for run_start.resumed_turn; the resumed run numbers its own
	// turns from 1 again.
	LastTurn int
	// Completed reports that the last llm_response carried no tool calls -
	// the old run reached a final answer, so continuing it needs a new
	// instruction from the caller.
	Completed bool
	// Pending lists the tool call ids that were dispatched but whose result
	// never reached the log, in call order. Each got the PendingResultMessage
	// tool message; nothing was re-executed.
	Pending []string
	// Messages is the rebuilt history: the task as a user message, then the
	// assistant turns and tool messages in log order. It NEVER contains a
	// system message - the system prompt is not logged, and the resumed run
	// takes it from the current config.
	Messages []llm.Message
	// Carriers reports that at least one assistant message got its provider
	// reasoning payload back (see Options.Provider).
	Carriers bool
}

// Options tunes carrier restoration.
type Options struct {
	// Provider is the current primary's identity (session.Event.Provider
	// spelling, e.g. "anthropic" or "openai/deepseek"). Reasoning payloads are
	// restored only when it equals the log's run_start.provider: a payload is
	// signed or hash-checked by the backend that produced it, so replaying one
	// into a different backend is at best rejected and at worst charged for.
	// Empty means "do not restore" - it is also what a pre-v1.8 log carries,
	// and two unknowns are not a match.
	Provider string
}

// The typed failures of a read. CONTRACT: the caller (cmd) maps all three onto
// exit code 2, and ErrClipped's message names the config key that makes a log
// resumable, so an operator can act on it without reading the docs.
var (
	// ErrClipped means a field the rebuilt conversation needs was shortened by
	// the log's own byte bound and is no longer what the model saw.
	ErrClipped = errors.New("session log is clipped")
	// ErrNotResumable means the file is a well-formed log of something that
	// cannot be continued: an interactive chat, a log with no run_start, or a
	// schema version this build does not read.
	ErrNotResumable = errors.New("session log is not resumable")
	// ErrMalformed means the file does not obey the event contract: a line
	// that is not JSON (including a torn last line after a hard kill), a tool
	// event whose id no turn requested, or two runs concatenated into one
	// file.
	ErrMalformed = errors.New("session log is malformed")
)

// PendingResultMessage is the tool message text that stands in for a result
// the interrupted run never wrote.
//
// CONTRACT: it is addressed to the MODEL, not to an operator: it says what
// happened and hands the decision back, because the harness must not re-run a
// tool call on its own - the call may have had a side effect that did land
// (the log records the request, not the outcome).
const PendingResultMessage = "error: the previous run was interrupted before this tool call completed; call it again if the result is still needed"

// chatTask is the fixed run_start.task of a chat session
// (docs/contracts/jsonl-events.md). A chat is a conversation with a person in
// it, not one task to finish, so its log is refused rather than continued.
const chatTask = "interactive chat"

// The event types this package understands. The rest are ignored by name, not
// by omission, so an added event type cannot silently change a replay.
const (
	eventRunStart         = "run_start"
	eventLLMResponse      = "llm_response"
	eventToolCall         = "tool_call"
	eventToolResult       = "tool_result"
	eventProviderFallback = "provider_fallback"
)

// Read parses the log at path and rebuilds the history. Failures are one of
// ErrClipped, ErrNotResumable or ErrMalformed (match with errors.Is), each
// wrapped with the path so a message can be printed as-is.
func Read(path string, opts Options) (*Replay, error) {
	f, err := os.Open(path) //nolint:gosec // G304: the path is the operator's own --resume argument.
	if err != nil {
		return nil, fmt.Errorf("opening session log: %w", err)
	}
	defer func() { _ = f.Close() }()
	rep, err := ReadFrom(f, opts)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return rep, nil
}

// ReadFrom rebuilds the history from an already-open log. It is the tested
// core of Read; callers with a file path want Read.
func ReadFrom(r io.Reader, opts Options) (*Replay, error) {
	events, err := decode(r)
	if err != nil {
		return nil, err
	}
	b, err := newBuilder(events, opts)
	if err != nil {
		return nil, err
	}
	// events[0] is the run_start newBuilder consumed.
	for _, ev := range events[1:] {
		if err := b.event(ev); err != nil {
			return nil, err
		}
	}
	b.closeTurn()
	return &b.rep, nil
}

// decode reads the whole log into memory.
//
// Whole, because two passes are needed anyway (a provider_fallback anywhere in
// the file disables carriers for the turns before it), and a session log is
// bounded by the run's own token and tool-output budgets. json.Decoder rather
// than a line scanner: a full-record log (limits.max_logged_field: 0) can carry
// a tool result far past any line-buffer size worth picking, and a reader that
// fails on long lines would fail exactly on the logs resume exists for.
func decode(r io.Reader) ([]session.Event, error) {
	dec := json.NewDecoder(r)
	var events []session.Event
	for {
		var ev session.Event
		err := dec.Decode(&ev)
		if errors.Is(err, io.EOF) {
			return events, nil
		}
		if err != nil {
			// One JSON value per line is the format, so the count of values
			// already read names the offending line.
			return nil, fmt.Errorf("%w: line %d: %v", ErrMalformed, len(events)+1, err)
		}
		events = append(events, ev)
	}
}

// callState tracks one requested tool call across the events of its turn.
type callState struct {
	// called records that the tool_call event was seen, which is what makes
	// the call reproducible (it carries the name and arguments).
	called bool
	// answered records that a tool_result was seen, so the turn does not need
	// a synthetic one.
	answered bool
}

// builder walks the events and grows the Replay. It holds the state of exactly
// one open turn: the log is chronological, so a turn is finished the moment the
// next llm_response starts.
type builder struct {
	rep Replay
	// carriers is whether restoration is ALLOWED for this log (decided once,
	// from the provider identity and the fallback history). rep.Carriers says
	// whether any payload actually was restored.
	carriers bool
	// turn is the open turn's number, used to name the turn in clip errors.
	turn int
	// assistant indexes the open turn's assistant message in rep.Messages.
	assistant int
	// expected is the open turn's tool_call_ids in the model's call order,
	// and calls is their state by id. Both are replaced on every llm_response.
	expected []string
	calls    map[string]*callState
}

// newBuilder applies the gates that can be decided from the log as a whole and
// from its run_start, then seeds the history with the task.
func newBuilder(events []session.Event, opts Options) (*builder, error) {
	if len(events) == 0 {
		return nil, fmt.Errorf("%w: the log is empty", ErrNotResumable)
	}
	for i, ev := range events {
		if ev.V != session.SchemaVersion {
			return nil, fmt.Errorf("%w: line %d carries schema version %d, this build reads version %d",
				ErrNotResumable, i+1, ev.V, session.SchemaVersion)
		}
	}
	// CONTRACT: run_start is the first line of every log (the ordering
	// guarantees). Anything else is a fragment, a rotated tail, or another
	// tool's JSONL - none of which has a task to continue.
	start := events[0]
	if start.Type != eventRunStart {
		return nil, fmt.Errorf("%w: the log does not begin with a run_start event", ErrNotResumable)
	}
	if start.Task == "" {
		return nil, fmt.Errorf("%w: its run_start carries no task", ErrNotResumable)
	}
	if start.Task == chatTask {
		return nil, fmt.Errorf("%w: it records an interactive chat, which has no task to continue", ErrNotResumable)
	}
	if err := gateClip("task", 0, start.Task); err != nil {
		return nil, err
	}
	b := &builder{assistant: -1}
	b.rep.Task, b.rep.Model, b.rep.Provider = start.Task, start.Model, start.Provider
	b.rep.Messages = []llm.Message{{Role: llm.RoleUser, Content: start.Task}}
	// Three conditions, all necessary: an identity to compare, the SAME
	// identity, and a run that never moved off it. A fallback anywhere in the
	// file taints the whole replay because the history is sent as one document
	// - the surviving payloads belong to a backend that is no longer the one
	// being talked to.
	b.carriers = opts.Provider != "" && opts.Provider == start.Provider && !hasFallback(events)
	return b, nil
}

// hasFallback reports whether the run ever moved along its fallback chain.
func hasFallback(events []session.Event) bool {
	for _, ev := range events {
		if ev.Type == eventProviderFallback {
			return true
		}
	}
	return false
}

// event applies one event to the replay under construction.
func (b *builder) event(ev session.Event) error {
	switch ev.Type {
	case eventRunStart:
		// Exactly one run_start per file; a second one means two runs were
		// concatenated, and continuing "both" is not a thing.
		return fmt.Errorf("%w: a second run_start event - two runs are concatenated in one file", ErrMalformed)
	case eventLLMResponse:
		return b.llmResponse(ev)
	case eventToolCall:
		return b.toolCall(ev)
	case eventToolResult:
		return b.toolResult(ev)
	default:
		// mcp_*, provider_fallback and run_end record what the HARNESS did;
		// none of them changed the conversation the model was shown.
		return nil
	}
}

// llmResponse closes the previous turn and opens a new assistant message.
func (b *builder) llmResponse(ev session.Event) error {
	b.closeTurn()
	if err := gateClip("content", ev.Turn, ev.Content); err != nil {
		return err
	}
	msg := llm.Message{Role: llm.RoleAssistant, Content: ev.Content}
	if err := b.restoreCarrier(&msg, ev); err != nil {
		return err
	}
	calls := make(map[string]*callState, len(ev.ToolCallIDs))
	for _, id := range ev.ToolCallIDs {
		if id == "" {
			return fmt.Errorf("%w: turn %d requests a tool call with an empty id", ErrMalformed, ev.Turn)
		}
		if _, dup := calls[id]; dup {
			return fmt.Errorf("%w: turn %d requests tool call id %q twice", ErrMalformed, ev.Turn, id)
		}
		calls[id] = &callState{}
	}
	b.turn, b.expected, b.calls = ev.Turn, ev.ToolCallIDs, calls
	b.assistant = len(b.rep.Messages)
	b.rep.Messages = append(b.rep.Messages, msg)
	if ev.Turn > b.rep.LastTurn {
		b.rep.LastTurn = ev.Turn
	}
	// Last one wins: a turn with no tool calls is a final answer, and only the
	// last turn of the log decides whether the run got there.
	b.rep.Completed = len(ev.ToolCallIDs) == 0
	return nil
}

// restoreCarrier puts the turn's reasoning payload back on the message when the
// log's backend is still the one being talked to.
//
// CONTRACT: the payload goes back as the RAW bytes the log holds - it is
// signed (Anthropic) or hash-checked (DeepSeek) by the provider that produced
// it, so re-encoding it would invalidate it. ReasoningField is deliberately
// left empty: the log does not record which wire key the payload arrived on, so
// the client's own default key applies.
//
// A turn with reasoning_bytes > 0 and no reasoning text is the ordinary,
// default-config shape (log_reasoning is off, or the log predates v1.5): the
// content was never written, so there is nothing to restore and the message is
// rebuilt without a carrier.
func (b *builder) restoreCarrier(msg *llm.Message, ev session.Event) error {
	if !b.carriers || ev.Reasoning == "" {
		return nil
	}
	// Gated only here: a clipped payload that would not be echoed anyway is
	// not a reason to refuse an otherwise faithful log.
	if err := gateClip("reasoning", ev.Turn, ev.Reasoning); err != nil {
		return err
	}
	msg.Reasoning = json.RawMessage(ev.Reasoning)
	b.rep.Carriers = true
	return nil
}

// toolCall records one dispatched call on the open assistant message.
func (b *builder) toolCall(ev session.Event) error {
	st, err := b.expect(ev.CallID, eventToolCall)
	if err != nil {
		return err
	}
	if st.called {
		return fmt.Errorf("%w: a second tool_call for id %q in turn %d", ErrMalformed, ev.CallID, b.turn)
	}
	if err := gateClip("args", b.turn, ev.Args); err != nil {
		return err
	}
	st.called = true
	// CONTRACT: within a turn the tool events are in the model's call order,
	// so appending in event order reproduces the assistant message's own
	// order.
	m := &b.rep.Messages[b.assistant]
	m.ToolCalls = append(m.ToolCalls, llm.ToolCall{ID: ev.CallID, Name: ev.Tool, Arguments: ev.Args})
	return nil
}

// toolResult appends the tool message that answers one call.
func (b *builder) toolResult(ev session.Event) error {
	st, err := b.expect(ev.CallID, eventToolResult)
	if err != nil {
		return err
	}
	// The loop logs the call before it dispatches it, so a result without one
	// is a log this package cannot rebuild: the call's arguments - which the
	// assistant message must carry - were never written.
	if !st.called {
		return fmt.Errorf("%w: a tool_result for id %q arrives before its tool_call", ErrMalformed, ev.CallID)
	}
	if st.answered {
		return fmt.Errorf("%w: a second tool_result for id %q", ErrMalformed, ev.CallID)
	}
	if err := gateClip("result", b.turn, ev.Result); err != nil {
		return err
	}
	st.answered = true
	b.rep.Messages = append(b.rep.Messages, llm.Message{
		Role: llm.RoleTool, ToolCallID: ev.CallID, Content: ev.Result,
	})
	return nil
}

// expect looks up a tool event's call in the open turn. An id the turn never
// requested is a dangling event: the history it would produce is one no
// provider accepts (a tool message answering nothing), so the read fails
// instead of dropping it quietly.
func (b *builder) expect(id, kind string) (*callState, error) {
	st, ok := b.calls[id]
	if !ok {
		return nil, fmt.Errorf("%w: %s for id %q, which turn %d did not request", ErrMalformed, kind, id, b.turn)
	}
	return st, nil
}

// closeTurn finishes the open turn: every call that was dispatched but never
// answered gets the synthetic tool message and is reported as Pending. Calls
// that were never dispatched are skipped - see the package comment for why
// they are dropped rather than invented.
//
// It runs before each new llm_response and once at the end of the file, which
// is what makes an interrupted run (no run_end at all) resumable.
func (b *builder) closeTurn() {
	for _, id := range b.expected {
		st := b.calls[id]
		if st == nil || !st.called || st.answered {
			continue
		}
		b.rep.Messages = append(b.rep.Messages, llm.Message{
			Role: llm.RoleTool, ToolCallID: id, Content: PendingResultMessage,
		})
		b.rep.Pending = append(b.rep.Pending, id)
	}
	b.expected, b.calls = nil, nil
}

// gateClip fails the read when a field the history needs was shortened by the
// log's byte bound.
//
// CONTRACT: the marker is session.ClipMarker, the exact string the writer
// appends, and the message names limits.max_logged_field: 0 because that is the
// only way to produce a resumable log. The task belongs to no turn and is
// reported as turn 0.
func gateClip(field string, turn int, value string) error {
	if !strings.HasSuffix(value, session.ClipMarker) {
		return nil
	}
	return fmt.Errorf("%w: %s in turn %d ends in the clip marker; write the log with limits.max_logged_field: 0 to make it resumable",
		ErrClipped, field, turn)
}
