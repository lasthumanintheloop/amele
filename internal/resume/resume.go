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
//   - validator_feedback (JSONL v1.10) is the user turn the output.schema
//     validator sent back after rejecting a final answer: it closes that turn
//     and appends a user message carrying the feedback, so the retry that
//     follows is rebuilt in the conversation it actually happened in.
//   - closing a turn (the next llm_response, or the end of the file) gives
//     every requested-and-dispatched call that never got a result a synthetic
//     tool message (PendingResultMessage) and lists it in Replay.Pending, so
//     the history a provider sees is complete without anything being re-run.
//   - every other event (mcp_*, provider_fallback, run_end) says nothing about
//     what the model saw and is ignored here; provider_fallback is read in a
//     pre-scan, for carriers only.
//
// # Chains
//
// A log written by a resumed run names the log it continued in
// run_start.resumed_from, and records the follow-up instruction that run was
// given in run_start.resumed_instruction (JSONL v1.11). Read follows that
// pointer: the named log's history is rebuilt first (recursively, up to
// MaxChainLinks), the instruction is appended as the user message it was, and
// this log's own turns follow - so a chain of resumes is one growing
// conversation rather than a chain of shortening ones (issue #31). Every link
// passes the same gates (schema version, clip marker, malformed events) and
// decides its own reasoning carriers (Options), so a link produced by another
// model keeps its carriers to itself while the links that match still get
// theirs back.
//
// Three things refuse a chain rather than guess at it. A link that cannot be
// read - missing, moved, or named through a path that redaction rewrote -
// because continuing from the first readable link would silently drop the
// conversation the missing one held, which is exactly what following the
// chain exists to avoid. A link written before v1.11, which does not say
// whether its resume was given an instruction (the key is absent, not empty),
// because the rebuilt history would either invent a user turn or skip one.
// And a parent that no longer ends where the link continued it - its turn
// count or its pending calls differ from what the link's run_start recorded -
// because the link's run saw the parent as it was then, and the file has
// since grown or been cut.
//
// A file whose LAST line is torn - a prefix of an event with no newline after
// it, which is what a SIGKILL mid-write leaves behind - ends there: the events
// before it are complete, and that is the position the resumed run continues
// from. A line that does not parse anywhere else is a damaged file
// (ErrMalformed), and so is a file whose only content is such a line: a torn
// tail ends a history, so with no complete event before it there is no history
// to end.
//
// Two llm_response events in a row where the first requested no tool calls and
// no validator_feedback sits between them mean a user turn happened that the
// log does not record - the shape every pre-v1.10 log of a schema retry has.
// Such a log is refused with ErrNotResumable rather than rebuilt into a
// conversation that never happened.
//
// A call the turn requested but never dispatched - no tool_call event, because
// the run died between the two writes - is DROPPED from the assistant message
// instead of being invented: its name and arguments were never written, so
// there is nothing faithful to send, and a call that never left the harness has
// no result to stand in for either. It is therefore not Pending; the model is
// free to ask again. When that drop leaves the turn with no text, no call and
// no reasoning carrier, the assistant message goes too: an empty assistant turn
// is not a message any provider takes (Anthropic answers 400 to the
// "content": null it becomes, Gemini discards it), so the history simply ends
// at the last turn that said something. LastTurn still counts the dropped turn:
// it reports where the old run got to, not what the rebuilt history kept.
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
// gate - the redacted text is the text of the run being continued. The one
// exception is a reasoning carrier: a provider verifies those bytes, so a
// carrier that contains "[REDACTED]" is dropped instead of being echoed back.
package resume

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"

	"github.com/lasthumanintheloop/amele/internal/ctxfile"
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
	// Links is how many session logs the history was rebuilt from: 1 for a
	// log that started from its own task, more for a log that continued an
	// earlier one (see Chains in the package comment). LastTurn counts the
	// turns of every link.
	Links int
}

// MaxChainLinks bounds how many logs a resume follows through resumed_from.
// It is a guard against a cycle in operator-edited files (a log copied over
// the one it points at), not a working limit: a chain that long has been
// resumed more times than any run should be.
const MaxChainLinks = 32

// Options tunes carrier restoration.
//
// CONTRACT: a reasoning payload comes back only when the log's run_start names
// BOTH the same provider identity and the same model as the run about to be
// made, and the old run never fell back. The backend signs or hash-checks the
// payload, and it signs it for the model that minted it - a thinking block
// belongs to that model, not merely to that vendor - so `--resume` with a
// changed model (`--model`, `--set model=`, an edited YAML) replays the
// conversation carrier-less rather than offering one model's signed reasoning
// to another.
type Options struct {
	// Provider is the current primary's identity (session.Event.Provider
	// spelling, e.g. "anthropic" or "openai/deepseek"). Reasoning payloads are
	// restored only when it equals the log's run_start.provider: a payload is
	// signed or hash-checked by the backend that produced it, so replaying one
	// into a different backend is at best rejected and at worst charged for.
	// Empty means "do not restore" - it is also what a pre-v1.8 log carries,
	// and two unknowns are not a match.
	Provider string
	// Model is the model the resumed run will call, after every override the
	// caller applied. It must equal the log's run_start.model for the same
	// reason Provider must match: the signature is over that model's own
	// output. Empty means "do not restore" for every log that names a model,
	// which is every log the writer produces.
	Model string
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
	// that is not JSON, a tool event whose id no turn requested, or two runs
	// concatenated into one file. A torn LAST line is the exception and not an
	// error at all - the prefix a hard kill leaves mid-write ends the history
	// there, which is the crashed run resume exists for - unless it is the
	// only content the file has: a torn tail needs a history to end, so a file
	// that is nothing but damage is reported as damage.
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

// redactedMarker is what session.SecretSet.Redact leaves in place of a secret.
// It is spelled here rather than imported because session offers no name for
// it; the two must not drift, which is why the resume tests pin the string.
const redactedMarker = "[REDACTED]"

// The event types this package understands. The rest are ignored by name, not
// by omission, so an added event type cannot silently change a replay.
const (
	eventRunStart          = "run_start"
	eventLLMResponse       = "llm_response"
	eventToolCall          = "tool_call"
	eventToolResult        = "tool_result"
	eventProviderFallback  = "provider_fallback"
	eventValidatorFeedback = "validator_feedback"
)

// Read parses the log at path and rebuilds the history, following the chain
// of logs it continues (see Chains in the package comment). Failures are one
// of ErrClipped, ErrNotResumable or ErrMalformed (match with errors.Is), each
// wrapped with the path of the link that failed so a message can be printed
// as-is - or the context's own error when ctx ends before a file is read: the
// logs are operator-named, and a path that blocks (a FIFO with no writer, a
// hung mount) must not hold the run past limits.timeout or a SIGTERM (issue
// #29).
func Read(ctx context.Context, path string, opts Options) (*Replay, error) {
	return readLink(ctx, path, opts, 1)
}

// readLink reads one log of the chain. depth is this link's 1-based position
// counted from the log the operator named; it is what MaxChainLinks bounds.
func readLink(ctx context.Context, path string, opts Options, depth int) (*Replay, error) {
	if depth > MaxChainLinks {
		return nil, fmt.Errorf("%w: the resume chain is longer than %d logs (a resumed_from cycle?)", ErrNotResumable, MaxChainLinks)
	}
	data, err := ctxfile.ReadFile(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("opening session log: %w", err)
	}
	link, err := readOne(bytes.NewReader(data), opts)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if link.from == "" {
		return &link.rep, nil
	}
	if link.instruction == nil {
		// Absent, not empty: a v1.9/v1.10 writer, which sent the instruction
		// but never recorded it. Whether one was given cannot be known here.
		return nil, fmt.Errorf("%w: %s continues %s but was written before JSONL v1.11 and does not record whether its resume was given an instruction; resume %s instead",
			ErrNotResumable, path, link.from, link.from)
	}
	// The path is followed exactly as the old run's operator typed it,
	// resolved against the current working directory like every other
	// operator path: it is a pointer the log recorded, not one this reader
	// invents. A parent that fails to read fails the whole resume - see the
	// package comment for why the first readable link is not a fallback.
	parent, err := readLink(ctx, link.from, opts, depth+1)
	if err != nil {
		return nil, fmt.Errorf("%s: following resumed_from: %w", path, err)
	}
	if err := link.continues(parent); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return chain(parent, link), nil
}

// ReadFrom rebuilds the history from an already-open log. It reads that log
// alone - with no path there is no chain to follow - and is the tested core
// of Read; callers with a file path want Read.
func ReadFrom(r io.Reader, opts Options) (*Replay, error) {
	link, err := readOne(r, opts)
	if err != nil {
		return nil, err
	}
	return &link.rep, nil
}

// link is one log of a chain as readOne yields it: its own replay, plus the
// two run_start facts that tie it to the log it continued.
type link struct {
	rep Replay
	// from is run_start.resumed_from - empty for a log that started from its
	// own task, which ends the chain.
	from string
	// instruction is run_start.resumed_instruction: the user message that
	// sat between the parent's history and this log's first turn. nil means
	// the key was absent - a pre-v1.11 writer - which is not the same as an
	// empty instruction (see session.Event.ResumedInstruction).
	instruction *string
	// turn and pending are run_start.resumed_turn and resumed_pending: what
	// the parent looked like when this log's run continued it, checked
	// against the parent as it reads now (continues).
	turn    int
	pending []string
	// opened is whether the log holds at least one llm_response; it decides
	// whose Completed verdict a chain takes (see chain).
	opened bool
}

// continues checks that parent still ends where this link's run continued it.
//
// CONTRACT: the link's run sent the parent's history as the parent was THEN
// - so many turns, these pending calls - and recorded both on its run_start.
// A parent that has since grown (its run went on after the resume, or a
// pending call was answered) or been cut (a torn tail, a cleanup) would put
// turns into the rebuilt chain that the link's run never saw, or take away
// ones it did. Two numbers and a list are not a byte digest, but they are the
// facts the v1.9 run_start already records, and every way a log can change
// moves at least one of them.
func (l *link) continues(parent *Replay) error {
	if parent.LastTurn != l.turn {
		return fmt.Errorf("%w: %s no longer ends where this log continued it (turn %d then, turn %d now)",
			ErrNotResumable, l.from, l.turn, parent.LastTurn)
	}
	if !slices.Equal(parent.Pending, l.pending) {
		return fmt.Errorf("%w: %s no longer has the pending tool calls this log continued from (%v then, %v now)",
			ErrNotResumable, l.from, l.pending, parent.Pending)
	}
	return nil
}

// readOne decodes and rebuilds a single log.
func readOne(r io.Reader, opts Options) (*link, error) {
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
	b.rep.Links = 1
	start := events[0]
	return &link{
		rep: b.rep, from: start.ResumedFrom, instruction: start.ResumedInstruction,
		turn: start.ResumedTurn, pending: start.ResumedPending, opened: b.opened,
	}, nil
}

// chain appends one link's conversation to the history of the log it
// continued, reproducing what that link's run actually sent: the parent's
// rebuilt history (including the synthetic results of its pending calls -
// the same ones the link's run stood in), then the instruction as the last
// user message when there was one, then the link's own turns. The link's own
// task message is dropped: it is the parent's task, copied.
//
// The facts of the newest link win where only one log can answer: Model and
// Provider are the link's (the last backend the conversation ran on), Pending
// is the link's alone (the parent's pending calls were answered - with the
// stand-in - in the link's run, and the link's run_start already lists them),
// and Completed is the link's verdict when it produced a turn at all; a link
// that died before its first turn hands the question back to the parent,
// unless it added an instruction nobody has answered yet.
func chain(parent *Replay, l *link) *Replay {
	rep := l.rep
	rep.Task = parent.Task
	rep.Links = parent.Links + 1
	rep.LastTurn += parent.LastTurn
	rep.Carriers = rep.Carriers || parent.Carriers
	msgs := make([]llm.Message, 0, len(parent.Messages)+1+len(l.rep.Messages))
	msgs = append(msgs, parent.Messages...)
	// readLink refused a nil instruction before getting here.
	instruction := *l.instruction
	if instruction != "" {
		msgs = append(msgs, llm.Message{Role: llm.RoleUser, Content: instruction})
	}
	msgs = append(msgs, l.rep.Messages[1:]...)
	rep.Messages = msgs
	if !l.opened {
		rep.Completed = parent.Completed && instruction == ""
	}
	return &rep
}

// decode reads the whole log into memory, one JSONL line at a time.
//
// Whole, because two passes are needed anyway (a provider_fallback anywhere in
// the file disables carriers for the turns before it), and a session log is
// bounded by the run's own token and tool-output budgets. Lines rather than a
// json.Decoder stream: the torn-line rule below is a statement about POSITION
// in the file, and only a line split can tell "the last thing in the file" from
// "something in the middle of it". Splitting on the newline is safe for JSONL -
// a JSON string cannot contain a literal newline - and there is no line-buffer
// cap to trip over, which matters because a full-record log
// (limits.max_logged_field: 0) is exactly the log resume exists for.
//
// A torn LAST line is end of file, not corruption: the writer emits one line
// per Write, so a SIGKILL or a power loss can leave a prefix of the final event
// on disk with no newline after it. That is rule (d) of the resume design - the
// history ends at the last complete position - and refusing there would refuse
// precisely the crashed runs resume is for. Anywhere else a line that does not
// parse is a damaged file, and damage is never resumed from silently.
func decode(r io.Reader) ([]session.Event, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("reading session log: %w", err)
	}
	lines := bytes.Split(data, []byte("\n"))
	var events []session.Event
	for i, line := range lines {
		// A complete file ends in a newline, so the final split element is
		// usually empty; a blank line anywhere else is likewise nothing to
		// decode.
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var ev session.Event
		if err := json.Unmarshal(line, &ev); err != nil {
			if i == len(lines)-1 && len(events) > 0 {
				// Nothing follows it, not even a newline: a torn tail.
				return events, nil
			}
			// A torn tail ENDS a history, so it needs a history to end. With
			// nothing complete before it the file is not a cut-short log at
			// all - it is a file that is not a log - and reporting "the log is
			// empty" there would send the operator looking for a missing run
			// instead of at the damaged bytes on line 1.
			return nil, fmt.Errorf("%w: line %d: %v", ErrMalformed, i+1, err)
		}
		events = append(events, ev)
	}
	return events, nil
}

// callState tracks one requested tool call across the events of its turn.
type callState struct {
	// call is the reproduced tool call, filled in from the tool_call event.
	call llm.ToolCall
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
	// assistant indexes the open turn's assistant message in rep.Messages, or
	// -1 when there is none to patch - before the first turn, and after
	// closeTurn dropped an empty one.
	assistant int
	// opened records that an llm_response was seen at all, which is a
	// statement about the FILE rather than about the rebuilt history: the
	// skipped-user-turn rule in llmResponse reads the sequence of turns in the
	// log, so a turn closeTurn dropped still counts as one that happened.
	opened bool
	// answered records that the open turn's final answer was followed by a
	// validator_feedback event, which is the one thing that makes a second
	// final-shaped llm_response legitimate (see llmResponse).
	answered bool
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
	if start.ResumedInstruction != nil {
		if err := gateClip("instruction", 0, *start.ResumedInstruction); err != nil {
			return nil, err
		}
	}
	b := &builder{assistant: -1}
	b.rep.Task, b.rep.Model, b.rep.Provider = start.Task, start.Model, start.Provider
	b.rep.Messages = []llm.Message{{Role: llm.RoleUser, Content: start.Task}}
	// Four conditions, all necessary: an identity to compare, the SAME
	// identity, the same MODEL behind it, and a run that never moved off
	// either. A fallback anywhere in the file taints the whole replay because
	// the history is sent as one document - the surviving payloads belong to a
	// backend (and a model) that is no longer the one being talked to.
	b.carriers = opts.Provider != "" && opts.Provider == start.Provider &&
		opts.Model != "" && opts.Model == start.Model && !hasFallback(events)
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
	case eventValidatorFeedback:
		return b.validatorFeedback(ev)
	default:
		// Everything else is ignored, by design and without inspection: the
		// event types this reader knows to skip (mcp_connect,
		// mcp_tools_listed, mcp_disconnect, provider_fallback, run_end)
		// record what the HARNESS did rather than what the model was shown,
		// and an event type added by a newer writer within schema v1 is
		// additive by contract, so ignoring it is the forward-compatible
		// reading too. provider_fallback is not lost here - hasFallback read
		// it before the walk.
		return nil
	}
}

// llmResponse closes the previous turn and opens a new assistant message.
func (b *builder) llmResponse(ev session.Event) error {
	// CONTRACT: a turn with no tool calls is a final answer, so a second
	// llm_response after one means SOMETHING was said to the model in
	// between - the output.schema validator's feedback, which a v1.10 log
	// records as a validator_feedback event and an older log does not.
	// Rebuilding without it would hand the model a conversation it never had
	// (and, for a signed-reasoning provider, one that no longer matches its
	// own carrier), so the older log is refused rather than guessed at.
	if b.opened && len(b.expected) == 0 && !b.answered {
		return fmt.Errorf("%w: the log skips a user turn (an output.schema retry's feedback is not logged before JSONL v1.10); the run is not resumable",
			ErrNotResumable)
	}
	b.closeTurn()
	b.answered = false
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
	b.opened = true
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
	// SECURITY-driven, correctness-visible: the log's redactor rewrote a
	// secret inside this payload, so the bytes on disk are no longer the bytes
	// the provider signed or hashed. Echoing them back fails that check (a 400
	// for the whole request), and un-redacting is not a thing anyone should
	// want, so the turn is simply rebuilt without a carrier - the conversation
	// is intact, only the thinking payload is gone.
	if strings.Contains(ev.Reasoning, redactedMarker) {
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

// validatorFeedback closes the rejected final answer's turn and appends the
// feedback as the user message it was.
//
// CONTRACT: the event belongs to a final-shaped turn (no tool calls) and comes
// once per turn, on that turn's number - anything else is a file the writer
// never produces, and rebuilding it would put a user message where no provider
// accepts one. The turn is closed BEFORE the message is appended so the
// empty-assistant drop (closeTurn) cannot truncate the feedback away with the
// answer it follows. Completed is cleared: the answer this feedback rejects is
// not one the run finished on, so a log that ends here continues without a new
// instruction - the model's next move is to answer the feedback.
func (b *builder) validatorFeedback(ev session.Event) error {
	if !b.opened || len(b.expected) != 0 {
		return fmt.Errorf("%w: validator_feedback for turn %d, which is not a final answer", ErrMalformed, ev.Turn)
	}
	if ev.Turn != b.turn {
		return fmt.Errorf("%w: validator_feedback for turn %d while turn %d is open", ErrMalformed, ev.Turn, b.turn)
	}
	if b.answered {
		return fmt.Errorf("%w: a second validator_feedback for turn %d", ErrMalformed, ev.Turn)
	}
	if err := gateClip("feedback", ev.Turn, ev.Content); err != nil {
		return err
	}
	b.closeTurn()
	b.rep.Messages = append(b.rep.Messages, llm.Message{Role: llm.RoleUser, Content: ev.Content})
	b.answered = true
	b.rep.Completed = false
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
	st.call = llm.ToolCall{ID: ev.CallID, Name: ev.Tool, Arguments: ev.Args}
	// The call is NOT appended to the assistant message here: closeTurn puts
	// the turn's calls back in tool_call_ids order, which is the model's own
	// order by definition. The events are in that order too (the ordering
	// guarantees), so this is the same result made structural instead of
	// dependent on a promise about the file.
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
// they are dropped rather than invented - and a turn left with nothing at all
// by that drop is itself dropped.
//
// It runs before each new llm_response, on a validator_feedback, and once at
// the end of the file, which is what makes an interrupted run (no run_end at
// all) resumable. It is idempotent: a turn closed by the feedback event is not
// patched again when the retry opens the next one.
func (b *builder) closeTurn() {
	if b.assistant >= 0 {
		// CONTRACT: tool_call_ids IS the model's call order, so rebuilding
		// from it needs no assumption about the order the events were written
		// in. Calls with no tool_call event are skipped here - that is the
		// drop rule. Left nil when the turn dispatched nothing, so a
		// tool-less turn keeps a nil slice.
		var calls []llm.ToolCall
		dropped := false
		for _, id := range b.expected {
			if st := b.calls[id]; st != nil && st.called {
				calls = append(calls, st.call)
			} else {
				dropped = true
			}
		}
		b.rep.Messages[b.assistant].ToolCalls = calls
		if dropped {
			// The carrier is the turn's raw content array, and that array
			// still announces the tool_use blocks the drop rule just removed.
			// Echoed verbatim it would rebuild exactly the announced-but-
			// unanswered history the rule exists to avoid (a 400 on the
			// wires that verify it), so the turn loses its carrier with the
			// call and is replayed from its text alone - or, with no text,
			// not at all (dropEmptyAssistant).
			b.rep.Messages[b.assistant].Reasoning = nil
		}
		b.dropEmptyAssistant()
		b.assistant = -1
	}
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

// dropEmptyAssistant removes the closed turn's assistant message when the drop
// rule left nothing in it: no text, no tool call, no reasoning carrier.
//
// CONTRACT: this completes the never-dispatched drop. A message with all three
// empty is not something a provider takes - the Anthropic wire would receive
// "content": null and answer 400, and the Gemini wire drops such a turn on the
// floor anyway - so the harness must not build one. It is the shape a crash
// between the llm_response line and its tool_call lines leaves behind (also a
// torn last tool_call, and an empty final answer): the calls were announced but
// never written, so there is nothing faithful to send and the model is free to
// ask again.
//
// The message is always the LAST one at this point: a turn reaches here empty
// only when no call was dispatched, and no tool message can exist for a call
// that was not (toolResult refuses a result before its tool_call), so nothing
// is appended after it that this truncation could take with it.
func (b *builder) dropEmptyAssistant() {
	msg := b.rep.Messages[b.assistant]
	if msg.Content != "" || len(msg.ToolCalls) > 0 || len(msg.Reasoning) > 0 {
		return
	}
	b.rep.Messages = b.rep.Messages[:b.assistant]
	// Nothing left to patch; the turn itself is still counted (b.opened).
	b.assistant = -1
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
