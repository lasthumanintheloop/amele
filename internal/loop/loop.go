// Package loop implements the agent turn loop: send the conversation to the
// provider, dispatch requested tools, feed results back, and stop when the
// model produces a final answer - or when a budget kill switch fires.
//
// Turns are strictly sequential. Within ONE turn the tool calls a model asked
// for may run concurrently (ParallelTools), but every observable effect - the
// session events, the message history, the progress feed - is still produced in
// the model's original call order, which is what keeps the append-only session
// log a faithful replay source (docs/contracts/jsonl-events.md).
package loop

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/lasthumanintheloop/amele/internal/llm"
	"github.com/lasthumanintheloop/amele/internal/session"
	"github.com/lasthumanintheloop/amele/internal/tools"
)

// Sentinel errors mapped to the frozen exit code contract (docs/engineering.md §7).
var (
	// ErrBudgetExceeded fires when max_turns, max_tokens or the run
	// timeout is hit. CONTRACT: exit code 3.
	ErrBudgetExceeded = errors.New("budget exceeded")
	// ErrPermissionDenied fires when the approval policy rejects a tool
	// call. CONTRACT: exit code 4.
	ErrPermissionDenied = errors.New("permission denied")
	// ErrOutputRejected fires when FinalValidator kept rejecting the model's
	// final answer past MaxFinalRejections. CONTRACT: exit code 6 (output
	// schema could not be satisfied).
	ErrOutputRejected = errors.New("output rejected")
)

// Decision is an Approver's ruling on one tool call.
type Decision int

// Approver decisions. The zero value is Allow so a forgotten return keeps
// Phase 1's allow-all behavior rather than silently blocking runs.
const (
	// Allow permits the call.
	Allow Decision = iota
	// DenyContinue rejects the call but keeps the run alive: the model sees
	// a "permission denied" tool result and can adapt (the Phase 2 "ask →
	// user said no" path).
	DenyContinue
	// DenyAbort rejects the call and aborts the run with ErrPermissionDenied.
	// CONTRACT: exit code 4.
	DenyAbort
)

// DenialReason explains WHY an approver refused a call. It is carried
// separately from the Decision because the two answer different questions: the
// decision says what the loop must DO, the reason says what an operator reading
// the session log afterwards must be able to reconstruct - "the profile says
// deny", "nobody was at the terminal", "a human said no" are three different
// incidents with three different fixes.
//
// CONTRACT: the reasons are serialized into the tool_result event's `outcome`
// field (docs/contracts/jsonl-events.md). Adding one means adding an enum value
// there.
type DenialReason int

// Denial reasons. DeniedByPolicy is the zero value because it is the
// unremarkable case: an approver that refuses without elaborating refused by
// policy, which is what every denial before the ask profiles existed was.
const (
	// DeniedByPolicy means a configured rule refused the call outright.
	DeniedByPolicy DenialReason = iota
	// DeniedNoTTY means an "ask" policy could not be answered because no
	// terminal was attached, so it fell closed (the headless fail-safe).
	DeniedNoTTY
	// DeniedAskRefused means a human was asked and said no.
	DeniedAskRefused
)

// Ruling is an Approver's answer: what to do with the call, plus why when the
// answer is a denial. The zero value is "allow, no reason", so a policy that
// forgets to fill it in keeps the documented allow-all default rather than
// silently blocking runs.
type Ruling struct {
	// Decision is what the loop must do.
	Decision Decision
	// Reason explains a denial and is ignored for Allow.
	Reason DenialReason
}

// Approver decides whether a tool call may proceed. It receives the full call
// so policies can rule on arguments (e.g. fs_write paths), not just the tool
// name. A returned error means the policy itself failed and aborts the run -
// a broken policy must never fail open. Phase 1 ships allow-all; the
// TTY-aware ask/deny profiles land in Phase 2 on top of this same hook.
type Approver func(ctx context.Context, call llm.ToolCall) (Ruling, error)

// Limits are the loop budgets, already defaulted by the config package.
type Limits struct {
	// MaxTurns bounds provider round-trips (> 0 after defaulting).
	MaxTurns int
	// MaxTokens bounds cumulative provider-reported tokens; 0 disables.
	MaxTokens int
}

// Clock supplies time for duration accounting; injected per docs/engineering.md §5.4.
type Clock func() time.Time

// Tuning carries the provider-neutral request knobs a config sets once and
// every turn repeats: reasoning depth, sampling, the output cap and the raw
// params escape hatch. The loop does not interpret any of them - it forwards
// them verbatim on every round-trip and lets the client map them to its wire.
//
// CONTRACT: the knobs ride on EVERY request, not only the first. A tool turn
// that dropped the reasoning knob would think shallower exactly where a task
// needs it most, and a dropped cap would leave the rest of the run uncapped.
//
// The zero value asks for nothing, which is the pre-tuning behavior: every
// field stays at the provider's own default.
type Tuning struct {
	// MaxOutputTokens caps the tokens the provider may generate per call.
	// Zero sends no cap.
	MaxOutputTokens int
	// Reasoning is the neutral thinking knob; nil sends none.
	Reasoning *llm.ReasoningSpec
	// Temperature and TopP are the sampling knobs, pointers so an explicit 0
	// stays distinguishable from "unset".
	Temperature *float64
	TopP        *float64
	// Extra is provider.params, already serialized to JSON by the caller: the
	// loop never re-encodes user YAML, and never inspects these keys.
	Extra map[string]json.RawMessage
}

// Loop wires one run's collaborators together.
type Loop struct {
	Provider llm.Provider
	Registry *tools.Registry
	// Session may be nil (logging disabled); *session.Writer is nil-safe.
	Session *session.Writer
	Limits  Limits
	// Approve may be nil, meaning allow-all.
	Approve Approver
	// AutoApprove reports whether a call's permission policy resolves to a
	// plain "allow" - a ruling that needs no human and produces no side effect
	// of its own. It exists ONLY as the gate on concurrent dispatch: a call
	// that might stop to ask somebody must never run beside another call.
	//
	// SECURITY: nil means "the caller did not say", and an unknown policy is
	// never assumed safe - with an Approve hook installed and no predicate the
	// loop stays sequential. It is ignored entirely when Approve is nil, which
	// is already allow-all.
	AutoApprove func(call llm.ToolCall) bool
	// ParallelTools enables concurrent dispatch of the tool calls within one
	// turn. It is only ever an upper bound: see parallelizable for the three
	// conditions that must all hold before a turn actually runs concurrently.
	// The zero value (false) keeps the pre-v0.2 strictly sequential loop.
	ParallelTools bool
	// MaxToolResultBytes is the ceiling on the text of any one tool result as
	// the model receives it, whatever family produced it. Each family caps its
	// own output already; this bound exists because the subprocess framing
	// (stdout plus stderr plus status lines) could otherwise hand the model
	// twice the per-stream cap. 0 means no ceiling - the pre-v0.3 behavior.
	//
	// CONTRACT: limits.max_tool_result_bytes; a cut here is reported as
	// tool_result.truncated exactly like a cut inside the tool.
	MaxToolResultBytes int
	// Clock may be nil, meaning time.Now.
	//
	// CONCURRENCY: with ParallelTools enabled AND a Progress hook installed,
	// the clock is read from the per-call goroutines (it times each tool
	// invocation), so an injected Clock must be safe for concurrent use.
	// time.Now is; a test clock counting its reads needs a mutex.
	Clock Clock

	// FinalValidator may be nil, meaning every clean final answer is
	// accepted (the Phase 1 behavior). When set, it is consulted on final
	// answers only - a turn with no tool calls that badFinish already judged
	// clean. Returning ok=false rejects the answer: the assistant message
	// stays in the history, feedback is appended as a user message, and the
	// loop asks the model again. This is the hook structured output
	// (output.schema) rides on; the loop itself stays schema-agnostic.
	FinalValidator func(text string) (feedback string, ok bool)
	// MaxFinalRejections caps how many feedback retries a rejected final
	// answer may buy: N means the model is sent repair feedback at most N
	// times, and the rejection after the last granted retry ends the run with
	// ErrOutputRejected (so N rejections are survivable, the N+1-th is fatal).
	//
	// Rejections are counted as a TOTAL over the whole run, not as a
	// consecutive streak: a model that interleaves tool calls with unusable
	// final answers stays bounded by the same cap.
	//
	// 0 means "no cap of its own" - not unbounded, because retries burn the
	// ordinary turn and token budgets. Callers that set FinalValidator should
	// set a positive value.
	MaxFinalRejections int

	// Progress, when non-nil, receives one short line per interesting event:
	// a tool call being dispatched, its result, and the acceptance of a final
	// answer. It exists so a caller can show live progress (the CLI's -v) or
	// feed a monitor, without the loop knowing anything about terminals: the
	// event carries no prefix, no color and no trailing newline, and the
	// consumer decides how - and whether - to render it. nil is a no-op.
	//
	// Events are informational only: nothing the hook does changes the run,
	// and a nil hook produces byte-identical behavior. The session log stays
	// the record of what happened (docs/contracts/jsonl-events.md); this is a progress feed.
	//
	// CONCURRENCY: Progress is called synchronously and sequentially from the
	// goroutine running RunMessages, in event order, and never concurrently
	// with itself - so an implementation needs no locking of its own. It does
	// need to be quick: the loop is blocked for the duration of the call.
	// This holds for concurrent tool dispatch too: the per-call goroutines
	// never touch the hook, so a parallel turn reports its calls in the
	// model's order - all "model requested" lines first, then all results,
	// once the slowest call has returned.
	//
	// SECURITY: event text embeds model-controlled strings (tool names, tool
	// arguments, tool error text). The loop clips them so one event cannot
	// carry a huge payload, but it does NOT make them terminal-safe - a
	// consumer writing events to a terminal must strip control bytes itself
	// (cmd/amele routes every event through safeForTerminal).
	Progress func(event string)

	// ResponseFormat, when non-nil, is forwarded verbatim on every provider
	// request so providers with native structured output constrain decoding.
	// The loop stays schema-agnostic: it never inspects or validates the
	// schema - FinalValidator is what actually enforces the contract, and
	// this field is only an optimization that makes the model right the first
	// time more often.
	ResponseFormat *llm.ResponseFormat

	// Tuning holds the provider-neutral request knobs forwarded on every
	// round-trip. The zero value sends nothing.
	Tuning Tuning

	// TurnBase is added to the turn number recorded in session events. It
	// exists for callers that drive several RunMessages calls inside ONE
	// run_start/run_end pair (chat): without it every call would restart the
	// logged turn at 1, and the frozen JSONL contract requires `turn` to be
	// monotonic within a run and run_end.turns to equal the highest turn
	// logged. Budgets are unaffected - MaxTurns still counts the turns of the
	// current call, so the caller subtracts what it already spent instead.
	// Zero (the one-shot default) means the log is numbered from 1.
	TurnBase int

	Model        string
	SystemPrompt string
}

// Result summarizes a completed (or aborted) run.
type Result struct {
	// FinalText is the model's last final-shaped answer (a tool-call-free
	// turn), regardless of whether the run ultimately succeeded: it is set
	// on the ErrOutputRejected path (the last answer the validator kept
	// rejecting) and on the badFinish failure path (a clean-looking but
	// truncated/filtered/errored/empty turn), not only on success. It stays
	// empty when no such turn was ever produced - e.g. a budget or
	// permission failure that aborted the run mid-tool-call.
	FinalText string
	Turns     int
	ToolCalls int
	Usage     llm.Usage
	Duration  time.Duration
	// SchemaEnforcementDropped is true when ANY provider response in the run
	// was produced without native schema enforcement despite ResponseFormat
	// being set (see llm.Response.SchemaEnforcementDropped). The loop only
	// aggregates the flag; cmd decides whether and how to warn the operator.
	SchemaEnforcementDropped bool
}

// Run executes the loop for one user task: it builds the opening history
// (system prompt, if any, plus the task as a user message), logs run_start,
// and hands off to RunMessages. On error the returned Result still carries
// the accounting collected so far, so callers can log a truthful run_end even
// for failed runs.
func (l *Loop) Run(ctx context.Context, task string) (*Result, error) {
	messages := make([]llm.Message, 0, 2)
	if l.SystemPrompt != "" {
		messages = append(messages, llm.Message{Role: llm.RoleSystem, Content: l.SystemPrompt})
	}
	messages = append(messages, llm.Message{Role: llm.RoleUser, Content: task})

	// run_start lives here rather than in RunMessages because only this entry
	// point knows the task string; callers driving their own history (chat,
	// resume) log their own opening event.
	l.Session.RunStart(l.Model, task)

	return l.RunMessages(ctx, messages)
}

// RunMessages is the loop core: it drives an already-built conversation to a
// final answer. The history is sent verbatim - RunMessages adds no system
// prompt and no framing of its own, so callers that own the conversation
// (multi-turn chat, replay, validator retries) decide exactly what the model
// sees. The caller's slice is never mutated.
//
// On error the returned Result still carries the accounting collected so far.
func (l *Loop) RunMessages(ctx context.Context, history []llm.Message) (*Result, error) {
	start := l.now()

	res := &Result{}
	finish := func(err error) (*Result, error) {
		res.Duration = l.now().Sub(start)
		return res, err
	}

	// Copy: the loop appends assistant/tool/feedback messages as it goes and
	// must not write into the caller's backing array.
	messages := slices.Clone(history)

	// rejections counts FinalValidator refusals over the whole run, not just
	// consecutive ones: a model that alternates tool calls with unusable
	// answers must stay bounded too.
	rejections := 0

	for turn := 1; ; turn++ {
		// CONTRACT: max_turns counts provider round-trips; exceeding it is
		// a budget failure (exit 3), not a normal stop - a cron job must
		// notice that its agent spun without converging.
		if turn > l.Limits.MaxTurns {
			return finish(fmt.Errorf("%w: max_turns (%d) reached without a final answer", ErrBudgetExceeded, l.Limits.MaxTurns))
		}
		// The run timeout arrives via ctx; check before an expensive call.
		if err := ctx.Err(); err != nil {
			return finish(wrapContextErr(err))
		}

		// The attempt counts as a turn even if it fails: run_end must never
		// claim fewer provider round-trips than were actually made.
		res.Turns = turn

		resp, err := l.Provider.Chat(ctx, llm.Request{
			Model:          l.Model,
			Messages:       messages,
			Tools:          l.Registry.Defs(),
			ResponseFormat: l.ResponseFormat,
			// The tuning knobs are repeated on every turn by design - see
			// Tuning's contract note.
			MaxOutputTokens: l.Tuning.MaxOutputTokens,
			Reasoning:       l.Tuning.Reasoning,
			Temperature:     l.Tuning.Temperature,
			TopP:            l.Tuning.TopP,
			Extra:           l.Tuning.Extra,
		})
		if err != nil {
			// A provider failure caused by our own context ending is not a
			// provider outage - map it to the context cause instead.
			if ctxErr := ctx.Err(); ctxErr != nil {
				return finish(wrapContextErr(ctxErr))
			}
			return finish(err)
		}

		// Saturating add, not +=: a provider reporting near-max token counts
		// must never wrap the accumulator negative, which checkTokenBudget
		// would read as "under budget" and let the run spend unbounded.
		res.Usage = res.Usage.Add(resp.Usage)
		// Sticky OR: one unenforced response is enough to make the run's
		// output contract validate+retry-only, so the flag never resets.
		if resp.SchemaEnforcementDropped {
			res.SchemaEnforcementDropped = true
		}
		// The logged number is offset by TurnBase so a multi-call session
		// (chat) keeps counting up; the loop's own budget still uses `turn`.
		// The reasoning payload's SIZE is always logged (JSONL v1.4); its
		// content is handed over too, but the writer drops it unless the
		// operator opted in with log_reasoning (session.Options.LogReasoning).
		// This string is a copy made for the LOG only - the echo path keeps
		// resp.Message.Reasoning's original bytes, which dialects that
		// hash-check their reasoning require back unaltered.
		l.Session.LLMResponse(session.LLMResponse{
			Turn: l.TurnBase + turn, Content: resp.Message.Content,
			ToolCallIDs: toolCallIDs(resp.Message.ToolCalls),
			InputTokens: resp.Usage.InputTokens, OutputTokens: resp.Usage.OutputTokens,
			FinishReason: resp.FinishReason, ReasoningBytes: len(resp.Message.Reasoning),
			Reasoning: string(resp.Message.Reasoning),
		})

		if err := l.checkTokenBudget(res, resp); err != nil {
			return finish(err)
		}

		messages = append(messages, resp.Message)

		if len(resp.Message.ToolCalls) == 0 {
			// A turn with no tool calls is a final answer only if the model
			// actually finished cleanly. Anything else - truncation, a
			// filter, an upstream error (OpenRouter reports mid-generation
			// failures as finish_reason "error"), or an empty body under a
			// non-stop reason - means the task did NOT complete. A cron
			// consumer must never mistake a half or failed answer for
			// success. CONTRACT: these map to exit 1 (task failed), not 0.
			res.FinalText = resp.Message.Content
			if reason := badFinish(resp); reason != "" {
				return finish(fmt.Errorf("model did not complete the task: %s", reason))
			}
			// The answer is a clean final; only now may the validator judge
			// its content. Validating a truncated or filtered turn would send
			// the model repair feedback for a failure it cannot repair.
			feedback, accepted, err := l.validateFinal(resp.Message.Content, rejections)
			if err != nil {
				return finish(err)
			}
			if !accepted {
				// The rejected assistant message stays in the history (it was
				// appended above) so the model can see and repair its own
				// output; the feedback arrives as the next user turn.
				// CONTRACT: retries are ordinary turns - already counted in
				// res.Turns with their tokens accounted, so max_turns and
				// max_tokens still bound a model that never converges.
				rejections++
				messages = append(messages, llm.Message{Role: llm.RoleUser, Content: feedback})
				continue
			}
			// The answer is final and accepted: the run is over. The token
			// count is this turn's OUTPUT tokens - how much the model spent
			// saying it - because the run total is already the closing summary's
			// job and a progress line that repeated it would say nothing new.
			l.progressf("turn %d: final answer (%d tokens)", l.TurnBase+turn, resp.Usage.OutputTokens)
			return finish(nil)
		}

		// Outputs come back positionally: whether the calls ran one after the
		// other or side by side, outputs[i] answers ToolCalls[i], and a run
		// that aborted mid-turn returns the outputs it managed to produce.
		outputs, published, err := l.dispatchCalls(ctx, l.TurnBase+turn, resp.Message.ToolCalls)
		// CONTRACT: run_end.tool_calls counts what the LOG says ran, not what
		// the model was told about. On the abort path those two diverge: a call
		// that was already in flight is published and then withheld from the
		// history, and counting the history undercounted the tool_result events
		// sitting in the same file.
		res.ToolCalls += published
		for i, output := range outputs {
			messages = append(messages, llm.Message{
				Role:       llm.RoleTool,
				Content:    output,
				ToolCallID: resp.Message.ToolCalls[i].ID,
			})
		}
		if err != nil {
			// Approval denial aborts the run (conservative Phase 1
			// stance); everything else has already been converted to
			// model-visible text by dispatch.
			return finish(err)
		}
	}
}

// validateFinal consults FinalValidator on a clean final answer. It reports
// the feedback to hand back to the model and whether the answer was accepted;
// with no validator configured every answer is accepted. rejections is the
// number of refusals *before* this one, so the (MaxFinalRejections+1)-th
// refusal returns ErrOutputRejected instead of another retry.
func (l *Loop) validateFinal(text string, rejections int) (feedback string, accepted bool, err error) {
	if l.FinalValidator == nil {
		return "", true, nil
	}
	feedback, ok := l.FinalValidator(text)
	if ok {
		return "", true, nil
	}
	// CONTRACT: exhausting the rejection budget is exit 6 - the run produced
	// an answer, but never one the caller's contract could accept.
	if l.MaxFinalRejections > 0 && rejections >= l.MaxFinalRejections {
		return "", false, fmt.Errorf("%w: the final answer was rejected %d times (limit %d): %s",
			ErrOutputRejected, rejections+1, l.MaxFinalRejections, feedback)
	}
	if feedback == "" {
		// A refusal with no feedback is a caller bug, but an empty user
		// message is rejected outright by some OpenAI-compatible endpoints -
		// send something the model can act on instead.
		feedback = "your previous answer was rejected; produce a corrected final answer"
	}
	return feedback, false, nil
}

// badFinish reports why a tool-call-free turn is not a valid completion, or
// "" if it is a clean final answer. Known-good finish reasons are "stop" and
// "" (some OpenAI-compatible providers omit it); everything else, or an empty
// body under a non-stop reason, indicates the model did not finish the task.
func badFinish(resp *llm.Response) string {
	switch resp.FinishReason {
	case "length":
		return "output was truncated at the token limit (finish_reason: length)"
	case "content_filter":
		return "response was blocked by a content filter (finish_reason: content_filter)"
	case "error":
		return "the provider reported an error mid-generation (finish_reason: error)"
	case "stop", "":
		if resp.Message.Content == "" {
			return "the model returned an empty final answer"
		}
		return ""
	default:
		// Unknown reason with no content is suspicious; with content we
		// accept it (providers use various custom reasons for normal stops).
		if resp.Message.Content == "" {
			return fmt.Sprintf("the model returned no answer (finish_reason: %s)", resp.FinishReason)
		}
		return ""
	}
}

// checkTokenBudget enforces limits.max_tokens against cumulative usage.
// CONTRACT: the budget uses provider-reported totals (docs/engineering.md §7) and
// fails closed - an endpoint that reports no usage cannot be budgeted, so
// the run stops rather than spending unbounded. Checked after accounting so
// the log shows the overshooting turn.
func (l *Loop) checkTokenBudget(res *Result, resp *llm.Response) error {
	if l.Limits.MaxTokens <= 0 {
		return nil
	}
	if resp.UsageMissing {
		return fmt.Errorf("%w: limits.max_tokens is set but the provider did not report token usage; remove the limit or use an endpoint that reports usage", ErrBudgetExceeded)
	}
	// Budget check runs after the response arrives: it can overshoot by one
	// turn, deliberately (threat-model §5).
	if res.Usage.Total() > l.Limits.MaxTokens {
		return fmt.Errorf("%w: max_tokens (%d) exceeded: %d used", ErrBudgetExceeded, l.Limits.MaxTokens, res.Usage.Total())
	}
	return nil
}

// wrapContextErr maps a context error onto the exit-code contract: only a
// deadline (the configured run timeout) is a budget kill (exit 3); a
// cancellation (SIGINT/SIGTERM) is an interrupted task (exit 1).
// CONTRACT: exit 3 is reserved for configured budgets - an operator pressing
// Ctrl-C must not look like a budget overrun in monitoring.
func wrapContextErr(err error) error {
	if errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("%w: %v", ErrBudgetExceeded, err)
	}
	return fmt.Errorf("run interrupted: %w", err)
}

// callResult is one dispatched call's outcome, held until the dispatcher is
// ready to publish it.
//
// It exists so that RUNNING a call and REPORTING it are two separate steps:
// concurrent calls finish in whatever order their tools happen to return, but
// the session events, the progress feed and the message history must still be
// produced in the model's original call order (docs/contracts/jsonl-events.md).
// The sequential path builds and publishes one of these immediately, so both
// paths emit byte-identical output for a single call.
type callResult struct {
	// output is the text handed back to the model. Empty when fail is set:
	// a call that aborts the run never reaches the history.
	output string
	// logged is the tool_result event's text, which differs from output on the
	// abort paths (the model never sees them, an operator must).
	logged string
	// isErr fills the frozen is_error field: a harness dispatch failure.
	isErr    bool
	outcome  session.ToolOutcome
	exitCode *int
	// truncated reports that the text the model receives is short of what the
	// tool produced - the tool cut it, the run's ceiling cut it, or both. The
	// dispatch paths that never ran a tool (denial, unknown tool, error) leave
	// it false: their text is complete.
	truncated bool
	// progress is the preformatted progress line, or "" when no hook is
	// installed (an unobserved run must not pay for the formatting).
	progress string
	// fail, when non-nil, aborts the whole run.
	fail error
}

// dispatchCalls runs one turn's tool calls and returns their outputs
// positionally: outputs[i] answers calls[i]. A non-nil error aborts the run,
// and the outputs returned alongside it are the ones that must still be
// appended to the history (the sequential loop's behavior, preserved).
//
// published is how many results were reported to the session, excluding the
// one that aborted the run. It is returned SEPARATELY from the outputs because
// the two answer different questions: outputs is what the model gets to see
// next, published is what the log already says happened. They only differ on
// the parallel abort path, where calls that were already in flight are logged
// but never reach the model.
//
// turn is the session-numbered turn (TurnBase already applied) and is used for
// progress events only - the loop's own budget counts elsewhere.
func (l *Loop) dispatchCalls(ctx context.Context, turn int, calls []llm.ToolCall) (outputs []string, published int, err error) {
	if l.parallelizable(calls) {
		return l.dispatchParallel(ctx, turn, calls)
	}
	outputs = make([]string, 0, len(calls))
	for _, call := range calls {
		l.announce(turn, call)
		r := l.approveCall(ctx, turn, call)
		if r == nil {
			ran := l.runCall(ctx, turn, call)
			r = &ran
		}
		output, err := l.publish(call, *r)
		if err != nil {
			return outputs, published, err
		}
		published++
		outputs = append(outputs, output)
	}
	return outputs, published, nil
}

// parallelizable reports whether this turn's calls may run concurrently. All
// three conditions must hold, and each is a separate promise:
//
//   - more than one call: one call has nothing to overlap with, and taking the
//     sequential path keeps the common case byte-identical to pre-v0.2;
//   - ParallelTools: the operator did not opt out (tools.parallel: false);
//   - every call auto-approved: see autoApproved.
func (l *Loop) parallelizable(calls []llm.ToolCall) bool {
	if len(calls) < 2 || !l.ParallelTools {
		return false
	}
	for _, call := range calls {
		if !l.autoApproved(call) {
			return false
		}
	}
	return true
}

// autoApproved reports whether call's policy is known to resolve without
// asking anybody.
//
// SECURITY: this is the gate that keeps an "ask" policy off the concurrent
// path. Two calls prompting at once would interleave two questions on one
// terminal, and a human answering "y" could not tell which call they just
// granted. Anything short of a positive "this is a plain allow" - an unlisted
// predicate, a policy the caller did not describe - falls back to sequential.
func (l *Loop) autoApproved(call llm.ToolCall) bool {
	if l.Approve == nil {
		// No approver at all is the documented allow-all default: nothing can
		// stop for a human because nothing is asked.
		return true
	}
	if l.AutoApprove == nil {
		return false
	}
	return l.AutoApprove(call)
}

// maxParallelToolCalls bounds how many of one turn's tool calls may be running
// at the same time.
//
// SECURITY: the length of the tool-call list is model output, and model output
// is untrusted input. A well-behaved provider keeps it to single digits, but
// nothing on the wire enforces that: a hostile or broken response can carry
// thousands of calls within the response body cap, and one goroutine (plus one
// subprocess, its pipes and its file descriptors) per call turned that number
// straight into a PID/memory/fd spike. This bound is the spike limiter only -
// the budgets (max_turns, max_tokens, timeout) remain the limiters on total
// VOLUME, and every call the model asked for still runs.
//
// Eight is chosen to sit above what real providers emit, so the common turn
// never queues and stays byte-identical to the unbounded dispatcher.
const maxParallelToolCalls = 8

// dispatchParallel runs the calls of one turn concurrently and publishes their
// results in the model's original call order.
//
// CONTRACT: the concurrency bound is invisible in the output. Calls are
// announced, approved and published in the model's call order regardless of
// which wave they ran in, and results are slotted by index, so a turn wider
// than maxParallelToolCalls produces exactly the events an unbounded
// dispatcher would - only spread over more waves.
func (l *Loop) dispatchParallel(ctx context.Context, turn int, calls []llm.ToolCall) (outputs []string, published int, err error) {
	// Announced up front, in call order: the tool_call events say what the
	// MODEL asked for, and that fact is known before any tool runs.
	for _, call := range calls {
		l.announce(turn, call)
	}

	// The permission checks stay sequential and in call order even here: an
	// Approver is caller code with no concurrency contract, and every call on
	// this path is auto-approved anyway, so the checks are cheap.
	//
	// A check that refuses is therefore only reachable when AutoApprove and the
	// Approver disagree (a caller bug). It is still honored to the letter - the
	// RULING decides, never the predicate - and the call simply does not run.
	results := make([]callResult, len(calls))
	refused := make([]bool, len(calls))
	for i, call := range calls {
		if r := l.approveCall(ctx, turn, call); r != nil {
			results[i], refused[i] = *r, true
		}
	}

	// OWNERSHIP: every goroutine started here is owned by this call. It writes
	// to exactly one slot of `results` (no two goroutines share an index, so
	// the slice needs no lock), touches no other loop state, and is joined by
	// the Wait below before this function returns - including when ctx is
	// already done, because the tools honor ctx themselves and returning early
	// would leave them writing into a turn nobody is reading.
	//
	// SECURITY: the slot is taken BEFORE the goroutine is started, not inside
	// it, so a turn carrying thousands of calls costs thousands of iterations
	// rather than thousands of live goroutines (see maxParallelToolCalls). The
	// channel cannot deadlock: every started goroutine releases its slot on
	// every exit path, and nothing a tool does can block that release.
	var wg sync.WaitGroup
	slots := make(chan struct{}, maxParallelToolCalls)
	for i, call := range calls {
		if refused[i] {
			continue // the policy already answered for this call
		}
		slots <- struct{}{}
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-slots }()
			results[i] = l.runCall(ctx, turn, call)
		}()
	}
	wg.Wait()

	outputs = make([]string, 0, len(calls))
	var failure error
	for i, call := range calls {
		output, pubErr := l.publish(call, results[i])
		if pubErr != nil {
			// The first failure in CALL order is the run's failure - not the
			// first in completion order, which would make the exit code depend
			// on tool latency. The remaining results are still logged: those
			// tools ran, and a tool_call without its tool_result would be a
			// hole in the audit trail.
			if failure == nil {
				failure = pubErr
			}
			continue
		}
		// Counted whether or not the run is already over: this call ran and
		// its tool_result is in the file, so a summary that skipped it would
		// contradict the log it summarizes. The abort short-circuit below
		// governs the HISTORY only.
		published++
		if failure == nil {
			// After a failure the run is over, so later outputs are not
			// appended to the history - the sequential path would never have
			// produced them either.
			outputs = append(outputs, output)
		}
	}
	return outputs, published, failure
}

// announce logs the tool_call event and the "model requested" progress line.
// CONTRACT: it runs before the permission check, so denied and unknown-tool
// calls appear in the log too (docs/contracts/jsonl-events.md).
func (l *Loop) announce(turn int, call llm.ToolCall) {
	l.Session.ToolCall(call.ID, call.Name, call.Arguments)
	l.progressf("turn %d: model requested %s %s", turn,
		clipRunes(call.Name, maxProgressName), clipRunes(call.Arguments, maxProgressArgs))
}

// approveCall consults the Approver. It returns nil when the call may run, and
// the result to publish when it may not.
func (l *Loop) approveCall(ctx context.Context, turn int, call llm.ToolCall) *callResult {
	if l.Approve == nil {
		return nil
	}
	name := clipRunes(call.Name, maxProgressName)
	ruling, err := l.Approve(ctx, call)
	if err != nil {
		// CONTRACT: a policy that could not answer is a harness dispatch
		// failure (`error`), not a denial - nothing ruled on this call.
		return &callResult{
			logged:   "approval check failed: " + err.Error(),
			isErr:    true,
			outcome:  session.OutcomeError,
			progress: l.eventf("turn %d: %s error: approval check failed: %s", turn, name, clipRunes(err.Error(), maxProgressArgs)),
			fail:     fmt.Errorf("approval check for tool %q: %w", call.Name, err),
		}
	}
	// CONTRACT: the denial reason is recorded so the session file answers
	// "why did the agent stop using that tool?" without the operator having
	// to re-derive the profile and the TTY state of a run that is over
	// (live-test finding C-3).
	outcome := denialOutcome(ruling.Reason)
	switch ruling.Decision {
	case DenyAbort:
		return &callResult{
			logged:   "permission denied",
			isErr:    true,
			outcome:  outcome,
			progress: l.eventf("turn %d: %s error: permission denied", turn, name),
			fail:     fmt.Errorf("%w: tool %q", ErrPermissionDenied, call.Name),
		}
	case DenyContinue:
		msg := fmt.Sprintf("permission denied: tool %q was not approved for this call", call.Name)
		return &callResult{
			output:   msg,
			logged:   msg,
			isErr:    true,
			outcome:  outcome,
			progress: l.eventf("turn %d: %s error: permission denied", turn, name),
		}
	case Allow:
		return nil
	default:
		// SECURITY: an out-of-range Decision is a policy bug, and a
		// broken policy must never fail open - abort instead of running
		// the tool.
		return &callResult{
			logged:   "approval policy returned an invalid decision",
			isErr:    true,
			outcome:  session.OutcomeError,
			progress: l.eventf("turn %d: %s error: approval policy returned an invalid decision", turn, name),
			fail:     fmt.Errorf("approval check for tool %q returned invalid decision %d", call.Name, ruling.Decision),
		}
	}
}

// runCall looks the tool up and invokes it. Tool-level failures (unknown tool,
// bad arguments, tool errors) come back as result text so the model can adapt;
// nothing here can abort the run.
//
// CONCURRENCY: this is the only part of a dispatch that runs on a per-call
// goroutine. It writes no loop state - the caller stores the returned value -
// and it reads the injected Clock, which is why a parallel run needs a
// concurrency-safe one (see Loop.Clock).
func (l *Loop) runCall(ctx context.Context, turn int, call llm.ToolCall) callResult {
	name := clipRunes(call.Name, maxProgressName)

	tool, ok := l.Registry.Get(call.Name)
	if !ok {
		// The model hallucinated a tool. Tell it what actually exists so
		// the next turn can recover instead of failing the whole run.
		msg := fmt.Sprintf("error: unknown tool %q; available tools: %v", call.Name, l.Registry.Names())
		// The registry listing is deliberately left out of the event: it is
		// for the model, and it would push the operator's line off the screen.
		return callResult{
			output: msg, logged: msg, isErr: true, outcome: session.OutcomeError,
			progress: l.eventf("turn %d: %s error: unknown tool", turn, name),
		}
	}

	// Timed around the invocation alone: an "ask" policy may have kept the
	// call waiting for a human, and reporting that wait as tool latency would
	// make an operator hunt a slow tool that was never slow.
	//
	// Both reads are conditional on the hook: the duration exists only to fill
	// in the event below, and an unobserved run must pay nothing for the
	// feature - including not advancing an injected Clock, which is caller
	// state a record/replay run observes.
	var started time.Time
	if l.Progress != nil {
		started = l.now()
	}
	output, outcome, err := invokeTool(ctx, tool, call.Arguments)
	if err != nil {
		msg := fmt.Sprintf("error: %v", err)
		return callResult{
			output: msg, logged: msg, isErr: true, outcome: session.OutcomeError,
			progress: l.eventf("turn %d: %s error: %s", turn, name, clipRunes(err.Error(), maxProgressArgs)),
		}
	}
	// CONTRACT: is_error stays false here whatever the outcome was - a tool
	// that RAN and failed is not a harness dispatch failure, and that boolean
	// is frozen (docs/contracts/jsonl-events.md). The classification travels in
	// the additive `outcome`/`exit_code` fields instead, so a non-zero exit is
	// visible without being renamed an error.
	kind, exitCode := toolOutcome(outcome)
	// The ceiling is applied to the FINISHED text, after the tool's own cap and
	// after whatever framing it added, and the clipped string is what both the
	// model and the log get: the session file must show what the model read,
	// not a fuller version of it.
	output, truncated := l.capResult(output, outcome.Truncated)
	r := callResult{output: output, logged: output, outcome: kind, exitCode: exitCode, truncated: truncated}
	if l.Progress != nil {
		// Guarded rather than left to eventf: the arguments - the second
		// clock read - are evaluated before the call, so only the guard here
		// keeps the read out of an unobserved run.
		suffix := ""
		if truncated {
			// Said on the operator's line too: a model that answered from half
			// a file looks like a model that answered badly, and only this
			// word tells the two apart while the run is still on screen.
			suffix = " (truncated)"
		}
		r.progress = l.eventf("turn %d: %s %s (%.1fs)%s", turn, name, outcome, l.now().Sub(started).Seconds(), suffix)
	}
	return r
}

// capResult applies the run's tool-result ceiling to one finished result and
// reports whether the model's copy of the text is short of what the tool
// produced - either because the tool said so or because the ceiling cut it.
//
// It is a method rather than three lines inside runCall only to keep that
// function under the cyclomatic budget (docs/engineering.md §5.1).
func (l *Loop) capResult(output string, toolTruncated bool) (string, bool) {
	if l.MaxToolResultBytes <= 0 {
		return output, toolTruncated
	}
	output, cut := tools.CapText(output, l.MaxToolResultBytes)
	return output, toolTruncated || cut
}

// publish writes a finished call's session event and progress line, in that
// order, and reports what the model gets to see. It is the single point where
// a dispatch becomes visible, which is what lets the concurrent path decide
// WHEN that happens without duplicating any of the reporting.
func (l *Loop) publish(call llm.ToolCall, r callResult) (string, error) {
	l.logToolResult(call, r.logged, r.isErr, r.outcome, r.exitCode, r.truncated)
	if r.progress != "" && l.Progress != nil {
		l.Progress(r.progress)
	}
	return r.output, r.fail
}

// logToolResult writes the one tool_result event for a dispatched call. Every
// exit from dispatch goes through it so no path can forget the observability
// fields - a tool_result without an outcome would be indistinguishable from a
// pre-v0.1.0 log line.
func (l *Loop) logToolResult(call llm.ToolCall, result string, isErr bool, outcome session.ToolOutcome, exitCode *int, truncated bool) {
	l.Session.ToolResult(session.ToolResult{
		CallID: call.ID, Tool: call.Name, Result: result,
		IsErr: isErr, Outcome: outcome, ExitCode: exitCode, Truncated: truncated,
	})
}

// denialOutcome maps an approver's reason onto the published outcome enum.
func denialOutcome(reason DenialReason) session.ToolOutcome {
	switch reason {
	case DeniedNoTTY:
		return session.OutcomeDeniedNoTTY
	case DeniedAskRefused:
		return session.OutcomeAskRefused
	case DeniedByPolicy:
		return session.OutcomeDeniedPolicy
	default:
		// An unknown reason is still a denial; recording it as the plain policy
		// denial keeps the event honest about WHAT happened when the loop
		// cannot be honest about why.
		return session.OutcomeDeniedPolicy
	}
}

// toolOutcome maps a tool's internal classification onto the published outcome
// enum and the optional exit status.
//
// The tools package deliberately keeps its own vocabulary (it classifies what a
// child process did, not what a log consumer needs), so the translation lives
// here rather than in either package's type: tools stays free of the schema,
// and session stays free of process semantics.
//
// Note the two rejections that share `denied_policy`: the shell tool's
// allow/deny patterns refuse a command before it runs, which is the same
// statement the permission profile's `deny` makes about a tool. They stay
// distinguishable in the event anyway - a permission denial sets is_error and
// says "permission denied", a shell rejection does not (it is content the model
// reads and adapts to).
func toolOutcome(o tools.Outcome) (session.ToolOutcome, *int) {
	switch o.Kind {
	case tools.OutcomeOK:
		return session.OutcomeOK, nil
	case tools.OutcomeRejected:
		return session.OutcomeDeniedPolicy, nil
	case tools.OutcomeTimedOut:
		return session.OutcomeTimeout, nil
	case tools.OutcomeAborted:
		return session.OutcomeAborted, nil
	case tools.OutcomeExit:
		// The exit status is copied out of the value so the pointer cannot
		// alias a caller's Outcome.
		code := o.ExitCode
		return session.OutcomeNonzeroExit, &code
	case tools.OutcomeToolError:
		return session.OutcomeToolError, nil
	case tools.OutcomeIndeterminate:
		return session.OutcomeIndeterminate, nil
	default:
		// Unreachable unless a kind is added without a case here. Reporting
		// `error` beats a confident `ok`: an unknown state is not a success,
		// and the operator-facing progress line makes the same choice.
		return session.OutcomeError, nil
	}
}

// outcomeInvoker is the richer half of tools.Tool: a tool that can end in a way
// its result text reports but its error return cannot (a rejected command, a
// timeout, a non-zero exit) implements it to say which one happened.
//
// It is an OPTIONAL interface rather than a widening of tools.Tool because a
// tool need not have anything to say: a tool that either does its job or
// returns a Go error is fully described by the error branch in runCall, and the
// zero Outcome reads as exactly that. Declaring it here, in the consuming
// package (docs/engineering.md §5.1), keeps tools free of any assumption about
// who is watching.
type outcomeInvoker interface {
	InvokeOutcome(ctx context.Context, rawArgs string) (string, tools.Outcome, error)
}

// invokeTool calls a tool through the richer path when it offers one. A tool
// that does not implement outcomeInvoker gets the zero Outcome, which reads as
// "ok" - correct, because for those tools a nil error IS success.
func invokeTool(ctx context.Context, tool tools.Tool, rawArgs string) (string, tools.Outcome, error) {
	if oi, ok := tool.(outcomeInvoker); ok {
		return oi.InvokeOutcome(ctx, rawArgs)
	}
	out, err := tool.Invoke(ctx, rawArgs)
	return out, tools.Outcome{}, err
}

// Progress event size caps, in runes. Tool names, tool arguments and tool
// error text all come from the model, so they are clipped before they are
// embedded in an event.
//
// SECURITY: a progress line is operator-facing, and an operator who cannot see
// the whole line cannot read it. The args bound must exceed any plausible
// secret length: cmd redacts secrets from events BY VALUE after the loop
// composes them, so a clip below secret length would cut a secret in half and
// let its prefix through unmatched (review finding, TestVerboseRedactsLong-
// Secrets). 512 runes clears real-world keys and JWT headers while a 100KB
// fs_write payload still cannot flood a monitor; the terminal display length
// is separately capped by cmd's whole-line backstop. Names get the same
// 64-rune bound the config schema puts on honest tool names - the name in a
// tool call comes from the model, not from the reviewed YAML.
const (
	maxProgressArgs = 512
	maxProgressName = 64
)

// clipMarker marks event text that clipRunes shortened.
const clipMarker = "... (clipped)"

// clipRunes shortens s to at most max runes, marking the cut. Runes, not
// bytes, because the cap exists to bound what a human sees on one line; a
// rune-wise cut also cannot split a multi-byte character in half.
func clipRunes(s string, max int) string {
	runes := []rune(s)
	if len(runes) <= max {
		return s
	}
	return string(runes[:max]) + clipMarker
}

// progressf emits one formatted progress event, or does nothing when no hook
// is installed. Formatting is skipped entirely in the nil case so an unobserved
// run pays nothing for the feature.
func (l *Loop) progressf(format string, args ...any) {
	if l.Progress == nil {
		return
	}
	l.Progress(fmt.Sprintf(format, args...))
}

// eventf formats a progress line for LATER emission, returning "" when no hook
// is installed. It is progressf split in half for the calls whose text is
// composed where it may not yet be reported: a concurrent call's line is built
// on its own goroutine and emitted by the dispatcher, in call order. The empty
// string is what publish reads as "nothing to say", so an unobserved run still
// formats nothing.
func (l *Loop) eventf(format string, args ...any) string {
	if l.Progress == nil {
		return ""
	}
	return fmt.Sprintf(format, args...)
}

// now reads the injected clock, defaulting to time.Now (docs/engineering.md §5.4).
func (l *Loop) now() time.Time {
	if l.Clock == nil {
		return time.Now()
	}
	return l.Clock()
}

// toolCallIDs extracts the IDs of a message's tool calls for session logging.
func toolCallIDs(calls []llm.ToolCall) []string {
	if len(calls) == 0 {
		return nil
	}
	ids := make([]string, len(calls))
	for i, c := range calls {
		ids[i] = c.ID
	}
	return ids
}
