package loop

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/llm"
	"github.com/lasthumanintheloop/amele/internal/session"
	"github.com/lasthumanintheloop/amele/internal/tools"
)

// update refreshes golden files: go test ./internal/loop -update
var update = flag.Bool("update", false, "rewrite golden files")

// fixedSessionClock ticks one second per call from a fixed origin, making the
// session log byte-stable for the golden test.
func fixedSessionClock() session.Clock {
	t := time.Date(2026, 8, 12, 9, 0, 0, 0, time.UTC)
	return func() time.Time {
		t = t.Add(time.Second)
		return t
	}
}

// newLoop wires a loop over a fake provider and an echo subprocess tool.
func newLoop(t *testing.T, fake *llm.Fake, limits Limits) *Loop {
	t.Helper()
	reg := tools.NewRegistry()
	echo := tools.NewSubprocess(config.SubprocessTool{
		Name: "echo_tool", Description: "echoes stdin", Command: []string{"cat"},
	}, t.TempDir())
	if err := reg.Register(echo); err != nil {
		t.Fatal(err)
	}
	if limits.MaxTurns == 0 {
		limits.MaxTurns = 10
	}
	return &Loop{
		Provider:     fake,
		Registry:     reg,
		Limits:       limits,
		Model:        "test-model",
		SystemPrompt: "You are a test agent.",
	}
}

func usage(in, out int) llm.Usage { return llm.Usage{InputTokens: in, OutputTokens: out} }

func TestRunTextOnly(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{llm.TextResponse("answer", usage(10, 5))}}
	l := newLoop(t, fake, Limits{})

	res, err := l.Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	if res.FinalText != "answer" || res.Turns != 1 || res.ToolCalls != 0 {
		t.Errorf("result: %+v", res)
	}
	if res.Usage.Total() != 15 {
		t.Errorf("usage: %+v", res.Usage)
	}

	// The provider must have received system + user messages and the tool defs.
	req := fake.Requests[0]
	if len(req.Messages) != 2 || req.Messages[0].Role != llm.RoleSystem || req.Messages[1].Content != "task" {
		t.Errorf("request messages: %+v", req.Messages)
	}
	if len(req.Tools) != 1 || req.Tools[0].Name != "echo_tool" {
		t.Errorf("request tools: %+v", req.Tools)
	}
}

func TestRunToolRoundTrip(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.ToolCallResponse("c1", "echo_tool", `{"stdin": "ping"}`, usage(10, 5)),
		llm.TextResponse("done", usage(20, 5)),
	}}
	l := newLoop(t, fake, Limits{})

	res, err := l.Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	if res.FinalText != "done" || res.Turns != 2 || res.ToolCalls != 1 {
		t.Errorf("result: %+v", res)
	}

	// Second request must carry the assistant tool call and the tool result,
	// linked by tool_call_id.
	second := fake.Requests[1]
	last := second.Messages[len(second.Messages)-1]
	if last.Role != llm.RoleTool || last.ToolCallID != "c1" || last.Content != "ping" {
		t.Errorf("tool message: %+v", last)
	}
}

func TestRunUnknownToolRecoverable(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.ToolCallResponse("c1", "ghost_tool", `{}`, usage(1, 1)),
		llm.TextResponse("recovered", usage(1, 1)),
	}}
	l := newLoop(t, fake, Limits{})

	res, err := l.Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	if res.FinalText != "recovered" {
		t.Errorf("result: %+v", res)
	}
	// The model must have been told what tools actually exist.
	second := fake.Requests[1]
	last := second.Messages[len(second.Messages)-1]
	if !strings.Contains(last.Content, "unknown tool") || !strings.Contains(last.Content, "echo_tool") {
		t.Errorf("recovery message: %q", last.Content)
	}
}

func TestRunMaxTurns(t *testing.T) {
	// The model asks for a tool forever; the loop must kill it at the cap.
	responses := make([]llm.Response, 5)
	for i := range responses {
		responses[i] = llm.ToolCallResponse("c", "echo_tool", `{}`, usage(1, 1))
	}
	fake := &llm.Fake{Responses: responses}
	l := newLoop(t, fake, Limits{MaxTurns: 3})

	res, err := l.Run(context.Background(), "task")
	if !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("want ErrBudgetExceeded, got %v", err)
	}
	if res.Turns != 3 {
		t.Errorf("turns: %d", res.Turns)
	}
	if len(fake.Requests) != 3 {
		t.Errorf("provider calls: %d", len(fake.Requests))
	}
}

func TestRunMaxTokens(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.ToolCallResponse("c1", "echo_tool", `{}`, usage(60, 20)),
		llm.TextResponse("never returned", usage(60, 20)),
	}}
	l := newLoop(t, fake, Limits{MaxTokens: 100})

	res, err := l.Run(context.Background(), "task")
	if !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("want ErrBudgetExceeded, got %v", err)
	}
	// Both turns ran (80 then 160 > 100), and accounting reflects the
	// overshooting turn - the log must show what was actually spent.
	if res.Usage.Total() != 160 {
		t.Errorf("usage: %+v", res.Usage)
	}
}

// TestRunUsageSaturates: accumulating token counts across turns must not wrap
// the accumulator negative, which would read as "under budget" and defeat the
// fail-closed token limit. The loop must accumulate through the saturating
// llm.Usage.Add, not a raw += (Codex bug review, overflow guard). MaxTokens is
// left at maxInt so the budget does not trip early and the accumulation path
// across two turns is what is exercised.
func TestRunUsageSaturates(t *testing.T) {
	const maxInt = int(^uint(0) >> 1)
	const near = maxInt - 10
	fake := &llm.Fake{Responses: []llm.Response{
		llm.ToolCallResponse("c1", "echo_tool", `{}`, usage(near, 0)),
		llm.TextResponse("done", usage(near, 0)),
	}}
	l := newLoop(t, fake, Limits{MaxTokens: maxInt})

	res, err := l.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Two turns of ~maxInt input tokens each: a raw += wraps InputTokens
	// negative; the saturating Add pins it at maxInt.
	if res.Usage.InputTokens < 0 || res.Usage.Total() < 0 {
		t.Errorf("accumulated usage wrapped negative: %+v", res.Usage)
	}
}

func TestRunPermissionDenied(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.ToolCallResponse("c1", "echo_tool", `{}`, usage(1, 1)),
	}}
	l := newLoop(t, fake, Limits{})
	l.Approve = approver(DenyAbort, DeniedByPolicy)

	_, err := l.Run(context.Background(), "task")
	if !errors.Is(err, ErrPermissionDenied) {
		t.Fatalf("want ErrPermissionDenied, got %v", err)
	}
}

// TestApproverSeesFullCall: the approver receives the whole tool call so
// Phase 2 policies can rule on arguments (e.g. fs_write paths), not just the
// tool name.
func TestApproverSeesFullCall(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.ToolCallResponse("c1", "echo_tool", `{"stdin": "ping"}`, usage(1, 1)),
		llm.TextResponse("done", usage(1, 1)),
	}}
	l := newLoop(t, fake, Limits{})

	var got llm.ToolCall
	l.Approve = func(_ context.Context, call llm.ToolCall) (Ruling, error) {
		got = call
		return Ruling{Decision: Allow}, nil
	}
	if _, err := l.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	if got.Name != "echo_tool" || got.ID != "c1" || !strings.Contains(got.Arguments, "ping") {
		t.Errorf("approver saw incomplete call: %+v", got)
	}
}

// TestApproverDenyContinue: a non-aborting denial becomes a tool result the
// model can see and adapt to - the Phase 2 "ask → denied, keep going" path.
func TestApproverDenyContinue(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.ToolCallResponse("c1", "echo_tool", `{}`, usage(1, 1)),
		llm.TextResponse("adapted", usage(1, 1)),
	}}
	l := newLoop(t, fake, Limits{})
	l.Approve = approver(DenyContinue, DeniedByPolicy)

	res, err := l.Run(context.Background(), "task")
	if err != nil {
		t.Fatalf("DenyContinue must not abort the run: %v", err)
	}
	if res.FinalText != "adapted" {
		t.Errorf("result: %+v", res)
	}
	second := fake.Requests[1]
	last := second.Messages[len(second.Messages)-1]
	if last.Role != llm.RoleTool || !strings.Contains(last.Content, "permission denied") {
		t.Errorf("model must see the denial as a tool result: %+v", last)
	}
}

// TestApproverInvalidDecisionFailsClosed: an out-of-range Decision (a buggy
// Phase 2 profile returning a bad cast) must NOT run the tool - the "a broken
// policy must never fail open" contract applies to nonsense values too
// (round-2 review finding P1-1).
func TestApproverInvalidDecisionFailsClosed(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.ToolCallResponse("c1", "echo_tool", `{"stdin": "must not run"}`, usage(1, 1)),
		llm.TextResponse("never reached", usage(1, 1)),
	}}
	l := newLoop(t, fake, Limits{})
	l.Approve = approver(Decision(42), DeniedByPolicy)

	res, err := l.Run(context.Background(), "task")
	if err == nil {
		t.Fatal("invalid decision must abort the run, not execute the tool")
	}
	if res.ToolCalls != 0 {
		t.Errorf("tool ran despite invalid decision: %d calls", res.ToolCalls)
	}
}

// TestApproverError: a failure inside the policy itself aborts the run - a
// broken policy must never fail open.
func TestApproverError(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.ToolCallResponse("c1", "echo_tool", `{}`, usage(1, 1)),
	}}
	l := newLoop(t, fake, Limits{})
	policyErr := errors.New("policy backend unreachable")
	l.Approve = func(_ context.Context, _ llm.ToolCall) (Ruling, error) { return Ruling{}, policyErr }

	_, err := l.Run(context.Background(), "task")
	if !errors.Is(err, policyErr) {
		t.Fatalf("approver error must abort the run: %v", err)
	}
}

func TestRunProviderError(t *testing.T) {
	fake := &llm.Fake{Errs: []error{llm.ErrProvider}}
	l := newLoop(t, fake, Limits{})

	res, err := l.Run(context.Background(), "task")
	if !errors.Is(err, llm.ErrProvider) {
		t.Fatalf("want ErrProvider, got %v", err)
	}
	// Truthful accounting: the request WAS attempted, so the failed turn
	// counts - run_end must not claim fewer round-trips than were made.
	if res.Turns != 1 {
		t.Errorf("failed attempt must count as a turn: got %d want 1", res.Turns)
	}
}

func TestRunContextTimeoutIsBudget(t *testing.T) {
	// A run timeout arrives as a context deadline; the loop must map it to
	// the budget sentinel (exit 3), not a provider error.
	fake := &llm.Fake{Responses: []llm.Response{llm.TextResponse("x", usage(1, 1))}}
	l := newLoop(t, fake, Limits{})

	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	_, err := l.Run(ctx, "task")
	if !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("want ErrBudgetExceeded, got %v", err)
	}
}

// TestRunContextCancelIsNotBudget: SIGINT-style cancellation must NOT map to
// the budget sentinel - exit 3 is reserved for configured budgets (Codex
// review finding F7).
func TestRunContextCancelIsNotBudget(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{llm.TextResponse("x", usage(1, 1))}}
	l := newLoop(t, fake, Limits{})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := l.Run(ctx, "task")
	if err == nil {
		t.Fatal("expected error")
	}
	if errors.Is(err, ErrBudgetExceeded) {
		t.Errorf("cancellation must not be a budget error: %v", err)
	}
}

// TestRunTruncatedAnswerFails: finish_reason length/content_filter means the
// task did NOT complete; success exit would hide silent truncation in cron
// (Codex review finding F8).
func TestRunTruncatedAnswerFails(t *testing.T) {
	for _, reason := range []string{"length", "content_filter"} {
		t.Run(reason, func(t *testing.T) {
			resp := llm.TextResponse("partial...", usage(1, 1))
			resp.FinishReason = reason
			fake := &llm.Fake{Responses: []llm.Response{resp}}
			l := newLoop(t, fake, Limits{})

			res, err := l.Run(context.Background(), "task")
			if err == nil {
				t.Fatal("truncated answer must be an error")
			}
			if errors.Is(err, ErrBudgetExceeded) || errors.Is(err, llm.ErrProvider) {
				t.Errorf("must map to task failure (exit 1), got %v", err)
			}
			// The partial text is preserved for the session log.
			if res.FinalText != "partial..." {
				t.Errorf("partial text lost: %q", res.FinalText)
			}
		})
	}
}

// TestRunErrorFinishReasonFails: OpenRouter reports mid-generation failures
// as finish_reason "error" with an empty body. This was observed live with
// google/gemini-3.5-flash-lite and must map to task failure, not success
// (live-test finding B1).
func TestRunErrorFinishReasonFails(t *testing.T) {
	resp := llm.TextResponse("", usage(5, 0))
	resp.FinishReason = "error"
	fake := &llm.Fake{Responses: []llm.Response{resp}}
	l := newLoop(t, fake, Limits{})

	_, err := l.Run(context.Background(), "task")
	if err == nil {
		t.Fatal("finish_reason=error must be an error")
	}
	if errors.Is(err, ErrBudgetExceeded) || errors.Is(err, llm.ErrProvider) {
		t.Errorf("must map to task failure (exit 1), got %v", err)
	}
}

// TestRunEmptyAnswerFails: a clean stop but empty content is not a usable
// answer for a pipe consumer.
func TestRunEmptyAnswerFails(t *testing.T) {
	resp := llm.TextResponse("", usage(1, 1))
	resp.FinishReason = "stop"
	fake := &llm.Fake{Responses: []llm.Response{resp}}
	l := newLoop(t, fake, Limits{})

	if _, err := l.Run(context.Background(), "task"); err == nil {
		t.Error("empty final answer must be an error")
	}
}

// TestRunMissingUsageFailsClosed: a configured token budget with a provider
// that reports no usage must abort, not silently run unbudgeted (Codex
// review finding F9).
func TestRunMissingUsageFailsClosed(t *testing.T) {
	resp := llm.TextResponse("ok", llm.Usage{})
	resp.UsageMissing = true
	fake := &llm.Fake{Responses: []llm.Response{resp}}

	// With a token budget: fail closed.
	l := newLoop(t, fake, Limits{MaxTokens: 100})
	if _, err := l.Run(context.Background(), "task"); !errors.Is(err, ErrBudgetExceeded) {
		t.Errorf("missing usage with max_tokens must fail closed: %v", err)
	}

	// Without a token budget: missing usage is acceptable.
	fake2 := &llm.Fake{Responses: []llm.Response{resp}}
	l2 := newLoop(t, fake2, Limits{})
	if _, err := l2.Run(context.Background(), "task"); err != nil {
		t.Errorf("missing usage without max_tokens must pass: %v", err)
	}
}

// TestTurnBaseOffsetsLoggedTurns pins the contract TurnBase exists for: a
// caller that drives several RunMessages calls inside one run_start/run_end
// pair (chat) must produce monotonically numbered turns, while the per-call
// budget keeps counting from 1.
func TestTurnBaseOffsetsLoggedTurns(t *testing.T) {
	w, err := session.New(t.TempDir(), session.Options{})
	if err != nil {
		t.Fatal(err)
	}

	fake := &llm.Fake{Responses: []llm.Response{
		llm.TextResponse("first", usage(10, 5)),
		llm.TextResponse("second", usage(10, 5)),
	}}
	l := newLoop(t, fake, Limits{MaxTurns: 1})
	l.Session = w

	history := []llm.Message{{Role: llm.RoleUser, Content: "one"}}
	if _, err := l.RunMessages(context.Background(), history); err != nil {
		t.Fatal(err)
	}
	// The second exchange starts a fresh per-call budget (MaxTurns 1 again)
	// but continues the session's turn numbering.
	l.TurnBase = 1
	if _, err := l.RunMessages(context.Background(), history); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(w.Path())
	if err != nil {
		t.Fatal(err)
	}
	var turns []int
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var ev session.Event
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("invalid JSONL line %q: %v", line, err)
		}
		if ev.Type == "llm_response" {
			turns = append(turns, ev.Turn)
		}
	}
	if len(turns) != 2 || turns[0] != 1 || turns[1] != 2 {
		t.Errorf("logged turns = %v, want [1 2]", turns)
	}
}

// TestSessionEventOrder pins the loop→session interaction at unit level: a
// tool round-trip must emit the exact ordered, ID-linked event sequence the
// replay contract depends on (previously only e2e substring checks existed).
func TestSessionEventOrder(t *testing.T) {
	w, err := session.New(t.TempDir(), session.Options{})
	if err != nil {
		t.Fatal(err)
	}

	fake := &llm.Fake{Responses: []llm.Response{
		llm.ToolCallResponse("c1", "echo_tool", `{"stdin": "ping"}`, usage(10, 5)),
		llm.TextResponse("done", usage(20, 5)),
	}}
	l := newLoop(t, fake, Limits{})
	l.Session = w

	if _, err := l.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}

	data, err := os.ReadFile(w.Path())
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	var events []session.Event
	for _, line := range lines {
		var ev session.Event
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("invalid JSONL line %q: %v", line, err)
		}
		events = append(events, ev)
	}

	wantTypes := []string{"run_start", "llm_response", "tool_call", "tool_result", "llm_response"}
	if len(events) != len(wantTypes) {
		t.Fatalf("event count: got %d want %d (%v)", len(events), len(wantTypes), lines)
	}
	for i, want := range wantTypes {
		if events[i].Type != want {
			t.Errorf("event[%d]: got %q want %q", i, events[i].Type, want)
		}
	}

	if len(events[1].ToolCallIDs) != 1 || events[1].ToolCallIDs[0] != "c1" {
		t.Errorf("llm_response tool_call_ids: %v", events[1].ToolCallIDs)
	}
	if events[2].CallID != "c1" || events[3].CallID != "c1" {
		t.Errorf("tool events must link by ID: %q / %q", events[2].CallID, events[3].CallID)
	}
	if events[3].Result != "ping" {
		t.Errorf("tool_result: %q", events[3].Result)
	}
	if events[4].Content != "done" {
		t.Errorf("final llm_response content: %q", events[4].Content)
	}
}

// TestRunMessagesPreservesHistory: RunMessages is the message-history core -
// it must send the caller's history verbatim and add nothing of its own (no
// system prompt), so callers that own the conversation (chat, structured
// output retries) stay in control of what the model sees.
func TestRunMessagesPreservesHistory(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{llm.TextResponse("answer", usage(1, 1))}}
	l := newLoop(t, fake, Limits{})

	history := []llm.Message{
		{Role: llm.RoleSystem, Content: "caller system prompt"},
		{Role: llm.RoleUser, Content: "first question"},
		{Role: llm.RoleAssistant, Content: "first answer"},
		{Role: llm.RoleUser, Content: "second question"},
	}

	res, err := l.RunMessages(context.Background(), history)
	if err != nil {
		t.Fatal(err)
	}
	if res.FinalText != "answer" || res.Turns != 1 {
		t.Errorf("result: %+v", res)
	}
	if got := fake.Requests[0].Messages; !reflect.DeepEqual(got, history) {
		t.Errorf("history not sent verbatim:\n got %+v\nwant %+v", got, history)
	}
}

// TestRunMessagesDoesNotMutateCallerHistory: RunMessages appends to its own
// copy. Regression guard for a slice-aliasing bug that a single-turn test
// cannot catch - the caller's history is built with spare capacity, so an
// append inside the loop would overwrite the caller's backing array in place.
// A caller reusing its history (chat, retries) must never see loop-internal
// messages appear behind its back.
func TestRunMessagesDoesNotMutateCallerHistory(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.ToolCallResponse("c1", "echo_tool", `{"stdin": "ping"}`, usage(1, 1)),
		llm.TextResponse("bad", usage(1, 1)),
		llm.TextResponse("good", usage(1, 1)),
	}}
	l := newLoop(t, fake, Limits{})
	l.FinalValidator = func(text string) (string, bool) {
		if text == "good" {
			return "", true
		}
		return "fix it", false
	}
	l.MaxFinalRejections = 3

	// Spare capacity is the point: len 2, cap 8.
	history := make([]llm.Message, 0, 8)
	history = append(history,
		llm.Message{Role: llm.RoleSystem, Content: "caller system prompt"},
		llm.Message{Role: llm.RoleUser, Content: "caller task"},
	)
	want := []llm.Message{
		{Role: llm.RoleSystem, Content: "caller system prompt"},
		{Role: llm.RoleUser, Content: "caller task"},
	}

	// The run must append an assistant turn, a tool result, a rejected
	// assistant turn and feedback - none of which may land in history.
	res, err := l.RunMessages(context.Background(), history)
	if err != nil {
		t.Fatal(err)
	}
	if res.Turns != 3 || res.ToolCalls != 1 {
		t.Fatalf("test must exercise multiple appends: %+v", res)
	}

	if len(history) != 2 {
		t.Fatalf("caller history length changed: %d", len(history))
	}
	if !reflect.DeepEqual(history, want) {
		t.Errorf("caller history mutated:\n got %+v\nwant %+v", history, want)
	}
	// Reslicing to full capacity exposes writes past len - the aliasing bug
	// this test exists for is invisible through len alone.
	if grown := history[:cap(history)]; !reflect.DeepEqual(grown[2:], make([]llm.Message, cap(history)-2)) {
		t.Errorf("loop wrote into the caller's spare capacity: %+v", grown[2:])
	}
}

// TestResponseFormatForwarded: the loop hands ResponseFormat to the provider
// verbatim on EVERY round-trip, retries included - a repair turn that silently
// dropped the native schema constraint would make the retry likelier to fail
// than the attempt it is repairing.
func TestResponseFormatForwarded(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.TextResponse("bad", usage(1, 1)),
		llm.TextResponse("good", usage(1, 1)),
	}}
	rf := &llm.ResponseFormat{Name: "amele_output", Schema: json.RawMessage(`{"type":"object"}`)}
	l := newLoop(t, fake, Limits{})
	l.ResponseFormat = rf
	l.FinalValidator = func(text string) (string, bool) { return "fix it", text == "good" }

	if _, err := l.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	if len(fake.Requests) != 2 {
		t.Fatalf("provider calls: %d want 2", len(fake.Requests))
	}
	for i, req := range fake.Requests {
		if req.ResponseFormat != rf {
			t.Errorf("request %d: ResponseFormat = %+v, want the configured one", i, req.ResponseFormat)
		}
	}
}

// TestTuningForwarded: the provider-neutral knobs the config sets once
// (reasoning depth, sampling, output cap, raw params) ride on EVERY request of
// the run, not just the first. A tool turn that dropped the reasoning knob
// would think shallower exactly where the model needs it most, and a dropped
// cap would leave the later turns of a run uncapped.
func TestTuningForwarded(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.ToolCallResponse("c1", "echo_tool", `{"stdin": "ping"}`, usage(1, 1)),
		llm.TextResponse("done", usage(1, 1)),
	}}
	temperature, topP := 0.2, 0.9
	tuning := Tuning{
		MaxOutputTokens: 65536,
		Reasoning:       &llm.ReasoningSpec{Effort: "high"},
		Temperature:     &temperature,
		TopP:            &topP,
		Extra:           map[string]json.RawMessage{"verbosity": json.RawMessage(`"low"`)},
	}
	l := newLoop(t, fake, Limits{})
	l.Tuning = tuning

	if _, err := l.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	if len(fake.Requests) != 2 {
		t.Fatalf("provider calls: %d want 2", len(fake.Requests))
	}
	for i, req := range fake.Requests {
		if req.MaxOutputTokens != tuning.MaxOutputTokens {
			t.Errorf("request %d: MaxOutputTokens = %d, want %d", i, req.MaxOutputTokens, tuning.MaxOutputTokens)
		}
		if req.Reasoning != tuning.Reasoning {
			t.Errorf("request %d: Reasoning = %+v, want the configured spec", i, req.Reasoning)
		}
		if req.Temperature != tuning.Temperature || req.TopP != tuning.TopP {
			t.Errorf("request %d: sampling = %v/%v, want the configured pointers", i, req.Temperature, req.TopP)
		}
		if !reflect.DeepEqual(req.Extra, tuning.Extra) {
			t.Errorf("request %d: Extra = %v, want %v", i, req.Extra, tuning.Extra)
		}
	}
}

// TestTuningZeroValueSendsNothing: a loop with no Tuning asks for nothing -
// the pre-tuning behavior, where every knob is the provider's own default.
func TestTuningZeroValueSendsNothing(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{llm.TextResponse("answer", usage(1, 1))}}
	l := newLoop(t, fake, Limits{})

	if _, err := l.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	req := fake.Requests[0]
	if req.MaxOutputTokens != 0 || req.Reasoning != nil || req.Temperature != nil || req.TopP != nil || req.Extra != nil {
		t.Errorf("untuned request carries knobs: %+v", req)
	}
}

// TestFinalValidatorRetry: a rejected final answer is not the end of the run -
// the rejected assistant message stays in history and the feedback comes back
// as a user message, which is what lets the model repair its own output.
func TestFinalValidatorRetry(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.TextResponse("bad", usage(1, 1)),
		llm.TextResponse("good", usage(1, 1)),
	}}
	l := newLoop(t, fake, Limits{})
	l.FinalValidator = func(text string) (string, bool) {
		if text == "good" {
			return "", true
		}
		return "fix it", false
	}

	res, err := l.Run(context.Background(), "task")
	if err != nil {
		t.Fatal(err)
	}
	if res.FinalText != "good" || res.Turns != 2 {
		t.Errorf("result: %+v", res)
	}

	if len(fake.Requests) != 2 {
		t.Fatalf("provider calls: %d want 2", len(fake.Requests))
	}
	second := fake.Requests[1].Messages
	if len(second) < 2 {
		t.Fatalf("second request too short: %+v", second)
	}
	rejected, feedback := second[len(second)-2], second[len(second)-1]
	if rejected.Role != llm.RoleAssistant || rejected.Content != "bad" {
		t.Errorf("rejected answer must stay in history: %+v", rejected)
	}
	if feedback.Role != llm.RoleUser || feedback.Content != "fix it" {
		t.Errorf("feedback must be appended as a user message: %+v", feedback)
	}
}

// TestFinalValidatorEmptyFeedback: a refusal carrying no feedback must still
// produce a non-empty user message - some OpenAI-compatible endpoints reject
// a request containing an empty-content message outright.
func TestFinalValidatorEmptyFeedback(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.TextResponse("bad", usage(1, 1)),
		llm.TextResponse("good", usage(1, 1)),
	}}
	l := newLoop(t, fake, Limits{})
	l.FinalValidator = func(text string) (string, bool) { return "", text == "good" }

	if _, err := l.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	second := fake.Requests[1].Messages
	last := second[len(second)-1]
	if last.Role != llm.RoleUser || last.Content == "" {
		t.Errorf("feedback message: %+v", last)
	}
}

// TestFinalValidatorExhausted: past MaxFinalRejections the run fails with the
// ErrOutputRejected sentinel (exit 6) while still carrying the last rejected
// answer, so the operator can see what the model kept producing.
func TestFinalValidatorExhausted(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.TextResponse("bad", usage(1, 1)),
		llm.TextResponse("bad", usage(1, 1)),
		llm.TextResponse("bad", usage(1, 1)),
	}}
	l := newLoop(t, fake, Limits{})
	l.FinalValidator = func(string) (string, bool) { return "fix it", false }
	l.MaxFinalRejections = 2

	res, err := l.Run(context.Background(), "task")
	if !errors.Is(err, ErrOutputRejected) {
		t.Fatalf("want ErrOutputRejected, got %v", err)
	}
	if res.Turns != 3 {
		t.Errorf("turns: %d want 3", res.Turns)
	}
	if res.FinalText != "bad" {
		t.Errorf("last rejected answer must be preserved: %q", res.FinalText)
	}
}

// TestFinalValidatorRetriesCountAgainstBudgets: CONTRACT - validator retries
// are ordinary turns. A cron job's max_turns cap must still fire even when the
// extra round-trips come from schema repair rather than from tool use.
func TestFinalValidatorRetriesCountAgainstBudgets(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.TextResponse("bad", usage(1, 1)),
		llm.TextResponse("bad", usage(1, 1)),
		llm.TextResponse("bad", usage(1, 1)),
	}}
	l := newLoop(t, fake, Limits{MaxTurns: 2})
	l.FinalValidator = func(string) (string, bool) { return "fix it", false }
	l.MaxFinalRejections = 10

	res, err := l.Run(context.Background(), "task")
	if !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("want ErrBudgetExceeded, got %v", err)
	}
	if res.Turns != 2 {
		t.Errorf("turns: %d want 2", res.Turns)
	}
	if len(fake.Requests) != 2 {
		t.Errorf("provider calls: %d want 2", len(fake.Requests))
	}
}

func TestRunNoSystemPrompt(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{llm.TextResponse("ok", usage(1, 1))}}
	l := newLoop(t, fake, Limits{})
	l.SystemPrompt = ""

	if _, err := l.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	if len(fake.Requests[0].Messages) != 1 {
		t.Errorf("empty system prompt must not produce a system message: %+v", fake.Requests[0].Messages)
	}
}

// stepClock returns a Clock starting at a fixed instant and advancing by step
// on every call, so the durations rendered into progress events are
// deterministic (docs/engineering.md §5.4). One tool dispatch reads the clock twice, so
// a tool call always takes exactly one step.
func stepClock(step time.Duration) Clock {
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	var n int
	return func() time.Time {
		t := base.Add(time.Duration(n) * step)
		n++
		return t
	}
}

// recordProgress installs a recording Progress hook and returns the slice it
// appends to. Reading the slice is only safe after the run has returned.
func recordProgress(l *Loop) *[]string {
	var events []string
	l.Progress = func(event string) { events = append(events, event) }
	return &events
}

// TestProgressEventSequence pins the hook's contract: one event per tool call
// dispatch, one per tool result, one when a final answer is accepted - in that
// order, with the turn number the session log uses.
func TestProgressEventSequence(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.ToolCallResponse("c1", "echo_tool", `{"stdin": "ping"}`, usage(10, 5)),
		llm.TextResponse("done", usage(20, 7)),
	}}
	l := newLoop(t, fake, Limits{})
	l.Clock = stepClock(time.Second)
	events := recordProgress(l)

	if _, err := l.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	want := []string{
		`turn 1: model requested echo_tool {"stdin": "ping"}`,
		`turn 1: echo_tool ok (1.0s)`,
		`turn 2: final answer (7 tokens)`,
	}
	if !reflect.DeepEqual(*events, want) {
		t.Errorf("progress events:\n got %q\nwant %q", *events, want)
	}
}

// TestProgressNilIsNoOp: the hook is optional, and a loop without one must
// behave exactly as before (nil = no-op, no panic).
func TestProgressNilIsNoOp(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.ToolCallResponse("c1", "echo_tool", `{}`, usage(1, 1)),
		llm.TextResponse("done", usage(1, 1)),
	}}
	l := newLoop(t, fake, Limits{})
	if l.Progress != nil {
		t.Fatal("Progress must default to nil")
	}
	if _, err := l.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
}

// TestProgressNilDoesNotReadClock pins the other half of "an unobserved run
// pays nothing": the per-dispatch timing reads exist only to render the
// `ok (1.2s)` event, so a loop without a hook must not make them. The Clock is
// injected state a caller can observe (a recorded/replayed run advances it per
// read), and spending two reads per tool call on an event nobody receives
// would change what that caller sees.
func TestProgressNilDoesNotReadClock(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{
		llm.ToolCallResponse("c1", "echo_tool", `{"stdin":"a"}`, usage(1, 1)),
		llm.ToolCallResponse("c2", "echo_tool", `{"stdin":"b"}`, usage(1, 1)),
		llm.TextResponse("done", usage(1, 1)),
	}}
	l := newLoop(t, fake, Limits{})
	var reads int
	l.Clock = func() time.Time {
		reads++
		return time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	}

	if _, err := l.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	// The run's own start and end reads (Result.Duration) are the only ones
	// an unobserved run is entitled to, whatever the tool call count.
	if reads != 2 {
		t.Errorf("clock reads with a nil Progress hook = %d, want 2 (run start + run end)", reads)
	}
}

// TestProgressToolFailureEvents pins that every way a tool call can end badly
// still produces exactly one result event, so a verbose operator never sees a
// dispatch with no outcome.
func TestProgressToolFailureEvents(t *testing.T) {
	tests := []struct {
		name    string
		tool    string
		args    string
		approve Approver
		want    []string
	}{
		{
			name: "unknown tool",
			tool: "ghost_tool",
			args: `{}`,
			want: []string{
				`turn 1: model requested ghost_tool {}`,
				`turn 1: ghost_tool error: unknown tool`,
			},
		},
		{
			name: "tool error",
			tool: "echo_tool",
			args: `{"args": ["extra"]}`,
			want: []string{
				`turn 1: model requested echo_tool {"args": ["extra"]}`,
				`turn 1: echo_tool error: tool "echo_tool" does not accept extra args (set allow_args: true in the config to permit them)`,
			},
		},
		{
			name:    "denied but continuing",
			tool:    "echo_tool",
			args:    `{}`,
			approve: approver(DenyContinue, DeniedByPolicy),
			want: []string{
				`turn 1: model requested echo_tool {}`,
				`turn 1: echo_tool error: permission denied`,
			},
		},
		{
			name:    "denied and aborting",
			tool:    "echo_tool",
			args:    `{}`,
			approve: approver(DenyAbort, DeniedByPolicy),
			want: []string{
				`turn 1: model requested echo_tool {}`,
				`turn 1: echo_tool error: permission denied`,
			},
		},
		{
			name: "approval policy failure",
			tool: "echo_tool",
			args: `{}`,
			approve: func(context.Context, llm.ToolCall) (Ruling, error) {
				return Ruling{}, errors.New("policy backend unreachable")
			},
			want: []string{
				`turn 1: model requested echo_tool {}`,
				`turn 1: echo_tool error: approval check failed: policy backend unreachable`,
			},
		},
		{
			name:    "invalid decision",
			tool:    "echo_tool",
			args:    `{}`,
			approve: approver(Decision(42), DeniedByPolicy),
			want: []string{
				`turn 1: model requested echo_tool {}`,
				`turn 1: echo_tool error: approval policy returned an invalid decision`,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fake := &llm.Fake{Responses: []llm.Response{
				llm.ToolCallResponse("c1", tt.tool, tt.args, usage(1, 1)),
				llm.TextResponse("done", usage(1, 1)),
			}}
			l := newLoop(t, fake, Limits{})
			l.Approve = tt.approve
			l.Clock = stepClock(time.Second)
			events := recordProgress(l)

			// The run's outcome differs per case (some abort, some recover);
			// only the emitted events are under test here.
			_, _ = l.Run(context.Background(), "task")

			got := *events
			if len(got) > len(tt.want) {
				got = got[:len(tt.want)] // drop the trailing final-answer event
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("progress events:\n got %q\nwant %q", got, tt.want)
			}
		})
	}
}

// TestProgressClipsModelControlledText: tool names, arguments and tool error
// text are model-controlled, so one event must not be able to carry a 100KB
// payload into whatever renders it (SECURITY, docs/engineering.md §5.5).
func TestProgressClipsModelControlledText(t *testing.T) {
	longArgs := `{"stdin": "` + strings.Repeat("A", 5000) + `"}`
	longName := strings.Repeat("n", 500)
	fake := &llm.Fake{Responses: []llm.Response{
		llm.ToolCallResponse("c1", longName, longArgs, usage(1, 1)),
		llm.TextResponse("done", usage(1, 1)),
	}}
	l := newLoop(t, fake, Limits{})
	events := recordProgress(l)

	if _, err := l.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	for _, event := range *events {
		// Bound: the two clipped fragments plus fixed framing. The args cap
		// is deliberately larger than any real secret (see maxProgressArgs);
		// what matters here is that a 5KB payload cannot get through.
		if n := len([]rune(event)); n > maxProgressArgs+maxProgressName+2*len(clipMarker)+64 {
			t.Errorf("event is %d runes, want it clipped: %q", n, event)
		}
	}
	if !strings.Contains((*events)[0], clipMarker) {
		t.Errorf("clipped arguments must be marked: %q", (*events)[0])
	}
}

// newShellLoop wires a loop over a fake provider and the builtin shell tool,
// configured so every way a command can end badly is reachable: `rm *` is
// denied by policy and the tool timeout is short enough to fire in a test.
func newShellLoop(t *testing.T, fake *llm.Fake) *Loop {
	t.Helper()
	reg := tools.NewRegistry()
	sh, err := tools.NewShell(config.ShellConfig{
		Deny:    []string{"rm *"},
		Timeout: config.Duration(100 * time.Millisecond),
	}, t.TempDir())
	if err != nil {
		t.Skipf("shell tool unavailable: %v", err)
	}
	if err := reg.Register(sh); err != nil {
		t.Fatal(err)
	}
	return &Loop{
		Provider:     fake,
		Registry:     reg,
		Limits:       Limits{MaxTurns: 10},
		Model:        "test-model",
		SystemPrompt: "You are a test agent.",
	}
}

// failingToolResponses scripts the three tool endings that report themselves as
// ordinary result text with a nil Go error: a policy rejection, a non-zero
// exit, and a tool timeout.
func failingToolResponses() []llm.Response {
	return []llm.Response{
		llm.ToolCallResponse("c1", "shell", `{"command":"rm -rf /tmp/x"}`, usage(10, 5)),
		llm.ToolCallResponse("c2", "shell", `{"command":"echo out; echo err >&2; exit 3"}`, usage(10, 5)),
		llm.ToolCallResponse("c3", "shell", `{"command":"sleep 5"}`, usage(10, 5)),
		llm.TextResponse("done", usage(20, 5)),
	}
}

// TestToolOutcomesInTheSessionLog pins the exact bytes of a log whose tool
// calls were rejected, failed and timed out - the published tool_result fields
// (`outcome`, `exit_code`, `result_bytes`) and everything they must not
// disturb. The golden was regenerated once, deliberately, when v0.1.0 added
// those fields (docs/contracts/jsonl-events.md, "Change policy"); a diff after
// that is either a contract change or a bug, and both must be seen in review.
//
// The second half pins the other direction: the bytes must not depend on
// whether a Progress hook is installed - observing a run must not change it.
func TestToolOutcomesInTheSessionLog(t *testing.T) {
	logFor := func(withProgress bool) []byte {
		w, err := session.New(t.TempDir(), session.Options{Clock: fixedSessionClock()})
		if err != nil {
			t.Fatal(err)
		}
		l := newShellLoop(t, &llm.Fake{Responses: failingToolResponses()})
		l.Session = w
		l.Clock = stepClock(time.Second)
		if withProgress {
			recordProgress(l)
		}
		if _, err := l.Run(context.Background(), "task"); err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(w.Path())
		if err != nil {
			t.Fatal(err)
		}
		return data
	}

	got := logFor(false)
	if observed := logFor(true); string(observed) != string(got) {
		t.Errorf("a Progress hook changed the session log.\nwith:\n%s\nwithout:\n%s", observed, got)
	}

	goldenPath := filepath.Join("testdata", "golden", "tool-outcomes.jsonl")
	if *update {
		if err := os.MkdirAll(filepath.Dir(goldenPath), 0o750); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(goldenPath, got, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(goldenPath) //nolint:gosec // G304: fixed testdata path.
	if err != nil {
		t.Fatalf("reading golden (run with -update to create): %v", err)
	}
	if string(got) != string(want) {
		t.Errorf("session log differs from golden.\ngot:\n%s\nwant:\n%s", got, want)
	}
}

// toolResultEvents runs l over its own session file and returns the tool_result
// events it produced. The session log - not the progress hook - is what an
// operator reads the morning after a cron run, so the outcome tests assert on
// the file.
func toolResultEvents(t *testing.T, ctx context.Context, l *Loop) []session.Event {
	t.Helper()
	w, err := session.New(t.TempDir(), session.Options{Clock: fixedSessionClock()})
	if err != nil {
		t.Fatal(err)
	}
	l.Session = w
	// The run's own outcome differs per case (some abort, some recover); only
	// the events are under test.
	_, _ = l.Run(ctx, "task")

	data, err := os.ReadFile(w.Path())
	if err != nil {
		t.Fatal(err)
	}
	var results []session.Event
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var ev session.Event
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("invalid JSONL line %q: %v", line, err)
		}
		if ev.Type == "tool_result" {
			results = append(results, ev)
		}
	}
	return results
}

// approver returns a fixed Ruling, for the table cases that only care about
// what the loop does with a policy answer.
func approver(decision Decision, reason DenialReason) Approver {
	return func(context.Context, llm.ToolCall) (Ruling, error) {
		return Ruling{Decision: decision, Reason: reason}, nil
	}
}

// TestSessionRecordsToolOutcome walks every value of the tool_result `outcome`
// enum from the loop's side: each one must be reachable from a distinct code
// path, so that a permission denial, a rejected command and a failed command
// are told apart in the session file (live-test findings C-2 and C-3) instead
// of all reading as "a tool result with is_error".
//
// The dispatch-failure cases live here; the ones that need a real child process
// are in TestSessionRecordsProcessOutcomes.
func TestSessionRecordsToolOutcome(t *testing.T) {
	tests := []struct {
		name      string
		tool      string
		args      string
		approve   Approver
		want      session.ToolOutcome
		wantIsErr bool
	}{
		{
			name: "successful call",
			tool: "echo_tool", args: `{"stdin": "ping"}`,
			want: session.OutcomeOK,
		},
		{
			name: "unknown tool is a dispatch failure",
			tool: "ghost_tool", args: `{}`,
			want: session.OutcomeError, wantIsErr: true,
		},
		{
			name: "unusable arguments are a dispatch failure",
			tool: "echo_tool", args: `{"args": ["extra"]}`,
			want: session.OutcomeError, wantIsErr: true,
		},
		{
			name: "denied by policy, run continues",
			tool: "echo_tool", args: `{}`,
			approve: approver(DenyContinue, DeniedByPolicy),
			want:    session.OutcomeDeniedPolicy, wantIsErr: true,
		},
		{
			name: "denied by policy, run aborts",
			tool: "echo_tool", args: `{}`,
			approve: approver(DenyAbort, DeniedByPolicy),
			want:    session.OutcomeDeniedPolicy, wantIsErr: true,
		},
		{
			name: "ask auto-denied with no TTY",
			tool: "echo_tool", args: `{}`,
			approve: approver(DenyContinue, DeniedNoTTY),
			want:    session.OutcomeDeniedNoTTY, wantIsErr: true,
		},
		{
			name: "the human said no",
			tool: "echo_tool", args: `{}`,
			approve: approver(DenyContinue, DeniedAskRefused),
			want:    session.OutcomeAskRefused, wantIsErr: true,
		},
		{
			name: "broken approval check",
			tool: "echo_tool", args: `{}`,
			approve: func(context.Context, llm.ToolCall) (Ruling, error) {
				return Ruling{}, errors.New("policy backend unreachable")
			},
			want: session.OutcomeError, wantIsErr: true,
		},
		{
			name: "invalid decision fails closed",
			tool: "echo_tool", args: `{}`,
			approve: approver(Decision(42), DeniedByPolicy),
			want:    session.OutcomeError, wantIsErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := newLoop(t, &llm.Fake{Responses: []llm.Response{
				llm.ToolCallResponse("c1", tt.tool, tt.args, usage(1, 1)),
				llm.TextResponse("done", usage(1, 1)),
			}}, Limits{})
			l.Approve = tt.approve

			events := toolResultEvents(t, context.Background(), l)
			if len(events) != 1 {
				t.Fatalf("tool_result events: got %d, want 1", len(events))
			}
			if got := events[0].Outcome; got != tt.want {
				t.Errorf("outcome = %q, want %q", got, tt.want)
			}
			if events[0].IsErr != tt.wantIsErr {
				t.Errorf("is_error = %v, want %v", events[0].IsErr, tt.wantIsErr)
			}
			if events[0].ExitCode != nil {
				t.Errorf("exit_code must be absent when no process ran, got %d", *events[0].ExitCode)
			}
			if events[0].ResultBytes != len(events[0].Result) {
				t.Errorf("result_bytes = %d, want %d (the result is short enough to be unclipped)",
					events[0].ResultBytes, len(events[0].Result))
			}
		})
	}
}

// TestSessionRecordsProcessOutcomes covers the three endings a real child
// process produces, which is where the enum earns its keep: all three come back
// as ordinary result text with a nil Go error (so the model can react), and
// before this field the log could not tell them from a clean run.
func TestSessionRecordsProcessOutcomes(t *testing.T) {
	events := toolResultEvents(t, context.Background(), newShellLoop(t, &llm.Fake{Responses: failingToolResponses()}))
	if len(events) != 3 {
		t.Fatalf("tool_result events: got %d, want 3", len(events))
	}
	// The shell's allow/deny patterns are an operator policy refusing the
	// command before it runs - the same statement the permission profile's
	// `deny` makes, so it carries the same outcome. is_error stays absent:
	// the rejection is content the model reads and adapts to.
	if got := events[0].Outcome; got != session.OutcomeDeniedPolicy {
		t.Errorf("rejected command outcome = %q, want %q", got, session.OutcomeDeniedPolicy)
	}
	if events[0].IsErr {
		t.Error("a shell-policy rejection is not a harness dispatch failure")
	}
	if got := events[1].Outcome; got != session.OutcomeNonzeroExit {
		t.Errorf("failing command outcome = %q, want %q", got, session.OutcomeNonzeroExit)
	}
	if events[1].ExitCode == nil || *events[1].ExitCode != 3 {
		t.Errorf("exit_code = %v, want 3", events[1].ExitCode)
	}
	if events[1].IsErr {
		t.Error("a non-zero exit is not an error: grep exits 1 when it finds nothing")
	}
	if got := events[2].Outcome; got != session.OutcomeTimeout {
		t.Errorf("timed-out command outcome = %q, want %q", got, session.OutcomeTimeout)
	}
	if events[2].ExitCode != nil {
		t.Errorf("a killed command has no exit status, got %d", *events[2].ExitCode)
	}
}

// cancelingProvider cancels the run right after answering, which is how a
// SIGINT or the overall run timeout arrives in the middle of a tool call. It
// makes the `aborted` outcome reachable without racing a real clock.
type cancelingProvider struct {
	inner  *llm.Fake
	cancel context.CancelFunc
}

func (p *cancelingProvider) Chat(ctx context.Context, req llm.Request) (*llm.Response, error) {
	resp, err := p.inner.Chat(ctx, req)
	p.cancel()
	return resp, err
}

// TestSessionRecordsAbortedOutcome: a command the RUN killed (Ctrl-C, overall
// timeout) must not be logged as the tool's own timeout - an operator who reads
// `timeout` goes looking for the wrong knob.
func TestSessionRecordsAbortedOutcome(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	l := newShellLoop(t, nil)
	l.Provider = &cancelingProvider{
		inner: &llm.Fake{Responses: []llm.Response{
			llm.ToolCallResponse("c1", "shell", `{"command":"echo hi"}`, usage(1, 1)),
		}},
		cancel: cancel,
	}

	events := toolResultEvents(t, ctx, l)
	if len(events) != 1 {
		t.Fatalf("tool_result events: got %d, want 1", len(events))
	}
	if got := events[0].Outcome; got != session.OutcomeAborted {
		t.Errorf("outcome = %q, want %q", got, session.OutcomeAborted)
	}
}

// TestSessionResultBytesSurviveClipping is the log-side half of live-test
// finding C-5: the model may read up to 64 KiB of tool output while the log
// keeps 8 KiB, and without a recorded size nothing in the file says so.
func TestSessionResultBytesSurviveClipping(t *testing.T) {
	const size = 20000
	l := newShellLoop(t, &llm.Fake{Responses: []llm.Response{
		llm.ToolCallResponse("c1", "shell", `{"command":"printf 'x%.0s' $(seq 20000)"}`, usage(1, 1)),
		llm.TextResponse("done", usage(1, 1)),
	}})

	events := toolResultEvents(t, context.Background(), l)
	if len(events) != 1 {
		t.Fatalf("tool_result events: got %d, want 1", len(events))
	}
	if events[0].ResultBytes != size {
		t.Errorf("result_bytes = %d, want %d", events[0].ResultBytes, size)
	}
	if len(events[0].Result) >= size {
		t.Errorf("the logged result should be clipped, got %d bytes", len(events[0].Result))
	}
}

// TestProgressReportsToolOutcomeTruthfully is the regression for live-test
// finding C-2: a subprocess/shell tool reports a rejection, a timeout and a
// non-zero exit as ordinary result text with a nil error - deliberately, so the
// model sees the failure as content - and the verbose line used to call all
// three "ok". An operator watching `-v` must be able to tell a working tool
// from a failing one.
func TestProgressReportsToolOutcomeTruthfully(t *testing.T) {
	l := newShellLoop(t, &llm.Fake{Responses: failingToolResponses()})
	l.Clock = stepClock(time.Second)
	events := recordProgress(l)

	if _, err := l.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	want := []string{
		`turn 1: model requested shell {"command":"rm -rf /tmp/x"}`,
		`turn 1: shell rejected (1.0s)`,
		`turn 2: model requested shell {"command":"echo out; echo err >&2; exit 3"}`,
		`turn 2: shell exit 3 (1.0s)`,
		`turn 3: model requested shell {"command":"sleep 5"}`,
		`turn 3: shell timed out (1.0s)`,
		`turn 4: final answer (5 tokens)`,
	}
	if !reflect.DeepEqual(*events, want) {
		t.Errorf("progress events:\n got %q\nwant %q", *events, want)
	}
}

// TestProgressPlainToolStillReportsOK: outcome reporting is optional. A tool
// that does not classify its endings - the fs tools, which either do their job
// or return a Go error - must still produce the plain `ok` line, not a blank
// or a bogus one.
func TestProgressPlainToolStillReportsOK(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.txt"), []byte("hi"), 0o600); err != nil {
		t.Fatal(err)
	}
	fsTools, err := tools.NewFSTools(dir, tools.FSOptions{})
	if err != nil {
		t.Fatal(err)
	}
	reg := tools.NewRegistry()
	for _, tool := range fsTools {
		if err := reg.Register(tool); err != nil {
			t.Fatal(err)
		}
	}
	l := &Loop{
		Provider: &llm.Fake{Responses: []llm.Response{
			llm.ToolCallResponse("c1", "fs_read", `{"path":"a.txt"}`, usage(1, 1)),
			llm.TextResponse("done", usage(1, 1)),
		}},
		Registry: reg,
		Limits:   Limits{MaxTurns: 10},
		Model:    "test-model",
		Clock:    stepClock(time.Second),
	}
	events := recordProgress(l)

	if _, err := l.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}
	if want := `turn 1: fs_read ok (1.0s)`; (*events)[1] != want {
		t.Errorf("progress event = %q, want %q", (*events)[1], want)
	}
}

// TestProgressUsesSessionTurnNumbers: chat drives several RunMessages calls
// inside one session, so progress events must carry the same offset turn
// numbers the session log records - not a per-call count restarting at 1.
func TestProgressUsesSessionTurnNumbers(t *testing.T) {
	fake := &llm.Fake{Responses: []llm.Response{llm.TextResponse("second", usage(1, 3))}}
	l := newLoop(t, fake, Limits{})
	l.TurnBase = 4
	events := recordProgress(l)

	if _, err := l.RunMessages(context.Background(), []llm.Message{{Role: llm.RoleUser, Content: "hi"}}); err != nil {
		t.Fatal(err)
	}
	want := []string{"turn 5: final answer (3 tokens)"}
	if !reflect.DeepEqual(*events, want) {
		t.Errorf("progress events:\n got %q\nwant %q", *events, want)
	}
}

// TestToolOutcomeMapping pins the translation from the runner's internal
// vocabulary (tools.OutcomeKind) to the frozen enum the JSONL event publishes
// (session.ToolOutcome, docs/contracts/jsonl-events.md). It is a table rather
// than a golden run because the mapping is a contract in its own right: a new
// kind must be given a published outcome deliberately, and an unmapped one must
// report `error` instead of a confident `ok`.
func TestToolOutcomeMapping(t *testing.T) {
	three := 3
	tests := []struct {
		name     string
		in       tools.Outcome
		want     session.ToolOutcome
		wantCode *int
	}{
		{"ok", tools.Outcome{}, session.OutcomeOK, nil},
		{"rejected", tools.Outcome{Kind: tools.OutcomeRejected}, session.OutcomeDeniedPolicy, nil},
		{"timed out", tools.Outcome{Kind: tools.OutcomeTimedOut}, session.OutcomeTimeout, nil},
		{"aborted", tools.Outcome{Kind: tools.OutcomeAborted}, session.OutcomeAborted, nil},
		{"exit", tools.Outcome{Kind: tools.OutcomeExit, ExitCode: 3}, session.OutcomeNonzeroExit, &three},
		{"tool error", tools.Outcome{Kind: tools.OutcomeToolError}, session.OutcomeToolError, nil},
		{"indeterminate", tools.Outcome{Kind: tools.OutcomeIndeterminate}, session.OutcomeIndeterminate, nil},
		{"unknown", tools.Outcome{Kind: tools.OutcomeKind(99)}, session.OutcomeError, nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, code := toolOutcome(tt.in)
			if got != tt.want {
				t.Errorf("toolOutcome(%+v) = %q, want %q", tt.in, got, tt.want)
			}
			switch {
			case tt.wantCode == nil && code != nil:
				t.Errorf("exit code = %d, want nil", *code)
			case tt.wantCode != nil && code == nil:
				t.Errorf("exit code = nil, want %d", *tt.wantCode)
			case tt.wantCode != nil && *code != *tt.wantCode:
				t.Errorf("exit code = %d, want %d", *code, *tt.wantCode)
			}
		})
	}
}
