package loop

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lasthumanintheloop/amele/internal/llm"
	"github.com/lasthumanintheloop/amele/internal/session"
	"github.com/lasthumanintheloop/amele/internal/tools"
)

// rendezvousTimeout bounds every cross-goroutine wait in this file. A test that
// needs two tools to run at the same time can only prove it by having each wait
// for the other, and a sequential loop would then hang forever - the timeout
// turns that hang into a readable failure instead of a test-binary timeout.
const rendezvousTimeout = 5 * time.Second

// scriptedTool is a registry tool whose whole behavior is one injected
// function, so a test can make a call block, signal, sleep or count without a
// new type per case.
type scriptedTool struct {
	name string
	run  func(ctx context.Context, rawArgs string) (string, error)
}

func (s scriptedTool) Def() llm.ToolDef {
	return llm.ToolDef{Name: s.name, Description: "scripted test tool"}
}

func (s scriptedTool) Invoke(ctx context.Context, rawArgs string) (string, error) {
	return s.run(ctx, rawArgs)
}

// concurrencyMeter records how many scripted tools were inside Invoke at the
// same time. It is the observable that separates "ran in parallel" from "ran
// fast": a sequential loop can never push the peak above 1.
type concurrencyMeter struct {
	mu      sync.Mutex
	current int
	peak    int
}

func (m *concurrencyMeter) enter() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.current++
	if m.current > m.peak {
		m.peak = m.current
	}
}

func (m *concurrencyMeter) leave() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.current--
}

func (m *concurrencyMeter) Peak() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.peak
}

// countingTool returns a tool that reports the meter's peak occupancy. The
// short sleep gives a parallel loop a real chance to overlap; a sequential one
// still cannot, which is exactly what the sequential assertions rely on.
func countingTool(name string, m *concurrencyMeter) tools.Tool {
	return scriptedTool{name: name, run: func(context.Context, string) (string, error) {
		m.enter()
		defer m.leave()
		time.Sleep(20 * time.Millisecond)
		return name + "-done", nil
	}}
}

// rendezvousTools returns one tool per name; each announces its arrival and
// then waits for ALL of them to arrive. Only a loop that really ran them at the
// same time can get past the barrier, so a concurrency assertion built on these
// cannot pass by lucky timing - and a sequential loop fails with a message
// instead of an unexplained peak of 1. The meter (optional) therefore peaks at
// exactly len(names).
func rendezvousTools(t *testing.T, meter *concurrencyMeter, names ...string) []tools.Tool {
	t.Helper()
	var (
		mu      sync.Mutex
		arrived int
		allHere = make(chan struct{})
	)
	out := make([]tools.Tool, 0, len(names))
	for _, name := range names {
		out = append(out, scriptedTool{name: name, run: func(context.Context, string) (string, error) {
			if meter != nil {
				meter.enter()
				defer meter.leave()
			}
			mu.Lock()
			arrived++
			if arrived == len(names) {
				close(allHere)
			}
			mu.Unlock()
			waitFor(t, allHere, fmt.Sprintf("all %d calls to start", len(names)))
			return name + "-result", nil
		}})
	}
	return out
}

// waitFor blocks until ch is closed, failing the test if the loop never let
// another call reach its signal (i.e. it ran the calls sequentially).
func waitFor(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(rendezvousTimeout):
		t.Errorf("timed out waiting for %s: the calls did not run concurrently", what)
	}
}

// toolCallsResponse builds one assistant turn requesting several tools at once,
// which is the shape this whole file is about (llm.ToolCallResponse only makes
// single-call turns).
func toolCallsResponse(calls ...llm.ToolCall) llm.Response {
	return llm.Response{
		Message:      llm.Message{Role: llm.RoleAssistant, ToolCalls: calls},
		Usage:        usage(1, 1),
		FinishReason: "tool_calls",
	}
}

// call is shorthand for one requested tool call with empty arguments.
func call(id, name string) llm.ToolCall {
	return llm.ToolCall{ID: id, Name: name, Arguments: `{}`}
}

// parallelLoop wires a loop over a fake provider and the given tools with
// concurrent dispatch enabled.
func parallelLoop(t *testing.T, fake *llm.Fake, ts ...tools.Tool) *Loop {
	t.Helper()
	reg := tools.NewRegistry()
	for _, tool := range ts {
		if err := reg.Register(tool); err != nil {
			t.Fatal(err)
		}
	}
	return &Loop{
		Provider:      fake,
		Registry:      reg,
		Limits:        Limits{MaxTurns: 10},
		Model:         "test-model",
		ParallelTools: true,
	}
}

// runWithEvents runs the loop against a fresh session file and returns every
// logged event, which is where the ordering contract is actually visible.
func runWithEvents(t *testing.T, ctx context.Context, l *Loop) ([]session.Event, *Result, error) {
	t.Helper()
	w, err := session.New(t.TempDir(), session.Options{Clock: fixedSessionClock()})
	if err != nil {
		t.Fatal(err)
	}
	l.Session = w

	res, runErr := l.Run(ctx, "task")

	data, err := os.ReadFile(w.Path())
	if err != nil {
		t.Fatal(err)
	}
	var events []session.Event
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var ev session.Event
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatalf("invalid JSONL line %q: %v", line, err)
		}
		events = append(events, ev)
	}
	return events, res, runErr
}

// eventIDs renders the events of one type as "type:call_id" strings, so an
// ordering assertion reads as a list instead of a pile of indexes.
func eventIDs(events []session.Event, typ string) []string {
	var out []string
	for _, ev := range events {
		if ev.Type == typ {
			out = append(out, ev.CallID)
		}
	}
	return out
}

// toolMessages returns the tool-result messages of the last request the fake
// provider received, as "call_id=content" pairs.
func toolMessages(req llm.Request) []string {
	var out []string
	for _, m := range req.Messages {
		if m.Role == llm.RoleTool {
			out = append(out, m.ToolCallID+"="+m.Content)
		}
	}
	return out
}

// TestParallelCallsKeepModelOrder is the core determinism contract: the two
// calls finish in the REVERSE of the order the model asked for them, and both
// the session log and the message history must still read in call order.
// Concurrency is proven, not assumed - each tool waits for the other to start.
func TestParallelCallsKeepModelOrder(t *testing.T) {
	slowStarted, fastStarted := make(chan struct{}), make(chan struct{})

	slow := scriptedTool{name: "slow_tool", run: func(context.Context, string) (string, error) {
		close(slowStarted)
		waitFor(t, fastStarted, "the second call to start")
		// Finishing last while being asked first is the whole point: a loop
		// that logged in completion order would put this result second.
		time.Sleep(30 * time.Millisecond)
		return "slow-result", nil
	}}
	fast := scriptedTool{name: "fast_tool", run: func(context.Context, string) (string, error) {
		close(fastStarted)
		waitFor(t, slowStarted, "the first call to start")
		return "fast-result", nil
	}}

	fake := &llm.Fake{Responses: []llm.Response{
		toolCallsResponse(call("c1", "slow_tool"), call("c2", "fast_tool")),
		llm.TextResponse("done", usage(1, 1)),
	}}
	l := parallelLoop(t, fake, slow, fast)

	events, res, err := runWithEvents(t, context.Background(), l)
	if err != nil {
		t.Fatal(err)
	}
	if res.ToolCalls != 2 {
		t.Errorf("tool calls: got %d want 2", res.ToolCalls)
	}

	if got, want := eventIDs(events, "tool_call"), []string{"c1", "c2"}; !equalStrings(got, want) {
		t.Errorf("tool_call order: got %v want %v", got, want)
	}
	if got, want := eventIDs(events, "tool_result"), []string{"c1", "c2"}; !equalStrings(got, want) {
		t.Errorf("tool_result order: got %v want %v", got, want)
	}

	got := toolMessages(fake.Requests[1])
	want := []string{"c1=slow-result", "c2=fast-result"}
	if !equalStrings(got, want) {
		t.Errorf("history order: got %v want %v", got, want)
	}
}

// TestParallelEightCallsRace runs a turn wide enough to matter under -race:
// eight tools that all have to be in flight at once (each waits for the last
// one to arrive) while the loop collects their results.
func TestParallelEightCallsRace(t *testing.T) {
	const n = 8
	names := make([]string, 0, n)
	calls := make([]llm.ToolCall, 0, n)
	want := make([]string, 0, n)
	for i := range n {
		name := fmt.Sprintf("tool_%d", i)
		names = append(names, name)
		id := fmt.Sprintf("c%d", i)
		calls = append(calls, call(id, name))
		want = append(want, id+"="+name+"-result")
	}
	meter := &concurrencyMeter{}
	toolset := rendezvousTools(t, meter, names...)

	fake := &llm.Fake{Responses: []llm.Response{
		toolCallsResponse(calls...),
		llm.TextResponse("done", usage(1, 1)),
	}}
	l := parallelLoop(t, fake, toolset...)

	events, res, err := runWithEvents(t, context.Background(), l)
	if err != nil {
		t.Fatal(err)
	}
	if res.ToolCalls != n {
		t.Errorf("tool calls: got %d want %d", res.ToolCalls, n)
	}
	if peak := meter.Peak(); peak != n {
		t.Errorf("peak concurrency: got %d want %d", peak, n)
	}
	if got := toolMessages(fake.Requests[1]); !equalStrings(got, want) {
		t.Errorf("history order: got %v want %v", got, want)
	}
	ids := make([]string, 0, n)
	for i := range n {
		ids = append(ids, fmt.Sprintf("c%d", i))
	}
	if got := eventIDs(events, "tool_result"); !equalStrings(got, ids) {
		t.Errorf("tool_result order: got %v want %v", got, ids)
	}
}

// TestParallelBoundsInFlightCalls is the SECURITY regression for the review
// finding: the model decides how many tool calls one turn carries, and a
// hostile or broken response can carry thousands of them under the body cap.
// One goroutine (and one subprocess) per call turned that number straight into
// a resource spike, so the dispatcher must run them in bounded waves.
//
// The barrier makes both halves of the contract observable at once: the first
// maxParallelToolCalls calls have to be in flight together (a bound that
// serialized the turn would hang on it), and the meter must never see more
// than that many, however wide the turn is.
func TestParallelBoundsInFlightCalls(t *testing.T) {
	// Wide enough that an unbounded dispatcher's peak is unmistakable, small
	// enough to stay a fast unit test.
	const n = 4 * maxParallelToolCalls

	var (
		mu       sync.Mutex
		arrived  int
		waveFull = make(chan struct{})
	)
	meter := &concurrencyMeter{}
	toolset := make([]tools.Tool, 0, n)
	calls := make([]llm.ToolCall, 0, n)
	wantHistory := make([]string, 0, n)
	wantIDs := make([]string, 0, n)
	for i := range n {
		name := fmt.Sprintf("tool_%02d", i)
		id := fmt.Sprintf("c%02d", i)
		calls = append(calls, call(id, name))
		wantHistory = append(wantHistory, id+"="+name+"-result")
		wantIDs = append(wantIDs, id)
		toolset = append(toolset, scriptedTool{name: name, run: func(context.Context, string) (string, error) {
			meter.enter()
			defer meter.leave()
			mu.Lock()
			arrived++
			if arrived == maxParallelToolCalls {
				close(waveFull)
			}
			mu.Unlock()
			// Only the first wave really waits; once the barrier is open the
			// later waves sail through it, so the test does not depend on how
			// the bound schedules the remaining calls.
			waitFor(t, waveFull, fmt.Sprintf("%d calls to be in flight at once", maxParallelToolCalls))
			// Held open after the barrier so an UNBOUNDED dispatcher would
			// pile every call into the meter at once instead of draining the
			// wave fast enough to hide the spike.
			time.Sleep(20 * time.Millisecond)
			return name + "-result", nil
		}})
	}

	fake := &llm.Fake{Responses: []llm.Response{
		toolCallsResponse(calls...),
		llm.TextResponse("done", usage(1, 1)),
	}}
	l := parallelLoop(t, fake, toolset...)

	events, res, err := runWithEvents(t, context.Background(), l)
	if err != nil {
		t.Fatal(err)
	}
	if peak := meter.Peak(); peak != maxParallelToolCalls {
		t.Errorf("peak concurrency: got %d want %d (the dispatcher must run %d calls in bounded waves)",
			peak, maxParallelToolCalls, n)
	}
	// The bound is a spike limiter, not a call limiter: every call the model
	// asked for still runs, and still answers in call order.
	if res.ToolCalls != n {
		t.Errorf("tool calls: got %d want %d", res.ToolCalls, n)
	}
	if got := toolMessages(fake.Requests[1]); !equalStrings(got, wantHistory) {
		t.Errorf("history order: got %v want %v", got, wantHistory)
	}
	if got := eventIDs(events, "tool_result"); !equalStrings(got, wantIDs) {
		t.Errorf("tool_result order: got %v want %v", got, wantIDs)
	}
}

// TestParallelDisabledRunsSequentially: with ParallelTools off, two calls in
// one turn must never be in flight together - the opt-out has to be real, not
// advisory.
func TestParallelDisabledRunsSequentially(t *testing.T) {
	meter := &concurrencyMeter{}
	fake := &llm.Fake{Responses: []llm.Response{
		toolCallsResponse(call("c1", "a_tool"), call("c2", "b_tool")),
		llm.TextResponse("done", usage(1, 1)),
	}}
	l := parallelLoop(t, fake, countingTool("a_tool", meter), countingTool("b_tool", meter))
	l.ParallelTools = false

	events, _, err := runWithEvents(t, context.Background(), l)
	if err != nil {
		t.Fatal(err)
	}
	if peak := meter.Peak(); peak != 1 {
		t.Errorf("peak concurrency: got %d want 1 (parallel dispatch is off)", peak)
	}
	// The sequential path interleaves call/result per tool, which is the
	// pre-parallel event shape and must not change.
	var types []string
	for _, ev := range events {
		if ev.Type == "tool_call" || ev.Type == "tool_result" {
			types = append(types, ev.Type+":"+ev.CallID)
		}
	}
	want := []string{"tool_call:c1", "tool_result:c1", "tool_call:c2", "tool_result:c2"}
	if !equalStrings(types, want) {
		t.Errorf("sequential event order: got %v want %v", types, want)
	}
}

// TestSingleCallStaysSequential: a one-call turn takes the old path unchanged
// (no goroutine, and the same interleaved events), because that is the common
// case and it must not change shape just because the feature exists.
func TestSingleCallStaysSequential(t *testing.T) {
	meter := &concurrencyMeter{}
	fake := &llm.Fake{Responses: []llm.Response{
		toolCallsResponse(call("c1", "a_tool")),
		llm.TextResponse("done", usage(1, 1)),
	}}
	l := parallelLoop(t, fake, countingTool("a_tool", meter))

	events, _, err := runWithEvents(t, context.Background(), l)
	if err != nil {
		t.Fatal(err)
	}
	var types []string
	for _, ev := range events {
		if ev.Type == "tool_call" || ev.Type == "tool_result" {
			types = append(types, ev.Type+":"+ev.CallID)
		}
	}
	if want := []string{"tool_call:c1", "tool_result:c1"}; !equalStrings(types, want) {
		t.Errorf("single-call event order: got %v want %v", types, want)
	}
	if peak := meter.Peak(); peak != 1 {
		t.Errorf("peak concurrency: got %d want 1", peak)
	}
}

// TestPermissionGatingForcesSequential is the SECURITY assertion: a call whose
// policy is not a plain allow (an "ask" that may stop for a human, or an
// approver the caller never described) must drag the whole turn back onto the
// sequential path, so no TTY prompt can ever interleave with another call.
func TestPermissionGatingForcesSequential(t *testing.T) {
	cases := []struct {
		name        string
		approve     Approver
		autoApprove func(llm.ToolCall) bool
		wantPeak    int
	}{
		{
			name:    "ask policy on one call",
			approve: approver(Allow, DeniedByPolicy),
			// b_tool is the "ask" one; one non-auto call is enough.
			autoApprove: func(c llm.ToolCall) bool { return c.Name != "b_tool" },
			wantPeak:    1,
		},
		{
			name:        "approver without an auto-approve predicate",
			approve:     approver(Allow, DeniedByPolicy),
			autoApprove: nil,
			wantPeak:    1,
		},
		{
			name:        "every call auto-approved",
			approve:     approver(Allow, DeniedByPolicy),
			autoApprove: func(llm.ToolCall) bool { return true },
			wantPeak:    2,
		},
		{
			name:        "no approver at all is allow-all",
			approve:     nil,
			autoApprove: nil,
			wantPeak:    2,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			meter := &concurrencyMeter{}
			fake := &llm.Fake{Responses: []llm.Response{
				toolCallsResponse(call("c1", "a_tool"), call("c2", "b_tool")),
				llm.TextResponse("done", usage(1, 1)),
			}}
			// The concurrent cases use the barrier tools (a sequential loop
			// could not finish them at all); the sequential cases use plain
			// counting tools, which a concurrent loop would overlap.
			toolset := []tools.Tool{countingTool("a_tool", meter), countingTool("b_tool", meter)}
			if tc.wantPeak > 1 {
				toolset = rendezvousTools(t, meter, "a_tool", "b_tool")
			}
			l := parallelLoop(t, fake, toolset...)
			l.Approve = tc.approve
			l.AutoApprove = tc.autoApprove

			if _, _, err := runWithEvents(t, context.Background(), l); err != nil {
				t.Fatal(err)
			}
			if peak := meter.Peak(); peak != tc.wantPeak {
				t.Errorf("peak concurrency: got %d want %d", peak, tc.wantPeak)
			}
		})
	}
}

// TestApproverStaysSequential: even on the concurrent path the permission
// check itself runs one call at a time, in call order, so an Approver written
// for the old loop keeps working unchanged.
func TestApproverStaysSequential(t *testing.T) {
	var (
		mu     sync.Mutex
		seen   []string
		inside int
		peak   int
	)
	fake := &llm.Fake{Responses: []llm.Response{
		toolCallsResponse(call("c1", "a_tool"), call("c2", "b_tool")),
		llm.TextResponse("done", usage(1, 1)),
	}}
	meter := &concurrencyMeter{}
	l := parallelLoop(t, fake, rendezvousTools(t, meter, "a_tool", "b_tool")...)
	l.AutoApprove = func(llm.ToolCall) bool { return true }
	l.Approve = func(_ context.Context, c llm.ToolCall) (Ruling, error) {
		mu.Lock()
		inside++
		if inside > peak {
			peak = inside
		}
		seen = append(seen, c.ID)
		mu.Unlock()
		time.Sleep(5 * time.Millisecond)
		mu.Lock()
		inside--
		mu.Unlock()
		return Ruling{Decision: Allow}, nil
	}

	if _, _, err := runWithEvents(t, context.Background(), l); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if peak != 1 {
		t.Errorf("approver concurrency: got %d want 1", peak)
	}
	if want := []string{"c1", "c2"}; !equalStrings(seen, want) {
		t.Errorf("approver call order: got %v want %v", seen, want)
	}
	if meter.Peak() != 2 {
		t.Errorf("tools still had to run concurrently, peak %d", meter.Peak())
	}
}

// TestParallelDenialKeepsOrder: a denial mixed into a concurrent turn is still
// logged in call order, and the run still aborts on the abort decision.
func TestParallelDeniedCallAborts(t *testing.T) {
	meter := &concurrencyMeter{}
	fake := &llm.Fake{Responses: []llm.Response{
		toolCallsResponse(call("c1", "a_tool"), call("c2", "b_tool")),
		llm.TextResponse("done", usage(1, 1)),
	}}
	l := parallelLoop(t, fake, countingTool("a_tool", meter), countingTool("b_tool", meter))
	l.AutoApprove = func(llm.ToolCall) bool { return true }
	// A policy that lies about being auto-approvable still must not run the
	// tool: the ruling, not the predicate, decides.
	l.Approve = func(_ context.Context, c llm.ToolCall) (Ruling, error) {
		if c.ID == "c1" {
			return Ruling{Decision: DenyAbort, Reason: DeniedByPolicy}, nil
		}
		return Ruling{Decision: Allow}, nil
	}

	events, _, err := runWithEvents(t, context.Background(), l)
	if err == nil {
		t.Fatal("expected the run to abort on the denial")
	}
	if got, want := eventIDs(events, "tool_result"), []string{"c1", "c2"}; !equalStrings(got, want) {
		t.Errorf("tool_result order: got %v want %v", got, want)
	}
	if peak := meter.Peak(); peak != 1 {
		t.Errorf("the denied call must not have run: peak %d want 1", peak)
	}
}

// TestParallelContextCancellationAwaitsCalls: when the run's context ends
// mid-flight the loop still waits for every goroutine it started, so no tool
// keeps running (or writing to the session) after RunMessages returned.
func TestParallelContextCancellationAwaitsCalls(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var (
		mu       sync.Mutex
		finished int
		started  = make(chan struct{}, 2)
	)
	cancelable := func(name string) tools.Tool {
		return scriptedTool{name: name, run: func(ctx context.Context, _ string) (string, error) {
			started <- struct{}{}
			<-ctx.Done()
			// A slow unwind is the interesting case: the loop must wait for
			// it rather than return while the goroutine is still alive.
			time.Sleep(20 * time.Millisecond)
			mu.Lock()
			finished++
			mu.Unlock()
			return name + "-aborted", nil
		}}
	}
	fake := &llm.Fake{Responses: []llm.Response{
		toolCallsResponse(call("c1", "a_tool"), call("c2", "b_tool")),
		llm.TextResponse("done", usage(1, 1)),
	}}
	l := parallelLoop(t, fake, cancelable("a_tool"), cancelable("b_tool"))

	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _, _ = runWithEvents(t, ctx, l)
	}()

	for range 2 {
		select {
		case <-started:
		case <-time.After(rendezvousTimeout):
			t.Fatal("both calls did not start")
		}
	}
	cancel()

	select {
	case <-done:
	case <-time.After(rendezvousTimeout):
		t.Fatal("the loop did not return after cancellation")
	}
	mu.Lock()
	defer mu.Unlock()
	if finished != 2 {
		t.Errorf("the loop returned before both goroutines finished: %d of 2", finished)
	}
}

// TestParallelLeavesNoGoroutines: the dispatcher owns every goroutine it
// starts and joins them all before the turn ends (docs/engineering.md §5.4).
func TestParallelLeavesNoGoroutines(t *testing.T) {
	before := runtime.NumGoroutine()

	for range 5 {
		meter := &concurrencyMeter{}
		fake := &llm.Fake{Responses: []llm.Response{
			toolCallsResponse(call("c1", "a_tool"), call("c2", "b_tool"), call("c3", "c_tool")),
			llm.TextResponse("done", usage(1, 1)),
		}}
		l := parallelLoop(t, fake,
			countingTool("a_tool", meter), countingTool("b_tool", meter), countingTool("c_tool", meter))
		if _, _, err := runWithEvents(t, context.Background(), l); err != nil {
			t.Fatal(err)
		}
	}

	// Polled, not sampled once: a finished goroutine can take a moment to be
	// reaped, and a flaky leak check is worse than no leak check.
	deadline := time.Now().Add(2 * time.Second)
	for {
		after := runtime.NumGoroutine()
		if after <= before {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("goroutines leaked: %d before, %d after", before, after)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// TestParallelProgressStaysOrdered: progress events are emitted from the
// loop's own goroutine in call order, so a -v run reads as one story even
// though the tools finished in the other order.
func TestParallelProgressStaysOrdered(t *testing.T) {
	slowStarted, fastStarted := make(chan struct{}), make(chan struct{})
	slow := scriptedTool{name: "slow_tool", run: func(context.Context, string) (string, error) {
		close(slowStarted)
		waitFor(t, fastStarted, "the second call to start")
		time.Sleep(20 * time.Millisecond)
		return "slow-result", nil
	}}
	fast := scriptedTool{name: "fast_tool", run: func(context.Context, string) (string, error) {
		close(fastStarted)
		waitFor(t, slowStarted, "the first call to start")
		return "fast-result", nil
	}}
	fake := &llm.Fake{Responses: []llm.Response{
		toolCallsResponse(call("c1", "slow_tool"), call("c2", "fast_tool")),
		llm.TextResponse("done", usage(1, 1)),
	}}
	l := parallelLoop(t, fake, slow, fast)
	l.Clock = safeStepClock(time.Second)
	events := recordProgress(l)

	if _, err := l.Run(context.Background(), "task"); err != nil {
		t.Fatal(err)
	}

	want := []string{
		"turn 1: model requested slow_tool {}",
		"turn 1: model requested fast_tool {}",
		"turn 1: slow_tool ok",
		"turn 1: fast_tool ok",
	}
	got := *events
	if len(got) < len(want) {
		t.Fatalf("progress events: got %v", got)
	}
	for i, w := range want {
		if !strings.HasPrefix(got[i], w) {
			t.Errorf("progress[%d]: got %q want prefix %q", i, got[i], w)
		}
	}
}

// safeStepClock is stepClock for the concurrent path: the loop reads the clock
// from the tool goroutines when a progress hook is installed, so an injected
// clock must be race-free.
func safeStepClock(step time.Duration) Clock {
	var mu sync.Mutex
	now := time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC)
	return func() time.Time {
		mu.Lock()
		defer mu.Unlock()
		now = now.Add(step)
		return now
	}
}

// equalStrings compares two string slices, treating nil and empty as equal.
func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
