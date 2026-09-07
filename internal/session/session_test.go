package session

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"
)

// update refreshes golden files: go test ./internal/session -update
var update = flag.Bool("update", false, "rewrite golden files")

// fixedClock ticks one second per call from a fixed origin, making session
// output byte-stable for the golden test.
func fixedClock() Clock {
	t := time.Date(2026, 8, 10, 12, 0, 0, 0, time.UTC)
	return func() time.Time {
		t = t.Add(time.Second)
		return t
	}
}

func TestWriterGolden(t *testing.T) {
	dir := t.TempDir()
	w, err := New(dir, Options{Clock: fixedClock(), Secrets: []string{"sk-supersecret"}})
	if err != nil {
		t.Fatal(err)
	}

	w.RunStart("test-model", "scan the logs")
	// A turn that carried reasoning reports its SIZE only - the log never
	// carries the thinking itself.
	w.LLMResponse(LLMResponse{Turn: 1, Content: "let me read the log", ToolCallIDs: []string{"call_1"}, InputTokens: 100, OutputTokens: 20, FinishReason: "tool_calls", ReasoningBytes: len(`"the log is where the errors are"`)})
	w.ToolCall("call_1", "fs_read", `{"path":"app.log"}`)
	// The secret arrives via tool output and must be redacted by value.
	w.ToolResult(ToolResult{CallID: "call_1", Tool: "fs_read", Result: "line with sk-supersecret token", Outcome: OutcomeOK})
	// A second result whose text was cut to the tool-result byte cap - this is
	// the one line in the golden that carries "truncated":true, proving the
	// untruncated call above still omits the key.
	w.ToolResult(ToolResult{CallID: "call_2", Tool: "fs_read", Result: "clipped output", Outcome: OutcomeOK, Truncated: true})
	// The second turn's prompt was partly served from the provider's cache,
	// so this is the one line in the golden carrying the v1.7 cache keys -
	// which makes turn 1 above the proof that a cache-less turn omits them.
	w.LLMResponse(LLMResponse{Turn: 2, Content: "all clear", InputTokens: 150, OutputTokens: 30, FinishReason: "stop", CacheReadTokens: 120, CacheWriteTokens: 25})
	w.RunEnd("success", 0, 2, 1, 300, 0, 1500*time.Millisecond)

	got, err := os.ReadFile(w.Path())
	if err != nil {
		t.Fatal(err)
	}

	goldenPath := filepath.Join("testdata", "golden", "session.jsonl")
	if *update {
		if err := os.MkdirAll(filepath.Dir(goldenPath), 0o750); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(goldenPath, got, 0o600); err != nil { //nolint:gosec // G703: goldenPath is a fixed testdata constant, not tainted input.
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

// TestWriterAppendsUnconditionally pins the append-only property as a SYSCALL
// guarantee rather than a single-writer convention (live-test finding B-A04):
// the writer must never seek back over bytes it did not write. The scenario is
// the cheap way to observe the O_APPEND flag from Go without a dependency on
// x/sys: something else grows the file between two of the writer's own writes.
// Without O_APPEND the writer's file offset is stale and the second event
// overwrites the foreign bytes; with it, every write lands at the current end
// of file.
func TestWriterAppendsUnconditionally(t *testing.T) {
	dir := t.TempDir()
	w, err := New(dir, Options{Clock: fixedClock()})
	if err != nil {
		t.Fatal(err)
	}
	w.RunStart("test-model", "first")

	// A second writer - a rotator, an operator's `echo >>`, a concurrent run
	// that got the same name - extends the file behind the writer's back.
	const foreign = "{\"type\":\"foreign\"}\n"
	f, err := os.OpenFile(w.Path(), os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(foreign); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	w.RunEnd("success", 0, 1, 0, 10, 0, time.Second)

	data, err := os.ReadFile(w.Path())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), foreign) {
		t.Errorf("the writer overwrote bytes it did not write; log:\n%s", data)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 3 || !strings.Contains(lines[2], `"run_end"`) {
		t.Errorf("run_end must land after the foreign line; log:\n%s", data)
	}
}

// TestLLMResponseReasoningBytes pins the v1.4 field: an assistant turn that
// carried a reasoning payload reports its SIZE, and a turn without one omits
// the field entirely (omitempty, so pre-1.4 consumers see the log they knew).
//
// SECURITY: the reasoning CONTENT is deliberately absent from the log. It is
// the model's unfiltered scratchpad - it can restate a secret the redactor
// never saw as a value - and replay does not need it. The size is what an
// operator asking "why did this turn cost that much?" actually needs.
func TestLLMResponseReasoningBytes(t *testing.T) {
	dir := t.TempDir()
	w, err := New(dir, Options{Clock: fixedClock()})
	if err != nil {
		t.Fatal(err)
	}
	reasoning := `{"type":"thinking","thinking":"the answer is 4"}`
	w.LLMResponse(LLMResponse{Turn: 1, Content: "thinking hard", InputTokens: 10, OutputTokens: 5, FinishReason: "stop", ReasoningBytes: len(reasoning)})
	w.LLMResponse(LLMResponse{Turn: 2, Content: "no reasoning here", InputTokens: 10, OutputTokens: 5, FinishReason: "stop"})

	data, err := os.ReadFile(w.Path())
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("want 2 events, got:\n%s", data)
	}
	if want := fmt.Sprintf(`"reasoning_bytes":%d`, len(reasoning)); !strings.Contains(lines[0], want) {
		t.Errorf("first event does not carry %s:\n%s", want, lines[0])
	}
	if strings.Contains(lines[1], "reasoning_bytes") {
		t.Errorf("a turn with no reasoning must omit the field:\n%s", lines[1])
	}
	if strings.Contains(string(data), "the answer is 4") {
		t.Errorf("reasoning CONTENT leaked into the log:\n%s", data)
	}
}

// TestLLMResponseCacheTokens pins the llm_response half of the v1.7 cache
// accounting: a turn whose prompt was partly served from a provider cache
// reports both counts, and a turn that touched no cache omits both keys
// entirely (omitempty), so a pre-v1.7 consumer sees byte-for-byte the line it
// already knew.
func TestLLMResponseCacheTokens(t *testing.T) {
	dir := t.TempDir()
	w, err := New(dir, Options{Clock: fixedClock()})
	if err != nil {
		t.Fatal(err)
	}
	w.LLMResponse(LLMResponse{
		Turn: 1, Content: "cached turn", InputTokens: 1000, OutputTokens: 20,
		FinishReason: "stop", CacheReadTokens: 800, CacheWriteTokens: 150,
	})
	w.LLMResponse(LLMResponse{Turn: 2, Content: "cold turn", InputTokens: 1000, OutputTokens: 20, FinishReason: "stop"})

	data, err := os.ReadFile(w.Path())
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("want 2 events, got:\n%s", data)
	}
	for _, want := range []string{`"cache_read_tokens":800`, `"cache_write_tokens":150`} {
		if !strings.Contains(lines[0], want) {
			t.Errorf("cached turn does not carry %s:\n%s", want, lines[0])
		}
	}
	for _, absent := range []string{"cache_read_tokens", "cache_write_tokens"} {
		if strings.Contains(lines[1], absent) {
			t.Errorf("a turn that read no cache must omit %s:\n%s", absent, lines[1])
		}
	}
}

// TestRunEndCacheReadTokens pins the run_end half of v1.7: the run's
// cumulative cache-READ total, and only that one - a write count is a
// per-turn fact about where a breakpoint landed, while "how much of this run
// came from cache" is the number an operator bills against. Zero is absent,
// which is also the pre-v1.7 shape.
func TestRunEndCacheReadTokens(t *testing.T) {
	tests := []struct {
		name      string
		cacheRead int
		want      string // "" means the key must be absent
	}{
		{"no cache reads", 0, ""},
		{"cache reads", 5, `"cache_read_tokens":5`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w, err := New(t.TempDir(), Options{Clock: fixedClock()})
			if err != nil {
				t.Fatal(err)
			}
			path := w.Path()
			w.RunEnd("success", 0, 1, 0, 1000, tt.cacheRead, time.Second)

			data, err := os.ReadFile(path) //nolint:gosec // G304: path comes from t.TempDir
			if err != nil {
				t.Fatal(err)
			}
			line := strings.TrimSpace(string(data))
			if tt.want == "" {
				if strings.Contains(line, "cache_read_tokens") {
					t.Errorf("a run with no cache reads must omit the key:\n%s", line)
				}
				return
			}
			if !strings.Contains(line, tt.want) {
				t.Errorf("run_end does not carry %s:\n%s", tt.want, line)
			}
			// The write count is per-turn only; run_end must never carry it.
			if strings.Contains(line, "cache_write_tokens") {
				t.Errorf("run_end must not carry a cache write total:\n%s", line)
			}
		})
	}
}

func TestRedaction(t *testing.T) {
	dir := t.TempDir()
	w, err := New(dir, Options{Clock: fixedClock(), Secrets: []string{"sk-verysecret", "tiny"}})
	if err != nil {
		t.Fatal(err)
	}
	w.ToolResult(ToolResult{CallID: "call_1", Tool: "t", Result: "key=sk-verysecret and tiny goes too", Outcome: OutcomeOK})
	// The model may echo a secret in its answer text; that path must be
	// redacted too now that content is logged.
	w.LLMResponse(LLMResponse{Turn: 2, Content: "the key is sk-verysecret", InputTokens: 1, OutputTokens: 1, FinishReason: "stop"})
	w.RunEnd("success", 0, 1, 1, 10, 0, time.Second)

	data, _ := os.ReadFile(w.Path())
	if strings.Contains(string(data), "sk-verysecret") {
		t.Error("secret leaked into session log")
	}
	if !strings.Contains(string(data), "[REDACTED]") {
		t.Error("redaction marker missing")
	}
	// SECURITY: registered secrets are redacted unconditionally - a short
	// DB password is still a credential (Codex review finding F4).
	if strings.Contains(string(data), "tiny") {
		t.Error("short registered secrets must be redacted too")
	}
}

func TestClipLongFields(t *testing.T) {
	dir := t.TempDir()
	w, err := New(dir, Options{Clock: fixedClock()})
	if err != nil {
		t.Fatal(err)
	}
	w.ToolResult(ToolResult{CallID: "call_1", Tool: "t", Result: strings.Repeat("x", maxLoggedField*2), Outcome: OutcomeOK})
	w.RunEnd("success", 0, 1, 1, 10, 0, time.Second)

	data, _ := os.ReadFile(w.Path())
	if len(data) > maxLoggedField+1024 {
		t.Errorf("event not clipped: %d bytes", len(data))
	}
	if !strings.Contains(string(data), "[clipped]") {
		t.Error("clip marker missing")
	}
}

// TestReplaySource: the log must carry everything needed to reconstruct the
// conversation - assistant text and the tool-call linkage IDs. Token counts
// alone cannot answer "what did the agent say at 03:00?" (review finding
// P1-3; log = session = replay is the package contract).
func TestReplaySource(t *testing.T) {
	dir := t.TempDir()
	w, err := New(dir, Options{Clock: fixedClock()})
	if err != nil {
		t.Fatal(err)
	}
	w.LLMResponse(LLMResponse{Turn: 1, Content: "reading the log now", ToolCallIDs: []string{"call_9"}, InputTokens: 10, OutputTokens: 5, FinishReason: "tool_calls"})
	w.ToolCall("call_9", "fs_read", `{"path":"x"}`)
	w.ToolResult(ToolResult{CallID: "call_9", Tool: "fs_read", Result: "data", Outcome: OutcomeOK})
	w.RunEnd("success", 0, 1, 1, 15, 0, time.Second)

	data, _ := os.ReadFile(w.Path())
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")

	var llmEv, callEv, resultEv Event
	if err := json.Unmarshal([]byte(lines[0]), &llmEv); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[1]), &callEv); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(lines[2]), &resultEv); err != nil {
		t.Fatal(err)
	}

	if llmEv.Content != "reading the log now" {
		t.Errorf("llm_response must log the assistant text, got %q", llmEv.Content)
	}
	if len(llmEv.ToolCallIDs) != 1 || llmEv.ToolCallIDs[0] != "call_9" {
		t.Errorf("llm_response must log the requested tool-call IDs, got %v", llmEv.ToolCallIDs)
	}
	if callEv.CallID != "call_9" || resultEv.CallID != "call_9" {
		t.Errorf("tool_call/tool_result must carry the linking ID, got %q / %q", callEv.CallID, resultEv.CallID)
	}
}

// TestClipRuneBoundary: clipping must back up to a rune boundary - a cut in
// the middle of a multi-byte rune would make json.Marshal emit U+FFFD.
func TestClipRuneBoundary(t *testing.T) {
	dir := t.TempDir()
	w, err := New(dir, Options{Clock: fixedClock()})
	if err != nil {
		t.Fatal(err)
	}
	// "€" is 3 bytes; maxLoggedField is not a multiple of 3, so a naive
	// byte-index cut lands mid-rune.
	w.ToolResult(ToolResult{CallID: "id", Tool: "t", Result: strings.Repeat("€", maxLoggedField), Outcome: OutcomeOK})
	w.RunEnd("success", 0, 1, 1, 1, 0, time.Second)

	data, _ := os.ReadFile(w.Path())
	var ev Event
	if err := json.Unmarshal([]byte(strings.SplitN(string(data), "\n", 2)[0]), &ev); err != nil {
		t.Fatal(err)
	}
	clipped := strings.TrimSuffix(ev.Result, "...[clipped]")
	if !utf8.ValidString(clipped) || strings.ContainsRune(ev.Result, utf8.RuneError) {
		t.Errorf("clip cut inside a rune: %q...", ev.Result[:24])
	}
}

// TestNilWriterIsSafe: a nil *Writer must be a no-op, because callers use it
// directly when session_dir is unset.
func TestNilWriterIsSafe(t *testing.T) {
	var w *Writer
	w.RunStart("m", "t")
	w.LLMResponse(LLMResponse{Turn: 1, Content: "c", ToolCallIDs: []string{"id"}, InputTokens: 1, OutputTokens: 1, FinishReason: "stop"})
	w.ToolCall("id", "t", "{}")
	w.ToolResult(ToolResult{CallID: "id", Tool: "t", Result: "r", Outcome: OutcomeOK})
	w.MCPConnect(MCPConnect{Server: "s", Transport: "stdio", OK: true})
	w.MCPToolsListed(MCPToolsListed{Server: "s", Tools: []MCPToolListed{{Name: "s__t"}}})
	w.MCPDisconnect(MCPDisconnect{Server: "s", Reason: "run_end"})
	w.SetMCPErrors(2)
	w.RunEnd("success", 0, 1, 1, 1, 0, time.Second)
	if w.Path() != "" {
		t.Error("nil writer path should be empty")
	}
}

func TestDuplicateFilenameFails(t *testing.T) {
	dir := t.TempDir()
	// Same fixed instant → same filename → second create must fail loudly
	// instead of interleaving two runs into one file.
	same := func() time.Time { return time.Date(2026, 8, 10, 12, 0, 0, 0, time.UTC) }
	if _, err := New(dir, Options{Clock: Clock(same)}); err != nil {
		t.Fatal(err)
	}
	if _, err := New(dir, Options{Clock: Clock(same)}); err == nil {
		t.Error("second writer with identical timestamp must fail")
	}
}

func TestSummary(t *testing.T) {
	tests := []struct {
		name      string
		ok        bool
		turns     int
		toolCalls int
		tokens    int
		cached    int
		want      string
	}{
		{"plural", true, 8, 3, 41234, 0, "✓ 8 turns, 3 tool calls, 41.2k tokens, 34.0s"},
		{"failure mark", false, 8, 3, 900, 0, "✗ 8 turns, 3 tool calls, 900 tokens, 34.0s"},
		{"millions of tokens", true, 8, 3, 2_500_000, 0, "✓ 8 turns, 3 tool calls, 2.5M tokens, 34.0s"},
		// The one-turn run is the common case for a cron agent that answers
		// without calling a tool, so "1 turns, 0 tool calls" was the line
		// operators read most often.
		{"single turn", true, 1, 1, 12, 0, "✓ 1 turn, 1 tool call, 12 tokens, 34.0s"},
		{"zero counts stay plural", true, 0, 0, 0, 0, "✓ 0 turns, 0 tool calls, 0 tokens, 34.0s"},
		// The cached parenthetical: present only when the run actually read
		// from a cache, so an uncached run prints the pre-v0.3 line unchanged
		// (the four rows above are that claim).
		{"cached share", true, 8, 3, 41000, 28000, "✓ 8 turns, 3 tool calls, 41.0k tokens (28.0k cached), 34.2s"},
		// Small cached counts stay bare integers, like the token total does.
		{"small cached share", true, 1, 0, 1200, 999, "✓ 1 turn, 0 tool calls, 1.2k tokens (999 cached), 34.2s"},
		// A failed run still reports what its cache reads saved.
		{"cached on failure", false, 2, 0, 5000, 1000, "✗ 2 turns, 0 tool calls, 5.0k tokens (1.0k cached), 34.2s"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// The cached rows want a fractional second in the line, so the
			// duration differs per row: 34.0s for the uncached legacy rows.
			d := 34 * time.Second
			if tt.cached > 0 {
				d = 34200 * time.Millisecond
			}
			got := Summary(tt.ok, tt.turns, tt.toolCalls, tt.tokens, tt.cached, d)
			if got != tt.want {
				t.Errorf("Summary: got %q want %q", got, tt.want)
			}
		})
	}
}

// intPtr is the one-line helper the tool_result tests need for the optional
// exit_code field ("known" vs "not applicable" is the whole point of *int).
func intPtr(n int) *int { return &n }

// firstEvent decodes the first line of a session file as a generic JSON object,
// which is what a log consumer actually sees - a struct would silently accept a
// field that was never written.
func firstEvent(t *testing.T, path string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(path) //nolint:gosec // G304: path comes from the writer under test.
	if err != nil {
		t.Fatal(err)
	}
	var ev map[string]any
	if err := json.Unmarshal([]byte(strings.SplitN(string(data), "\n", 2)[0]), &ev); err != nil {
		t.Fatal(err)
	}
	return ev
}

// TestToolResultOutcomeFields walks every value of the outcome enum published
// in docs/contracts/jsonl-events.md. The table IS the contract: a value that
// stops being serialized, or is renamed, breaks a log consumer that greps for
// it, so each spelling is pinned literally rather than via the constant.
func TestToolResultOutcomeFields(t *testing.T) {
	tests := []struct {
		name         string
		result       ToolResult
		wantOutcome  string
		wantExitCode any // nil = the key must be absent
		wantIsErr    bool
	}{
		{
			name:        "ok",
			result:      ToolResult{CallID: "c1", Tool: "shell", Result: "hello", Outcome: OutcomeOK},
			wantOutcome: "ok",
		},
		{
			name:        "timeout",
			result:      ToolResult{CallID: "c1", Tool: "shell", Result: "command timed out after 1s", Outcome: OutcomeTimeout},
			wantOutcome: "timeout",
		},
		{
			// The whole point of the enum: a non-zero exit is NOT an error
			// (`grep` exits 1 when it finds nothing), so is_error stays absent
			// and the status travels in exit_code.
			name:         "nonzero exit carries its status",
			result:       ToolResult{CallID: "c1", Tool: "shell", Result: "exit status 3", Outcome: OutcomeNonzeroExit, ExitCode: intPtr(3)},
			wantOutcome:  "nonzero_exit",
			wantExitCode: 3.0,
		},
		{
			name:        "aborted",
			result:      ToolResult{CallID: "c1", Tool: "shell", Result: "command aborted", Outcome: OutcomeAborted},
			wantOutcome: "aborted",
		},
		{
			name:        "denied by policy",
			result:      ToolResult{CallID: "c1", Tool: "shell", Result: "permission denied", IsErr: true, Outcome: OutcomeDeniedPolicy},
			wantOutcome: "denied_policy",
			wantIsErr:   true,
		},
		{
			name:        "denied for lack of a TTY",
			result:      ToolResult{CallID: "c1", Tool: "shell", Result: "permission denied", IsErr: true, Outcome: OutcomeDeniedNoTTY},
			wantOutcome: "denied_no_tty",
			wantIsErr:   true,
		},
		{
			name:        "ask refused",
			result:      ToolResult{CallID: "c1", Tool: "shell", Result: "permission denied", IsErr: true, Outcome: OutcomeAskRefused},
			wantOutcome: "ask_refused",
			wantIsErr:   true,
		},
		{
			name:        "harness error",
			result:      ToolResult{CallID: "c1", Tool: "ghost", Result: "error: unknown tool", IsErr: true, Outcome: OutcomeError},
			wantOutcome: "error",
			wantIsErr:   true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w, err := New(t.TempDir(), Options{Clock: fixedClock()})
			if err != nil {
				t.Fatal(err)
			}
			w.ToolResult(tt.result)
			w.RunEnd("success", 0, 1, 1, 1, 0, time.Second)

			ev := firstEvent(t, w.Path())
			if got := ev["outcome"]; got != tt.wantOutcome {
				t.Errorf("outcome = %v, want %q", got, tt.wantOutcome)
			}
			got, present := ev["exit_code"]
			if tt.wantExitCode == nil && present {
				t.Errorf("exit_code must be absent when the exit status is unknown, got %v", got)
			}
			if tt.wantExitCode != nil && got != tt.wantExitCode {
				t.Errorf("exit_code = %v, want %v", got, tt.wantExitCode)
			}
			if isErr, _ := ev["is_error"].(bool); isErr != tt.wantIsErr {
				t.Errorf("is_error = %v, want %v (it is frozen: harness dispatch failures only)", isErr, tt.wantIsErr)
			}
			if want := float64(len(tt.result.Result)); ev["result_bytes"] != want {
				t.Errorf("result_bytes = %v, want %v", ev["result_bytes"], want)
			}
		})
	}
}

// TestToolResultBytesArePreClip is the C-5 finding in one assertion: the log
// clips a result at 8 KiB while the model saw up to 64 KiB, and without a
// recorded size nothing in the file says how much was dropped.
func TestToolResultBytesArePreClip(t *testing.T) {
	w, err := New(t.TempDir(), Options{Clock: fixedClock()})
	if err != nil {
		t.Fatal(err)
	}
	const size = maxLoggedField * 3
	w.ToolResult(ToolResult{CallID: "c1", Tool: "shell", Result: strings.Repeat("x", size), Outcome: OutcomeOK})
	w.RunEnd("success", 0, 1, 1, 1, 0, time.Second)

	ev := firstEvent(t, w.Path())
	if got := ev["result_bytes"]; got != float64(size) {
		t.Errorf("result_bytes = %v, want %d (the FULL size, not the clipped one)", got, size)
	}
	if logged, _ := ev["result"].(string); len(logged) > maxLoggedField+64 {
		t.Errorf("the result itself must still be clipped, got %d bytes", len(logged))
	}
}

// TestToolResultTruncatedFlag pins the v1.6 additive field: a tool_result
// whose text was cut to the byte cap says so with "truncated":true, and one
// that was not cut omits the key rather than writing "truncated":false.
func TestToolResultTruncatedFlag(t *testing.T) {
	w, err := New(t.TempDir(), Options{Clock: fixedClock()})
	if err != nil {
		t.Fatal(err)
	}
	w.ToolResult(ToolResult{CallID: "c1", Tool: "fs_read", Result: "x", Outcome: OutcomeOK, Truncated: true})
	w.ToolResult(ToolResult{CallID: "c2", Tool: "fs_read", Result: "y", Outcome: OutcomeOK})
	w.RunEnd("success", 0, 1, 2, 2, 0, time.Second)

	data, err := os.ReadFile(w.Path())
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	if !strings.Contains(lines[0], `"truncated":true`) {
		t.Fatalf("first event lacks truncated: %s", lines[0])
	}
	if strings.Contains(lines[1], "truncated") {
		t.Fatalf("second event must omit truncated: %s", lines[1])
	}
}

// TestOutcomeFieldsAreToolResultOnly guards the additive-change promise: the
// new keys belong to tool_result, and every other event type must serialize
// exactly as it did before v0.1.0 (docs/contracts/jsonl-events.md, "Change
// policy"). run_end keeps its own exit_code, which is a different field with
// the same name - the assertion below is per event type for that reason.
func TestOutcomeFieldsAreToolResultOnly(t *testing.T) {
	w, err := New(t.TempDir(), Options{Clock: fixedClock()})
	if err != nil {
		t.Fatal(err)
	}
	w.RunStart("m", "task")
	w.LLMResponse(LLMResponse{Turn: 1, Content: "text", ToolCallIDs: []string{"c1"}, InputTokens: 1, OutputTokens: 1, FinishReason: "tool_calls"})
	w.ToolCall("c1", "shell", "{}")
	w.ToolResult(ToolResult{CallID: "c1", Tool: "shell", Result: "out", Outcome: OutcomeOK})
	w.RunEnd("success", 0, 1, 1, 2, 0, time.Second)

	data, err := os.ReadFile(w.Path())
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var ev map[string]any
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatal(err)
		}
		isToolResult := ev["type"] == "tool_result"
		for _, field := range []string{"outcome", "result_bytes"} {
			if _, present := ev[field]; present != isToolResult {
				t.Errorf("%v event: %q present = %v, want %v (line: %s)", ev["type"], field, present, isToolResult, line)
			}
		}
		if ev["type"] == "run_end" {
			if _, present := ev["exit_code"]; !present {
				t.Errorf("run_end lost its exit_code: %s", line)
			}
		}
	}
}

// TestRedactorOverlappingSecrets is a regression test: redaction used to run
// in slice order, so a short secret that is a PREFIX of a longer one ("sk-"
// before "sk-secret-value") consumed the prefix and left the rest of the long
// secret in the log. Longest-first ordering must make the result independent
// of the order the secrets were registered in.
func TestRedactorOverlappingSecrets(t *testing.T) {
	const long = "sk-secret-value"
	tests := []struct {
		name    string
		secrets []string
	}{
		{"short first", []string{"sk-", long}},
		{"long first", []string{long, "sk-"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Redactor(tt.secrets)("token=" + long + " end")
			if strings.Contains(got, "secret-value") {
				t.Errorf("secret leaked: %q", got)
			}
			if want := "token=[REDACTED] end"; got != want {
				t.Errorf("got %q, want %q", got, want)
			}
		})
	}
}

// TestGoldenMCP pins the v1.2 MCP additions (mcp_connect, mcp_tools_listed,
// mcp_disconnect, the tool_error/indeterminate outcomes and
// run_end.mcp_errors) to a golden file, so a change to the wire shape of any
// of them shows up as a diff in a `contract:`-sized review rather than as a
// silent break of a log consumer.
func TestGoldenMCP(t *testing.T) {
	dir := t.TempDir()
	// SECURITY: the header value is a configured secret, so it must be
	// redacted where it lands - here inside a connect error message.
	w, err := New(dir, Options{Clock: fixedClock(), Secrets: []string{"Bearer sekrit"}})
	if err != nil {
		t.Fatal(err)
	}

	w.RunStart("test-model", "list issues")
	w.MCPConnect(MCPConnect{
		Server: "github", Transport: "http", OK: true, DurationMS: 12,
		ProtocolVersion: "2025-06-18", ServerName: "gh", ServerVersion: "1.0",
		SessionFP: "ab12cd34", ToolCount: 2, Auth: "oauth",
	})
	w.MCPConnect(MCPConnect{
		Server: "flaky", Transport: "stdio", OK: false,
		ErrorClass: "network", Error: `handshake failed with header "Bearer sekrit"`,
		DurationMS: 3,
	})
	w.MCPToolsListed(MCPToolsListed{
		Server: "github",
		Tools: []MCPToolListed{{
			Name: "github__x", SHA256: "9f2c" + strings.Repeat("0", 60),
			Bytes: 120, Annotations: map[string]bool{"readOnly": true},
		}},
		TotalBytes: 120,
		Skipped:    []MCPSkippedTool{{Name: "a.b", Reason: "excluded"}},
	})
	w.ToolResult(ToolResult{CallID: "call_1", Tool: "github__x", Result: "not found", Outcome: OutcomeToolError})
	w.ToolResult(ToolResult{CallID: "call_2", Tool: "github__x", Result: "no response", Outcome: OutcomeIndeterminate})
	w.MCPDisconnect(MCPDisconnect{Server: "github", Reason: "run_end"})
	w.SetMCPErrors(1)
	w.RunEnd("success", 0, 2, 2, 300, 0, 1500*time.Millisecond)

	got, err := os.ReadFile(w.Path())
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(got), "sekrit") {
		t.Errorf("secret survived into the log:\n%s", got)
	}
	if !strings.Contains(string(got), "[REDACTED]") {
		t.Errorf("mcp_connect error was not redacted:\n%s", got)
	}
	// v1.3: the authenticated connect says so, and the one that used no
	// credential says nothing - `auth` is omitempty, and an absent field must
	// keep meaning "no OAuth", not "unknown".
	if strings.Count(string(got), `"auth":"oauth"`) != 1 {
		t.Errorf("mcp_connect.auth is not written exactly once:\n%s", got)
	}

	goldenPath := filepath.Join("testdata", "golden", "session-mcp.jsonl")
	if *update {
		if err := os.MkdirAll(filepath.Dir(goldenPath), 0o750); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(goldenPath, got, 0o600); err != nil { //nolint:gosec // G703: goldenPath is a fixed testdata constant, not tainted input.
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

// TestWriterConcurrentUse pins the Writer's own locking: the MCP connect phase
// (and, on the roadmap, parallel tool calls) drive one writer from several
// goroutines, and `go test -race` turns any unguarded access here into a
// failure.
func TestWriterConcurrentUse(t *testing.T) {
	dir := t.TempDir()
	w, err := New(dir, Options{Clock: fixedClock()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	path := w.Path()
	var wg sync.WaitGroup
	for i := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			w.MCPConnect(MCPConnect{Server: "s", Transport: "stdio", OK: true})
			w.SetMCPErrors(i)
			w.MCPDisconnect(MCPDisconnect{Server: "s", Reason: "reconnect"})
		}()
	}
	wg.Wait()
	w.RunEnd("success", 0, 1, 0, 0, 0, 0)

	data, rerr := os.ReadFile(path) //nolint:gosec // G304: path comes from t.TempDir
	if rerr != nil {
		t.Fatalf("reading log: %v", rerr)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 17 { // 8x(connect+disconnect) + run_end
		t.Fatalf("got %d lines, want 17", len(lines))
	}
	for _, line := range lines {
		var e Event
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Fatalf("torn event line %q: %v", line, err)
		}
	}
}

// TestSecretSetAddMidRun: the whole reason SecretSet exists. OAuth mints an
// access token AFTER the run started, and the sinks were wired at startup;
// a secret added later must be scrubbed by the same set every sink already
// holds.
func TestSecretSetAddMidRun(t *testing.T) {
	set := NewSecretSet([]string{"first"})
	if got := set.Redact("x first y late z"); strings.Contains(got, "first") {
		t.Fatalf("initial secret leaked: %q", got)
	}
	set.Add("late")
	if got := set.Redact("x late z"); strings.Contains(got, "late") {
		t.Fatalf("added secret leaked: %q", got)
	}
}

// TestSecretSetLongestFirstAfterAdd: the longest-first rule Redactor pins for
// a frozen slice must survive incremental Add - a refreshed token registered
// after a short one must still be replaced whole, not left with its tail in
// the log.
func TestSecretSetLongestFirstAfterAdd(t *testing.T) {
	set := NewSecretSet(nil)
	set.Add("sk-")
	set.Add("sk-longer-token")
	got := set.Redact("token=sk-longer-token end")
	if strings.Contains(got, "longer-token") {
		t.Fatalf("secret leaked: %q", got)
	}
	if want := "token=[REDACTED] end"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

// TestSecretSetEmptyValuesIgnored: replacing "" would corrupt every event, so
// empty values are dropped exactly as Redactor drops them.
func TestSecretSetEmptyValuesIgnored(t *testing.T) {
	set := NewSecretSet([]string{"", "tok"})
	set.Add("")
	if got, want := set.Redact("a tok b"), "a [REDACTED] b"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

// TestSecretSetRedactsJSONEscapedForm is the SECURITY regression for the
// review finding: a secret reaches the log through a PROVIDER's JSON error
// body, where the server re-encoded it as a JSON string. A value carrying a
// quote or a backslash then appears escaped (`a\"b`) and no longer equals the
// bytes registered here, so a literal-only search walked straight past it.
//
// The secret below is a fabricated value that merely LOOKS awkward; it is not
// a credential of any kind.
func TestSecretSetRedactsJSONEscapedForm(t *testing.T) {
	const fakeValue = `test"quote\secret`
	// The spelling a JSON encoder produces for that value, computed rather
	// than typed so the test cannot drift from what a server would emit.
	encoded, err := json.Marshal(fakeValue)
	if err != nil {
		t.Fatal(err)
	}
	escaped := string(encoded[1 : len(encoded)-1])
	if escaped == fakeValue {
		t.Fatalf("test is vacuous: %q needs a character JSON escapes", fakeValue)
	}

	set := NewSecretSet([]string{fakeValue})
	// A 400 body of the shape every OpenAI-compatible gateway echoes back:
	// the offending value, quoted, inside the provider's own JSON.
	snippet := `status 400: {"error":{"message":"Unrecognized value for 'x-api-key': ` + escaped + `","type":"invalid_request_error"}}`
	got := set.Redact(snippet)
	if strings.Contains(got, escaped) {
		t.Errorf("JSON-escaped secret survived redaction: %q", got)
	}
	// The literal spelling must keep working: the same value still reaches
	// plain-text sinks (tool output, argv echoes).
	if plain := set.Redact("key=" + fakeValue + " end"); strings.Contains(plain, fakeValue) {
		t.Errorf("literal secret survived redaction: %q", plain)
	}
	if !strings.Contains(got, "[REDACTED]") {
		t.Errorf("redaction marker missing: %q", got)
	}
}

// jsonInterior renders v as a JSON string and strips the encoder's quotes -
// the spelling a value has once a server has spliced it into a JSON document.
// Computed rather than typed so these tests cannot drift from what a real
// encoder emits.
func jsonInterior(t *testing.T, v string) string {
	t.Helper()
	encoded, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded[1 : len(encoded)-1])
}

// TestSecretSetRedactsDeepEscapedForms is the SECURITY regression for the
// review finding that one level of JSON escaping is not enough.
//
// Two shapes were walking past a literal search. A gateway that wraps the
// upstream response - already a JSON document - inside its OWN JSON error body
// escapes the escapes, so `a"b` arrives as `a\\\"b`. And an encoder that
// prefers unicode escapes spells the same quote `\u0022`, which shares no
// bytes with either of the forms that were registered.
//
// The secrets below are fabricated values that merely LOOK awkward; none of
// them is a credential of any kind.
func TestSecretSetRedactsDeepEscapedForms(t *testing.T) {
	const fakeValue = `test"quote\secret`
	const plainValue = "sk-plain-token"
	unicodeEscaped := strings.NewReplacer(`"`, `\u0022`, `\`, `\u005c`).Replace(fakeValue)

	tests := []struct {
		name    string
		secrets []string
		// spelling is the form of the secret the body carries; the body is
		// built around it so every case redacts the same way.
		spelling func(t *testing.T) string
	}{
		{
			name:     "literal",
			secrets:  []string{fakeValue},
			spelling: func(*testing.T) string { return fakeValue },
		},
		{
			name:     "one json level",
			secrets:  []string{fakeValue},
			spelling: func(t *testing.T) string { return jsonInterior(t, fakeValue) },
		},
		{
			name:    "two json levels: a gateway wrapping the upstream body",
			secrets: []string{fakeValue},
			spelling: func(t *testing.T) string {
				return jsonInterior(t, jsonInterior(t, fakeValue))
			},
		},
		{
			name:     "unicode escapes",
			secrets:  []string{fakeValue},
			spelling: func(*testing.T) string { return unicodeEscaped },
		},
		{
			name:    "unicode escapes wrapped once more",
			secrets: []string{fakeValue},
			spelling: func(t *testing.T) string {
				return jsonInterior(t, unicodeEscaped)
			},
		},
		{
			// The extra variants must not disturb longest-first ordering: a
			// short secret registered alongside a longer one it prefixes still
			// must not eat the prefix and leave the tail in the log.
			name:     "overlapping secrets keep longest-first",
			secrets:  []string{`test"`, fakeValue},
			spelling: func(t *testing.T) string { return jsonInterior(t, fakeValue) },
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spelling := tt.spelling(t)
			set := NewSecretSet(tt.secrets)
			body := `status 400: {"error":{"message":"bad key: ` + spelling + `"}}`
			got := set.Redact(body)
			if strings.Contains(got, "quote") || strings.Contains(got, "secret") {
				t.Errorf("secret survived redaction: %q", got)
			}
			if !strings.Contains(got, "[REDACTED]") {
				t.Errorf("redaction marker missing: %q", got)
			}
		})
	}

	// A secret made of ordinary token characters has exactly one spelling, so
	// the deeper variants must add nothing and change nothing.
	t.Run("plain secret unaffected", func(t *testing.T) {
		if got, want := len(secretVariants(plainValue)), 1; got != want {
			t.Errorf("variants of a plain secret: got %d want %d", got, want)
		}
		set := NewSecretSet([]string{plainValue})
		if got, want := set.Redact("key="+plainValue+" end"), "key=[REDACTED] end"; got != want {
			t.Errorf("got %q, want %q", got, want)
		}
	})

	// SECURITY: "" must still register nothing - a variant of the empty string
	// would replace every byte boundary in every event.
	t.Run("empty secret registers nothing", func(t *testing.T) {
		if got := secretVariants(""); len(got) != 0 {
			t.Errorf("variants of an empty secret: got %v want none", got)
		}
	})
}

// TestSecretSetConcurrent: the registry is shared by sinks running on
// different goroutines (the MCP stderr relays) while a token refresh adds to
// it, so Add and Redact must be safe together under -race.
func TestSecretSetConcurrent(t *testing.T) {
	set := NewSecretSet([]string{"seed"})
	var wg sync.WaitGroup
	for i := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range 50 {
				set.Add(strings.Repeat("t", i+1) + string(rune('a'+j%26)))
				if got := set.Redact("line with seed in it"); strings.Contains(got, "seed") {
					t.Errorf("seed leaked under concurrency: %q", got)
					return
				}
			}
		}()
	}
	wg.Wait()
}

// TestWriterRedactsSecretAddedAfterNew: the Writer must redact through the
// LIVE set, not through a copy frozen at New - that copy is the bug this task
// removes.
func TestWriterRedactsSecretAddedAfterNew(t *testing.T) {
	dir := t.TempDir()
	secrets := NewSecretSet(nil)
	w, err := New(dir, Options{Clock: fixedClock(), SecretSource: secrets})
	if err != nil {
		t.Fatal(err)
	}
	secrets.Add("tok-123")
	w.ToolResult(ToolResult{CallID: "c1", Tool: "mcp__x__y", Result: "auth used tok-123", Outcome: OutcomeOK})
	w.RunEnd("success", 0, 1, 1, 10, 0, time.Second)

	data, err := os.ReadFile(w.Path())
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "tok-123") {
		t.Errorf("mid-run secret leaked into the session log:\n%s", data)
	}
	if !strings.Contains(string(data), "[REDACTED]") {
		t.Errorf("session log lost the redaction marker:\n%s", data)
	}
}

// TestClipLimitConfigurable pins limits.max_logged_field's writer half:
// 0 keeps the 8192 default (today's behavior for every existing caller),
// a positive value becomes the bound, negative disables clipping entirely.
// The ...[clipped] marker still rides ON TOP of the limit (contract:
// limit + 12 bytes), and redaction still runs first in every mode.
func TestClipLimitConfigurable(t *testing.T) {
	const marker = "...[clipped]"
	long := strings.Repeat("a", 9000)
	cases := []struct {
		name       string
		maxField   int
		content    string
		wantLen    int // expected len of the logged content field
		wantMarker bool
	}{
		{"zero means the 8192 default", 0, long, maxLoggedField + len(marker), true},
		{"custom small bound", 16, long, 16 + len(marker), true},
		{"negative means unbounded", -1, long, 9000, false},
		{"under any bound is untouched", 16, "short", len("short"), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w, err := New(t.TempDir(), Options{Clock: fixedClock(), MaxLoggedField: tc.maxField})
			if err != nil {
				t.Fatal(err)
			}
			w.LLMResponse(LLMResponse{Turn: 1, Content: tc.content, FinishReason: "stop"})

			ev := firstLLMEvent(t, w.Path())
			if len(ev.Content) != tc.wantLen {
				t.Errorf("logged content is %d bytes, want %d", len(ev.Content), tc.wantLen)
			}
			if got := strings.HasSuffix(ev.Content, marker); got != tc.wantMarker {
				t.Errorf("clip marker present = %v, want %v", got, tc.wantMarker)
			}
		})
	}

	// SECURITY: redaction runs BEFORE the bound in every mode, unbounded
	// included - "no clipping" must never mean "no redaction".
	t.Run("unbounded still redacts", func(t *testing.T) {
		w, err := New(t.TempDir(), Options{Clock: fixedClock(), MaxLoggedField: -1, Secrets: []string{"sk-live"}})
		if err != nil {
			t.Fatal(err)
		}
		w.LLMResponse(LLMResponse{Turn: 1, Content: "key is sk-live" + long, FinishReason: "stop"})

		ev := firstLLMEvent(t, w.Path())
		if strings.Contains(ev.Content, "sk-live") {
			t.Error("secret leaked when clipping was disabled")
		}
		if !strings.Contains(ev.Content, "[REDACTED]") {
			t.Error("redaction marker missing in unbounded mode")
		}
	})
}

// TestReasoningLoggedOnlyWhenEnabled pins log_reasoning's writer half: the
// reasoning payload is dropped unless the writer was opened with LogReasoning,
// and once opted in it travels the same redact+clip path as every other
// free-text field. The size (reasoning_bytes) is recorded either way.
func TestReasoningLoggedOnlyWhenEnabled(t *testing.T) {
	const payload = `{"x":"secret-value"}`

	t.Run("default off drops the content", func(t *testing.T) {
		w, err := New(t.TempDir(), Options{Clock: fixedClock()})
		if err != nil {
			t.Fatal(err)
		}
		w.LLMResponse(LLMResponse{Turn: 1, Content: "hi", FinishReason: "stop", Reasoning: payload, ReasoningBytes: len(payload)})

		data, err := os.ReadFile(w.Path())
		if err != nil {
			t.Fatal(err)
		}
		line := strings.TrimSpace(string(data))
		if strings.Contains(line, `"reasoning"`) {
			t.Errorf("reasoning key written without the opt-in:\n%s", line)
		}
		if strings.Contains(line, "secret-value") {
			t.Errorf("reasoning content leaked without the opt-in:\n%s", line)
		}
		var ev Event
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			t.Fatal(err)
		}
		if ev.ReasoningBytes != len(payload) {
			t.Errorf("reasoning_bytes = %d, want %d", ev.ReasoningBytes, len(payload))
		}
	})

	t.Run("opt-in logs the payload", func(t *testing.T) {
		w, err := New(t.TempDir(), Options{Clock: fixedClock(), LogReasoning: true})
		if err != nil {
			t.Fatal(err)
		}
		w.LLMResponse(LLMResponse{Turn: 1, Content: "hi", FinishReason: "stop", Reasoning: payload, ReasoningBytes: len(payload)})

		ev := firstLLMEvent(t, w.Path())
		if ev.Reasoning != payload {
			t.Errorf("reasoning = %q, want %q", ev.Reasoning, payload)
		}
	})

	t.Run("opt-in still redacts", func(t *testing.T) {
		w, err := New(t.TempDir(), Options{Clock: fixedClock(), LogReasoning: true, Secrets: []string{"secret-value"}})
		if err != nil {
			t.Fatal(err)
		}
		w.LLMResponse(LLMResponse{Turn: 1, Content: "hi", FinishReason: "stop", Reasoning: payload, ReasoningBytes: len(payload)})

		ev := firstLLMEvent(t, w.Path())
		if strings.Contains(ev.Reasoning, "secret-value") {
			t.Errorf("secret survived in the logged reasoning: %q", ev.Reasoning)
		}
		if !strings.Contains(ev.Reasoning, "[REDACTED]") {
			t.Errorf("reasoning was not redacted: %q", ev.Reasoning)
		}
	})

	t.Run("opt-in still clips", func(t *testing.T) {
		w, err := New(t.TempDir(), Options{Clock: fixedClock(), LogReasoning: true, MaxLoggedField: 16})
		if err != nil {
			t.Fatal(err)
		}
		long := strings.Repeat("r", 40)
		w.LLMResponse(LLMResponse{Turn: 1, Content: "hi", FinishReason: "stop", Reasoning: long, ReasoningBytes: len(long)})

		ev := firstLLMEvent(t, w.Path())
		if want := strings.Repeat("r", 16) + "...[clipped]"; ev.Reasoning != want {
			t.Errorf("reasoning = %q, want %q", ev.Reasoning, want)
		}
	})
}

// firstLLMEvent decodes the first line of a session file as an Event.
func firstLLMEvent(t *testing.T, path string) Event {
	t.Helper()
	data, err := os.ReadFile(path) //nolint:gosec // G304: path comes from t.TempDir
	if err != nil {
		t.Fatal(err)
	}
	var ev Event
	if err := json.Unmarshal([]byte(strings.SplitN(strings.TrimSpace(string(data)), "\n", 2)[0]), &ev); err != nil {
		t.Fatal(err)
	}
	return ev
}
