package session

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"strings"
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
	w.LLMResponse(1, "let me read the log", []string{"call_1"}, 100, 20, "tool_calls")
	w.ToolCall("call_1", "fs_read", `{"path":"app.log"}`)
	// The secret arrives via tool output and must be redacted by value.
	w.ToolResult(ToolResult{CallID: "call_1", Tool: "fs_read", Result: "line with sk-supersecret token", Outcome: OutcomeOK})
	w.LLMResponse(2, "all clear", nil, 150, 30, "stop")
	w.RunEnd("success", 0, 2, 1, 300, 1500*time.Millisecond)

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

	w.RunEnd("success", 0, 1, 0, 10, time.Second)

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

func TestRedaction(t *testing.T) {
	dir := t.TempDir()
	w, err := New(dir, Options{Clock: fixedClock(), Secrets: []string{"sk-verysecret", "tiny"}})
	if err != nil {
		t.Fatal(err)
	}
	w.ToolResult(ToolResult{CallID: "call_1", Tool: "t", Result: "key=sk-verysecret and tiny goes too", Outcome: OutcomeOK})
	// The model may echo a secret in its answer text; that path must be
	// redacted too now that content is logged.
	w.LLMResponse(2, "the key is sk-verysecret", nil, 1, 1, "stop")
	w.RunEnd("success", 0, 1, 1, 10, time.Second)

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
	w.RunEnd("success", 0, 1, 1, 10, time.Second)

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
	w.LLMResponse(1, "reading the log now", []string{"call_9"}, 10, 5, "tool_calls")
	w.ToolCall("call_9", "fs_read", `{"path":"x"}`)
	w.ToolResult(ToolResult{CallID: "call_9", Tool: "fs_read", Result: "data", Outcome: OutcomeOK})
	w.RunEnd("success", 0, 1, 1, 15, time.Second)

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
	w.RunEnd("success", 0, 1, 1, 1, time.Second)

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
	w.LLMResponse(1, "c", []string{"id"}, 1, 1, "stop")
	w.ToolCall("id", "t", "{}")
	w.ToolResult(ToolResult{CallID: "id", Tool: "t", Result: "r", Outcome: OutcomeOK})
	w.RunEnd("success", 0, 1, 1, 1, time.Second)
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
		want      string
	}{
		{"plural", true, 8, 3, 41234, "✓ 8 turns, 3 tool calls, 41.2k tokens, 34.0s"},
		{"failure mark", false, 8, 3, 900, "✗ 8 turns, 3 tool calls, 900 tokens, 34.0s"},
		{"millions of tokens", true, 8, 3, 2_500_000, "✓ 8 turns, 3 tool calls, 2.5M tokens, 34.0s"},
		// The one-turn run is the common case for a cron agent that answers
		// without calling a tool, so "1 turns, 0 tool calls" was the line
		// operators read most often.
		{"single turn", true, 1, 1, 12, "✓ 1 turn, 1 tool call, 12 tokens, 34.0s"},
		{"zero counts stay plural", true, 0, 0, 0, "✓ 0 turns, 0 tool calls, 0 tokens, 34.0s"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Summary(tt.ok, tt.turns, tt.toolCalls, tt.tokens, 34*time.Second)
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
			w.RunEnd("success", 0, 1, 1, 1, time.Second)

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
	w.RunEnd("success", 0, 1, 1, 1, time.Second)

	ev := firstEvent(t, w.Path())
	if got := ev["result_bytes"]; got != float64(size) {
		t.Errorf("result_bytes = %v, want %d (the FULL size, not the clipped one)", got, size)
	}
	if logged, _ := ev["result"].(string); len(logged) > maxLoggedField+64 {
		t.Errorf("the result itself must still be clipped, got %d bytes", len(logged))
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
	w.LLMResponse(1, "text", []string{"c1"}, 1, 1, "tool_calls")
	w.ToolCall("c1", "shell", "{}")
	w.ToolResult(ToolResult{CallID: "c1", Tool: "shell", Result: "out", Outcome: OutcomeOK})
	w.RunEnd("success", 0, 1, 1, 2, time.Second)

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
