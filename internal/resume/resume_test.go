package resume_test

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/lasthumanintheloop/amele/internal/llm"
	"github.com/lasthumanintheloop/amele/internal/resume"
	"github.com/lasthumanintheloop/amele/internal/session"
)

// user, assistant and toolMsg keep the expectation tables readable: the
// fixtures assert on whole []llm.Message slices, and a literal struct per line
// buries the one field that differs.
func user(content string) llm.Message {
	return llm.Message{Role: llm.RoleUser, Content: content}
}

func assistant(content string, calls ...llm.ToolCall) llm.Message {
	return llm.Message{Role: llm.RoleAssistant, Content: content, ToolCalls: calls}
}

func toolMsg(id, content string) llm.Message {
	return llm.Message{Role: llm.RoleTool, ToolCallID: id, Content: content}
}

func TestReadFixtures(t *testing.T) {
	tests := []struct {
		name      string
		fixture   string
		opts      resume.Options
		task      string
		model     string
		provider  string
		lastTurn  int
		completed bool
		pending   []string
		carriers  bool
		messages  []llm.Message
	}{
		{
			name:     "complete tool turn resumes after the results",
			fixture:  "complete-tool-turn.jsonl",
			opts:     resume.Options{Provider: "openai"},
			task:     "scan yesterday's logs and report anything unusual",
			model:    "gpt-4o",
			provider: "openai",
			lastTurn: 1,
			messages: []llm.Message{
				user("scan yesterday's logs and report anything unusual"),
				assistant("I will read both log files.",
					llm.ToolCall{ID: "call_1", Name: "fs_read", Arguments: `{"path":"app.log"}`},
					llm.ToolCall{ID: "call_2", Name: "fs_read", Arguments: `{"path":"nginx.log"}`}),
				toolMsg("call_1", "ERROR failed to bind port 8080"),
				toolMsg("call_2", "200 GET /health"),
			},
		},
		{
			name:     "interrupted dispatch gets a synthetic result and drops the undispatched call",
			fixture:  "interrupted-mid-dispatch.jsonl",
			opts:     resume.Options{Provider: "openai"},
			task:     "check the disk and the queue",
			model:    "gpt-4o",
			provider: "openai",
			lastTurn: 1,
			pending:  []string{"call_2"},
			messages: []llm.Message{
				user("check the disk and the queue"),
				assistant("Checking disk, queue and uptime.",
					llm.ToolCall{ID: "call_1", Name: "disk", Arguments: `{}`},
					llm.ToolCall{ID: "call_2", Name: "queue", Arguments: `{"name":"mail"}`}),
				toolMsg("call_1", "/dev/sda1 61% used"),
				toolMsg("call_2", resume.PendingResultMessage),
			},
		},
		{
			name:      "final answer is a completed run",
			fixture:   "final-answer.jsonl",
			opts:      resume.Options{Provider: "openai"},
			task:      "summarize the release notes",
			model:     "gpt-4o",
			provider:  "openai",
			lastTurn:  2,
			completed: true,
			messages: []llm.Message{
				user("summarize the release notes"),
				assistant("", llm.ToolCall{ID: "call_1", Name: "fs_read", Arguments: `{"path":"CHANGELOG.md"}`}),
				toolMsg("call_1", "## v0.2.1\n- three fixes"),
				assistant("Everything looks fine: three fixes, no regressions."),
			},
		},
		{
			name:     "a run that died on the provider ends at the last complete position",
			fixture:  "provider-error.jsonl",
			opts:     resume.Options{Provider: "openai"},
			task:     "rotate the logs and confirm the free space",
			model:    "gpt-4o",
			provider: "openai",
			lastTurn: 1,
			messages: []llm.Message{
				user("rotate the logs and confirm the free space"),
				assistant("Rotating first.", llm.ToolCall{ID: "call_1", Name: "rotate", Arguments: `{"path":"/var/log/app.log"}`}),
				toolMsg("call_1", "rotated 1 file"),
			},
		},
		{
			name:      "a run that fell back never restores carriers",
			fixture:   "fallback.jsonl",
			opts:      resume.Options{Provider: "openai"},
			task:      "tail the queue and report the backlog",
			model:     "gpt-4o",
			provider:  "openai",
			lastTurn:  3,
			completed: true,
			messages: []llm.Message{
				user("tail the queue and report the backlog"),
				assistant("Reading the queue.", llm.ToolCall{ID: "call_1", Name: "queue", Arguments: `{"name":"mail"}`}),
				toolMsg("call_1", "backlog 12"),
				assistant("The mail queue has a backlog of 12."),
			},
		},
		{
			name:      "carriers are restored for the same provider",
			fixture:   "carriers-anthropic.jsonl",
			opts:      resume.Options{Provider: "anthropic"},
			task:      "check app.log for anything unusual",
			model:     "claude-sonnet-4",
			provider:  "anthropic",
			lastTurn:  2,
			completed: true,
			carriers:  true,
			messages: []llm.Message{
				user("check app.log for anything unusual"),
				{
					Role:      llm.RoleAssistant,
					ToolCalls: []llm.ToolCall{{ID: "call_1", Name: "fs_read", Arguments: `{"path":"app.log"}`}},
					Reasoning: json.RawMessage(`[{"type":"thinking","thinking":"The log lives at app.log; read it first.","signature":"Ej8BCkYIBBgCIkA="}]`),
				},
				toolMsg("call_1", "WARN retrying in 5s"),
				{
					Role:      llm.RoleAssistant,
					Content:   "One retry warning, nothing unusual.",
					Reasoning: json.RawMessage(`[{"type":"thinking","thinking":"A retry warning is routine.","signature":"Ej8BCkYIBBgCIkB="}]`),
				},
			},
		},
		{
			name:      "a different provider gets no carriers",
			fixture:   "carriers-anthropic.jsonl",
			opts:      resume.Options{Provider: "openai"},
			task:      "check app.log for anything unusual",
			model:     "claude-sonnet-4",
			provider:  "anthropic",
			lastTurn:  2,
			completed: true,
			messages: []llm.Message{
				user("check app.log for anything unusual"),
				assistant("", llm.ToolCall{ID: "call_1", Name: "fs_read", Arguments: `{"path":"app.log"}`}),
				toolMsg("call_1", "WARN retrying in 5s"),
				assistant("One retry warning, nothing unusual."),
			},
		},
		{
			name:      "an unnamed provider gets no carriers",
			fixture:   "carriers-anthropic.jsonl",
			opts:      resume.Options{},
			task:      "check app.log for anything unusual",
			model:     "claude-sonnet-4",
			provider:  "anthropic",
			lastTurn:  2,
			completed: true,
			messages: []llm.Message{
				user("check app.log for anything unusual"),
				assistant("", llm.ToolCall{ID: "call_1", Name: "fs_read", Arguments: `{"path":"app.log"}`}),
				toolMsg("call_1", "WARN retrying in 5s"),
				assistant("One retry warning, nothing unusual."),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := resume.Read(filepath.Join("testdata", tt.fixture), tt.opts)
			if err != nil {
				t.Fatalf("Read(%s) = %v, want no error", tt.fixture, err)
			}
			if got.Task != tt.task {
				t.Errorf("Task = %q, want %q", got.Task, tt.task)
			}
			if got.Model != tt.model {
				t.Errorf("Model = %q, want %q", got.Model, tt.model)
			}
			if got.Provider != tt.provider {
				t.Errorf("Provider = %q, want %q", got.Provider, tt.provider)
			}
			if got.LastTurn != tt.lastTurn {
				t.Errorf("LastTurn = %d, want %d", got.LastTurn, tt.lastTurn)
			}
			if got.Completed != tt.completed {
				t.Errorf("Completed = %v, want %v", got.Completed, tt.completed)
			}
			if got.Carriers != tt.carriers {
				t.Errorf("Carriers = %v, want %v", got.Carriers, tt.carriers)
			}
			if !reflect.DeepEqual(got.Pending, tt.pending) {
				t.Errorf("Pending = %#v, want %#v", got.Pending, tt.pending)
			}
			assertMessages(t, got.Messages, tt.messages)
		})
	}
}

// assertMessages compares message by message so a failure names the entry that
// differs instead of dumping two whole conversations.
func assertMessages(t *testing.T, got, want []llm.Message) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d messages, want %d:\ngot  %+v\nwant %+v", len(got), len(want), got, want)
	}
	for i := range want {
		if !reflect.DeepEqual(got[i], want[i]) {
			t.Errorf("message %d =\n  %+v\nwant\n  %+v", i, got[i], want[i])
		}
	}
}

// The reasoning payload is echoed back to a provider that signs it, so the
// restored bytes must be the logged bytes - not a re-encoding of them.
func TestCarrierBytesAreExact(t *testing.T) {
	got, err := resume.Read(filepath.Join("testdata", "carriers-anthropic.jsonl"), resume.Options{Provider: "anthropic"})
	if err != nil {
		t.Fatalf("Read = %v, want no error", err)
	}
	want := `[{"type":"thinking","thinking":"The log lives at app.log; read it first.","signature":"Ej8BCkYIBBgCIkA="}]`
	if string(got.Messages[1].Reasoning) != want {
		t.Errorf("Reasoning = %q, want %q", got.Messages[1].Reasoning, want)
	}
	if got.Messages[1].ReasoningField != "" {
		t.Errorf("ReasoningField = %q, want empty (the client's default key applies)", got.Messages[1].ReasoningField)
	}
}

const runStart = `{"v":1,"type":"run_start","ts":"2026-09-05T03:00:00Z","model":"gpt-4o","provider":"openai","task":"scan the logs"}`

func TestReadRejects(t *testing.T) {
	tests := []struct {
		name     string
		log      string
		opts     resume.Options
		want     error
		contains []string
	}{
		{
			name:     "a chat log is not resumable",
			log:      readFixture(t, "chat.jsonl"),
			want:     resume.ErrNotResumable,
			contains: []string{"chat"},
		},
		{
			name:     "a clipped tool result names the config key that fixes it",
			log:      readFixture(t, "clipped.jsonl"),
			want:     resume.ErrClipped,
			contains: []string{"result", "turn 1", "limits.max_logged_field: 0"},
		},
		{
			name:     "a clipped task",
			log:      `{"v":1,"type":"run_start","ts":"2026-09-05T03:00:00Z","model":"gpt-4o","provider":"openai","task":"scan the l...[clipped]"}`,
			want:     resume.ErrClipped,
			contains: []string{"task"},
		},
		{
			name: "clipped assistant content",
			log: runStart + "\n" +
				`{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":1,"content":"I will...[clipped]","finish_reason":"stop"}`,
			want:     resume.ErrClipped,
			contains: []string{"content", "turn 1"},
		},
		{
			name: "clipped tool arguments",
			log: runStart + "\n" +
				`{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":2,"tool_call_ids":["c1"],"finish_reason":"tool_calls"}` + "\n" +
				`{"v":1,"type":"tool_call","ts":"2026-09-05T03:00:02Z","tool_call_id":"c1","tool":"fs_read","args":"{\"path\":\"app...[clipped]"}`,
			want:     resume.ErrClipped,
			contains: []string{"args", "turn 2"},
		},
		{
			name: "clipped reasoning that would be restored",
			log: runStart + "\n" +
				`{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":1,"content":"done","reasoning_bytes":9000,"reasoning":"[{\"type\":\"thinking\"...[clipped]","finish_reason":"stop"}`,
			opts:     resume.Options{Provider: "openai"},
			want:     resume.ErrClipped,
			contains: []string{"reasoning", "turn 1"},
		},
		{
			name: "a tool_result for an id nobody requested",
			log: runStart + "\n" +
				`{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":1,"tool_call_ids":["c1"],"finish_reason":"tool_calls"}` + "\n" +
				`{"v":1,"type":"tool_call","ts":"2026-09-05T03:00:02Z","tool_call_id":"c1","tool":"fs_read","args":"{}"}` + "\n" +
				`{"v":1,"type":"tool_result","ts":"2026-09-05T03:00:03Z","tool_call_id":"c9","tool":"fs_read","result":"x","outcome":"ok"}`,
			want:     resume.ErrMalformed,
			contains: []string{"c9"},
		},
		{
			name: "a tool_call for an id the turn did not request",
			log: runStart + "\n" +
				`{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":1,"tool_call_ids":["c1"],"finish_reason":"tool_calls"}` + "\n" +
				`{"v":1,"type":"tool_call","ts":"2026-09-05T03:00:02Z","tool_call_id":"c2","tool":"fs_read","args":"{}"}`,
			want:     resume.ErrMalformed,
			contains: []string{"c2"},
		},
		{
			name: "a tool_result that arrives before its tool_call",
			log: runStart + "\n" +
				`{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":1,"tool_call_ids":["c1"],"finish_reason":"tool_calls"}` + "\n" +
				`{"v":1,"type":"tool_result","ts":"2026-09-05T03:00:02Z","tool_call_id":"c1","tool":"fs_read","result":"x","outcome":"ok"}`,
			want:     resume.ErrMalformed,
			contains: []string{"c1"},
		},
		{
			name: "a turn that requests an empty tool call id",
			log: runStart + "\n" +
				`{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":1,"tool_call_ids":[""],"finish_reason":"tool_calls"}`,
			want:     resume.ErrMalformed,
			contains: []string{"empty id"},
		},
		{
			name: "a second tool_call for one id",
			log: runStart + "\n" +
				`{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":1,"tool_call_ids":["c1"],"finish_reason":"tool_calls"}` + "\n" +
				`{"v":1,"type":"tool_call","ts":"2026-09-05T03:00:02Z","tool_call_id":"c1","tool":"fs_read","args":"{}"}` + "\n" +
				`{"v":1,"type":"tool_call","ts":"2026-09-05T03:00:03Z","tool_call_id":"c1","tool":"fs_read","args":"{}"}`,
			want:     resume.ErrMalformed,
			contains: []string{"second tool_call", "c1"},
		},
		{
			name: "a second tool_result for one call",
			log: runStart + "\n" +
				`{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":1,"tool_call_ids":["c1"],"finish_reason":"tool_calls"}` + "\n" +
				`{"v":1,"type":"tool_call","ts":"2026-09-05T03:00:02Z","tool_call_id":"c1","tool":"fs_read","args":"{}"}` + "\n" +
				`{"v":1,"type":"tool_result","ts":"2026-09-05T03:00:03Z","tool_call_id":"c1","tool":"fs_read","result":"x","outcome":"ok"}` + "\n" +
				`{"v":1,"type":"tool_result","ts":"2026-09-05T03:00:04Z","tool_call_id":"c1","tool":"fs_read","result":"x","outcome":"ok"}`,
			want:     resume.ErrMalformed,
			contains: []string{"c1"},
		},
		{
			name: "a turn that requests one id twice",
			log: runStart + "\n" +
				`{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":1,"tool_call_ids":["c1","c1"],"finish_reason":"tool_calls"}`,
			want:     resume.ErrMalformed,
			contains: []string{"c1"},
		},
		{
			name:     "a line that is not JSON",
			log:      runStart + "\nnot json at all\n",
			want:     resume.ErrMalformed,
			contains: []string{"line 2"},
		},
		{
			name:     "a future schema version",
			log:      `{"v":2,"type":"run_start","ts":"2026-09-05T03:00:00Z","model":"gpt-4o","task":"scan the logs"}`,
			want:     resume.ErrNotResumable,
			contains: []string{"schema version 2"},
		},
		{
			name:     "a log that does not start with run_start",
			log:      `{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":1,"content":"hi","finish_reason":"stop"}`,
			want:     resume.ErrNotResumable,
			contains: []string{"run_start"},
		},
		{
			name:     "an empty log",
			log:      "",
			want:     resume.ErrNotResumable,
			contains: []string{"empty"},
		},
		{
			name:     "a run_start without a task",
			log:      `{"v":1,"type":"run_start","ts":"2026-09-05T03:00:00Z","model":"gpt-4o","provider":"openai"}`,
			want:     resume.ErrNotResumable,
			contains: []string{"task"},
		},
		{
			name:     "two runs concatenated into one file",
			log:      runStart + "\n" + runStart,
			want:     resume.ErrMalformed,
			contains: []string{"run_start"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := resume.ReadFrom(strings.NewReader(tt.log), tt.opts)
			if !errors.Is(err, tt.want) {
				t.Fatalf("ReadFrom error = %v, want %v", err, tt.want)
			}
			for _, want := range tt.contains {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not mention %q", err, want)
				}
			}
		})
	}
}

// Clipped reasoning is only a gate when the reasoning would actually be used:
// a run continuing on another provider never echoes it, so the log is fine.
func TestClippedReasoningIsNotAGateWhenUnused(t *testing.T) {
	log := runStart + "\n" +
		`{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":1,"content":"done","reasoning_bytes":9000,"reasoning":"[{\"type\":\"thinking\"...[clipped]","finish_reason":"stop"}`
	got, err := resume.ReadFrom(strings.NewReader(log), resume.Options{Provider: "anthropic"})
	if err != nil {
		t.Fatalf("ReadFrom = %v, want no error", err)
	}
	if got.Carriers {
		t.Error("Carriers = true, want false for a provider that does not match the log")
	}
	if got.Messages[1].Reasoning != nil {
		t.Errorf("Reasoning = %q, want nil", got.Messages[1].Reasoning)
	}
}

// A log whose run died before the first turn still yields the task, so the
// resumed run simply starts over with the same instruction.
func TestReadRunStartOnly(t *testing.T) {
	got, err := resume.ReadFrom(strings.NewReader(runStart), resume.Options{Provider: "openai"})
	if err != nil {
		t.Fatalf("ReadFrom = %v, want no error", err)
	}
	if got.LastTurn != 0 || got.Completed {
		t.Errorf("LastTurn/Completed = %d/%v, want 0/false", got.LastTurn, got.Completed)
	}
	assertMessages(t, got.Messages, []llm.Message{user("scan the logs")})
}

// A pre-v1.8 log carries no provider identity, so there is nothing to match
// the current primary against and carriers stay off.
func TestPreV18LogRestoresNoCarriers(t *testing.T) {
	log := `{"v":1,"type":"run_start","ts":"2026-09-05T03:00:00Z","model":"gpt-4o","task":"scan the logs"}` + "\n" +
		`{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":1,"content":"done","reasoning_bytes":20,"reasoning":"\"thinking\"","finish_reason":"stop"}`
	got, err := resume.ReadFrom(strings.NewReader(log), resume.Options{Provider: ""})
	if err != nil {
		t.Fatalf("ReadFrom = %v, want no error", err)
	}
	if got.Carriers {
		t.Error("Carriers = true, want false for a log without a provider identity")
	}
	if got.Messages[1].Reasoning != nil {
		t.Errorf("Reasoning = %q, want nil", got.Messages[1].Reasoning)
	}
}

// Redaction is not a fidelity problem: the model saw exactly this text.
func TestRedactedTextPassesThrough(t *testing.T) {
	log := runStart + "\n" +
		`{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":1,"tool_call_ids":["c1"],"finish_reason":"tool_calls"}` + "\n" +
		`{"v":1,"type":"tool_call","ts":"2026-09-05T03:00:02Z","tool_call_id":"c1","tool":"fs_read","args":"{}"}` + "\n" +
		`{"v":1,"type":"tool_result","ts":"2026-09-05T03:00:03Z","tool_call_id":"c1","tool":"fs_read","result":"token [REDACTED] seen","outcome":"ok"}`
	got, err := resume.ReadFrom(strings.NewReader(log), resume.Options{})
	if err != nil {
		t.Fatalf("ReadFrom = %v, want no error", err)
	}
	if got.Messages[2].Content != "token [REDACTED] seen" {
		t.Errorf("tool content = %q, want the redacted text unchanged", got.Messages[2].Content)
	}
}

// The fidelity gate matches the marker the writer actually appends.
func TestClipMarkerIsSessionsMarker(t *testing.T) {
	if session.ClipMarker != "...[clipped]" {
		t.Errorf("session.ClipMarker = %q, want %q", session.ClipMarker, "...[clipped]")
	}
}

// Read names the file it refused, so cmd can print the error as-is.
func TestReadNamesThePath(t *testing.T) {
	path := filepath.Join("testdata", "chat.jsonl")
	_, err := resume.Read(path, resume.Options{})
	if !errors.Is(err, resume.ErrNotResumable) {
		t.Fatalf("Read(%s) = %v, want ErrNotResumable", path, err)
	}
	if !strings.Contains(err.Error(), path) {
		t.Errorf("error %q does not name the path %q", err, path)
	}
}

func TestReadMissingFile(t *testing.T) {
	_, err := resume.Read(filepath.Join(t.TempDir(), "nope.jsonl"), resume.Options{})
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("Read of a missing file = %v, want a not-exist error", err)
	}
}

func readFixture(t *testing.T, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name)) //nolint:gosec // G304: fixed testdata path.
	if err != nil {
		t.Fatalf("reading fixture %s: %v", name, err)
	}
	return string(b)
}
