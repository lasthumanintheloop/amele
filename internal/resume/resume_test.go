package resume_test

import (
	"context"
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
			// A v1.10 log of a schema retry: the feedback is a user turn of
			// the rebuilt conversation, between the rejected answer and the
			// retry. The run finished, so it needs an instruction to go on.
			name:      "a logged schema retry is rebuilt with its feedback turn",
			fixture:   "schema-retry-logged.jsonl",
			opts:      resume.Options{Provider: "openai"},
			task:      "score the incident report from 1 to 5",
			model:     "gpt-4o",
			provider:  "openai",
			lastTurn:  2,
			completed: true,
			messages: []llm.Message{
				user("score the incident report from 1 to 5"),
				assistant(`{"score": "four"}`),
				user("your answer did not match the schema: score must be an integer"),
				assistant(`{"score": 4}`),
			},
		},
		{
			// The run died after writing the feedback and before the retry:
			// the history ends on the feedback, which the model has not
			// answered - so the run is NOT complete and resumes on its own.
			name:     "a run interrupted after the feedback continues from it",
			fixture:  "schema-retry-interrupted.jsonl",
			opts:     resume.Options{Provider: "openai"},
			task:     "score the incident report from 1 to 5",
			model:    "gpt-4o",
			provider: "openai",
			lastTurn: 1,
			messages: []llm.Message{
				user("score the incident report from 1 to 5"),
				assistant(`{"score": "four"}`),
				user("your answer did not match the schema: score must be an integer"),
			},
		},
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
			// The crash between an llm_response and the tool_call lines it
			// announced: the turn requested a call, nothing was dispatched, so
			// the message has no text, no call and no carrier left. Sending it
			// is what the anthropic wire refuses as "content": null, so the
			// whole turn is dropped and only the task survives.
			name:     "a turn that dispatched nothing is dropped whole",
			fixture:  "undispatched-only.jsonl",
			opts:     resume.Options{Provider: "openai", Model: "gpt-4o"},
			task:     "restart the worker and confirm it came back",
			model:    "gpt-4o",
			provider: "openai",
			lastTurn: 1,
			messages: []llm.Message{
				user("restart the worker and confirm it came back"),
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
			name:      "carriers are restored for the same provider and model",
			fixture:   "carriers-anthropic.jsonl",
			opts:      resume.Options{Provider: "anthropic", Model: "claude-sonnet-4"},
			task:      "check app.log and deploy.log for anything unusual",
			model:     "claude-sonnet-4",
			provider:  "anthropic",
			lastTurn:  3,
			completed: true,
			carriers:  true,
			messages: []llm.Message{
				user("check app.log and deploy.log for anything unusual"),
				{
					Role:      llm.RoleAssistant,
					ToolCalls: []llm.ToolCall{{ID: "call_1", Name: "fs_read", Arguments: `{"path":"app.log"}`}},
					Reasoning: json.RawMessage(carrierTurn1),
				},
				toolMsg("call_1", "WARN retrying in 5s"),
				// Turn 2's payload was rewritten by the redactor, so it is not
				// echoed back - the turn keeps its text and tool call only.
				assistant("", llm.ToolCall{ID: "call_2", Name: "fs_read", Arguments: `{"path":"deploy.log"}`}),
				toolMsg("call_2", "deployed 2026-09-04 with key [REDACTED]"),
				{
					Role:      llm.RoleAssistant,
					Content:   "One retry warning, nothing unusual.",
					Reasoning: json.RawMessage(carrierTurn3),
				},
			},
		},
		{
			// Same backend, another model: a thinking block is signed by the
			// model that minted it, so `--resume --model other` replays the
			// conversation carrier-less rather than handing one model's
			// signed reasoning to another.
			name:      "a different model gets no carriers",
			fixture:   "carriers-anthropic.jsonl",
			opts:      resume.Options{Provider: "anthropic", Model: "claude-opus-4"},
			task:      "check app.log and deploy.log for anything unusual",
			model:     "claude-sonnet-4",
			provider:  "anthropic",
			lastTurn:  3,
			completed: true,
			messages: []llm.Message{
				user("check app.log and deploy.log for anything unusual"),
				assistant("", llm.ToolCall{ID: "call_1", Name: "fs_read", Arguments: `{"path":"app.log"}`}),
				toolMsg("call_1", "WARN retrying in 5s"),
				assistant("", llm.ToolCall{ID: "call_2", Name: "fs_read", Arguments: `{"path":"deploy.log"}`}),
				toolMsg("call_2", "deployed 2026-09-04 with key [REDACTED]"),
				assistant("One retry warning, nothing unusual."),
			},
		},
		{
			name:      "a different provider gets no carriers",
			fixture:   "carriers-anthropic.jsonl",
			opts:      resume.Options{Provider: "openai", Model: "claude-sonnet-4"},
			task:      "check app.log and deploy.log for anything unusual",
			model:     "claude-sonnet-4",
			provider:  "anthropic",
			lastTurn:  3,
			completed: true,
			messages: []llm.Message{
				user("check app.log and deploy.log for anything unusual"),
				assistant("", llm.ToolCall{ID: "call_1", Name: "fs_read", Arguments: `{"path":"app.log"}`}),
				toolMsg("call_1", "WARN retrying in 5s"),
				assistant("", llm.ToolCall{ID: "call_2", Name: "fs_read", Arguments: `{"path":"deploy.log"}`}),
				toolMsg("call_2", "deployed 2026-09-04 with key [REDACTED]"),
				assistant("One retry warning, nothing unusual."),
			},
		},
		{
			name:      "an unnamed provider gets no carriers",
			fixture:   "carriers-anthropic.jsonl",
			opts:      resume.Options{},
			task:      "check app.log and deploy.log for anything unusual",
			model:     "claude-sonnet-4",
			provider:  "anthropic",
			lastTurn:  3,
			completed: true,
			messages: []llm.Message{
				user("check app.log and deploy.log for anything unusual"),
				assistant("", llm.ToolCall{ID: "call_1", Name: "fs_read", Arguments: `{"path":"app.log"}`}),
				toolMsg("call_1", "WARN retrying in 5s"),
				assistant("", llm.ToolCall{ID: "call_2", Name: "fs_read", Arguments: `{"path":"deploy.log"}`}),
				toolMsg("call_2", "deployed 2026-09-04 with key [REDACTED]"),
				assistant("One retry warning, nothing unusual."),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := resume.Read(context.Background(), filepath.Join("testdata", tt.fixture), tt.opts)
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

// The carriers as the anthropic client stores them: the ENTIRE raw content
// array, thinking blocks interleaved with the tool_use/text blocks they were
// produced with, which is the unit Anthropic requires back byte-exact.
const (
	carrierTurn1 = `[{"type":"thinking","thinking":"The application log lives at app.log; read it first.","signature":"Ej8BCkYIBBgCIkA="},{"type":"tool_use","id":"call_1","name":"fs_read","input":{"path":"app.log"}}]`
	carrierTurn3 = `[{"type":"thinking","thinking":"A retry warning is routine.","signature":"Ej8BCkYIBBgCIkC="},{"type":"text","text":"One retry warning, nothing unusual."}]`
)

// The reasoning payload is echoed back to a provider that signs it, so the
// restored bytes must be the logged bytes - not a re-encoding of them.
func TestCarrierBytesAreExact(t *testing.T) {
	got, err := resume.Read(context.Background(), filepath.Join("testdata", "carriers-anthropic.jsonl"),
		resume.Options{Provider: "anthropic", Model: "claude-sonnet-4"})
	if err != nil {
		t.Fatalf("Read = %v, want no error", err)
	}
	if string(got.Messages[1].Reasoning) != carrierTurn1 {
		t.Errorf("Reasoning = %q, want %q", got.Messages[1].Reasoning, carrierTurn1)
	}
	if got.Messages[1].ReasoningField != "" {
		t.Errorf("ReasoningField = %q, want empty (the client's default key applies)", got.Messages[1].ReasoningField)
	}
	// The redacted payload is dropped, not repaired: its bytes no longer match
	// the signature the provider gave them.
	if got.Messages[3].Reasoning != nil {
		t.Errorf("redacted carrier = %q, want nil", got.Messages[3].Reasoning)
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
			opts:     resume.Options{Provider: "openai", Model: "gpt-4o"},
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
			// Not the last line: something complete follows it, so the file is
			// damaged rather than cut short.
			name: "a torn line in the middle of the file",
			log: runStart + "\n" + `{"v":1,"type":"llm_res` + "\n" +
				`{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":1,"content":"done","finish_reason":"stop"}` + "\n",
			want:     resume.ErrMalformed,
			contains: []string{"line 2"},
		},
		{
			name:     "a pre-v1.10 log that skips the validator's feedback turn",
			log:      readFixture(t, "schema-retry.jsonl"),
			want:     resume.ErrNotResumable,
			contains: []string{"skips a user turn", "output.schema", "v1.10"},
		},
		{
			name: "validator_feedback after a turn that requested tools",
			log: `{"v":1,"type":"run_start","ts":"2026-09-13T03:00:00Z","model":"gpt-4o","task":"scan the logs"}
{"v":1,"type":"llm_response","ts":"2026-09-13T03:00:01Z","turn":1,"tool_call_ids":["c1"],"finish_reason":"tool_calls"}
{"v":1,"type":"tool_call","ts":"2026-09-13T03:00:01Z","tool_call_id":"c1","tool":"fs_read","args":"{}"}
{"v":1,"type":"validator_feedback","ts":"2026-09-13T03:00:02Z","turn":1,"content":"fix it"}`,
			want:     resume.ErrMalformed,
			contains: []string{"validator_feedback", "not a final answer"},
		},
		{
			name: "validator_feedback before any turn",
			log: `{"v":1,"type":"run_start","ts":"2026-09-13T03:00:00Z","model":"gpt-4o","task":"scan the logs"}
{"v":1,"type":"validator_feedback","ts":"2026-09-13T03:00:02Z","turn":1,"content":"fix it"}`,
			want:     resume.ErrMalformed,
			contains: []string{"validator_feedback", "not a final answer"},
		},
		{
			name: "validator_feedback on the wrong turn number",
			log: `{"v":1,"type":"run_start","ts":"2026-09-13T03:00:00Z","model":"gpt-4o","task":"scan the logs"}
{"v":1,"type":"llm_response","ts":"2026-09-13T03:00:01Z","turn":1,"content":"{}","finish_reason":"stop"}
{"v":1,"type":"validator_feedback","ts":"2026-09-13T03:00:02Z","turn":2,"content":"fix it"}`,
			want:     resume.ErrMalformed,
			contains: []string{"turn 2", "turn 1 is open"},
		},
		{
			name: "two validator_feedback events for one turn",
			log: `{"v":1,"type":"run_start","ts":"2026-09-13T03:00:00Z","model":"gpt-4o","task":"scan the logs"}
{"v":1,"type":"llm_response","ts":"2026-09-13T03:00:01Z","turn":1,"content":"{}","finish_reason":"stop"}
{"v":1,"type":"validator_feedback","ts":"2026-09-13T03:00:02Z","turn":1,"content":"fix it"}
{"v":1,"type":"validator_feedback","ts":"2026-09-13T03:00:02Z","turn":1,"content":"fix it again"}`,
			want:     resume.ErrMalformed,
			contains: []string{"second validator_feedback"},
		},
		{
			name: "a clipped validator_feedback",
			log: `{"v":1,"type":"run_start","ts":"2026-09-13T03:00:00Z","model":"gpt-4o","task":"scan the logs"}
{"v":1,"type":"llm_response","ts":"2026-09-13T03:00:01Z","turn":1,"content":"{}","finish_reason":"stop"}
{"v":1,"type":"validator_feedback","ts":"2026-09-13T03:00:02Z","turn":1,"content":"fix it...[clipped]"}`,
			want:     resume.ErrClipped,
			contains: []string{"feedback", "turn 1", "limits.max_logged_field: 0"},
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
			// A torn tail ends a history; with no complete line before it
			// there is no history to end, so the file is damage rather than
			// an empty log - and the two ask different things of the operator.
			name:     "a file that is only a torn line",
			log:      "hello",
			want:     resume.ErrMalformed,
			contains: []string{"line 1"},
		},
		{
			name:     "a file that is only a garbage line with a newline",
			log:      "hello\n",
			want:     resume.ErrMalformed,
			contains: []string{"line 1"},
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
	_, err := resume.Read(context.Background(), path, resume.Options{})
	if !errors.Is(err, resume.ErrNotResumable) {
		t.Fatalf("Read(%s) = %v, want ErrNotResumable", path, err)
	}
	if !strings.Contains(err.Error(), path) {
		t.Errorf("error %q does not name the path %q", err, path)
	}
}

// A hard kill leaves a prefix of the event it was writing on disk. That is the
// case resume exists for, so the history ends at the last complete line.
func TestTornFinalLineEndsTheHistory(t *testing.T) {
	log := runStart + "\n" +
		`{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":1,"tool_call_ids":["c1"],"finish_reason":"tool_calls"}` + "\n" +
		`{"v":1,"type":"tool_call","ts":"2026-09-05T03:00:02Z","tool_call_id":"c1","tool":"fs_read","args":"{}"}` + "\n" +
		`{"v":1,"type":"tool_result","ts":"2026-09-05T03:00:03Z","tool_call_id":"c1","tool":"fs_read","resu`
	got, err := resume.ReadFrom(strings.NewReader(log), resume.Options{})
	if err != nil {
		t.Fatalf("ReadFrom = %v, want no error", err)
	}
	if got.LastTurn != 1 {
		t.Errorf("LastTurn = %d, want 1", got.LastTurn)
	}
	// The torn tool_result never happened as far as the log can prove, so its
	// call is pending, not answered.
	if !reflect.DeepEqual(got.Pending, []string{"c1"}) {
		t.Errorf("Pending = %#v, want [c1]", got.Pending)
	}
	assertMessages(t, got.Messages, []llm.Message{
		user("scan the logs"),
		assistant("", llm.ToolCall{ID: "c1", Name: "fs_read", Arguments: "{}"}),
		toolMsg("c1", resume.PendingResultMessage),
	})
}

// Pending and the synthetic messages follow the model's call order, which is
// tool_call_ids order - not the order the events happen to be written in.
func TestPendingKeepsCallOrder(t *testing.T) {
	log := runStart + "\n" +
		`{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":1,"content":"both","tool_call_ids":["c1","c2"],"finish_reason":"tool_calls"}` + "\n" +
		`{"v":1,"type":"tool_call","ts":"2026-09-05T03:00:02Z","tool_call_id":"c1","tool":"disk","args":"{}"}` + "\n" +
		`{"v":1,"type":"tool_call","ts":"2026-09-05T03:00:02Z","tool_call_id":"c2","tool":"queue","args":"{}"}` + "\n"
	got, err := resume.ReadFrom(strings.NewReader(log), resume.Options{})
	if err != nil {
		t.Fatalf("ReadFrom = %v, want no error", err)
	}
	if !reflect.DeepEqual(got.Pending, []string{"c1", "c2"}) {
		t.Errorf("Pending = %#v, want [c1 c2]", got.Pending)
	}
	assertMessages(t, got.Messages, []llm.Message{
		user("scan the logs"),
		assistant("both",
			llm.ToolCall{ID: "c1", Name: "disk", Arguments: "{}"},
			llm.ToolCall{ID: "c2", Name: "queue", Arguments: "{}"}),
		toolMsg("c1", resume.PendingResultMessage),
		toolMsg("c2", resume.PendingResultMessage),
	})
}

// The assistant message's calls come back in tool_call_ids order even when the
// events were written in another one, because that order is the model's.
func TestToolCallsFollowTheRequestedOrder(t *testing.T) {
	log := runStart + "\n" +
		`{"v":1,"type":"llm_response","ts":"2026-09-05T03:00:01Z","turn":1,"tool_call_ids":["c1","c2"],"finish_reason":"tool_calls"}` + "\n" +
		`{"v":1,"type":"tool_call","ts":"2026-09-05T03:00:02Z","tool_call_id":"c2","tool":"queue","args":"{}"}` + "\n" +
		`{"v":1,"type":"tool_call","ts":"2026-09-05T03:00:02Z","tool_call_id":"c1","tool":"disk","args":"{}"}` + "\n" +
		`{"v":1,"type":"tool_result","ts":"2026-09-05T03:00:03Z","tool_call_id":"c2","tool":"queue","result":"empty","outcome":"ok"}` + "\n" +
		`{"v":1,"type":"tool_result","ts":"2026-09-05T03:00:04Z","tool_call_id":"c1","tool":"disk","result":"61%","outcome":"ok"}` + "\n"
	got, err := resume.ReadFrom(strings.NewReader(log), resume.Options{})
	if err != nil {
		t.Fatalf("ReadFrom = %v, want no error", err)
	}
	want := []llm.ToolCall{
		{ID: "c1", Name: "disk", Arguments: "{}"},
		{ID: "c2", Name: "queue", Arguments: "{}"},
	}
	if !reflect.DeepEqual(got.Messages[1].ToolCalls, want) {
		t.Errorf("ToolCalls = %#v, want %#v", got.Messages[1].ToolCalls, want)
	}
}

// errReader fails mid-stream the way a truncated pipe or an unreadable file
// does: the read error is reported, not mistaken for end of file.
type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, errors.New("disk went away") }

func TestReadFromReaderError(t *testing.T) {
	_, err := resume.ReadFrom(errReader{}, resume.Options{})
	if err == nil || !strings.Contains(err.Error(), "disk went away") {
		t.Fatalf("ReadFrom error = %v, want the read failure", err)
	}
}

func TestReadMissingFile(t *testing.T) {
	_, err := resume.Read(context.Background(), filepath.Join(t.TempDir(), "nope.jsonl"), resume.Options{})
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

// A carrier is the raw content array, and for a turn whose calls were never
// dispatched that array still announces those tool_use blocks - echoing it
// would rebuild exactly the unanswered-call history the drop rule exists to
// avoid. So a turn that loses a call to the drop rule loses its carrier too.
func TestUndispatchedCallDropsTheCarrier(t *testing.T) {
	const log = `{"v":1,"type":"run_start","ts":"2026-09-05T09:00:00Z","model":"claude-sonnet-4","provider":"anthropic","task":"check the log"}
{"v":1,"type":"llm_response","ts":"2026-09-05T09:00:02Z","turn":1,"tool_call_ids":["call_1"],"finish_reason":"tool_use","reasoning_bytes":150,"reasoning":"[{\"type\":\"thinking\",\"thinking\":\"Read it.\",\"signature\":\"Ej8=\"},{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"fs_read\",\"input\":{\"path\":\"app.log\"}}]"}
`
	got, err := resume.ReadFrom(strings.NewReader(log), resume.Options{Provider: "anthropic", Model: "claude-sonnet-4"})
	if err != nil {
		t.Fatalf("ReadFrom = %v, want no error", err)
	}
	// No text, no dispatched call, and the carrier is gone with the call:
	// the empty turn is dropped and the history is the task alone.
	assertMessages(t, got.Messages, []llm.Message{user("check the log")})
	if len(got.Pending) != 0 {
		t.Errorf("Pending = %v, want none (the call was never dispatched)", got.Pending)
	}
}

// writeChainLog writes one log of a chain into dir and returns its path. The
// lines are given whole so a test reads like the files an operator has.
func writeChainLog(t *testing.T, dir, name string, lines ...string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// resumedStart renders the run_start of a log that continued another one.
func resumedStart(from, instruction string) string {
	ev := map[string]any{
		"v": 1, "type": "run_start", "ts": "2026-09-13T03:00:00Z",
		"model": "gpt-4o", "provider": "openai", "task": "scan the logs",
		"resumed_from": from, "resumed_turn": 1,
	}
	if instruction != "" {
		ev["resumed_instruction"] = instruction
	}
	b, _ := json.Marshal(ev)
	return string(b)
}

const (
	turn1Call   = `{"v":1,"type":"llm_response","ts":"2026-09-13T03:00:01Z","turn":1,"content":"reading","tool_call_ids":["c1"],"finish_reason":"tool_calls"}`
	turn1Dispat = `{"v":1,"type":"tool_call","ts":"2026-09-13T03:00:02Z","tool_call_id":"c1","tool":"fs_read","args":"{}"}`
	turn1Result = `{"v":1,"type":"tool_result","ts":"2026-09-13T03:00:03Z","tool_call_id":"c1","tool":"fs_read","result":"ERROR at 03:00"}`
	finalTurn   = `{"v":1,"type":"llm_response","ts":"2026-09-13T03:00:05Z","turn":1,"content":"one error at 03:00","finish_reason":"stop"}`
)

// TestReadFollowsTheChain (issue #31): a log produced by a resume names the log
// it continued, and reading it rebuilds BOTH - the original run's turns, the
// instruction the resume was given, then the retry's turns - as one
// conversation, with the turn count and the link count of the whole chain.
func TestReadFollowsTheChain(t *testing.T) {
	dir := t.TempDir()
	// Run 1 died with c1 dispatched and unanswered.
	root := writeChainLog(t, dir, "run-1.jsonl", runStart, turn1Call, turn1Dispat)
	// Run 2 resumed it with an instruction and finished.
	retry := writeChainLog(t, dir, "run-2.jsonl", resumedStart(root, "be brief"), finalTurn)
	// Run 3 resumed the retry, added another instruction, and died before its
	// first turn.
	third := writeChainLog(t, dir, "run-3.jsonl", resumedStart(retry, "now count them"))

	got, err := resume.Read(context.Background(), third, resume.Options{Provider: "openai", Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("Read = %v", err)
	}
	if got.Links != 3 || got.LastTurn != 2 {
		t.Errorf("Links = %d, LastTurn = %d; want 3 and 2 (one turn per run that made one)", got.Links, got.LastTurn)
	}
	if got.Task != "scan the logs" || got.Completed {
		t.Errorf("Task = %q, Completed = %v; want the root task and an unanswered instruction", got.Task, got.Completed)
	}
	if len(got.Pending) != 0 {
		t.Errorf("Pending = %v; the root's pending call was answered (stood in) by run 2", got.Pending)
	}
	assertMessages(t, got.Messages, []llm.Message{
		user("scan the logs"),
		assistant("reading", llm.ToolCall{ID: "c1", Name: "fs_read", Arguments: "{}"}),
		toolMsg("c1", resume.PendingResultMessage),
		user("be brief"),
		assistant("one error at 03:00"),
		user("now count them"),
	})

	t.Run("the middle link alone", func(t *testing.T) {
		got, err := resume.Read(context.Background(), retry, resume.Options{})
		if err != nil {
			t.Fatalf("Read = %v", err)
		}
		if got.Links != 2 || got.LastTurn != 2 || !got.Completed {
			t.Errorf("Links = %d, LastTurn = %d, Completed = %v; want 2, 2, true", got.Links, got.LastTurn, got.Completed)
		}
		if n := len(got.Messages); n != 5 || got.Messages[n-1].Content != "one error at 03:00" {
			t.Errorf("Messages = %+v", got.Messages)
		}
	})
}

// TestChainCompletedFollowsTheParent: a link that died before its first turn
// and added no instruction leaves the question "is there anything left to
// answer?" to the log it continued.
func TestChainCompletedFollowsTheParent(t *testing.T) {
	dir := t.TempDir()
	root := writeChainLog(t, dir, "run-1.jsonl", runStart, finalTurn)
	empty := writeChainLog(t, dir, "run-2.jsonl", resumedStart(root, ""))
	got, err := resume.Read(context.Background(), empty, resume.Options{})
	if err != nil {
		t.Fatalf("Read = %v", err)
	}
	if !got.Completed || got.LastTurn != 1 || got.Links != 2 {
		t.Errorf("Completed = %v, LastTurn = %d, Links = %d; want true, 1, 2", got.Completed, got.LastTurn, got.Links)
	}
}

// TestChainCarriersPerLink: each link decides its own carriers. A link run by
// another model keeps its payloads out of the replay while the links that
// match the current model still get theirs back.
func TestChainCarriersPerLink(t *testing.T) {
	dir := t.TempDir()
	reasoning := `{"v":1,"type":"llm_response","ts":"2026-09-13T03:00:05Z","turn":1,"content":"done","finish_reason":"stop","reasoning_bytes":11,"reasoning":"[{\"a\":1}]"}`
	root := writeChainLog(t, dir, "run-1.jsonl", runStart, reasoning)
	otherModel := strings.Replace(resumedStart(root, "again"), `"model":"gpt-4o"`, `"model":"gpt-4o-mini"`, 1)
	retry := writeChainLog(t, dir, "run-2.jsonl", otherModel, reasoning)

	got, err := resume.Read(context.Background(), retry, resume.Options{Provider: "openai", Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("Read = %v", err)
	}
	if !got.Carriers {
		t.Fatal("the root link matches the current model, so its carrier must come back")
	}
	// Messages: task, root assistant (carrier), instruction, retry assistant (no carrier).
	if len(got.Messages) != 4 {
		t.Fatalf("Messages = %+v", got.Messages)
	}
	if len(got.Messages[1].Reasoning) == 0 {
		t.Errorf("root turn lost its carrier: %+v", got.Messages[1])
	}
	if len(got.Messages[3].Reasoning) != 0 {
		t.Errorf("a turn produced by another model must not carry its payload: %+v", got.Messages[3])
	}
}

// TestChainRefusals: a chain is refused whole when a link cannot be read, is
// clipped where the history needs it, or never ends.
func TestChainRefusals(t *testing.T) {
	dir := t.TempDir()
	t.Run("a missing link names both files", func(t *testing.T) {
		missing := filepath.Join(dir, "gone.jsonl")
		child := writeChainLog(t, dir, "run-2.jsonl", resumedStart(missing, ""), finalTurn)
		_, err := resume.Read(context.Background(), child, resume.Options{})
		if err == nil || !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("err = %v, want a not-exist error", err)
		}
		for _, want := range []string{child, "following resumed_from", missing} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("err %q does not name %q", err, want)
			}
		}
	})
	t.Run("a clipped instruction", func(t *testing.T) {
		root := writeChainLog(t, dir, "run-1.jsonl", runStart, finalTurn)
		child := writeChainLog(t, dir, "run-clip.jsonl", resumedStart(root, "be brief"+session.ClipMarker))
		_, err := resume.Read(context.Background(), child, resume.Options{})
		if !errors.Is(err, resume.ErrClipped) || !strings.Contains(err.Error(), "instruction") {
			t.Fatalf("err = %v, want ErrClipped naming the instruction", err)
		}
	})
	t.Run("a cycle hits the link cap", func(t *testing.T) {
		self := filepath.Join(dir, "loop.jsonl")
		writeChainLog(t, dir, "loop.jsonl", resumedStart(self, ""), finalTurn)
		_, err := resume.Read(context.Background(), self, resume.Options{})
		if !errors.Is(err, resume.ErrNotResumable) || !strings.Contains(err.Error(), "longer than") {
			t.Fatalf("err = %v, want ErrNotResumable naming the chain cap", err)
		}
	})
	t.Run("a cancelled context stops at the first link", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		root := writeChainLog(t, dir, "run-ctx.jsonl", runStart, finalTurn)
		_, err := resume.Read(ctx, root, resume.Options{})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want context.Canceled", err)
		}
	})
}
