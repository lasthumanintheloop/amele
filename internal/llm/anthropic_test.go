package llm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// anthropicServer runs an httptest server for the Anthropic Messages API wire
// format. It asserts the invariants every request must satisfy (path and
// protocol headers) and hands the decoded body to the test's handler.
func anthropicServer(t *testing.T, handler func(w http.ResponseWriter, req map[string]any)) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		if got := r.Header.Get("anthropic-version"); got != "2023-06-01" {
			t.Errorf("anthropic-version header: got %q", got)
		}
		if got := r.Header.Get("Content-Type"); got != "application/json" {
			t.Errorf("content-type header: got %q", got)
		}
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decoding request: %v", err)
		}
		handler(w, req)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// headerCapture records the named request header before delegating, so a test
// can assert on auth headers without threading them through anthropicServer or
// chatServer. Shared by both clients' tests (x-api-key and Authorization).
func headerCapture(next http.Handler, name string, into *string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*into = r.Header.Get(name)
		next.ServeHTTP(w, r)
	})
}

func anOKBody(text string) string {
	return `{
		"content": [{"type": "text", "text": ` + jsonStr(text) + `}],
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`
}

// TestAnthropicChatBasics pins the request shape: system prompt hoisted to the
// top-level "system" field (Anthropic rejects a system role inside messages),
// the required max_tokens defaulting to defaultAnthropicMaxOutput, tool
// definitions with input_schema passed through verbatim, and the x-api-key
// auth header. The BaseURL carries a trailing slash to pin the trim.
func TestAnthropicChatBasics(t *testing.T) {
	var gotKey string
	srv := anthropicServer(t, func(w http.ResponseWriter, req map[string]any) {
		assertBasicRequestShape(t, req)
		_, _ = w.Write([]byte(anOKBody("hello")))
	})
	srv.Config.Handler = headerCapture(srv.Config.Handler, "x-api-key", &gotKey)

	client := &AnthropicClient{BaseURL: srv.URL + "/", APIKey: testAPIKey}
	resp, err := client.Chat(context.Background(), Request{
		Model: "claude-test",
		Messages: []Message{
			{Role: RoleSystem, Content: "sys"},
			{Role: RoleUser, Content: "hi"},
		},
		Tools: []ToolDef{{Name: "fs_read", Description: "read", Parameters: json.RawMessage(`{"type":"object"}`)}},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Message.Content != "hello" {
		t.Errorf("content: got %q", resp.Message.Content)
	}
	if resp.Message.Role != RoleAssistant {
		t.Errorf("role: got %q", resp.Message.Role)
	}
	if resp.Usage.Total() != 15 {
		t.Errorf("usage total: got %d", resp.Usage.Total())
	}
	if resp.FinishReason != "stop" {
		t.Errorf("finish reason: got %q", resp.FinishReason)
	}
	if gotKey != testAPIKey {
		t.Errorf("x-api-key header: got %q", gotKey)
	}
}

// testAPIKey is a fake credential for header assertions; the neutral spelling
// keeps secret scanners quiet.
const testAPIKey = "unit-test-key"

// assertBasicRequestShape checks the wire body of TestAnthropicChatBasics:
// model, defaulted max_tokens, hoisted system prompt, user text block, and
// verbatim input_schema passthrough.
func assertBasicRequestShape(t *testing.T, req map[string]any) {
	t.Helper()
	if req["model"] != "claude-test" {
		t.Errorf("model: got %v", req["model"])
	}
	if req["max_tokens"] != float64(defaultAnthropicMaxOutput) {
		t.Errorf("max_tokens: got %v, want %d", req["max_tokens"], defaultAnthropicMaxOutput)
	}
	if req["system"] != "sys" {
		t.Errorf("system: got %v", req["system"])
	}
	msgs := req["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages: got %d, want 1 (system must be extracted)", len(msgs))
	}
	m := msgs[0].(map[string]any)
	if m["role"] != "user" {
		t.Errorf("role: got %v", m["role"])
	}
	blocks := m["content"].([]any)
	b := blocks[0].(map[string]any)
	if b["type"] != "text" || b["text"] != "hi" {
		t.Errorf("content block: %v", b)
	}
	tools := req["tools"].([]any)
	tool := tools[0].(map[string]any)
	if tool["name"] != "fs_read" || tool["description"] != "read" {
		t.Errorf("tool: %v", tool)
	}
	schema, err := json.Marshal(tool["input_schema"])
	if err != nil {
		t.Errorf("marshaling input_schema: %v", err)
	}
	if string(schema) != `{"type":"object"}` {
		t.Errorf("input_schema: got %s", schema)
	}
}

// TestAnthropicMaxOutputTokensOverride: a configured MaxOutputTokens must be
// sent instead of the default.
func TestAnthropicMaxOutputTokensOverride(t *testing.T) {
	srv := anthropicServer(t, func(w http.ResponseWriter, req map[string]any) {
		if req["max_tokens"] != float64(512) {
			t.Errorf("max_tokens: got %v, want 512", req["max_tokens"])
		}
		_, _ = w.Write([]byte(anOKBody("ok")))
	})
	client := &AnthropicClient{BaseURL: srv.URL, MaxOutputTokens: 512}
	if _, err := client.Chat(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}}); err != nil {
		t.Fatal(err)
	}
}

// TestAnthropicToolUseResponse: tool_use blocks become ToolCalls with the
// input compacted to a JSON string, and text blocks concatenate into Content.
func TestAnthropicToolUseResponse(t *testing.T) {
	srv := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = w.Write([]byte(`{
			"content": [
				{"type": "text", "text": "I'll read "},
				{"type": "text", "text": "the file."},
				{"type": "tool_use", "id": "toolu_1", "name": "fs_read", "input": {"path": "a.txt"}}
			],
			"stop_reason": "tool_use",
			"usage": {"input_tokens": 1, "output_tokens": 2}
		}`))
	})

	client := &AnthropicClient{BaseURL: srv.URL}
	resp, err := client.Chat(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Content != "I'll read the file." {
		t.Errorf("content: got %q", resp.Message.Content)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls: got %d", len(resp.Message.ToolCalls))
	}
	tc := resp.Message.ToolCalls[0]
	if tc.ID != "toolu_1" || tc.Name != "fs_read" || tc.Arguments != `{"path":"a.txt"}` {
		t.Errorf("tool call: %+v", tc)
	}
	if resp.FinishReason != "tool_calls" {
		t.Errorf("finish reason: got %q", resp.FinishReason)
	}
}

// TestAnthropicAssistantEchoAndToolResultMerge pins the multi-turn tool wire
// shape: the assistant tool_use echo becomes content blocks (text first when
// non-empty, then one tool_use per call, empty Arguments => {}), and the two
// CONSECUTIVE RoleTool messages the loop emits sequentially are merged into a
// SINGLE user message - Anthropic requires all parallel tool results in one
// user turn.
func TestAnthropicAssistantEchoAndToolResultMerge(t *testing.T) {
	srv := anthropicServer(t, func(w http.ResponseWriter, req map[string]any) {
		msgs := req["messages"].([]any)
		if len(msgs) != 3 {
			t.Fatalf("messages: got %d, want 3 (user, assistant, merged tool results)", len(msgs))
		}
		assertAssistantEcho(t, msgs[1].(map[string]any))
		assertMergedToolResults(t, msgs[2].(map[string]any))
		_, _ = w.Write([]byte(anOKBody("done")))
	})

	client := &AnthropicClient{BaseURL: srv.URL}
	_, err := client.Chat(context.Background(), Request{
		Model: "m",
		Messages: []Message{
			{Role: RoleUser, Content: "read a and b"},
			{Role: RoleAssistant, Content: "reading both", ToolCalls: []ToolCall{
				{ID: "c1", Name: "fs_read", Arguments: `{"path":"a.txt"}`},
				{ID: "c2", Name: "fs_list", Arguments: ""},
			}},
			{Role: RoleTool, ToolCallID: "c1", Content: "aaa"},
			{Role: RoleTool, ToolCallID: "c2", Content: "bbb"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// assertAssistantEcho checks the assistant tool_use echo: an optional leading
// text block, then one tool_use block per call with Arguments embedded as raw
// JSON (empty Arguments => {}).
func assertAssistantEcho(t *testing.T, asst map[string]any) {
	t.Helper()
	if asst["role"] != "assistant" {
		t.Errorf("assistant role: got %v", asst["role"])
	}
	blocks := asst["content"].([]any)
	if len(blocks) != 3 {
		t.Fatalf("assistant blocks: got %d, want 3 (text + 2 tool_use)", len(blocks))
	}
	text := blocks[0].(map[string]any)
	if text["type"] != "text" || text["text"] != "reading both" {
		t.Errorf("assistant text block: %v", text)
	}
	use1 := blocks[1].(map[string]any)
	if use1["type"] != "tool_use" || use1["id"] != "c1" || use1["name"] != "fs_read" {
		t.Errorf("first tool_use: %v", use1)
	}
	in1, _ := json.Marshal(use1["input"])
	if string(in1) != `{"path":"a.txt"}` {
		t.Errorf("first tool_use input: got %s", in1)
	}
	use2 := blocks[2].(map[string]any)
	if use2["id"] != "c2" {
		t.Errorf("second tool_use: %v", use2)
	}
	in2, _ := json.Marshal(use2["input"])
	if string(in2) != `{}` {
		t.Errorf("empty Arguments must become {}: got %s", in2)
	}
}

// assertMergedToolResults checks that both tool results landed in ONE user
// message with the correct tool_use_id on each block.
func assertMergedToolResults(t *testing.T, results map[string]any) {
	t.Helper()
	if results["role"] != "user" {
		t.Errorf("tool results role: got %v (must be user)", results["role"])
	}
	rblocks := results["content"].([]any)
	if len(rblocks) != 2 {
		t.Fatalf("tool result blocks: got %d, want 2 merged into one message", len(rblocks))
	}
	r1 := rblocks[0].(map[string]any)
	if r1["type"] != "tool_result" || r1["tool_use_id"] != "c1" || r1["content"] != "aaa" {
		t.Errorf("first tool_result: %v", r1)
	}
	r2 := rblocks[1].(map[string]any)
	if r2["type"] != "tool_result" || r2["tool_use_id"] != "c2" || r2["content"] != "bbb" {
		t.Errorf("second tool_result: %v", r2)
	}
}

// TestAnthropicToolResultsDoNotMergeAcrossUserTurn: only CONSECUTIVE RoleTool
// messages merge - a tool result arriving after an intervening user message
// starts a fresh user turn instead of being folded into the earlier one.
func TestAnthropicToolResultsDoNotMergeAcrossUserTurn(t *testing.T) {
	srv := anthropicServer(t, func(w http.ResponseWriter, req map[string]any) {
		msgs := req["messages"].([]any)
		if len(msgs) != 4 {
			t.Fatalf("messages: got %d, want 4", len(msgs))
		}
		last := msgs[3].(map[string]any)
		blocks := last["content"].([]any)
		if len(blocks) != 1 {
			t.Errorf("last message blocks: got %d, want 1 (no merge across the user turn)", len(blocks))
		}
		_, _ = w.Write([]byte(anOKBody("ok")))
	})

	client := &AnthropicClient{BaseURL: srv.URL}
	_, err := client.Chat(context.Background(), Request{
		Model: "m",
		Messages: []Message{
			{Role: RoleTool, ToolCallID: "c1", Content: "aaa"},
			{Role: RoleUser, Content: "and now?"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c2", Name: "t", Arguments: "{}"}}},
			{Role: RoleTool, ToolCallID: "c2", Content: "bbb"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestAnthropicStopReasonMapping pins the full translation table into the
// OpenAI-compatible finish reasons the loop understands; unknown values pass
// through verbatim for the loop's defensive badFinish handling.
func TestAnthropicStopReasonMapping(t *testing.T) {
	tests := []struct {
		stopReason string
		want       string
	}{
		{"end_turn", "stop"},
		{"max_tokens", "length"},
		// Context-window exhaustion is a truncation: mapping it to "length"
		// routes it into badFinish's hard-failure branch. Left to the
		// passthrough default it would reach the unknown-reason branch, which
		// accepts non-empty content - a truncated answer exiting 0 in cron.
		{"model_context_window_exceeded", "length"},
		{"refusal", "content_filter"},
		{"tool_use", "tool_calls"},
		{"pause_turn", "pause_turn"},
		{"", ""},
	}
	for _, tt := range tests {
		t.Run(tt.stopReason, func(t *testing.T) {
			srv := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
				_, _ = w.Write([]byte(`{
					"content": [{"type": "text", "text": "x"}],
					"stop_reason": ` + jsonStr(tt.stopReason) + `,
					"usage": {"input_tokens": 1, "output_tokens": 1}
				}`))
			})
			client := &AnthropicClient{BaseURL: srv.URL}
			resp, err := client.Chat(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}})
			if err != nil {
				t.Fatal(err)
			}
			if resp.FinishReason != tt.want {
				t.Errorf("finish reason: got %q, want %q", resp.FinishReason, tt.want)
			}
		})
	}
}

// TestAnthropicMissingUsage: an endpoint omitting the usage object must be
// reported as UsageMissing so token budgets fail closed (same contract as the
// OpenAI client).
func TestAnthropicMissingUsage(t *testing.T) {
	srv := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = w.Write([]byte(`{"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn"}`))
	})
	client := &AnthropicClient{BaseURL: srv.URL}
	resp, err := client.Chat(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}})
	if err != nil {
		t.Fatal(err)
	}
	if !resp.UsageMissing {
		t.Error("UsageMissing must be true when the provider omits usage")
	}

	srv2 := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = w.Write([]byte(anOKBody("hi")))
	})
	client2 := &AnthropicClient{BaseURL: srv2.URL}
	resp2, err := client2.Chat(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}})
	if err != nil {
		t.Fatal(err)
	}
	if resp2.UsageMissing {
		t.Error("UsageMissing must be false when usage is reported")
	}
}

// TestAnthropicRetryAfterHonored: a 429 carrying Retry-After (seconds form)
// must stretch the backoff to the provider's wish.
func TestAnthropicRetryAfterHonored(t *testing.T) {
	var calls int
	srv := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		if calls == 1 {
			w.Header().Set("Retry-After", "7")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_, _ = w.Write([]byte(anOKBody("ok")))
	})

	var delays []time.Duration
	client := &AnthropicClient{BaseURL: srv.URL, Sleep: func(_ context.Context, d time.Duration) error {
		delays = append(delays, d)
		return nil
	}}
	if _, err := client.Chat(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}}); err != nil {
		t.Fatal(err)
	}
	if len(delays) != 1 || delays[0] != 7*time.Second {
		t.Errorf("delays: got %v want [7s]", delays)
	}
}

// anthropicRetryTwiceServer answers the first two calls with 429 and then
// succeeds, so a test observes exactly two backoff waits.
func anthropicRetryTwiceServer(t *testing.T) *httptest.Server {
	t.Helper()
	var calls int
	return anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		if calls <= 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_, _ = w.Write([]byte(anOKBody("ok")))
	})
}

// TestAnthropicBackoffHonorsInitialBackoff: the native wire honors the same
// configured retry rhythm as the OpenAI-compatible one - a policy that applied
// to only one of the two clients would be a trap when a config switches wires.
func TestAnthropicBackoffHonorsInitialBackoff(t *testing.T) {
	srv := anthropicRetryTwiceServer(t)

	var delays []time.Duration
	client := &AnthropicClient{
		BaseURL:        srv.URL,
		InitialBackoff: 200 * time.Millisecond,
		Sleep:          recordDelays(&delays),
	}
	if _, err := client.Chat(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}}); err != nil {
		t.Fatal(err)
	}
	want := []time.Duration{200 * time.Millisecond, 400 * time.Millisecond}
	if !slices.Equal(delays, want) {
		t.Errorf("delays: got %v want %v", delays, want)
	}
}

// TestAnthropicBackoffDefaultsUnchanged: no InitialBackoff means the 1s/2s
// ladder this client has always used.
func TestAnthropicBackoffDefaultsUnchanged(t *testing.T) {
	srv := anthropicRetryTwiceServer(t)

	var delays []time.Duration
	client := &AnthropicClient{BaseURL: srv.URL, Sleep: recordDelays(&delays)}
	if _, err := client.Chat(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}}); err != nil {
		t.Fatal(err)
	}
	want := []time.Duration{time.Second, 2 * time.Second}
	if !slices.Equal(delays, want) {
		t.Errorf("delays: got %v want %v", delays, want)
	}
}

// TestAnthropic529Retried: 529 is Anthropic's "overloaded" status and must be
// retried like any transient server-side failure.
func TestAnthropic529Retried(t *testing.T) {
	var calls int
	srv := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		if calls == 1 {
			w.WriteHeader(529)
			_, _ = w.Write([]byte(`{"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}`))
			return
		}
		_, _ = w.Write([]byte(anOKBody("recovered")))
	})

	var sleeps int
	client := &AnthropicClient{BaseURL: srv.URL, Sleep: noSleep(t, &sleeps)}
	resp, err := client.Chat(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Content != "recovered" {
		t.Errorf("content: %q", resp.Message.Content)
	}
	if calls != 2 || sleeps != 1 {
		t.Errorf("calls=%d sleeps=%d, want 2 and 1", calls, sleeps)
	}
}

// TestAnthropicDoesNotRetryOn400: request errors are permanent - retrying
// them burns the whole budget on a request the provider already rejected.
func TestAnthropicDoesNotRetryOn400(t *testing.T) {
	var calls int
	srv := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"type": "error", "error": {"type": "invalid_request_error", "message": "bad model"}}`))
	})

	client := &AnthropicClient{BaseURL: srv.URL}
	_, err := client.Chat(context.Background(), Request{Model: "m"})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, ErrProvider) {
		t.Errorf("should wrap ErrProvider: %v", err)
	}
	if !strings.Contains(err.Error(), "bad model") {
		t.Errorf("error should carry body snippet: %v", err)
	}
	if calls != 1 {
		t.Errorf("400 must not be retried, got %d calls", calls)
	}
}

// TestAnthropicRequestTimeoutNotRetried: a generation that exceeds the
// per-request budget will exceed it again - the failure must be immediate and
// name the provider.request_timeout knob (parity with the OpenAI client).
func TestAnthropicRequestTimeoutNotRetried(t *testing.T) {
	// atomic: the handler is still sleeping in its own goroutine when the
	// test (whose client already timed out) reads the counter.
	var calls atomic.Int32
	srv := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls.Add(1)
		time.Sleep(300 * time.Millisecond)
		_, _ = w.Write([]byte(anOKBody("too late")))
	})

	var sleeps int
	client := &AnthropicClient{BaseURL: srv.URL, RequestTimeout: 30 * time.Millisecond, Sleep: noSleep(t, &sleeps)}
	_, err := client.Chat(context.Background(), Request{Model: "m"})
	if err == nil {
		t.Fatal("expected timeout error")
	}
	if !errors.Is(err, ErrProvider) {
		t.Errorf("should wrap ErrProvider: %v", err)
	}
	if !strings.Contains(err.Error(), "request_timeout") {
		t.Errorf("error must name the config knob: %v", err)
	}
	if calls.Load() != 1 {
		t.Errorf("request timeout must not be retried, got %d calls", calls.Load())
	}
}

// TestAnthropicRetriesExhausted: persistent 5xx must end with the retries
// exhausted error after defaultMaxAttempts tries.
func TestAnthropicRetriesExhausted(t *testing.T) {
	var calls int
	srv := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		w.WriteHeader(http.StatusServiceUnavailable)
	})

	var sleeps int
	client := &AnthropicClient{BaseURL: srv.URL, Sleep: noSleep(t, &sleeps)}
	_, err := client.Chat(context.Background(), Request{Model: "m"})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "retries exhausted") {
		t.Errorf("error: %v", err)
	}
	if calls != defaultMaxAttempts {
		t.Errorf("calls: got %d want %d", calls, defaultMaxAttempts)
	}
}

// The schema path moved: this API has native json_schema enforcement via
// output_config.format, so the request DOES carry the schema and the response
// is not flagged. See TestAnthropicNativeSchemaIsSent and
// TestAnthropicOutputConfigFallback in anthropic_thinking_test.go, which
// replaced the former TestAnthropicResponseFormatIgnored.

// TestAnthropicChatResponseBodyIsBounded mirrors the OpenAI client's cap on
// the success body: an endless stream must end as a provider error, not as
// unbounded memory growth.
func TestAnthropicChatResponseBodyIsBounded(t *testing.T) {
	srv := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"`))
		chunk := strings.Repeat("a", 64*1024)
		for written := 0; written < maxResponseBody+len(chunk); written += len(chunk) {
			if _, err := w.Write([]byte(chunk)); err != nil {
				return
			}
		}
	})
	client := &AnthropicClient{BaseURL: srv.URL}
	_, err := client.Chat(context.Background(), Request{
		Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}},
	})
	if !errors.Is(err, ErrProvider) {
		t.Fatalf("oversized body: got %v, want a provider error", err)
	}
}

// TestAnthropicChatUsageIsSanitized mirrors the OpenAI client's regression:
// provider-reported counts feed the token budget directly, so negative or
// absurd values must never reach the loop's accumulator.
func TestAnthropicChatUsageIsSanitized(t *testing.T) {
	tests := []struct {
		name        string
		usage       string
		wantIn      int
		wantOut     int
		wantMissing bool
	}{
		{"sane", `{"input_tokens": 10, "output_tokens": 5}`, 10, 5, false},
		{"negative", `{"input_tokens": -5, "output_tokens": -1}`, 0, 0, true},
		{
			"absurd",
			`{"input_tokens": 9000000000000000000, "output_tokens": 9000000000000000000}`,
			maxTokensPerResponse, maxTokensPerResponse, false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
				_, _ = w.Write([]byte(`{"content": [{"type": "text", "text": "hi"}],
					"stop_reason": "end_turn", "usage": ` + tt.usage + `}`))
			})
			client := &AnthropicClient{BaseURL: srv.URL}
			resp, err := client.Chat(context.Background(), Request{
				Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}},
			})
			if err != nil {
				t.Fatalf("Chat: %v", err)
			}
			if resp.Usage.InputTokens != tt.wantIn || resp.Usage.OutputTokens != tt.wantOut {
				t.Errorf("usage: got %+v, want {%d %d}", resp.Usage, tt.wantIn, tt.wantOut)
			}
			if resp.UsageMissing != tt.wantMissing {
				t.Errorf("UsageMissing: got %v, want %v", resp.UsageMissing, tt.wantMissing)
			}
		})
	}
}
