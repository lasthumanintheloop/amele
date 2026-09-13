package llm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// anSSE renders Anthropic-style events: `event: <type>` plus the JSON.
func anSSE(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		b.WriteString("event: x\ndata: " + e + "\n\n")
	}
	return b.String()
}

// decodeRequest decodes a request body as a generic JSON object.
func decodeRequest(t *testing.T, r *http.Request) map[string]any {
	t.Helper()
	var req map[string]any
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		t.Errorf("decoding request: %v", err)
	}
	return req
}

// anStreamServer answers every request with body and records the request
// body's stream field.
func anStreamServer(t *testing.T, body string) (*httptest.Server, *map[string]any) {
	t.Helper()
	var seen map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = decodeRequest(t, r)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv, &seen
}

func TestAnthropicChatStreamAssemblesEverything(t *testing.T) {
	srv, seen := anStreamServer(t, anSSE(
		`{"type":"message_start","message":{"id":"m1","usage":{"input_tokens":25,"cache_read_input_tokens":10,"cache_creation_input_tokens":5,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"read the "}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"log"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"c2ln"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Let me "}}`,
		`{"type":"ping"}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"look."}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"fs_read","input":{}}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":" \"app.log\"}"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":42}}`,
		`{"type":"message_stop"}`,
	))
	client := &AnthropicClient{BaseURL: srv.URL, APIKey: "k"}
	var got []string
	resp, err := client.ChatStream(context.Background(), Request{Model: "claude", Messages: []Message{{Role: RoleUser, Content: "x"}}}, collect(&got))
	if err != nil {
		t.Fatal(err)
	}
	if (*seen)["stream"] != true {
		t.Errorf("request did not ask to stream: %v", *seen)
	}
	if strings.Join(got, "|") != "Let me |look." {
		t.Errorf("sink got %q (thinking and tool input must never reach it)", got)
	}
	if resp.Message.Content != "Let me look." || resp.FinishReason != "tool_calls" {
		t.Errorf("content %q, finish %q", resp.Message.Content, resp.FinishReason)
	}
	if len(resp.Message.ToolCalls) != 1 || resp.Message.ToolCalls[0] != (ToolCall{ID: "toolu_1", Name: "fs_read", Arguments: `{"path":"app.log"}`}) {
		t.Errorf("tool calls = %+v", resp.Message.ToolCalls)
	}
	wantCarrier := `[{"type":"thinking","thinking":"read the log","signature":"c2ln"},{"type":"text","text":"Let me look."},{"type":"tool_use","id":"toolu_1","name":"fs_read","input":{"path":"app.log"}}]`
	if string(resp.Message.Reasoning) != wantCarrier {
		t.Errorf("carrier =\n%s\nwant\n%s", resp.Message.Reasoning, wantCarrier)
	}
	// input_tokens + both cache counters, as on the non-streaming path.
	if resp.UsageMissing || resp.Usage.InputTokens != 40 || resp.Usage.OutputTokens != 42 || resp.Usage.CacheReadTokens != 10 || resp.Usage.CacheWriteTokens != 5 {
		t.Errorf("usage = %+v (missing %v)", resp.Usage, resp.UsageMissing)
	}
}

func TestAnthropicChatStreamPlainAnswerHasNoCarrier(t *testing.T) {
	srv, _ := anStreamServer(t, anSSE(
		`{"type":"message_start","message":{"usage":{"input_tokens":3,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`,
		`{"type":"message_stop"}`,
	))
	client := &AnthropicClient{BaseURL: srv.URL, APIKey: "k"}
	resp, err := client.ChatStream(context.Background(), Request{Model: "claude", Messages: []Message{{Role: RoleUser, Content: "x"}}}, func(string) {})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Content != "hi" || resp.FinishReason != "stop" || resp.Message.Reasoning != nil {
		t.Errorf("response = %+v", resp)
	}
	if resp.Usage.InputTokens != 3 || resp.Usage.OutputTokens != 2 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestAnthropicChatStreamFailures(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{"a mid-stream error event", anSSE(
			`{"type":"message_start","message":{"usage":{"input_tokens":3,"output_tokens":1}}}`,
			`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`,
		), "overloaded_error: Overloaded"},
		{"a stream cut before message_stop", anSSE(
			`{"type":"message_start","message":{"usage":{"input_tokens":3,"output_tokens":1}}}`,
			`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"par"}}`,
		), "before message_stop"},
		{"a delta for an unknown block", anSSE(
			`{"type":"message_start","message":{"usage":{"input_tokens":3,"output_tokens":1}}}`,
			`{"type":"content_block_delta","index":4,"delta":{"type":"text_delta","text":"x"}}`,
		), "unknown block 4"},
		{"no message at all", anSSE(`{"type":"ping"}`), "no message"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := anStreamServer(t, tc.body)
			client := &AnthropicClient{BaseURL: srv.URL, APIKey: "k", MaxAttempts: 1}
			_, err := client.ChatStream(context.Background(), Request{Model: "claude", Messages: []Message{{Role: RoleUser, Content: "x"}}}, func(string) {})
			if !errors.Is(err, ErrProvider) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want ErrProvider containing %q", err, tc.want)
			}
		})
	}
}

func TestAnthropicChatStreamFallsBackWhenStreamingIsRejected(t *testing.T) {
	var streamed, plain int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		req := decodeRequest(t, r)
		if req["stream"] == true {
			streamed++
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"stream: not supported"}}`))
			return
		}
		plain++
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"whole"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	t.Cleanup(srv.Close)
	client := &AnthropicClient{BaseURL: srv.URL, APIKey: "k"}
	var got []string
	resp, err := client.ChatStream(context.Background(), Request{Model: "claude", Messages: []Message{{Role: RoleUser, Content: "x"}}}, collect(&got))
	if err != nil {
		t.Fatal(err)
	}
	if streamed != 1 || plain != 1 || resp.Message.Content != "whole" || strings.Join(got, "") != "whole" {
		t.Errorf("streamed %d plain %d content %q sink %q", streamed, plain, resp.Message.Content, got)
	}
}
