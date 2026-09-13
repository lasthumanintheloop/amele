package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"
)

// sseBody joins chunks into a text/event-stream body with the [DONE]
// terminator the OpenAI wire sends.
func sseBody(chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString("data: " + c + "\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

// streamServer answers every request with the given SSE body and records the
// decoded request for assertions on the stream fields.
func streamServer(t *testing.T, body string) (string, *map[string]any) {
	t.Helper()
	var seen map[string]any
	srv := chatServer(t, func(w http.ResponseWriter, req map[string]any) {
		seen = req
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(body))
	})
	return srv.URL + "/v1", &seen
}

// collect returns a sink appending to got.
func collect(got *[]string) func(string) {
	return func(s string) { *got = append(*got, s) }
}

func TestChatStreamAssemblesTextAndUsage(t *testing.T) {
	base, seen := streamServer(t, sseBody(
		`{"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":4}}}`,
	))
	client := &OpenAIClient{BaseURL: base}
	var got []string
	resp, err := client.ChatStream(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}}, collect(&got))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Content != "Hello" || resp.FinishReason != "stop" || resp.Message.Role != RoleAssistant {
		t.Errorf("response = %+v", resp)
	}
	if strings.Join(got, "|") != "Hel|lo" {
		t.Errorf("sink got %q", got)
	}
	if resp.UsageMissing || resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 2 || resp.Usage.CacheReadTokens != 4 {
		t.Errorf("usage = %+v (missing %v)", resp.Usage, resp.UsageMissing)
	}
	if (*seen)["stream"] != true {
		t.Errorf("request did not ask to stream: %v", *seen)
	}
	if so, ok := (*seen)["stream_options"].(map[string]any); !ok || so["include_usage"] != true {
		t.Errorf("request did not ask for usage on the final chunk: %v", (*seen)["stream_options"])
	}
	if resp.Message.Reasoning != nil {
		t.Errorf("a stream with no reasoning must yield no carrier, got %s", resp.Message.Reasoning)
	}
}

func TestChatStreamAssemblesToolCalls(t *testing.T) {
	base, _ := streamServer(t, sseBody(
		`{"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"fs_read","arguments":""}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"c2","type":"function","function":{"name":"fs_list","arguments":"{}"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"a.txt\"}"}}]},"finish_reason":null}]}`,
		`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
	))
	client := &OpenAIClient{BaseURL: base}
	var got []string
	resp, err := client.ChatStream(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}}, collect(&got))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Errorf("tool arguments must not reach the sink: %q", got)
	}
	want := []ToolCall{{ID: "c1", Name: "fs_read", Arguments: `{"path":"a.txt"}`}, {ID: "c2", Name: "fs_list", Arguments: "{}"}}
	if len(resp.Message.ToolCalls) != 2 || resp.Message.ToolCalls[0] != want[0] || resp.Message.ToolCalls[1] != want[1] {
		t.Errorf("tool calls = %+v, want %+v", resp.Message.ToolCalls, want)
	}
	if resp.FinishReason != "tool_calls" {
		t.Errorf("finish = %q", resp.FinishReason)
	}
}

// TestChatStreamReasoningCarriers: the carrier of a streamed turn is the same
// JSON value the provider sends whole - a string for reasoning_content and
// the bare reasoning, an array of index-merged items for reasoning_details -
// on the dialect's own key, so the echo path needs no streaming knowledge.
func TestChatStreamReasoningCarriers(t *testing.T) {
	t.Run("deepseek reasoning_content is one JSON string", func(t *testing.T) {
		base, _ := streamServer(t, sseBody(
			`{"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"first, ","content":null},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{"reasoning_content":"think <b>","content":null},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{"reasoning_content":null,"content":"answer"},"finish_reason":"stop"}]}`,
		))
		client := &OpenAIClient{BaseURL: base, Dialect: DialectDeepSeek}
		var got []string
		resp, err := client.ChatStream(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}}, collect(&got))
		if err != nil {
			t.Fatal(err)
		}
		// The concatenated text, re-encoded by encoding/json - which escapes
		// < and > exactly as it does in every request amele sends.
		var want string
		if err := json.Unmarshal(resp.Message.Reasoning, &want); err != nil || want != "first, think <b>" || resp.Message.ReasoningField != fieldReasoningContent {
			t.Errorf("carrier = %s on %q (decode: %v)", resp.Message.Reasoning, resp.Message.ReasoningField, err)
		}
		if !bytes.Contains(resp.Message.Reasoning, []byte(`\u003c`)) {
			t.Errorf("carrier is not encoding/json's own escaping: %s", resp.Message.Reasoning)
		}
		if strings.Join(got, "") != "answer" || resp.Message.Content != "answer" {
			t.Errorf("reasoning leaked into the text: sink %q, content %q", got, resp.Message.Content)
		}
	})
	t.Run("openrouter reasoning_details merge by index", func(t *testing.T) {
		base, _ := streamServer(t, sseBody(
			`{"choices":[{"index":0,"delta":{"role":"assistant","reasoning_details":[{"type":"reasoning.text","text":"I sho","index":0}]},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{"reasoning_details":[{"type":"reasoning.text","text":"uld read","index":0},{"type":"reasoning.encrypted","data":"abc","index":1}]},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{"reasoning_details":[{"type":"reasoning.text","text":" the log","signature":"sig","index":0}]},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{"content":"done"},"finish_reason":"stop"}]}`,
		))
		client := &OpenAIClient{BaseURL: base, Dialect: DialectOpenRouter}
		resp, err := client.ChatStream(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}}, func(string) {})
		if err != nil {
			t.Fatal(err)
		}
		want := `[{"type":"reasoning.text","text":"I should read the log","index":0,"signature":"sig"},{"type":"reasoning.encrypted","data":"abc","index":1}]`
		if string(resp.Message.Reasoning) != want || resp.Message.ReasoningField != fieldReasoningDetails {
			t.Errorf("carrier = %s on %q\nwant    %s", resp.Message.Reasoning, resp.Message.ReasoningField, want)
		}
	})
	t.Run("groq bare reasoning is captured for the log", func(t *testing.T) {
		base, _ := streamServer(t, sseBody(
			`{"choices":[{"index":0,"delta":{"role":"assistant","reasoning":"hm"},"finish_reason":null}]}`,
			`{"choices":[{"index":0,"delta":{"reasoning":"m","content":"ok"},"finish_reason":"stop"}]}`,
		))
		client := &OpenAIClient{BaseURL: base, Dialect: DialectGroq}
		resp, err := client.ChatStream(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}}, func(string) {})
		if err != nil {
			t.Fatal(err)
		}
		if string(resp.Message.Reasoning) != `"hmm"` {
			t.Errorf("carrier = %s", resp.Message.Reasoning)
		}
	})
}

func TestChatStreamFailures(t *testing.T) {
	t.Run("a mid-stream error chunk is a provider error", func(t *testing.T) {
		base, _ := streamServer(t, sseBody(
			`{"choices":[{"index":0,"delta":{"content":"par"},"finish_reason":null}]}`,
			`{"error":{"message":"upstream reset","code":502}}`,
		))
		client := &OpenAIClient{BaseURL: base, MaxAttempts: 1}
		_, err := client.ChatStream(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}}, func(string) {})
		if !errors.Is(err, ErrProvider) || !strings.Contains(err.Error(), "upstream reset") {
			t.Fatalf("err = %v", err)
		}
	})
	t.Run("a stream with no choice is an empty response", func(t *testing.T) {
		base, _ := streamServer(t, sseBody(`{"usage":{"prompt_tokens":1,"completion_tokens":0}}`))
		client := &OpenAIClient{BaseURL: base, MaxAttempts: 1}
		_, err := client.ChatStream(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}}, func(string) {})
		if !errors.Is(err, ErrProvider) || !strings.Contains(err.Error(), "no choices") {
			t.Fatalf("err = %v", err)
		}
	})
	t.Run("a malformed chunk is a provider error", func(t *testing.T) {
		base, _ := streamServer(t, "data: {not json\n\n")
		client := &OpenAIClient{BaseURL: base, MaxAttempts: 1}
		_, err := client.ChatStream(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}}, func(string) {})
		if !errors.Is(err, ErrProvider) || !strings.Contains(err.Error(), "decoding stream chunk") {
			t.Fatalf("err = %v", err)
		}
	})
}

// TestChatStreamFallsBackWhenStreamingIsRejected: an endpoint that answers
// `stream` with a 400 naming it is asked once more without it, and the
// whole text then reaches the sink at the end.
func TestChatStreamFallsBackWhenStreamingIsRejected(t *testing.T) {
	var streamed, plain int
	srv := chatServer(t, func(w http.ResponseWriter, req map[string]any) {
		if req["stream"] == true {
			streamed++
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"stream is not supported on this endpoint"}}`))
			return
		}
		plain++
		_, _ = w.Write([]byte(okBody("whole answer")))
	})
	client := &OpenAIClient{BaseURL: srv.URL + "/v1"}
	var got []string
	resp, err := client.ChatStream(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}}, collect(&got))
	if err != nil {
		t.Fatal(err)
	}
	if streamed != 1 || plain != 1 {
		t.Errorf("requests: streamed %d, plain %d; want one of each", streamed, plain)
	}
	if resp.Message.Content != "whole answer" || strings.Join(got, "") != "whole answer" {
		t.Errorf("content %q, sink %q", resp.Message.Content, got)
	}
}

// TestChatDoesNotStreamWithoutASink pins the non-streaming request bytes: no
// stream key, no stream_options, exactly as before ChatStream existed.
func TestChatDoesNotStreamWithoutASink(t *testing.T) {
	var seen map[string]any
	srv := chatServer(t, func(w http.ResponseWriter, req map[string]any) {
		seen = req
		_, _ = w.Write([]byte(okBody("hi")))
	})
	client := &OpenAIClient{BaseURL: srv.URL + "/v1"}
	if _, err := client.Chat(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}}); err != nil {
		t.Fatal(err)
	}
	if _, ok := seen["stream"]; ok {
		t.Errorf("Chat sent a stream key: %v", seen)
	}
	if _, ok := seen["stream_options"]; ok {
		t.Errorf("Chat sent stream_options: %v", seen)
	}
}
