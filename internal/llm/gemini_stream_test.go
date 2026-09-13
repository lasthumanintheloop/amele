package llm

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// gemSSE renders streamGenerateContent events.
func gemSSE(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		b.WriteString("data: " + e + "\r\n\r\n")
	}
	return b.String()
}

func gemStreamServer(t *testing.T, body string) (*httptest.Server, *http.Request) {
	t.Helper()
	var seen http.Request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = *r.Clone(context.Background())
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv, &seen
}

func TestGeminiChatStreamAssemblesEverything(t *testing.T) {
	srv, seen := gemStreamServer(t, gemSSE(
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Thinking about it","thought":true}]},"index":0}],"usageMetadata":{"promptTokenCount":5,"thoughtsTokenCount":3}}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":" more","thought":true,"thoughtSignature":"c2ln"}]},"index":0}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Let me "}]},"index":0}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"read it."},{"functionCall":{"id":"fc1","name":"fs_read","args":{"path":"app.log"}},"thoughtSignature":"Y2FsbA=="}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":4,"thoughtsTokenCount":6,"totalTokenCount":15}}`,
	))
	client := &GeminiClient{BaseURL: srv.URL, APIKey: "AIza"}
	var got []string
	resp, err := client.ChatStream(context.Background(), Request{Model: "gemini-3-flash", Messages: []Message{{Role: RoleUser, Content: "x"}}}, collect(&got))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(seen.URL.Path, ":streamGenerateContent") || seen.URL.Query().Get("alt") != "sse" {
		t.Errorf("request went to %s?%s, want streamGenerateContent with alt=sse", seen.URL.Path, seen.URL.RawQuery)
	}
	if strings.Join(got, "|") != "Let me |read it." {
		t.Errorf("sink got %q (thought parts must never reach it)", got)
	}
	if resp.Message.Content != "Let me read it." || resp.FinishReason != "stop" {
		t.Errorf("content %q, finish %q", resp.Message.Content, resp.FinishReason)
	}
	if len(resp.Message.ToolCalls) != 1 || resp.Message.ToolCalls[0] != (ToolCall{ID: "fc1", Name: "fs_read", Arguments: `{"path":"app.log"}`}) {
		t.Errorf("tool calls = %+v", resp.Message.ToolCalls)
	}
	want := `[{"text":"Thinking about it more","thought":true,"thoughtSignature":"c2ln"},{"text":"Let me read it."},{"functionCall":{"id":"fc1","name":"fs_read","args":{"path":"app.log"}},"thoughtSignature":"Y2FsbA=="}]`
	if string(resp.Message.Reasoning) != want {
		t.Errorf("carrier =\n%s\nwant\n%s", resp.Message.Reasoning, want)
	}
	// The LAST usageMetadata is the turn's: prompt 5, output 4+6 thoughts.
	if resp.UsageMissing || resp.Usage.InputTokens != 5 || resp.Usage.OutputTokens != 10 {
		t.Errorf("usage = %+v (missing %v)", resp.Usage, resp.UsageMissing)
	}
}

func TestGeminiChatStreamPlainAnswer(t *testing.T) {
	srv, _ := gemStreamServer(t, gemSSE(
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"hel"}]},"index":0}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"lo"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1}}`,
	))
	client := &GeminiClient{BaseURL: srv.URL, APIKey: "AIza"}
	resp, err := client.ChatStream(context.Background(), Request{Model: "g", Messages: []Message{{Role: RoleUser, Content: "x"}}}, func(string) {})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Content != "hello" || resp.FinishReason != "stop" || resp.Message.Reasoning != nil {
		t.Errorf("response = %+v", resp)
	}
}

func TestGeminiChatStreamFailures(t *testing.T) {
	cases := []struct {
		name string
		body string
		want string
	}{
		{"an empty stream", "", "no event"},
		{"a blocked prompt", gemSSE(`{"promptFeedback":{"blockReason":"SAFETY"}}`), "prompt blocked: SAFETY"},
		{"a truncated answer is a length finish", gemSSE(`{"candidates":[{"content":{"parts":[{"text":"cut"}]},"finishReason":"MAX_TOKENS"}]}`), ""},
		{"a malformed event", "data: {oops\n\n", "decoding stream event"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, _ := gemStreamServer(t, tc.body)
			client := &GeminiClient{BaseURL: srv.URL, APIKey: "AIza", MaxAttempts: 1}
			resp, err := client.ChatStream(context.Background(), Request{Model: "g", Messages: []Message{{Role: RoleUser, Content: "x"}}}, func(string) {})
			if tc.want == "" {
				if err != nil || resp.FinishReason != "length" {
					t.Fatalf("resp = %+v, err = %v; want a length finish", resp, err)
				}
				return
			}
			if !errors.Is(err, ErrProvider) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err = %v, want ErrProvider containing %q", err, tc.want)
			}
		})
	}
}
