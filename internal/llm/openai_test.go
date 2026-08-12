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

// noSleep makes retry tests instant while still counting backoff calls.
func noSleep(t *testing.T, count *int) func(context.Context, time.Duration) error {
	t.Helper()
	return func(ctx context.Context, _ time.Duration) error {
		*count++
		return ctx.Err()
	}
}

// chatServer runs an httptest server whose handler receives the decoded wire
// request and can script arbitrary responses.
func chatServer(t *testing.T, handler func(w http.ResponseWriter, req map[string]any)) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path %q", r.URL.Path)
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

func okBody(content string) string {
	return `{
		"choices": [{"message": {"role": "assistant", "content": ` + jsonStr(content) + `}, "finish_reason": "stop"}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5}
	}`
}

func jsonStr(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func TestChatBasics(t *testing.T) {
	var gotAuth string
	srv := chatServer(t, func(w http.ResponseWriter, req map[string]any) {
		if req["model"] != "test-model" {
			t.Errorf("model: got %v", req["model"])
		}
		msgs := req["messages"].([]any)
		if len(msgs) != 2 {
			t.Fatalf("messages: got %d", len(msgs))
		}
		tools := req["tools"].([]any)
		fn := tools[0].(map[string]any)["function"].(map[string]any)
		if fn["name"] != "fs_read" {
			t.Errorf("tool name: got %v", fn["name"])
		}
		_, _ = w.Write([]byte(okBody("hello")))
	})
	// Capturing the auth header inside chatServer's handler is awkward;
	// assert it in a dedicated middleware instead (see headerCapture).
	srv.Config.Handler = headerCapture(srv.Config.Handler, "Authorization", &gotAuth)

	client := &OpenAIClient{BaseURL: srv.URL + "/v1", APIKey: "sk-secret"}
	resp, err := client.Chat(context.Background(), Request{
		Model: "test-model",
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
	if resp.Usage.Total() != 15 {
		t.Errorf("usage total: got %d", resp.Usage.Total())
	}
	if gotAuth != "Bearer sk-secret" {
		t.Errorf("auth header: got %q", gotAuth)
	}
}

func TestChatToolCalls(t *testing.T) {
	srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = w.Write([]byte(`{
			"choices": [{"message": {"role": "assistant", "content": "",
				"tool_calls": [{"id": "c1", "type": "function",
					"function": {"name": "fs_read", "arguments": "{\"path\":\"a.txt\"}"}}]},
				"finish_reason": "tool_calls"}],
			"usage": {"prompt_tokens": 1, "completion_tokens": 2}
		}`))
	})

	client := &OpenAIClient{BaseURL: srv.URL + "/v1"}
	resp, err := client.Chat(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls: got %d", len(resp.Message.ToolCalls))
	}
	tc := resp.Message.ToolCalls[0]
	if tc.ID != "c1" || tc.Name != "fs_read" || tc.Arguments != `{"path":"a.txt"}` {
		t.Errorf("tool call: %+v", tc)
	}
}

func TestChatRetriesOn500Then429(t *testing.T) {
	var calls int
	srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		switch calls {
		case 1:
			w.WriteHeader(http.StatusInternalServerError)
		case 2:
			w.WriteHeader(http.StatusTooManyRequests)
		default:
			_, _ = w.Write([]byte(okBody("recovered")))
		}
	})

	var sleeps int
	client := &OpenAIClient{BaseURL: srv.URL + "/v1", Sleep: noSleep(t, &sleeps)}
	resp, err := client.Chat(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}}})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Content != "recovered" {
		t.Errorf("content: %q", resp.Message.Content)
	}
	if calls != 3 || sleeps != 2 {
		t.Errorf("calls=%d sleeps=%d, want 3 and 2", calls, sleeps)
	}
}

func TestChatDoesNotRetryOn400(t *testing.T) {
	var calls int
	srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error": {"message": "bad model"}}`))
	})

	client := &OpenAIClient{BaseURL: srv.URL + "/v1"}
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

func TestChatRetriesExhausted(t *testing.T) {
	var calls int
	srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		w.WriteHeader(http.StatusServiceUnavailable)
	})

	var sleeps int
	client := &OpenAIClient{BaseURL: srv.URL + "/v1", Sleep: noSleep(t, &sleeps)}
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

// TestChatRequestTimeoutNotRetried: a generation that exceeds the per-request
// budget will exceed it again - retrying multiplies the wasted wall-clock
// (3 × 120s before exit 5). The failure must be immediate and name the config
// knob (review finding P1-4).
func TestChatRequestTimeoutNotRetried(t *testing.T) {
	// atomic: the handler is still sleeping in its own goroutine when the
	// test (whose client already timed out) reads the counter.
	var calls atomic.Int32
	srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls.Add(1)
		time.Sleep(300 * time.Millisecond)
		_, _ = w.Write([]byte(okBody("too late")))
	})

	var sleeps int
	client := &OpenAIClient{BaseURL: srv.URL + "/v1", RequestTimeout: 30 * time.Millisecond, Sleep: noSleep(t, &sleeps)}
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

// TestChatRetryAfterHonored: a 429 carrying Retry-After (seconds form) must
// stretch the backoff to what the provider asked for - fixed 1s/2s waits
// inside a longer rate-limit window burn all attempts for nothing (P2-9).
func TestChatRetryAfterHonored(t *testing.T) {
	var calls int
	srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		if calls == 1 {
			w.Header().Set("Retry-After", "7")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_, _ = w.Write([]byte(okBody("ok")))
	})

	var delays []time.Duration
	client := &OpenAIClient{BaseURL: srv.URL + "/v1", Sleep: func(_ context.Context, d time.Duration) error {
		delays = append(delays, d)
		return nil
	}}
	if _, err := client.Chat(context.Background(), Request{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	if len(delays) != 1 || delays[0] != 7*time.Second {
		t.Errorf("delays: got %v want [7s]", delays)
	}
}

// TestChatRetryAfterCapped: an absurd Retry-After must not stall the run for
// hours - the wait is capped (the run timeout would fire anyway, but exit 3
// would then misattribute a provider problem to the user's budget).
func TestChatRetryAfterCapped(t *testing.T) {
	var calls int
	srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		if calls == 1 {
			w.Header().Set("Retry-After", "3600")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_, _ = w.Write([]byte(okBody("ok")))
	})

	var delays []time.Duration
	client := &OpenAIClient{BaseURL: srv.URL + "/v1", Sleep: func(_ context.Context, d time.Duration) error {
		delays = append(delays, d)
		return nil
	}}
	if _, err := client.Chat(context.Background(), Request{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	if len(delays) != 1 || delays[0] != maxRetryAfter {
		t.Errorf("delays: got %v want [%v]", delays, maxRetryAfter)
	}
}

func TestChatContextCancel(t *testing.T) {
	srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
		w.WriteHeader(http.StatusInternalServerError) // would retry...
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // ...but the context is already gone
	client := &OpenAIClient{BaseURL: srv.URL + "/v1"}
	_, err := client.Chat(ctx, Request{Model: "m"})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, ErrProvider) {
		t.Errorf("should wrap ErrProvider: %v", err)
	}
}

func TestChatMalformedResponses(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{"not json", "garbage"},
		{"no choices", `{"choices": [], "usage": {}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
				_, _ = w.Write([]byte(tt.body))
			})
			client := &OpenAIClient{BaseURL: srv.URL + "/v1"}
			_, err := client.Chat(context.Background(), Request{Model: "m"})
			if !errors.Is(err, ErrProvider) {
				t.Errorf("should wrap ErrProvider: %v", err)
			}
		})
	}
}

// TestChatMissingUsage: an endpoint omitting the usage object must be
// reported as UsageMissing so token budgets can fail closed (Codex review
// finding F9).
func TestChatMissingUsage(t *testing.T) {
	srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = w.Write([]byte(`{"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}`))
	})
	client := &OpenAIClient{BaseURL: srv.URL + "/v1"}
	resp, err := client.Chat(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if !resp.UsageMissing {
		t.Error("UsageMissing must be true when the provider omits usage")
	}
	// And the inverse: a present usage object clears the flag.
	srv2 := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = w.Write([]byte(okBody("hi")))
	})
	client2 := &OpenAIClient{BaseURL: srv2.URL + "/v1"}
	resp2, err := client2.Chat(context.Background(), Request{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if resp2.UsageMissing {
		t.Error("UsageMissing must be false when usage is reported")
	}
}

// TestChatSendsResponseFormat pins the exact provider-native wire shape. The
// assertion re-marshals the decoded value so key order is canonical (Go sorts
// map keys), making a literal string comparison meaningful.
func TestChatSendsResponseFormat(t *testing.T) {
	var got string
	srv := chatServer(t, func(w http.ResponseWriter, req map[string]any) {
		b, err := json.Marshal(req["response_format"])
		if err != nil {
			t.Errorf("marshaling response_format: %v", err)
		}
		got = string(b)
		_, _ = w.Write([]byte(okBody(`{"ok":true}`)))
	})

	client := &OpenAIClient{BaseURL: srv.URL + "/v1"}
	_, err := client.Chat(context.Background(), Request{
		Model:    "m",
		Messages: []Message{{Role: RoleUser, Content: "x"}},
		ResponseFormat: &ResponseFormat{
			Name:   "amele_output",
			Schema: json.RawMessage(`{"type": "object"}`),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := `{"json_schema":{"name":"amele_output","schema":{"type":"object"},"strict":true},"type":"json_schema"}`
	if got != want {
		t.Errorf("response_format wire shape:\n got %s\nwant %s", got, want)
	}
}

// TestChatOmitsResponseFormatWhenNil: plain-text runs must not send the field
// at all - providers without schema support reject unknown keys.
func TestChatOmitsResponseFormatWhenNil(t *testing.T) {
	var present bool
	srv := chatServer(t, func(w http.ResponseWriter, req map[string]any) {
		_, present = req["response_format"]
		_, _ = w.Write([]byte(okBody("hi")))
	})

	client := &OpenAIClient{BaseURL: srv.URL + "/v1"}
	if _, err := client.Chat(context.Background(), Request{Model: "m"}); err != nil {
		t.Fatal(err)
	}
	if present {
		t.Error("response_format must be omitted when Request.ResponseFormat is nil")
	}
}

// TestChatFallsBackWhenResponseFormatRejected: a provider that 400s on
// response_format gets exactly one extra round-trip without the field, inside
// the same Chat call, and that round-trip costs no retry budget and no
// backoff sleep (it is capability discovery, not a transient failure).
func TestChatFallsBackWhenResponseFormatRejected(t *testing.T) {
	var calls int
	var secondHadFormat bool
	srv := chatServer(t, func(w http.ResponseWriter, req map[string]any) {
		calls++
		if calls == 1 {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error": {"message": "response_format is not supported by this model"}}`))
			return
		}
		_, secondHadFormat = req["response_format"]
		_, _ = w.Write([]byte(okBody("plain")))
	})

	var sleeps int
	client := &OpenAIClient{BaseURL: srv.URL + "/v1", Sleep: noSleep(t, &sleeps)}
	resp, err := client.Chat(context.Background(), Request{
		Model:          "m",
		Messages:       []Message{{Role: RoleUser, Content: "x"}},
		ResponseFormat: &ResponseFormat{Name: "amele_output", Schema: json.RawMessage(`{"type":"object"}`)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Content != "plain" {
		t.Errorf("content: %q", resp.Message.Content)
	}
	if calls != 2 {
		t.Errorf("calls: got %d, want 2 (original + one fallback)", calls)
	}
	if secondHadFormat {
		t.Error("fallback request must not carry response_format")
	}
	if sleeps != 0 {
		t.Errorf("fallback must not sleep on backoff, got %d sleeps", sleeps)
	}
	// The caller asked for native enforcement and did not get it; the
	// response must say so, or the downgrade is silent.
	if !resp.SchemaEnforcementDropped {
		t.Error("fallback response must be flagged SchemaEnforcementDropped")
	}
}

// TestChatFallbackBodyReusedOnLaterRetries: once the fallback has been taken,
// every remaining retry must send the stripped body. The single-line swap
// (body, fallbackBody = fallbackBody, nil) is all that guarantees it, so this
// pins the property: a 429 after a successful downgrade must not resurrect the
// rejected response_format body - that would 400 forever and burn the whole
// retry budget on a request the provider already refused.
func TestChatFallbackBodyReusedOnLaterRetries(t *testing.T) {
	var hadFormat []bool
	srv := chatServer(t, func(w http.ResponseWriter, req map[string]any) {
		_, present := req["response_format"]
		hadFormat = append(hadFormat, present)
		switch len(hadFormat) {
		case 1:
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error": {"message": "response_format is not supported"}}`))
		case 2:
			w.WriteHeader(http.StatusTooManyRequests)
		default:
			_, _ = w.Write([]byte(okBody("recovered")))
		}
	})

	var sleeps int
	client := &OpenAIClient{BaseURL: srv.URL + "/v1", Sleep: noSleep(t, &sleeps)}
	resp, err := client.Chat(context.Background(), Request{
		Model:          "m",
		Messages:       []Message{{Role: RoleUser, Content: "x"}},
		ResponseFormat: &ResponseFormat{Name: "amele_output", Schema: json.RawMessage(`{"type":"object"}`)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Content != "recovered" {
		t.Errorf("content: %q", resp.Message.Content)
	}
	want := []bool{true, false, false}
	if !slices.Equal(hadFormat, want) {
		t.Errorf("response_format presence per request: got %v, want %v", hadFormat, want)
	}
	// One sleep only: the 429 retry backs off, the fallback does not.
	if sleeps != 1 {
		t.Errorf("sleeps: got %d, want 1 (429 backoff only)", sleeps)
	}
	// The success arrived on a retry AFTER the fallback fired; it was still
	// produced without native enforcement and must carry the flag.
	if !resp.SchemaEnforcementDropped {
		t.Error("post-fallback retry response must be flagged SchemaEnforcementDropped")
	}
}

// TestChatSchemaEnforcementNotDropped: a provider that ACCEPTS
// response_format produced a natively-constrained answer, so the flag stays
// false - and so does a request that asked for no schema at all (there was
// nothing to drop).
func TestChatSchemaEnforcementNotDropped(t *testing.T) {
	srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = w.Write([]byte(okBody("fine")))
	})
	client := &OpenAIClient{BaseURL: srv.URL + "/v1"}

	withSchema, err := client.Chat(context.Background(), Request{
		Model:          "m",
		Messages:       []Message{{Role: RoleUser, Content: "x"}},
		ResponseFormat: &ResponseFormat{Name: "amele_output", Schema: json.RawMessage(`{"type":"object"}`)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if withSchema.SchemaEnforcementDropped {
		t.Error("accepted response_format must not set SchemaEnforcementDropped")
	}

	plain, err := client.Chat(context.Background(), Request{
		Model:    "m",
		Messages: []Message{{Role: RoleUser, Content: "x"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if plain.SchemaEnforcementDropped {
		t.Error("schema-less request must not set SchemaEnforcementDropped")
	}
}

// TestChatFallbackOnlyOnce: if the provider keeps 400ing, the caller gets the
// error - no loop, no retry-budget consumption.
func TestChatFallbackOnlyOnce(t *testing.T) {
	var calls int
	srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error": "response_format rejected"}`))
	})

	client := &OpenAIClient{BaseURL: srv.URL + "/v1"}
	_, err := client.Chat(context.Background(), Request{
		Model:          "m",
		ResponseFormat: &ResponseFormat{Name: "amele_output", Schema: json.RawMessage(`{"type":"object"}`)},
	})
	if !errors.Is(err, ErrProvider) {
		t.Fatalf("want ErrProvider, got %v", err)
	}
	if calls != 2 {
		t.Errorf("calls: got %d, want 2 (original + one fallback)", calls)
	}
}

// TestChatNoFallbackOnUnrelated400: an unrelated 400 (bad model, bad key) must
// stay a hard failure - retrying it without the schema would silently degrade
// output enforcement for a problem the fallback cannot fix.
func TestChatNoFallbackOnUnrelated400(t *testing.T) {
	var calls int
	srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error": {"message": "bad model"}}`))
	})

	client := &OpenAIClient{BaseURL: srv.URL + "/v1"}
	_, err := client.Chat(context.Background(), Request{
		Model:          "m",
		ResponseFormat: &ResponseFormat{Name: "amele_output", Schema: json.RawMessage(`{"type":"object"}`)},
	})
	if !errors.Is(err, ErrProvider) {
		t.Fatalf("want ErrProvider, got %v", err)
	}
	if !strings.Contains(err.Error(), "bad model") {
		t.Errorf("error should carry body snippet: %v", err)
	}
	if calls != 1 {
		t.Errorf("unrelated 400 must not trigger the fallback, got %d calls", calls)
	}
}

// TestFakeRecordsResponseFormat: later layers assert on the schema the loop
// asked for, so the fake's verbatim recording must include it.
func TestFakeRecordsResponseFormat(t *testing.T) {
	fake := &Fake{Responses: []Response{TextResponse("{}", Usage{})}}
	rf := &ResponseFormat{Name: "amele_output", Schema: json.RawMessage(`{"type":"object"}`)}
	if _, err := fake.Chat(context.Background(), Request{Model: "m", ResponseFormat: rf}); err != nil {
		t.Fatal(err)
	}
	if len(fake.Requests) != 1 || fake.Requests[0].ResponseFormat != rf {
		t.Errorf("fake must record ResponseFormat verbatim: %+v", fake.Requests)
	}
}

func TestFakeProvider(t *testing.T) {
	fake := &Fake{
		Responses: []Response{
			ToolCallResponse("c1", "t", "{}", Usage{InputTokens: 1, OutputTokens: 1}),
			TextResponse("done", Usage{InputTokens: 1, OutputTokens: 1}),
		},
	}
	ctx := context.Background()

	r1, err := fake.Chat(ctx, Request{Model: "m"})
	if err != nil || len(r1.Message.ToolCalls) != 1 {
		t.Fatalf("first call: %+v, %v", r1, err)
	}
	r2, err := fake.Chat(ctx, Request{Model: "m"})
	if err != nil || r2.Message.Content != "done" {
		t.Fatalf("second call: %+v, %v", r2, err)
	}
	// Script exhaustion must fail loudly with the provider sentinel.
	if _, err := fake.Chat(ctx, Request{Model: "m"}); !errors.Is(err, ErrProvider) {
		t.Errorf("exhausted script: %v", err)
	}
	if len(fake.Requests) != 3 {
		t.Errorf("recorded requests: %d", len(fake.Requests))
	}
}

// TestChatUsageIsSanitized is a regression test for a budget bypass: the
// client copied the provider's token counts through verbatim, so a hostile or
// broken endpoint could report negative counts (shrinking the loop's running
// total, buying unlimited extra turns) or absurd ones (overflowing the
// accumulator into a negative number, same effect).
func TestChatUsageIsSanitized(t *testing.T) {
	tests := []struct {
		name        string
		usage       string
		wantIn      int
		wantOut     int
		wantMissing bool
	}{
		{"sane", `{"prompt_tokens": 10, "completion_tokens": 5}`, 10, 5, false},
		{"negative", `{"prompt_tokens": -5, "completion_tokens": -1}`, 0, 0, true},
		{"one negative", `{"prompt_tokens": 10, "completion_tokens": -1}`, 10, 0, true},
		{
			"absurd",
			`{"prompt_tokens": 9000000000000000000, "completion_tokens": 9000000000000000000}`,
			maxTokensPerResponse, maxTokensPerResponse, false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
				_, _ = w.Write([]byte(`{"choices": [{"message": {"role": "assistant", "content": "hi"},
					"finish_reason": "stop"}], "usage": ` + tt.usage + `}`))
			})
			client := &OpenAIClient{BaseURL: srv.URL + "/v1"}
			resp, err := client.Chat(context.Background(), Request{
				Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}},
			})
			if err != nil {
				t.Fatalf("Chat: %v", err)
			}
			if resp.Usage.InputTokens != tt.wantIn || resp.Usage.OutputTokens != tt.wantOut {
				t.Errorf("usage: got %+v, want {%d %d}", resp.Usage, tt.wantIn, tt.wantOut)
			}
			if resp.Usage.Total() < 0 {
				t.Errorf("total wrapped negative: %d", resp.Usage.Total())
			}
			// CONTRACT: an impossible report must fail closed, not read as
			// "this turn was free".
			if resp.UsageMissing != tt.wantMissing {
				t.Errorf("UsageMissing: got %v, want %v", resp.UsageMissing, tt.wantMissing)
			}
		})
	}
}

// TestChatResponseBodyIsBounded: the SUCCESS body used to be decoded straight
// off the socket, so a hostile or broken endpoint could stream forever and
// grow the process until the OOM killer ended the run. The cap must turn that
// into a plain provider error.
func TestChatResponseBodyIsBounded(t *testing.T) {
	srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
		// A valid prefix followed by an unterminated string: the decoder can
		// never finish, so only the read cap can stop this.
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"`))
		chunk := strings.Repeat("a", 64*1024)
		for written := 0; written < maxResponseBody+len(chunk); written += len(chunk) {
			if _, err := w.Write([]byte(chunk)); err != nil {
				return // the client hung up at the cap, which is the point
			}
		}
	})
	client := &OpenAIClient{BaseURL: srv.URL + "/v1"}
	_, err := client.Chat(context.Background(), Request{
		Model: "m", Messages: []Message{{Role: RoleUser, Content: "x"}},
	})
	if !errors.Is(err, ErrProvider) {
		t.Fatalf("oversized body: got %v, want a provider error", err)
	}
}

// TestUsageAddSaturates pins the accumulator helper: adding two huge usages
// must saturate instead of wrapping into a negative total that would defeat
// the max_tokens budget.
func TestUsageAddSaturates(t *testing.T) {
	huge := Usage{InputTokens: maxInt - 1, OutputTokens: maxInt - 1}
	got := huge.Add(Usage{InputTokens: 100, OutputTokens: 100})
	if got.InputTokens != maxInt || got.OutputTokens != maxInt {
		t.Errorf("Add did not saturate: %+v", got)
	}
	if got.Total() != maxInt {
		t.Errorf("Total did not saturate: %d", got.Total())
	}
}
