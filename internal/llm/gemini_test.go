package llm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// geminiServer runs an httptest server for the generateContent wire format. It
// asserts the invariants every request must satisfy (the versioned model path
// and the JSON content type) and hands the decoded body to the test's handler.
func geminiServer(t *testing.T, handler func(w http.ResponseWriter, req map[string]any)) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Path, "/v1beta/models/") || !strings.HasSuffix(r.URL.Path, ":generateContent") {
			t.Errorf("unexpected path %q", r.URL.Path)
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

// gemOKBody is the smallest successful generateContent response: one candidate
// with one text part, stopped normally, with a usage report.
func gemOKBody(text string) string {
	return `{
		"candidates": [{"content": {"role": "model", "parts": [{"text": ` + jsonStr(text) + `}]}, "finishReason": "STOP"}],
		"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}
	}`
}

// gemUserRequest is the smallest request the client can send, used by the tests
// that care about transport rather than about message mapping.
func gemUserRequest() Request {
	return Request{Model: "gemini-3-pro", Messages: []Message{{Role: RoleUser, Content: "x"}}}
}

// TestGeminiChatBasics pins the transport contract: the model rides in the URL
// path (not the body), the key travels in x-goog-api-key, and a plain answer
// decodes into the neutral message with its usage and finish reason. The
// BaseURL carries a trailing slash to pin the trim.
func TestGeminiChatBasics(t *testing.T) {
	var gotKey, gotPath string
	srv := geminiServer(t, func(w http.ResponseWriter, req map[string]any) {
		if _, ok := req["model"]; ok {
			t.Errorf("model must not be a body field on this wire: %v", req["model"])
		}
		_, _ = w.Write([]byte(gemOKBody("hello")))
	})
	srv.Config.Handler = headerCapture(pathCapture(srv.Config.Handler, &gotPath), "x-goog-api-key", &gotKey)

	client := &GeminiClient{BaseURL: srv.URL + "/", APIKey: testAPIKey}
	resp, err := client.Chat(context.Background(), Request{
		Model: "gemini-3-pro",
		Messages: []Message{
			{Role: RoleSystem, Content: "sys"},
			{Role: RoleUser, Content: "hi"},
		},
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
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 {
		t.Errorf("usage: got %+v", resp.Usage)
	}
	if resp.FinishReason != "stop" {
		t.Errorf("finish reason: got %q", resp.FinishReason)
	}
	if gotKey != testAPIKey {
		t.Errorf("x-goog-api-key header: got %q", gotKey)
	}
	if gotPath != "/v1beta/models/gemini-3-pro:generateContent" {
		t.Errorf("path: got %q", gotPath)
	}
}

// pathCapture records the request path before delegating, so a test can pin the
// endpoint this wire builds from the model name.
func pathCapture(next http.Handler, into *string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*into = r.URL.Path
		next.ServeHTTP(w, r)
	})
}

// TestGeminiNoAPIKeyNoHeader: an empty key sends NO auth header at all. A
// gateway in front of the API may authenticate by other means, and an empty
// header value is a credential claim amele never made.
func TestGeminiNoAPIKeyNoHeader(t *testing.T) {
	var gotKey string
	srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = w.Write([]byte(gemOKBody("ok")))
	})
	srv.Config.Handler = headerCapture(srv.Config.Handler, "x-goog-api-key", &gotKey)

	client := &GeminiClient{BaseURL: srv.URL}
	if _, err := client.Chat(context.Background(), gemUserRequest()); err != nil {
		t.Fatal(err)
	}
	if gotKey != "" {
		t.Errorf("x-goog-api-key header: got %q, want no header", gotKey)
	}
}

// gemWireCase is one request rendered by the gemini client. The golden holds
// the exact bytes that would go on the wire.
type gemWireCase struct {
	name   string
	golden string
	req    Request
}

func gemWireCases() []gemWireCase {
	return []gemWireCase{
		{
			// The plain-text baseline: the system prompt hoists to
			// systemInstruction (this wire has no system role) and the user
			// turn becomes a user-role content with one text part.
			name:   "baseline",
			golden: "gemini-baseline.json",
			req:    Request{Model: "gemini-3-pro", Messages: baseMessages()},
		},
		{
			// A multi-turn conversation: RoleAssistant is spelled "model" here,
			// and a second system message appends instead of being dropped.
			name:   "conversation with a second system message",
			golden: "gemini-conversation.json",
			req: Request{
				Model: "gemini-3-pro",
				Messages: []Message{
					{Role: RoleSystem, Content: "you are a log sentry"},
					{Role: RoleUser, Content: "scan today's log"},
					{Role: RoleAssistant, Content: "nothing unusual"},
					{Role: RoleSystem, Content: "answer in English"},
					{Role: RoleUser, Content: "and yesterday's?"},
				},
			},
		},
		{
			// The cap and the sampling knobs land inside generationConfig, and
			// provider.params merge at the body ROOT beside it.
			name:   "sampling, cap and params",
			golden: "gemini-sampling-params.json",
			req: Request{
				Model:           "gemini-3-pro",
				Messages:        baseMessages(),
				MaxOutputTokens: 4096,
				Temperature:     ptr(0.0),
				TopP:            ptr(0.9),
				Extra:           map[string]json.RawMessage{"labels": json.RawMessage(`{"team":"ops"}`)},
			},
		},
		{
			// CONTRACT for the task boundary: tools, the reasoning knob and the
			// response format are NOT on this wire yet (Task 3 sanitizes tool
			// schemas, Task 4 maps thinkingConfig/responseJsonSchema). A
			// request carrying them must encode byte-identically to the
			// baseline, so an accidental early emission breaks this golden.
			name:   "unmapped knobs are not emitted yet",
			golden: "gemini-baseline.json",
			req: Request{
				Model:          "gemini-3-pro",
				Messages:       baseMessages(),
				Tools:          []ToolDef{{Name: "fs_read", Description: "read", Parameters: json.RawMessage(`{"type":"object"}`)}},
				Reasoning:      &ReasoningSpec{Effort: "high"},
				ResponseFormat: &ResponseFormat{Name: "amele_output", Schema: json.RawMessage(`{"type":"object"}`)},
			},
		},
	}
}

// TestGeminiToWireGolden is the request contract in bytes: exactly these keys
// with exactly these values leave the process.
func TestGeminiToWireGolden(t *testing.T) {
	for _, tc := range gemWireCases() {
		t.Run(tc.name, func(t *testing.T) {
			client := &GeminiClient{}
			wire, fields := client.toWire(tc.req)
			got, err := encodeBody(wire, fields)
			if err != nil {
				t.Fatalf("encodeBody: %v", err)
			}
			if !json.Valid(got) {
				t.Fatalf("body is not valid JSON: %s", got)
			}

			goldenPath := filepath.Join("testdata", "golden", "wire", tc.golden)
			if *update {
				if err := os.MkdirAll(filepath.Dir(goldenPath), 0o750); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(goldenPath, append(got, '\n'), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			raw, err := os.ReadFile(goldenPath) //nolint:gosec // G304: fixed testdata path.
			if err != nil {
				t.Fatalf("reading golden (run with -update to create): %v", err)
			}
			want := strings.TrimSuffix(string(raw), "\n")
			if string(got) != want {
				t.Errorf("request body differs from golden %s.\ngot:  %s\nwant: %s", tc.golden, got, want)
			}
		})
	}
}

// TestGeminiResponseTextParts: text parts concatenate in order, and a THOUGHT
// part is not part of the answer - it is a thinking summary, which would
// otherwise be handed to the user (and to output.schema) as content.
func TestGeminiResponseTextParts(t *testing.T) {
	srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = w.Write([]byte(`{
			"candidates": [{"content": {"role": "model", "parts": [
				{"text": "let me think", "thought": true},
				{"text": "I'll read "},
				{"text": "the file."}
			]}, "finishReason": "STOP"}],
			"usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2}
		}`))
	})

	client := &GeminiClient{BaseURL: srv.URL}
	resp, err := client.Chat(context.Background(), gemUserRequest())
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Content != "I'll read the file." {
		t.Errorf("content: got %q", resp.Message.Content)
	}
}

// TestGeminiEndpoint pins the URL built from base and model: the version
// segment is the client's to add, a "models/" prefix an operator copied from
// the docs is not doubled, and a model name cannot smuggle anything into the
// path.
func TestGeminiEndpoint(t *testing.T) {
	tests := []struct {
		name  string
		base  string
		model string
		want  string
	}{
		{
			"default host",
			"", "gemini-3-pro",
			"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro:generateContent",
		},
		{
			"trailing slash trimmed",
			"https://proxy.example.com/", "gemini-3-pro",
			"https://proxy.example.com/v1beta/models/gemini-3-pro:generateContent",
		},
		{
			"models/ prefix is not doubled",
			"https://proxy.example.com", "models/gemini-3-pro",
			"https://proxy.example.com/v1beta/models/gemini-3-pro:generateContent",
		},
		{
			// SECURITY: a model name names a model; it may not append a query
			// string or climb the path.
			"path injection is escaped",
			"https://proxy.example.com", "gemini-3-pro?key=leak",
			"https://proxy.example.com/v1beta/models/gemini-3-pro%3Fkey=leak:generateContent",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &GeminiClient{BaseURL: tt.base}
			if got := client.endpoint(tt.model); got != tt.want {
				t.Errorf("endpoint: got %q, want %q", got, tt.want)
			}
		})
	}
}

// TestGeminiPartsNotAnArray: a parts field this client cannot read must name
// the response instead of yielding a silently empty answer.
func TestGeminiPartsNotAnArray(t *testing.T) {
	srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = w.Write([]byte(`{"candidates": [{"content": {"role": "model", "parts": "hi"}, "finishReason": "STOP"}]}`))
	})

	client := &GeminiClient{BaseURL: srv.URL, Sleep: failingSleep(t)}
	_, err := client.Chat(context.Background(), gemUserRequest())
	if !errors.Is(err, ErrProvider) {
		t.Fatalf("err: got %v, want a provider error", err)
	}
	if !strings.Contains(err.Error(), "parts") {
		t.Errorf("error %q does not name the field it could not read", err)
	}
}

// TestGeminiNoCandidates: a 200 with no candidate is a failed turn, not an
// empty answer. The prompt-level block reason is named when the API sent one,
// because that is the only thing that explains the missing candidate.
func TestGeminiNoCandidates(t *testing.T) {
	srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = w.Write([]byte(`{"promptFeedback": {"blockReason": "SAFETY"}}`))
	})

	client := &GeminiClient{BaseURL: srv.URL}
	_, err := client.Chat(context.Background(), gemUserRequest())
	if !errors.Is(err, ErrProvider) {
		t.Fatalf("err: got %v, want a provider error", err)
	}
	if !strings.Contains(err.Error(), "SAFETY") {
		t.Errorf("error %q does not name the block reason", err)
	}
}

// TestGeminiFinishReasonMapping walks the documented vocabulary. The mapping is
// load-bearing: the loop hard-fails a truncated turn on "length" and would
// accept it as a finished answer under any other name.
func TestGeminiFinishReasonMapping(t *testing.T) {
	tests := []struct {
		reason  string
		want    string
		wantErr bool
	}{
		{"STOP", "stop", false},
		{"MAX_TOKENS", "length", false},
		{"SAFETY", "content_filter", false},
		{"PROHIBITED_CONTENT", "content_filter", false},
		{"BLOCKLIST", "content_filter", false},
		{"SPII", "content_filter", false},
		{"RECITATION", "recitation", false},
		{"LANGUAGE", "language", false},
		{"OTHER", "other", false},
		{"", "", false},
		{"MALFORMED_FUNCTION_CALL", "", true},
	}
	for _, tt := range tests {
		t.Run(tt.reason, func(t *testing.T) {
			got, err := mapGeminiFinishReason(tt.reason)
			if tt.wantErr {
				if !errors.Is(err, ErrProvider) {
					t.Fatalf("err: got %v, want a provider error", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

// TestGeminiMalformedFunctionCallIsAnError: the API answers 200 with this
// finish reason, so nothing below the client would notice the turn failed. It
// must surface as a provider error that names the situation.
func TestGeminiMalformedFunctionCallIsAnError(t *testing.T) {
	var calls int
	srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		_, _ = w.Write([]byte(`{"candidates": [{"content": {"role": "model", "parts": []},
			"finishReason": "MALFORMED_FUNCTION_CALL"}]}`))
	})

	client := &GeminiClient{BaseURL: srv.URL, Sleep: failingSleep(t)}
	_, err := client.Chat(context.Background(), gemUserRequest())
	if !errors.Is(err, ErrProvider) {
		t.Fatalf("err: got %v, want a provider error", err)
	}
	if calls != 1 {
		t.Errorf("calls: got %d, want 1 (a malformed call is not retryable)", calls)
	}
}

// failingSleep fails the test if the client ever backs off, for the paths that
// must not retry.
func failingSleep(t *testing.T) func(context.Context, time.Duration) error {
	t.Helper()
	return func(_ context.Context, d time.Duration) error {
		t.Errorf("unexpected backoff of %s", d)
		return nil
	}
}

// TestGeminiUsageArithmetic pins the billing truth: thinking tokens are billed
// as OUTPUT, so they must be added to candidatesTokenCount or the token budget
// undercounts every reasoning run. Negative and absurd counts are sanitized at
// this boundary like on every other wire.
func TestGeminiUsageArithmetic(t *testing.T) {
	tests := []struct {
		name        string
		usage       string
		wantIn      int
		wantOut     int
		wantMissing bool
	}{
		{"no thoughts", `{"promptTokenCount": 10, "candidatesTokenCount": 5}`, 10, 5, false},
		{"thoughts bill as output", `{"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 7}`, 10, 12, false},
		{"absent usage", `null`, 0, 0, true},
		{"negative thoughts", `{"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": -1}`, 10, 0, true},
		{"negative prompt", `{"promptTokenCount": -5, "candidatesTokenCount": 5}`, 0, 5, true},
		{
			"absurd counts saturate",
			`{"promptTokenCount": 9000000000000000000, "candidatesTokenCount": 9000000000000000000, "thoughtsTokenCount": 9000000000000000000}`,
			maxTokensPerResponse, maxTokensPerResponse, false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
				_, _ = w.Write([]byte(`{"candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]},
					"finishReason": "STOP"}], "usageMetadata": ` + tt.usage + `}`))
			})
			client := &GeminiClient{BaseURL: srv.URL}
			resp, err := client.Chat(context.Background(), gemUserRequest())
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

// gemErrorBody is the documented error envelope of this API.
func gemErrorBody(code int, status, message, details string) string {
	return `{"error": {"code": ` + strconv.Itoa(code) + `, "message": ` + jsonStr(message) +
		`, "status": ` + jsonStr(status) + `, "details": [` + details + `]}}`
}

// TestGeminiErrorEnvelope: a 400 keeps the API's own message and status in the
// typed statusError, is not retried, and reaches the caller as ErrProvider.
func TestGeminiErrorEnvelope(t *testing.T) {
	var calls int
	srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(gemErrorBody(400, "INVALID_ARGUMENT", "Unknown name \"thinkingLevel\"", "")))
	})

	client := &GeminiClient{BaseURL: srv.URL, Sleep: failingSleep(t)}
	_, err := client.Chat(context.Background(), gemUserRequest())
	if !errors.Is(err, ErrProvider) {
		t.Fatalf("err: got %v, want a provider error", err)
	}
	var se *statusError
	if !errors.As(err, &se) {
		t.Fatalf("err %v is not a statusError", err)
	}
	if se.code != http.StatusBadRequest {
		t.Errorf("code: got %d", se.code)
	}
	if !strings.Contains(se.snippet, "INVALID_ARGUMENT") || !strings.Contains(se.snippet, "thinkingLevel") {
		t.Errorf("snippet does not carry the API's message: %q", se.snippet)
	}
	if calls != 1 {
		t.Errorf("calls: got %d, want 1 (400 is not retryable)", calls)
	}
}

// TestParseGoogleRetryDelay covers the RetryInfo detail in both duration
// spellings plus the shapes that must yield "no wish" rather than a wrong wait.
func TestParseGoogleRetryDelay(t *testing.T) {
	tests := []struct {
		name    string
		details string
		want    time.Duration
	}{
		{"whole seconds", `{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3s"}`, 3 * time.Second},
		{"fractional seconds", `{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3.5s"}`, 3500 * time.Millisecond},
		{
			"retry info after another detail",
			`{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": []},` +
				`{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "12s"}`,
			12 * time.Second,
		},
		{"no retry info", `{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": []}`, 0},
		{"unparseable duration", `{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "soon"}`, 0},
		{"negative duration", `{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "-3s"}`, 0},
		{"no details", ``, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var env gemErrorEnvelope
			if err := json.Unmarshal([]byte(gemErrorBody(429, "RESOURCE_EXHAUSTED", "quota", tt.details)), &env); err != nil {
				t.Fatalf("decoding fixture: %v", err)
			}
			if got := parseGoogleRetryDelay(env.Error.Details); got != tt.want {
				t.Errorf("got %s, want %s", got, tt.want)
			}
		})
	}
}

// TestGeminiRetryDelayHonored: this API sends NO Retry-After header - the wait
// it wants lives in the 429 BODY - so the body-borne wish must reach the same
// backoff machinery every other wire feeds from the header.
func TestGeminiRetryDelayHonored(t *testing.T) {
	tests := []struct {
		name    string
		details string
		want    []time.Duration
	}{
		{
			"body wish stretches the ladder",
			`{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3.5s"}`,
			[]time.Duration{3500 * time.Millisecond},
		},
		{
			// CONTRACT: a wish is capped like any other, so a misbehaving
			// endpoint cannot stall the run past the budget it would then be
			// blamed on (maxRetryAfter).
			"absurd wish is capped",
			`{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "600s"}`,
			[]time.Duration{maxRetryAfter},
		},
		{
			// No wish: the configured exponential ladder stands untouched.
			"no retry info falls back to the ladder",
			``,
			[]time.Duration{200 * time.Millisecond},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls int
			srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
				calls++
				if calls == 1 {
					w.WriteHeader(http.StatusTooManyRequests)
					_, _ = w.Write([]byte(gemErrorBody(429, "RESOURCE_EXHAUSTED", "quota exceeded", tt.details)))
					return
				}
				_, _ = w.Write([]byte(gemOKBody("ok")))
			})

			var delays []time.Duration
			client := &GeminiClient{
				BaseURL:        srv.URL,
				InitialBackoff: 200 * time.Millisecond,
				Sleep:          recordDelays(&delays),
			}
			if _, err := client.Chat(context.Background(), gemUserRequest()); err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(delays, tt.want) {
				t.Errorf("delays: got %v want %v", delays, tt.want)
			}
		})
	}
}

// TestGeminiRetriesExhausted: a persistent 503 consumes the attempt budget and
// then surfaces as a provider error naming the last failure.
func TestGeminiRetriesExhausted(t *testing.T) {
	var calls int
	srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(gemErrorBody(503, "UNAVAILABLE", "model is overloaded", "")))
	})

	var delays []time.Duration
	client := &GeminiClient{BaseURL: srv.URL, MaxAttempts: 2, Sleep: recordDelays(&delays)}
	_, err := client.Chat(context.Background(), gemUserRequest())
	if !errors.Is(err, ErrProvider) {
		t.Fatalf("err: got %v, want a provider error", err)
	}
	if calls != 2 {
		t.Errorf("calls: got %d, want 2", calls)
	}
	if len(delays) != 1 {
		t.Errorf("delays: got %v, want one wait", delays)
	}
	if !strings.Contains(err.Error(), "overloaded") {
		t.Errorf("error %q does not carry the last failure", err)
	}
}

// TestGeminiOwnedWireFieldsCoversTheRequest is the anti-drift check for the
// params escape hatch: every key the request struct writes must be refused as a
// provider.params key, or an operator could overwrite it silently.
func TestGeminiOwnedWireFieldsCoversTheRequest(t *testing.T) {
	body, err := json.Marshal(gemRequest{
		Contents:          []gemContent{{Role: "user", Parts: []gemPart{{Text: "x"}}}},
		SystemInstruction: &gemContent{Parts: []gemPart{{Text: "s"}}},
		Tools:             []gemTool{{}},
		GenerationConfig:  &gemGenerationConfig{MaxOutputTokens: 1},
	})
	if err != nil {
		t.Fatalf("marshaling request: %v", err)
	}
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(body, &keys); err != nil {
		t.Fatalf("decoding request: %v", err)
	}
	owned := GeminiOwnedWireFields()
	for key := range keys {
		if !slices.Contains(owned, key) {
			t.Errorf("request key %q is not in GeminiOwnedWireFields()", key)
		}
	}
}

// TestGeminiRequestTimeoutNotRetried: a client-side timeout is the operator's
// own ceiling, so retrying it only multiplies the wasted wall-clock; the error
// must name the knob to turn.
func TestGeminiRequestTimeoutNotRetried(t *testing.T) {
	// atomic: the handler is still sleeping in its own goroutine when the test
	// (whose client already timed out) reads the counter.
	var calls atomic.Int32
	srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls.Add(1)
		time.Sleep(300 * time.Millisecond)
		_, _ = w.Write([]byte(gemOKBody("too late")))
	})

	client := &GeminiClient{BaseURL: srv.URL, RequestTimeout: 30 * time.Millisecond, Sleep: failingSleep(t)}
	_, err := client.Chat(context.Background(), gemUserRequest())
	if !errors.Is(err, ErrProvider) {
		t.Fatalf("err: got %v, want a provider error", err)
	}
	if !strings.Contains(err.Error(), "provider.request_timeout") {
		t.Errorf("error %q does not name the knob", err)
	}
	if calls.Load() != 1 {
		t.Errorf("calls: got %d, want 1", calls.Load())
	}
}

// TestGeminiChatResponseBodyIsBounded: a hostile endpoint must not be able to
// stream until the process dies; the shared cap applies on this wire too.
func TestGeminiChatResponseBodyIsBounded(t *testing.T) {
	srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates": [{"content": {"role": "model", "parts": [{"text": "`))
		chunk := strings.Repeat("a", 1<<20)
		for range 9 {
			_, _ = w.Write([]byte(chunk))
		}
	})

	client := &GeminiClient{BaseURL: srv.URL}
	_, err := client.Chat(context.Background(), gemUserRequest())
	if !errors.Is(err, ErrProvider) {
		t.Fatalf("oversized body: got %v, want a provider error", err)
	}
}
