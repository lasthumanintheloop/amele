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
			// A message that would STILL produce no parts is dropped: an empty
			// text part encodes to {}, a shape protobuf-JSON rejects, and a
			// turn with no parts is a 400. An assistant turn with neither text
			// nor tool calls is the only shape left in that branch now that
			// tool calls fill parts of their own.
			name:   "a message with no parts is dropped",
			golden: "gemini-baseline.json",
			req: Request{
				Model: "gemini-3-pro",
				Messages: append(baseMessages(),
					Message{Role: RoleAssistant},
					Message{Role: RoleUser},
				),
			},
		},
		{
			// The tool loop in one request: the assistant turn becomes text
			// plus one functionCall part per call, and both results ride back
			// in a SINGLE user turn of functionResponse parts - a plain output
			// wrapped under "output", a JSON object passed through.
			name:   "a tool call and its results",
			golden: "gemini-tool-loop.json",
			req: Request{
				Model: "gemini-3-pro",
				Messages: []Message{
					{Role: RoleSystem, Content: "you are a log sentry"},
					{Role: RoleUser, Content: "scan today's log"},
					{Role: RoleAssistant, Content: "reading it", ToolCalls: []ToolCall{
						{ID: "call_a", Name: "fs_read", Arguments: `{"path":"app.log"}`},
						{ID: "call_b", Name: "fs_list", Arguments: ""},
					}},
					{Role: RoleTool, ToolCallID: "call_a", Content: "ERROR disk full"},
					{Role: RoleTool, ToolCallID: "call_b", Content: `{"entries":["app.log"]}`},
				},
			},
		},
		{
			// Tool declarations share one tools entry, and every schema passes
			// the sanitizer first: additionalProperties and $schema are hard
			// 400s on this wire.
			name:   "tools are declared through the sanitizer",
			golden: "gemini-tools.json",
			req: Request{
				Model:    "gemini-3-pro",
				Messages: baseMessages(),
				Tools: []ToolDef{
					{Name: "fs_read", Description: "read a file", Parameters: json.RawMessage(
						`{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object",` +
							`"properties":{"path":{"type":"string","description":"Relative file path"}},` +
							`"required":["path"],"additionalProperties":false}`)},
					{Name: "ping", Description: "no arguments"},
				},
			},
		},
		{
			// The reasoning knob: an effort this wire HAS travels as a
			// thinkingLevel of the same word, inside generationConfig.
			name:   "reasoning effort becomes a thinking level",
			golden: "gemini-thinking-high.json",
			req: Request{
				Model:     "gemini-3-pro",
				Messages:  baseMessages(),
				Reasoning: &ReasoningSpec{Effort: "high"},
			},
		},
		{
			// xhigh has no counterpart here - high is the deepest level this
			// wire knows - so the request is byte-identical to the high case
			// and the rounding is reported in the notes instead.
			name:   "an effort above high rounds to high",
			golden: "gemini-thinking-high.json",
			req: Request{
				Model:     "gemini-3-pro",
				Messages:  baseMessages(),
				Reasoning: &ReasoningSpec{Effort: "xhigh"},
			},
		},
		{
			// A budget passes through as thinkingBudget and sends NO level:
			// the two fields together are a 400 on this wire.
			name:   "reasoning budget becomes a thinking budget",
			golden: "gemini-thinking-budget.json",
			req: Request{
				Model:     "gemini-3-pro",
				Messages:  baseMessages(),
				Reasoning: &ReasoningSpec{BudgetTokens: 2048},
			},
		},
		{
			// effort: none is the 2.5-era "thinking off" instruction, which
			// this wire spells as a budget of zero. The pointer is what keeps
			// the key on the wire: omitempty would drop the whole instruction.
			name:   "reasoning effort none sends a zero budget",
			golden: "gemini-thinking-off.json",
			req: Request{
				Model:     "gemini-3-pro",
				Messages:  baseMessages(),
				Reasoning: &ReasoningSpec{Effort: "none"},
			},
		},
		{
			// Structured output is a PAIR: the schema verbatim plus the JSON
			// mime type. The schema is not sanitized (response schemas tolerate
			// the keywords FunctionDeclaration.parameters rejects) and the
			// format's Name has no field on this wire.
			name:   "output schema and its mime type",
			golden: "gemini-schema.json",
			req: Request{
				Model:    "gemini-3-pro",
				Messages: baseMessages(),
				ResponseFormat: &ResponseFormat{Name: "amele_output", Schema: json.RawMessage(
					`{"type":"object","properties":{"ok":{"type":"boolean"}},"additionalProperties":false}`)},
			},
		},
		{
			// A format with no schema is plain JSON mode: the mime type alone,
			// never an empty responseJsonSchema (a shape this API rejects).
			// The validate+retry layer above is then what enforces the schema.
			name:   "a schemaless format is plain json mode",
			golden: "gemini-json-mode.json",
			req: Request{
				Model:          "gemini-3-pro",
				Messages:       baseMessages(),
				ResponseFormat: &ResponseFormat{Name: "amele_output"},
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
		{
			// SECURITY: the separators are what makes traversal work, and they
			// are encoded - so a normalizing proxy has no ".." SEGMENT to
			// resolve and the request stays under /v1beta/models/.
			"traversal cannot escape the models path",
			"https://proxy.example.com", "../../../foo",
			"https://proxy.example.com/v1beta/models/..%2F..%2F..%2Ffoo:generateContent",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &GeminiClient{BaseURL: tt.base}
			got, err := client.endpoint(tt.model)
			if err != nil {
				t.Fatalf("endpoint: %v", err)
			}
			if got != tt.want {
				t.Errorf("endpoint: got %q, want %q", got, tt.want)
			}
		})
	}
}

// TestGeminiCandidateWithoutParts: a candidate whose content carries NO parts
// key at all is an empty turn, not a decode failure - MAX_TOKENS and a blocked
// candidate really do produce one. The client must hand the loop an empty
// assistant message with the mapped finish reason and let the loop decide.
func TestGeminiCandidateWithoutParts(t *testing.T) {
	srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = w.Write([]byte(`{"candidates": [{"content": {"role": "model"}, "finishReason": "MAX_TOKENS"}],
			"usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 0}}`))
	})

	client := &GeminiClient{BaseURL: srv.URL, Sleep: failingSleep(t)}
	resp, err := client.Chat(context.Background(), gemUserRequest())
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp.Message.Content != "" {
		t.Errorf("content: got %q, want empty", resp.Message.Content)
	}
	if resp.Message.Role != RoleAssistant {
		t.Errorf("role: got %q", resp.Message.Role)
	}
	if resp.FinishReason != "length" {
		t.Errorf("finish reason: got %q, want length", resp.FinishReason)
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
//
// CONTRACT: there is no passthrough. A reason this table does not name is a
// provider error naming it, because badFinish ACCEPTS an unknown reason
// whenever the turn carried content - which is how a RECITATION turn with a
// preamble used to exit 0 in an unattended run.
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
		{"RECITATION", "content_filter", false},
		{"LANGUAGE", "content_filter", false},
		{"OTHER", "", true},
		{"FINISH_REASON_UNSPECIFIED", "", true},
		// An absent finishReason is not an unknown one: the field simply was
		// not sent, and the loop's own "" branch already refuses an empty
		// answer under it.
		{"", "", false},
		{"MALFORMED_FUNCTION_CALL", "", true},
		{"MISSING_THOUGHT_SIGNATURE", "", true},
		// The future reason nobody has written a case for yet.
		{"MODEL_ARMOR", "", true},
	}
	for _, tt := range tests {
		t.Run(tt.reason, func(t *testing.T) {
			got, err := mapGeminiFinishReason(tt.reason)
			if tt.wantErr {
				if !errors.Is(err, ErrProvider) {
					t.Fatalf("err: got %v, want a provider error", err)
				}
				if !strings.Contains(err.Error(), tt.reason) {
					t.Errorf("error %q does not name the finish reason %q", err, tt.reason)
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
			// CONTRACT: the tool-use prompt is INPUT that promptTokenCount does
			// not include - the API reports the tokens the tool declarations
			// cost in a counter of their own. A run with a large MCP toolset
			// pays for it every turn, so leaving it out understated the input
			// half of every budgeted tool run.
			"tool-use prompt tokens are input",
			`{"promptTokenCount": 10, "toolUsePromptTokenCount": 4, "candidatesTokenCount": 5}`,
			14, 5, false,
		},
		{
			"negative tool-use prompt",
			`{"promptTokenCount": 10, "toolUsePromptTokenCount": -1, "candidatesTokenCount": 5}`,
			0, 5, true,
		},
		{
			"absurd counts saturate",
			`{"promptTokenCount": 9000000000000000000, "toolUsePromptTokenCount": 9000000000000000000,
			  "candidatesTokenCount": 9000000000000000000, "thoughtsTokenCount": 9000000000000000000}`,
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
		{
			// A detail that is not an object at all: the entry is skipped
			// rather than failing the whole read, so a later well-formed
			// RetryInfo would still be honored.
			"detail is not an object", `"oops"`, 0,
		},
		{"negative duration", `{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "-3s"}`, 0},
		{
			// CONTRACT: an unusable detail is skipped, not an answer about the
			// ones behind it. Returning 0 here discarded a well-formed wish.
			"a good retry info behind a bad one",
			`{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "soon"},` +
				`{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "5s"}`,
			5 * time.Second,
		},
		{
			"a good retry info behind a non-positive one",
			`{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "0s"},` +
				`{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "2s"}`,
			2 * time.Second,
		},
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

// TestGeminiRefusesRedirects is the credential-leak guard on this wire.
//
// SECURITY: Go's redirect follower drops the Authorization header across hosts
// but preserves every CUSTOM header, so an x-goog-api-key would follow a 302 to
// wherever it pointed. generateContent never redirects, so nothing legitimate
// is lost by refusing: the 3xx surfaces as the status failure it is, and the
// key stays on the host the config named. (The MCP transport refuses redirects
// for the same reason.)
func TestGeminiRefusesRedirects(t *testing.T) {
	var elsewhereHits int
	elsewhere := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		elsewhereHits++
		if r.Header.Get("x-goog-api-key") != "" {
			t.Errorf("the api key followed a redirect to another host")
		}
		_, _ = w.Write([]byte(gemOKBody("stolen")))
	}))
	t.Cleanup(elsewhere.Close)

	// A fixed destination, not one derived from the request: the test is about
	// what the CLIENT does with a 302, and echoing the request path back into
	// the Location header is the open-redirect shape gosec (rightly) flags.
	target := elsewhere.URL + "/v1beta/models/gemini-3-pro:generateContent"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target, http.StatusFound)
	}))
	t.Cleanup(srv.Close)

	client := &GeminiClient{BaseURL: srv.URL, APIKey: "k", Sleep: failingSleep(t)}
	_, err := client.Chat(context.Background(), gemUserRequest())
	if !errors.Is(err, ErrProvider) {
		t.Fatalf("err: got %v, want a provider error", err)
	}
	var se *statusError
	if !errors.As(err, &se) {
		t.Fatalf("err %v is not a statusError", err)
	}
	if se.code != http.StatusFound {
		t.Errorf("code: got %d, want 302 (the redirect itself is the answer)", se.code)
	}
	if elsewhereHits != 0 {
		t.Errorf("the redirect was followed %d times", elsewhereHits)
	}
}

// TestGeminiRetryInfoBeyondTheDisplaySnippet: the retry wish lives in the same
// body as the error text, but the two are read with DIFFERENT bounds. A 429
// from this API carries its quota violations first and the RetryInfo detail
// last, so a body cut at the 2 KiB display snippet no longer parsed as JSON and
// the endpoint's own wish was lost - silently, since the exponential ladder
// then stood in for it.
func TestGeminiRetryInfoBeyondTheDisplaySnippet(t *testing.T) {
	// Comfortably past maxErrorBody, and shaped like the real thing: the quota
	// violations come first.
	violations := strings.Repeat(`{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
		"violations": [{"subject": "`+strings.Repeat("q", 200)+`", "description": "quota exceeded"}]},`, 20)
	details := violations + `{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}`

	var calls int
	srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		if calls == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			body := gemErrorBody(429, "RESOURCE_EXHAUSTED", "quota exceeded", details)
			if len(body) <= maxErrorBody {
				t.Fatalf("fixture is only %d bytes: it must exceed the %d-byte display snippet", len(body), maxErrorBody)
			}
			_, _ = w.Write([]byte(body))
			return
		}
		_, _ = w.Write([]byte(gemOKBody("ok")))
	})

	var delays []time.Duration
	client := &GeminiClient{BaseURL: srv.URL, InitialBackoff: 200 * time.Millisecond, Sleep: recordDelays(&delays)}
	resp, err := client.Chat(context.Background(), gemUserRequest())
	if err != nil {
		t.Fatal(err)
	}
	if resp.Message.Content != "ok" {
		t.Errorf("content: got %q", resp.Message.Content)
	}
	if !slices.Equal(delays, []time.Duration{7 * time.Second}) {
		t.Errorf("delays: got %v, want [7s] (the RetryInfo past the snippet)", delays)
	}
}

// TestGeminiRetryInfoAfterAnUnreadableDetail: one detail that cannot be read is
// not an answer about the others. The scan used to return 0 - "no wish" - the
// moment a RetryInfo carried an unparseable duration, so a second, well-formed
// one behind it was never reached.
func TestGeminiRetryInfoAfterAnUnreadableDetail(t *testing.T) {
	details := `{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "soon"},` +
		`{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "5s"}`

	var calls int
	srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		if calls == 1 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(gemErrorBody(429, "RESOURCE_EXHAUSTED", "quota exceeded", details)))
			return
		}
		_, _ = w.Write([]byte(gemOKBody("ok")))
	})

	var delays []time.Duration
	client := &GeminiClient{BaseURL: srv.URL, InitialBackoff: 200 * time.Millisecond, Sleep: recordDelays(&delays)}
	if _, err := client.Chat(context.Background(), gemUserRequest()); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(delays, []time.Duration{5 * time.Second}) {
		t.Errorf("delays: got %v, want [5s] (the second detail is still an answer)", delays)
	}
}

// TestGeminiErrorSnippetStaysBounded: reading further for the RetryInfo must
// not widen what a misbehaving endpoint can print into the operator's logs.
func TestGeminiErrorSnippetStaysBounded(t *testing.T) {
	srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(strings.Repeat("x", 4*maxErrorBody)))
	})

	client := &GeminiClient{BaseURL: srv.URL, Sleep: failingSleep(t)}
	_, err := client.Chat(context.Background(), gemUserRequest())
	var se *statusError
	if !errors.As(err, &se) {
		t.Fatalf("err %v is not a statusError", err)
	}
	if len(se.snippet) > maxErrorBody {
		t.Errorf("snippet is %d bytes, want at most %d", len(se.snippet), maxErrorBody)
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
