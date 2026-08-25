package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// gemScriptedServer answers each request with the next scripted body and
// records the exact bytes it received, which is what a byte-exactness claim
// has to be checked against.
func gemScriptedServer(t *testing.T, replies ...string) (*httptest.Server, *[][]byte) {
	t.Helper()
	var recorded [][]byte
	var n int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("reading request body: %v", err)
		}
		recorded = append(recorded, body)
		if n >= len(replies) {
			t.Errorf("unscripted request %d", n+1)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte(replies[n]))
		n++
	}))
	t.Cleanup(srv.Close)
	return srv, &recorded
}

// gemSentContent is one contents entry as it left the process. Parts stays raw
// so the test compares BYTES, not a decoded structure.
type gemSentContent struct {
	Role  string          `json:"role"`
	Parts json.RawMessage `json:"parts"`
}

func gemContentsOf(t *testing.T, body []byte) []gemSentContent {
	t.Helper()
	var parsed struct {
		Contents []gemSentContent `json:"contents"`
		Tools    json.RawMessage  `json:"tools"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("decoding recorded request: %v\nbody: %s", err, body)
	}
	return parsed.Contents
}

// gemSignedParts is a Gemini 3 candidate's parts array: a signed thinking part,
// a signed functionCall part and the answer text.
//
// It carries the two payloads a re-encoding round trip would damage: non-ASCII
// text and a backslash escape (\" and \/) inside the signature. If anything
// between the decoder and the next request re-serialized this array, these
// bytes would come back different.
const gemSignedParts = `[{"text":"günlük bakıyorum","thought":true,"thoughtSignature":"Cq4B\/sig\"one\""},` +
	`{"functionCall":{"id":"call_1","name":"fs_read","args":{"path":"app \"prod\".log"}},"thoughtSignature":"Ep8CtwoB"},` +
	`{"text":"okuyorum"}]`

func gemCandidateBody(parts, finish string) string {
	return `{"candidates":[{"content":{"role":"model","parts":` + parts + `},"finishReason":"` + finish + `"}],
		"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}}`
}

// TestGeminiThoughtSignatureRoundTrip is the echo contract end to end.
//
// CONTRACT: Gemini 3 requires the thought signature back on the first
// functionCall part of every step and rejects a modified or reordered parts
// array with a 400. The client stores the WHOLE raw array and re-emits those
// exact bytes as the model turn, so signature position and order are preserved
// by construction - no code here has to remember the rule.
func TestGeminiThoughtSignatureRoundTrip(t *testing.T) {
	srv, recorded := gemScriptedServer(t,
		gemCandidateBody(gemSignedParts, "STOP"),
		gemCandidateBody(`[{"text":"done"}]`, "STOP"),
	)
	client := &GeminiClient{BaseURL: srv.URL}

	first, err := client.Chat(context.Background(), Request{
		Model:    "gemini-3-pro",
		Messages: []Message{{Role: RoleUser, Content: "scan app.log"}},
	})
	if err != nil {
		t.Fatalf("first Chat: %v", err)
	}
	if got := string(first.Message.Reasoning); got != gemSignedParts {
		t.Fatalf("captured carrier:\ngot:  %s\nwant: %s", got, gemSignedParts)
	}
	// The typed decode still drives the loop: the thought part is not answer
	// text, and the functionCall part is a tool call.
	if first.Message.Content != "okuyorum" {
		t.Errorf("content: got %q", first.Message.Content)
	}
	if len(first.Message.ToolCalls) != 1 {
		t.Fatalf("tool calls: got %d, want 1", len(first.Message.ToolCalls))
	}
	call := first.Message.ToolCalls[0]
	if call.ID != "call_1" || call.Name != "fs_read" || call.Arguments != `{"path":"app \"prod\".log"}` {
		t.Errorf("tool call: got %+v", call)
	}

	// The second turn is exactly what the loop sends: the assistant message
	// unchanged, then the tool result.
	_, err = client.Chat(context.Background(), Request{
		Model: "gemini-3-pro",
		Messages: []Message{
			{Role: RoleUser, Content: "scan app.log"},
			first.Message,
			{Role: RoleTool, ToolCallID: "call_1", Content: "ERROR disk full"},
		},
	})
	if err != nil {
		t.Fatalf("second Chat: %v", err)
	}

	if len(*recorded) != 2 {
		t.Fatalf("recorded %d requests, want 2", len(*recorded))
	}
	contents := gemContentsOf(t, (*recorded)[1])
	if len(contents) != 3 {
		t.Fatalf("contents: got %d entries, want 3", len(contents))
	}
	if contents[1].Role != geminiRoleModel {
		t.Errorf("echoed role: got %q, want model", contents[1].Role)
	}
	if got := string(contents[1].Parts); got != gemSignedParts {
		t.Errorf("model parts are not byte-identical.\ngot:  %s\nwant: %s", got, gemSignedParts)
	}
	// The function NAME is not in the neutral tool-result message; it is
	// recovered from the call the assistant turn announced - which the raw
	// echo path must keep track of just as the rebuilt one does.
	wantResult := `[{"functionResponse":{"id":"call_1","name":"fs_read","response":{"output":"ERROR disk full"}}}]`
	if got := string(contents[2].Parts); got != wantResult {
		t.Errorf("tool result parts:\ngot:  %s\nwant: %s", got, wantResult)
	}
	if contents[2].Role != geminiRoleUser {
		t.Errorf("tool result role: got %q, want user", contents[2].Role)
	}
}

// TestGeminiNoSignatureNoCarrier: an unsigned answer stores NO carrier, and the
// assistant turn is then rebuilt from the neutral fields on the way back. Only
// the signed case pays for the echo path.
func TestGeminiNoSignatureNoCarrier(t *testing.T) {
	parts := `[{"text":"reading"},{"functionCall":{"id":"call_9","name":"fs_read","args":{"path":"a.log"}}}]`
	srv, recorded := gemScriptedServer(t,
		gemCandidateBody(parts, "STOP"),
		gemCandidateBody(`[{"text":"done"}]`, "STOP"),
	)
	client := &GeminiClient{BaseURL: srv.URL}

	first, err := client.Chat(context.Background(), Request{
		Model:    "gemini-3-pro",
		Messages: []Message{{Role: RoleUser, Content: "scan"}},
	})
	if err != nil {
		t.Fatalf("first Chat: %v", err)
	}
	if first.Message.Reasoning != nil {
		t.Fatalf("carrier: got %s, want nil", first.Message.Reasoning)
	}

	if _, err := client.Chat(context.Background(), Request{
		Model:    "gemini-3-pro",
		Messages: []Message{{Role: RoleUser, Content: "scan"}, first.Message},
	}); err != nil {
		t.Fatalf("second Chat: %v", err)
	}
	contents := gemContentsOf(t, (*recorded)[1])
	if len(contents) != 2 {
		t.Fatalf("contents: got %d entries, want 2", len(contents))
	}
	want := `[{"text":"reading"},{"functionCall":{"id":"call_9","name":"fs_read","args":{"path":"a.log"}}}]`
	if got := string(contents[1].Parts); got != want {
		t.Errorf("rebuilt model parts:\ngot:  %s\nwant: %s", got, want)
	}
}

// TestGeminiCarrierFromAnotherWireIsNotEchoed: a reasoning payload captured
// from a different provider (a reasoning_content string) is not a parts array
// and must not be sent as one. Rebuilding from the neutral fields is a
// well-formed request where echoing would be a guaranteed 400.
func TestGeminiCarrierFromAnotherWireIsNotEchoed(t *testing.T) {
	srv, recorded := gemScriptedServer(t, gemCandidateBody(`[{"text":"ok"}]`, "STOP"))
	client := &GeminiClient{BaseURL: srv.URL}

	if _, err := client.Chat(context.Background(), Request{
		Model: "gemini-3-pro",
		Messages: []Message{
			{Role: RoleUser, Content: "hi"},
			{Role: RoleAssistant, Content: "hello", Reasoning: json.RawMessage(`"let me think"`), ReasoningField: "reasoning_content"},
		},
	}); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	contents := gemContentsOf(t, (*recorded)[0])
	if len(contents) != 2 {
		t.Fatalf("contents: got %d entries, want 2", len(contents))
	}
	if got := string(contents[1].Parts); got != `[{"text":"hello"}]` {
		t.Errorf("model parts: got %s", got)
	}
}

// TestGeminiParallelFunctionCalls: several functionCall parts in one candidate
// are the parallel dispatch the loop already knows. Order is the model's and
// must survive, because the loop answers outputs[i] to ToolCalls[i].
func TestGeminiParallelFunctionCalls(t *testing.T) {
	tests := []struct {
		name    string
		parts   string
		wantIDs []string
	}{
		{
			name: "gemini 3 sends ids",
			parts: `[{"functionCall":{"id":"call_a","name":"fs_read","args":{"path":"a.log"}}},` +
				`{"functionCall":{"id":"call_b","name":"fs_read","args":{"path":"b.log"}}}]`,
			wantIDs: []string{"call_a", "call_b"},
		},
		{
			// The 2.5-era models send no id at all; an empty id is tolerated
			// rather than invented, and the pairing falls back to call order.
			name: "2.5-era sends no id",
			parts: `[{"functionCall":{"name":"fs_read","args":{"path":"a.log"}}},` +
				`{"functionCall":{"name":"fs_list"}}]`,
			wantIDs: []string{"", ""},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, _ := gemScriptedServer(t, gemCandidateBody(tt.parts, "STOP"))
			client := &GeminiClient{BaseURL: srv.URL}
			resp, err := client.Chat(context.Background(), Request{
				Model:    "gemini-3-pro",
				Messages: []Message{{Role: RoleUser, Content: "scan both"}},
			})
			if err != nil {
				t.Fatalf("Chat: %v", err)
			}
			if len(resp.Message.ToolCalls) != len(tt.wantIDs) {
				t.Fatalf("tool calls: got %d, want %d", len(resp.Message.ToolCalls), len(tt.wantIDs))
			}
			for i, want := range tt.wantIDs {
				if got := resp.Message.ToolCalls[i].ID; got != want {
					t.Errorf("call %d id: got %q, want %q", i, got, want)
				}
			}
			if resp.Message.ToolCalls[0].Name != "fs_read" {
				t.Errorf("call 0 name: got %q", resp.Message.ToolCalls[0].Name)
			}
			// A zero-argument call still carries an arguments object: the tool
			// layer decodes a JSON object, not an empty string.
			if got := resp.Message.ToolCalls[len(tt.wantIDs)-1].Arguments; got == "" {
				t.Errorf("arguments of the last call are empty")
			}
		})
	}
}

// TestGeminiParallelResultsShareOneTurn: both results of a parallel step go
// back in ONE user content, in call order, each paired to its own call - the
// shape this wire documents for parallel function responses.
func TestGeminiParallelResultsShareOneTurn(t *testing.T) {
	client := &GeminiClient{}
	wire, _ := client.toWire(Request{
		Model: "gemini-3-pro",
		Messages: []Message{
			{Role: RoleUser, Content: "scan both"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{
				{ID: "call_a", Name: "fs_read", Arguments: `{"path":"a.log"}`},
				{ID: "call_b", Name: "fs_list", Arguments: `{}`},
			}},
			// The loop appends results in call order; a provider is free to
			// accept them in any order but amele keeps the deterministic one.
			{Role: RoleTool, ToolCallID: "call_a", Content: "clean"},
			{Role: RoleTool, ToolCallID: "call_b", Content: `{"entries":["a.log"]}`},
		},
	})
	if len(wire.Contents) != 3 {
		t.Fatalf("contents: got %d entries, want 3 (user, model, one tool turn)", len(wire.Contents))
	}
	results := wire.Contents[2]
	if results.Role != geminiRoleUser {
		t.Errorf("tool turn role: got %q", results.Role)
	}
	if len(results.Parts) != 2 {
		t.Fatalf("tool turn parts: got %d, want 2", len(results.Parts))
	}
	if results.Parts[0].FunctionResponse.Name != "fs_read" || results.Parts[1].FunctionResponse.Name != "fs_list" {
		t.Errorf("names out of order: %q, %q",
			results.Parts[0].FunctionResponse.Name, results.Parts[1].FunctionResponse.Name)
	}
	if results.Parts[0].FunctionResponse.ID != "call_a" || results.Parts[1].FunctionResponse.ID != "call_b" {
		t.Errorf("ids out of order: %q, %q",
			results.Parts[0].FunctionResponse.ID, results.Parts[1].FunctionResponse.ID)
	}
}

// TestGeminiToolResponseObject pins the wrapping rule: functionResponse.response
// must be an OBJECT on this wire, so a plain tool output (which is what every
// amele tool returns) travels under an "output" key while a tool that already
// answers with a JSON object passes through unchanged.
func TestGeminiToolResponseObject(t *testing.T) {
	tests := []struct {
		name   string
		result string
		want   string
	}{
		{"plain text wraps", "ERROR disk full", `{"output":"ERROR disk full"}`},
		{"empty output wraps", "", `{"output":""}`},
		{"json object passes through", `{"entries":["a.log"]}`, `{"entries":["a.log"]}`},
		{"json object is compacted", "{\n  \"ok\": true\n}", `{"ok":true}`},
		{
			// Not an object: a JSON array, a number or a bare string is still
			// text as far as this wire is concerned.
			"json array wraps", `["a","b"]`, `{"output":"[\"a\",\"b\"]"}`,
		},
		{"number wraps", "42", `{"output":"42"}`},
		{
			// Looks like an object but is not parseable: wrapping keeps the
			// bytes the tool produced instead of sending a malformed body.
			"malformed object wraps", `{"broken":`, `{"output":"{\"broken\":"}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := string(geminiToolResponse(tt.result)); got != tt.want {
				t.Errorf("got %s, want %s", got, tt.want)
			}
		})
	}
}

// TestGeminiOrphanToolResult: a tool result with no matching call cannot become
// a functionResponse (this wire demands the function name, which only the call
// carries). It travels as plain user text instead - the output still reaches
// the model, and nothing invents a pairing the transcript does not support.
func TestGeminiOrphanToolResult(t *testing.T) {
	client := &GeminiClient{}
	wire, _ := client.toWire(Request{
		Model: "gemini-3-pro",
		Messages: []Message{
			{Role: RoleUser, Content: "hi"},
			{Role: RoleTool, ToolCallID: "call_gone", Content: "stale output"},
		},
	})
	if len(wire.Contents) != 2 {
		t.Fatalf("contents: got %d entries, want 2", len(wire.Contents))
	}
	part := wire.Contents[1].Parts[0]
	if part.FunctionResponse != nil {
		t.Errorf("orphan became a functionResponse: %+v", part.FunctionResponse)
	}
	if part.Text != "stale output" {
		t.Errorf("orphan text: got %q", part.Text)
	}
}

// TestGeminiToolResultIDMismatch: an id that names no announced call is not
// resolved by position. The id is evidence about which call the output answers,
// and pairing it with another function's call would mislabel the result.
func TestGeminiToolResultIDMismatch(t *testing.T) {
	client := &GeminiClient{}
	wire, _ := client.toWire(Request{
		Model: "gemini-3-pro",
		Messages: []Message{
			{Role: RoleUser, Content: "scan"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call_a", Name: "fs_read", Arguments: `{}`}}},
			{Role: RoleTool, ToolCallID: "call_from_another_run", Content: "stale output"},
		},
	})
	if len(wire.Contents) != 3 {
		t.Fatalf("contents: got %d entries, want 3", len(wire.Contents))
	}
	part := wire.Contents[2].Parts[0]
	if part.FunctionResponse != nil {
		t.Errorf("mismatched id became a functionResponse: %+v", part.FunctionResponse)
	}
	if part.Text != "stale output" {
		t.Errorf("text: got %q", part.Text)
	}
}

// TestGeminiCallArgs pins the args rendering: a zero-argument call sends no
// args key (rather than a shape this wire would reject), and arguments that are
// not JSON - only reachable from a history captured on another wire - are not
// silently replaced with an empty object.
func TestGeminiCallArgs(t *testing.T) {
	tests := []struct {
		name string
		args string
		want string
	}{
		{"absent", "", ""},
		{"blank", "   ", ""},
		{"null", "null", ""},
		{"object is compacted", "{\n  \"path\": \"a.log\"\n}", `{"path":"a.log"}`},
		{"not json travels unchanged", "path=a.log", "path=a.log"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := string(geminiCallArgs(tt.args)); got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

// TestGeminiToolsAreSanitized: the declarations that reach the wire carry the
// sanitized schema, so amele's own builtin (which ships additionalProperties)
// does not 400 the whole request.
func TestGeminiToolsAreSanitized(t *testing.T) {
	client := &GeminiClient{}
	wire, _ := client.toWire(Request{
		Model:    "gemini-3-pro",
		Messages: baseMessages(),
		Tools: []ToolDef{
			{Name: "fs_write", Description: "write a file", Parameters: json.RawMessage(fsWriteSchema)},
			{Name: "ping", Description: "no arguments"},
		},
	})
	if len(wire.Tools) != 1 {
		t.Fatalf("tools: got %d entries, want 1 (all declarations share one tool object)", len(wire.Tools))
	}
	decls := wire.Tools[0].FunctionDeclarations
	if len(decls) != 2 {
		t.Fatalf("declarations: got %d, want 2", len(decls))
	}
	if got := string(decls[0].Parameters); got != `{"properties":{"content":{"description":"Full file content","type":"string"},`+
		`"path":{"description":"Relative file path","type":"string"}},"required":["path","content"],"type":"object"}` {
		t.Errorf("fs_write parameters: got %s", got)
	}
	// A tool without a schema declares no parameters key at all.
	if decls[1].Parameters != nil {
		t.Errorf("ping parameters: got %s, want none", decls[1].Parameters)
	}
}
