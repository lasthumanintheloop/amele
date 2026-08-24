package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// anThinkingBlock is a recorded thinking block: signed, non-ASCII (Turkish, an
// em dash, CJK) and carrying an escaped quote. A client that decoded the block
// into Go values and marshalled it again could still land on these bytes, but
// one that re-flowed or re-ordered anything would not - and Anthropic answers a
// modified thinking block with a 400 (research §"Load-bearing quirks" #3).
const anThinkingBlock = `{"type":"thinking","thinking":"planı kurdum: \"app.log\" oku — 思考","signature":"c2lnbmF0dXJl"}`

// anRedactedBlock is the encrypted sibling. It always travels in the same array
// as the thinking blocks, never separately.
const anRedactedBlock = `{"type":"redacted_thinking","data":"RURBQ1RFRA=="}`

const (
	anTextBlock    = `{"type":"text","text":"reading it"}`
	anToolUseBlock = `{"type":"tool_use","id":"toolu_1","name":"fs_read","input":{"path":"app.log"}}`
)

// anContent renders a compact content array. Compact matters: the round-trip
// assertion below is byte-for-byte, and the JSON encoder compacts whatever it
// re-emits, so a fixture with whitespace would fail for a reason that has
// nothing to do with the contract under test.
func anContent(blocks ...string) string {
	return "[" + strings.Join(blocks, ",") + "]"
}

// anBody wraps a content array in the Messages API response envelope.
func anBody(content, stopReason string) string {
	return `{"content":` + content + `,"stop_reason":"` + stopReason +
		`","usage":{"input_tokens":10,"output_tokens":5}}`
}

// anScriptedReply is one scripted HTTP reply for anScriptedServer.
type anScriptedReply struct {
	status int
	body   string
}

// anScriptedServer answers with scripted status+body pairs in order and records
// the RAW request bodies. Raw, because both contracts this file pins - the
// byte-exact echo and the single-shot output_config fallback - are about the
// bytes on the wire; a decode/re-encode in the test would hide exactly the
// corruption it hunts.
func anScriptedServer(t *testing.T, replies ...anScriptedReply) (*httptest.Server, *[][]byte) {
	t.Helper()
	var recorded [][]byte
	var turn int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw := new(bytes.Buffer)
		if _, err := raw.ReadFrom(r.Body); err != nil {
			t.Errorf("reading request body: %v", err)
		}
		recorded = append(recorded, raw.Bytes())
		if turn >= len(replies) {
			t.Errorf("unexpected request %d: %s", turn+1, raw.Bytes())
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		reply := replies[turn]
		turn++
		if reply.status != 0 && reply.status != http.StatusOK {
			w.WriteHeader(reply.status)
		}
		_, _ = w.Write([]byte(reply.body))
	}))
	t.Cleanup(srv.Close)
	return srv, &recorded
}

// anMessagesOf splits a recorded request body into role + raw content array per
// message. json.RawMessage keeps the original bytes, so an assertion on Content
// is structural AND byte-exact at once.
func anMessagesOf(t *testing.T, body []byte) []struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
} {
	t.Helper()
	var parsed struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("decoding recorded body: %v\nbody: %s", err, body)
	}
	return parsed.Messages
}

// TestAnthropicThinkingRoundTrip is the echo-back contract end to end: a
// response whose content array contains thinking (or redacted_thinking) blocks
// must come back out of the client as Message.Reasoning holding the WHOLE
// array, and the next request must send those exact bytes as the assistant
// message's content - signature included, order untouched, nothing rebuilt.
func TestAnthropicThinkingRoundTrip(t *testing.T) {
	tests := []struct {
		name    string
		content string
		// wantText and wantToolCall are what the loop must still see: capturing
		// the raw array may not disturb text/tool_use parsing.
		wantText     string
		wantToolCall bool
	}{
		{
			name:         "thinking, text and tool_use",
			content:      anContent(anThinkingBlock, anTextBlock, anToolUseBlock),
			wantText:     "reading it",
			wantToolCall: true,
		},
		{
			// Both types always travel together inside the one array; storing
			// the array is what makes that automatic.
			name:         "thinking and redacted_thinking together",
			content:      anContent(anThinkingBlock, anRedactedBlock, anTextBlock, anToolUseBlock),
			wantText:     "reading it",
			wantToolCall: true,
		},
		{
			name:     "redacted_thinking alone",
			content:  anContent(anRedactedBlock, anTextBlock),
			wantText: "reading it",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, recorded := anScriptedServer(t,
				anScriptedReply{body: anBody(tt.content, "tool_use")},
				anScriptedReply{body: anOKBody("done")},
			)
			client := &AnthropicClient{BaseURL: srv.URL}

			first, err := client.Chat(context.Background(), Request{
				Model:    "claude-opus-5",
				Messages: []Message{{Role: RoleUser, Content: "scan app.log"}},
			})
			if err != nil {
				t.Fatalf("first Chat: %v", err)
			}
			if got := string(first.Message.Reasoning); got != tt.content {
				t.Fatalf("captured carrier:\ngot:  %s\nwant: %s", got, tt.content)
			}
			if first.Message.Content != tt.wantText {
				t.Errorf("text: got %q, want %q", first.Message.Content, tt.wantText)
			}
			if got := len(first.Message.ToolCalls) == 1; got != tt.wantToolCall {
				t.Errorf("tool calls: got %d, wantToolCall=%v", len(first.Message.ToolCalls), tt.wantToolCall)
			}

			// Second turn: exactly what the loop does - the assistant message
			// goes back unchanged, followed by the tool result.
			msgs := []Message{
				{Role: RoleUser, Content: "scan app.log"},
				first.Message,
			}
			if tt.wantToolCall {
				msgs = append(msgs, Message{Role: RoleTool, ToolCallID: "toolu_1", Content: "ERROR disk full"})
			}
			if _, err := client.Chat(context.Background(), Request{Model: "claude-opus-5", Messages: msgs}); err != nil {
				t.Fatalf("second Chat: %v", err)
			}

			if len(*recorded) != 2 {
				t.Fatalf("recorded %d requests, want 2", len(*recorded))
			}
			sent := anMessagesOf(t, (*recorded)[1])
			if len(sent) != len(msgs) {
				t.Fatalf("messages in echo request: got %d, want %d", len(sent), len(msgs))
			}
			if sent[1].Role != RoleAssistant {
				t.Errorf("echoed message role: got %q", sent[1].Role)
			}
			if got := string(sent[1].Content); got != tt.content {
				t.Errorf("assistant content array is not byte-identical.\ngot:  %s\nwant: %s", got, tt.content)
			}
		})
	}
}

// TestAnthropicEchoDoesNotDuplicateBlocks: the raw array already holds the text
// and tool_use blocks, so the reconstruction path must not run beside it. A
// second text block would be a silent corruption of the transcript and a
// second tool_use an outright protocol error.
func TestAnthropicEchoDoesNotDuplicateBlocks(t *testing.T) {
	content := anContent(anThinkingBlock, anTextBlock, anToolUseBlock)
	srv, recorded := anScriptedServer(t,
		anScriptedReply{body: anBody(content, "tool_use")},
		anScriptedReply{body: anOKBody("done")},
	)
	client := &AnthropicClient{BaseURL: srv.URL}

	first, err := client.Chat(context.Background(), Request{
		Model:    "claude-opus-5",
		Messages: []Message{{Role: RoleUser, Content: "scan app.log"}},
	})
	if err != nil {
		t.Fatalf("first Chat: %v", err)
	}
	if _, err := client.Chat(context.Background(), Request{
		Model: "claude-opus-5",
		Messages: []Message{
			{Role: RoleUser, Content: "scan app.log"},
			first.Message,
			{Role: RoleTool, ToolCallID: "toolu_1", Content: "ERROR disk full"},
		},
	}); err != nil {
		t.Fatalf("second Chat: %v", err)
	}

	var blocks []json.RawMessage
	if err := json.Unmarshal(anMessagesOf(t, (*recorded)[1])[1].Content, &blocks); err != nil {
		t.Fatalf("decoding echoed content: %v", err)
	}
	if len(blocks) != 3 {
		t.Fatalf("echoed blocks: got %d, want 3 (the raw array, nothing rebuilt): %s", len(blocks), blocks)
	}
	if n := bytes.Count((*recorded)[1], []byte(`"type":"tool_use"`)); n != 1 {
		t.Errorf("tool_use blocks in the request: got %d, want 1", n)
	}
}

// TestAnthropicNoThinkingNoCarrier is the silence half: a response without
// thinking blocks leaves the carrier nil, and the assistant message is
// reconstructed from text + tool calls exactly as before this feature existed.
func TestAnthropicNoThinkingNoCarrier(t *testing.T) {
	srv, recorded := anScriptedServer(t,
		anScriptedReply{body: anBody(anContent(anTextBlock, anToolUseBlock), "tool_use")},
		anScriptedReply{body: anOKBody("done")},
	)
	client := &AnthropicClient{BaseURL: srv.URL}

	first, err := client.Chat(context.Background(), Request{
		Model:    "claude-opus-5",
		Messages: []Message{{Role: RoleUser, Content: "scan app.log"}},
	})
	if err != nil {
		t.Fatalf("first Chat: %v", err)
	}
	if first.Message.Reasoning != nil {
		t.Fatalf("carrier must stay nil, got %s", first.Message.Reasoning)
	}
	if _, err := client.Chat(context.Background(), Request{
		Model: "claude-opus-5",
		Messages: []Message{
			{Role: RoleUser, Content: "scan app.log"},
			first.Message,
			{Role: RoleTool, ToolCallID: "toolu_1", Content: "ERROR disk full"},
		},
	}); err != nil {
		t.Fatalf("second Chat: %v", err)
	}
	rebuilt := string(anMessagesOf(t, (*recorded)[1])[1].Content)
	if !strings.Contains(rebuilt, `"text":"reading it"`) || !strings.Contains(rebuilt, `"id":"toolu_1"`) {
		t.Errorf("assistant message must be reconstructed from the neutral fields: %s", rebuilt)
	}
}

// TestAnthropicCarrierFromAnotherWireIsNotEchoed guards the one case where the
// carrier cannot be trusted as a content array: a history captured from the
// OpenAI wire (a reasoning_content string) replayed against this client. The
// echo path takes arrays only; anything else falls back to reconstruction
// rather than sending a content field the API cannot parse.
func TestAnthropicCarrierFromAnotherWireIsNotEchoed(t *testing.T) {
	client := &AnthropicClient{}
	wire, _ := client.toWire(Request{
		Model: "claude-opus-5",
		Messages: []Message{
			{Role: RoleUser, Content: "hi"},
			{Role: RoleAssistant, Content: "hello", Reasoning: json.RawMessage(`"a plaintext reasoning summary"`)},
		},
	})
	body, err := encodeBody(wire, nil)
	if err != nil {
		t.Fatalf("encodeBody: %v", err)
	}
	if bytes.Contains(body, []byte("plaintext reasoning summary")) {
		t.Errorf("a non-array carrier must not become the content field: %s", body)
	}
	if !bytes.Contains(body, []byte(`"text":"hello"`)) {
		t.Errorf("assistant message must fall back to reconstruction: %s", body)
	}
}

// anWireCase is one request rendered by the Anthropic client. The golden holds
// the EXACT bytes that would go on the wire: the point of this task is which
// keys and values the provider receives.
type anWireCase struct {
	name   string
	golden string
	client AnthropicClient
	req    Request
}

func anWireCases() []anWireCase {
	base := []Message{
		{Role: RoleSystem, Content: "you are a log sentry"},
		{Role: RoleUser, Content: "scan today's log"},
	}
	schema := &ResponseFormat{Name: "amele_output", Schema: json.RawMessage(`{"type":"object","required":["verdict"]}`)}
	return []anWireCase{
		{
			// The pre-thinking baseline: a config that asks for nothing new
			// sends exactly what it sent before this task.
			name:   "baseline",
			golden: "anthropic-baseline.json",
			req:    Request{Model: "claude-opus-5", Messages: base},
		},
		{
			// Current models (4.7+): adaptive thinking, level in output_config.
			name:   "adaptive effort",
			golden: "anthropic-adaptive-effort.json",
			req: Request{
				Model:           "claude-opus-5",
				Messages:        base,
				MaxOutputTokens: 32000,
				Reasoning:       &ReasoningSpec{Effort: "high"},
			},
		},
		{
			// A budget switches to the legacy shape (Haiku 4.5 and older),
			// which has no effort field.
			name:   "budget legacy",
			golden: "anthropic-budget-legacy.json",
			client: AnthropicClient{MaxOutputTokens: 16000},
			req: Request{
				Model:     "claude-haiku-4-5",
				Messages:  base,
				Reasoning: &ReasoningSpec{BudgetTokens: 8192},
			},
		},
		{
			name:   "thinking disabled",
			golden: "anthropic-thinking-disabled.json",
			req: Request{
				Model:     "claude-opus-5",
				Messages:  base,
				Reasoning: &ReasoningSpec{Effort: effortNone},
			},
		},
		{
			// Native structured output: GA via output_config.format.
			name:   "native schema",
			golden: "anthropic-schema.json",
			req: Request{
				Model:          "claude-opus-5",
				Messages:       base,
				ResponseFormat: schema,
			},
		},
		{
			// output_config is ONE object: effort and format share it.
			name:   "effort and schema merged",
			golden: "anthropic-effort-schema.json",
			req: Request{
				Model:           "claude-opus-5",
				Messages:        base,
				MaxOutputTokens: 64000,
				Reasoning:       &ReasoningSpec{Effort: "xhigh"},
				ResponseFormat:  schema,
				Tools: []ToolDef{{
					Name:        "fs_read",
					Description: "read a file",
					Parameters:  json.RawMessage(`{"type":"object"}`),
				}},
			},
		},
		{
			name:   "sampling and params",
			golden: "anthropic-sampling-params.json",
			req: Request{
				Model:       "claude-sonnet-5",
				Messages:    base,
				Temperature: ptr(0.0),
				TopP:        ptr(0.9),
				Extra: map[string]json.RawMessage{
					"metadata":    json.RawMessage(`{"user_id":"abc"}`),
					"top_k":       json.RawMessage(` 40 `),
					"stop_sequen": json.RawMessage(`["END"]`),
				},
			},
		},
	}
}

// TestAnthropicToWireGolden pins the request body byte for byte.
func TestAnthropicToWireGolden(t *testing.T) {
	for _, tc := range anWireCases() {
		t.Run(tc.name, func(t *testing.T) {
			client := tc.client
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
			if want := strings.TrimSuffix(string(raw), "\n"); string(got) != want {
				t.Errorf("request body differs from golden %s.\ngot:  %s\nwant: %s", tc.golden, got, want)
			}
		})
	}
}

// TestMapAnthropicThinking walks the whole neutral vocabulary against the two
// wire shapes Anthropic splits it across. It is the completeness check the
// thinking mapping needs: a new effort value has to be answered here explicitly.
func TestMapAnthropicThinking(t *testing.T) {
	tests := []struct {
		spec         ReasoningSpec
		wantThinking string
		wantEffort   string
	}{
		{ReasoningSpec{}, "", ""},
		{ReasoningSpec{Effort: "none"}, `{"type":"disabled"}`, ""},
		{ReasoningSpec{Effort: "low"}, `{"type":"adaptive"}`, "low"},
		{ReasoningSpec{Effort: "medium"}, `{"type":"adaptive"}`, "medium"},
		{ReasoningSpec{Effort: "high"}, `{"type":"adaptive"}`, "high"},
		// xhigh and max are in Anthropic's own effort vocabulary: they pass
		// through unrounded.
		{ReasoningSpec{Effort: "xhigh"}, `{"type":"adaptive"}`, "xhigh"},
		{ReasoningSpec{Effort: "max"}, `{"type":"adaptive"}`, "max"},
		// A budget is the legacy shape, and it wins over a level: the two
		// spellings target different model generations.
		{ReasoningSpec{BudgetTokens: 1024}, `{"type":"enabled","budget_tokens":1024}`, ""},
		{ReasoningSpec{Effort: "high", BudgetTokens: 1024}, `{"type":"enabled","budget_tokens":1024}`, ""},
	}

	for _, tt := range tests {
		t.Run(tt.spec.Effort+"/"+itoaTest(tt.spec.BudgetTokens), func(t *testing.T) {
			thinking, effort := mapAnthropicThinking(tt.spec)
			var got string
			if thinking != nil {
				raw, err := json.Marshal(thinking)
				if err != nil {
					t.Fatalf("marshaling thinking: %v", err)
				}
				got = string(raw)
			}
			if got != tt.wantThinking {
				t.Errorf("thinking: got %s, want %s", got, tt.wantThinking)
			}
			if effort != tt.wantEffort {
				t.Errorf("effort: got %q, want %q", effort, tt.wantEffort)
			}
		})
	}
}

// itoaTest names a subtest by its budget without pulling strconv into the
// assertions.
func itoaTest(n int) string {
	if n == 0 {
		return "nobudget"
	}
	return "budget"
}

// TestAnthropicNativeSchemaIsSent is the contract change this task makes: the
// Messages API has native json_schema enforcement (GA via output_config.format),
// so the schema goes on the wire and the response is NOT flagged as a dropped
// enforcement. It replaces the older TestAnthropicResponseFormatIgnored, which
// pinned the opposite behavior back when this client sent no schema at all.
func TestAnthropicNativeSchemaIsSent(t *testing.T) {
	srv := anthropicServer(t, func(w http.ResponseWriter, req map[string]any) {
		// Errorf, not Fatalf: this runs on the server's goroutine, where a
		// Fatalf would kill the handler mid-reply and surface as a confusing
		// transport error instead of the assertion that failed.
		if cfg, ok := req["output_config"].(map[string]any); !ok {
			t.Errorf("request must carry output_config: %v", req)
		} else if format, ok := cfg["format"].(map[string]any); !ok {
			t.Errorf("output_config must carry format: %v", cfg)
		} else if format["type"] != "json_schema" {
			t.Errorf("format type: got %v", format["type"])
		} else if _, present := format["name"]; present {
			t.Errorf("anthropic's format object takes no name: %v", format)
		}
		if _, present := req["response_format"]; present {
			t.Error("response_format is the OpenAI spelling and must never appear here")
		}
		_, _ = w.Write([]byte(anOKBody(`{"ok":true}`)))
	})

	client := &AnthropicClient{BaseURL: srv.URL}
	resp, err := client.Chat(context.Background(), Request{
		Model:          "claude-opus-5",
		Messages:       []Message{{Role: RoleUser, Content: "x"}},
		ResponseFormat: &ResponseFormat{Name: "amele_output", Schema: json.RawMessage(`{"type":"object"}`)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.SchemaEnforcementDropped {
		t.Error("a natively enforced schema must not be flagged as dropped")
	}

	// With no schema requested there is nothing to drop and nothing to send.
	plainSrv := anthropicServer(t, func(w http.ResponseWriter, req map[string]any) {
		if _, present := req["output_config"]; present {
			t.Errorf("schema-less request must not carry output_config: %v", req)
		}
		_, _ = w.Write([]byte(anOKBody("hi")))
	})
	plain, err := (&AnthropicClient{BaseURL: plainSrv.URL}).Chat(context.Background(), Request{
		Model:    "claude-opus-5",
		Messages: []Message{{Role: RoleUser, Content: "x"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if plain.SchemaEnforcementDropped {
		t.Error("schema-less request must not set SchemaEnforcementDropped")
	}
}

// bodyOutputConfigRejected is the 400 an endpoint that does not implement
// output_config answers with (Anthropic's own strict-field error shape, which
// the Anthropic-compatible gateways of DeepSeek/GLM/Kimi mirror).
const bodyOutputConfigRejected = `{"type":"error","error":{"type":"invalid_request_error","message":"output_config: Extra inputs are not permitted"}}`

// TestAnthropicOutputConfigFallback: a 400 naming output_config is capability
// discovery, not a transient failure. The call repeats ONCE without the field,
// costs no retry budget and no backoff sleep, and every response after that
// point is flagged SchemaEnforcementDropped - the validate+retry layer above is
// then the only thing enforcing output.schema.
func TestAnthropicOutputConfigFallback(t *testing.T) {
	srv, recorded := anScriptedServer(t,
		anScriptedReply{status: http.StatusBadRequest, body: bodyOutputConfigRejected},
		anScriptedReply{body: anOKBody(`{"ok":true}`)},
	)
	var sleeps int
	client := &AnthropicClient{BaseURL: srv.URL, Sleep: noSleep(t, &sleeps)}

	resp, err := client.Chat(context.Background(), Request{
		Model:          "claude-opus-5",
		Messages:       []Message{{Role: RoleUser, Content: "x"}},
		Reasoning:      &ReasoningSpec{Effort: "high"},
		ResponseFormat: &ResponseFormat{Name: "amele_output", Schema: json.RawMessage(`{"type":"object"}`)},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if !resp.SchemaEnforcementDropped {
		t.Error("a response produced after the fallback must be flagged SchemaEnforcementDropped")
	}
	if len(*recorded) != 2 {
		t.Fatalf("requests: got %d, want 2 (one fallback, exactly once)", len(*recorded))
	}
	if !bytes.Contains((*recorded)[0], []byte(`"output_config"`)) {
		t.Errorf("first request must carry output_config: %s", (*recorded)[0])
	}
	if bytes.Contains((*recorded)[1], []byte(`"output_config"`)) {
		t.Errorf("fallback request must drop output_config entirely: %s", (*recorded)[1])
	}
	// The thinking object survives the fallback: only the rejected field goes.
	if !bytes.Contains((*recorded)[1], []byte(`"thinking":{"type":"adaptive"}`)) {
		t.Errorf("fallback must keep the thinking control: %s", (*recorded)[1])
	}
	if sleeps != 0 {
		t.Errorf("the fallback must not sleep, got %d sleeps", sleeps)
	}
}

// TestAnthropicUnrelated400NoFallback: only a 400 that NAMES output_config is
// capability discovery. Any other rejection stays a hard failure instead of
// being silently downgraded to a schema-less request.
func TestAnthropicUnrelated400NoFallback(t *testing.T) {
	srv, recorded := anScriptedServer(t,
		anScriptedReply{status: http.StatusBadRequest, body: `{"type":"error","error":{"message":"bad model"}}`},
	)
	client := &AnthropicClient{BaseURL: srv.URL}
	_, err := client.Chat(context.Background(), Request{
		Model:          "claude-opus-5",
		Messages:       []Message{{Role: RoleUser, Content: "x"}},
		ResponseFormat: &ResponseFormat{Name: "amele_output", Schema: json.RawMessage(`{"type":"object"}`)},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, ErrProvider) {
		t.Errorf("should wrap ErrProvider: %v", err)
	}
	if len(*recorded) != 1 {
		t.Errorf("requests: got %d, want 1 (no fallback for an unrelated 400)", len(*recorded))
	}
}

// The Anthropic sampling rejections these heuristics exist for. Current models
// answer a non-default temperature with a 400 unconditionally, and the older
// ones do it while thinking is enabled (research §matrix "temperature/top_p").
const (
	bodyAnthropicTemperatureWithThinking = `{"type":"error","error":{"type":"invalid_request_error","message":"` +
		"`temperature`" + ` may only be set to 1 when thinking is enabled."}}`
	bodyAnthropicTemperatureNotSupported = `{"type":"error","error":{"type":"invalid_request_error","message":"temperature is not supported with this model."}}`
	// The same two phrasings for top_p. A config may set top_p alone, and
	// before 2026-08-24 that 400 matched nothing at all.
	bodyAnthropicTopPWithThinking = `{"type":"error","error":{"type":"invalid_request_error","message":"` +
		"`top_p`" + ` may only be set to 1 when thinking is enabled."}}`
	bodyAnthropicTopPNotSupported = `{"type":"error","error":{"type":"invalid_request_error","message":"top_p is not supported with this model."}}`
)

// TestAnthropic400AdviceForSampling: the recognized sampling 400 gains the hint
// naming the config keys to remove, and nothing else changes - same message,
// same single final call.
func TestAnthropic400AdviceForSampling(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{"thinking forbids sampling", bodyAnthropicTemperatureWithThinking, adviceNoSampling},
		{"model rejects sampling", bodyAnthropicTemperatureNotSupported, adviceNoSampling},
		{"thinking forbids top_p", bodyAnthropicTopPWithThinking, adviceNoSampling},
		{"model rejects top_p", bodyAnthropicTopPNotSupported, adviceNoSampling},
		{"unrelated 400 keeps its message", `{"error":{"message":"bad model"}}`, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls int
			srv := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
				calls++
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(tt.body))
			})
			client := &AnthropicClient{BaseURL: srv.URL}
			_, err := client.Chat(context.Background(), Request{
				Model:       "claude-opus-5",
				Messages:    []Message{{Role: RoleUser, Content: "x"}},
				Temperature: ptr(0.7),
			})
			if err == nil {
				t.Fatal("expected error")
			}
			want := "provider error: status 400: " + tt.body
			if tt.want != "" {
				want += " — " + tt.want
			}
			if err.Error() != want {
				t.Errorf("message:\n got %q\nwant %q", err.Error(), want)
			}
			if calls != 1 {
				t.Errorf("400 must not be retried, got %d calls", calls)
			}
		})
	}
}

// TestAnthropicContentDecoding: the content array is decoded separately from
// the envelope now that its raw bytes are the carrier, so both ends of that
// split need pinning - an absent content field is an empty turn, a content
// field that is not an array is a decode error naming the response rather than
// a silently empty answer.
func TestAnthropicContentDecoding(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		wantErr bool
	}{
		{"content absent", `{"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`, false},
		{"content null", `{"content":null,"stop_reason":"end_turn"}`, false},
		{"content is not an array", `{"content":{"type":"text"},"stop_reason":"end_turn"}`, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
				_, _ = w.Write([]byte(tt.body))
			})
			client := &AnthropicClient{BaseURL: srv.URL}
			resp, err := client.Chat(context.Background(), Request{
				Model:    "claude-opus-5",
				Messages: []Message{{Role: RoleUser, Content: "x"}},
			})
			if tt.wantErr {
				if !errors.Is(err, ErrProvider) {
					t.Fatalf("got %v, want a provider error", err)
				}
				if !strings.Contains(err.Error(), "decoding response") {
					t.Errorf("error must name the decode step: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("Chat: %v", err)
			}
			if resp.Message.Content != "" || resp.Message.Reasoning != nil {
				t.Errorf("empty turn: got %+v", resp.Message)
			}
		})
	}
}

// TestAnthropicRequestMaxOutputTokensWins: max_tokens is required on every
// Anthropic request, and the per-call value is the more specific instruction -
// it overrides the client-level default so the cmd wiring can pass the config's
// cap through the neutral Request like every other dialect does.
func TestAnthropicRequestMaxOutputTokensWins(t *testing.T) {
	srv := anthropicServer(t, func(w http.ResponseWriter, req map[string]any) {
		if req["max_tokens"] != float64(4096) {
			t.Errorf("max_tokens: got %v, want 4096", req["max_tokens"])
		}
		_, _ = w.Write([]byte(anOKBody("ok")))
	})
	client := &AnthropicClient{BaseURL: srv.URL, MaxOutputTokens: 512}
	if _, err := client.Chat(context.Background(), Request{
		Model:           "claude-opus-5",
		Messages:        []Message{{Role: RoleUser, Content: "x"}},
		MaxOutputTokens: 4096,
	}); err != nil {
		t.Fatal(err)
	}
}
