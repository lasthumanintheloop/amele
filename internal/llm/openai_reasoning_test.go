package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// reasoningPayload is the reasoning text every fixture carries, as it appears
// INSIDE the JSON body. It is deliberately hostile to a re-encoding client:
// non-ASCII (Turkish, an em dash, CJK) and escaped quotes. A client that
// decoded the payload into a Go string and marshalled it again would still
// produce these bytes, but one that re-flowed or re-escaped them would not -
// and the providers that sign or hash their reasoning reject the difference.
const reasoningPayload = `"planı kurdum: \"app.log\" oku — 思考"`

// detailsPayload is OpenRouter's typed array form of the same thing, including
// the signature field that makes byte-exactness load-bearing.
const detailsPayload = `[{"type":"reasoning.text","text":"planı kurdum: \"app.log\" oku — 思考","format":"unknown","signature":"c2ln=="}]`

// recordingServer answers with scripted bodies in order and records the RAW
// request bodies. Recording bytes rather than a decoded map is the point of
// this file: the echo-back contract is about the bytes on the wire, and a
// decode/re-encode in the test would hide exactly the corruption it hunts.
func recordingServer(t *testing.T, bodies ...string) (*httptest.Server, *[][]byte) {
	t.Helper()
	var recorded [][]byte
	var turn int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("reading request body: %v", err)
		}
		recorded = append(recorded, raw)
		if turn >= len(bodies) {
			t.Errorf("unexpected request %d: %s", turn+1, raw)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte(bodies[turn]))
		turn++
	}))
	t.Cleanup(srv.Close)
	return srv, &recorded
}

// toolCallResponse is a first-turn reply that thinks and then calls a tool -
// the exact shape whose follow-up request must carry the reasoning back.
// extra is spliced into the message object verbatim so each dialect's fixture
// can carry its own reasoning fields.
func toolCallResponse(extra string) string {
	return `{"choices":[{"message":{"role":"assistant","content":"",` + extra +
		`"tool_calls":[{"id":"call_1","type":"function","function":{"name":"fs_read","arguments":"{\"path\":\"app.log\"}"}}]},
		"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`
}

// messagesOf splits a recorded request body into the raw bytes of each message
// object. json.RawMessage keeps the original bytes, so the assertions below
// are structural (which message carries the key) AND byte-exact at once.
func messagesOf(t *testing.T, body []byte) []json.RawMessage {
	t.Helper()
	var parsed struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		t.Fatalf("decoding recorded body: %v\nbody: %s", err, body)
	}
	return parsed.Messages
}

// TestReasoningRoundTrip is the echo-back contract end to end: a response that
// carries reasoning must come back out of the client in Message.Reasoning
// unchanged, and the NEXT request must put those exact bytes under the key
// this dialect expects, on the assistant message that produced them.
func TestReasoningRoundTrip(t *testing.T) {
	tests := []struct {
		name string
		// dialect under test.
		dialect Dialect
		// extra is the reasoning part of the response message.
		extra string
		// wantCarrier is the byte-exact payload the client must capture.
		wantCarrier string
		// wantKey is the message key the echo must use; absentKey must never
		// appear anywhere in the request.
		wantKey   string
		absentKey string
		// absentText is response content that must never reach the next
		// request under any key.
		absentText string
	}{
		{
			// DeepSeek thinks by DEFAULT: this request never set a reasoning
			// knob, and omitting the echo in a tool loop is a hard 400.
			name:        "deepseek",
			dialect:     DialectDeepSeek,
			extra:       `"reasoning_content":` + reasoningPayload + `,`,
			wantCarrier: reasoningPayload,
			wantKey:     "reasoning_content",
			absentKey:   "reasoning_details",
		},
		{
			name:        "glm",
			dialect:     DialectGLM,
			extra:       `"reasoning_content":` + reasoningPayload + `,`,
			wantCarrier: reasoningPayload,
			wantKey:     "reasoning_content",
			absentKey:   "reasoning_details",
		},
		{
			name:        "kimi",
			dialect:     DialectKimi,
			extra:       `"reasoning_content":` + reasoningPayload + `,`,
			wantCarrier: reasoningPayload,
			wantKey:     "reasoning_content",
			absentKey:   "reasoning_details",
		},
		{
			// Gateways in front of a thinking model can put reasoning_content
			// on the openai wire too; capturing it is free and echoing it is
			// what the model behind the gateway needs.
			name:        "openai",
			dialect:     DialectOpenAI,
			extra:       `"reasoning_content":` + reasoningPayload + `,`,
			wantCarrier: reasoningPayload,
			wantKey:     "reasoning_content",
			absentKey:   "reasoning_details",
		},
		{
			name:        "groq",
			dialect:     DialectGroq,
			extra:       `"reasoning_content":` + reasoningPayload + `,`,
			wantCarrier: reasoningPayload,
			wantKey:     "reasoning_content",
			absentKey:   "reasoning_details",
		},
		{
			// OpenRouter returns BOTH forms. The typed array is the one that
			// carries signatures and survives echo-back; the plaintext
			// "reasoning" field is display sugar and must be ignored.
			name:        "openrouter prefers reasoning_details",
			dialect:     DialectOpenRouter,
			extra:       `"reasoning":"plaintext summary that must not be echoed","reasoning_details":` + detailsPayload + `,`,
			wantCarrier: detailsPayload,
			wantKey:     "reasoning_details",
			absentKey:   "reasoning_content",
			absentText:  "plaintext summary that must not be echoed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, recorded := recordingServer(t,
				toolCallResponse(tt.extra),
				okBody("done"),
			)
			client := &OpenAIClient{BaseURL: srv.URL + "/v1", Dialect: tt.dialect}

			first, err := client.Chat(context.Background(), Request{
				Model:    "m",
				Messages: []Message{{Role: RoleUser, Content: "scan app.log"}},
			})
			if err != nil {
				t.Fatalf("first Chat: %v", err)
			}
			if got := string(first.Message.Reasoning); got != tt.wantCarrier {
				t.Fatalf("captured reasoning:\ngot:  %s\nwant: %s", got, tt.wantCarrier)
			}

			// Second turn: exactly what the loop does - the assistant message
			// goes back unchanged, followed by the tool result.
			if _, err := client.Chat(context.Background(), Request{
				Model: "m",
				Messages: []Message{
					{Role: RoleUser, Content: "scan app.log"},
					first.Message,
					{Role: RoleTool, ToolCallID: "call_1", Content: "ERROR disk full"},
				},
			}); err != nil {
				t.Fatalf("second Chat: %v", err)
			}

			if len(*recorded) != 2 {
				t.Fatalf("recorded %d requests, want 2", len(*recorded))
			}
			body := (*recorded)[1]
			if bytes.Contains(body, []byte(tt.absentKey)) {
				t.Errorf("request carries the wrong key %q: %s", tt.absentKey, body)
			}
			if tt.absentText != "" && bytes.Contains(body, []byte(tt.absentText)) {
				t.Errorf("request echoed the plaintext summary: %s", body)
			}
			msgs := messagesOf(t, body)
			if len(msgs) != 3 {
				t.Fatalf("messages in echo request: got %d, want 3", len(msgs))
			}
			want := []byte(`"` + tt.wantKey + `":` + tt.wantCarrier)
			if !bytes.Contains(msgs[1], want) {
				t.Errorf("assistant message lost the verbatim payload.\ngot:  %s\nwant to contain: %s", msgs[1], want)
			}
			for _, i := range []int{0, 2} {
				if bytes.Contains(msgs[i], []byte("reasoning")) {
					t.Errorf("message %d must carry no reasoning: %s", i, msgs[i])
				}
			}
		})
	}
}

// TestNoReasoningNoKey is the silence half of the contract: nothing the
// provider did not send may appear in the request. A stray reasoning key is
// not cosmetic - the strict dialects answer an unknown or empty field with a
// 400, and a `null` echoed back is exactly such a field.
func TestNoReasoningNoKey(t *testing.T) {
	tests := []struct {
		name  string
		extra string
	}{
		{"field absent", ""},
		{"field null", `"reasoning_content":null,`},
		{"details null", `"reasoning_details":null,`},
	}

	for _, tt := range tests {
		for _, dialect := range dialects {
			t.Run(tt.name+"/"+string(dialect), func(t *testing.T) {
				srv, recorded := recordingServer(t,
					toolCallResponse(tt.extra),
					okBody("done"),
				)
				client := &OpenAIClient{BaseURL: srv.URL + "/v1", Dialect: dialect}

				first, err := client.Chat(context.Background(), Request{
					Model:    "m",
					Messages: []Message{{Role: RoleUser, Content: "scan app.log"}},
				})
				if err != nil {
					t.Fatalf("first Chat: %v", err)
				}
				if first.Message.Reasoning != nil {
					t.Fatalf("carrier must stay nil, got %s", first.Message.Reasoning)
				}

				if _, err := client.Chat(context.Background(), Request{
					Model: "m",
					Messages: []Message{
						{Role: RoleUser, Content: "scan app.log"},
						first.Message,
						{Role: RoleTool, ToolCallID: "call_1", Content: "ERROR disk full"},
					},
				}); err != nil {
					t.Fatalf("second Chat: %v", err)
				}
				if body := (*recorded)[1]; bytes.Contains(body, []byte("reasoning")) {
					t.Errorf("request invented a reasoning key: %s", body)
				}
			})
		}
	}
}

// TestReasoningEchoSurvivesHTMLEscaping pins what "verbatim" means for a
// payload containing <, > or &: encoding/json escapes those in every string it
// writes, so the echoed BYTES differ while the decoded VALUE - the thing a
// provider signs, hashes and re-reads - is identical. The alternative
// (disabling HTML escaping for the whole body) would change every request
// amele sends to buy nothing; this test is the record of that choice.
func TestReasoningEchoSurvivesHTMLEscaping(t *testing.T) {
	const payload = `"compared a<b && c>d, then \"quoted\""`
	srv, recorded := recordingServer(t,
		toolCallResponse(`"reasoning_content":`+payload+`,`),
		okBody("done"),
	)
	client := &OpenAIClient{BaseURL: srv.URL + "/v1", Dialect: DialectDeepSeek}

	first, err := client.Chat(context.Background(), Request{
		Model:    "m",
		Messages: []Message{{Role: RoleUser, Content: "go"}},
	})
	if err != nil {
		t.Fatalf("first Chat: %v", err)
	}
	if got := string(first.Message.Reasoning); got != payload {
		t.Fatalf("captured reasoning:\ngot:  %s\nwant: %s", got, payload)
	}
	if _, err := client.Chat(context.Background(), Request{
		Model:    "m",
		Messages: []Message{{Role: RoleUser, Content: "go"}, first.Message},
	}); err != nil {
		t.Fatalf("second Chat: %v", err)
	}

	var echoed struct {
		Messages []struct {
			ReasoningContent string `json:"reasoning_content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal((*recorded)[1], &echoed); err != nil {
		t.Fatalf("decoding recorded body: %v", err)
	}
	var want string
	if err := json.Unmarshal([]byte(payload), &want); err != nil {
		t.Fatal(err)
	}
	if got := echoed.Messages[1].ReasoningContent; got != want {
		t.Errorf("decoded reasoning changed across the round-trip:\ngot:  %q\nwant: %q", got, want)
	}
}

// TestReasoningEchoedOnAssistantOnly guards the role check. A carrier can only
// come from an assistant message, so one attached to a user or tool message is
// a bug upstream; echoing it would hand a strict provider an unknown field on
// a message shape that never has one.
func TestReasoningEchoedOnAssistantOnly(t *testing.T) {
	client := &OpenAIClient{Dialect: DialectDeepSeek}
	wire, fields := client.toWire(Request{
		Model: "m",
		Messages: []Message{
			{Role: RoleUser, Content: "hi", Reasoning: json.RawMessage(reasoningPayload)},
			{Role: RoleTool, ToolCallID: "call_1", Content: "out", Reasoning: json.RawMessage(reasoningPayload)},
		},
	})
	body, err := encodeBody(wire, fields)
	if err != nil {
		t.Fatalf("encodeBody: %v", err)
	}
	if bytes.Contains(body, []byte("reasoning")) {
		t.Errorf("non-assistant message echoed reasoning: %s", body)
	}
}
