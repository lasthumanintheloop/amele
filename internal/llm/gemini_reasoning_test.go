package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestGeminiThinkingMapping pins the whole effort vocabulary against the wire
// object AND the notes at once, which is the point of the single-mapping-
// function discipline: the report cannot describe a request the client does not
// send, because both come from mapGeminiThinking.
func TestGeminiThinkingMapping(t *testing.T) {
	tests := []struct {
		name      string
		spec      ReasoningSpec
		wantWire  string // the encoded thinkingConfig, "" when nothing is sent
		wantNotes []string
	}{
		{
			name:     "no knob sends nothing",
			spec:     ReasoningSpec{},
			wantWire: "",
		},
		{
			name:      "low",
			spec:      ReasoningSpec{Effort: "low"},
			wantWire:  `{"thinkingLevel":"low"}`,
			wantNotes: []string{"reasoning.effort: low -> thinkingConfig.thinkingLevel: low"},
		},
		{
			name:      "medium",
			spec:      ReasoningSpec{Effort: "medium"},
			wantWire:  `{"thinkingLevel":"medium"}`,
			wantNotes: []string{"reasoning.effort: medium -> thinkingConfig.thinkingLevel: medium"},
		},
		{
			name:      "high",
			spec:      ReasoningSpec{Effort: "high"},
			wantWire:  `{"thinkingLevel":"high"}`,
			wantNotes: []string{"reasoning.effort: high -> thinkingConfig.thinkingLevel: high"},
		},
		{
			// The rounding is DOWNWARD here, unlike the openai-wire dialects
			// that round up: high is the deepest level this wire has, so there
			// is nothing above it to round to. The note is what keeps that
			// visible instead of silent.
			name:      "xhigh rounds to high",
			spec:      ReasoningSpec{Effort: "xhigh"},
			wantWire:  `{"thinkingLevel":"high"}`,
			wantNotes: []string{"reasoning.effort: xhigh -> thinkingConfig.thinkingLevel: high (gemini has no level above high)"},
		},
		{
			name:      "max rounds to high",
			spec:      ReasoningSpec{Effort: "max"},
			wantWire:  `{"thinkingLevel":"high"}`,
			wantNotes: []string{"reasoning.effort: max -> thinkingConfig.thinkingLevel: high (gemini has no level above high)"},
		},
		{
			// none is a real instruction, not the absence of one, and this wire
			// spells it as a zero budget. It only works on the 2.5-era models;
			// a Gemini 3 model answers with a 400 the signature table explains.
			name:      "none sends a zero budget",
			spec:      ReasoningSpec{Effort: "none"},
			wantWire:  `{"thinkingBudget":0}`,
			wantNotes: []string{"reasoning.effort: none -> thinkingConfig.thinkingBudget: 0 (gemini 3 models cannot disable thinking; this 400s there)"},
		},
		{
			name:      "budget passthrough",
			spec:      ReasoningSpec{BudgetTokens: 2048},
			wantWire:  `{"thinkingBudget":2048}`,
			wantNotes: []string{"reasoning.budget_tokens: 2048 -> thinkingConfig.thinkingBudget: 2048"},
		},
		{
			// Unreachable through validate (config refuses the pair), but the
			// mapping must stay total: sending BOTH fields is a 400, so the
			// budget - the more specific instruction - wins and the dropped
			// effort is reported rather than silently ignored.
			name:     "budget and effort together send only the budget",
			spec:     ReasoningSpec{Effort: "high", BudgetTokens: 512},
			wantWire: `{"thinkingBudget":512}`,
			wantNotes: []string{
				"reasoning.budget_tokens: 512 -> thinkingConfig.thinkingBudget: 512",
				"reasoning.effort: high not sent: the gemini wire takes a thinking level or a budget, not both",
			},
		},
		{
			// The vocabulary is validated in config; an unknown value passes
			// through so the provider's own 400 names the typo precisely.
			name:      "an unknown level passes through",
			spec:      ReasoningSpec{Effort: "ultra"},
			wantWire:  `{"thinkingLevel":"ultra"}`,
			wantNotes: []string{"reasoning.effort: ultra -> thinkingConfig.thinkingLevel: ultra"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			thinking := mapGeminiThinking(tt.spec)
			got := ""
			if thinking != nil {
				encoded, err := json.Marshal(thinking)
				if err != nil {
					t.Fatalf("marshalling thinkingConfig: %v", err)
				}
				got = string(encoded)
			}
			if got != tt.wantWire {
				t.Errorf("thinkingConfig: got %s, want %s", got, tt.wantWire)
			}
			notes := GeminiReasoningNotes(tt.spec)
			if len(notes) != len(tt.wantNotes) {
				t.Fatalf("notes: got %q, want %q", notes, tt.wantNotes)
			}
			for i, want := range tt.wantNotes {
				if notes[i] != want {
					t.Errorf("note %d:\n got %q\nwant %q", i, notes[i], want)
				}
			}
		})
	}
}

// TestGeminiSamplingNote: Google recommends the default temperature on the
// Gemini 3 family and amele still SENDS what the config asked for, so the
// caveat is the only thing standing between a non-default value and a run that
// quietly degrades.
func TestGeminiSamplingNote(t *testing.T) {
	tests := []struct {
		name        string
		temperature *float64
		want        string
	}{
		{"unset", nil, ""},
		{"the recommended default", ptr(1.0), ""},
		{"zero is a non-default value", ptr(0.0), geminiSamplingCaveat},
		{"a non-default value", ptr(0.2), geminiSamplingCaveat},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GeminiSamplingNote(tt.temperature); got != tt.want {
				t.Errorf("GeminiSamplingNote: got %q, want %q", got, tt.want)
			}
		})
	}
}

// geminiSamplingCaveat is asserted verbatim: the wording is what the operator
// reads in `amele explain`, so it is contract, not prose.
const geminiSamplingCaveat = "google recommends the default 1.0 on gemini 3 models; non-default may degrade output"

// gemReply is one scripted HTTP answer: status 0 means 200.
type gemReply struct {
	status int
	body   string
}

// gemStatusServer is gemScriptedServer with a status code per reply, which the
// capability-fallback tests need (the first answer is a 400).
func gemStatusServer(t *testing.T, replies ...gemReply) (*httptest.Server, *[][]byte) {
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
			t.Errorf("unscripted request %d: %s", n+1, body)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		reply := replies[n]
		n++
		if reply.status != 0 && reply.status != http.StatusOK {
			w.WriteHeader(reply.status)
		}
		_, _ = w.Write([]byte(reply.body))
	}))
	t.Cleanup(srv.Close)
	return srv, &recorded
}

// gemSchemaRequest asks for structured output, which is what arms the fallback.
func gemSchemaRequest() Request {
	return Request{
		Model:          "gemini-3-pro",
		Messages:       []Message{{Role: RoleUser, Content: "x"}},
		ResponseFormat: &ResponseFormat{Name: "amele_output", Schema: json.RawMessage(`{"type":"object"}`)},
	}
}

// The 400 the fallback exists for, in both spellings this API uses for the
// field (camelCase on the wire, snake_case in protobuf-JSON error text).
const (
	bodyGemSchemaRejectedCamel = `{"error":{"code":400,"message":"Invalid JSON payload received. Unknown name \"responseJsonSchema\" at 'generation_config': Cannot find field.","status":"INVALID_ARGUMENT"}}`
	bodyGemSchemaRejectedSnake = `{"error":{"code":400,"message":"Invalid JSON payload received. Unknown name \"response_json_schema\" at 'generation_config': Cannot find field.","status":"INVALID_ARGUMENT"}}`
)

// TestGeminiResponseSchemaFallback: a 400 naming the schema field is capability
// discovery, not a transient failure. The call repeats ONCE without the schema,
// costs no retry budget and no backoff sleep, and the response is flagged
// SchemaEnforcementDropped so the validate+retry layer above knows it is now
// the only thing enforcing output.schema.
func TestGeminiResponseSchemaFallback(t *testing.T) {
	spellings := []struct {
		name string
		body string
	}{
		{"the wire spelling", bodyGemSchemaRejectedCamel},
		{"the protobuf spelling", bodyGemSchemaRejectedSnake},
	}
	for _, spelling := range spellings {
		body := spelling.body
		t.Run(spelling.name, func(t *testing.T) {
			srv, recorded := gemStatusServer(t,
				gemReply{status: http.StatusBadRequest, body: body},
				gemReply{body: gemOKBody(`{"ok":true}`)},
			)
			var sleeps int
			client := &GeminiClient{BaseURL: srv.URL, Sleep: noSleep(t, &sleeps)}

			resp, err := client.Chat(context.Background(), gemSchemaRequest())
			if err != nil {
				t.Fatalf("Chat: %v", err)
			}
			if !resp.SchemaEnforcementDropped {
				t.Error("a response produced after the fallback must be flagged SchemaEnforcementDropped")
			}
			if len(*recorded) != 2 {
				t.Fatalf("requests: got %d, want 2 (one fallback, exactly once)", len(*recorded))
			}
			if !bytes.Contains((*recorded)[0], []byte(`"responseJsonSchema"`)) {
				t.Errorf("first request must carry the schema: %s", (*recorded)[0])
			}
			if bytes.Contains((*recorded)[1], []byte(`"responseJsonSchema"`)) {
				t.Errorf("fallback request must drop the schema entirely: %s", (*recorded)[1])
			}
			// The mime type survives: JSON mode alone still gets the model to
			// answer with JSON, which is what the local validate+retry layer
			// then checks against the schema.
			if !bytes.Contains((*recorded)[1], []byte(`"responseMimeType":"application/json"`)) {
				t.Errorf("fallback must keep the JSON mime type: %s", (*recorded)[1])
			}
			if sleeps != 0 {
				t.Errorf("the fallback must not sleep, got %d sleeps", sleeps)
			}
		})
	}
}

// TestGeminiNoSchemaProbeOnSuccess: the fallback costs nothing when the
// endpoint honors the schema - one request, and no false claim that native
// enforcement was dropped. A request with no schema at all is never flagged
// either.
func TestGeminiNoSchemaProbeOnSuccess(t *testing.T) {
	srv, recorded := gemStatusServer(t, gemReply{body: gemOKBody(`{"ok":true}`)})
	client := &GeminiClient{BaseURL: srv.URL, Sleep: failingSleep(t)}
	resp, err := client.Chat(context.Background(), gemSchemaRequest())
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(*recorded) != 1 {
		t.Fatalf("requests: got %d, want 1 (no probe when the schema is accepted)", len(*recorded))
	}
	if resp.SchemaEnforcementDropped {
		t.Error("an accepted schema must not be flagged SchemaEnforcementDropped")
	}

	// A format with no schema is plain JSON mode: nothing native is enforcing
	// output.schema, which the caller must be told BEFORE any 400 - there is no
	// field to probe for.
	jsonModeSrv, jsonModeRecorded := gemStatusServer(t, gemReply{body: gemOKBody(`{"ok":true}`)})
	jsonModeClient := &GeminiClient{BaseURL: jsonModeSrv.URL, Sleep: failingSleep(t)}
	jsonMode, err := jsonModeClient.Chat(context.Background(), Request{
		Model:          "gemini-3-pro",
		Messages:       []Message{{Role: RoleUser, Content: "x"}},
		ResponseFormat: &ResponseFormat{Name: "amele_output"},
	})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if len(*jsonModeRecorded) != 1 {
		t.Fatalf("requests: got %d, want 1 (nothing to probe for)", len(*jsonModeRecorded))
	}
	if !jsonMode.SchemaEnforcementDropped {
		t.Error("json mode without a schema must be flagged SchemaEnforcementDropped")
	}

	plainSrv, _ := gemStatusServer(t, gemReply{body: gemOKBody("hi")})
	plainClient := &GeminiClient{BaseURL: plainSrv.URL, Sleep: failingSleep(t)}
	plain, err := plainClient.Chat(context.Background(), gemUserRequest())
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if plain.SchemaEnforcementDropped {
		t.Error("a schema-less request must not set SchemaEnforcementDropped")
	}
}

// TestGeminiUnrelated400NoFallback: only a 400 that NAMES the schema field is
// capability discovery. Any other rejection stays a hard failure instead of
// being silently downgraded to a schema-less request.
func TestGeminiUnrelated400NoFallback(t *testing.T) {
	srv, recorded := gemStatusServer(t, gemReply{
		status: http.StatusBadRequest,
		body:   gemErrorBody(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.", ""),
	})
	client := &GeminiClient{BaseURL: srv.URL, Sleep: failingSleep(t)}
	if _, err := client.Chat(context.Background(), gemSchemaRequest()); err == nil {
		t.Fatal("expected the 400 to reach the caller")
	}
	if len(*recorded) != 1 {
		t.Errorf("requests: got %d, want 1 (no fallback for an unrelated 400)", len(*recorded))
	}
}

// The 400 bodies the signature table exists for. Each message is the documented
// wording of the failure it describes; the advice strings below are the
// contract this task freezes, so both are asserted verbatim.
const (
	msgGemMissingSignature = "Unable to submit request because Function Call is missing a `thought_signature`. " +
		"Please ensure the thought_signature received in the model response is sent back unmodified."
	msgGemThinkingConflict = "Invalid JSON payload received. Only one of thinkingLevel and thinkingBudget may be set in thinkingConfig."
	msgGemBudgetZero       = "Budget 0 is invalid. This model only supports thinking budgets of at least 128 tokens."
	msgGemCannotDisable    = "Thinking cannot be disabled for this model."
	// The same failure phrased the other way round, which is why the matcher
	// checks two forms rather than one sentence.
	msgGemCannotDisableAlt = "This model reasons by default and you cannot disable thinking for it."
	msgGemUnknownName      = "Invalid JSON payload received. Unknown name \"topK\" at 'generation_config': Cannot find field."
)

const (
	adviceGemSignatureBug   = "amele must echo signatures automatically - this is a bug, please report it with the session log"
	adviceGemThinkingBoth   = "set only one of provider.reasoning.effort or budget_tokens"
	adviceGemCannotDisable  = "this model cannot disable thinking; remove reasoning.effort: none"
	adviceGemUnknownField   = "the gemini API rejects unknown fields; if this key came from provider.params, remove it"
	adviceGemMissingSigTurn = "amele echoes thought signatures automatically - this is a bug; please report it with the session log"
)

// TestGemini400AdviceForKnownSignatures: a recognized 400 keeps the API's own
// message and gains the hint that names the config knob to turn. Diagnostics
// only - the request is not retried, rewritten or downgraded.
func TestGemini400AdviceForKnownSignatures(t *testing.T) {
	tests := []struct {
		name    string
		message string
		advice  string
	}{
		{"a missing thought signature is amele's bug", msgGemMissingSignature, adviceGemSignatureBug},
		{"thinking level and budget together", msgGemThinkingConflict, adviceGemThinkingBoth},
		{"a zero budget on a model that always thinks", msgGemBudgetZero, adviceGemCannotDisable},
		{"the cannot-disable wording", msgGemCannotDisable, adviceGemCannotDisable},
		{"the other cannot-disable wording", msgGemCannotDisableAlt, adviceGemCannotDisable},
		{"an unknown field", msgGemUnknownName, adviceGemUnknownField},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := gemErrorBody(400, "INVALID_ARGUMENT", tt.message, "")
			var calls int
			srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
				calls++
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(body))
			})

			client := &GeminiClient{BaseURL: srv.URL, Sleep: failingSleep(t)}
			_, err := client.Chat(context.Background(), gemUserRequest())
			if err == nil {
				t.Fatal("expected error")
			}
			if !errors.Is(err, ErrProvider) {
				t.Errorf("should wrap ErrProvider: %v", err)
			}
			want := "provider error: status 400: " + body + " — " + tt.advice
			if err.Error() != want {
				t.Errorf("message:\n got %q\nwant %q", err.Error(), want)
			}
			if calls != 1 {
				t.Errorf("400 must not be retried, got %d calls", calls)
			}
		})
	}
}

// TestGeminiUnrecognized400KeepsItsMessage: a 400 nothing in the table
// recognizes reads exactly as it did before the table existed - no invented
// hint pointing at an innocent config key.
func TestGeminiUnrecognized400KeepsItsMessage(t *testing.T) {
	body := gemErrorBody(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.", "")
	srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(body))
	})
	client := &GeminiClient{BaseURL: srv.URL, Sleep: failingSleep(t)}
	_, err := client.Chat(context.Background(), gemUserRequest())
	if err == nil {
		t.Fatal("expected error")
	}
	if strings.Contains(err.Error(), " — ") {
		t.Errorf("unrecognized 400 must carry no advice: %q", err.Error())
	}
}

// TestGeminiMissingThoughtSignatureFinishReason is the OTHER surface of the
// same failure, and the one that would be silent: the API answers 200 with
// finishReason MISSING_THOUGHT_SIGNATURE, so nothing below this client would
// notice the turn failed - a non-empty answer would exit 0 in an unattended
// run. It must become a provider error, like MALFORMED_FUNCTION_CALL.
func TestGeminiMissingThoughtSignatureFinishReason(t *testing.T) {
	srv := geminiServer(t, func(w http.ResponseWriter, _ map[string]any) {
		_, _ = w.Write([]byte(gemCandidateBody(`[{"text":"here is the answer"}]`, "MISSING_THOUGHT_SIGNATURE")))
	})
	client := &GeminiClient{BaseURL: srv.URL, Sleep: failingSleep(t)}
	_, err := client.Chat(context.Background(), gemUserRequest())
	if err == nil {
		t.Fatal("a MISSING_THOUGHT_SIGNATURE turn must not succeed")
	}
	if !errors.Is(err, ErrProvider) {
		t.Errorf("should wrap ErrProvider: %v", err)
	}
	if !strings.Contains(err.Error(), "MISSING_THOUGHT_SIGNATURE") {
		t.Errorf("the message must name the finish reason: %q", err.Error())
	}
	if !strings.Contains(err.Error(), adviceGemMissingSigTurn) {
		t.Errorf("the message must carry the bug-report advice: %q", err.Error())
	}
}
