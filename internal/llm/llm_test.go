package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"reflect"
	"testing"
)

// TestFakeCarriesReasoningAndRequestFields pins the neutral request/message
// surface the dialect-aware clients are built on: the sampling and reasoning
// knobs reach the provider unchanged, and an assistant message's opaque
// reasoning payload survives a full turn round-trip byte-for-byte. The
// byte-identity assertion is the load-bearing one - providers that sign
// (Anthropic) or hash-check (DeepSeek) their reasoning reject an altered echo,
// so anything in this package normalizing the carrier is a bug.
func TestFakeCarriesReasoningAndRequestFields(t *testing.T) {
	// A payload with signature-ish content and non-alphabetical key order: a
	// re-serialization anywhere in the path would reorder or reformat it.
	reasoning := json.RawMessage(`{"blocks":[{"type":"thinking","signature":"sig-abc==","text":"step 1"}],"id":"r1"}`)

	fake := &Fake{Responses: []Response{
		ToolCallResponse("call-1", "fs_read", `{"path":"a.txt"}`, Usage{InputTokens: 10, OutputTokens: 5}).
			WithReasoning(reasoning),
		TextResponse("done", Usage{InputTokens: 20, OutputTokens: 3}),
	}}

	temperature, topP := 0.2, 0.9
	req := Request{
		Model:           "reasoner-1",
		Messages:        []Message{{Role: RoleUser, Content: "read a.txt"}},
		MaxOutputTokens: 65536,
		Reasoning:       &ReasoningSpec{Effort: "high", BudgetTokens: 8192},
		Temperature:     &temperature,
		TopP:            &topP,
		Extra:           map[string]json.RawMessage{"verbosity": json.RawMessage(`"low"`)},
	}

	ctx := context.Background()
	resp, err := fake.Chat(ctx, req)
	if err != nil {
		t.Fatalf("first Chat: %v", err)
	}
	if !bytes.Equal(resp.Message.Reasoning, reasoning) {
		t.Fatalf("response carrier mangled:\n got %s\nwant %s", resp.Message.Reasoning, reasoning)
	}

	// Second turn, built the way the loop builds it: the assistant message is
	// appended to history unchanged and the tool result follows it.
	next := req
	next.Messages = append(append([]Message{}, req.Messages...),
		resp.Message,
		Message{Role: RoleTool, ToolCallID: "call-1", Content: "file body"},
	)
	if _, err := fake.Chat(ctx, next); err != nil {
		t.Fatalf("second Chat: %v", err)
	}

	if len(fake.Requests) != 2 {
		t.Fatalf("recorded %d requests, want 2", len(fake.Requests))
	}
	history := fake.Requests[1].Messages
	if len(history) != 3 {
		t.Fatalf("history has %d messages, want 3", len(history))
	}
	if got := history[1].Reasoning; string(got) != string(reasoning) {
		t.Errorf("echoed carrier not byte-identical:\n got %s\nwant %s", got, reasoning)
	}
	// A message that never carried reasoning must stay empty: an absent
	// carrier is absent, never an empty JSON document.
	if got := history[2].Reasoning; got != nil {
		t.Errorf("tool message grew a carrier: %s", got)
	}

	assertRecordedKnobs(t, fake.Requests[0])
}

// assertRecordedKnobs checks that the fake recorded every neutral knob of the
// request it was handed, which is how later tasks assert what the cmd wiring
// and the clients ask a provider for.
func assertRecordedKnobs(t *testing.T, recorded Request) {
	t.Helper()

	if recorded.Temperature == nil || recorded.TopP == nil || recorded.Reasoning == nil {
		t.Fatalf("recorded request lost a pointer knob: %+v", recorded)
	}
	tests := []struct {
		name string
		got  any
		want any
	}{
		{"max_output_tokens", recorded.MaxOutputTokens, 65536},
		{"temperature", *recorded.Temperature, 0.2},
		{"top_p", *recorded.TopP, 0.9},
		{"reasoning", *recorded.Reasoning, ReasoningSpec{Effort: "high", BudgetTokens: 8192}},
		{"extra", recorded.Extra, map[string]json.RawMessage{"verbosity": json.RawMessage(`"low"`)}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if !reflect.DeepEqual(tc.got, tc.want) {
				t.Errorf("recorded %s = %#v, want %#v", tc.name, tc.got, tc.want)
			}
		})
	}
}

// TestZeroRequestCarriesNoKnobs pins the opt-in rule: a Request that mentions
// none of the new fields must leave them nil/zero, because that is exactly
// what the clients read as "send no such wire field" - a config that stays
// silent must not start overriding provider defaults.
func TestZeroRequestCarriesNoKnobs(t *testing.T) {
	bare := Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "hi"}}}

	if bare.Reasoning != nil {
		t.Errorf("Reasoning = %+v, want nil", bare.Reasoning)
	}
	if bare.Temperature != nil || bare.TopP != nil {
		t.Errorf("sampling knobs set: temperature=%v top_p=%v", bare.Temperature, bare.TopP)
	}
	if bare.Extra != nil {
		t.Errorf("Extra = %v, want nil", bare.Extra)
	}
	if bare.MaxOutputTokens != 0 {
		t.Errorf("MaxOutputTokens = %d, want 0", bare.MaxOutputTokens)
	}
	if bare.Messages[0].Reasoning != nil {
		t.Errorf("message carrier = %s, want nil", bare.Messages[0].Reasoning)
	}
}
