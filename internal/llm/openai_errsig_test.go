package llm

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"
)

// The fixture bodies below are the provider replies these heuristics exist for.
// The gpt-5.6 message is quoted verbatim from
// docs/superpowers/specs/2026-08-24-provider-dialects-research.md
// §"Load-bearing quirks" #1; the sampling and cap messages are the OpenAI
// "Unsupported value:" / "Unsupported parameter:" families the same document
// and the design doc §"Error-signature detection" name.
const (
	body56ToolsReasoning = `{"error":{"message":"Function tools with reasoning_effort are not supported for gpt-5.6-sol in /v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort to 'none'.","type":"invalid_request_error","param":"reasoning_effort","code":"unsupported_value"}}`

	bodyTemperatureUnsupportedValue = `{"error":{"message":"Unsupported value: 'temperature' does not support 0.7 with this model. Only the default (1) value is supported.","type":"invalid_request_error","param":"temperature","code":"unsupported_value"}}`

	bodyTemperatureNotSupported = `{"error":{"message":"Invalid request: 'temperature' is not supported with this model. The K-series uses a fixed temperature of 1.0.","type":"invalid_request_error","param":"temperature"}}`

	bodyWrongOutputCapField = `{"error":{"message":"Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.","type":"invalid_request_error","param":"max_tokens","code":"unsupported_parameter"}}`

	bodyUnrelated400 = `{"error": {"message": "bad model"}}`
)

// The advice strings are the contract this task freezes: they name the config
// key to turn, so they are asserted verbatim rather than by keyword.
const (
	adviceReasoningEffortNone = "set provider.reasoning.effort: none for this model on chat/completions, or use a different model"
	adviceNoSampling          = "this model rejects non-default sampling; remove provider.temperature/top_p"
	adviceCapField            = "this model requires max_completion_tokens; set provider.dialect to a dialect that maps it (openai/groq/kimi)"
)

func TestChat400AdviceForKnownSignatures(t *testing.T) {
	tests := []struct {
		name   string
		body   string
		advice string
	}{
		{"gpt-5.6 function tools with reasoning_effort", body56ToolsReasoning, adviceReasoningEffortNone},
		{"sampling rejected (unsupported value)", bodyTemperatureUnsupportedValue, adviceNoSampling},
		{"sampling rejected (not supported)", bodyTemperatureNotSupported, adviceNoSampling},
		{"wrong output cap field", bodyWrongOutputCapField, adviceCapField},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls int
			srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
				calls++
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(tt.body))
			})

			client := &OpenAIClient{BaseURL: srv.URL + "/v1"}
			_, err := client.Chat(context.Background(), Request{Model: "m"})
			if err == nil {
				t.Fatal("expected error")
			}
			if !errors.Is(err, ErrProvider) {
				t.Errorf("should wrap ErrProvider: %v", err)
			}
			// The snippet stays in the message; the advice is appended to it.
			want := "provider error: status 400: " + tt.body + " — " + tt.advice
			if err.Error() != want {
				t.Errorf("message:\n got %q\nwant %q", err.Error(), want)
			}
			// Diagnostics only: a recognized 400 is still a single, final call.
			if calls != 1 {
				t.Errorf("400 must not be retried, got %d calls", calls)
			}
			var se *statusError
			if !errors.As(err, &se) {
				t.Fatalf("typed status error must survive: %v", err)
			}
		})
	}
}

// An unrelated 400 keeps today's message byte for byte: the table may only add
// text to failures it actually recognizes.
func TestChat400UnrelatedMessageUnchanged(t *testing.T) {
	srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(bodyUnrelated400))
	})

	client := &OpenAIClient{BaseURL: srv.URL + "/v1"}
	_, err := client.Chat(context.Background(), Request{Model: "m"})
	if err == nil {
		t.Fatal("expected error")
	}
	want := "provider error: status 400: " + bodyUnrelated400
	if err.Error() != want {
		t.Errorf("message:\n got %q\nwant %q", err.Error(), want)
	}
}

// A retryable status carrying a signature body gets no advice: these are
// diagnostics for a request the provider will refuse every time, not for a
// server that happens to be unhappy right now.
func TestChatAdviceOnlyOnBadRequest(t *testing.T) {
	var calls int
	srv := chatServer(t, func(w http.ResponseWriter, _ map[string]any) {
		calls++
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(bodyTemperatureUnsupportedValue))
	})

	client := &OpenAIClient{BaseURL: srv.URL + "/v1", MaxAttempts: 1}
	_, err := client.Chat(context.Background(), Request{Model: "m"})
	if err == nil {
		t.Fatal("expected error")
	}
	if strings.Contains(err.Error(), adviceNoSampling) {
		t.Errorf("429 must not carry 400 advice: %v", err)
	}
	if calls != 1 {
		t.Errorf("MaxAttempts 1 means one call, got %d", calls)
	}
}

// The table is ordered: a body naming both max_tokens and max_completion_tokens
// reads as the cap-field mistake even though it also says "is not supported".
func TestChat400AdviceIsFirstMatch(t *testing.T) {
	se := &statusError{code: http.StatusBadRequest, snippet: bodyWrongOutputCapField}
	if got := adviceFor(errorSignatures, se); got != adviceCapField {
		t.Errorf("advice: got %q, want %q", got, adviceCapField)
	}
}
