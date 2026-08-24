package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
)

// Fake is a scripted Provider for tests. It returns its responses in order
// and records every request it receives. Unit tests never touch the network
// (docs/engineering.md §6); Fake is how the loop and CLI are exercised hermetically.
type Fake struct {
	mu sync.Mutex
	// Responses are returned one per Chat call, in order.
	Responses []Response
	// Errs, when non-nil at the call index, is returned instead of the
	// response, letting tests script transient failures.
	Errs []error
	// Requests records every Request Chat received, whole and unaltered -
	// including the sampling, reasoning and Extra knobs - so tests can assert
	// what the loop and the cmd wiring actually asked the provider for.
	Requests []Request

	calls int
}

// Chat implements Provider by replaying the script.
func (f *Fake) Chat(_ context.Context, req Request) (*Response, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.Requests = append(f.Requests, req)
	idx := f.calls
	f.calls++

	if idx < len(f.Errs) && f.Errs[idx] != nil {
		return nil, f.Errs[idx]
	}
	if idx >= len(f.Responses) {
		// Exhausting the script is always a test bug; fail loudly with
		// the provider sentinel so exit-code mapping stays realistic.
		return nil, fmt.Errorf("%w: fake script exhausted after %d calls", ErrProvider, idx)
	}
	resp := f.Responses[idx]
	return &resp, nil
}

// TextResponse is a convenience constructor for a plain assistant answer.
func TextResponse(text string, usage Usage) Response {
	return Response{
		Message:      Message{Role: RoleAssistant, Content: text},
		Usage:        usage,
		FinishReason: "stop",
	}
}

// ToolCallResponse is a convenience constructor for an assistant turn that
// requests a single tool invocation.
func ToolCallResponse(id, name, args string, usage Usage) Response {
	return Response{
		Message: Message{
			Role:      RoleAssistant,
			ToolCalls: []ToolCall{{ID: id, Name: name, Arguments: args}},
		},
		Usage:        usage,
		FinishReason: "tool_calls",
	}
}

// WithReasoning returns a copy of the response whose assistant message
// carries the given opaque reasoning payload. It composes with the
// constructors above (a reasoning model answers with text or tool calls, and
// either way the carrier rides along) so tests scripting a reasoning provider
// need no third constructor. The bytes are stored as given; the fake never
// re-encodes them, which is what makes byte-identity assertions meaningful.
func (r Response) WithReasoning(reasoning json.RawMessage) Response {
	r.Message.Reasoning = reasoning
	return r
}
