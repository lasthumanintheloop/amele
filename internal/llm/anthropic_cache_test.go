package llm

import (
	"encoding/json"
	"strings"
	"testing"
)

// anCacheAssistantArray is a recorded assistant content array: a signed
// thinking block beside the tool_use that opened the tool turn. It is the
// carrier the loop hands back, and the echo contract forbids touching it - so
// no cache breakpoint may be written into it either.
const anCacheAssistantArray = `[{"type":"thinking","thinking":"read the log first","signature":"c2ln"},` +
	`{"type":"tool_use","id":"toolu_1","name":"fs_read","input":{"path":"app.log"}}]`

// anCacheTools are two tool definitions. Two, not one: the breakpoint belongs
// on the LAST tool, and a single-tool fixture could not tell "last" from "all".
func anCacheTools() []ToolDef {
	return []ToolDef{
		{Name: "fs_read", Description: "read a file", Parameters: json.RawMessage(`{"type":"object"}`)},
		{Name: "fs_list", Description: "list a directory", Parameters: json.RawMessage(`{"type":"object"}`)},
	}
}

// anCacheWireCases are the request shapes prompt caching must render. They
// reuse anWireCase so both golden tests share one harness; every case sets
// PromptCache, since the flag-off bytes are already pinned by anWireCases.
func anCacheWireCases() []anWireCase {
	base := []Message{
		{Role: RoleSystem, Content: "you are a log sentry"},
		{Role: RoleUser, Content: "scan today's log"},
	}
	return []anWireCase{
		{
			// The full shape: a marker on the last tool, on the system block
			// and on the last content block of the last message.
			name:   "tools, system and messages",
			golden: "anthropic-cache-tools-system-messages.json",
			client: AnthropicClient{PromptCache: true},
			req: Request{
				Model:    "claude-opus-5",
				Messages: base,
				Tools:    anCacheTools(),
			},
		},
		{
			// No tools: the prefix breakpoint lands on the system block alone.
			name:   "no tools",
			golden: "anthropic-cache-no-tools.json",
			client: AnthropicClient{PromptCache: true},
			req:    Request{Model: "claude-opus-5", Messages: base},
		},
		{
			// An empty system prompt keeps the key OFF the wire - it must not
			// become an empty text block just because caching is on.
			name:   "empty system",
			golden: "anthropic-cache-empty-system.json",
			client: AnthropicClient{PromptCache: true},
			req: Request{
				Model:    "claude-opus-5",
				Messages: []Message{{Role: RoleUser, Content: "scan today's log"}},
				Tools:    anCacheTools(),
			},
		},
		{
			// A tool turn: the echoed assistant array stays untouched and the
			// moving breakpoint sits on the LAST of the two tool_result blocks
			// that Anthropic requires to share one user message.
			name:   "tool turn",
			golden: "anthropic-cache-tool-turn.json",
			client: AnthropicClient{PromptCache: true},
			req: Request{
				Model: "claude-opus-5",
				Messages: []Message{
					{Role: RoleSystem, Content: "you are a log sentry"},
					{Role: RoleUser, Content: "scan today's log"},
					{Role: RoleAssistant, Reasoning: json.RawMessage(anCacheAssistantArray)},
					{Role: RoleTool, ToolCallID: "toolu_1", Content: "ERROR disk full"},
					{Role: RoleTool, ToolCallID: "toolu_2", Content: "app.log\nweb.log"},
				},
				Tools: anCacheTools(),
			},
		},
	}
}

// TestAnthropicCacheToWireGolden pins the cached request body byte for byte:
// which blocks carry cache_control is the whole contract of this feature.
func TestAnthropicCacheToWireGolden(t *testing.T) {
	for _, tc := range anCacheWireCases() {
		t.Run(tc.name, func(t *testing.T) {
			client := tc.client
			wire, fields := client.toWire(tc.req)
			got, err := encodeBody(wire, fields)
			if err != nil {
				t.Fatalf("encodeBody: %v", err)
			}
			checkWireGolden(t, got, tc.golden)
		})
	}
}

// TestAnthropicCacheOffByDefault is the no-regression guard: a client that was
// never told about caching sends what it sent before, so the field's zero value
// must be false and no request may grow a cache_control key.
func TestAnthropicCacheOffByDefault(t *testing.T) {
	if (AnthropicClient{}).PromptCache {
		t.Fatal("PromptCache zero value is true; it must default to off so untouched configs keep today's bytes")
	}
	for _, tc := range append(anWireCases(), anCacheWireCases()...) {
		t.Run(tc.name, func(t *testing.T) {
			client := tc.client
			client.PromptCache = false
			wire, fields := client.toWire(tc.req)
			got, err := encodeBody(wire, fields)
			if err != nil {
				t.Fatalf("encodeBody: %v", err)
			}
			if strings.Contains(string(got), "cache_control") {
				t.Errorf("caching is off but the body carries cache_control: %s", got)
			}
			// The uncached system prompt is a bare JSON string, not a block
			// array: the type change behind this feature must not leak.
			var parsed struct {
				System json.RawMessage `json:"system"`
			}
			if err := json.Unmarshal(got, &parsed); err != nil {
				t.Fatalf("decoding body: %v", err)
			}
			if len(parsed.System) > 0 && parsed.System[0] != '"' {
				t.Errorf("uncached system is not a JSON string: %s", parsed.System)
			}
		})
	}
}

// TestAnthropicCacheSystemKeyOmitted: an empty system prompt omits the key on
// BOTH sides of the flag. Anthropic rejects a blank text block, and the pointer
// that makes the string/array switch possible is exactly the thing that could
// silently start sending "system":"" or [{"text":""}].
func TestAnthropicCacheSystemKeyOmitted(t *testing.T) {
	for _, promptCache := range []bool{false, true} {
		t.Run(map[bool]string{false: "caching off", true: "caching on"}[promptCache], func(t *testing.T) {
			client := AnthropicClient{PromptCache: promptCache}
			wire, fields := client.toWire(Request{
				Model:    "claude-opus-5",
				Messages: []Message{{Role: RoleUser, Content: "scan today's log"}},
			})
			got, err := encodeBody(wire, fields)
			if err != nil {
				t.Fatalf("encodeBody: %v", err)
			}
			if strings.Contains(string(got), `"system"`) {
				t.Errorf("empty system prompt still wrote a system key: %s", got)
			}
		})
	}
}

// TestAnthropicCacheBreakpointBudget: the API allows at most 4 cache_control
// breakpoints per request and answers a fifth with a 400. amele places at most
// three (tools, system, last message), so no reachable request can overrun the
// budget - this pins that the placement stays a fixed set and never a per-item
// loop.
func TestAnthropicCacheBreakpointBudget(t *testing.T) {
	for _, tc := range anCacheWireCases() {
		t.Run(tc.name, func(t *testing.T) {
			client := tc.client
			wire, fields := client.toWire(tc.req)
			got, err := encodeBody(wire, fields)
			if err != nil {
				t.Fatalf("encodeBody: %v", err)
			}
			if n := strings.Count(string(got), `"cache_control"`); n > 3 {
				t.Errorf("placed %d breakpoints, want at most 3: %s", n, got)
			}
		})
	}
}

// TestAnthropicCacheSkipsEchoedAssistantArray: an echoed content array is
// signed, and inserting a cache_control key into it would be a 400. The loop
// never sends a history ending in an assistant turn, so this is the defensive
// branch - and a defensive branch nobody tests is a branch that rots.
func TestAnthropicCacheSkipsEchoedAssistantArray(t *testing.T) {
	client := AnthropicClient{PromptCache: true}
	wire, fields := client.toWire(Request{
		Model: "claude-opus-5",
		Messages: []Message{
			{Role: RoleSystem, Content: "you are a log sentry"},
			{Role: RoleUser, Content: "scan today's log"},
			{Role: RoleAssistant, Reasoning: json.RawMessage(anCacheAssistantArray)},
		},
	})
	got, err := encodeBody(wire, fields)
	if err != nil {
		t.Fatalf("encodeBody: %v", err)
	}
	if !strings.Contains(string(got), anCacheAssistantArray) {
		t.Errorf("echoed assistant array is not byte-identical: %s", got)
	}
	// Only the system breakpoint survives: no tools, and the last message is
	// the untouchable echo.
	if n := strings.Count(string(got), `"cache_control"`); n != 1 {
		t.Errorf("placed %d breakpoints, want 1 (system only): %s", n, got)
	}
}
