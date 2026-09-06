package mcp

import (
	"encoding/json"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/lasthumanintheloop/amele/internal/schema"
	"github.com/lasthumanintheloop/amele/internal/tools"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// inputRequiredResult builds a result whose resultType is "input_required".
// The field is unexported in the SDK, so the only way to reach that state from
// a test is the wire format - which is also how a real server reaches it.
func inputRequiredResult(t *testing.T) *sdk.CallToolResult {
	t.Helper()
	var res sdk.CallToolResult
	if err := json.Unmarshal([]byte(`{"content":[],"resultType":"input_required"}`), &res); err != nil {
		t.Fatalf("building input-required result: %v", err)
	}
	if !res.NeedsInput() {
		t.Fatal("test fixture does not report NeedsInput")
	}
	return &res
}

func mustCompile(t *testing.T, s string) *schema.Validator {
	t.Helper()
	v, err := schema.Compile([]byte(s))
	if err != nil {
		t.Fatalf("compiling schema: %v", err)
	}
	return v
}

func TestRenderResult(t *testing.T) {
	objectSchema := mustCompile(t, `{"type":"object","required":["a"]}`)

	for _, tc := range []struct {
		name      string
		res       *sdk.CallToolResult
		validator *schema.Validator
		want      string
		wantKind  tools.OutcomeKind
	}{
		{
			name: "text content",
			res:  &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: "hello"}}},
			want: "hello",
		},
		{
			name: "parts are joined by newline",
			res: &sdk.CallToolResult{Content: []sdk.Content{
				&sdk.TextContent{Text: "one"},
				&sdk.TextContent{Text: "two"},
			}},
			want: "one\ntwo",
		},
		{
			name:     "is error prefixes and marks the outcome",
			res:      &sdk.CallToolResult{IsError: true, Content: []sdk.Content{&sdk.TextContent{Text: "boom"}}},
			want:     "error: boom",
			wantKind: tools.OutcomeToolError,
		},
		{
			name: "image",
			res:  &sdk.CallToolResult{Content: []sdk.Content{&sdk.ImageContent{MIMEType: "image/png", Data: []byte("abc")}}},
			want: "[image: image/png, 3 bytes]",
		},
		{
			name: "audio by pointer",
			res:  &sdk.CallToolResult{Content: []sdk.Content{&sdk.AudioContent{MIMEType: "audio/wav", Data: []byte("ab")}}},
			want: "[audio: audio/wav, 2 bytes]",
		},
		{
			name: "nil content block",
			res:  &sdk.CallToolResult{Content: []sdk.Content{nil}},
			want: "[unsupported content]",
		},
		{
			name: "missing mime type",
			res:  &sdk.CallToolResult{Content: []sdk.Content{&sdk.ImageContent{Data: []byte("a")}}},
			want: "[image: unknown, 1 bytes]",
		},
		{
			name: "embedded text resource",
			res: &sdk.CallToolResult{Content: []sdk.Content{&sdk.EmbeddedResource{
				Resource: &sdk.ResourceContents{URI: "file:///a.txt", Text: "body"},
			}}},
			want: "file:///a.txt\nbody",
		},
		{
			name: "embedded blob resource",
			res: &sdk.CallToolResult{Content: []sdk.Content{&sdk.EmbeddedResource{
				Resource: &sdk.ResourceContents{URI: "file:///a.bin", MIMEType: "application/octet-stream", Blob: []byte("abcd")},
			}}},
			want: "[resource: file:///a.bin, application/octet-stream, 4 bytes]",
		},
		{
			name: "resource link",
			res:  &sdk.CallToolResult{Content: []sdk.Content{&sdk.ResourceLink{URI: "https://x/y", Name: "y"}}},
			want: "[resource link: https://x/y y]",
		},
		{
			name: "unknown content",
			res:  &sdk.CallToolResult{Content: []sdk.Content{&sdk.EmbeddedResource{}}},
			want: "[unsupported content]",
		},
		{
			name: "empty result",
			res:  &sdk.CallToolResult{},
			want: "(empty result)",
		},
		{
			name: "nil result",
			res:  nil,
			want: "(empty result)",
		},
		{
			name: "structured content wins over content parts",
			res: &sdk.CallToolResult{
				Content:           []sdk.Content{&sdk.TextContent{Text: "ignored"}},
				StructuredContent: map[string]any{"a": 1},
			},
			want: `{"a":1}`,
		},
		{
			name:      "structured content matching the schema",
			res:       &sdk.CallToolResult{StructuredContent: map[string]any{"a": 1}},
			validator: objectSchema,
			want:      `{"a":1}`,
		},
		{
			name:      "structured content violating the schema",
			res:       &sdk.CallToolResult{StructuredContent: map[string]any{"b": 1}},
			validator: objectSchema,
			want:      `error: structured output does not match outputSchema:`,
			wantKind:  tools.OutcomeToolError,
		},
		{
			name: "is error with structured content",
			res: &sdk.CallToolResult{
				IsError:           true,
				StructuredContent: map[string]any{"a": 1},
			},
			want:     `error: {"a":1}`,
			wantKind: tools.OutcomeToolError,
		},
		{
			name:     "needs input",
			res:      inputRequiredResult(t),
			want:     "error: server requested interactive input; not supported in headless mode",
			wantKind: tools.OutcomeToolError,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			text, out := RenderResult(tc.res, tc.validator, 0)
			if tc.name == "structured content violating the schema" {
				if !strings.HasPrefix(text, tc.want) {
					t.Errorf("text = %q, want prefix %q", text, tc.want)
				}
			} else if text != tc.want {
				t.Errorf("text = %q, want %q", text, tc.want)
			}
			if out.Kind != tc.wantKind {
				t.Errorf("outcome = %v, want %v", out.Kind, tc.wantKind)
			}
		})
	}
}

func TestRenderResultCapsSize(t *testing.T) {
	// The cap is a parameter now, so the table drives it: the legacy value
	// (0, meaning DefaultMaxResultBytes) and small explicit caps that make the
	// boundary cases readable without allocating 64 KiB per row.
	cases := []struct {
		name     string
		text     string
		maxBytes int
		wantBody string
		wantCut  bool
	}{
		{
			name:     "zero means the default cap",
			text:     strings.Repeat("x", DefaultMaxResultBytes+100),
			maxBytes: 0,
			wantBody: strings.Repeat("x", DefaultMaxResultBytes),
			wantCut:  true,
		},
		{
			name:     "explicit cap cuts to that many bytes",
			text:     strings.Repeat("x", 20),
			maxBytes: 8,
			wantBody: "xxxxxxxx",
			wantCut:  true,
		},
		{
			name:     "exactly the cap is returned whole and unmarked",
			text:     "xxxxxxxx",
			maxBytes: 8,
			wantBody: "xxxxxxxx",
			wantCut:  false,
		},
		{
			// Five two-byte runes: a byte-exact cut at 5 would split the third,
			// so only two runes (4 bytes) may survive.
			name:     "cut moves back to a rune boundary",
			text:     "ééééé",
			maxBytes: 5,
			wantBody: "éé",
			wantCut:  true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res := &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: tc.text}}}
			text, out := RenderResult(res, nil, tc.maxBytes)
			if out.Kind != tools.OutcomeOK {
				t.Errorf("outcome = %v, want OK", out.Kind)
			}
			if out.Truncated != tc.wantCut {
				t.Errorf("Truncated = %v, want %v", out.Truncated, tc.wantCut)
			}
			if got := strings.HasSuffix(text, tools.TruncationMarker); got != tc.wantCut {
				t.Errorf("marker present = %v, want %v", got, tc.wantCut)
			}
			if body := strings.TrimSuffix(text, tools.TruncationMarker); body != tc.wantBody {
				t.Errorf("body = %d bytes (%q), want %d bytes", len(body), truncateForMsg(body), len(tc.wantBody))
			}
		})
	}
}

// truncateForMsg keeps a failure message readable when the body is a 64 KiB
// wall of x's.
func truncateForMsg(s string) string {
	if len(s) <= 40 {
		return s
	}
	return s[:40] + "..."
}

func TestRenderResultCutsOnARuneBoundary(t *testing.T) {
	// Every rune is 2 bytes, so a byte-exact cut would land mid-rune.
	big := strings.Repeat("ü", DefaultMaxResultBytes)
	text, out := RenderResult(&sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: big}}}, nil, 0)
	body := strings.TrimSuffix(text, tools.TruncationMarker)
	if !utf8ValidString(body) {
		t.Error("truncated text is not valid UTF-8")
	}
	if len(body) > DefaultMaxResultBytes {
		t.Errorf("kept %d bytes, want <= %d", len(body), DefaultMaxResultBytes)
	}
	if !out.Truncated {
		t.Error("Truncated = false, want true")
	}
}

// utf8ValidString is a thin alias so the intent of the assertion above reads
// clearly at the call site.
func utf8ValidString(s string) bool { return utf8.ValidString(s) }
