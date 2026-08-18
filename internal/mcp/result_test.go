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
			name:     "needs input",
			res:      inputRequiredResult(t),
			want:     "error: server requested interactive input; not supported in headless mode",
			wantKind: tools.OutcomeToolError,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			text, out := RenderResult(tc.res, tc.validator)
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
	big := strings.Repeat("x", MaxResultBytes+100)
	text, out := RenderResult(&sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: big}}}, nil)
	if out.Kind != tools.OutcomeOK {
		t.Errorf("outcome = %v, want OK", out.Kind)
	}
	if !strings.HasSuffix(text, truncationMarker) {
		t.Fatalf("text does not end with the truncation marker: %q", text[len(text)-40:])
	}
	if got := len(text) - len(truncationMarker); got != MaxResultBytes {
		t.Errorf("kept %d bytes, want %d", got, MaxResultBytes)
	}
}

func TestRenderResultCutsOnARuneBoundary(t *testing.T) {
	// Every rune is 2 bytes, so a byte-exact cut would land mid-rune.
	big := strings.Repeat("ü", MaxResultBytes)
	text, _ := RenderResult(&sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: big}}}, nil)
	body := strings.TrimSuffix(text, truncationMarker)
	if !utf8ValidString(body) {
		t.Error("truncated text is not valid UTF-8")
	}
	if len(body) > MaxResultBytes {
		t.Errorf("kept %d bytes, want <= %d", len(body), MaxResultBytes)
	}
}

// utf8ValidString is a thin alias so the intent of the assertion above reads
// clearly at the call site.
func utf8ValidString(s string) bool { return utf8.ValidString(s) }
