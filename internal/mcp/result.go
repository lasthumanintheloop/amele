package mcp

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/lasthumanintheloop/amele/internal/schema"
	"github.com/lasthumanintheloop/amele/internal/tools"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

const (
	// DefaultMaxResultBytes caps how much of a tool result the model ever sees
	// when the caller names no cap of its own. It is the same cap subprocess
	// stdout gets (tools.DefaultMaxOutputBytes): a server the operator does not
	// control must not be able to spend the whole context window in one call.
	// CONTRACT: the returned text never exceeds the cap + the marker.
	DefaultMaxResultBytes = 64 * 1024
	// emptyResultText is what a tool that returned nothing shows the model. An
	// explicit phrase beats an empty string, which reads as a broken harness.
	emptyResultText = "(empty result)"
	// errorPrefix marks a result the server itself flagged as a failure.
	errorPrefix = "error: "
	// unknownMIME stands in when a server omits the media type.
	unknownMIME = "unknown"
	// unsupportedText describes a content block this version cannot render.
	unsupportedText = "[unsupported content]"
)

// RenderResult converts one CallToolResult into the text the model sees and
// the outcome the session log records.
//
// validator is the tool's compiled outputSchema, or nil when the server did
// not publish one; it is compiled once at discovery rather than per call.
//
// The rules, in order:
//   - a nil result, or one with no content at all, renders as "(empty result)";
//   - NeedsInput (the server wants interactive input) is a tool error: amele is
//     headless, so there is nobody to answer and the call is not retried;
//   - StructuredContent, when present, IS the result - it is marshaled compact
//     and the unstructured Content parts are ignored. If validator is non-nil
//     and the value does not satisfy the schema, that is a tool error carrying
//     the validator's feedback, so the model can correct its next call;
//   - otherwise the content parts are rendered one per line: text verbatim,
//     binary blobs as a one-line placeholder naming the media type and size
//     (amele passes no image or audio bytes to the model), resources by URI;
//   - IsError prefixes the text with "error: " and yields OutcomeToolError.
//
// maxBytes is how many bytes of result text the model may read; <= 0 means
// DefaultMaxResultBytes, so a zero-valued caller still gets the legacy cap.
// A result that was cut carries tools.TruncationMarker and reports
// Outcome.Truncated, which is what the session log records.
//
// SECURITY: the text is untrusted server data (docs/threat-model.md S9). It is
// neither parsed nor executed here, only capped at maxBytes on a rune
// boundary so a hostile server cannot flood the context window.
//
// RenderResult is pure: it performs no I/O and no retries.
func RenderResult(res *sdk.CallToolResult, validator *schema.Validator, maxBytes int) (text string, out tools.Outcome) {
	if res == nil {
		return emptyResultText, tools.Outcome{}
	}
	if res.NeedsInput() {
		return errorPrefix + "server requested interactive input; not supported in headless mode",
			tools.Outcome{Kind: tools.OutcomeToolError}
	}

	// One cap for every exit below: an error body is untrusted server text too,
	// so it gets bounded exactly like a successful one.
	limit := maxBytes
	if limit <= 0 {
		limit = DefaultMaxResultBytes
	}

	body, ok := renderBody(res, validator)
	if !ok {
		// body already reads as an error message; do not prefix it twice.
		text, cut := tools.CapText(body, limit)
		return text, tools.Outcome{Kind: tools.OutcomeToolError, Truncated: cut}
	}
	if res.IsError {
		// The prefix applies to structured content too: an error result is for
		// the model to read, not for a caller to parse, so the failure must be
		// visible even when the body happens to be JSON.
		text, cut := tools.CapText(errorPrefix+body, limit)
		return text, tools.Outcome{Kind: tools.OutcomeToolError, Truncated: cut}
	}
	text, cut := tools.CapText(body, limit)
	return text, tools.Outcome{Truncated: cut}
}

// renderBody produces the un-prefixed, un-capped text of a result. ok is false
// when the returned text is itself an error message (already prefixed).
func renderBody(res *sdk.CallToolResult, validator *schema.Validator) (body string, ok bool) {
	if res.StructuredContent != nil {
		return renderStructured(res.StructuredContent, validator)
	}
	if len(res.Content) == 0 {
		return emptyResultText, true
	}
	parts := make([]string, 0, len(res.Content))
	for _, c := range res.Content {
		parts = append(parts, renderContent(c))
	}
	return strings.Join(parts, "\n"), true
}

// renderStructured marshals structuredContent compactly and, when the tool
// published an output schema, checks it against that schema.
func renderStructured(value any, validator *schema.Validator) (body string, ok bool) {
	encoded, err := json.Marshal(value)
	if err != nil {
		// Defensive: the value came off the wire as JSON, so re-encoding it
		// should not fail. If it somehow does, say so rather than dropping the
		// result silently.
		return errorPrefix + "structured output could not be encoded: " + err.Error(), false
	}
	text := string(encoded)
	if validator == nil {
		return text, true
	}
	canonical, feedback, valid := validator.Validate(text)
	if !valid {
		return errorPrefix + "structured output does not match outputSchema: " + feedback, false
	}
	return canonical, true
}

// renderContent renders one content block. Unknown blocks are named rather
// than dropped so the model can tell that something was there.
//
// Every SDK content type satisfies sdk.Content only in its pointer form (the
// interface's unexported fromWire method has a pointer receiver), so the
// pointer cases below are exhaustive for SDK v1.7.0; anything else - a future
// content type, or a nil interface - falls through to the default.
func renderContent(c sdk.Content) string {
	switch v := c.(type) {
	case *sdk.TextContent:
		return v.Text
	case *sdk.ImageContent:
		return blobLine("image", v.MIMEType, len(v.Data))
	case *sdk.AudioContent:
		return blobLine("audio", v.MIMEType, len(v.Data))
	case *sdk.EmbeddedResource:
		return renderResource(v.Resource)
	case *sdk.ResourceLink:
		return fmt.Sprintf("[resource link: %s %s]", v.URI, v.Name)
	default:
		return unsupportedText
	}
}

// renderResource renders an embedded resource: its text if it has any (the
// URI first, so the model knows what it is reading), otherwise a placeholder.
func renderResource(r *sdk.ResourceContents) string {
	if r == nil {
		return unsupportedText
	}
	if r.Text != "" {
		return r.URI + "\n" + r.Text
	}
	return fmt.Sprintf("[resource: %s, %s, %d bytes]", r.URI, mimeOrUnknown(r.MIMEType), len(r.Blob))
}

// blobLine describes binary content the model never receives.
func blobLine(kind, mime string, n int) string {
	return fmt.Sprintf("[%s: %s, %d bytes]", kind, mimeOrUnknown(mime), n)
}

// mimeOrUnknown keeps a missing media type from rendering as an empty gap.
func mimeOrUnknown(mime string) string {
	if mime == "" {
		return unknownMIME
	}
	return mime
}
