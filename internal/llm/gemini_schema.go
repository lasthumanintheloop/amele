package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
)

// geminiSchemaKeywords is the ALLOWLIST of JSON Schema keywords the Gemini wire
// accepts inside a FunctionDeclaration's parameters, which is an OpenAPI-3.0
// SUBSET rather than JSON Schema (design doc §"Gemini-specific mechanics" 1).
//
// CONTRACT: it is an allowlist, not a denylist, because this wire answers an
// UNKNOWN key with a hard 400 that names one field and fails the whole request
// - every tool, not just the one with the offending schema. A denylist could
// only remove the keywords someone thought of; the allowlist removes everything
// the subset cannot carry, including keywords a future draft invents and the
// vendor extensions MCP servers ship. The cost is a constraint the model no
// longer sees (a "pattern", say); the alternative is a run that cannot start.
var geminiSchemaKeywords = map[string]bool{
	"type": true, "format": true, "title": true, "description": true,
	"enum": true, "items": true, "properties": true, "required": true,
	"nullable": true, "minimum": true, "maximum": true,
	"minItems": true, "maxItems": true, "anyOf": true, "propertyOrdering": true,
}

// SanitizeGeminiSchema rewrites a JSON Schema document into the subset the
// Gemini wire accepts and reports what it had to remove.
//
// clean is the compacted schema with every unsupported keyword gone,
// recursively, at every schema position (properties values, items, anyOf
// members). Object keys are emitted in sorted order so the request bytes - and
// the wire goldens - are stable. stripped holds one SCHEMA-RELATIVE path per
// removed keyword ("additionalProperties", "properties.path.$ref",
// "anyOf[1].const"), in a deterministic order; the caller composes the tool's
// own prefix for reporting (Task 5's explain).
//
// CONTRACT: nothing is dropped silently - a keyword that leaves the schema is
// always named in stripped. An empty document returns (nil, nil) so the
// declaration carries no parameters key at all, and a document that is not a
// JSON object is passed through compacted: there is no keyword to strip, and
// the wire's own error names it better than a guess here would.
//
// The input is never modified: every value that survives is a copy.
func SanitizeGeminiSchema(raw json.RawMessage) (clean json.RawMessage, stripped []string) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, nil
	}
	return sanitizeGeminiSchema(raw, "", &stripped), stripped
}

// sanitizeGeminiSchema is the recursive worker. path is the position of raw
// inside the document being sanitized ("" at the root) and removals append to
// stripped.
func sanitizeGeminiSchema(raw json.RawMessage, path string, stripped *[]string) json.RawMessage {
	obj, ok := decodeJSONObject(raw)
	if !ok {
		return compactJSON(raw)
	}
	out := make(map[string]json.RawMessage, len(obj))
	// Sorted iteration, not map order: the stripped paths are a report an
	// operator reads and a golden pins, and Go randomizes map iteration.
	for _, key := range slices.Sorted(maps.Keys(obj)) {
		if !geminiSchemaKeywords[key] {
			*stripped = append(*stripped, joinSchemaPath(path, key))
			continue
		}
		switch key {
		case "properties":
			// The KEYS here are property names, not keywords: a property
			// legitimately called "$ref" or "not" must survive untouched while
			// its schema is sanitized like any other.
			out[key] = sanitizeGeminiProperties(obj[key], joinSchemaPath(path, key), stripped)
		case "items", "anyOf":
			out[key] = sanitizeGeminiSchemaList(obj[key], joinSchemaPath(path, key), stripped)
		default:
			// Every other allowed keyword is a leaf as far as this subset goes
			// (no schema hides inside an enum or a required list), so the value
			// travels verbatim - compacted, never re-encoded.
			out[key] = compactJSON(obj[key])
		}
	}
	// Marshalling this map sorts the keys and compacts the values, and cannot
	// fail: every value is either a RawMessage that already decoded or output
	// of this function. The error is discarded deliberately rather than turned
	// into an unreachable branch.
	encoded, _ := json.Marshal(out)
	return encoded
}

// sanitizeGeminiProperties sanitizes each property's schema, keeping the
// property NAMES as they are.
func sanitizeGeminiProperties(raw json.RawMessage, path string, stripped *[]string) json.RawMessage {
	obj, ok := decodeJSONObject(raw)
	if !ok {
		return compactJSON(raw)
	}
	out := make(map[string]json.RawMessage, len(obj))
	for _, name := range slices.Sorted(maps.Keys(obj)) {
		out[name] = sanitizeGeminiSchema(obj[name], joinSchemaPath(path, name), stripped)
	}
	encoded, _ := json.Marshal(out) // cannot fail; see sanitizeGeminiSchema.
	return encoded
}

// sanitizeGeminiSchemaList sanitizes a position that holds either one schema
// (items) or a list of them (anyOf, and the tuple spelling of items).
func sanitizeGeminiSchemaList(raw json.RawMessage, path string, stripped *[]string) json.RawMessage {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '[' {
		return sanitizeGeminiSchema(raw, path, stripped)
	}
	var members []json.RawMessage
	if err := json.Unmarshal(trimmed, &members); err != nil {
		return compactJSON(raw)
	}
	out := make([]json.RawMessage, len(members))
	for i, member := range members {
		out[i] = sanitizeGeminiSchema(member, fmt.Sprintf("%s[%d]", path, i), stripped)
	}
	encoded, _ := json.Marshal(out) // cannot fail; see sanitizeGeminiSchema.
	return encoded
}

// joinSchemaPath appends one segment to a schema-relative path.
func joinSchemaPath(path, segment string) string {
	if path == "" {
		return segment
	}
	return path + "." + segment
}

// decodeJSONObject decodes a raw value as a JSON object, reporting false for
// anything else (a bool, an array, a document too deep or malformed for the
// decoder). Values come back as fresh copies, so nothing here aliases - let
// alone modifies - the caller's bytes.
func decodeJSONObject(raw json.RawMessage) (map[string]json.RawMessage, bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return nil, false
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &obj); err != nil {
		return nil, false
	}
	return obj, true
}

// compactJSON returns raw with insignificant whitespace removed, or a copy of
// raw when it is not valid JSON - a schema this sanitizer cannot read is passed
// through rather than silently replaced, so the wire's own 400 (or the encoder)
// names it instead of the model receiving a tool whose parameters quietly
// vanished.
func compactJSON(raw json.RawMessage) json.RawMessage {
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		return bytes.Clone(raw)
	}
	return buf.Bytes()
}
