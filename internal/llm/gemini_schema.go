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
// declaration carries no parameters key at all.
//
// CONTRACT: the allowlist answers unsupported KEYS; two unsupported SHAPES are
// answered as well, because neither could ever be caught by a key filter and
// each is a hard 400 that fails the WHOLE toolset, not just the declaration
// carrying it:
//
//   - a schema POSITION holding something that is not an object (a boolean
//     schema, a stray array, bytes that are not JSON at all) becomes {} - the
//     empty schema - and is reported with a "(non-object schema replaced)"
//     note. Passing it through instead used to fail the request at the encoder
//     or at the wire, which is the one outcome this function exists to prevent.
//   - the JSON-Schema union spelling of type ("type": ["string","null"]) is
//     narrowed to its first non-"null" member, with nullable: true when the
//     union admitted null. This wire's type is ONE OpenAPI type and nullability
//     is its own keyword; a type that narrows to nothing (a bare "null", an
//     empty or unreadable array, a number) is removed entirely.
//
// The notes ride in the same report as the stripped keys and are still PATHS
// plus a fixed phrase - no schema value is ever quoted, which is what lets
// `explain` and the run's warning line print this list for a remote MCP
// server's schemas (see geminiSchemaLines).
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
		// A schema position that is not a Schema object. See the shape half of
		// SanitizeGeminiSchema's contract: the empty schema is the one value
		// this wire is known to accept everywhere a schema is expected, and the
		// note is what keeps the loss visible.
		*stripped = append(*stripped, noteSchemaPath(path, "non-object schema replaced"))
		return json.RawMessage("{}")
	}
	out := make(map[string]json.RawMessage, len(obj))
	// nullable is remembered rather than written immediately: it is decided by
	// the type key and must survive an explicit "nullable" that contradicts it,
	// whichever order the two keys are visited in.
	nullable := false
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
		case "type":
			narrowed, addNullable, keep := narrowGeminiType(obj[key], joinSchemaPath(path, key), stripped)
			if keep {
				out[key] = narrowed
			}
			nullable = nullable || addNullable
		default:
			// Every other allowed keyword is a leaf as far as this subset goes
			// (no schema hides inside an enum or a required list), so the value
			// travels verbatim - compacted, never re-encoded.
			out[key] = compactJSON(obj[key])
		}
	}
	if nullable {
		// Overwrites an explicit nullable: false. The two contradict, and the
		// union that admits null is the more specific statement of the same
		// document - dropping it would silently tighten the model's contract.
		out["nullable"] = json.RawMessage("true")
	}
	// Marshalling this map sorts the keys and compacts the values, and cannot
	// fail: every value is either a RawMessage that already decoded or output
	// of this function. The error is discarded deliberately rather than turned
	// into an unreachable branch.
	encoded, _ := json.Marshal(out)
	return encoded
}

// narrowGeminiType maps a JSON Schema type value onto the single OpenAPI type
// this wire's Schema carries. It reports the value to emit, whether the schema
// must also gain nullable: true, and whether the key survives at all; a removal
// or a rewrite is appended to stripped under path.
//
// The union spelling is the reason this exists: "type": ["string","null"] is
// how JSON Schema says "nullable string", it rides on a key the allowlist
// ALLOWS, and this wire answers the array with a 400 that fails every tool in
// the request. The first non-"null" member is kept because a union amele cannot
// express has to collapse to something, and the first member is the one the
// author wrote first; the loss is reported rather than guessed at silently.
func narrowGeminiType(raw json.RawMessage, path string, stripped *[]string) (value json.RawMessage, nullable, keep bool) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) > 0 && trimmed[0] == '[' {
		var members []string
		if err := json.Unmarshal(trimmed, &members); err != nil {
			*stripped = append(*stripped, noteSchemaPath(path, "type removed"))
			return nil, false, false
		}
		// nullable follows the union's OWN membership, not its length: a
		// ["integer","string"] union loses a type but gains no nullability.
		hasNull := slices.Contains(members, "null")
		for _, member := range members {
			if member == "null" {
				continue
			}
			// Re-encoded rather than sliced out of the input: this is the one
			// place a value is rewritten, and the encoder is what guarantees
			// the result is a well-formed JSON string.
			encoded, err := json.Marshal(member)
			if err != nil {
				break
			}
			*stripped = append(*stripped, noteSchemaPath(path, "type array narrowed"))
			return encoded, hasNull, true
		}
		// Nothing but "null" (or an empty array): there is no type left to be
		// nullable ABOUT, so the keyword goes without one.
		*stripped = append(*stripped, noteSchemaPath(path, "type removed"))
		return nil, false, false
	}
	var single string
	if err := json.Unmarshal(trimmed, &single); err != nil || single == "null" {
		// Either a shape this wire cannot read (a number, an object) or the
		// null type, which has no OpenAPI spelling at all.
		*stripped = append(*stripped, noteSchemaPath(path, "type removed"))
		return nil, false, false
	}
	return compactJSON(raw), false, true
}

// sanitizeGeminiProperties sanitizes each property's schema, keeping the
// property NAMES as they are.
//
// A properties value that is not an object holds no property schema to walk and
// is a 400 on this wire, so it is replaced by the empty map and reported, the
// same answer a schema position gets.
func sanitizeGeminiProperties(raw json.RawMessage, path string, stripped *[]string) json.RawMessage {
	obj, ok := decodeJSONObject(raw)
	if !ok {
		*stripped = append(*stripped, noteSchemaPath(path, "non-object schema replaced"))
		return json.RawMessage("{}")
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

// isEmptyJSONObject reports whether a sanitized schema carries nothing at all.
// The comparison is against the exact bytes because every document this package
// produces is compacted, so "{}" is the only spelling an empty object can have.
func isEmptyJSONObject(raw json.RawMessage) bool {
	return string(raw) == "{}"
}

// noteSchemaPath renders one SHAPE rewrite for the stripped-path report: the
// position, then a fixed phrase saying what happened to it.
//
// The root of a document has no path, so it is named "root" rather than
// reported as an empty string - these lines are read by an operator and quoted
// verbatim by `explain`. The phrase is always a constant from this file: no
// part of the schema's own text may reach a report (see the CONTRACT on
// SanitizeGeminiSchema).
func noteSchemaPath(path, note string) string {
	if path == "" {
		path = "root"
	}
	return path + " (" + note + ")"
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
