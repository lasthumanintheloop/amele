package llm

import (
	"encoding/json"
	"slices"
	"testing"
)

// fsWriteSchema is the fs_write tool's real parameter schema, copied verbatim
// from internal/tools/fs.go (llm cannot import tools - tools imports llm). It
// is the fixture that matters most: amele's own builtin ships a keyword this
// wire rejects, so the sanitizer is load-bearing for the default tool set, and
// the indentation proves the output is compacted rather than echoed.
const fsWriteSchema = `{
			"type": "object",
			"properties": {
				"path": {"type": "string", "description": "Relative file path"},
				"content": {"type": "string", "description": "Full file content"}
			},
			"required": ["path", "content"],
			"additionalProperties": false
		}`

// TestSanitizeGeminiSchema walks the keyword policy: what survives onto the
// OpenAPI-3.0 subset this wire accepts, what is removed, and the PATH each
// removal is reported under.
//
// CONTRACT: nothing is dropped silently - every key that leaves the schema
// appears in the returned list, so `explain` can name it (Task 5).
func TestSanitizeGeminiSchema(t *testing.T) {
	tests := []struct {
		name         string
		raw          string
		want         string
		wantStripped []string
	}{
		{
			// The keep-list survives, values untouched, keys sorted so the
			// wire goldens are byte-stable.
			name: "supported keywords survive",
			raw: `{"type":"object","title":"t","description":"d","nullable":true,
				"properties":{"n":{"type":"integer","format":"int32","minimum":1,"maximum":9},
				"tags":{"type":"array","items":{"type":"string","enum":["a","b"]},"minItems":1,"maxItems":3}},
				"required":["n"],"propertyOrdering":["n","tags"]}`,
			want: `{"description":"d","nullable":true,"properties":{"n":{"format":"int32","maximum":9,"minimum":1,"type":"integer"},` +
				`"tags":{"items":{"enum":["a","b"],"type":"string"},"maxItems":3,"minItems":1,"type":"array"}},` +
				`"propertyOrdering":["n","tags"],"required":["n"],"title":"t","type":"object"}`,
		},
		{
			// Every keyword the design names, at the top level, in one pass.
			name: "every unsupported keyword is stripped and reported",
			raw: `{"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"x","$anchor":"a",
				"$defs":{"d":{"type":"string"}},"$ref":"#/$defs/d","additionalProperties":false,
				"oneOf":[{"type":"string"}],"allOf":[{"type":"string"}],"not":{"type":"string"},
				"patternProperties":{"^x":{"type":"string"}},"dependencies":{"a":["b"]},
				"const":"c","examples":["e"],"default":"d","type":"object"}`,
			want: `{"type":"object"}`,
			wantStripped: []string{
				"$anchor", "$defs", "$id", "$ref", "$schema", "additionalProperties",
				"allOf", "const", "default", "dependencies", "examples", "not",
				"oneOf", "patternProperties",
			},
		},
		{
			// The allowlist is what makes the guarantee absolute: this wire
			// 400s on ANY keyword outside the subset, including ones no
			// strip-list could have anticipated.
			name:         "an unknown keyword is stripped too",
			raw:          `{"type":"string","pattern":"^a","x-vendor":1}`,
			want:         `{"type":"string"}`,
			wantStripped: []string{"pattern", "x-vendor"},
		},
		{
			name: "nested properties report their path",
			raw: `{"type":"object","properties":{"path":{"type":"string","additionalProperties":false},
				"opts":{"type":"object","properties":{"deep":{"type":"string","$ref":"#/x"}}}}}`,
			want: `{"properties":{"opts":{"properties":{"deep":{"type":"string"}},"type":"object"},` +
				`"path":{"type":"string"}},"type":"object"}`,
			wantStripped: []string{
				"properties.opts.properties.deep.$ref",
				"properties.path.additionalProperties",
			},
		},
		{
			// Array-valued schema positions recurse element by element and
			// the index rides in the path.
			name: "arrays recurse with indexed paths",
			raw: `{"anyOf":[{"type":"string","const":"a"},{"type":"array","items":{"type":"string","$schema":"x"}}],
				"items":{"type":"string","default":"d"}}`,
			want:         `{"anyOf":[{"type":"string"},{"items":{"type":"string"},"type":"array"}],"items":{"type":"string"}}`,
			wantStripped: []string{"anyOf[0].const", "anyOf[1].items.$schema", "items.default"},
		},
		{
			// A PROPERTY NAME is data, not a keyword: a property called
			// "$ref" must survive while the keyword inside it is stripped.
			name:         "a property named like a keyword is kept",
			raw:          `{"type":"object","properties":{"$ref":{"type":"string","const":"x"},"not":{"type":"boolean"}}}`,
			want:         `{"properties":{"$ref":{"type":"string"},"not":{"type":"boolean"}},"type":"object"}`,
			wantStripped: []string{"properties.$ref.const"},
		},
		{
			// amele's own builtin: the default tool set does not reach this
			// wire without the sanitizer.
			name: "the fs_write builtin schema",
			raw:  fsWriteSchema,
			want: `{"properties":{"content":{"description":"Full file content","type":"string"},` +
				`"path":{"description":"Relative file path","type":"string"}},"required":["path","content"],"type":"object"}`,
			wantStripped: []string{"additionalProperties"},
		},
		{
			// A schema of nothing but unsupported keywords empties out to {}
			// rather than disappearing: this function's job is to remove
			// keywords, not to decide what an emptied schema means. Whether a
			// type-less Schema is a shape the service accepts is NOT settled
			// here - the caller decides what to do with an empty root, and
			// applyTools drops the parameters key entirely rather than betting
			// on it (unverified until the Task 6 live smoke).
			name:         "a schema emptied by the sanitizer stays an object",
			raw:          `{"$ref":"#/$defs/x"}`,
			want:         `{}`,
			wantStripped: []string{"$ref"},
		},
		{
			// Not an object, so there is no keyword to strip - but the wire
			// wants a Schema object here, and one bad declaration 400s the
			// WHOLE toolset. It becomes an empty schema, and the report says so.
			name:         "a non-object schema becomes an empty schema",
			raw:          `[ 1, 2 ]`,
			want:         `{}`,
			wantStripped: []string{"root (non-object schema replaced)"},
		},
		{
			// Not JSON at all: same answer. Passing it through used to fail the
			// whole request at the encoder, which is the one outcome this
			// sanitizer exists to prevent.
			name:         "an unparseable schema becomes an empty schema",
			raw:          `{"broken":`,
			want:         `{}`,
			wantStripped: []string{"root (non-object schema replaced)"},
		},
		{
			// Every schema position holds a Schema object after this function
			// runs, including the ones the input filled with something else.
			name: "unreadable schema positions become empty schemas",
			raw:  `{"properties":"nonsense","items":true,"anyOf":[1,"x"]}`,
			want: `{"anyOf":[{},{}],"items":{},"properties":{}}`,
			wantStripped: []string{
				"anyOf[0] (non-object schema replaced)",
				"anyOf[1] (non-object schema replaced)",
				"items (non-object schema replaced)",
				"properties (non-object schema replaced)",
			},
		},
		{
			// The JSON-Schema nullable idiom. The value is an ARRAY on an
			// ALLOWED key, so the keyword allowlist never sees it - and this
			// wire's type is a single OpenAPI type, with nullability spelled by
			// its own keyword.
			name:         "a nullable type union becomes a type plus nullable",
			raw:          `{"type":["string","null"],"description":"d"}`,
			want:         `{"description":"d","nullable":true,"type":"string"}`,
			wantStripped: []string{"type (type array narrowed)"},
		},
		{
			// A union of two real types keeps the first and gains NO nullable:
			// the constraint the model loses is reported, not invented.
			name:         "a type union without null narrows to its first member",
			raw:          `{"type":["integer","string"]}`,
			want:         `{"type":"integer"}`,
			wantStripped: []string{"type (type array narrowed)"},
		},
		{
			// An explicit nullable: false loses to a union that says null is
			// allowed - the two contradict, and the union is the one the
			// operator (or the MCP server) wrote last.
			name:         "the narrowed union wins over a contradicting nullable",
			raw:          `{"type":["string","null"],"nullable":false}`,
			want:         `{"nullable":true,"type":"string"}`,
			wantStripped: []string{"type (type array narrowed)"},
		},
		{
			// "the value is null" has no OpenAPI spelling at all, so the
			// keyword goes rather than travelling as a type this wire rejects.
			name:         "a null type is removed",
			raw:          `{"type":"null","title":"t"}`,
			want:         `{"title":"t"}`,
			wantStripped: []string{"type (type removed)"},
		},
		{
			// Nothing usable survives the union either.
			name:         "a union of nothing but null is removed",
			raw:          `{"type":["null"]}`,
			want:         `{}`,
			wantStripped: []string{"type (type removed)"},
		},
		{
			// A type that is not a string and not an array of them is a shape
			// this wire cannot read; it is removed rather than forwarded.
			name:         "a type that is neither a string nor an array is removed",
			raw:          `{"type":7}`,
			want:         `{}`,
			wantStripped: []string{"type (type removed)"},
		},
		{
			// The MCP-style fixture: both shapes in one declaration, nested,
			// next to an ordinary stripped keyword. This is the schema that
			// used to 400 every OTHER tool in the request too.
			name: "an mcp-style schema carrying both shapes",
			raw: `{"type":"object","additionalProperties":false,
				"properties":{"path":{"type":["string","null"],"description":"d"},
				"deep":true,"mode":{"type":"null"},
				"tags":{"type":"array","items":{"type":["string","null"]}}}}`,
			want: `{"properties":{"deep":{},"mode":{},` +
				`"path":{"description":"d","nullable":true,"type":"string"},` +
				`"tags":{"items":{"nullable":true,"type":"string"},"type":"array"}},"type":"object"}`,
			wantStripped: []string{
				"additionalProperties",
				"properties.deep (non-object schema replaced)",
				"properties.mode.type (type removed)",
				"properties.path.type (type array narrowed)",
				"properties.tags.items.type (type array narrowed)",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := json.RawMessage(tt.raw)
			before := string(raw)

			got, stripped := SanitizeGeminiSchema(raw)
			if string(got) != tt.want {
				t.Errorf("clean schema:\ngot:  %s\nwant: %s", got, tt.want)
			}
			if !slices.Equal(stripped, tt.wantStripped) {
				t.Errorf("stripped paths:\ngot:  %v\nwant: %v", stripped, tt.wantStripped)
			}
			if string(raw) != before {
				t.Errorf("input was mutated:\ngot:  %s\nwant: %s", raw, before)
			}
			// Determinism: goldens and explain output depend on a stable key
			// order and a stable report order, and Go map iteration is not.
			again, strippedAgain := SanitizeGeminiSchema(json.RawMessage(tt.raw))
			if string(again) != string(got) || !slices.Equal(strippedAgain, stripped) {
				t.Errorf("second run differs:\nschema %s\npaths %v", again, strippedAgain)
			}
		})
	}
}

// TestSanitizeGeminiSchemaEmpty: a tool with no parameter schema declares no
// parameters key at all, rather than an empty document this wire would reject.
func TestSanitizeGeminiSchemaEmpty(t *testing.T) {
	for _, raw := range []string{"", "  "} {
		got, stripped := SanitizeGeminiSchema(json.RawMessage(raw))
		if got != nil {
			t.Errorf("clean schema for %q: got %s, want nil", raw, got)
		}
		if stripped != nil {
			t.Errorf("stripped for %q: got %v, want none", raw, stripped)
		}
	}
}
