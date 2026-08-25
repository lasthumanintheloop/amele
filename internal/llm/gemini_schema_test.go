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
			// A schema of nothing but unsupported keywords becomes the empty
			// schema rather than disappearing: the tool still takes an
			// argument object, and the declaration must stay well-formed.
			name:         "a schema emptied by the sanitizer stays an object",
			raw:          `{"$ref":"#/$defs/x"}`,
			want:         `{}`,
			wantStripped: []string{"$ref"},
		},
		{
			// Not an object, so there is no keyword to strip: passed through
			// compacted, and the wire's own 400 names it if it is wrong.
			name: "a non-object schema passes through",
			raw:  `[ 1, 2 ]`,
			want: `[1,2]`,
		},
		{
			// Not JSON at all: nothing is invented and nothing is dropped, so
			// the failure stays visible (the encoder refuses the body) instead
			// of a tool quietly losing its parameters.
			name: "an unparseable schema passes through",
			raw:  `{"broken":`,
			want: `{"broken":`,
		},
		{
			// A schema position holding something that is not a schema is left
			// alone: recursing into it is impossible, and rewriting it would
			// be a guess.
			name: "unreadable schema positions are left alone",
			raw:  `{"properties":"nonsense","items":true,"anyOf":[1,"x"]}`,
			want: `{"anyOf":[1,"x"],"items":true,"properties":"nonsense"}`,
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
