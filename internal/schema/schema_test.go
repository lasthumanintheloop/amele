package schema_test

import (
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"unicode/utf8"

	"github.com/lasthumanintheloop/amele/internal/schema"
)

// personSchema is the workhorse fixture: one required string, one required
// integer, one optional array. It exercises required/type/nested errors
// without dragging in draft-specific keywords.
const personSchema = `{
  "type": "object",
  "required": ["name", "count"],
  "properties": {
    "name": {"type": "string"},
    "count": {"type": "integer"},
    "tags": {"type": "array", "items": {"type": "string"}}
  },
  "additionalProperties": false
}`

func mustCompile(t *testing.T, s string) *schema.Validator {
	t.Helper()
	v, err := schema.Compile([]byte(s))
	if err != nil {
		t.Fatalf("Compile(%s) returned error: %v", s, err)
	}
	return v
}

func TestCompile(t *testing.T) {
	tests := []struct {
		name    string
		schema  string
		wantErr bool
	}{
		{name: "object schema", schema: personSchema},
		{name: "empty schema accepts everything", schema: `{}`},
		{name: "boolean true schema", schema: `true`},
		{name: "malformed json", schema: `{"type":`, wantErr: true},
		{name: "trailing garbage", schema: `{} {}`, wantErr: true},
		{name: "empty input", schema: ``, wantErr: true},
		{name: "type is not a string", schema: `{"type": 123}`, wantErr: true},
		{name: "unknown type name", schema: `{"type": "nosuchtype"}`, wantErr: true},
		{name: "required is not an array", schema: `{"required": "name"}`, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v, err := schema.Compile([]byte(tt.schema))
			if tt.wantErr {
				if err == nil {
					t.Fatalf("Compile(%q) = nil error, want error", tt.schema)
				}
				if v != nil {
					t.Errorf("Compile(%q) returned a non-nil Validator alongside an error", tt.schema)
				}
				return
			}
			if err != nil {
				t.Fatalf("Compile(%q) returned error: %v", tt.schema, err)
			}
			if v == nil {
				t.Fatal("Compile returned nil Validator with nil error")
			}
		})
	}
}

// TestCompileErrorDoesNotLeakLocalPaths pins the message hygiene rule: the
// compiler's internal resource URL must not drag the working directory into
// an error the user (or the model) will read.
func TestCompileErrorDoesNotLeakLocalPaths(t *testing.T) {
	_, err := schema.Compile([]byte(`{"type": "nosuchtype"}`))
	if err == nil {
		t.Fatal("Compile of an invalid schema returned nil error")
	}
	if strings.Contains(err.Error(), "file://") {
		t.Errorf("Compile error leaks a filesystem URL: %v", err)
	}
}

// TestCompileRefs pins the hermetic-compilation rule: internal refs resolve,
// external ones (http, relative, file://) are refused instead of turning a
// config load into a network fetch or a local-file read.
func TestCompileRefs(t *testing.T) {
	tests := []struct {
		name    string
		schema  string
		wantErr bool
	}{
		{name: "internal defs ref", schema: `{"$ref":"#/$defs/x","$defs":{"x":{"type":"string"}}}`},
		{name: "draft-07 metaschema is bundled", schema: `{"$schema":"http://json-schema.org/draft-07/schema#","type":"object"}`},
		{name: "http ref", schema: `{"$ref":"https://example.com/x.json"}`, wantErr: true},
		{name: "relative ref", schema: `{"$ref":"./other.json"}`, wantErr: true},
		{name: "file ref", schema: `{"$ref":"file:///etc/hostname"}`, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := schema.Compile([]byte(tt.schema))
			if tt.wantErr {
				if err == nil {
					t.Fatalf("Compile(%s) = nil error, want error", tt.schema)
				}
				if !strings.Contains(err.Error(), "not supported") {
					t.Errorf("Compile(%s) error = %v, want it to explain that external refs are unsupported", tt.schema, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("Compile(%s) returned error: %v", tt.schema, err)
			}
		})
	}
}

func TestValidatorJSON(t *testing.T) {
	raw := []byte(personSchema)
	v := mustCompile(t, personSchema)

	got := v.JSON()
	if string(got) != personSchema {
		t.Errorf("JSON() = %q, want the schema verbatim %q", got, personSchema)
	}
	if !json.Valid(got) {
		t.Error("JSON() returned invalid JSON")
	}

	// CONTRACT: the compiled schema is immutable. Neither the caller's input
	// slice nor a previously returned slice may change what JSON() reports.
	raw[0] = 'X'
	got[0] = 'Y'
	if string(v.JSON()) != personSchema {
		t.Errorf("JSON() = %q after caller mutation, want %q", v.JSON(), personSchema)
	}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name          string
		schema        string
		text          string
		wantOK        bool
		wantCanonical string
		wantFeedback  []string // substrings the feedback must contain
	}{
		{
			name:          "bare json object",
			schema:        personSchema,
			text:          `{"name":"bob","count":1}`,
			wantOK:        true,
			wantCanonical: `{"name":"bob","count":1}`,
		},
		{
			name:          "surrounding whitespace is trimmed",
			schema:        personSchema,
			text:          "\n\n  {\"name\":\"bob\",\"count\":1}  \n",
			wantOK:        true,
			wantCanonical: `{"name":"bob","count":1}`,
		},
		{
			name:          "formatting and key order are preserved verbatim",
			schema:        personSchema,
			text:          "{ \"count\" : 1,\n  \"name\":   \"bob\" }",
			wantOK:        true,
			wantCanonical: "{ \"count\" : 1,\n  \"name\":   \"bob\" }",
		},
		{
			name:          "large integers keep their precision",
			schema:        `{"type":"object","properties":{"id":{"type":"integer"}}}`,
			text:          `{"id":12345678901234567890123}`,
			wantOK:        true,
			wantCanonical: `{"id":12345678901234567890123}`,
		},
		{
			name:          "json fenced block after prose",
			schema:        personSchema,
			text:          "Sure, here is the result:\n```json\n{\"name\":\"bob\",\"count\":1}\n```\nHope that helps.",
			wantOK:        true,
			wantCanonical: `{"name":"bob","count":1}`,
		},
		{
			name:          "fence info string is case-insensitive",
			schema:        personSchema,
			text:          "```JSON\n{\"name\":\"bob\",\"count\":1}\n```",
			wantOK:        true,
			wantCanonical: `{"name":"bob","count":1}`,
		},
		{
			name:          "bare fence whose content parses",
			schema:        personSchema,
			text:          "answer:\n```\n{\"name\":\"bob\",\"count\":1}\n```",
			wantOK:        true,
			wantCanonical: `{"name":"bob","count":1}`,
		},
		{
			name:          "first parseable fence wins",
			schema:        personSchema,
			text:          "```\nnot json at all\n```\nand then\n```json\n{\"name\":\"bob\",\"count\":1}\n```",
			wantOK:        true,
			wantCanonical: `{"name":"bob","count":1}`,
		},
		{
			name:          "whole text wins over a later fence",
			schema:        `{}`,
			text:          "{\"whole\":true}",
			wantOK:        true,
			wantCanonical: `{"whole":true}`,
		},
		{
			name:         "fence with a non-json info string is ignored",
			schema:       personSchema,
			text:         "```python\n{\"name\":\"bob\",\"count\":1}\n```",
			wantOK:       false,
			wantFeedback: []string{"JSON"},
		},
		{
			// REGRESSION: a rejected opener used to leave the scanner
			// "closed", so its closing ``` was read as an opening bare fence
			// and every later fence was inverted - the good block below was
			// never seen and a correct answer cost a retry.
			name:          "json fence after a rejected fence is still found",
			schema:        personSchema,
			text:          "```python\nprint(1)\n```\nAnd the answer:\n```json\n{\"name\":\"bob\",\"count\":1}\n```",
			wantOK:        true,
			wantCanonical: `{"name":"bob","count":1}`,
		},
		{
			// REGRESSION: the same parity desync could also pick up text
			// BETWEEN two fences as if it were a block, shipping the wrong
			// JSON as the final answer.
			name:          "text between a rejected fence and a json fence is not a block",
			schema:        `{}`,
			text:          "```text\nfoo\n```\n{\"WRONG\":1}\n```json\n{\"RIGHT\":2}\n```",
			wantOK:        true,
			wantCanonical: `{"RIGHT":2}`,
		},
		{
			name:          "bare fence before a json fence wins on source order",
			schema:        `{}`,
			text:          "```\n{\"FIRST\":1}\n```\n```json\n{\"SECOND\":2}\n```",
			wantOK:        true,
			wantCanonical: `{"FIRST":1}`,
		},
		{
			name:         "unterminated fence is not a block",
			schema:       personSchema,
			text:         "```json\n{\"name\":\"bob\",\"count\":1}",
			wantOK:       false,
			wantFeedback: []string{"JSON"},
		},
		{
			name:          "empty schema accepts any json value",
			schema:        `{}`,
			text:          `[1, 2, 3]`,
			wantOK:        true,
			wantCanonical: `[1, 2, 3]`,
		},
		{
			name:          "empty schema accepts a bare string",
			schema:        `{}`,
			text:          `"hello"`,
			wantOK:        true,
			wantCanonical: `"hello"`,
		},
		{
			name:          "empty schema accepts null",
			schema:        `{}`,
			text:          `null`,
			wantOK:        true,
			wantCanonical: `null`,
		},
		{
			name:         "missing required field",
			schema:       personSchema,
			text:         `{"count":1}`,
			wantOK:       false,
			wantFeedback: []string{"JSON", "schema", "name"},
		},
		{
			name:         "wrong type",
			schema:       personSchema,
			text:         `{"name":"bob","count":"many"}`,
			wantOK:       false,
			wantFeedback: []string{"JSON", "count"},
		},
		{
			name:         "nested item type violation",
			schema:       personSchema,
			text:         `{"name":"bob","count":1,"tags":[7]}`,
			wantOK:       false,
			wantFeedback: []string{"JSON", "tags"},
		},
		{
			name:         "additional property rejected",
			schema:       personSchema,
			text:         `{"name":"bob","count":1,"extra":true}`,
			wantOK:       false,
			wantFeedback: []string{"JSON", "extra"},
		},
		{
			name:         "prose only",
			schema:       personSchema,
			text:         "I could not complete the task, sorry.",
			wantOK:       false,
			wantFeedback: []string{"JSON"},
		},
		{
			name:         "empty text",
			schema:       personSchema,
			text:         "",
			wantOK:       false,
			wantFeedback: []string{"JSON"},
		},
		{
			name:         "whitespace only",
			schema:       personSchema,
			text:         "   \n\t ",
			wantOK:       false,
			wantFeedback: []string{"JSON"},
		},
		{
			name:         "two concatenated values are not one json document",
			schema:       `{}`,
			text:         `{"a":1} {"b":2}`,
			wantOK:       false,
			wantFeedback: []string{"JSON"},
		},
		{
			name:         "truncated json",
			schema:       personSchema,
			text:         `{"name":"bob","count":`,
			wantOK:       false,
			wantFeedback: []string{"JSON"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := mustCompile(t, tt.schema)
			canonical, feedback, ok := v.Validate(tt.text)

			if ok != tt.wantOK {
				t.Fatalf("Validate(%q) ok = %v, want %v (feedback: %s)", tt.text, ok, tt.wantOK, feedback)
			}
			if tt.wantOK {
				if canonical != tt.wantCanonical {
					t.Errorf("Validate(%q) canonical = %q, want %q", tt.text, canonical, tt.wantCanonical)
				}
				if feedback != "" {
					t.Errorf("Validate(%q) feedback = %q, want empty on success", tt.text, feedback)
				}
				return
			}
			if canonical != "" {
				t.Errorf("Validate(%q) canonical = %q, want empty on failure", tt.text, canonical)
			}
			for _, want := range tt.wantFeedback {
				if !strings.Contains(feedback, want) {
					t.Errorf("Validate(%q) feedback = %q, want it to mention %q", tt.text, feedback, want)
				}
			}
		})
	}
}

// TestValidateFeedbackIsBounded pins the retry-cost guard: feedback goes back
// to the model as a user message, so a pathological instance must not blow up
// the prompt.
func TestValidateFeedbackIsBounded(t *testing.T) {
	var props, required []string
	for _, name := range []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"} {
		props = append(props, `"field_`+name+`":{"type":"string","minLength":500}`)
		required = append(required, `"field_`+name+`"`)
	}
	s := `{"type":"object","required":[` + strings.Join(required, ",") + `],"properties":{` + strings.Join(props, ",") + `}}`

	v := mustCompile(t, s)
	_, feedback, ok := v.Validate(`{"field_a":1,"field_b":2,"field_c":3,"field_d":4,"field_e":5}`)
	if ok {
		t.Fatal("Validate accepted an instance that violates the schema")
	}
	if len(feedback) > schema.MaxFeedbackBytes {
		t.Errorf("feedback is %d bytes, want <= %d", len(feedback), schema.MaxFeedbackBytes)
	}
	if n := strings.Count(feedback, "\n  - "); n > schema.MaxFeedbackErrors {
		t.Errorf("feedback lists %d errors, want <= %d:\n%s", n, schema.MaxFeedbackErrors, feedback)
	}
}

// TestValidateFeedbackTruncates exercises the byte cap on a message whose
// three errors alone overflow it, and pins that the cut lands on a rune
// boundary - a mangled trailing rune would travel back to the provider.
func TestValidateFeedbackTruncates(t *testing.T) {
	long := strings.Repeat("ü", 300) // multibyte on purpose
	var props, required []string
	for _, name := range []string{"a", "b", "c"} {
		props = append(props, `"`+name+long+`":{"type":"string"}`)
		required = append(required, `"`+name+long+`"`)
	}
	s := `{"type":"object","required":[` + strings.Join(required, ",") + `],"properties":{` + strings.Join(props, ",") + `}}`

	v := mustCompile(t, s)
	_, feedback, ok := v.Validate(`{"a` + long + `":1,"b` + long + `":2,"c` + long + `":3}`)
	if ok {
		t.Fatal("Validate accepted an instance that violates the schema")
	}
	if len(feedback) > schema.MaxFeedbackBytes {
		t.Fatalf("feedback is %d bytes, want <= %d", len(feedback), schema.MaxFeedbackBytes)
	}
	if len(feedback) < schema.MaxFeedbackBytes/2 {
		t.Fatalf("feedback is only %d bytes; the truncation path was not exercised", len(feedback))
	}
	if !utf8.ValidString(feedback) {
		t.Error("truncated feedback is not valid UTF-8")
	}
	if !strings.HasSuffix(feedback, "(truncated)") {
		t.Errorf("truncated feedback lacks its marker: %q", feedback[len(feedback)-40:])
	}
}

// TestValidateFeedbackIsDeterministic guards against Go's randomized map
// iteration order leaking into the retry message: the same rejected output
// must always produce byte-identical feedback, otherwise sessions stop being
// replayable.
func TestValidateFeedbackIsDeterministic(t *testing.T) {
	v := mustCompile(t, personSchema)
	const bad = `{"count":"many","tags":[1,2],"extra":true}`

	_, first, ok := v.Validate(bad)
	if ok {
		t.Fatal("Validate accepted an instance that violates the schema")
	}
	for i := 0; i < 100; i++ {
		_, got, _ := v.Validate(bad)
		if got != first {
			t.Fatalf("feedback differs between runs:\nfirst: %q\ngot:   %q", first, got)
		}
	}
}

// TestValidateConcurrent pins the documented concurrency contract: one
// compiled Validator is shared by every retry and (later) by every run.
func TestValidateConcurrent(t *testing.T) {
	v := mustCompile(t, personSchema)

	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if i%2 == 0 {
				if _, _, ok := v.Validate(`{"name":"bob","count":1}`); !ok {
					t.Errorf("valid instance rejected")
				}
				return
			}
			if _, fb, ok := v.Validate(`{"count":"many"}`); ok || fb == "" {
				t.Errorf("invalid instance accepted (ok=%v feedback=%q)", ok, fb)
			}
		}(i)
	}
	wg.Wait()
}
