package config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/lasthumanintheloop/amele/internal/schema"
)

// docsSchemaPath locates the published copy of the schema relative to this
// package directory. The source of truth is the embedded file next to this
// test; the docs copy exists so the contract is browsable without a checkout
// of internal/.
const docsSchemaPath = "../../docs/contracts/config.schema.json"

// TestSchemaJSONBytesCompiles pins that the embedded schema is a valid JSON
// Schema: it must compile under the same engine that validates output.schema,
// so `amele schema` can never publish a document editors reject.
func TestSchemaJSONBytesCompiles(t *testing.T) {
	raw := SchemaJSONBytes()
	if len(raw) == 0 {
		t.Fatal("SchemaJSONBytes returned empty bytes")
	}
	if _, err := schema.Compile(raw); err != nil {
		t.Fatalf("embedded config schema does not compile: %v", err)
	}
}

// TestSchemaJSONBytesReturnsCopy pins the defensive-copy contract: a caller
// mutating the returned slice must not corrupt the embedded schema.
func TestSchemaJSONBytesReturnsCopy(t *testing.T) {
	first := SchemaJSONBytes()
	first[0] = 'X'
	if second := SchemaJSONBytes(); second[0] == 'X' {
		t.Fatal("SchemaJSONBytes returned a view of the embedded bytes, not a copy")
	}
}

// TestSchemaDocsCopyInSync byte-compares the embedded source of truth with
// the published copy under docs/contracts. go:embed cannot reach ../../docs,
// so the copy is kept honest by this test instead of by the build.
func TestSchemaDocsCopyInSync(t *testing.T) {
	published, err := os.ReadFile(docsSchemaPath)
	if err != nil {
		t.Fatalf("reading published schema copy: %v", err)
	}
	if !bytes.Equal(published, SchemaJSONBytes()) {
		t.Fatalf("%s differs from internal/config/config.schema.json; copy the embedded file over it", docsSchemaPath)
	}
}

// TestExamplesValidateAgainstSchema validates every example YAML shipped in
// the repo against the published schema. ${VAR} references are plain strings
// at this level: the schema describes structure, not resolved values.
func TestExamplesValidateAgainstSchema(t *testing.T) {
	validator, err := schema.Compile(SchemaJSONBytes())
	if err != nil {
		t.Fatalf("compiling config schema: %v", err)
	}

	var paths []string
	for _, glob := range []string{"../../examples/*.yaml", "../../examples/*/agent.yaml", "../../testdata/live-scenarios/*.yaml"} {
		matches, err := filepath.Glob(glob)
		if err != nil {
			t.Fatal(err)
		}
		paths = append(paths, matches...)
	}
	if len(paths) == 0 {
		t.Fatal("no example YAML files found; the repo layout changed under this test")
	}

	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			raw, err := os.ReadFile(path) //nolint:gosec // G304: fixed repo-relative testdata path.
			if err != nil {
				t.Fatal(err)
			}
			jsonDoc := yamlToJSON(t, raw)
			if _, feedback, ok := validator.Validate(jsonDoc); !ok {
				t.Errorf("%s does not validate against the config schema:\n%s", path, feedback)
			}
		})
	}
}

// TestSchemaAcceptsEnvReferencesInTypedFields pins the interpolation escape
// hatch: runtime interpolation legally fills integer and duration fields from
// the environment (`max_tokens: ${MAXTOK}` passes amele validate), so at
// editor time - before interpolation - the schema must accept the ${VAR}
// reference form in those fields rather than red-squiggling the exact
// headless/cron idiom the project pushes.
func TestSchemaAcceptsEnvReferencesInTypedFields(t *testing.T) {
	validator, err := schema.Compile(SchemaJSONBytes())
	if err != nil {
		t.Fatalf("compiling config schema: %v", err)
	}

	const doc = `
model: test-model
provider:
  base_url: https://api.example.com/v1
  api_key: ${API_KEY}
  request_timeout: ${REQ_TIMEOUT}
  max_output_tokens: ${MAXOUT}
limits:
  max_turns: ${TURNS}
  max_tokens: ${MAXTOK}
  timeout: ${RUN_TIMEOUT}
  max_logged_field: ${LOGCLIP}
output:
  max_schema_retries: ${RETRIES}
`
	if _, feedback, ok := validator.Validate(yamlToJSON(t, []byte(doc))); !ok {
		t.Errorf("config with ${VAR} in integer and duration fields does not validate:\n%s", feedback)
	}
}

// TestSchemaMaxToolResultBytesFloor pins runtime/schema agreement for the
// tool-result guard: Validate refuses anything below 1 KiB (there is no "off"
// setting), so the published schema must refuse it at editor time too - while
// still accepting the whole-value ${VAR} form the headless idiom uses.
func TestSchemaMaxToolResultBytesFloor(t *testing.T) {
	validator, err := schema.Compile(SchemaJSONBytes())
	if err != nil {
		t.Fatalf("compiling config schema: %v", err)
	}

	const head = `
model: test-model
provider:
  base_url: https://api.example.com/v1
limits:
  max_tool_result_bytes: `

	tests := []struct {
		name  string
		value string
		want  bool // whether the document must validate
	}{
		{"below the floor is rejected", "512", false},
		{"zero is rejected", "0", false},
		{"the floor itself validates", "1024", true},
		{"an env reference validates", "\"${CAP}\"", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, feedback, ok := validator.Validate(yamlToJSON(t, []byte(head+tt.value+"\n")))
			if ok != tt.want {
				t.Errorf("schema validation = %v, want %v:\n%s", ok, tt.want, feedback)
			}
		})
	}
}

// TestSchemaProviderTypeGemini pins the published enum against the runtime:
// `type: gemini` is a valid third wire value in the schema editors consume, and
// a near-miss spelling is not.
func TestSchemaProviderTypeGemini(t *testing.T) {
	validator, err := schema.Compile(SchemaJSONBytes())
	if err != nil {
		t.Fatalf("compiling config schema: %v", err)
	}

	const doc = `
model: gemini-3-pro
provider:
  type: gemini
  api_key: ${GEMINI_API_KEY}
  reasoning:
    budget_tokens: 8192
`
	if _, feedback, ok := validator.Validate(yamlToJSON(t, []byte(doc))); !ok {
		t.Errorf("a gemini provider block does not validate:\n%s", feedback)
	}

	const misspelled = `
model: gemini-3-pro
provider:
  type: Gemini
`
	if _, _, ok := validator.Validate(yamlToJSON(t, []byte(misspelled))); ok {
		t.Error("schema accepts provider.type \"Gemini\" although the runtime rejects it")
	}

}

// TestSchemaVertexBlock pins the vertex block on the editor side: the block a
// Vertex config writes validates, and a misspelled key inside it is caught
// rather than silently ignored (additionalProperties: false, the house rule for
// every closed block in this schema).
//
// Presence IS schema-expressible, so the block requires project and location:
// an editor can then say "location is required" while the YAML is being typed,
// which is the whole job of the published schema. What stays OUT are the
// genuine CROSS-FIELD relations - never together with api_key, gemini only -
// because encoding those means maintaining the same conditional in two copies
// of a frozen contract while `amele validate` already states them in the
// operator's own words.
func TestSchemaVertexBlock(t *testing.T) {
	validator, err := schema.Compile(SchemaJSONBytes())
	if err != nil {
		t.Fatalf("compiling config schema: %v", err)
	}

	const doc = `
model: gemini-3-pro
provider:
  type: gemini
  vertex:
    project: my-project
    location: europe-west4
    credentials: /etc/amele/sa.json
`
	if _, feedback, ok := validator.Validate(yamlToJSON(t, []byte(doc))); !ok {
		t.Errorf("a vertex provider block does not validate:\n%s", feedback)
	}

	const typo = `
model: gemini-3-pro
provider:
  type: gemini
  vertex:
    project: my-project
    region: europe-west4
`
	if _, _, ok := validator.Validate(yamlToJSON(t, []byte(typo))); ok {
		t.Error("schema accepts provider.vertex.region although the runtime rejects unknown keys")
	}

	// A vertex request is addressed by project AND location; neither has a
	// default, and a missing one is exit 2 at run time. The editor may as well
	// say so while the file is still being typed.
	for name, incomplete := range map[string]string{
		"no location": "model: m\nprovider: {type: gemini, vertex: {project: my-project}}\n",
		"no project":  "model: m\nprovider: {type: gemini, vertex: {location: europe-west4}}\n",
		"empty block": "model: m\nprovider: {type: gemini, vertex: {}}\n",
	} {
		if _, _, ok := validator.Validate(yamlToJSON(t, []byte(incomplete))); ok {
			t.Errorf("schema accepts a vertex block with %s although the runtime refuses it", name)
		}
	}
}

// yamlToJSON decodes a config YAML document and re-encodes it as JSON, the
// form the schema validator consumes. Both steps must succeed for the test to
// mean anything, so a failure is fatal rather than reported.
func yamlToJSON(t *testing.T, raw []byte) string {
	t.Helper()
	var doc any
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("parsing YAML: %v", err)
	}
	jsonDoc, err := json.Marshal(normalizeYAML(doc))
	if err != nil {
		t.Fatalf("converting to JSON: %v", err)
	}
	return string(jsonDoc)
}

// normalizeYAML rewrites yaml.v3's decoded tree into a json.Marshal-able one:
// map keys become strings at every depth. yaml.v3 already produces
// map[string]any for string keys, but non-string keys (or older behaviors)
// arrive as map[any]any, which json.Marshal rejects.
func normalizeYAML(v any) any {
	switch val := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(val))
		for k, item := range val {
			out[k] = normalizeYAML(item)
		}
		return out
	case map[any]any:
		out := make(map[string]any, len(val))
		for k, item := range val {
			out[fmt.Sprint(k)] = normalizeYAML(item)
		}
		return out
	case []any:
		out := make([]any, len(val))
		for i, item := range val {
			out[i] = normalizeYAML(item)
		}
		return out
	default:
		return v
	}
}

// TestSchemaStructTwoWaySync is the drift guard: every yaml-tagged field in
// the Config struct tree must exist as a schema property, and every schema
// property must correspond to a struct field. Adding a config field without
// updating the schema (or vice versa) fails here, naming the missing path.
func TestSchemaStructTwoWaySync(t *testing.T) {
	structPaths := map[string]bool{}
	collectYAMLPaths(reflect.TypeOf(Config{}), "", structPaths)

	var doc map[string]any
	if err := json.Unmarshal(SchemaJSONBytes(), &doc); err != nil {
		t.Fatalf("parsing embedded schema: %v", err)
	}
	schemaPaths := map[string]bool{}
	collectSchemaPaths(doc, "", schemaPaths)

	for _, path := range sortedKeys(structPaths) {
		if !schemaPaths[path] {
			t.Errorf("config field %q has no property in config.schema.json", path)
		}
	}
	for _, path := range sortedKeys(schemaPaths) {
		if !structPaths[path] {
			t.Errorf("schema property %q has no yaml-tagged field in the Config struct tree", path)
		}
	}
}

// collectYAMLPaths walks a struct type, recording the dotted yaml-tag path of
// every exported field. Slices of structs contribute their element fields
// under a "[]" marker; maps and scalars are leaves - their value shape is the
// schema's business (additionalProperties), not a named property.
func collectYAMLPaths(t reflect.Type, prefix string, out map[string]bool) {
	for i := range t.NumField() {
		field := t.Field(i)
		if !field.IsExported() {
			continue // derived state (e.g. interpolated), not YAML schema
		}
		name := strings.Split(field.Tag.Get("yaml"), ",")[0]
		if name == "-" {
			continue
		}
		if name == "" {
			// yaml.v3's default for an untagged field.
			name = strings.ToLower(field.Name)
		}
		path := prefix + name
		out[path] = true

		ft := field.Type
		// An optional block is a pointer to a struct (nil = absent, e.g.
		// mcp.servers[].auth); its fields are still named schema properties.
		if ft.Kind() == reflect.Pointer && ft.Elem().Kind() == reflect.Struct {
			ft = ft.Elem()
		}
		switch ft.Kind() {
		case reflect.Struct:
			collectYAMLPaths(ft, path+".", out)
		case reflect.Slice:
			if ft.Elem().Kind() == reflect.Struct {
				collectYAMLPaths(ft.Elem(), path+"[].", out)
			}
		default:
			// Leaf: scalar, map, or slice of scalars.
		}
	}
}

// collectSchemaPaths mirrors collectYAMLPaths on the JSON side: it records
// every named property, descending into nested "properties" and into array
// "items" that declare properties. Objects validated only by
// additionalProperties (maps) and free-form objects (output.schema) are
// leaves, matching the struct walk.
func collectSchemaPaths(node map[string]any, prefix string, out map[string]bool) {
	props, ok := node["properties"].(map[string]any)
	if !ok {
		return
	}
	for name, raw := range props {
		sub, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		path := prefix + name
		out[path] = true
		if _, nested := sub["properties"]; nested {
			collectSchemaPaths(sub, path+".", out)
			continue
		}
		if items, ok := sub["items"].(map[string]any); ok {
			if _, nested := items["properties"]; nested {
				collectSchemaPaths(items, path+"[].", out)
			}
		}
	}
}

// sortedKeys returns the map's keys in stable order so failure output is
// deterministic.
func sortedKeys(m map[string]bool) []string {
	return slices.Sorted(maps.Keys(m))
}

// TestSchemaRejectsEmptyExecutable pins runtime/schema agreement: Validate
// rejects `command: [""]`, so the published schema (the public contract
// editors and external validators consume) must reject it too.
func TestSchemaRejectsEmptyExecutable(t *testing.T) {
	validator, err := schema.Compile(SchemaJSONBytes())
	if err != nil {
		t.Fatalf("compiling config schema: %v", err)
	}

	const doc = `
model: test-model
provider:
  base_url: https://api.example.com/v1
tools:
  subprocess:
    - name: t
      description: d
      command: [""]
`
	if _, _, ok := validator.Validate(yamlToJSON(t, []byte(doc))); ok {
		t.Error("schema accepts command [\"\"] although the runtime rejects it")
	}
}

// TestSchemaRejectsEnvAssignment pins runtime/schema agreement for env
// allowlists: Validate rejects "NAME=value" entries (the list carries names,
// not assignments), so the published schema must reject them too - for both
// the shell tool and subprocess tools.
func TestSchemaRejectsEnvAssignment(t *testing.T) {
	validator, err := schema.Compile(SchemaJSONBytes())
	if err != nil {
		t.Fatalf("compiling config schema: %v", err)
	}

	docs := map[string]string{
		"shell": `
model: test-model
provider:
  base_url: https://api.example.com/v1
tools:
  shell:
    enabled: true
    env: ["PATH=/usr/bin"]
`,
		"subprocess": `
model: test-model
provider:
  base_url: https://api.example.com/v1
tools:
  subprocess:
    - name: t
      description: d
      command: ["true"]
      env: ["KEY=val"]
`,
	}
	for name, doc := range docs {
		t.Run(name, func(t *testing.T) {
			if _, _, ok := validator.Validate(yamlToJSON(t, []byte(doc))); ok {
				t.Error("schema accepts an env assignment although the runtime rejects it")
			}
		})
	}
}

// TestSchemaMCPAgreesWithRuntime pins runtime/schema agreement for the mcp
// block: what Validate rejects (an unknown transport type, an out-of-charset
// server name, an env assignment) the published schema must reject too, and
// the canonical shape must validate cleanly.
func TestSchemaMCPAgreesWithRuntime(t *testing.T) {
	validator, err := schema.Compile(SchemaJSONBytes())
	if err != nil {
		t.Fatalf("compiling config schema: %v", err)
	}

	const head = `
model: test-model
provider:
  base_url: https://api.example.com/v1
mcp:
  servers:
`
	rejected := map[string]string{
		"bad transport type": `
    - name: github
      transport:
        type: sse
        url: https://mcp.example.com/mcp
`,
		"bad server name": `
    - name: GitHub
      transport:
        type: stdio
        command: ["srv"]
`,
		"empty executable": `
    - name: github
      transport:
        type: stdio
        command: [""]
`,
		"env assignment": `
    - name: github
      transport:
        type: stdio
        command: ["srv"]
        env: ["KEY=val"]
`,
		"missing transport": `
    - name: github
`,
		"unknown key": `
    - name: github
      transports:
        type: stdio
        command: ["srv"]
`,
		"unknown auth type": `
    - name: github
      transport:
        type: http
        url: https://mcp.example.com/mcp
      auth:
        type: basic
`,
		"auth with an unknown field": `
    - name: github
      transport:
        type: http
        url: https://mcp.example.com/mcp
      auth:
        type: oauth
        client_secret: shhh
`,
		"empty auth scope": `
    - name: github
      transport:
        type: http
        url: https://mcp.example.com/mcp
      auth:
        type: oauth
        scopes: [""]
`,
	}
	for name, tail := range rejected {
		t.Run(name, func(t *testing.T) {
			if _, _, ok := validator.Validate(yamlToJSON(t, []byte(head+tail))); ok {
				t.Error("schema accepts a config the runtime rejects")
			}
		})
	}

	t.Run("canonical example", func(t *testing.T) {
		const doc = head + `
    - name: github
      transport:
        type: http
        url: https://mcp.example.com/mcp
        headers:
          Authorization: "Bearer ${GITHUB_TOKEN}"
      tools:
        include: ["issue_*"]
        exclude: ["issue_delete"]
      call_timeout: 90s
      required: false
    - name: remote-oauth
      transport:
        type: http
        url: https://oauth.example.com/mcp
      auth:
        type: oauth
        client_id: amele-cli
        scopes: ["repo"]
    - name: local-fs
      transport:
        type: stdio
        command: ["./tools/fs-server", "--root", "."]
        env: ["HOME"]
`
		if _, feedback, ok := validator.Validate(yamlToJSON(t, []byte(doc))); !ok {
			t.Errorf("canonical mcp config does not validate:\n%s", feedback)
		}
	})
}

// TestSchemaProviderTuningAgreesWithRuntime pins runtime/schema agreement for
// the tuning surface: an editor must red-squiggle exactly what `amele validate`
// refuses (an unknown dialect or effort, an out-of-range sampling value, a
// misspelled key), and must leave the canonical block - including the free-form
// params escape hatch - alone.
func TestSchemaProviderTuningAgreesWithRuntime(t *testing.T) {
	validator, err := schema.Compile(SchemaJSONBytes())
	if err != nil {
		t.Fatalf("compiling config schema: %v", err)
	}

	const head = `
model: test-model
provider:
  base_url: https://api.example.com/v1
`
	rejected := map[string]string{
		"unknown dialect":           "  dialect: gemini\n",
		"dialect case":              "  dialect: DeepSeek\n",
		"unknown effort":            "  reasoning:\n    effort: insane\n",
		"negative budget":           "  reasoning:\n    budget_tokens: -1\n",
		"unknown reasoning key":     "  reasoning:\n    depth: high\n",
		"temperature too high":      "  temperature: 2.5\n",
		"temperature negative":      "  temperature: -0.5\n",
		"top_p zero":                "  top_p: 0\n",
		"top_p above one":           "  top_p: 1.5\n",
		"misspelled key":            "  temprature: 0.5\n",
		"negative attempts":         "  retry:\n    max_attempts: -1\n",
		"attempts above ten":        "  retry:\n    max_attempts: 11\n",
		"unknown retry key":         "  retry:\n    max_retries: 4\n",
		"backoff is not a duration": "  retry:\n    initial_backoff: soon\n",
	}
	for name, tail := range rejected {
		t.Run(name, func(t *testing.T) {
			if _, _, ok := validator.Validate(yamlToJSON(t, []byte(head+tail))); ok {
				t.Error("schema accepts a config the runtime rejects")
			}
		})
	}

	t.Run("canonical tuning block", func(t *testing.T) {
		const doc = head + `  dialect: openrouter
  max_output_tokens: 65536
  reasoning:
    effort: high
    budget_tokens: 8192
  temperature: 0.2
  top_p: 0.9
  retry:
    max_attempts: 5
    initial_backoff: 500ms
  params:
    verbosity: low
    provider:
      require_parameters: true
`
		if _, feedback, ok := validator.Validate(yamlToJSON(t, []byte(doc))); !ok {
			t.Errorf("canonical provider tuning does not validate:\n%s", feedback)
		}
	})

	// The runtime reads 0 (and an empty duration) as "omitted, use the
	// default", so the schema must not red-squiggle a config `amele validate`
	// accepts.
	t.Run("zero retry knobs mean the defaults", func(t *testing.T) {
		const doc = head + `  retry:
    max_attempts: 0
    initial_backoff: ""
`
		if _, feedback, ok := validator.Validate(yamlToJSON(t, []byte(doc))); !ok {
			t.Errorf("zero retry knobs do not validate:\n%s", feedback)
		}
	})

	// The headless idiom: budgets and sampling filled from the environment.
	// The schema must not flag the exact form the project pushes.
	t.Run("env references in numeric tuning fields", func(t *testing.T) {
		const doc = head + `  reasoning:
    budget_tokens: ${BUDGET}
  temperature: ${TEMP}
  top_p: ${TOPP}
  retry:
    max_attempts: ${ATTEMPTS}
    initial_backoff: ${BACKOFF}
`
		if _, feedback, ok := validator.Validate(yamlToJSON(t, []byte(doc))); !ok {
			t.Errorf("config with ${VAR} in tuning fields does not validate:\n%s", feedback)
		}
	})
}
