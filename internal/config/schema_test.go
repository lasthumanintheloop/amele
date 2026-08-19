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
output:
  max_schema_retries: ${RETRIES}
`
	if _, feedback, ok := validator.Validate(yamlToJSON(t, []byte(doc))); !ok {
		t.Errorf("config with ${VAR} in integer and duration fields does not validate:\n%s", feedback)
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
