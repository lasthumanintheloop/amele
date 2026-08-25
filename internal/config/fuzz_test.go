package config

import (
	"os"
	"path/filepath"
	"testing"
)

// FuzzLoad throws arbitrary bytes at the YAML loader. The config parser is a
// security surface (it consumes untrusted recipe files), so it must never
// panic - any input either loads or returns an error.
func FuzzLoad(f *testing.F) {
	f.Add([]byte(minimalYAML))
	f.Add([]byte("model: m\nprovider: {base_url: 'https://x', api_key: '${K}'}\n"))
	f.Add([]byte("limits:\n  timeout: 5x\n"))
	f.Add([]byte("$$${A}$${B}$"))
	f.Add([]byte{0xff, 0xfe, 0x00})
	// The optional auth block: a pointer field whose nested keys are strict-
	// decoded, so it is its own parser shape worth mutating.
	f.Add([]byte("mcp:\n  servers:\n    - name: s\n      transport: {type: http, url: 'https://x/mcp'}\n      auth: {type: oauth, client_id: c, scopes: ['a']}\n"))
	// The provider tuning surface: an optional nested block, two pointer
	// floats and a free-form map whose values reach json.Marshal during
	// validation - four parser shapes the other seeds do not exercise.
	f.Add([]byte("model: m\nprovider:\n  base_url: 'https://x'\n  dialect: openrouter\n  max_output_tokens: 65536\n  reasoning: {effort: high, budget_tokens: 8192}\n  temperature: 0.2\n  top_p: 0.9\n  params: {verbosity: low, provider: {require_parameters: true}}\n"))
	// The gemini wire: a third provider.type whose validation takes different
	// branches (no base_url requirement, no dialect, its own owned-params set),
	// so mutating from here reaches code the other seeds never enter.
	f.Add([]byte("model: m\nprovider:\n  type: gemini\n  api_key: '${K}'\n  reasoning: {budget_tokens: 8192}\n  params: {labels: {team: ops}}\n"))
	// The vertex block: another optional pointer struct, and the only strings
	// in the config that are held to a charset because they become part of a
	// HOSTNAME. Mutating from here is what walks ValidVertexID.
	f.Add([]byte("model: m\nprovider:\n  type: gemini\n  vertex: {project: my-project, location: europe-west4, credentials: /etc/amele/sa.json}\n"))
	// A base_url carrying a PATH, next to a vertex block: the shape validation
	// refuses (in vertex mode base_url names the host and nothing else), so
	// this seed enters the rejection branch the other provider seeds never do.
	f.Add([]byte("model: m\nprovider:\n  type: gemini\n  base_url: 'https://vpc-sc.example.com/v1beta/models'\n  vertex: {project: p, location: global}\n"))

	f.Fuzz(func(t *testing.T, data []byte) {
		dir := t.TempDir()
		path := filepath.Join(dir, "fuzz.yaml")
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Skip()
		}
		env := func(string) (string, bool) { return "v", true }
		cfg, err := Load(path, env)
		if err != nil {
			return // errors are fine; panics are not
		}
		// A successfully loaded config must also survive validation
		// without panicking.
		_ = cfg.Validate()
	})
}
