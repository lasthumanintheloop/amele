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
