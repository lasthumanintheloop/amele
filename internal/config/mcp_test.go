package config

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

// mcpEnv is the environment every MCP test loads against: one credential for
// header interpolation plus the baseline API key.
func mcpEnv() LookupEnv {
	return envMap(map[string]string{"API_KEY": "sk-test", "TOK": "t0ken-value"})
}

// loadMCP writes minimalYAML plus the given fragment to a temp config and
// loads it, failing the test when the load itself fails.
func loadMCP(t *testing.T, fragment string) *Config {
	t.Helper()
	path := writeConfig(t, t.TempDir(), minimalYAML+fragment)
	cfg, err := Load(path, mcpEnv())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return cfg
}

// validMCPYAML is a complete, valid http server declaration reused as the
// positive control across tests.
const validMCPYAML = `
mcp:
  servers:
    - name: github
      transport:
        type: http
        url: https://mcp.example.com/mcp
        headers:
          Authorization: "Bearer ${TOK}"
          X-Client-Name: amele
      tools:
        include: ["issue_*"]
`

func TestMCPValidServer(t *testing.T) {
	cfg := loadMCP(t, validMCPYAML)

	if msgs := cfg.Violations(); len(msgs) != 0 {
		t.Fatalf("valid mcp config reported violations: %v", msgs)
	}
	if len(cfg.MCP.Servers) != 1 {
		t.Fatalf("got %d servers, want 1", len(cfg.MCP.Servers))
	}
	s := cfg.MCP.Servers[0]
	if !s.IsRequired() {
		t.Error("an omitted required: must mean required (fail-fast is the default)")
	}
	// Defaulting the call timeout belongs to the mcp package; config keeps
	// the file literal so explain can show "unset".
	if s.CallTimeout.Std() != 0 {
		t.Errorf("CallTimeout = %v, want 0 (config must not default it)", s.CallTimeout.Std())
	}
	if got := s.Transport.Headers["Authorization"]; got != "Bearer t0ken-value" {
		t.Errorf("header not interpolated: %q", got)
	}
}

func TestMCPRequiredFalse(t *testing.T) {
	cfg := loadMCP(t, `
mcp:
  servers:
    - name: opt
      required: false
      transport:
        type: stdio
        command: ["srv"]
`)
	if cfg.MCP.Servers[0].IsRequired() {
		t.Error("required: false must be honored")
	}
}

func TestMCPViolations(t *testing.T) {
	tests := []struct {
		name     string
		fragment string
		want     []string
		notWant  []string
	}{
		{
			name: "bad name",
			fragment: `
mcp:
  servers:
    - name: GitHub
      transport: {type: stdio, command: ["srv"]}
`,
			want: []string{`mcp.servers[0].name "GitHub" must match ^[a-z0-9_-]{1,32}$`},
		},
		{
			name: "duplicate name",
			fragment: `
mcp:
  servers:
    - name: x
      transport: {type: stdio, command: ["srv"]}
    - name: x
      transport: {type: stdio, command: ["srv"]}
`,
			want: []string{`mcp.servers[1].name "x" is declared twice (first at mcp.servers[0])`},
		},
		{
			name: "reserved builtin name",
			fragment: `
mcp:
  servers:
    - name: shell
      transport: {type: stdio, command: ["srv"]}
`,
			want: []string{`mcp.servers[0].name "shell" is reserved for a builtin tool`},
		},
		{
			name: "collides with subprocess tool",
			fragment: `
tools:
  subprocess:
    - name: mailer
      description: send mail
      command: ["msmtp"]
mcp:
  servers:
    - name: mailer
      transport: {type: stdio, command: ["srv"]}
`,
			want: []string{`mcp.servers[0].name "mailer" collides with a tools.subprocess entry`},
		},
		{
			name: "unsupported transport type",
			fragment: `
mcp:
  servers:
    - name: s
      transport: {type: sse, url: https://x.example.com/mcp}
`,
			want: []string{`mcp.servers[0].transport.type "sse" is not supported (stdio, http)`},
		},
		{
			name: "missing transport type",
			fragment: `
mcp:
  servers:
    - name: s
      transport: {url: https://x.example.com/mcp}
`,
			want: []string{`mcp.servers[0].transport.type is required (stdio, http)`},
		},
		{
			name: "http without url",
			fragment: `
mcp:
  servers:
    - name: s
      transport: {type: http}
`,
			want: []string{`mcp.servers[0].transport.url is required for type http`},
		},
		{
			name: "stdio without command",
			fragment: `
mcp:
  servers:
    - name: s
      transport: {type: stdio}
`,
			want: []string{`mcp.servers[0].transport.command is required for type stdio`},
		},
		{
			name: "stdio with empty executable",
			fragment: `
mcp:
  servers:
    - name: s
      transport: {type: stdio, command: [""]}
`,
			want: []string{`mcp.servers[0].transport.command is required for type stdio`},
		},
		{
			name: "http with command",
			fragment: `
mcp:
  servers:
    - name: s
      transport:
        type: http
        url: https://x.example.com/mcp
        command: ["srv"]
        env: ["PATH"]
`,
			want: []string{
				`mcp.servers[0].transport.command is only valid for type stdio`,
				`mcp.servers[0].transport.env is only valid for type stdio`,
			},
		},
		{
			name: "stdio with url and headers",
			fragment: `
mcp:
  servers:
    - name: s
      transport:
        type: stdio
        command: ["srv"]
        url: https://x.example.com/mcp
        headers: {X-Client-Name: amele}
`,
			want: []string{
				`mcp.servers[0].transport.url is only valid for type http`,
				`mcp.servers[0].transport.headers is only valid for type http`,
			},
		},
		{
			name: "plain http remote",
			fragment: `
mcp:
  servers:
    - name: s
      transport: {type: http, url: "http://example.com/mcp"}
`,
			want: []string{`must use https (plain http is allowed only for localhost/127.0.0.1/::1)`},
		},
		{
			name: "plain http localhost is fine",
			fragment: `
mcp:
  servers:
    - name: s
      transport: {type: http, url: "http://localhost:8080/mcp"}
`,
			notWant: []string{"must use https"},
		},
		{
			name: "plain http loopback ip is fine",
			fragment: `
mcp:
  servers:
    - name: s
      transport: {type: http, url: "http://127.0.0.1:8080/mcp"}
`,
			notWant: []string{"must use https"},
		},
		{
			name: "non-http scheme",
			fragment: `
mcp:
  servers:
    - name: s
      transport: {type: http, url: "ftp://x.example.com/mcp"}
`,
			want: []string{`must be an http(s) URL`},
		},
		{
			name: "url without host",
			fragment: `
mcp:
  servers:
    - name: s
      transport: {type: http, url: "not a url"}
`,
			want: []string{`is not a valid absolute URL`},
		},
		{
			name: "non-sensitive header literal is fine",
			fragment: `
mcp:
  servers:
    - name: s
      transport:
        type: http
        url: https://x.example.com/mcp
        headers: {X-Client-Name: amele}
`,
			notWant: []string{"sensitive"},
		},
		{
			name: "duplicate header case-insensitive",
			fragment: `
mcp:
  servers:
    - name: s
      transport:
        type: http
        url: https://x.example.com/mcp
        headers:
          Authorization: "Bearer ${TOK}"
          authorization: "Bearer ${TOK}"
`,
			want: []string{`duplicate header (case-insensitive)`},
		},
		{
			name: "managed header",
			fragment: `
mcp:
  servers:
    - name: s
      transport:
        type: http
        url: https://x.example.com/mcp
        headers: {Host: x.example.com}
`,
			want: []string{`header "Host" is managed by amele and cannot be set`},
		},
		{
			name: "header value with CRLF",
			fragment: `
mcp:
  servers:
    - name: s
      transport:
        type: http
        url: https://x.example.com/mcp
        headers: {X-Client-Name: "a\nb"}
`,
			want: []string{`must not contain CR/LF`},
		},
		{
			name: "env assignment on stdio",
			fragment: `
mcp:
  servers:
    - name: s
      transport: {type: stdio, command: ["srv"], env: ["A=B"]}
`,
			want: []string{`mcp.servers[0].transport.env[0]: "A=B" must be a variable name, not an assignment`},
		},
		{
			name: "negative call timeout",
			fragment: `
mcp:
  servers:
    - name: s
      call_timeout: -1s
      transport: {type: stdio, command: ["srv"]}
`,
			want: []string{`mcp.servers[0].call_timeout must not be negative`},
		},
		{
			name: "empty tool filter entries",
			fragment: `
mcp:
  servers:
    - name: s
      transport: {type: stdio, command: ["srv"]}
      tools:
        include: [""]
        exclude: [""]
`,
			want: []string{
				`mcp.servers[0].tools.include[0] must not be empty`,
				`mcp.servers[0].tools.exclude[0] must not be empty`,
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := loadMCP(t, tc.fragment)
			joined := strings.Join(cfg.Violations(), "\n")
			for _, want := range tc.want {
				if !strings.Contains(joined, want) {
					t.Errorf("violations missing %q; got:\n%s", want, joined)
				}
			}
			for _, notWant := range tc.notWant {
				if strings.Contains(joined, notWant) {
					t.Errorf("violations unexpectedly contain %q; got:\n%s", notWant, joined)
				}
			}
		})
	}
}

// TestMCPLiteralSensitiveHeaderRejected pins the pre-interpolation guard: a
// credential written literally in the YAML is a load error (exit 2), because
// after interpolation a literal and a substituted value look identical.
func TestMCPLiteralSensitiveHeaderRejected(t *testing.T) {
	tests := []struct {
		name     string
		fragment string
		want     string // empty means the load must succeed
	}{
		{
			name: "authorization literal",
			fragment: `
mcp:
  servers:
    - name: s
      transport:
        type: http
        url: https://x.example.com/mcp
        headers: {Authorization: "Bearer abc"}
`,
			want: `mcp.servers[0].transport.headers.Authorization is sensitive and must reference an environment variable (${VAR})`,
		},
		{
			name: "api key header literal",
			fragment: `
mcp:
  servers:
    - name: s
      transport:
        type: http
        url: https://x.example.com/mcp
        headers: {X-Api-Key: literal}
`,
			want: `mcp.servers[0].transport.headers.X-Api-Key is sensitive and must reference an environment variable (${VAR})`,
		},
		{
			name: "partially literal is still literal",
			fragment: `
mcp:
  servers:
    - name: s
      transport:
        type: http
        url: https://x.example.com/mcp
        headers: {Authorization: "Bearer sk-live-${TOK}"}
`,
			want: `mcp.servers[0].transport.headers.Authorization is sensitive and must reference an environment variable (${VAR})`,
		},
		{
			name:     "reference is accepted",
			fragment: validMCPYAML,
		},
		{
			name: "non-sensitive literal is accepted",
			fragment: `
mcp:
  servers:
    - name: s
      transport:
        type: http
        url: https://x.example.com/mcp
        headers: {X-Client-Name: amele}
`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			path := writeConfig(t, t.TempDir(), minimalYAML+tc.fragment)
			_, err := Load(path, mcpEnv())
			if tc.want == "" {
				if err != nil {
					t.Fatalf("Load: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatal("Load accepted a literal credential header")
			}
			if !errors.Is(err, ErrInvalid) {
				t.Errorf("error does not wrap ErrInvalid: %v", err)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not contain %q", err, tc.want)
			}
		})
	}
}

func TestSensitiveHeaderName(t *testing.T) {
	sensitive := []string{"Authorization", "authorization", "Cookie", "X-Api-Key", "x-token", "My-Secret", "X-Password", "X-Credential"}
	for _, name := range sensitive {
		if !SensitiveHeaderName(name) {
			t.Errorf("SensitiveHeaderName(%q) = false, want true", name)
		}
	}
	plain := []string{"X-Client-Name", "Accept-Language", "User-Agent", "X-Request-Id"}
	for _, name := range plain {
		if SensitiveHeaderName(name) {
			t.Errorf("SensitiveHeaderName(%q) = true, want false", name)
		}
	}
}

// TestMCPRelativeCommandResolved pins that a stdio server's executable follows
// the same pack-relocatability rule as a subprocess tool's.
func TestMCPRelativeCommandResolved(t *testing.T) {
	dir := t.TempDir()
	path := writeConfig(t, dir, minimalYAML+`
mcp:
  servers:
    - name: local
      transport: {type: stdio, command: ["./tools/srv", "--flag"]}
    - name: bare
      transport: {type: stdio, command: ["srv"]}
`)
	cfg, err := Load(path, mcpEnv())
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got, want := cfg.MCP.Servers[0].Transport.Command[0], filepath.Join(dir, "tools/srv"); got != want {
		t.Errorf("command[0] = %q, want %q", got, want)
	}
	if got := cfg.MCP.Servers[0].Transport.Command[1]; got != "--flag" {
		t.Errorf("arguments must not be rewritten, got %q", got)
	}
	if got := cfg.MCP.Servers[1].Transport.Command[0]; got != "srv" {
		t.Errorf("bare command must resolve from PATH, got %q", got)
	}
}

// TestMCPHeaderSecrets pins the redaction surface: a value substituted into a
// sensitive header must reach both the header-secret list and the general
// interpolated-secret list, and must be marked as a credential binding.
func TestMCPHeaderSecrets(t *testing.T) {
	cfg := loadMCP(t, validMCPYAML)

	if got := cfg.MCPHeaderSecrets(); len(got) != 1 || got[0] != "Bearer t0ken-value" {
		t.Errorf("MCPHeaderSecrets() = %v, want [\"Bearer t0ken-value\"]", got)
	}
	if !contains(cfg.InterpolatedSecrets(), "t0ken-value") {
		t.Errorf("InterpolatedSecrets() = %v, want it to contain the header value", cfg.InterpolatedSecrets())
	}
	var marked bool
	for _, b := range cfg.EnvBindings() {
		if b.Name == "TOK" {
			marked = b.APIKey
		}
	}
	if !marked {
		t.Error("a ${VAR} used in a sensitive header must be marked as a credential binding")
	}
}

// TestMCPHeaderSecretsSkipsPlainHeaders keeps the redaction list tight: a
// non-sensitive header value is not a secret, and redacting it would blank out
// ordinary text in the session log.
func TestMCPHeaderSecretsSkipsPlainHeaders(t *testing.T) {
	cfg := loadMCP(t, `
mcp:
  servers:
    - name: s
      transport:
        type: http
        url: https://x.example.com/mcp
        headers: {X-Client-Name: amele}
`)
	if got := cfg.MCPHeaderSecrets(); len(got) != 0 {
		t.Errorf("MCPHeaderSecrets() = %v, want none", got)
	}
}

// TestMCPCredentialPath pins the dotted paths interpolateNode produces, which
// is what credentialPath matches against. Sequence elements inherit their
// parent's path (no index), so the header path carries no [i].
func TestMCPCredentialPath(t *testing.T) {
	tests := []struct {
		path string
		want bool
	}{
		{"provider.api_key", true},
		{"mcp.servers.transport.headers.Authorization", true},
		{"mcp.servers.transport.headers.X-Api-Key", true},
		{"mcp.servers.transport.headers.X-Client-Name", false},
		{"mcp.servers.transport.url", false},
		{"prompt", false},
	}
	for _, tc := range tests {
		if got := credentialPath(tc.path); got != tc.want {
			t.Errorf("credentialPath(%q) = %v, want %v", tc.path, got, tc.want)
		}
	}
}

// TestMCPMissingEnvVar pins the load contract for a header reference: Load
// fails, LoadTolerant reports it under EnvMissing so explain can list it.
func TestMCPMissingEnvVar(t *testing.T) {
	path := writeConfig(t, t.TempDir(), minimalYAML+validMCPYAML)
	env := envMap(map[string]string{"API_KEY": "sk-test"})

	if _, err := Load(path, env); err == nil || !strings.Contains(err.Error(), "TOK") {
		t.Fatalf("Load error = %v, want it to name TOK", err)
	}
	cfg, err := LoadTolerant(path, env)
	if err != nil {
		t.Fatalf("LoadTolerant: %v", err)
	}
	if !contains(cfg.EnvMissing(), "TOK") {
		t.Errorf("EnvMissing() = %v, want it to contain TOK", cfg.EnvMissing())
	}
}

// TestMCPNotSettable pins that connecting a server is a grant of capability
// and therefore stays out of the --set allowlist.
func TestMCPNotSettable(t *testing.T) {
	for _, key := range SettableKeys() {
		if strings.HasPrefix(key, "mcp") {
			t.Fatalf("settable key %q grants capability and must not be overridable", key)
		}
	}
	cfg := loadMCP(t, validMCPYAML)
	err := ApplyOverrides(cfg, []string{"mcp.servers[0].transport.url=https://evil.example.com/mcp"}, t.TempDir())
	if err == nil {
		t.Fatal("ApplyOverrides accepted an mcp key")
	}
	if !strings.Contains(err.Error(), `cannot override "mcp.servers[0].transport.url" from the command line`) {
		t.Errorf("unexpected error: %v", err)
	}
}

// contains reports whether values holds want.
func contains(values []string, want string) bool {
	for _, v := range values {
		if v == want {
			return true
		}
	}
	return false
}
