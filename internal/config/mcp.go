package config

// MCP block: see docs/superpowers/specs/2026-08-18-mcp-client-design.md.
//
// It lives at the top level (not under tools:) because one connection will
// later also carry resources and prompts; a block named "tools" would lie
// then, and a renamed block is a breaking schema change.

import (
	"fmt"
	"net/url"
	"regexp"
	"slices"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Transport type values accepted by mcp.servers[].transport.type. Both are
// spelled out in every error message, so the set is small on purpose.
const (
	// MCPTransportStdio runs the server as a child process and speaks JSON-RPC
	// over its stdin/stdout.
	MCPTransportStdio = "stdio"
	// MCPTransportHTTP talks to a remote server over streamable HTTP.
	MCPTransportHTTP = "http"
)

// DefaultMCPCallTimeout bounds a single tool call when a server does not set
// call_timeout. It is applied by the consumer (internal/mcp), not at load
// time, so `explain` can still show which value the file actually chose.
const DefaultMCPCallTimeout = 60 * time.Second

// mcpTransportValues lists the accepted transport spellings for messages.
const mcpTransportValues = "stdio, http"

var (
	// mcpServerNameRe is stricter than toolNameRe: a server name is the prefix
	// of every tool it contributes ("<server>__<tool>"), so it must stay short
	// and unambiguous inside a name the model has to type back verbatim.
	mcpServerNameRe = regexp.MustCompile(`^[a-z0-9_-]{1,32}$`)

	// SECURITY: sensitiveHeaderRe decides which header values must be ${ENV}
	// references and which are redacted from logs. It is deliberately broad -
	// a false positive costs an operator one environment variable, a false
	// negative writes a bearer token into a git-committed YAML.
	sensitiveHeaderRe = regexp.MustCompile(`(?i)^authorization$|^cookie$|key|token|secret|passw|cred`)

	// managedHeaders are set by the transport itself; an operator value would
	// either be overwritten silently or break the protocol, so it is reported
	// as a config error instead of being ignored at run time.
	managedHeaders = map[string]bool{
		"host": true, "content-length": true, "content-type": true,
		"accept": true, "mcp-session-id": true, "mcp-protocol-version": true,
		"last-event-id": true,
	}
)

// MCPConfig is the top-level `mcp:` block: the set of MCP servers whose tools
// are offered to the model.
type MCPConfig struct {
	// Servers are the declared connections, in file order. Names must be
	// unique across the block and must not collide with a builtin or
	// subprocess tool name.
	Servers []MCPServer `yaml:"servers"`
}

// MCPServer is one declared MCP connection.
type MCPServer struct {
	// Name prefixes every tool this server contributes ("<name>__<tool>").
	// It must match ^[a-z0-9_-]{1,32}$.
	Name string `yaml:"name"`
	// Transport says how to reach the server: a child process (stdio) or a
	// remote endpoint (http).
	Transport MCPTransport `yaml:"transport"`
	// Tools optionally narrows which of the server's tools are exposed.
	Tools MCPToolFilter `yaml:"tools"`
	// CallTimeout bounds a single tool call. Zero means the consumer's
	// default (DefaultMCPCallTimeout).
	CallTimeout Duration `yaml:"call_timeout"`
	// Required decides whether a failure to connect aborts the run. nil (the
	// field omitted) means true; see IsRequired.
	Required *bool `yaml:"required"`
}

// IsRequired reports whether a connect failure must abort the run.
//
// The zero value (nil) means required: fail-fast is the amele default because
// a headless run that silently loses half its tools produces a plausible wrong
// answer, and an operator opts OUT of that explicitly with `required: false`.
func (s MCPServer) IsRequired() bool { return s.Required == nil || *s.Required }

// MCPTransport is the connection detail of one server. Which fields are legal
// depends on Type; mixing them is a validation error rather than a silently
// ignored key, so a misplaced `url:` under a stdio server is never mistaken
// for a working remote connection.
type MCPTransport struct {
	// Type is "stdio" or "http". Required - there is no default, because
	// guessing from the other fields would make a typo mean something.
	Type string `yaml:"type"`
	// URL is the endpoint for type http. Must be https unless the host is
	// loopback.
	URL string `yaml:"url"`
	// Headers are extra HTTP headers for type http. SECURITY: values of
	// credential-shaped header names must be ${ENV} references.
	Headers map[string]string `yaml:"headers"`
	// Command is the argv vector for type stdio. A path-like relative
	// command[0] resolves against the config file's directory.
	Command []string `yaml:"command"`
	// Env is the environment allowlist for the child process of a stdio
	// server, with the same semantics as a subprocess tool's env.
	Env []string `yaml:"env"`
}

// MCPToolFilter narrows the tool set a server contributes. Both lists hold
// glob patterns matched against the server-side tool name (before the
// "<server>__" prefix is added). Empty Include means "every tool"; Exclude is
// applied after Include.
type MCPToolFilter struct {
	// Include lists patterns a tool must match to be exposed.
	Include []string `yaml:"include"`
	// Exclude lists patterns that remove a tool that Include let through.
	Exclude []string `yaml:"exclude"`
}

// SensitiveHeaderName reports whether a header name conventionally carries a
// credential.
//
// SECURITY: values of such headers must be ${ENV} references (never a literal
// in the YAML, see rejectLiteralMCPHeaders) and are registered as secrets so
// the session log and progress output redact them.
func SensitiveHeaderName(name string) bool { return sensitiveHeaderRe.MatchString(name) }

// MCPHeaderSecrets returns the (already interpolated) values of every
// sensitive MCP header, deduplicated and without empties, for registration
// with the session log's redactor.
//
// It exists next to InterpolatedSecrets because the two cover different
// leaks: InterpolatedSecrets protects what the ENVIRONMENT supplied, while a
// header value is a composed string ("Bearer <token>") that must be redacted
// as a whole as well.
func (c *Config) MCPHeaderSecrets() []string {
	var values []string
	seen := map[string]bool{}
	for _, s := range c.MCP.Servers {
		for _, name := range sortedHeaderNames(s.Transport.Headers) {
			value := s.Transport.Headers[name]
			if value == "" || seen[value] || !SensitiveHeaderName(name) {
				continue
			}
			seen[value] = true
			values = append(values, value)
		}
	}
	return values
}

// sortedHeaderNames returns the map's keys in stable order, so violations and
// secret lists do not shuffle between runs (Go map iteration order would
// otherwise defeat golden-file testing).
func sortedHeaderNames(headers map[string]string) []string {
	names := make([]string, 0, len(headers))
	for name := range headers {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}

// validateMCP checks the mcp block. Split from Violations to keep each
// function under the complexity budget.
func (c *Config) validateMCP(add func(format string, args ...any)) {
	seen := map[string]int{}
	subprocessNames := map[string]bool{}
	for _, t := range c.Tools.Subprocess {
		subprocessNames[t.Name] = true
	}
	for i, s := range c.MCP.Servers {
		where := fmt.Sprintf("mcp.servers[%d]", i)
		validateMCPName(add, where, s.Name, seen, subprocessNames)
		seen[s.Name] = i
		validateMCPTransport(add, where+".transport", s.Transport)
		if s.CallTimeout < 0 {
			add("%s.call_timeout must not be negative", where)
		}
		for k, pattern := range s.Tools.Include {
			if pattern == "" {
				add("%s.tools.include[%d] must not be empty", where, k)
			}
		}
		for k, pattern := range s.Tools.Exclude {
			if pattern == "" {
				add("%s.tools.exclude[%d] must not be empty", where, k)
			}
		}
	}
}

// validateMCPName checks one server name against the charset rule, the names
// already seen, the reserved builtin names and the subprocess tools. A server
// name is a tool-name prefix, so a collision would let one declaration shadow
// another's tools - and a permission entry would then govern the wrong thing.
func validateMCPName(add func(format string, args ...any), where, name string, seen map[string]int, subprocess map[string]bool) {
	if !mcpServerNameRe.MatchString(name) {
		add("%s.name %q must match %s", where, name, mcpServerNameRe.String())
	}
	if first, dup := seen[name]; dup {
		add("%s.name %q is declared twice (first at mcp.servers[%d])", where, name, first)
	}
	if slices.Contains(builtinToolNames, name) {
		add("%s.name %q is reserved for a builtin tool", where, name)
	}
	if subprocess[name] {
		add("%s.name %q collides with a tools.subprocess entry", where, name)
	}
}

// validateMCPTransport checks that the transport block is internally
// consistent: exactly the fields of the declared type, and nothing else. A
// field belonging to the other type is reported rather than ignored, because
// an ignored `url:` under a stdio server reads like a configured endpoint.
func validateMCPTransport(add func(format string, args ...any), where string, t MCPTransport) {
	switch t.Type {
	case MCPTransportStdio:
		if len(t.Command) == 0 || t.Command[0] == "" {
			add("%s.command is required for type %s", where, MCPTransportStdio)
		}
		if t.URL != "" {
			add("%s.url is only valid for type %s", where, MCPTransportHTTP)
		}
		if len(t.Headers) > 0 {
			add("%s.headers is only valid for type %s", where, MCPTransportHTTP)
		}
		validateEnvAllowlist(add, where+".env", t.Env)
	case MCPTransportHTTP:
		if len(t.Command) > 0 {
			add("%s.command is only valid for type %s", where, MCPTransportStdio)
		}
		if len(t.Env) > 0 {
			add("%s.env is only valid for type %s", where, MCPTransportStdio)
		}
		validateMCPURL(add, where+".url", t.URL)
		validateMCPHeaders(add, where+".headers", t.Headers)
	case "":
		add("%s.type is required (%s)", where, mcpTransportValues)
	default:
		add("%s.type %q is not supported (%s)", where, t.Type, mcpTransportValues)
	}
}

// validateMCPURL checks the endpoint of an http server.
//
// SECURITY: plain http is refused except against a loopback host. An MCP
// connection carries the credential of whatever the server fronts, and the
// tool results it returns steer the rest of the run; on the open network that
// is a plaintext credential and an unauthenticated instruction channel. A
// local dev server is exempt because there is no network to sniff.
func validateMCPURL(add func(format string, args ...any), where, raw string) {
	if raw == "" {
		add("%s is required for type %s", where, MCPTransportHTTP)
		return
	}
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" {
		add("%s %q is not a valid absolute URL", where, raw)
		return
	}
	switch u.Scheme {
	case "https":
	case "http":
		if !isLoopbackHost(u.Hostname()) {
			add("%s %q must use https (plain http is allowed only for localhost/127.0.0.1/::1)", where, raw)
		}
	default:
		add("%s %q must be an http(s) URL", where, raw)
	}
}

// isLoopbackHost reports whether host names the local machine. The list is
// literal rather than a DNS resolution: validation must not depend on what a
// name resolves to on the validating machine.
func isLoopbackHost(host string) bool {
	switch strings.ToLower(host) {
	case "localhost", "127.0.0.1", "::1":
		return true
	default:
		return false
	}
}

// validateMCPHeaders checks header names and values. The literal-credential
// rule is NOT checked here: after interpolation a substituted value and a
// hardcoded one are indistinguishable, so it is enforced on the raw YAML at
// load time (rejectLiteralMCPHeaders).
func validateMCPHeaders(add func(format string, args ...any), where string, headers map[string]string) {
	canonical := map[string]string{} // lowercase name -> name as first written
	for _, name := range sortedHeaderNames(headers) {
		lower := strings.ToLower(name)
		if first, dup := canonical[lower]; dup {
			add("%s: %q is a duplicate header (case-insensitive) of %q", where, name, first)
			continue
		}
		canonical[lower] = name
		if managedHeaders[lower] {
			add("%s: header %q is managed by amele and cannot be set", where, name)
		}
		// SECURITY: a CR/LF in a header value is request splitting. net/http
		// would reject it at send time; catching it here keeps the failure a
		// config error (exit 2) instead of a mid-run provider error.
		if strings.ContainsAny(headers[name], "\r\n") {
			add("%s.%s value must not contain CR/LF", where, name)
		}
	}
}

// rejectLiteralMCPHeaders probes the parsed but NOT yet interpolated node tree
// and fails when a sensitive header holds anything other than ${ENV_VAR}
// references.
//
// SECURITY: this is the MCP twin of rejectLiteralAPIKey. It must run before
// interpolation, because afterwards "Bearer abc" and "Bearer ${TOK}" are the
// same string - and a token committed to git is already leaked.
func rejectLiteralMCPHeaders(root *yaml.Node) error {
	for i, server := range mappingSeq(root, "mcp", "servers") {
		headers := mappingChild(mappingChild(server, "transport"), "headers")
		if headers == nil || headers.Kind != yaml.MappingNode {
			continue
		}
		for j := 1; j < len(headers.Content); j += 2 {
			name := headers.Content[j-1].Value
			if !SensitiveHeaderName(name) {
				continue
			}
			if literalHeaderValue(headers.Content[j].Value) {
				return fmt.Errorf("mcp.servers[%d].transport.headers.%s is sensitive and must reference an environment variable (${VAR})", i, name)
			}
		}
	}
	return nil
}

// authSchemeRe matches what may legally remain of a sensitive header value
// after every ${VAR} reference is stripped: nothing, or a bare authentication
// scheme word ("Bearer ", "Basic "). Anything else - punctuation, digits, a
// key prefix like "sk-live-" - is secret material.
var authSchemeRe = regexp.MustCompile(`^[A-Za-z]*\s*$`)

// literalHeaderValue reports whether a sensitive header's raw YAML value
// carries secret material of its own. A value must reference at least one
// ${VAR} and leave nothing but an auth scheme word behind: "Bearer
// sk-live-${SUFFIX}" is still a committed secret even though it interpolates.
func literalHeaderValue(raw string) bool {
	if !interpRe.MatchString(raw) {
		return true
	}
	stripped := interpRe.ReplaceAllString(raw, "")
	stripped = strings.ReplaceAll(stripped, "$$", "")
	return !authSchemeRe.MatchString(stripped)
}

// mappingSeq resolves a dotted path of mapping keys ending in a sequence and
// returns its elements, or nil when any step is absent or of another kind.
// The tree is user input at this point, so every step is defensive: a
// malformed document must reach the strict decoder's error, not a panic here.
func mappingSeq(root *yaml.Node, keys ...string) []*yaml.Node {
	node := root
	if node != nil && node.Kind == yaml.DocumentNode && len(node.Content) > 0 {
		node = node.Content[0]
	}
	for _, key := range keys {
		node = mappingChild(node, key)
	}
	if node == nil || node.Kind != yaml.SequenceNode {
		return nil
	}
	return node.Content
}

// mappingChild returns the value node stored under key, or nil.
func mappingChild(node *yaml.Node, key string) *yaml.Node {
	if node == nil || node.Kind != yaml.MappingNode {
		return nil
	}
	for i := 1; i < len(node.Content); i += 2 {
		if node.Content[i-1].Value == key {
			return node.Content[i]
		}
	}
	return nil
}

// mcpHeaderPathRe matches the dotted field path interpolateNode builds for an
// MCP header value. Sequence elements inherit their parent's path (there is no
// [i] in it), which is why the pattern has no index: the header NAME is what
// decides credential-ness, and it is the same decision for every server.
var mcpHeaderPathRe = regexp.MustCompile(`^mcp\.servers\.transport\.headers\.(.+)$`)

// credentialPath reports whether a dotted field path names a field whose
// ${VAR} references are credentials regardless of the variable's name (see
// EnvBinding.APIKey): provider.api_key and any sensitive MCP header.
func credentialPath(path string) bool {
	if path == apiKeyPath {
		return true
	}
	if m := mcpHeaderPathRe.FindStringSubmatch(path); m != nil {
		return SensitiveHeaderName(m[1])
	}
	return false
}
