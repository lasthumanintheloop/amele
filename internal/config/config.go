// Package config loads, interpolates and validates the single YAML file that
// defines an agent. The YAML file is the declarative entry point of amele: it
// is the only input a user needs to describe a complete agent (model,
// provider, prompts, tools and budgets).
//
// The package deliberately fails fast: unknown YAML keys, undefined
// environment variables and invalid values are reported at load time so that
// a broken config never reaches a headless (cron/CI) run.
package config

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/lasthumanintheloop/amele/internal/llm"
)

// ErrInvalid is the sentinel wrapped by every validation failure produced by
// this package. Callers map it to the frozen exit code 2 (config error).
var ErrInvalid = errors.New("invalid config")

// policyValues lists the accepted policy spellings for error messages.
const policyValues = "allow, ask, deny"

// builtinToolNames are the tool names amele owns. They are reserved against
// every other name source (subprocess tools, MCP servers) so a name means one
// thing across every config - which is what makes a permissions.tools entry
// govern what the operator thinks it governs. "shell" is included even when
// the builtin is disabled.
var builtinToolNames = []string{"fs_read", "fs_write", "fs_list", "shell"}

// toolNameRe restricts tool names to a conservative charset so that names are
// safe to embed in provider tool definitions and log lines.
var toolNameRe = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

// interpRe matches ${VAR} environment references inside the raw YAML text.
var interpRe = regexp.MustCompile(`\$\{([A-Za-z_][A-Za-z0-9_]*)\}`)

// Duration wraps time.Duration so YAML values can be written in the human
// form used across the config file ("30s", "5m").
type Duration time.Duration

// UnmarshalYAML parses a Go duration string. An empty value stays zero, which
// every consumer treats as "use the default".
func (d *Duration) UnmarshalYAML(value *yaml.Node) error {
	// A non-scalar node ("timeout: [5m]") also has an empty Value; treating
	// it as zero would silently disable a budget kill switch.
	if value.Kind != yaml.ScalarNode {
		return fmt.Errorf("duration must be a scalar string like \"30s\" or \"5m\", got a %s", nodeKindName(value.Kind))
	}
	if value.Value == "" {
		*d = 0
		return nil
	}
	parsed, err := time.ParseDuration(value.Value)
	if err != nil {
		return fmt.Errorf("parsing duration %q: %w", value.Value, err)
	}
	*d = Duration(parsed)
	return nil
}

// Std returns the wrapped standard-library duration.
func (d Duration) Std() time.Duration { return time.Duration(d) }

// nodeKindName names a YAML node kind for error messages.
func nodeKindName(k yaml.Kind) string {
	switch k {
	case yaml.SequenceNode:
		return "list"
	case yaml.MappingNode:
		return "mapping"
	case yaml.AliasNode:
		return "alias"
	default:
		return "non-scalar value"
	}
}

// The provider protocol selectors accepted by ProviderConfig.Type.
const (
	// ProviderTypeOpenAI selects the OpenAI-compatible client. It is also
	// the meaning of an empty Type: every config written before the field
	// existed keeps its behavior.
	ProviderTypeOpenAI = "openai"
	// ProviderTypeAnthropic selects the native Anthropic Messages client.
	ProviderTypeAnthropic = "anthropic"
	// ProviderTypeGemini selects the native Google Gemini generateContent
	// client. It is a THIRD wire family, not a variation of the OpenAI one:
	// the request body, the reasoning knob and the tool protocol all differ,
	// which is why Dialect (an openai-wire concept) is refused with it.
	ProviderTypeGemini = "gemini"
)

// providerTypeValues lists the accepted provider.type spellings for error
// messages.
const providerTypeValues = "openai, anthropic, gemini"

// ProviderConfig describes the LLM endpoint the agent talks to: an
// OpenAI-compatible HTTP API (the default), the native Anthropic Messages
// API, or the native Google Gemini API, selected by Type.
type ProviderConfig struct {
	// Type selects the wire protocol: "" or "openai" for OpenAI-compatible
	// endpoints, "anthropic" for the native Anthropic Messages API, "gemini"
	// for the native Google Gemini generateContent API. Empty means openai so
	// pre-existing configs keep working unchanged.
	Type string `yaml:"type"`
	// BaseURL is the API root, e.g. "https://api.openai.com/v1". With
	// type anthropic it may be empty (the client defaults to the official
	// endpoint) and must NOT include the /v1 suffix - the client appends
	// /v1/messages itself. With type gemini it may likewise be empty (the
	// client defaults to the fixed AI Studio host); a value overrides the host
	// for proxied deployments.
	BaseURL string `yaml:"base_url"`
	// APIKey is the provider credential. Each wire carries it in its own
	// header - Authorization: Bearer on the OpenAI-compatible path, x-api-key
	// with type anthropic, x-goog-api-key with type gemini - so the config
	// names the secret and the client decides where it goes. It is normally
	// injected via ${ENV_VAR} interpolation; an empty key is valid for local
	// endpoints (Ollama).
	APIKey string `yaml:"api_key"`
	// RequestTimeout bounds a single HTTP round-trip to the provider. Zero
	// means the client default (120s). Distinct from limits.timeout, which
	// bounds the whole run: slow reasoning models can need several minutes
	// for one generation.
	RequestTimeout Duration `yaml:"request_timeout"`
	// MaxOutputTokens is the per-request output ceiling. Zero means "send no
	// cap", leaving the provider default in force. It is honored on both
	// paths: the Anthropic API requires the field on every request, and on
	// the OpenAI wire the Dialect picks the field name (max_completion_tokens
	// or max_tokens). Reasoning tokens are billed against this ceiling on
	// every provider that reports them, so a reasoning agent that needs a cap
	// must leave room for the thinking as well as the answer.
	MaxOutputTokens int `yaml:"max_output_tokens"`
	// Dialect names the variation of the OpenAI-compatible wire format this
	// endpoint speaks ("deepseek", "glm", "kimi", "groq", "openrouter"), which
	// decides how the fields below are mapped onto the request. Empty means
	// the OpenAI baseline, so configs written before dialects existed keep
	// their behavior. It is ignored with type anthropic and refused with type
	// gemini (see validateProviderTuning). It is explicit rather than sniffed
	// from BaseURL:
	// guessing wrong would send a knob to the wrong field name and the
	// consequence would surface as a provider error much later.
	Dialect string `yaml:"dialect"`
	// Reasoning is the provider-neutral thinking knob. A nil block means the
	// config never asks, which is NOT the same as asking for none: most
	// reasoning models think by default and several cannot be switched off.
	Reasoning *ReasoningConfig `yaml:"reasoning"`
	// Temperature is the sampling temperature, in [0, 2]. The target can
	// narrow that: the anthropic wire and the glm dialect both accept only
	// [0, 1]. It is a pointer because 0 is a legal, useful value - the usual
	// choice for a judge agent - so "unset" must stay distinguishable from
	// "zero".
	Temperature *float64 `yaml:"temperature"`
	// TopP is nucleus sampling, in (0, 1]. Pointer for the same reason as
	// Temperature: absent means "the provider decides".
	TopP *float64 `yaml:"top_p"`
	// Retry tunes how a transient provider failure (429, 5xx, a dropped
	// connection) is retried. A nil block means the client defaults, which is
	// what every config written before the block existed gets.
	Retry *RetryConfig `yaml:"retry"`
	// Vertex, when present, points the gemini wire at Vertex AI instead of the
	// AI Studio host: a different endpoint (project- and location-addressed), a
	// different credential (Google OAuth rather than an API key) and a
	// different API version. A nil block means AI Studio, which is what every
	// gemini config written before this block carries. It is refused with any
	// other provider.type - the block describes ONE wire's endpoint.
	Vertex *VertexConfig `yaml:"vertex"`
	// Params is the escape hatch: arbitrary keys merged verbatim into the
	// request body root, for provider extras amele has no neutral field for.
	// Keys amele writes itself ON THE ACTIVE TARGET are rejected at validation
	// time (see ownedParamsKeys), so this can extend a request but never
	// rewrite one - while a key that target never writes (thinking on kimi)
	// stays reachable.
	Params map[string]any `yaml:"params"`
}

// ReasoningConfig is the provider-neutral reasoning knob. Both fields are
// optional; the dialect decides how they are mapped onto the wire, and
// `amele explain` reports the mapping (including any rounding) before a run.
type ReasoningConfig struct {
	// Effort is the reasoning depth: none, low, medium, high, xhigh or max -
	// the union of what the supported providers accept. Empty means the
	// config sends no knob at all and the provider default applies. A dialect
	// that has no equivalent for a value rounds it to the nearest one it
	// does have, visibly.
	Effort string `yaml:"effort"`
	// BudgetTokens caps the thinking with a token count instead of a level.
	// Zero means unset. Only budget-based targets can carry it (the Anthropic
	// wire, the Gemini wire's thinkingConfig.thinkingBudget, or the openrouter
	// gateway which converts it); elsewhere it is a validation error rather
	// than a silently dropped field. On the Gemini wire it is an ALTERNATIVE
	// to Effort, never a companion: the API refuses a request carrying both.
	BudgetTokens int `yaml:"budget_tokens"`
}

// VertexConfig points the gemini wire at a Vertex AI deployment: a Google
// Cloud project and the location whose regional endpoint serves it.
//
// Project and Location are both required, and neither is ever guessed: the
// location decides where the prompt is PROCESSED, which is a data-residency
// commitment rather than a routing convenience (see internal/llm's
// vertexEndpoint). amele has no default for either, so an operator can only
// get the endpoint they wrote.
type VertexConfig struct {
	// Project is the Google Cloud project id (or project number) that owns the
	// quota and the billing for the request. It travels as a URL path segment.
	Project string `yaml:"project"`
	// Location is the Vertex location: a region id ("us-central1",
	// "europe-west4"), a jurisdictional multi-region ("us", "eu"), or "global".
	// It selects BOTH the endpoint host and the locations/ path segment, and
	// amele never rewrites it - see the SECURITY note on vertexEndpoint.
	Location string `yaml:"location"`
	// Credentials optionally names a service-account key file. Empty means
	// Application Default Credentials (the GOOGLE_APPLICATION_CREDENTIALS
	// variable, then gcloud's user credentials, then the metadata server).
	// The path is resolved by the auth layer, not here: this block only
	// carries the operator's answer to "which identity".
	Credentials string `yaml:"credentials"`
}

// RetryConfig tunes the transient-failure retry loop both provider clients
// share. WHICH failures are retried is not configurable: 429, 5xx and network
// errors are transient by definition, and everything else is a request the
// provider will refuse just as firmly on the next attempt.
type RetryConfig struct {
	// MaxAttempts is the TOTAL number of tries for one provider call (1
	// initial + retries), so 1 disables retrying. Zero means the client
	// default (3).
	MaxAttempts int `yaml:"max_attempts"`
	// InitialBackoff is the wait before the second attempt; each further
	// attempt doubles it, up to a 60s ceiling per wait. A provider's
	// Retry-After header still stretches an individual wait (under the same
	// ceiling): retrying earlier than the rate limiter allows only burns the
	// attempt budget. Zero means the client default (1s).
	InitialBackoff Duration `yaml:"initial_backoff"`
}

// The bounds on provider.retry. They are wide enough for every rhythm an
// operator has a reason to ask for and narrow enough that a slipped digit
// (max_attempts: 100, initial_backoff: 10m) cannot turn one rate-limited turn
// into an unattended run that looks hung. They bound each knob, not their
// product: the wall-clock kill switch for a whole run stays limits.timeout,
// which cuts a backoff wait short like any other wait.
const (
	retryMinAttempts = 1
	retryMaxAttempts = 10
	retryMinBackoff  = 100 * time.Millisecond
	retryMaxBackoff  = 60 * time.Second
)

// effortValues lists the accepted provider.reasoning.effort spellings, in
// increasing depth, for validation and for error messages.
//
// CONTRACT: this is the UNION of the vocabularies of the supported providers -
// no single provider accepts all six. Values a dialect cannot express are
// rounded by the wire mapping, not rejected here, so one config can move
// between providers without being rewritten.
var effortValues = []string{"none", "low", "medium", "high", "xhigh", "max"}

// reservedWireFields are request-body keys provider.params must not carry even
// though NO target writes them, so they are not "owned" in the sense
// llm.OwnedWireFields means.
//
// CONTRACT: they are refused because amele's own machinery cannot survive them,
// not because they would clobber a field it writes:
//   - stream: the clients read a single JSON body. An SSE stream decodes as a
//     parse error, so the whole run would die on the first turn (streaming is a
//     later roadmap slice, README §Roadmap).
//   - tool_choice: the loop owns the tool protocol - it offers the tools and
//     stops when the model answers without calling one. A pinned "required"
//     removes that stopping condition and every run ends at max_turns (exit 3).
var reservedWireFields = []string{"stream", "tool_choice"}

// SubprocessTool declares an external executable the model may invoke as a
// tool. The command is a fixed argv vector: there is no shell involved, so
// the model can never inject into a shell string.
type SubprocessTool struct {
	// Name is the tool name exposed to the model.
	Name string `yaml:"name"`
	// Description tells the model what the tool does and when to use it.
	Description string `yaml:"description"`
	// Command is the argv vector to execute. Command[0] is the executable.
	Command []string `yaml:"command"`
	// Timeout bounds a single invocation. Zero means the default (60s).
	Timeout Duration `yaml:"timeout"`
	// AllowArgs lets the model append extra argv elements after Command.
	// SECURITY: disabled by default - with it off the model controls stdin
	// only, never the argument vector.
	AllowArgs bool `yaml:"allow_args"`
	// Env, when non-empty, is the environment allowlist for this tool's
	// process: the child sees ONLY the listed variables (plus PATH, HOME and
	// LANG, which are always passed so base tools function), with values taken
	// from amele's own environment; listed names that are unset are silently
	// skipped. Empty means the child inherits amele's entire environment
	// (backward compatible).
	// SECURITY: the allowlist reduces credential exposure to model-driven
	// commands, but the boundary remains the OS/container the agent runs in.
	Env []string `yaml:"env"`
}

// ShellConfig configures the builtin `shell` tool, which runs a model-written
// command line through `sh -c` in the workspace.
//
// SECURITY: the tool is off unless Enabled is explicitly true - the zero value
// (an absent `tools.shell:` block) never grants a shell. Allow/Deny are an
// ACCIDENT-PREVENTION layer, not a security boundary: a pattern like "git *"
// still reaches arbitrary code through `git -c core.fsmonitor=cmd`, aliases and
// hooks. The security boundary is the OS/container the agent runs in
// (docs/threat-model.md).
type ShellConfig struct {
	// Enabled turns the tool on. Default (false) means the model has no shell.
	Enabled bool `yaml:"enabled"`
	// Allow lists glob patterns (`*` matches any substring) matched against
	// the whole command string. An empty list allows everything Deny does not
	// catch; a non-empty list requires at least one match.
	Allow []string `yaml:"allow"`
	// Deny lists glob patterns checked BEFORE Allow. Any match rejects the
	// command.
	Deny []string `yaml:"deny"`
	// Env, when non-empty, is the environment allowlist for shell commands:
	// the child sees ONLY the listed variables (plus PATH, HOME and LANG,
	// which are always passed so base tools function), with values taken from
	// amele's own environment; listed names that are unset are silently
	// skipped. Empty means the child inherits amele's entire environment
	// (backward compatible).
	// SECURITY: the allowlist reduces credential exposure to model-written
	// commands (e.g. `printenv API_KEY`), but the boundary remains the
	// OS/container the agent runs in.
	Env []string `yaml:"env"`
	// Timeout bounds a single command. Zero means the default (60s).
	Timeout Duration `yaml:"timeout"`
}

// ToolsConfig selects which builtin tools are enabled and declares custom
// subprocess tools.
type ToolsConfig struct {
	// FS enables the builtin sandboxed filesystem tools
	// (fs_read/fs_write/fs_list).
	FS bool `yaml:"fs"`
	// Shell configures the builtin `shell` tool. Absent means disabled.
	Shell ShellConfig `yaml:"shell"`
	// Subprocess lists the custom executable tools.
	Subprocess []SubprocessTool `yaml:"subprocess"`
	// Parallel decides whether the tool calls of ONE model turn may run
	// concurrently. nil (the field omitted) means true; see IsParallel.
	Parallel *bool `yaml:"parallel"`
}

// IsParallel reports whether the tool calls within one turn may run at the
// same time.
//
// The zero value (nil) means yes: a model that asks for three independent
// reads in one turn should not pay for them one after the other, and every
// tool amele dispatches is already an isolated process or request. An operator
// whose tools are NOT independent - two subprocess tools writing the same file,
// a server that serializes badly - opts out with `parallel: false`, and the
// whole turn goes back to one call at a time.
//
// Concurrency never changes what is recorded: the session events, the message
// history and the progress feed stay in the model's call order either way
// (docs/contracts/jsonl-events.md).
func (t ToolsConfig) IsParallel() bool { return t.Parallel == nil || *t.Parallel }

// Limits are the run budgets. They are the kill switches that make an agent
// safe to leave unattended in cron.
type Limits struct {
	// MaxTurns bounds the number of provider round-trips. Zero means the
	// default (20).
	MaxTurns int `yaml:"max_turns"`
	// MaxTokens bounds the cumulative input+output tokens as reported by
	// the provider. Zero disables the token budget.
	// CONTRACT: tokens are the primary budget unit; USD is informational
	// only (docs/engineering.md §7).
	MaxTokens int `yaml:"max_tokens"`
	// Timeout bounds the whole run wall-clock. Zero disables it.
	Timeout Duration `yaml:"timeout"`
}

// OutputConfig constrains the run's final answer to a JSON Schema.
type OutputConfig struct {
	// Schema is an inline JSON Schema written as a YAML mapping. Empty (nil
	// or {}) means no constraint: the run's final answer is unconstrained
	// plain text, same as Phase 1.
	Schema map[string]any `yaml:"schema"`
	// MaxSchemaRetries bounds how many times the model gets validator
	// feedback before the run fails with exit 6. Zero (unset) means "use the
	// default" (2); the default itself is applied by cmd, not here, so a
	// config round-tripped through Load+Validate+re-marshal never gains an
	// explicit value the user didn't write.
	MaxSchemaRetries int `yaml:"max_schema_retries"`
}

// SchemaJSON marshals Schema to JSON Schema text for compilation by
// internal/schema. A nil or empty map returns (nil, nil): "no schema" and
// "empty schema" are the same "no constraint" case, and callers can treat a
// nil result as license to skip schema compilation entirely.
//
// CONTRACT: config does not import internal/schema and never compiles or
// validates the schema itself - only cmd does (Task 5), mapping a compile
// failure to exit 2. Keeping config unaware of the schema package keeps the
// dependency graph a DAG (schema is a leaf package per its own doc comment)
// and keeps config's job narrow: parse YAML, don't interpret JSON Schema
// semantics.
func (c *Config) SchemaJSON() ([]byte, error) {
	if len(c.Output.Schema) == 0 {
		return nil, nil
	}
	raw, err := json.Marshal(c.Output.Schema)
	if err != nil {
		return nil, fmt.Errorf("marshaling output.schema: %w", err)
	}
	return raw, nil
}

// Policy is the permission ruling configured for a tool.
//
// CONTRACT: the three values below are the whole vocabulary; Validate rejects
// anything else, so a typo ("auto_approve") is a config error (exit 2) rather
// than a silently-wrong policy on an unattended run.
//
// The type lives in config, not in internal/perm, because config is where the
// YAML schema and its validation live - and perm imports config to read the
// block, so the reverse dependency would be an import cycle.
type Policy string

// The permission policies.
const (
	// PolicyAllow runs the tool without asking.
	PolicyAllow Policy = "allow"
	// PolicyAsk asks the human. SECURITY: with no TTY there is no human, so
	// consumers must degrade this to a denial (docs/engineering.md §5.5).
	PolicyAsk Policy = "ask"
	// PolicyDeny refuses the call. The run continues with the remaining
	// tools: by design a denial is information to the model, not a crash.
	PolicyDeny Policy = "deny"
)

// Valid reports whether p is one of the three known policies. The empty
// string is NOT valid: only Permissions.Default treats "unset" as allow, and
// it checks for the empty value explicitly.
func (p Policy) Valid() bool {
	switch p {
	case PolicyAllow, PolicyAsk, PolicyDeny:
		return true
	default:
		return false
	}
}

// Permissions is the per-run permission profile: one fallback policy plus
// per-tool overrides.
//
// An absent block (the zero value) means allow-everything, which is Phase 1
// parity: adding the feature must not silently break configs written before
// it existed.
type Permissions struct {
	// Default applies to any tool without an entry in Tools. Empty means
	// allow.
	Default Policy `yaml:"default"`
	// Tools maps tool name to policy, overriding Default.
	Tools map[string]Policy `yaml:"tools"`
}

// Config is the root of the agent YAML file.
type Config struct {
	// Model is the model identifier sent to the provider. Required unless
	// overridden on the CLI (--model).
	Model string `yaml:"model"`
	// Provider selects the LLM endpoint.
	Provider ProviderConfig `yaml:"provider"`
	// SystemPrompt is the inline system prompt. Mutually exclusive with
	// SystemPromptFile.
	SystemPrompt string `yaml:"system_prompt"`
	// SystemPromptFile loads the system prompt from a file relative to the
	// config file, keeping long prompts out of the YAML.
	SystemPromptFile string `yaml:"system_prompt_file"`
	// Prompt is the user-message template. It may reference {{args}} (CLI
	// task text) and {{input}} (stdin). Empty means a sensible default
	// combining both.
	Prompt string `yaml:"prompt"`
	// Workspace is the directory sandbox for fs tools and the working
	// directory for subprocess tools. Defaults to the config file's
	// directory.
	Workspace string `yaml:"workspace"`
	// Tools configures builtin and custom tools.
	Tools ToolsConfig `yaml:"tools"`
	// Limits are the run budgets.
	Limits Limits `yaml:"limits"`
	// SessionDir enables JSONL session logging when non-empty. Relative
	// paths are resolved against the config file's directory.
	SessionDir string `yaml:"session_dir"`
	// Lock makes `amele run` single-flight per config file: before doing
	// anything with the provider it takes a non-blocking advisory lock on
	// "<config path>.lock", and a run that finds it held exits 7 immediately.
	// Only `run` locks - validate/explain/chat never do.
	//
	// It defaults to false (opt-in) because locking by default would break a
	// supported pattern: the same YAML run concurrently with different tasks
	// (the LLM-judge recipe fans one config out over several inputs). Cron
	// users, whose risk is an overrunning run overlapping the next tick, set
	// it to true.
	Lock bool `yaml:"lock"`
	// Output constrains the run's final answer to a JSON Schema.
	Output OutputConfig `yaml:"output"`
	// Permissions is the tool approval profile. An absent block allows every
	// tool (Phase 1 parity).
	Permissions Permissions `yaml:"permissions"`
	// MCP declares MCP servers whose tools are offered to the model as
	// "<server>__<tool>". An absent block means no MCP connection is made.
	MCP MCPConfig `yaml:"mcp"`

	// promptConflict records that the file set both SystemPrompt and
	// SystemPromptFile. It is derived at load time because applyDefaults
	// overwrites SystemPrompt with the file's content on the valid path, after
	// which the two fields are indistinguishable from a legal
	// system_prompt_file config. Unexported: derived state, not part of the
	// YAML schema. Violations reports it; see applyDefaults for why the load
	// itself no longer fails.
	promptConflict bool

	// envBindings records every ${VAR} reference the file made, in
	// first-appearance order and deduplicated by name. It is the single source
	// behind InterpolatedSecrets, EnvReferenced and EnvMissing - one structure
	// instead of three parallel slices, so the views cannot drift apart.
	// Unexported: derived state, not part of the YAML schema.
	envBindings []EnvBinding
}

// EnvBinding is one ${VAR} reference resolved while loading a config: what the
// file asked for, and what the environment answered.
//
// SECURITY: Value may be a secret (a DB password in a prompt, not just the API
// key), so a consumer that prints it must apply its own display policy;
// internal/explain does, and internal/session redacts every value
// unconditionally.
type EnvBinding struct {
	// Name is the variable name as written between ${ and }.
	Name string
	// Value is what the environment returned. Empty when Missing, and also
	// empty when the variable is defined-but-empty.
	Value string
	// Missing reports that the variable was undefined at load time. Only
	// LoadTolerant can produce it; Load fails instead.
	Missing bool
	// APIKey reports that this variable was referenced by provider.api_key.
	// It is recorded during interpolation because it cannot be recovered
	// afterwards: the field is never printed, but the same value may appear
	// again somewhere that is (a subprocess argv, a shell pattern), and that
	// occurrence must still be treated as a credential.
	APIKey bool
}

// InterpolatedSecrets returns every environment value that was substituted
// into the config, deduplicated and without empties, for registration with
// the session log's redactor.
func (c *Config) InterpolatedSecrets() []string {
	values := make([]string, 0, len(c.envBindings))
	seen := make(map[string]bool, len(c.envBindings))
	for _, b := range c.envBindings {
		if b.Value == "" || seen[b.Value] {
			continue
		}
		seen[b.Value] = true
		values = append(values, b.Value)
	}
	return values
}

// EnvBindings returns every ${VAR} reference the config made, in
// first-appearance order, deduplicated by name. It is the detailed view behind
// EnvReferenced/EnvMissing, for callers (internal/explain) that must decide
// per variable whether its value may be displayed.
func (c *Config) EnvBindings() []EnvBinding {
	return append([]EnvBinding(nil), c.envBindings...)
}

// EnvReferenced returns every ${VAR} name the config references, in
// first-appearance order, deduplicated.
func (c *Config) EnvReferenced() []string {
	names := make([]string, 0, len(c.envBindings))
	for _, b := range c.envBindings {
		names = append(names, b.Name)
	}
	return names
}

// EnvMissing returns the referenced names that were undefined at load
// time. Always empty after Load (which fails on missing); populated only
// by LoadTolerant.
func (c *Config) EnvMissing() []string {
	var names []string
	for _, b := range c.envBindings {
		if b.Missing {
			names = append(names, b.Name)
		}
	}
	return names
}

// LookupEnv is the environment access used for ${VAR} interpolation. It is
// injected (instead of calling os.LookupEnv directly) so tests are hermetic.
type LookupEnv func(key string) (string, bool)

// Load reads, interpolates and parses the YAML file at path, then applies
// defaults. It does NOT validate; call Validate after applying any CLI
// overrides (e.g. --model) so overrides participate in validation.
func Load(path string, env LookupEnv) (*Config, error) { return load(path, env, false) }

// LoadTolerant is Load for inspection commands: an undefined ${VAR} does
// not fail the load - it substitutes as an empty string and is recorded in
// EnvMissing, so `explain` can tell a new user which variables to set
// before the config can run. Run/validate keep using Load: headless runs
// must fail loudly on missing env, not limp on with empty secrets.
func LoadTolerant(path string, env LookupEnv) (*Config, error) { return load(path, env, true) }

// load is the shared implementation behind Load and LoadTolerant. tolerant
// controls whether an undefined ${VAR} fails the load (false, Load's
// contract) or substitutes as "" and is merely recorded (true, LoadTolerant's
// contract).
func load(path string, env LookupEnv, tolerant bool) (*Config, error) {
	raw, err := os.ReadFile(path) //nolint:gosec // G304: loading the user-named config file is this function's purpose.
	if err != nil {
		return nil, fmt.Errorf("%w: reading %s: %v", ErrInvalid, path, err)
	}

	// SECURITY: literal credentials in YAML are rejected before anything
	// else - a key committed to git is already leaked, and interpolation
	// would otherwise mask the difference.
	if err := rejectLiteralAPIKey(raw); err != nil {
		return nil, fmt.Errorf("%w: %s: %v", ErrInvalid, path, err)
	}

	// Parse FIRST, interpolate afterwards on the node tree. Substituting
	// into the raw text would make ${VAR} inside comments a load failure and
	// let env values containing YAML metacharacters (newlines, ": ") alter
	// the document structure. Post-parse, an env value is always data.
	docs := yaml.NewDecoder(bytes.NewReader(raw))
	var doc yaml.Node
	if err := docs.Decode(&doc); err != nil {
		if errors.Is(err, io.EOF) {
			return nil, fmt.Errorf("%w: parsing %s: file is empty", ErrInvalid, path)
		}
		// Note: yaml.v3 sometimes reports the line number off by one;
		// upstream bug, don't fiddle with it.
		return nil, fmt.Errorf("%w: parsing %s: %v", ErrInvalid, path, err)
	}
	if doc.Kind == 0 {
		return nil, fmt.Errorf("%w: parsing %s: file is empty", ErrInvalid, path)
	}
	// CONTRACT: an agent is exactly ONE YAML document. Only the first was ever
	// decoded, so a second one silently dropped whatever it held - budgets, or
	// a "permissions.default: deny" block whose absence leaves the
	// allow-everything default in force. Refuse the file instead.
	if err := docs.Decode(new(yaml.Node)); !errors.Is(err, io.EOF) {
		if err != nil {
			return nil, fmt.Errorf("%w: parsing %s: %v", ErrInvalid, path, err)
		}
		return nil, fmt.Errorf("%w: parsing %s: file contains multiple YAML documents (separated by ---); an agent config must be a single document, so merge them into one", ErrInvalid, path)
	}

	// SECURITY: same rule as provider.api_key, enforced on the parsed but
	// still un-interpolated tree: after substitution a literal token and a
	// ${VAR} reference are the same string.
	if err := rejectLiteralMCPHeaders(&doc); err != nil {
		return nil, fmt.Errorf("%w: %s: %v", ErrInvalid, path, err)
	}

	bindings, err := interpolateNode(&doc, env, tolerant)
	if err != nil {
		return nil, fmt.Errorf("%w: %s: %v", ErrInvalid, path, err)
	}

	// Re-encode the interpolated tree and strict-decode that: the encoder
	// quotes substituted values as needed, and KnownFields still rejects
	// typo'd keys. (Line numbers in unknown-key errors refer to the
	// re-encoded document, so they can drift slightly from the source file;
	// the field name in the message is the reliable pointer.)
	interpolated, err := yaml.Marshal(&doc)
	if err != nil {
		return nil, fmt.Errorf("%w: parsing %s: %v", ErrInvalid, path, err)
	}

	cfg := &Config{envBindings: bindings}
	dec := yaml.NewDecoder(bytes.NewReader(interpolated))
	// Unknown keys are almost always typos ("max_token" instead of
	// "max_tokens") that would otherwise silently disable a budget, so
	// strict decoding is a safety feature, not pedantry.
	dec.KnownFields(true)
	if err := dec.Decode(cfg); err != nil {
		return nil, fmt.Errorf("%w: parsing %s: %v", ErrInvalid, path, err)
	}

	// The base directory is absolutized before any join: filepath.Dir of a
	// user-typed relative config path is itself relative, and joining tool
	// paths onto a relative base would leave them dependent on the child
	// process working directory (cmd.Dir) instead of the config location.
	baseDir, err := filepath.Abs(filepath.Dir(path))
	if err != nil {
		return nil, fmt.Errorf("%w: resolving config directory of %s: %v", ErrInvalid, path, err)
	}
	if err := cfg.applyDefaults(baseDir, tolerant); err != nil {
		return nil, err
	}
	return cfg, nil
}

// rejectLiteralAPIKey probes the raw (pre-interpolation) YAML and fails when
// provider.api_key holds anything other than an ${ENV_VAR} reference. Uses a
// lenient parse over just this field: other fields may legitimately contain
// text that only becomes valid for the strict schema after interpolation.
func rejectLiteralAPIKey(raw []byte) error {
	var probe struct {
		Provider struct {
			APIKey string `yaml:"api_key"`
		} `yaml:"provider"`
	}
	if err := yaml.Unmarshal(raw, &probe); err != nil {
		return nil // let the strict decoder produce the real parse error
	}
	// Stripping every ${VAR} reference (and the "$$" escape) must leave
	// nothing: "sk-live-secret-${SUFFIX}" is still a committed secret even
	// though it contains a reference.
	stripped := interpRe.ReplaceAllString(probe.Provider.APIKey, "")
	stripped = strings.ReplaceAll(stripped, "$$", "")
	if stripped != "" {
		return errors.New("provider.api_key must be built only from environment references (api_key: ${MY_API_KEY}); literal secrets in YAML are forbidden")
	}
	return nil
}

// apiKeyPath is the dotted field path of provider.api_key, whose references
// are marked as credentials regardless of the variable's name (see
// EnvBinding.APIKey and credentialPath, which also covers sensitive MCP
// headers).
const apiKeyPath = "provider.api_key"

// interpolateNode substitutes ${VAR} references inside every scalar VALUE of
// the parsed YAML tree and returns one EnvBinding per referenced variable, in
// first-appearance order and deduplicated by name.
// Referencing an undefined variable is an error UNLESS tolerant is true: in a
// headless run a silently-empty API key would surface much later as a
// confusing provider error, but inspection commands (LoadTolerant) need to
// keep going so they can report which variables are missing.
//
// SECURITY: mapping KEYS are deliberately left untouched. Substituting into a
// key lets a config name a field indirectly, and the literal-credential ban is
// enforced on the raw pre-interpolation file (rejectLiteralAPIKey), so
// `"${KEY}": sk-live-...` with KEY=api_key would have walked a plaintext
// credential past the guard (docs/engineering.md §5.5). Keys name schema fields, which
// are a fixed vocabulary the strict decoder already validates, so there is no
// legitimate use to lose: an uninterpolated key that isn't a schema field is
// rejected as an unknown key (exit 2).
func interpolateNode(root *yaml.Node, env LookupEnv, tolerant bool) ([]EnvBinding, error) {
	var bindings []EnvBinding
	index := map[string]int{} // name -> position in bindings, for deduplication

	// record merges one reference into the binding list. A name seen twice
	// keeps its first position but ORs the api_key flag in: `api_key: ${TOKEN}`
	// plus `command: [..., "${TOKEN}"]` is one variable that is a credential.
	record := func(name string, value string, missing, apiKey bool) {
		if i, ok := index[name]; ok {
			bindings[i].APIKey = bindings[i].APIKey || apiKey
			return
		}
		index[name] = len(bindings)
		bindings = append(bindings, EnvBinding{Name: name, Value: value, Missing: missing, APIKey: apiKey})
	}

	// path is the dotted field path of the node being walked, so a reference
	// can be attributed to provider.api_key. Sequence elements keep their
	// parent's path: no api_key field is a list, and an index in the path
	// would buy nothing.
	var walk func(n *yaml.Node, path string)
	walk = func(n *yaml.Node, path string) {
		if n.Kind == yaml.ScalarNode {
			replaced, refs := interpolateString(n.Value, env, tolerant)
			for _, r := range refs {
				record(r.name, r.value, r.missing, credentialPath(path))
			}
			if replaced != n.Value {
				n.Value = replaced
				// A plain scalar's tag was resolved from the placeholder
				// text; drop it so the encoder re-resolves from the new
				// value ("${N}" -> "5000" must still decode as an int, as
				// it did when substitution happened in the raw text).
				// Quoted scalars keep their explicit string identity.
				if n.Style == 0 {
					n.Tag = ""
				}
			}
			return
		}
		if n.Kind == yaml.MappingNode {
			// Content is [key, value, key, value, ...]; only the values are
			// interpolated (see the SECURITY note above).
			for i := 1; i < len(n.Content); i += 2 {
				child := n.Content[i-1].Value // the key names the field
				if path != "" {
					child = path + "." + child
				}
				walk(n.Content[i], child)
			}
			return
		}
		for _, child := range n.Content {
			walk(child, path)
		}
	}
	walk(root, "")

	if !tolerant {
		var missing []string
		for _, b := range bindings {
			if b.Missing {
				missing = append(missing, b.Name)
			}
		}
		if len(missing) > 0 {
			return nil, fmt.Errorf("undefined environment variable(s): %s", strings.Join(missing, ", "))
		}
	}
	return bindings, nil
}

// envRef is one ${VAR} occurrence resolved by interpolateString: the name as
// written, what the environment answered, and whether it answered at all.
type envRef struct {
	name    string
	value   string
	missing bool
}

// interpolateString substitutes ${VAR} references in a single scalar value.
// "$$" escapes a literal "$" so prompts can talk about ${} syntax without
// triggering interpolation. Every occurrence is reported in refs, in the order
// it appears, whether or not the variable was defined. An undefined name's
// replacement is the original "${VAR}" text when tolerant is false (Load's
// contract: the caller turns missing into a load error before the replacement
// matters) or "" when tolerant is true (LoadTolerant's contract: a missing
// secret becomes an empty string so the rest of the config can still be
// inspected).
func interpolateString(text string, env LookupEnv, tolerant bool) (out string, refs []envRef) {
	const escape = "\x00amele-dollar\x00"
	text = strings.ReplaceAll(text, "$$", escape)

	replaced := interpRe.ReplaceAllStringFunc(text, func(match string) string {
		name := interpRe.FindStringSubmatch(match)[1]
		value, ok := env(name)
		if !ok {
			refs = append(refs, envRef{name: name, missing: true})
			if tolerant {
				return ""
			}
			return match
		}
		refs = append(refs, envRef{name: name, value: value})
		return value
	})
	return strings.ReplaceAll(replaced, escape, "$"), refs
}

// applyDefaults resolves paths against the config file directory and fills
// zero values with the documented defaults. When tolerant is true (the
// LoadTolerant path) AND the load also has missing ${VAR}s, a
// system_prompt_file that fails to read is skipped instead of failing the
// whole load: the path itself may contain a missing ${VAR} substituted as
// "", and the requirements report (explain) must still render even though
// the prompt can't be resolved yet. Without a missing env var, the file name
// is exactly what the author wrote, so an unreadable prompt file is a real
// config error and the load fails: it is a FILE explain could not read, which
// is the one thing `amele explain` still refuses to report on (exit 2). The
// skip therefore cannot be unconditional on tolerant alone, or the missing
// prompt would go unmentioned by every command.
func (c *Config) applyDefaults(baseDir string, tolerant bool) error {
	if c.Limits.MaxTurns == 0 {
		c.Limits.MaxTurns = 20
	}
	if c.Workspace == "" {
		c.Workspace = baseDir
	} else if !filepath.IsAbs(c.Workspace) {
		c.Workspace = filepath.Join(baseDir, c.Workspace)
	}
	if c.SessionDir != "" && !filepath.IsAbs(c.SessionDir) {
		c.SessionDir = filepath.Join(baseDir, c.SessionDir)
	}

	// CONTRACT: path-like relative commands resolve against the config file's
	// directory; bare names resolve from PATH at exec time. This is what makes
	// a shared pack folder relocatable: "./tools/x.sh" keeps working no matter
	// where the workspace points or what the caller's cwd is. filepath.Separator
	// is checked alongside '/' so pack YAML written with portable forward
	// slashes is recognized on every platform.
	// The same rule governs an MCP stdio server's executable: a pack that
	// ships its own server must keep working wherever the folder is copied.
	for i := range c.Tools.Subprocess {
		resolveCommandBase(c.Tools.Subprocess[i].Command, baseDir)
	}
	for i := range c.MCP.Servers {
		resolveCommandBase(c.MCP.Servers[i].Transport.Command, baseDir)
	}

	return c.resolveSystemPrompt(baseDir, tolerant)
}

// resolveCommandBase rewrites a path-like relative command[0] in place so it
// resolves against baseDir (the config file's directory). Absolute paths, bare
// names (resolved from PATH at exec time) and empty vectors are left alone.
func resolveCommandBase(cmd []string, baseDir string) {
	if len(cmd) == 0 || cmd[0] == "" || filepath.IsAbs(cmd[0]) {
		return
	}
	if strings.ContainsRune(cmd[0], '/') || strings.ContainsRune(cmd[0], filepath.Separator) {
		cmd[0] = filepath.Join(baseDir, cmd[0])
	}
}

// resolveSystemPrompt loads system_prompt_file into SystemPrompt, resolving
// the path against the config file's directory. Setting both prompt forms is
// recorded as a violation rather than returned: failing the load here hid
// every other violation in the same file behind this one (see Violations).
// The file is deliberately NOT read in that case - the config cannot run
// either way, and a missing prompt file would abort the load and restore the
// very short-circuit this removes. The inline prompt stays as written so the
// report shows what the author wrote rather than a resolution amele invented.
func (c *Config) resolveSystemPrompt(baseDir string, tolerant bool) error {
	if c.SystemPromptFile == "" {
		return nil
	}
	if c.SystemPrompt != "" {
		c.promptConflict = true
		return nil
	}
	// Resolved against the config file's directory here; a --set override
	// of the same field resolves against the caller's working directory
	// instead (see ApplyOverrides) but shares this reader, so the two paths
	// cannot drift in how they read a prompt file.
	_, content, err := readPromptFile(c.SystemPromptFile, baseDir)
	if err != nil {
		if tolerant && len(c.EnvMissing()) > 0 {
			return nil
		}
		return fmt.Errorf("%w: reading system_prompt_file: %v", ErrInvalid, err)
	}
	c.SystemPrompt = content
	return nil
}

// Validate checks every semantic rule of the schema and returns all
// violations at once (joined), so users fix the file in one pass instead of
// replaying cron failures one error at a time. It is Violations with an
// ErrInvalid-wrapped message; the two can never disagree.
func (c *Config) Validate() error {
	msgs := c.Violations()
	if len(msgs) == 0 {
		return nil
	}
	// Indent every violation, not just the first: errors.Join renders with
	// plain "\n" separators, which would leave violations 2..N flush-left
	// under the "invalid config:" header. The contract (docs/engineering.md §7,
	// docs/contracts) is that validate reports every violation at once and
	// readably, so each line carries the same two-space indent.
	return fmt.Errorf("%w:\n  %s", ErrInvalid, strings.Join(msgs, "\n  "))
}

// Violations returns every semantic rule violation as its own message, in a
// fixed order, or nil for a valid config.
//
// It exists next to Validate because `explain` reports problems as individual
// lines of a report section rather than as one error blob: re-splitting
// Validate's joined message would make the report's shape depend on that
// message's formatting.
func (c *Config) Violations() []string {
	var msgs []string
	add := func(format string, args ...any) {
		msgs = append(msgs, fmt.Sprintf(format, args...))
	}

	if c.Model == "" {
		add("model is required (set it in the YAML or via --model)")
	}
	if c.promptConflict {
		add("system_prompt and system_prompt_file are mutually exclusive")
	}
	c.validateProvider(add)

	if c.Limits.MaxTurns < 0 {
		add("limits.max_turns must not be negative")
	}
	if c.Limits.MaxTokens < 0 {
		add("limits.max_tokens must not be negative")
	}
	if c.Limits.Timeout < 0 {
		add("limits.timeout must not be negative")
	}

	if info, err := os.Stat(c.Workspace); err != nil {
		add("workspace %q is not accessible: %v", c.Workspace, err)
	} else if !info.IsDir() {
		add("workspace %q is not a directory", c.Workspace)
	}

	if c.Output.MaxSchemaRetries < 0 {
		add("output.max_schema_retries must not be negative")
	}

	c.validateSubprocessTools(add)
	c.validateShell(add)
	c.validatePermissions(add)
	c.validateMCP(add)

	return msgs
}

// validateProvider checks the provider block. Split from Validate to keep
// each function under the complexity budget.
func (c *Config) validateProvider(add func(format string, args ...any)) {
	switch c.Provider.Type {
	case "", ProviderTypeOpenAI, ProviderTypeAnthropic, ProviderTypeGemini:
	default:
		add("provider.type %q is not a valid provider type (%s, or omit for openai)", c.Provider.Type, providerTypeValues)
	}
	anthropic := c.Provider.Type == ProviderTypeAnthropic

	if c.Provider.BaseURL == "" {
		// base_url stays required on the OpenAI-compatible path: there is no
		// canonical gateway address to default to (OpenAI, OpenRouter, Ollama
		// all differ). The Anthropic and Gemini paths are exempt because each
		// has ONE official host, fixed by its vendor, and the client falls back
		// to it on its own.
		if !anthropic && c.Provider.Type != ProviderTypeGemini {
			add("provider.base_url is required")
		}
	} else if problem := baseURLProblem(c.Provider.BaseURL); problem != "" {
		add("provider.base_url %q %s", c.Provider.BaseURL, problem)
	} else if problem := c.hostOrVersionProblem(); problem != "" {
		add("provider.base_url %q %s", c.Provider.BaseURL, problem)
	}

	if c.Provider.Type == ProviderTypeGemini && c.Provider.APIKey == "" && c.Provider.Vertex == nil {
		// The Gemini API has two auth paths and this wire speaks both: the AI
		// Studio key, and Vertex's Google credentials (named by the vertex
		// block, which carries the project and location rather than a secret).
		// With neither, the request would reach the wire unauthenticated and
		// come back a 401 from a run nobody was watching - so both successors
		// are NAMED here rather than left for the operator to infer from the
		// endpoint's refusal.
		//
		// Scoped to this wire: a keyless openai-compatible config is a local
		// Ollama server, which is a supported deployment.
		add("provider.api_key: gemini needs api_key (AI Studio) or a vertex block (Vertex AI)")
	}
	c.validateVertex(add)

	if c.Provider.RequestTimeout < 0 {
		add("provider.request_timeout must not be negative")
	}
	if c.Provider.MaxOutputTokens < 0 {
		add("provider.max_output_tokens must not be negative")
	}
	c.validateRetry(add)
	c.validateProviderTuning(add)
}

// validateVertex checks the Vertex AI block: presence relative to the wire, the
// two required coordinates, and the charset that keeps them addressable.
//
// CONTRACT: the charset is not restated here - llm.ValidVertexID owns it, and
// this function calls the same predicate the client will apply at request time.
// A second copy would be a config that passes `amele validate` and then fails
// its CONFIGURATION mid-run the day one side is loosened, which the frozen
// exit-code contract forbids (docs/engineering.md §7: exit 2 is decided before
// the run, never during it). The SECURITY reasoning behind the charset lives
// with the definition.
func (c *Config) validateVertex(add func(format string, args ...any)) {
	v := c.Provider.Vertex
	if v == nil {
		return
	}
	if c.Provider.Type != ProviderTypeGemini {
		// Refused rather than ignored: a config carrying a vertex block reads as
		// a Vertex deployment to everyone who opens it, and running it against
		// another wire would honor the file's letter while breaking its intent.
		add("provider.vertex is only valid with provider.type: gemini")
		return
	}
	if c.Provider.APIKey != "" {
		// The two credentials are alternatives, and not merely redundant
		// together: Vertex REFUSES API keys outright (live-verified - see
		// docs/superpowers/specs/2026-08-25-vertex-adc-research.md §3.3), so a
		// config naming both would look authenticated and fail at the endpoint.
		add("provider.vertex must not be combined with provider.api_key: vertex authenticates with google credentials (application default credentials, or vertex.credentials), and the AI Studio key is not accepted there")
	}
	if v.Project == "" {
		add("provider.vertex.project is required (the google cloud project id or number that owns the quota)")
	} else if !llm.ValidVertexID(v.Project) {
		add("provider.vertex.project %q must be a lowercase project id or project number (letters, digits and hyphens): it becomes a path segment of the endpoint", v.Project)
	}
	if v.Location == "" {
		// No default: the location decides where the prompt is processed, so
		// guessing one (us-central1, the historical habit) would be a residency
		// decision amele made on the operator's behalf - and it does not even
		// serve the current Gemini models.
		add("provider.vertex.location is required (e.g. us-central1, europe-west4, or global)")
	} else if !llm.ValidVertexID(v.Location) {
		add("provider.vertex.location %q must be a lowercase region id like us-central1 or global (letters, digits and hyphens): it becomes part of the endpoint host", v.Location)
	}
}

// validateRetry checks the retry policy bounds. Zero is "omitted" for both
// fields - the struct cannot tell an absent key from a written 0, and every
// other budget in this file spells "use the default" that way - so only a value
// the operator actually chose is range-checked.
func (c *Config) validateRetry(add func(format string, args ...any)) {
	r := c.Provider.Retry
	if r == nil {
		return
	}
	if r.MaxAttempts != 0 && (r.MaxAttempts < retryMinAttempts || r.MaxAttempts > retryMaxAttempts) {
		add("provider.retry.max_attempts must be between %d and %d (got %d; omit for the default 3, or set 1 to disable retrying)",
			retryMinAttempts, retryMaxAttempts, r.MaxAttempts)
	}
	// The bounds are spelled literally rather than formatted from the constants
	// above: time.Duration.String() renders the ceiling as "1m0s", and an error
	// message must show the units the operator typed in the file.
	if backoff := r.InitialBackoff.Std(); backoff != 0 && (backoff < retryMinBackoff || backoff > retryMaxBackoff) {
		add("provider.retry.initial_backoff must be between 100ms and 60s (got %s; omit for the default 1s)", backoff)
	}
}

// validateProviderTuning checks the dialect and the knobs whose legality
// depends on it (reasoning, sampling, raw params).
//
// CONTRACT: only rules that are TOTAL for a dialect live here - what the
// provider rejects for every model it serves. Model-dependent rules (a model
// that refuses a temperature its siblings accept) stay runtime errors with
// mapped messages: a config that passes validate must not be rejected for its
// CONFIGURATION at run time, but validate must not refuse a combination that
// works either.
func (c *Config) validateProviderTuning(add func(format string, args ...any)) {
	dialect, known := c.tuningDialect(add)
	c.validateReasoning(add, dialect, known)
	c.validateSampling(add, dialect, known)
	validateParams(add, c.Provider.Params, c.ownedParamsKeys(dialect, known), reservedWireFields)
}

// tuningDialect resolves the dialect the dialect-DEPENDENT rules are checked
// against, reporting the field's own violations. The bool is whether those
// rules may speak at all: a dialect that does not parse - or one written on a
// wire that has none - makes every one of them unanswerable, so they are
// skipped rather than guessed at. Reporting "kimi fixes sampling" for a config
// that says "kimi-k3" would send the operator to the wrong line. The
// dialect-INDEPENDENT rules still run, so one pass still reports everything
// actionable.
func (c *Config) tuningDialect(add func(format string, args ...any)) (llm.Dialect, bool) {
	if c.Provider.Type == ProviderTypeGemini && c.Provider.Dialect != "" {
		// The gemini wire is a family of its own, not a variation of the OpenAI
		// one, so the VALUE is not parsed here: whatever it spells, the fix is
		// to delete the line, and reporting the dialect vocabulary instead
		// would send the operator to correct a key that has to go. Refused
		// rather than ignored (the anthropic treatment) because the gemini wire
		// is new: nothing was written against it before this rule existed, so
		// strictness costs no working config and buys the operator certainty
		// that no knob is being quietly dropped.
		add("provider.dialect: %q applies to the openai wire; remove it for type gemini", c.Provider.Dialect)
		return llm.DialectOpenAI, false
	}
	dialect, err := llm.ParseDialect(c.Provider.Dialect)
	if err != nil {
		add("provider.dialect: %v", err)
		return dialect, false
	}
	return dialect, true
}

// ownedParamsKeys returns the request-body keys the ACTIVE target writes
// itself - the owned half of what provider.params must not carry. The
// reserved half (reservedWireFields, refused on every target regardless of
// what this returns) is passed separately by the validateParams call site, so
// the two get their own violation wording (issue #16): a key amele writes on
// this target would be silently overwritten or overwrite a contract, while a
// reserved key is refused because amele's own machinery cannot survive it at
// all - neither reason describes the other case.
//
// When the dialect did not parse it returns nil. Which key a dialect writes is
// a dialect question and is unanswerable then, so the same one-error rule the
// other dialect-dependent rules follow applies: the dialect is reported once
// instead of piling on a collision the operator cannot act on yet. The
// reserved keys are not a dialect question - they are refused on every target
// regardless - so the call site passes them unconditionally rather than
// costing the operator a second validate round for a violation that was
// already answerable, against validate's one-pass, every-violation contract.
func (c *Config) ownedParamsKeys(dialect llm.Dialect, known bool) []string {
	switch c.Provider.Type {
	case ProviderTypeAnthropic:
		// The dialect is not consulted on this wire, so a leftover (even an
		// unparseable) value cannot decide the answer.
		return llm.AnthropicOwnedWireFields()
	case ProviderTypeGemini:
		// Same reasoning, and here the dialect is not even legal: an illegal
		// one is reported on its own line and must not also cost the operator
		// the params answer, which this wire can give without it.
		return llm.GeminiOwnedWireFields()
	}
	if !known {
		return nil
	}
	return llm.OwnedWireFields(dialect)
}

// validateReasoning checks the thinking knob against the effort vocabulary and
// against the dialects that can actually carry a token budget.
func (c *Config) validateReasoning(add func(format string, args ...any), dialect llm.Dialect, known bool) {
	r := c.Provider.Reasoning
	if r == nil {
		return
	}
	if r.Effort != "" && !slices.Contains(effortValues, r.Effort) {
		add("provider.reasoning.effort %q is not a valid effort (%s, or omit for the provider default)",
			r.Effort, strings.Join(effortValues, ", "))
	}
	if r.BudgetTokens < 0 {
		add("provider.reasoning.budget_tokens must not be negative")
	}
	// Checked BEFORE the dialect early return: it is a relation between two
	// anthropic-wire fields, and that wire does not consult the dialect at all,
	// so an unparseable dialect cannot make it unanswerable. Leaving it below
	// the return cost the operator a second validate round for a violation the
	// first pass already knew - against validate's one-pass contract.
	c.validateThinkingBudgetFitsCap(add, r)
	// Checked before the dialect early return for the same reason: it is a
	// relation between two gemini-wire fields, and that wire refuses the
	// dialect outright, so an illegal one cannot make the pair unanswerable.
	c.validateGeminiThinkingChoice(add, r)
	if !known {
		return
	}
	// Everything else on the openai wire takes a LEVEL, not a count. Sending
	// the budget anyway would be dropped by the endpoint (or 400 on the strict
	// ones) while the config claims a bounded thinking cost.
	if r.BudgetTokens > 0 && c.Provider.Type != ProviderTypeAnthropic &&
		c.Provider.Type != ProviderTypeGemini && dialect != llm.DialectOpenRouter {
		add("provider.reasoning.budget_tokens is only mapped for the anthropic or gemini wire, or the openrouter dialect (use provider.reasoning.effort instead)")
	}
	// Kimi's K-series thinks unconditionally; "none" has nothing to map to.
	// Gated on the openai wire: the dialect describes an openai-wire variation
	// and is ignored when type is anthropic (documented in the published
	// schema), where "none" is a legal thinking setting.
	if c.dialectApplies(known) && dialect == llm.DialectKimi && r.Effort == "none" {
		add("provider.reasoning.effort %q: kimi models cannot disable thinking", r.Effort)
	}
}

// validateThinkingBudgetFitsCap checks the one relation BETWEEN two tuning
// fields: on the anthropic wire the thinking budget is drawn from max_tokens,
// so a budget that meets or exceeds provider.max_output_tokens leaves no room
// for the answer and the Messages API refuses the request.
//
// CONTRACT: this is a total rule of that wire, and both numbers are knowable
// before the run - exactly the mistake validate exists to catch (exit 2)
// instead of letting it surface as a provider error (exit 5) on the first
// unattended run. It is scoped to the anthropic wire deliberately: the
// openrouter gateway, the only other target that maps a budget, carves it out
// of max_tokens itself.
//
// CONTRACT: an unset provider.max_output_tokens is NOT an absent ceiling here.
// max_tokens is required on every Messages API request, so the client sends
// llm.DefaultAnthropicMaxOutput and the API measures the budget against that
// number. Checking only the explicit case let `budget_tokens: 8192` with no cap
// pass validate and 400 at the API - the exact split this function exists to
// close.
func (c *Config) validateThinkingBudgetFitsCap(add func(format string, args ...any), r *ReasoningConfig) {
	// Zero budget means "unset": there is no relation to check.
	if c.Provider.Type != ProviderTypeAnthropic || r.BudgetTokens <= 0 {
		return
	}
	capTokens, capNote := c.Provider.MaxOutputTokens, ""
	if capTokens <= 0 {
		capTokens = llm.DefaultAnthropicMaxOutput
		capNote = ", the default amele sends when provider.max_output_tokens is unset"
	}
	if r.BudgetTokens >= capTokens {
		add("provider.reasoning.budget_tokens must be below provider.max_output_tokens (%d >= %d%s): on the anthropic wire the thinking budget is drawn from the same output ceiling, so nothing is left for the answer",
			r.BudgetTokens, capTokens, capNote)
	}
}

// validateGeminiThinkingChoice checks the one relation between the two thinking
// fields on the gemini wire: they are ALTERNATIVES there. Effort maps to
// generationConfig.thinkingConfig.thinkingLevel and BudgetTokens to
// thinkingBudget, and the API answers a request carrying both with a 400.
//
// CONTRACT: a total rule of that wire, knowable from the file alone - so it is
// an exit-2 config error rather than an exit-5 provider error on the first
// unattended run.
func (c *Config) validateGeminiThinkingChoice(add func(format string, args ...any), r *ReasoningConfig) {
	if c.Provider.Type != ProviderTypeGemini {
		return
	}
	// Zero budget means "unset", and an empty effort means "send no knob":
	// only two values the operator actually chose are in conflict.
	if r.Effort != "" && r.BudgetTokens > 0 {
		add("provider.reasoning: gemini accepts thinkingLevel or thinkingBudget, not both (drop provider.reasoning.effort or provider.reasoning.budget_tokens)")
	}
}

// dialectApplies reports whether the dialect-DEPENDENT rules may speak. They
// may not on the anthropic wire: the dialect names a variation of the
// OpenAI-compatible format, and the published schema documents it as ignored
// when type is anthropic. Refusing `type: anthropic` + a leftover
// `dialect: kimi` would blame a provider the config is not talking to. They may
// not on the gemini wire either, for the stronger reason that a dialect is
// refused there outright - so no dialect-shaped rule can be about a request
// that wire would send. known carries whether the dialect parsed at all.
func (c *Config) dialectApplies(known bool) bool {
	return known && c.Provider.Type != ProviderTypeAnthropic && c.Provider.Type != ProviderTypeGemini
}

// validateSampling checks temperature/top_p against the range the target
// accepts for every model it serves.
func (c *Config) validateSampling(add func(format string, args ...any), dialect llm.Dialect, known bool) {
	temperature, topP := c.Provider.Temperature, c.Provider.TopP
	// The K-series pins temperature and top_p to fixed values and answers any
	// other value with a 400, so this is a config error, not a preference.
	if c.dialectApplies(known) && dialect == llm.DialectKimi && (temperature != nil || topP != nil) {
		add("provider: kimi K-series models fix sampling; remove temperature/top_p")
	}

	// Both ceilings below are TOTAL for their target, so they belong here
	// rather than in a runtime error: the Anthropic Messages API documents
	// temperature as 0..1 for every model, and so does GLM. The wire is
	// checked first because it wins - on the anthropic path the dialect is not
	// consulted at all, so naming it in the message would misdirect the fix.
	maxTemperature, ceilingNote := 2.0, ""
	switch {
	case c.Provider.Type == ProviderTypeAnthropic:
		maxTemperature, ceilingNote = 1.0, " on the anthropic wire"
	case c.dialectApplies(known) && dialect == llm.DialectGLM:
		maxTemperature, ceilingNote = 1.0, " for the glm dialect"
	}
	// Both checks are written as a NEGATED positive test, deliberately: every
	// comparison against NaN is false, so the natural "v < 0 || v > max" form
	// would wave a NaN through validation (strconv.ParseFloat accepts "NaN"
	// from --set, and YAML spells it ".nan") only for it to die as an
	// unserializable request body at run time.
	if temperature != nil && !(*temperature >= 0 && *temperature <= maxTemperature) {
		add("provider.temperature %g is out of range: must be between 0 and %g%s", *temperature, maxTemperature, ceilingNote)
	}
	// Zero is excluded rather than clamped: top_p 0 asks for an empty nucleus,
	// which providers answer with a 400 rather than with greedy decoding.
	if topP != nil && !(*topP > 0 && *topP <= 1) {
		add("provider.top_p %g is out of range: must be greater than 0 and at most 1", *topP)
	}
}

// validateParams checks the raw escape hatch: no collision with a key the
// active target writes itself (owned, empty when that is unanswerable) or with
// a key amele reserves on every target (reserved), and nothing JSON cannot
// express.
//
// reserved is read-only: the call site hands it the package-level
// reservedWireFields slice directly (no defensive copy), which is safe only
// because this function does no more than slices.Contains on it. Do not
// mutate, sort, or otherwise write through reserved - doing so would corrupt
// the shared package var for every future caller.
//
// CONTRACT: params is merged verbatim into the request body root, so a
// collision would either clobber an amele contract (tools, response_format) or
// be clobbered by it - both silently. The two lists get their own violation
// wording (issue #16): an owned-key collision would be silently overwritten
// or overwrite a contract on THIS target, while a reserved key is refused on
// EVERY target because amele's own machinery cannot survive it - neither
// reason describes the other case, so one message for both would misdirect
// the fix for whichever half it does not describe.
func validateParams(add func(format string, args ...any), params map[string]any, owned, reserved []string) {
	if len(params) == 0 {
		return
	}
	// Sorted so the joined message is deterministic; Go map iteration order
	// would otherwise shuffle violations between runs.
	for _, key := range slices.Sorted(maps.Keys(params)) {
		switch {
		case slices.Contains(reserved, key):
			// Checked first: a key could in principle appear in both lists, and
			// "refused on every target" is the stronger, always-true claim.
			add("provider.params key %q is reserved on every target (amele's own request machinery cannot run with it set); remove it", key)
		case slices.Contains(owned, key):
			add("provider.params key %q is a request field amele sets itself on this target; remove it (params carries provider-specific extras only)", key)
		}
	}
	// The values are serialized into the request body verbatim, so a value
	// JSON cannot express (a non-string mapping key, a NaN) must fail here
	// rather than on the first request of an unattended run.
	if _, err := json.Marshal(params); err != nil {
		add("provider.params is not JSON-serializable: %v", err)
	}
}

// baseURLProblem reports why raw cannot serve as a provider base URL, phrased
// to follow the quoted value in an error message, or "" when it is usable.
//
// CONTRACT: a config that passes `amele validate` must not fail its
// CONFIGURATION at run time. Accepting anything with a scheme and a host let
// "ftp://host" and the typo "htps://host" through validation only to die as a
// provider/network error (exit 5) on the first real run, which is the wrong
// exit code and the wrong moment. Query strings and fragments are refused for
// the same reason: the clients append request paths to this value, so
// "http://h/?q=1" becomes "http://h/?q=1/chat/completions".
func baseURLProblem(raw string) string {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme == "" || u.Host == "" {
		// Wording frozen: `amele validate`'s golden output pins this line.
		return "is not a valid absolute URL"
	}
	// url.Parse lowercases the scheme, so "HTTPS://..." is accepted here.
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Sprintf("must use the http or https scheme (got %q)", u.Scheme)
	}
	if u.RawQuery != "" || u.ForceQuery || u.Fragment != "" {
		return "must not contain a query string or fragment: the client appends request paths to it"
	}
	return ""
}

// hostOrVersionProblem reports what is wrong with a non-empty base_url beyond
// its URL syntax, phrased to follow the quoted value in an error message.
//
// The vertex endpoint reads base_url as a HOST override and nothing more (see
// internal/llm's vertexEndpoint): the request path is built from the project
// and location, so any path written here would be dropped. It is refused
// instead of dropped - a proxy prefix that silently disappears is a request
// sent somewhere other than where the config says.
func (c *Config) hostOrVersionProblem() string {
	// The type is consulted too: a vertex block outside the gemini wire is
	// already an error of its own, and the endpoint it describes is not the one
	// this base_url would serve, so the version rule stays the right complaint.
	if c.Provider.Vertex == nil || c.Provider.Type != ProviderTypeGemini {
		return versionedBaseURLProblem(c.Provider.Type, c.Provider.BaseURL)
	}
	u, err := url.Parse(c.Provider.BaseURL)
	if err != nil || strings.Trim(u.Path, "/") == "" {
		// A parse failure is already reported by baseURLProblem, which runs
		// first; there is nothing to add here.
		return ""
	}
	return "must not include a path when provider.vertex is set: only the scheme and host are taken from base_url, and the client appends /v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent itself"
}

// versionedBaseURLProblem reports the "the API version is still in base_url"
// mistake, phrased to follow the quoted value in an error message, or "" when
// there is none.
//
// It applies to the two NATIVE wires only. Both append their own versioned
// request path, while the OpenAI-compatible convention puts the version IN
// base_url - so an operator moving a config from a gateway reflexively keeps a
// suffix that then travels twice. The result is a 404 on the first real request
// of an unattended run, which reads as a broken endpoint rather than as the
// config error it is, so it is caught here instead.
func versionedBaseURLProblem(providerType, baseURL string) string {
	trimmed := strings.TrimRight(baseURL, "/")
	switch providerType {
	case ProviderTypeAnthropic:
		if strings.HasSuffix(trimmed, "/v1") {
			// Wording frozen: this line is pinned by tests.
			return "must not end with /v1 when provider.type is anthropic: the client appends /v1/messages itself"
		}
	case ProviderTypeGemini:
		// /v1 is refused beside /v1beta because it is the habit the operator
		// brings, not because this client would ever append it.
		if strings.HasSuffix(trimmed, "/v1beta") || strings.HasSuffix(trimmed, "/v1") {
			return "must not end with /v1beta or /v1 when provider.type is gemini: the client appends /v1beta/models/{model}:generateContent itself"
		}
	}
	return ""
}

// validatePermissions checks the permission profile's policy values.
//
// Keys are patterns, not just names: a key containing '*' matches by glob
// (internal/perm applies the precedence), which is how one line governs every
// tool of an MCP server. Nothing here rejects or interprets the pattern - a
// key is only ever read as text at this layer.
//
// It deliberately does NOT check that a named tool exists: a profile may be
// written before the tool it governs (or shared across configs that declare
// different subprocess tools), and an entry for an absent tool is inert
// rather than dangerous - it can only ever deny something that cannot be
// called. Rejecting it would turn a harmless leftover into a cron failure.
func (c *Config) validatePermissions(add func(format string, args ...any)) {
	// An empty default means "unset" and defaults to allow (Phase 1 parity);
	// an empty per-tool value is a typo ("fs_write:" with nothing after it),
	// because the only reason to write an entry is to override the default.
	if c.Permissions.Default != "" && !c.Permissions.Default.Valid() {
		add("permissions.default %q is not a valid policy (%s)", c.Permissions.Default, policyValues)
	}

	// Sorted so the joined error message is deterministic; Go map iteration
	// order would otherwise shuffle violations between runs and defeat
	// golden-file testing.
	names := make([]string, 0, len(c.Permissions.Tools))
	for name := range c.Permissions.Tools {
		names = append(names, name)
	}
	slices.Sort(names)
	for _, name := range names {
		if policy := c.Permissions.Tools[name]; !policy.Valid() {
			add("permissions.tools.%s: %q is not a valid policy (%s)", name, policy, policyValues)
		}
	}
}

// validateShell checks the builtin shell block.
//
// It deliberately does NOT reject allow/deny patterns written while the tool is
// disabled: a profile is often prepared before it is switched on, and inert
// patterns cannot hurt anyone. Nor does it judge the patterns themselves -
// there is no such thing as a "safe" pattern to check for (see ShellConfig's
// SECURITY note), so pretending to validate one would only sell false comfort.
func (c *Config) validateShell(add func(format string, args ...any)) {
	if c.Tools.Shell.Timeout < 0 {
		add("tools.shell.timeout must not be negative")
	}
	validateEnvAllowlist(add, "tools.shell.env", c.Tools.Shell.Env)
}

// validateEnvAllowlist checks one env allowlist (tools.shell.env or a
// subprocess tool's env). Empty entries are rejected - a "- " with nothing
// after it is a YAML slip, and silently allowing an empty name would make the
// list look longer than it is. Duplicates are rejected too: a repeated name
// is harmless to the child but almost always a copy-paste error worth
// surfacing at validation time rather than never.
func validateEnvAllowlist(add func(format string, args ...any), where string, env []string) {
	seen := map[string]bool{}
	for i, name := range env {
		if name == "" {
			add("%s[%d]: entry must be a non-empty environment variable name", where, i)
			continue
		}
		// SECURITY: "NAME=value" here means the author believed they were
		// setting a value for the child. The allowlist only passes names
		// through from amele's own environment; silently accepting the
		// assignment form would allowlist a variable that never matches
		// (the literal name "NAME=value") while the author assumes the
		// child got their value. Reject instead of guessing.
		if strings.Contains(name, "=") {
			add("%s[%d]: %q must be a variable name, not an assignment: the list passes variables through from amele's environment by name", where, i, name)
			continue
		}
		if seen[name] {
			add("%s[%d]: duplicate env name %q", where, i, name)
		}
		seen[name] = true
	}
}

// validateSubprocessTools checks the custom tool declarations. Split from
// Validate to keep each function under the complexity budget.
func (c *Config) validateSubprocessTools(add func(format string, args ...any)) {
	seen := map[string]bool{}
	for i, tool := range c.Tools.Subprocess {
		where := fmt.Sprintf("tools.subprocess[%d]", i)
		if !toolNameRe.MatchString(tool.Name) {
			add("%s: name %q must match %s", where, tool.Name, toolNameRe.String())
		}
		if seen[tool.Name] {
			add("%s: duplicate tool name %q", where, tool.Name)
		}
		seen[tool.Name] = true
		if tool.Description == "" {
			add("%s: description is required", where)
		}
		if len(tool.Command) == 0 {
			add("%s: command must have at least the executable", where)
		} else if tool.Command[0] == "" {
			// An empty executable would slip through path resolution and reach
			// exec as "", producing a confusing spawn error at run time; reject
			// it where every other shape error is reported.
			add("%s: command[0] must not be empty", where)
		}
		if tool.Timeout < 0 {
			add("%s: timeout must not be negative", where)
		}
		validateEnvAllowlist(add, where+".env", tool.Env)
	}
	// Builtin tool names are reserved so a subprocess tool can never shadow
	// (and silently replace) a builtin. "shell" is included even when the
	// builtin is disabled: the name means one thing across every config, so a
	// permission profile entry (permissions.tools.shell) always governs what
	// the operator thinks it governs.
	for _, reserved := range builtinToolNames {
		if seen[reserved] {
			add("tools.subprocess: name %q is reserved for a builtin tool", reserved)
		}
	}
}
