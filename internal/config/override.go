package config

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"
)

// settableKeys is the CLOSED allowlist of config fields the command line may
// override, sorted so error messages and documentation are deterministic.
//
// SECURITY: the fields that grant capability are deliberately absent -
// tools.* (which tools exist at all), mcp.* (which external servers the run
// connects to, and therefore which tools and credentials it carries),
// permissions.* (who may run them) and the provider's IDENTITY - type,
// base_url, api_key, which decide where the run's credentials go.
// docs/threat-model.md §2 puts
// the YAML file on the trusted side of the boundary precisely because it is
// the operator's reviewable, diffable declaration of authority: `amele explain
// agent.yaml` tells you what agent.yaml grants, and that answer must not be
// falsifiable by a flag appended to the cron line that invokes it. What stays
// settable is the parameters of a run whose authority is already fixed: which
// model, where it works, what it is told, what it may spend, where the log
// goes.
//
// SECURITY: `lock` was on this list until 2026-08-12 and was withdrawn for the
// same reason. It was the one settable key that could WEAKEN a run rather than
// retune it: `--set lock=false` disarmed the single-flight guard an audited
// `lock: true` had armed, from the invocation, with nothing in the file or its
// explain report to show for it. Removing it is a breaking change to the
// documented allowlist (docs/contracts/cli.md); `lock: true` in YAML is
// untouched.
//
// SECURITY: the four provider.* entries added on 2026-08-24 are TUNING, not
// identity - how hard the model thinks, how it samples, how long its answer may
// be. They belong with the budgets: the worst an operator can do with them is
// make a run spend differently. provider.dialect stays out because it reshapes
// every request rather than retuning one, and provider.params because it writes
// arbitrary keys into the request body.
//
// SECURITY: limits.max_logged_field (2026-08-31) is a BUDGET, not a
// disclosure switch: it bounds how many bytes of each field the session log
// keeps, so the worst an operator can do with it is make the log bigger or
// smaller. Its two companions in the same feature - log_reasoning and
// print_session_path - stay off the list deliberately: they decide WHAT is
// persisted and what is printed, which is a data-governance statement the
// audited YAML owns.
//
// SECURITY: limits.max_tool_result_bytes (2026-09-06) joins them on the same
// reading. It is a BUDGET, not a capability: it bounds how many bytes of a
// tool result reach the model, so the worst an operator can do with it is make
// the model read more or less of a result it was already allowed to fetch.
// WHICH tools exist and WHO may run them stay in the YAML, untouched by it,
// and the key cannot disarm the guard - Validate refuses anything under 1 KiB
// from a flag exactly as it does from the file.
var settableKeys = []string{
	"limits.max_logged_field",
	"limits.max_tokens",
	"limits.max_tool_result_bytes",
	"limits.max_turns",
	"limits.timeout",
	"model",
	"output.max_schema_retries",
	"prompt",
	"provider.max_output_tokens",
	"provider.reasoning.effort",
	"provider.temperature",
	"provider.top_p",
	"session_dir",
	"system_prompt_file",
	"workspace",
}

// SettableKeys returns the sorted allowlist of keys ApplyOverrides accepts.
// The result is a copy: the allowlist is a security boundary, not a variable.
func SettableKeys() []string {
	return slices.Clone(settableKeys)
}

// SplitOverride splits a "key=value" override pair on the FIRST "=". It
// reports ok=false when there is no "=" at all; an empty value (and even an
// empty key, which ApplyOverrides then rejects by name) is a successful split.
//
// Exported because internal/explain marks the overridden lines of its report
// and must derive keys with exactly the parser that applied them - two
// splitters would eventually disagree and mark the wrong line.
func SplitOverride(pair string) (key, value string, ok bool) {
	key, value, ok = strings.Cut(pair, "=")
	if !ok {
		return "", "", false
	}
	return key, value, true
}

// ApplyOverrides applies CLI `key=value` overrides to cfg in order, so a later
// duplicate wins over an earlier one. baseDir is the directory CLI-given paths
// (workspace, session_dir, system_prompt_file) resolve against - the caller's
// working directory, NOT the config file's directory: a path typed on the
// command line means what it means in the shell that typed it, while the same
// field written in YAML stays relative to the YAML (config.Load's rule).
//
// CONTRACT: call it between Load and Validate. Overrides participate in
// validation - that is what makes `--set model=X` valid for a config with no
// model, and what keeps a nonsense value (a negative budget, an unreachable
// workspace) an exit-2 config error instead of a mid-run surprise.
//
// Every failure wraps ErrInvalid and names the key, so the caller maps it to
// exit 2 like any other config error. cfg may be partially modified when an
// error is returned: the caller must abandon it, which it does - a config
// error aborts before anything is spent.
func ApplyOverrides(cfg *Config, pairs []string, baseDir string) error {
	for _, pair := range pairs {
		key, value, ok := SplitOverride(pair)
		if !ok {
			return fmt.Errorf("%w: --set %q is not in key=value form", ErrInvalid, pair)
		}
		if err := cfg.applyOverride(key, value, baseDir); err != nil {
			return err
		}
	}
	return nil
}

// applyOverride dispatches one already-split pair. The two halves are split by
// value shape (text/path versus parsed scalar) rather than by config section:
// it keeps each switch inside the complexity budget and puts the parsing rules
// of a kind next to each other.
func (c *Config) applyOverride(key, value, baseDir string) error {
	if handled, err := c.applyTextOverride(key, value, baseDir); handled {
		return err
	}
	if handled, err := c.applyScalarOverride(key, value); handled {
		return err
	}
	return fmt.Errorf("%w: cannot override %q from the command line; settable keys: %s",
		ErrInvalid, key, strings.Join(settableKeys, ", "))
}

// applyTextOverride handles the string-valued keys, including the three that
// carry a path. handled=false means the key belongs to another group (or to
// none at all).
func (c *Config) applyTextOverride(key, value, baseDir string) (handled bool, err error) {
	switch key {
	case "model":
		c.Model = value
	case "prompt":
		c.Prompt = value
	case "workspace":
		if value == "" {
			return true, fmt.Errorf("%w: --set workspace: the value must name a directory", ErrInvalid)
		}
		c.Workspace = resolveOverridePath(value, baseDir)
	case "session_dir":
		// The one key where empty is a setting rather than a slip: it turns
		// session logging off, which is how a caller runs one config both with
		// and without an audit trail.
		if value == "" {
			c.SessionDir = ""
			return true, nil
		}
		c.SessionDir = resolveOverridePath(value, baseDir)
	case "system_prompt_file":
		return true, c.overrideSystemPromptFile(value, baseDir)
	case "provider.reasoning.effort":
		c.overrideReasoningEffort(value)
	default:
		return false, nil
	}
	return true, nil
}

// overrideReasoningEffort sets the reasoning depth, allocating the block when
// the config has none. The value is not judged here: Validate owns the effort
// vocabulary, once, for YAML and --set alike.
//
// An empty value means "back to the provider default", which is exactly what an
// absent block means - so with no block to clear, none is created either. That
// keeps `--set provider.reasoning.effort=` from handing the wiring below an
// empty-but-present block to interpret.
func (c *Config) overrideReasoningEffort(value string) {
	if c.Provider.Reasoning == nil {
		if value == "" {
			return
		}
		c.Provider.Reasoning = &ReasoningConfig{}
	}
	c.Provider.Reasoning.Effort = value
}

// overrideSystemPromptFile points the system prompt at a different file and
// RE-READS it. Load already read the config's own prompt file into
// SystemPrompt, so replacing only the path would leave the old text in place -
// the model would be told one thing while the operator read another.
//
// The file's content wins over an inline system_prompt too (the mutual
// exclusion Violations reports between the two YAML fields does not apply
// here):
// the override is a deliberate, explicit replacement of whatever the config
// said.
func (c *Config) overrideSystemPromptFile(value, baseDir string) error {
	if value == "" {
		return fmt.Errorf("%w: --set system_prompt_file: the value must name a file", ErrInvalid)
	}
	path, content, err := readPromptFile(value, baseDir)
	if err != nil {
		return fmt.Errorf("%w: --set system_prompt_file: %v", ErrInvalid, err)
	}
	c.SystemPromptFile = path
	c.SystemPrompt = content
	// The override names the winning prompt outright, so the ambiguity the
	// exclusivity rule guards against is gone; leaving the violation set would
	// make a pack that ships both fields unfixable from the command line.
	c.promptConflict = false
	return nil
}

// readPromptFile resolves path against baseDir and reads it, returning the
// resolved path and the content. Shared by applyDefaults (baseDir = the config
// file's directory) and the system_prompt_file override (baseDir = the
// caller's working directory), so both read a prompt file the same way.
func readPromptFile(path, baseDir string) (resolved, content string, err error) {
	resolved = resolveOverridePath(path, baseDir)
	raw, err := os.ReadFile(resolved) //nolint:gosec // G304: the prompt file is named by the operator, in the config or on the command line - reading it is the point.
	if err != nil {
		return "", "", err
	}
	return resolved, string(raw), nil
}

// applyScalarOverride handles the keys whose value is parsed into a number or
// a duration. Range checks are deliberately left to Validate: one
// rule per field, stated once, whether the value came from YAML or from --set.
func (c *Config) applyScalarOverride(key, value string) (handled bool, err error) {
	switch key {
	case "limits.max_turns":
		return true, overrideInt(key, value, &c.Limits.MaxTurns)
	case "limits.max_tokens":
		return true, overrideInt(key, value, &c.Limits.MaxTokens)
	case "limits.max_logged_field":
		return true, overrideIntPtr(key, value, &c.Limits.MaxLoggedField)
	case "limits.max_tool_result_bytes":
		return true, overrideIntPtr(key, value, &c.Limits.MaxToolResultBytes)
	case "output.max_schema_retries":
		return true, overrideInt(key, value, &c.Output.MaxSchemaRetries)
	case "provider.max_output_tokens":
		return true, overrideInt(key, value, &c.Provider.MaxOutputTokens)
	case "provider.temperature":
		return true, overrideFloat(key, value, &c.Provider.Temperature)
	case "provider.top_p":
		return true, overrideFloat(key, value, &c.Provider.TopP)
	case "limits.timeout":
		parsed, err := time.ParseDuration(value)
		if err != nil {
			return true, fmt.Errorf("%w: --set limits.timeout: %q is not a duration like \"30s\" or \"5m\"", ErrInvalid, value)
		}
		c.Limits.Timeout = Duration(parsed)
	default:
		return false, nil
	}
	return true, nil
}

// overrideInt parses one integer override into dst, naming the key in the
// error so an operator editing a cron line knows which --set to fix.
func overrideInt(key, value string, dst *int) error {
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fmt.Errorf("%w: --set %s: %q is not an integer", ErrInvalid, key, value)
	}
	*dst = parsed
	return nil
}

// overrideIntPtr parses one integer override into dst, which is a POINTER
// field: "--set limits.max_logged_field=0" therefore means the explicit
// "unbounded" setting rather than "unset", the same distinction the YAML key
// carries.
//
// An empty value is a setting here, not a slip (as with session_dir): it
// clears the pointer, which is how a caller drops a bound the file set and
// falls back to the built-in default for this run. Range checks stay in
// Validate, like every other field's.
func overrideIntPtr(key, value string, dst **int) error {
	if value == "" {
		*dst = nil
		return nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return fmt.Errorf("%w: --set %s: %q is not an integer", ErrInvalid, key, value)
	}
	*dst = &parsed
	return nil
}

// overrideFloat parses one sampling override into dst, which is a POINTER
// field: the override sets it, so "--set provider.temperature=0" means the
// deterministic setting rather than "unset". Range checks stay in Validate,
// like every other field's.
func overrideFloat(key, value string, dst **float64) error {
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return fmt.Errorf("%w: --set %s: %q is not a number", ErrInvalid, key, value)
	}
	*dst = &parsed
	return nil
}

// resolveOverridePath makes a CLI-given path absolute against baseDir (the
// caller's working directory). Absolute rather than merely joined so the value
// is stable afterwards: nothing downstream re-resolves it, and a run that
// changes directory (or a lock/session path compared across invocations) must
// still mean the same place.
func resolveOverridePath(value, baseDir string) string {
	if filepath.IsAbs(value) {
		return filepath.Clean(value)
	}
	return filepath.Join(baseDir, value)
}
