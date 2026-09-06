package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"slices"
	"strings"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/llm"
)

// shellToolName is the name the builtin shell tool is registered under. It is
// reserved in the config schema so no subprocess tool can shadow it.
const shellToolName = "shell"

// rejectionPrefix starts every policy rejection result.
//
// CONTRACT: rejections are TOOL RESULTS, not Go errors - the model reads the
// text and adapts (picks an allowed command, or explains it cannot proceed)
// instead of the run aborting.
const rejectionPrefix = "command rejected by shell policy"

// Shell is the builtin `shell` tool: it runs one model-written command line
// through `sh -c` with the workspace as the working directory.
//
// SECURITY: this tool is the agent's widest capability and is therefore
// disabled by default (config.ShellConfig.Enabled). Two limits of it must be
// understood before enabling it.
//
// 1. The allow/deny pattern lists are an ACCIDENT-PREVENTION layer - they stop
// a model from casually running `rm -rf`, they do NOT contain a determined or
// prompt-injected one. `*` patterns match text, and text has escapes: "git *"
// permits `git -c core.fsmonitor=cmd status`, which executes arbitrary code;
// aliases, hooks, `--exec`, `-o ProxyCommand` and friends do the same for other
// binaries.
//
// 2. By default the child inherits amele's ENTIRE environment (as subprocess
// tools do), so `printenv OPENROUTER_API_KEY` hands the provider credential to
// the model as an ordinary tool result. No pattern list stops that: reading an
// env var takes no suspicious-looking command. It is worse for forensics than
// it looks - session-log redaction rewrites the leaked value to [REDACTED], so
// the audit trail HIDES the leak instead of recording it. The first-line
// mitigation is the config's env allowlist (config.ShellConfig.Env): when set,
// the child is started with only the listed variables plus PATH/HOME/LANG (see
// buildChildEnv). But the allowlist only scopes inheritance - a same-user
// child can still read amele's environment via /proc - so anything the agent
// must never read still must not be in amele's environment at all.
//
// Both point the same way: the real boundary is the OS. Run amele in a
// container/VM with only the filesystem, network, privileges and environment it
// should have (docs/threat-model.md). Anything stronger belongs in that layer, never
// in this pattern list.
type Shell struct {
	// shPath is the resolved `sh` executable, looked up once at construction
	// so a missing shell is a startup error rather than a mid-run surprise.
	shPath    string
	workspace string
	allow     []string
	deny      []string
	// env is the environment allowlist handed to runCommand; nil means the
	// child inherits amele's entire environment (config.ShellConfig.Env).
	env     []string
	timeout config.Duration
	// maxOutput caps each captured stream, resolved once at construction.
	maxOutput int
}

// ShellOptions tunes the builtin shell tool beyond its config entry.
type ShellOptions struct {
	// MaxOutputBytes caps each captured stream; <= 0 means
	// DefaultMaxOutputBytes.
	MaxOutputBytes int
}

// NewShell builds the builtin shell tool bound to workspace. It fails when no
// `sh` is on PATH: the tool cannot work without one, and reporting it at
// startup (exit 2) beats failing on the first model call halfway into a run.
// A zero opts.MaxOutputBytes selects DefaultMaxOutputBytes.
//
// The caller decides whether to build it at all; NewShell does not consult
// cfg.Enabled, because "should this tool exist" is a registry question
// (cmd/amele buildRegistry) and keeping the check there means there is exactly
// one place where the default-off contract lives.
func NewShell(cfg config.ShellConfig, workspace string, opts ShellOptions) (*Shell, error) {
	shPath, err := exec.LookPath("sh")
	if err != nil {
		return nil, fmt.Errorf("shell tool needs a POSIX shell on PATH: %w", err)
	}
	return &Shell{
		shPath:    shPath,
		workspace: workspace,
		allow:     slices.Clone(cfg.Allow),
		deny:      slices.Clone(cfg.Deny),
		env:       slices.Clone(cfg.Env),
		timeout:   cfg.Timeout,
		maxOutput: resolveOutputCap(opts.MaxOutputBytes),
	}, nil
}

// shellArgs is the single argument the model supplies: a command line.
type shellArgs struct {
	Command string `json:"command"`
}

// Def implements Tool. The description states the execution model plainly
// (one `sh -c` line, workspace cwd, bounded output) so the model does not have
// to guess whether pipes, redirection or `cd` persist between calls.
func (s *Shell) Def() llm.ToolDef {
	return llm.ToolDef{
		Name: shellToolName,
		// SECURITY: the description advertises the capability honestly, but it
		// is NOT a control - a model can ask for anything and the operator's
		// allow/deny patterns (accident prevention) plus the OS sandbox (the
		// actual boundary) decide what runs.
		Description: "Run a command line with `sh -c` in the workspace directory. Each call is a fresh shell: shell state (cd, variables) does not persist between calls. Stdout is returned; a non-zero exit is reported with its stderr. Some commands may be rejected by the operator's policy.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {"command": {"type": "string", "description": "Shell command line, e.g. \"git status --short\""}},
			"required": ["command"],
			"additionalProperties": false
		}`),
	}
}

// Invoke implements Tool. A policy rejection, a non-zero exit and a timeout are
// all returned as result text; only bad arguments and spawn failures are Go
// errors.
func (s *Shell) Invoke(ctx context.Context, rawArgs string) (string, error) {
	out, _, err := s.InvokeOutcome(ctx, rawArgs)
	return out, err
}

// InvokeOutcome is Invoke plus the out-of-band classification of how the call
// ended (see Outcome). Invoke is defined in terms of it so the two can never
// disagree about what happened.
//
// The rejection outcome is decided HERE rather than in runCommand: a rejected
// command never reaches a process, and this is the only code that knows the
// operator's patterns turned it away.
func (s *Shell) InvokeOutcome(ctx context.Context, rawArgs string) (string, Outcome, error) {
	var args shellArgs
	if err := decodeArgs(rawArgs, &args); err != nil {
		return "", Outcome{}, err
	}
	if strings.TrimSpace(args.Command) == "" {
		return "", Outcome{}, errors.New("shell: command is required")
	}

	if reason, ok := s.reject(args.Command); ok {
		return reason, Outcome{Kind: OutcomeRejected}, nil
	}

	out, outcome, err := runCommand(ctx, []string{s.shPath, "-c", args.Command}, s.workspace, "", s.env, s.timeout.Std(), s.maxOutput)
	if err != nil {
		return "", Outcome{}, fmt.Errorf("running shell command: %w", err)
	}
	return out, outcome, nil
}

// reject applies the operator's pattern lists to a command and returns the
// model-facing rejection text when it must not run.
//
// Matching is per LINE, not per command, and each line is whitespace-trimmed
// first (empty lines are ignored). Exact semantics:
//
//   - deny: ANY line matching ANY deny pattern rejects the whole command;
//   - allow (when non-empty): EVERY non-empty line must match at least one
//     allow pattern, otherwise the whole command is rejected;
//   - an empty allow list permits everything deny did not catch.
//
// Normalization is the point, not a nicety. Matching the raw command string
// meant `deny: ["rm *"]` was bypassed by a leading space, a tab, or - the
// common case, because well-behaved models write multi-line commands routinely
// - by putting the denied command on the second line of "cd build\nrm -rf .".
// A guard that a normal model defeats by accident does not do its one job.
//
// SECURITY: this is still ACCIDENT PREVENTION, not a security boundary, and
// per-line normalization does not change that. Patterns match text, and text
// has escapes: `git -c core.fsmonitor=cmd`, `find -exec`, `ssh host cmd`,
// aliases, hooks, `env`, `xargs` all walk straight through an allow list that
// names them, and a single line can still chain commands with `;`, `&&` or a
// pipe. Do not read a passing pattern check as "this command is safe"; the
// boundary that actually holds is the OS/container the agent runs in
// (docs/threat-model.md).
func (s *Shell) reject(command string) (string, bool) {
	lines := policyLines(command)

	for _, line := range lines {
		for _, pattern := range s.deny {
			if GlobMatch(pattern, line) {
				return fmt.Sprintf("%s: the line %q matches the deny pattern %q. Try a different command.",
					rejectionPrefix, line, pattern), true
			}
		}
	}
	if len(s.allow) == 0 {
		return "", false
	}
	for _, line := range lines {
		if !matchesAny(s.allow, line) {
			return fmt.Sprintf("%s: the line %q matches none of the allowed patterns (%s). Try a different command.",
				rejectionPrefix, line, strings.Join(quoteAll(s.allow), ", ")), true
		}
	}
	return "", false
}

// policyLines splits a command into the units the patterns are matched
// against: one per line, whitespace-trimmed, empties dropped. Trimming and
// splitting happen for MATCHING only - the command is executed exactly as the
// model wrote it, so normalization can never change what runs, only what is
// allowed to run.
func policyLines(command string) []string {
	raw := strings.Split(command, "\n")
	lines := make([]string, 0, len(raw))
	for _, line := range raw {
		// TrimSpace also drops the \r of CRLF input and the trailing tabs a
		// model uses to indent a continuation.
		if trimmed := strings.TrimSpace(line); trimmed != "" {
			lines = append(lines, trimmed)
		}
	}
	return lines
}

// matchesAny reports whether s matches at least one of the patterns.
func matchesAny(patterns []string, s string) bool {
	for _, pattern := range patterns {
		if GlobMatch(pattern, s) {
			return true
		}
	}
	return false
}

// quoteAll quotes each pattern for the rejection message, so a pattern made of
// spaces and stars is still readable to the model.
func quoteAll(patterns []string) []string {
	out := make([]string, 0, len(patterns))
	for _, p := range patterns {
		out = append(out, fmt.Sprintf("%q", p))
	}
	return out
}

// GlobMatch reports whether the whole string s matches pattern.
//
// The grammar is deliberately one rule wide: `*` matches any substring
// (including the empty one) and every other byte matches itself. Matching is
// case-sensitive and anchored at both ends. filepath/path.Match is NOT used on
// purpose - its `*` stops at a separator and it gives `?`, `[...]` and `\`
// meanings, which turns "rm *" into a rule an operator cannot predict when the
// subject is a command line rather than a path.
//
// It is exported because the same rule is shared by the shell allow/deny lists
// and the MCP include/exclude filters (and, later, the permission globs), so an
// operator learns one syntax and it behaves identically everywhere.
func GlobMatch(pattern, s string) bool {
	parts := strings.Split(pattern, "*")
	if len(parts) == 1 {
		return s == pattern // no wildcard: exact match
	}
	// The first and last segments are anchored; the middle ones are matched
	// greedily left to right. Consuming the prefix before the loop (and
	// checking the suffix after it) is what keeps the match anchored, and
	// slicing s as segments are consumed is what stops two segments from
	// matching the same characters twice ("a*a" must not match "a").
	if !strings.HasPrefix(s, parts[0]) {
		return false
	}
	s = s[len(parts[0]):]

	last := parts[len(parts)-1]
	for _, middle := range parts[1 : len(parts)-1] {
		idx := strings.Index(s, middle)
		if idx < 0 {
			return false
		}
		s = s[idx+len(middle):]
	}
	return strings.HasSuffix(s, last)
}
