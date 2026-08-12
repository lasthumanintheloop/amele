package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"slices"
	"strings"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/llm"
)

// DefaultSubprocessTimeout bounds a tool invocation when the config does not
// set one; an unbounded child process would defeat the run's kill switches.
const DefaultSubprocessTimeout = 60 * time.Second

// maxOutputBytes bounds captured stdout/stderr per invocation, for the same
// context-budget reason fs_read is bounded.
const maxOutputBytes = 64 * 1024

// Subprocess adapts one config.SubprocessTool into a Tool. The command is a
// fixed argv vector executed directly (no shell), with the workspace as the
// working directory.
type Subprocess struct {
	def       config.SubprocessTool
	workspace string
}

// NewSubprocess builds a subprocess tool from its validated config entry.
func NewSubprocess(def config.SubprocessTool, workspace string) *Subprocess {
	return &Subprocess{def: def, workspace: workspace}
}

// subprocessArgs is what the model may supply per invocation.
type subprocessArgs struct {
	// Stdin is piped to the process.
	Stdin string `json:"stdin,omitempty"`
	// Args are extra argv elements appended after the configured command.
	// Only honored when the config sets allow_args: true.
	Args []string `json:"args,omitempty"`
}

// Def implements Tool. The advertised schema depends on allow_args so the
// model is never invited to pass arguments it is not allowed to pass.
func (s *Subprocess) Def() llm.ToolDef {
	properties := `"stdin": {"type": "string", "description": "Text piped to the process on stdin"}`
	if s.def.AllowArgs {
		properties += `,
			"args": {"type": "array", "items": {"type": "string"}, "description": "Extra command-line arguments"}`
	}
	return llm.ToolDef{
		Name:        s.def.Name,
		Description: s.def.Description,
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {` + properties + `},
			"additionalProperties": false
		}`),
	}
}

// Invoke implements Tool. A non-zero exit is reported as result text (the
// model should see and react to failures); only harness-level problems
// (bad arguments, spawn failure) return a Go error.
func (s *Subprocess) Invoke(ctx context.Context, rawArgs string) (string, error) {
	out, _, err := s.InvokeOutcome(ctx, rawArgs)
	return out, err
}

// InvokeOutcome is Invoke plus the out-of-band classification of how the call
// ended (see Outcome). Invoke is defined in terms of it so the two can never
// disagree about what happened.
func (s *Subprocess) InvokeOutcome(ctx context.Context, rawArgs string) (string, Outcome, error) {
	var args subprocessArgs
	if rawArgs != "" {
		if err := decodeArgs(rawArgs, &args); err != nil {
			return "", Outcome{}, err
		}
	}
	// SECURITY: extra argv is opt-in per tool. Without allow_args the model
	// controls stdin only; the executable and its arguments are fixed by
	// the reviewed YAML.
	if len(args.Args) > 0 && !s.def.AllowArgs {
		return "", Outcome{}, fmt.Errorf("tool %q does not accept extra args (set allow_args: true in the config to permit them)", s.def.Name)
	}

	// Cloned before appending so the config's argv is never extended in place.
	argv := append(slices.Clone(s.def.Command), args.Args...)
	out, outcome, err := runCommand(ctx, argv, s.workspace, args.Stdin, s.def.Env, s.def.Timeout.Std())
	if err != nil {
		return "", Outcome{}, fmt.Errorf("running %q: %w", s.def.Name, err)
	}
	return out, outcome, nil
}

// runCommand executes one child process and renders the outcome the way the
// model should see it. It is the single implementation of amele's process
// discipline - process group, WaitDelay, output caps, timeout attribution -
// shared by every tool that spawns something (the subprocess tools and the
// builtin shell), so those properties cannot drift apart between them.
//
// timeout <= 0 means DefaultSubprocessTimeout. envAllow is the environment
// allowlist for the child: nil or empty means the child inherits amele's
// entire environment (the pre-allowlist behavior every existing config relies
// on); non-empty means the environment is BUILT from the list - see
// buildChildEnv. The returned error is reserved for harness-level failures
// (the executable does not exist, fork failed); a non-zero exit, a timeout
// and a run-level cancellation are all returned as result TEXT with a nil
// error, because they are task information the model must be able to react to.
//
// The returned Outcome says which of those three it was without parsing the
// text back apart. This function is the only place that distinguishes them -
// the tool timeout, the run-level cancellation and the exit status are all
// decided here - so classifying anywhere else would be a second, drifting
// answer to the same question.
func runCommand(ctx context.Context, argv []string, dir, stdin string, envAllow []string, timeout time.Duration) (string, Outcome, error) {
	if timeout <= 0 {
		timeout = DefaultSubprocessTimeout
	}
	// Kept so a kill can be attributed correctly: the run's own deadline or
	// a Ctrl-C also cancels the derived context, and blaming the tool
	// timeout for those would send the operator debugging the wrong knob.
	parent := ctx
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...) //nolint:gosec // G204: callers build argv from the user's reviewed YAML (subprocess) or from a policy-checked shell line the operator opted into (shell tool).
	cmd.Dir = dir
	cmd.Stdin = strings.NewReader(stdin)
	// SECURITY: a non-empty allowlist replaces inheritance - the child's
	// environment is built from the listed names, so a model-driven command
	// can no longer `printenv` every secret amele was started with. A nil/empty
	// list keeps full inheritance (cmd.Env == nil), the behavior every config
	// written before the allowlist existed depends on.
	if len(envAllow) > 0 {
		cmd.Env = buildChildEnv(envAllow)
	}
	// On timeout the whole process group must die, not just the direct
	// child - `sh -c "a | b"` style tools would otherwise leak grandchildren
	// past the run's kill switches. Platform-specific, no-op on Windows.
	setupProcessGroup(cmd)
	// SECURITY / CONTRACT: WaitDelay bounds how long Run may block AFTER
	// cancellation. Without it, a grandchild that survives the group-kill
	// (e.g. a nested amele tool spawning its own process group) can hold the
	// captured stdout pipe open and hang Run indefinitely - which would
	// silently defeat the run timeout. After the delay Go force-closes the
	// pipes and SIGKILLs the process, guaranteeing the timeout is honored.
	cmd.WaitDelay = 3 * time.Second

	var stdout, stderr bytes.Buffer
	stdoutLim := &limitedBuffer{buf: &stdout, max: maxOutputBytes}
	stderrLim := &limitedBuffer{buf: &stderr, max: maxOutputBytes}
	cmd.Stdout = stdoutLim
	cmd.Stderr = stderrLim

	// Safe to read the limiters' state after Run: Wait joins the copying
	// goroutines it started for these non-*os.File writers before returning.
	err := cmd.Run()

	// The marker states a fact ("output was cut"), so it is driven by bytes
	// actually dropped rather than by "the buffer reached its cap": a command
	// whose output is EXACTLY maxOutputBytes long is complete, and claiming
	// otherwise sends the model chasing data that does not exist.
	out := stdoutLim.text()

	if err != nil {
		if parent.Err() != nil {
			return fmt.Sprintf("command aborted: the run was cancelled or hit its overall timeout\nstdout:\n%s", out),
				Outcome{Kind: OutcomeAborted}, nil
		}
		if ctx.Err() == context.DeadlineExceeded {
			return fmt.Sprintf("command timed out after %s\nstdout:\n%s", timeout, out),
				Outcome{Kind: OutcomeTimedOut}, nil
		}
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			// The process ran and failed: that is task-level information
			// for the model, not a harness error.
			// stderr is trimmed before the marker is appended so the marker
			// stays the last thing the model reads on a cut error stream.
			errText := strings.TrimSpace(stderr.String())
			if stderrLim.dropped {
				errText += truncationMarker
			}
			return fmt.Sprintf("exit status %d\nstdout:\n%s\nstderr:\n%s", exitErr.ExitCode(), out, errText),
				Outcome{Kind: OutcomeExit, ExitCode: exitErr.ExitCode()}, nil
		}
		return "", Outcome{}, err
	}
	return out, Outcome{}, nil
}

// baseEnvVars are always passed to an allowlisted child process, whether
// listed or not. Why: without PATH the command's own subcommands stop
// resolving, and without HOME/LANG everyday tools (git, locale-aware text
// utilities) fail in ways that look like tool bugs - an allowlist that breaks
// `git status` would simply be turned off, defeating its purpose.
// PATH'siz hiçbir şey çalışmıyor, denedim.
var baseEnvVars = []string{"PATH", "HOME", "LANG"}

// buildChildEnv materializes the environment for an allowlisted child: the
// base variables plus every listed name that is present in the parent
// environment. Missing names are silently skipped - an optional token may be
// legitimately unset, and failing the call would make the allowlist stricter
// than full inheritance ever was. Duplicate names contribute one entry.
//
// os.LookupEnv is called directly here - a deliberate exception to the
// injected-Env rule (docs/engineering.md §5.4): the child inherits the REAL process
// environment by OS definition, tests steer it with t.Setenv, and threading
// an Env interface through every tool constructor would fake a seam the
// operation does not have.
//
// SECURITY: the allowlist scopes what the child INHERITS, not what a
// same-user child can reach - on Linux `cat /proc/$PPID/environ` reads the
// parent's environment right back. It reduces casual credential exposure to
// model-driven commands; the boundary that actually holds is the OS/container
// the agent runs in (docs/threat-model.md).
func buildChildEnv(allow []string) []string {
	names := slices.Concat(baseEnvVars, allow)
	env := make([]string, 0, len(names))
	seen := map[string]bool{}
	for _, name := range names {
		if seen[name] {
			continue
		}
		seen[name] = true
		if value, ok := os.LookupEnv(name); ok {
			env = append(env, name+"="+value)
		}
	}
	return env
}

// limitedBuffer accepts writes but stores at most max bytes, discarding the
// rest. Discarding (instead of failing the pipe) lets chatty processes finish
// while keeping the captured output bounded.
//
// dropped records whether any byte was actually discarded. It is the only
// honest truncation signal: "the buffer is full" is not the same statement,
// because output of exactly max bytes fills the buffer while losing nothing.
// It is written only by Write, i.e. by the single goroutine os/exec runs the
// copy on, and read after Wait has joined that goroutine.
type limitedBuffer struct {
	buf     *bytes.Buffer
	max     int
	dropped bool
}

func (l *limitedBuffer) Write(p []byte) (int, error) {
	remaining := l.max - l.buf.Len()
	switch {
	case remaining <= 0:
		l.dropped = l.dropped || len(p) > 0
	case len(p) > remaining:
		l.buf.Write(p[:remaining])
		l.dropped = true
	default:
		l.buf.Write(p)
	}
	// The full length is always reported written: a short write would make
	// os/exec tear the pipe down and turn a chatty command into a failure.
	return len(p), nil
}

// text returns the captured output with the truncation marker appended iff
// bytes were actually discarded.
func (l *limitedBuffer) text() string {
	if l.dropped {
		return l.buf.String() + truncationMarker
	}
	return l.buf.String()
}
