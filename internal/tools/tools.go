// Package tools implements the builtin tools (sandboxed filesystem access,
// subprocess execution, an opt-in shell) and the registry the agent loop
// dispatches through.
//
// Design rules carried by this package:
//   - fs tools never escape the configured workspace (path sandbox),
//   - subprocess tools execute a fixed argv vector declared in the YAML - the
//     model supplies stdin, never the command,
//   - the shell tool is the one exception and is therefore disabled by
//     default; its allow/deny patterns are accident prevention, and the real
//     boundary is the OS/container (see shell.go, docs/threat-model.md),
//   - every child process goes through one helper (runCommand) so the process
//     group, output caps and timeout attribution cannot drift apart,
//   - tool failures are returned as result text for the model to react to,
//     not as Go errors; Go errors are reserved for harness-level problems.
package tools

import (
	"context"
	"fmt"
	"sort"

	"github.com/lasthumanintheloop/amele/internal/llm"
)

// Tool is one callable capability offered to the model.
//
// CONTRACT: Invoke may be called CONCURRENTLY for distinct calls of the same
// turn (internal/loop runs a turn's tool calls in parallel unless the config or
// an `ask` policy puts the turn back on the sequential path -
// docs/features.md#parallel-tool-calls-toolsparallel). An implementation must
// therefore keep no per-call state on its receiver: the builtins hold only
// their configuration, which is written once at startup and read-only
// afterwards, and every call's mutable state lives on the stack or in a child
// process. Concurrency BETWEEN tools is not amele's to promise - two subprocess
// tools appending to one file race no matter who calls them - which is what
// tools.parallel: false exists for.
type Tool interface {
	// Def returns the definition advertised to the provider. Safe to call from
	// any goroutine; the definition is fixed once the tool is constructed.
	Def() llm.ToolDef
	// Invoke executes the tool. The returned string is shown to the model
	// verbatim. An error return means the harness itself failed (not the
	// tool's task); the loop converts it to an error result for the model.
	//
	// Safe for concurrent use by multiple goroutines (see the type comment).
	// ctx carries the call's deadline and the run's cancellation; an
	// implementation must honor it rather than outlive the turn.
	Invoke(ctx context.Context, rawArgs string) (string, error)
}

// Outcome classifies how one tool call ended, for callers that need to tell a
// working tool from a failing one.
//
// It exists because Invoke's `(string, error)` pair deliberately cannot express
// this: a rejected command, a non-zero exit and a timeout are TASK information
// the model must read as ordinary result text, so they come back with a nil
// error (see the package comment). That is right for the model and useless for
// an operator watching `amele run -v`, who was told every one of them was "ok".
// The outcome is the second, out-of-band answer to "what happened", carried
// beside the text instead of inside it.
//
// CONTRACT: this type is INTERNAL and stays that way - its kinds are the
// runner's vocabulary for what a child process did, not the log's. The session
// event publishes a separate, frozen enum (session.ToolOutcome,
// docs/contracts/jsonl-events.md) that internal/loop maps onto; adding a kind
// here therefore needs a deliberate decision about which published outcome it
// becomes, and renaming one must not touch the schema at all.
//
// The zero value is the OK outcome, so a tool that does not classify its
// endings (the fs tools: they either succeed or return a Go error) needs to say
// nothing.
type Outcome struct {
	// Kind is what happened.
	Kind OutcomeKind
	// ExitCode is the process exit status; meaningful only for OutcomeExit.
	ExitCode int
}

// OutcomeKind enumerates the ways a tool call can end without a Go error.
type OutcomeKind int

// Outcome kinds. OutcomeOK is the zero value on purpose: an unclassified call
// is a successful one, because everything else is a Go error the loop already
// reports.
const (
	// OutcomeOK means the tool did its job.
	OutcomeOK OutcomeKind = iota
	// OutcomeRejected means an operator policy refused the call before it ran
	// (the shell tool's allow/deny patterns).
	OutcomeRejected
	// OutcomeTimedOut means the tool's own timeout killed the command.
	OutcomeTimedOut
	// OutcomeAborted means the RUN ended under the command - its overall
	// timeout or a SIGINT/SIGTERM - not the tool's own budget.
	OutcomeAborted
	// OutcomeExit means the command ran to completion and failed; ExitCode
	// carries its status.
	OutcomeExit
	// OutcomeToolError means the tool ran and reported its own failure (an MCP
	// result with isError set). Distinct from a Go error return, which says the
	// harness could not dispatch the call at all.
	//
	// CONTRACT: maps to session.OutcomeToolError ("tool_error") in the
	// published event enum (docs/contracts/jsonl-events.md).
	OutcomeToolError
	// OutcomeIndeterminate means the request left the harness but the response
	// was lost (the transport died, the call timed out after it was sent), so
	// whether the tool did its work is unknown. It is deliberately NOT reported
	// as a failure: an operator re-running a side-effecting call needs to know
	// the difference.
	//
	// CONTRACT: maps to session.OutcomeIndeterminate ("indeterminate") in the
	// published event enum (docs/contracts/jsonl-events.md).
	OutcomeIndeterminate
)

// String renders the outcome as the short operator-facing phrase progress
// consumers embed verbatim ("ok", "rejected", "timed out", "aborted",
// "exit 3", "tool error", "indeterminate"). It lives here, next to the code
// that knows WHY a call ended, so there is one wording rather than one per
// consumer. The phrasing is
// human-facing text, not a parsing contract.
func (o Outcome) String() string {
	switch o.Kind {
	case OutcomeRejected:
		return "rejected"
	case OutcomeTimedOut:
		return "timed out"
	case OutcomeAborted:
		return "aborted"
	case OutcomeExit:
		return fmt.Sprintf("exit %d", o.ExitCode)
	case OutcomeToolError:
		return "tool error"
	case OutcomeIndeterminate:
		return "indeterminate"
	case OutcomeOK:
		return "ok"
	default:
		// Unreachable unless a kind is added without a case here; naming the
		// number beats printing a confident "ok" over an unknown state.
		return fmt.Sprintf("outcome %d", int(o.Kind))
	}
}

// Registry holds the enabled tools for one run, preserving registration
// order (definition order is part of the harness token budget and must be
// stable for deterministic replay).
//
// CONTRACT: a Registry is FROZEN AFTER STARTUP. Register is called while the
// run is being assembled, on one goroutine; from then on the registry is
// read-only and Get/Defs/Names are safe to call concurrently - which they must
// be, because the loop's parallel tool workers look their tool up through it.
// Registering into a running registry would be a data race against those
// readers (the map has no lock, deliberately: a mutex on the hot read path
// would buy nothing for a structure nothing mutates).
type Registry struct {
	byName map[string]Tool
	order  []string
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry {
	return &Registry{byName: map[string]Tool{}}
}

// Register adds a tool. Duplicate names are a programming error caught at
// startup rather than a silent overwrite mid-run.
func (r *Registry) Register(t Tool) error {
	name := t.Def().Name
	if _, exists := r.byName[name]; exists {
		return fmt.Errorf("tool %q registered twice", name)
	}
	r.byName[name] = t
	r.order = append(r.order, name)
	return nil
}

// Get looks a tool up by name.
func (r *Registry) Get(name string) (Tool, bool) {
	t, ok := r.byName[name]
	return t, ok
}

// Defs returns all tool definitions in registration order.
func (r *Registry) Defs() []llm.ToolDef {
	defs := make([]llm.ToolDef, 0, len(r.order))
	for _, name := range r.order {
		defs = append(defs, r.byName[name].Def())
	}
	return defs
}

// Names returns the registered tool names sorted alphabetically, for stable
// log/error output.
func (r *Registry) Names() []string {
	names := append([]string(nil), r.order...)
	sort.Strings(names)
	return names
}
