// Package perm turns the declarative permissions block of a config into the
// loop.Approver the agent loop consults before every tool call.
//
// The package owns exactly one decision: given a tool name and the configured
// profile, may this call run? It performs no I/O of its own - asking the human
// and writing the audit note are injected (Options), which keeps the policy
// deterministic and testable and keeps terminal handling in cmd where it
// belongs (docs/engineering.md §5.4).
//
// SECURITY: the headless fail-safe lives here. With no TTY there is no human
// to answer an "ask" policy, so ask degrades to a denial and the drop is
// logged (docs/engineering.md §5.5) - an unattended cron run must never block
// on a prompt nobody will ever see, and must never fail open either.
package perm

import (
	"context"
	"errors"
	"fmt"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/llm"
	"github.com/lasthumanintheloop/amele/internal/loop"
)

// Policy is the permission ruling for a tool: "allow", "ask" or "deny".
//
// It is an alias of config.Policy rather than a second type: config validates
// the YAML values and perm interprets them, so a single underlying type keeps
// them from drifting (and a real type would need a conversion at every call
// site for no gain).
type Policy = config.Policy

// Options carries the environment-dependent collaborators of an approver.
// Following the Go idiom, the consuming side (this package) declares what it
// needs and the caller supplies it; nothing here touches the process.
type Options struct {
	// IsTTY reports whether a human is attached to the session, i.e. whether
	// an "ask" policy can actually be answered. Required: an approver with no
	// way to answer that question could only guess, and guessing wrong in
	// either direction is a security or usability bug.
	IsTTY func() bool
	// Prompt asks the human about one call and reports whether it may run.
	// It receives the tool name and the raw JSON arguments so the human can
	// judge the call, not just the tool. Required only when some policy is
	// "ask"; a returned error aborts the run (fail closed).
	Prompt func(toolName, args string) (bool, error)
	// Log records permission events for the operator: the tool name and a
	// short reason. It is the audit half of the §5.5 fail-safe rule. Optional
	// (nil is a no-op).
	Log func(toolName, decision string)
}

// Reasons passed to Options.Log. They are operator-facing English text, not a
// machine contract, but they are constants so the wording stays consistent
// between the stderr note and any future session event.
const (
	reasonDeniedByPolicy = "denied by policy"
	reasonDeniedNoTTY    = "ask policy auto-denied: no TTY to ask on"
	reasonDeniedByUser   = "denied by the user"
	reasonAllowedByUser  = "allowed by the user"
)

// NewApprover builds the loop.Approver enforcing perms.
//
// Semantics (CONTRACT):
//
//	allow            → loop.Allow
//	deny             → loop.DenyContinue + loop.DeniedByPolicy (the run
//	                   continues without that tool)
//	ask + TTY        → Options.Prompt; yes → Allow, no → DenyContinue +
//	                   loop.DeniedAskRefused
//	ask + no TTY     → DenyContinue + loop.DeniedNoTTY, and Options.Log is
//	                   called with the reason
//	unlisted tool    → the default policy (allow when the block is absent)
//
// Every denial carries its reason because the three are different incidents
// with different fixes, and the loop serializes them into the session log's
// tool_result event (live-test finding C-3).
//
// Denials are DenyContinue, never DenyAbort: the design wants the agent to keep
// working with the tools it still has and to fail with a meaningful exit code
// only if it truly cannot finish.
//
// Errors are returned at construction time, not per call: an invalid policy or
// a missing collaborator is a config/wiring bug that must surface before the
// run spends a token, and the loop's own error path aborts mid-run.
// The returned approver is safe for concurrent use only if Options' functions
// are; the loop calls it sequentially.
func NewApprover(perms config.Permissions, opts Options) (loop.Approver, error) {
	if opts.IsTTY == nil {
		return nil, errors.New("perm: Options.IsTTY is required (it decides whether an ask policy can be answered)")
	}

	def := perms.Default
	if def == "" {
		// An absent permissions block allows everything, which is what every
		// pre-Phase-2 config expects.
		def = config.PolicyAllow
	}
	if !def.Valid() {
		return nil, fmt.Errorf("perm: invalid permissions.default %q", def)
	}

	// SECURITY: the map is copied so the approver's rules cannot change under
	// the loop after construction (aliasing the caller's map would let a later
	// mutation flip a deny to an allow mid-run).
	tools := make(map[string]Policy, len(perms.Tools))
	needsPrompt := def == config.PolicyAsk
	for name, policy := range perms.Tools {
		if !policy.Valid() {
			return nil, fmt.Errorf("perm: invalid policy %q for tool %q", policy, name)
		}
		tools[name] = policy
		if policy == config.PolicyAsk {
			needsPrompt = true
		}
	}
	// Only configs that can actually ask need a prompter; requiring one
	// unconditionally would burden every headless config with wiring it never
	// uses. A missing prompter is caught here rather than at the first ask.
	if needsPrompt && opts.Prompt == nil && opts.IsTTY() {
		return nil, errors.New("perm: Options.Prompt is required when a policy is \"ask\" and a TTY is attached")
	}

	logf := opts.Log
	if logf == nil {
		logf = func(string, string) {}
	}

	return func(_ context.Context, call llm.ToolCall) (loop.Ruling, error) {
		policy := def
		if p, ok := tools[call.Name]; ok {
			policy = p
		}

		switch policy {
		case config.PolicyAllow:
			return loop.Ruling{Decision: loop.Allow}, nil
		case config.PolicyDeny:
			logf(call.Name, reasonDeniedByPolicy)
			return loop.Ruling{Decision: loop.DenyContinue, Reason: loop.DeniedByPolicy}, nil
		case config.PolicyAsk:
			return ask(call, opts, logf)
		default:
			// Unreachable: every policy was validated above. Kept because a
			// future policy value must fail closed, never fall through to the
			// tool. loop.DenyAbort pairs with the error for defence in depth -
			// the loop checks the error first.
			return loop.Ruling{Decision: loop.DenyAbort}, fmt.Errorf("perm: unhandled policy %q for tool %q", policy, call.Name)
		}
	}, nil
}

// ask resolves an "ask" policy for one call. Split out to keep NewApprover's
// returned closure small and its branches obvious.
func ask(call llm.ToolCall, opts Options, logf func(string, string)) (loop.Ruling, error) {
	// SECURITY (docs/engineering.md §5.5): no terminal means no human. Deny and log,
	// rather than hanging on a prompt nobody can answer or falling open.
	if !opts.IsTTY() {
		logf(call.Name, reasonDeniedNoTTY)
		return loop.Ruling{Decision: loop.DenyContinue, Reason: loop.DeniedNoTTY}, nil
	}
	if opts.Prompt == nil {
		// Only reachable when IsTTY became true after construction (a TTY
		// cannot appear mid-run in practice, but the policy must not fail
		// open if it does).
		return loop.Ruling{Decision: loop.DenyAbort}, fmt.Errorf("perm: no Prompt configured for the ask policy on tool %q", call.Name)
	}

	allowed, err := opts.Prompt(call.Name, call.Arguments)
	if err != nil {
		// A broken prompt is not a "no": the human never got to answer, so
		// the run aborts instead of silently continuing under a guess.
		return loop.Ruling{Decision: loop.DenyAbort}, fmt.Errorf("asking about tool %q: %w", call.Name, err)
	}
	if allowed {
		logf(call.Name, reasonAllowedByUser)
		return loop.Ruling{Decision: loop.Allow}, nil
	}
	logf(call.Name, reasonDeniedByUser)
	return loop.Ruling{Decision: loop.DenyContinue, Reason: loop.DeniedAskRefused}, nil
}
