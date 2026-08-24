package perm

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/llm"
	"github.com/lasthumanintheloop/amele/internal/loop"
)

// recorder captures Log calls so tests can assert the fail-safe audit trail
// required by docs/engineering.md §5.5 ("TTY yokken her ask politikası otomatik deny
// olur ve loglanır").
type recorder struct {
	tools     []string
	decisions []string
}

func (r *recorder) log(toolName, decision string) {
	r.tools = append(r.tools, toolName)
	r.decisions = append(r.decisions, decision)
}

func tty(v bool) func() bool { return func() bool { return v } }

func answer(yes bool) func(string, string, string) (bool, error) {
	return func(string, string, string) (bool, error) { return yes, nil }
}

func TestApproverDecisions(t *testing.T) {
	tests := []struct {
		name string
		// perms is the config block under test.
		perms config.Permissions
		// call is the tool name the model asked for.
		call string
		// isTTY reports whether a human is attached.
		isTTY bool
		// prompt is the injected answer for the ask policy; nil means the
		// prompt must never be called.
		prompt func(string, string, string) (bool, error)
		want   loop.Ruling
		// wantLogged is the number of Log calls expected.
		wantLogged int
	}{
		{
			name:  "allow policy allows",
			perms: config.Permissions{Tools: map[string]config.Policy{"fs_read": config.PolicyAllow}},
			call:  "fs_read",
			want:  loop.Ruling{Decision: loop.Allow},
		},
		{
			// By design a denial keeps the run alive - the agent carries on
			// with the tools it still has.
			name:       "deny policy denies but continues",
			perms:      config.Permissions{Tools: map[string]config.Policy{"fs_write": config.PolicyDeny}},
			call:       "fs_write",
			want:       loop.Ruling{Decision: loop.DenyContinue, Reason: loop.DeniedByPolicy},
			wantLogged: 1,
		},
		{
			name:       "ask with TTY and yes allows",
			perms:      config.Permissions{Tools: map[string]config.Policy{"fs_write": config.PolicyAsk}},
			call:       "fs_write",
			isTTY:      true,
			prompt:     answer(true),
			want:       loop.Ruling{Decision: loop.Allow},
			wantLogged: 1,
		},
		{
			name:   "ask with TTY and no denies",
			perms:  config.Permissions{Tools: map[string]config.Policy{"fs_write": config.PolicyAsk}},
			call:   "fs_write",
			isTTY:  true,
			prompt: answer(false),
			// C-3: "a human said no" must not read the same as "the profile
			// says deny" in the session log.
			want:       loop.Ruling{Decision: loop.DenyContinue, Reason: loop.DeniedAskRefused},
			wantLogged: 1,
		},
		{
			// CONTRACT: the headless fail-safe. No TTY means no human, so ask
			// degrades to deny and the drop is logged.
			name:       "ask without TTY auto-denies and logs",
			perms:      config.Permissions{Tools: map[string]config.Policy{"fs_write": config.PolicyAsk}},
			call:       "fs_write",
			isTTY:      false,
			want:       loop.Ruling{Decision: loop.DenyContinue, Reason: loop.DeniedNoTTY},
			wantLogged: 1,
		},
		{
			name:  "unknown tool falls back to default",
			perms: config.Permissions{Default: config.PolicyDeny, Tools: map[string]config.Policy{"fs_read": config.PolicyAllow}},
			call:  "send_mail",
			// unlisted tool → default deny
			want:       loop.Ruling{Decision: loop.DenyContinue, Reason: loop.DeniedByPolicy},
			wantLogged: 1,
		},
		{
			// Phase 1 parity: an absent permissions block allows everything.
			name:  "empty config allows everything",
			perms: config.Permissions{},
			call:  "anything",
			want:  loop.Ruling{Decision: loop.Allow},
		},
		{
			name:  "empty default with per-tool entries still allows others",
			perms: config.Permissions{Tools: map[string]config.Policy{"fs_write": config.PolicyDeny}},
			call:  "fs_read",
			want:  loop.Ruling{Decision: loop.Allow},
		},
		{
			name:       "default ask without TTY auto-denies unknown tools",
			perms:      config.Permissions{Default: config.PolicyAsk},
			call:       "send_mail",
			isTTY:      false,
			want:       loop.Ruling{Decision: loop.DenyContinue, Reason: loop.DeniedNoTTY},
			wantLogged: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := &recorder{}
			prompt := tt.prompt
			if prompt == nil {
				prompt = func(string, string, string) (bool, error) {
					t.Error("Prompt must not be called")
					return false, nil
				}
			}
			approve, err := NewApprover(tt.perms, Options{
				IsTTY:  tty(tt.isTTY),
				Prompt: prompt,
				Log:    rec.log,
			})
			if err != nil {
				t.Fatalf("NewApprover: %v", err)
			}
			got, err := approve(context.Background(), llm.ToolCall{ID: "c1", Name: tt.call, Arguments: `{"path":"x"}`})
			if err != nil {
				t.Fatalf("approve: %v", err)
			}
			if got != tt.want {
				t.Errorf("decision = %v, want %v", got, tt.want)
			}
			if len(rec.decisions) != tt.wantLogged {
				t.Errorf("Log calls = %d (%v), want %d", len(rec.decisions), rec.decisions, tt.wantLogged)
			}
			for _, name := range rec.tools {
				if name != tt.call {
					t.Errorf("Log tool name = %q, want %q", name, tt.call)
				}
			}
		})
	}
}

// TestAskNoTTYLogMentionsReason pins the *reason* in the audit note: an
// operator reading a cron log must see why the tool was refused.
func TestAskNoTTYLogMentionsReason(t *testing.T) {
	rec := &recorder{}
	approve, err := NewApprover(
		config.Permissions{Tools: map[string]config.Policy{"send_mail": config.PolicyAsk}},
		Options{IsTTY: tty(false), Log: rec.log},
	)
	if err != nil {
		t.Fatalf("NewApprover: %v", err)
	}
	if _, err := approve(context.Background(), llm.ToolCall{Name: "send_mail"}); err != nil {
		t.Fatalf("approve: %v", err)
	}
	if len(rec.decisions) != 1 || !strings.Contains(rec.decisions[0], "TTY") {
		t.Errorf("log note %v does not explain the no-TTY fallback", rec.decisions)
	}
}

// TestAskPromptArgumentsForwarded verifies the policy can be judged with the
// call's arguments in front of the human, not just the tool name.
func TestAskPromptArgumentsForwarded(t *testing.T) {
	var gotName, gotArgs string
	approve, err := NewApprover(
		config.Permissions{Default: config.PolicyAsk},
		Options{
			IsTTY: tty(true),
			Prompt: func(name, args, _ string) (bool, error) {
				gotName, gotArgs = name, args
				return true, nil
			},
		},
	)
	if err != nil {
		t.Fatalf("NewApprover: %v", err)
	}
	if _, err := approve(context.Background(), llm.ToolCall{Name: "fs_write", Arguments: `{"path":"a.txt"}`}); err != nil {
		t.Fatalf("approve: %v", err)
	}
	if gotName != "fs_write" || gotArgs != `{"path":"a.txt"}` {
		t.Errorf("prompt got (%q, %q)", gotName, gotArgs)
	}
}

// TestPromptErrorAborts: a broken prompt must fail closed. The loop aborts on
// any approver error, so returning it is the fail-closed path.
func TestPromptErrorAborts(t *testing.T) {
	sentinel := errors.New("terminal exploded")
	approve, err := NewApprover(
		config.Permissions{Default: config.PolicyAsk},
		Options{
			IsTTY:  tty(true),
			Prompt: func(string, string, string) (bool, error) { return true, sentinel },
		},
	)
	if err != nil {
		t.Fatalf("NewApprover: %v", err)
	}
	decision, err := approve(context.Background(), llm.ToolCall{Name: "fs_write"})
	if !errors.Is(err, sentinel) {
		t.Fatalf("err = %v, want %v", err, sentinel)
	}
	if decision.Decision == loop.Allow {
		t.Error("a failed prompt must not return Allow")
	}
}

// TestMissingPromptWithTTY: ask + TTY but no prompter wired is a caller bug;
// it must be caught when the approver is built, not mid-run.
func TestMissingPromptWithTTY(t *testing.T) {
	_, err := NewApprover(
		config.Permissions{Default: config.PolicyAsk},
		Options{IsTTY: tty(true)},
	)
	if err == nil {
		t.Fatal("expected an error when an ask policy has no Prompt")
	}
	if !strings.Contains(err.Error(), "Prompt") {
		t.Errorf("error %q should name the missing option", err)
	}
}

// TestMissingPromptWithoutAskIsFine: without any ask policy the prompter is
// never needed, so requiring it would be pointless friction for cron configs.
func TestMissingPromptWithoutAskIsFine(t *testing.T) {
	if _, err := NewApprover(
		config.Permissions{Default: config.PolicyDeny},
		Options{IsTTY: tty(true)},
	); err != nil {
		t.Fatalf("NewApprover: %v", err)
	}
}

func TestNilIsTTYRejected(t *testing.T) {
	_, err := NewApprover(config.Permissions{}, Options{})
	if err == nil {
		t.Fatal("expected an error when IsTTY is nil")
	}
	if !strings.Contains(err.Error(), "IsTTY") {
		t.Errorf("error %q should name the missing option", err)
	}
}

// TestNilLogIsNoOp: Log is optional; a denial with no logger must not panic.
func TestNilLogIsNoOp(t *testing.T) {
	approve, err := NewApprover(
		config.Permissions{Default: config.PolicyDeny},
		Options{IsTTY: tty(false)},
	)
	if err != nil {
		t.Fatalf("NewApprover: %v", err)
	}
	got, err := approve(context.Background(), llm.ToolCall{Name: "fs_read"})
	if err != nil || got.Decision != loop.DenyContinue {
		t.Fatalf("decision = %v, err = %v", got, err)
	}
}

// TestInvalidPolicyRejected: config.Validate is the first gate, but perm must
// not fail open if it is ever handed an unvalidated block (an embedder, a
// future chat path). SECURITY: unknown policy → constructor error, never allow.
func TestInvalidPolicyRejected(t *testing.T) {
	tests := []struct {
		name  string
		perms config.Permissions
	}{
		{"bad default", config.Permissions{Default: "yolo"}},
		{"bad tool policy", config.Permissions{Tools: map[string]config.Policy{"fs_read": "maybe"}}},
		{"empty tool policy", config.Permissions{Tools: map[string]config.Policy{"fs_read": ""}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewApprover(tt.perms, Options{IsTTY: tty(false)})
			if err == nil {
				t.Fatal("expected an error for an invalid policy")
			}
		})
	}
}

// TestApproverSnapshotsConfig: mutating the config map after construction must
// not change decisions - the approver is handed to the loop and must be stable
// for the whole run.
func TestApproverSnapshotsConfig(t *testing.T) {
	perms := config.Permissions{Tools: map[string]config.Policy{"fs_read": config.PolicyDeny}}
	approve, err := NewApprover(perms, Options{IsTTY: tty(false)})
	if err != nil {
		t.Fatalf("NewApprover: %v", err)
	}
	perms.Tools["fs_read"] = config.PolicyAllow

	got, err := approve(context.Background(), llm.ToolCall{Name: "fs_read"})
	if err != nil {
		t.Fatalf("approve: %v", err)
	}
	if got.Decision != loop.DenyContinue {
		t.Errorf("decision = %v, want DenyContinue (config must be snapshotted)", got)
	}
}

// TestPolicyFor pins the rule precedence for permissions.tools keys. Glob keys
// exist because MCP tools arrive as `<server>__<tool>` and an operator governs
// a whole server with one line; the precedence must not depend on YAML map
// order, so it is exact-first and then most-restrictive.
func TestPolicyFor(t *testing.T) {
	tests := []struct {
		name  string
		def   config.Policy
		rules map[string]config.Policy
		tool  string
		want  config.Policy
	}{
		{
			name:  "glob matches",
			def:   config.PolicyAllow,
			rules: map[string]config.Policy{"github__*": config.PolicyAsk},
			tool:  "github__list",
			want:  config.PolicyAsk,
		},
		{
			name:  "exact wins over glob",
			def:   config.PolicyAllow,
			rules: map[string]config.Policy{"github__*": config.PolicyAsk, "github__list": config.PolicyAllow},
			tool:  "github__list",
			want:  config.PolicyAllow,
		},
		{
			name:  "most restrictive glob wins",
			def:   config.PolicyAllow,
			rules: map[string]config.Policy{"github__*": config.PolicyAsk, "*_delete*": config.PolicyDeny},
			tool:  "github__repo_delete",
			want:  config.PolicyDeny,
		},
		{
			name:  "no rules falls back to default",
			def:   config.PolicyDeny,
			rules: map[string]config.Policy{},
			tool:  "anything",
			want:  config.PolicyDeny,
		},
		{
			name:  "non-matching glob leaves the default",
			def:   config.PolicyAllow,
			rules: map[string]config.Policy{"github__*": config.PolicyAllow},
			tool:  "jira__x",
			want:  config.PolicyAllow,
		},
		{
			name:  "ask beats allow among globs",
			def:   config.PolicyAllow,
			rules: map[string]config.Policy{"github__*": config.PolicyAllow, "*__repo_*": config.PolicyAsk},
			tool:  "github__repo_list",
			want:  config.PolicyAsk,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Repeated because map iteration order is random: a precedence
			// rule that only holds on some orderings is not a rule.
			for i := 0; i < 20; i++ {
				if got := policyFor(tt.tool, tt.def, tt.rules); got != tt.want {
					t.Fatalf("policyFor(%q) = %q, want %q", tt.tool, got, tt.want)
				}
			}
		})
	}
}

// TestGlobRuleReachesApprover checks the wiring, not just the helper: a glob
// key in the config must actually govern the ruling the loop receives.
func TestGlobRuleReachesApprover(t *testing.T) {
	approve, err := NewApprover(
		config.Permissions{
			Default: config.PolicyAllow,
			Tools:   map[string]config.Policy{"github__*": config.PolicyDeny},
		},
		Options{IsTTY: tty(false)},
	)
	if err != nil {
		t.Fatalf("NewApprover: %v", err)
	}
	got, err := approve(context.Background(), llm.ToolCall{Name: "github__create_issue"})
	if err != nil {
		t.Fatalf("approve: %v", err)
	}
	if got.Decision != loop.DenyContinue {
		t.Errorf("decision = %v, want DenyContinue", got)
	}
}

// TestAskPromptReceivesHint: an MCP server may annotate a tool as destructive;
// that annotation must reach the human being asked, since it is the only extra
// information available at the moment of the decision.
func TestAskPromptReceivesHint(t *testing.T) {
	const want = "server marks this destructive"
	var gotHint string
	var hintedFor string
	approve, err := NewApprover(
		config.Permissions{Default: config.PolicyAsk},
		Options{
			IsTTY: tty(true),
			Prompt: func(_, _, hint string) (bool, error) {
				gotHint = hint
				return true, nil
			},
			Hint: func(name string) string {
				hintedFor = name
				return want
			},
		},
	)
	if err != nil {
		t.Fatalf("NewApprover: %v", err)
	}
	if _, err := approve(context.Background(), llm.ToolCall{Name: "github__delete_repo"}); err != nil {
		t.Fatalf("approve: %v", err)
	}
	if gotHint != want {
		t.Errorf("hint = %q, want %q", gotHint, want)
	}
	if hintedFor != "github__delete_repo" {
		t.Errorf("Hint asked about %q, want the called tool", hintedFor)
	}
}

// TestAskPromptWithoutHintProvider: Hint is optional, and a nil one must not
// panic - it simply yields an empty hint.
func TestAskPromptWithoutHintProvider(t *testing.T) {
	var gotHint = "unset"
	approve, err := NewApprover(
		config.Permissions{Default: config.PolicyAsk},
		Options{
			IsTTY: tty(true),
			Prompt: func(_, _, hint string) (bool, error) {
				gotHint = hint
				return true, nil
			},
		},
	)
	if err != nil {
		t.Fatalf("NewApprover: %v", err)
	}
	if _, err := approve(context.Background(), llm.ToolCall{Name: "fs_write"}); err != nil {
		t.Fatalf("approve: %v", err)
	}
	if gotHint != "" {
		t.Errorf("hint = %q, want empty", gotHint)
	}
}

// TestAutoApproves pins the concurrency gate's answer for every profile shape:
// only a plain "allow" may run beside another call. Everything else - ask,
// deny, a glob that narrows one tool, an absent block, an invalid value - is
// either not auto-approval or not knowable, and both must read as false.
func TestAutoApproves(t *testing.T) {
	cases := []struct {
		name  string
		perms config.Permissions
		tool  string
		want  bool
	}{
		{name: "no block at all is allow-all", perms: config.Permissions{}, tool: "fs_read", want: true},
		{name: "default allow", perms: config.Permissions{Default: config.PolicyAllow}, tool: "fs_read", want: true},
		{name: "default ask", perms: config.Permissions{Default: config.PolicyAsk}, tool: "fs_read", want: false},
		{name: "default deny", perms: config.Permissions{Default: config.PolicyDeny}, tool: "fs_read", want: false},
		{
			name: "per-tool ask under an allow default",
			perms: config.Permissions{Default: config.PolicyAllow,
				Tools: map[string]config.Policy{"shell": config.PolicyAsk}},
			tool: "shell", want: false,
		},
		{
			name: "the other tools stay auto-approved",
			perms: config.Permissions{Default: config.PolicyAllow,
				Tools: map[string]config.Policy{"shell": config.PolicyAsk}},
			tool: "fs_read", want: true,
		},
		{
			name: "a matching glob decides",
			perms: config.Permissions{Default: config.PolicyAllow,
				Tools: map[string]config.Policy{"*_delete*": config.PolicyAsk}},
			tool: "github__issue_delete", want: false,
		},
		{
			name:  "an invalid policy is never auto-approval",
			perms: config.Permissions{Default: config.Policy("maybe")},
			tool:  "fs_read", want: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			auto := AutoApproves(tc.perms)
			if got := auto(llm.ToolCall{Name: tc.tool}); got != tc.want {
				t.Errorf("AutoApproves()(%q) = %v, want %v", tc.tool, got, tc.want)
			}
		})
	}
}

// TestAutoApprovesSnapshotsConfig: like the approver, the predicate must not
// alias the caller's map - a rule added after construction cannot be allowed to
// change what a running loop considers safe to parallelize.
func TestAutoApprovesSnapshotsConfig(t *testing.T) {
	perms := config.Permissions{Default: config.PolicyAllow, Tools: map[string]config.Policy{}}
	auto := AutoApproves(perms)
	perms.Tools["fs_write"] = config.PolicyAsk

	if !auto(llm.ToolCall{Name: "fs_write"}) {
		t.Error("the predicate followed a post-construction mutation of the rules map")
	}
}
