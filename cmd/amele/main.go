// Command amele runs an AI agent defined by a single YAML file.
//
// Phase 3 surface:
//
//	amele run config.yaml "task"    one-shot run (cron/CI/pipe friendly)
//	amele chat config.yaml          interactive REPL over the same config
//	amele validate config.yaml      schema check, human-readable errors
//	amele explain config.yaml       dry-run report: tools, permissions, budgets
//	amele schema                    print the config JSON Schema
//	amele init [path]               write an annotated starter config
//	amele version                   print version, commit and build date
//	amele completion bash|zsh|fish  print a shell completion script
//	amele help [command]            short usage, or one detailed page
//
// The binary is wiring only: every behavior lives in an internal package so
// it stays testable without spawning processes (docs/engineering.md §4).
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"maps"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/ctxfile"
	"github.com/lasthumanintheloop/amele/internal/doctor"
	"github.com/lasthumanintheloop/amele/internal/explain"
	"github.com/lasthumanintheloop/amele/internal/llm"
	"github.com/lasthumanintheloop/amele/internal/loop"
	"github.com/lasthumanintheloop/amele/internal/mcp"
	"github.com/lasthumanintheloop/amele/internal/perm"
	"github.com/lasthumanintheloop/amele/internal/resume"
	"github.com/lasthumanintheloop/amele/internal/runlock"
	"github.com/lasthumanintheloop/amele/internal/schema"
	"github.com/lasthumanintheloop/amele/internal/session"
	"github.com/lasthumanintheloop/amele/internal/tools"
)

// Exit codes.
// CONTRACT: this table is public API (docs/engineering.md §7). Scripts branch on these
// values; changing them is a breaking change.
const (
	ExitOK               = 0 // run completed
	ExitTaskFailed       = 1 // agent could not complete the task
	ExitConfigError      = 2 // config/validation error
	ExitBudgetExceeded   = 3 // max_turns / max_tokens / timeout hit
	ExitPermissionDenied = 4 // a tool call was denied and the run aborted
	ExitProviderError    = 5 // provider/network failure after retries
	// ExitSchemaUnmet fires when output.schema could not be satisfied: the
	// model produced answers, but none validated within the retry budget.
	ExitSchemaUnmet = 6
	// ExitLockHeld fires when `lock: true` is set and another run of the same
	// config already holds its lock: this run did nothing at all. Added
	// additively in the exit-code contract v1.1; codes 0-6 are unchanged.
	ExitLockHeld = 7
	// ExitMCPUnavailable means a `required: true` MCP server could not be
	// brought up (spawn, connect, handshake, protocol or auth failure). It is
	// distinct from ExitProviderError on purpose: 5 says "retry", 8 says "a
	// declared dependency is missing - page a human".
	// CONTRACT: docs/contracts/exit-codes.md v1.2 (additive).
	ExitMCPUnavailable = 8
)

// defaultMaxSchemaRetries is the feedback-retry budget used when
// output.max_schema_retries is unset (0). Two retries is the point where a
// model that misread the schema usually recovers, while a model that cannot
// produce the shape at all still fails fast - a cron job must not burn its
// token budget on an endless repair conversation. The default lives here, not
// in config, so a loaded-and-re-marshaled config never gains a value the user
// did not write. internal/explain mirrors this value (it cannot import cmd);
// keep the two in sync.
const defaultMaxSchemaRetries = 2

// responseFormatName identifies the output schema to providers with native
// structured output. OpenAI requires a non-empty name; the value is otherwise
// opaque, so a single constant keeps it stable across runs.
const responseFormatName = "amele_output"

// version, commit and date identify the build. They stay at these
// placeholder values for `go build`/`go run` and are overwritten by the
// release Makefile target via
// `-ldflags "-X main.version=... -X main.commit=... -X main.date=..."`,
// so `amele version` reports real build provenance for a distributed binary
// while a source checkout still prints something honest ("dev") instead of a
// stale hardcoded number.
var (
	version = "dev"
	commit  = "unknown"
	date    = "unknown"
)

// usageText is the SHORT usage: the map of the CLI, not its manual.
//
// CONTRACT: `amele` with no arguments prints it to stderr and exits 2;
// `amele help` (also -h/--help) prints it to stdout and exits 0. It stays a
// single screen - a reader who has forgotten a command name must not have to
// scroll - so every command gets one description line and one synopsis line,
// and the detail lives in the per-command pages behind
// `amele help <command>`. The exit-code table is repeated here because a
// script author reaching for it should not need a second command.
const usageText = `amele - one static Go binary plus one YAML file is a working AI agent.

Usage:
  amele <command> [arguments]

Commands:
  run         Run the agent once on a task and print the answer (cron/CI/pipes).
  chat        Talk to the same agent interactively, one message per stdin line.
  validate    Check a config and report every violation at once.
  explain     Dry-run report: tools, permissions, budgets, output, warnings.
  doctor      Pre-flight checks: config, env, endpoints, workspace, TTY (exit 1 on a failure).
  schema      Print the config JSON Schema for editors and tooling.
  init        Write an annotated starter config (an existing file is kept).
  mcp         Log in to, inspect or log out of an MCP server's OAuth credential.
  version     Print this binary's version, commit and build date.
  completion  Print a shell completion script for bash, zsh or fish.
  help        Print this text, or the detailed page for one command.

Synopsis:
  amele run <config.yaml|dir> [--model MODEL] [--set key=value] [-w DIR] [-q|-v] [task...]
  amele chat <config.yaml|dir> [--model MODEL] [--set key=value] [-w DIR] [-q|-v]
  amele validate <config.yaml|dir> [--set key=value] [-w DIR]
  amele explain <config.yaml|dir> [--set key=value] [-w DIR]
  amele doctor <config.yaml|dir> [--set key=value] [-w DIR]
  amele schema
  amele init [path]
  amele mcp login|status|logout <config.yaml|dir> [server]
  amele version
  amele completion bash|zsh|fish
  amele help [command]

Run 'amele help <command>' for details - or 'amele <command> --help'.

Exit codes:
  0 success · 1 task failed · 2 config error · 3 budget exceeded
  4 permission denied · 5 provider error · 6 output schema unmet
  7 run lock held (another run of this config is in progress)
  8 required MCP server unavailable
`

// The one-line usage strings printed when an invocation is malformed.
//
// CONTRACT: each is exactly its command's help-page SYNOPSIS line with a
// "usage: " prefix, and a test pins that equality - the spelling an operator
// is shown at the moment they got it wrong must be the one the manual
// documents. They drifted once already (still advertising just
// [--model MODEL] after -q/-v shipped), which is why they are consts here
// rather than literals at the call sites.
const (
	usageRun        = "usage: amele run <config.yaml|dir> [--model MODEL] [--set key=value] [-w DIR] [--resume PATH] [-q|-v] [task...]"
	usageChat       = "usage: amele chat <config.yaml|dir> [--model MODEL] [--set key=value] [-w DIR] [-q|-v]"
	usageValidate   = "usage: amele validate <config.yaml|dir> [--set key=value] [-w DIR]"
	usageExplain    = "usage: amele explain <config.yaml|dir> [--set key=value] [-w DIR]"
	usageDoctor     = "usage: amele doctor <config.yaml|dir> [--set key=value] [-w DIR]"
	usageCompletion = "usage: amele completion bash|zsh|fish"
	usageMCP        = "usage: amele mcp login|status|logout <config.yaml|dir> [server]"
)

// The detailed help pages. One raw-string const per command, all built to the
// same man-page skeleton (SYNOPSIS, DESCRIPTION, FLAGS, STDIN, STDOUT, STDERR,
// EXIT CODES, EXAMPLES) so a reader who has learned one page can navigate the
// rest by muscle memory.
//
// CONTRACT: these pages must agree with docs/contracts/cli.md - the contract
// is the source, the page is its rendering for someone who has a terminal and
// no browser. Wording is copied from it deliberately rather than paraphrased,
// so a contract change that never reaches the help output is visible as a
// diff. Examples are runnable as written.
//
// Plain consts rather than a template engine: the pages are static text, and a
// static binary should not carry a rendering layer to print eight strings.
// Backticks never appear in the text so every page can stay a raw string.
const helpRun = `amele run - one-shot agent run

SYNOPSIS
  amele run <config.yaml|dir> [--model MODEL] [--set key=value] [-w DIR] [--resume PATH] [-q|-v] [task...]

DESCRIPTION
  Loads the config, runs the agent on the task, prints the final answer and
  exits per the exit-code contract. This is the headless mode: it needs no
  terminal, so it composes in cron jobs, CI steps and shell pipes.

  The config path comes first, then flags, then free-form task text. Flag
  parsing stops at the first non-flag argument, so anything after the task
  text - including --model - is task text.

  A directory argument is shorthand for <dir>/agent.yaml inside it, and a
  bare name that names nothing on disk is looked up as a saved agent:
  $XDG_CONFIG_HOME/amele/<name>.yaml or <name>/agent.yaml there (~/.config
  when the variable is unset). The run lock is derived from the resolved
  file, so every spelling of one config contends on the same lock.

  The message the model sees is built from the task text and, when needed,
  piped stdin. A config with a prompt template controls that composition
  itself: {{args}} is the task text, {{input}} is stdin. Without a template
  the message is the task text when there is any, and the piped input when
  there is none - stdin is never appended to task text. Combining an
  instruction with piped data is what the prompt template is for.

  Tools, permission profile, budgets and session logging all come from the
  YAML. Run "amele explain <config.yaml|dir>" to see exactly what a config
  grants before letting it loose.

FLAGS
  --model MODEL   Shortcut for --set model=MODEL. The override participates in
                  validation, so a config with no model plus --model X is
                  valid. An empty value (--model "" from an unset shell
                  variable) means "no override". Default: the config's model.
  --set KEY=VALUE Override one config field for this run, before validation.
                  Repeatable. Split on the FIRST "=", so the value may contain
                  more. Settable keys, and nothing else:
                    model, prompt, system_prompt_file, workspace, session_dir,
                    limits.max_turns, limits.max_tokens, limits.timeout,
                    limits.max_logged_field, limits.max_tool_result_bytes,
                    output.max_schema_retries,
                    provider.max_output_tokens, provider.reasoning.effort,
                    provider.temperature, provider.top_p
                  Tools, permissions, the provider's identity (type, base_url,
                  api_key), the run lock and what the session log RECORDS or
                  ANNOUNCES (log_reasoning, print_session_path) are deliberately
                  NOT settable: the YAML file stays the audited grant of
                  authority, so what "amele explain agent.yaml" reports cannot
                  be widened - or, in the lock's case, weakened - by a flag on
                  the cron line (docs/threat-model.md §2). The provider tuning
                  knobs above are settable because they only change what a run
                  spends, not what it may do.
                  workspace, session_dir and system_prompt_file resolve against
                  the CURRENT DIRECTORY, not the config's - a path typed in a
                  shell means what it means in that shell. system_prompt_file
                  is re-read and replaces whatever prompt the config carried.
                  An empty session_dir (--set session_dir=) turns session
                  logging off, an empty limits.max_logged_field drops back to
                  the default clip, and an empty limits.max_tool_result_bytes
                  drops back to the built-in per-tool result caps.
                  Default: nothing overridden.
  -w, --workspace DIR
                  Shortcut for --set workspace=DIR. Default: the config's
                  workspace (its own directory unless the YAML says otherwise).
  --resume PATH   Continue the run recorded in the session log at PATH: its
                  conversation is rebuilt and sent again before this run's
                  first turn, so the model picks up where it stopped. The task
                  comes from the log; task text given alongside is a follow-up
                  INSTRUCTION, appended as the last user message VERBATIM -
                  the config's prompt template is not applied to it, and stdin
                  is never read on this path.
                  The log must be a complete record of what the model saw:
                  write it with limits.max_logged_field: 0, or the resume is
                  refused (exit 2) rather than continued from text the log
                  shortened. A log of an interactive chat, a pre-v1.10 log
                  of an output.schema retry (its feedback turn was not
                  logged), or a log of a different schema version is refused
                  the same way.
                  No tool call is ever re-executed. A call the interrupted run
                  dispatched but never logged a result for gets a message in
                  its place telling the model the result is unknown, and the
                  new run's run_start records the origin (resumed_from,
                  resumed_turn, resumed_pending) so an operator can see which
                  side effects are unaccounted for. Turn numbering starts at 1
                  again; the log named by PATH is never appended to. A log
                  that was itself written by a resumed run names the log it
                  continued, and that chain is followed: naming the NEWEST
                  log rebuilds every earlier run's turns and instructions.
                  A log whose run already produced a final answer has nothing
                  to continue on its own: resuming it without an instruction
                  is exit 2. Default: off - the run starts from the task text.
  -q, --quiet     Drop the summary line and the non-error notes, so a run that
                  works says nothing at all. Errors, permission questions and
                  permission decisions still print, and the session log is
                  unchanged. Default: off.
  -v, --verbose   Print one progress line per loop event to stderr (see
                  STDERR). Default: off.
  -h, --help      Print this page to stdout and exit 0.

  -q and -v ask for opposite things: giving both is a usage error (exit 2).

  --model, -w and --set append to ONE ordered override list, in the order they
  are written, and the last entry for a key wins - there is no precedence
  between the spellings, so the effective value is always the one further
  right: "--model a --set model=b" sends b, "--set model=b --model a" sends a.

  Flags go after the config path and before the task text; a flag written
  after the task text is part of the task.

STDIN
  Read only when it is actually needed: the config's prompt template
  references {{input}}, or there is no prompt and no task text. A --resume run
  never reads it at all: its message history comes from the log.
  amele run cfg.yaml "task" never touches stdin, so it cannot hang on an open
  pipe. When stdin is an interactive terminal, nothing is read - a run never
  blocks waiting for typing. Piped input is capped at 10 MB; the cut is marked
  with [input truncated at 10MB by amele] so the model knows data is missing.
  A run whose final user message would be empty or whitespace-only (nothing
  piped and no task text, or a prompt template whose placeholders all rendered
  empty) is refused with exit 2 before the provider is contacted.

STDOUT
  On success, exactly the agent's final answer followed by one newline -
  nothing else, so runs compose in pipes. With output.schema set, stdout
  carries the canonical JSON the validator accepted (the model's fencing and
  prose framing are stripped). On any failure - including exit 6 - stdout gets
  nothing.

STDERR
  Config and run errors, permission questions
  (amele: allow tool X with {...}? [y/N]) and audit notes, and - unless -q -
  the one-line summary: ✓ 8 turns, 3 tool calls, 41.0k tokens, 34.2s (with a
  (N cached) suffix on the token figure when any turn was served from the
  provider's prompt cache; ✗ on failure).

  With -v, one line per loop event, as it happens:
    amele: turn 3: model requested fs_read {"path":"app.log"}
    amele: turn 3: fs_read ok (1.2s)
    amele: turn 3: shell exit 3 (0.4s)
    amele: turn 3: fs_read error: <message>
    amele: turn 4: final answer (312 tokens)
  A tool call that ran but did not work is named as such instead of ok:
  exit N, timed out, aborted (the run ended under it) or rejected (the shell
  policy refused the command).
  The token count is what the model produced in that turn. Tool names and
  arguments come from the model, so they are stripped of control characters
  and clipped before they reach the terminal, and every ${VAR} value the
  config interpolated (the API key included) is replaced with [REDACTED],
  exactly as in the session log.

RUN LOCK
  With lock: true in the config, run takes a non-blocking advisory lock on
  <absolute config path>.lock (created 0600) before reading stdin or
  contacting the provider, and releases it when the run ends - normally or
  not. A run that finds the lock held prints "another run holds the lock for
  this config (lock file: ...)" to stderr and exits 7, having spent nothing
  and written nothing. The lock file is never deleted. The default is off, so
  the same config can still be run concurrently with different tasks. Only
  run locks. The switch lives in the YAML alone - there is no --set for it, so
  an invocation cannot disarm the guard a reviewed config armed.

EXIT CODES
  0  the agent finished; its final answer is on stdout
  1  the task failed, or the run was interrupted (SIGINT/SIGTERM)
  2  usage or config error - reported before a single token is spent
  3  limits.max_turns, limits.max_tokens or limits.timeout stopped the run
  4  a permission denial aborted the run
  5  provider or network failure after the client's retries were exhausted
  6  output.schema was never satisfied within max_schema_retries
  7  lock: true is set and another run of this config holds the lock
  8  a required MCP server (mcp.servers[].required: true) could not be started
     or reached

EXAMPLES
  Run a task written on the command line:
    amele run agent.yaml "summarize today's incidents"

  Pipe a log file in as the whole task - no task text, so stdin is read:
    amele run agent.yaml < app.log

  Instruction plus piped data: the config carries a prompt template placing
  {{args}} and {{input}}, since without one only the task text is sent:
    amele run triage.yaml "only the ERROR lines" < app.log

  Score a diff with an output.schema config and read one field:
    amele run judge.yaml --model gpt-4o-mini < diff.txt | jq .score

  Run one config against another directory and a tighter budget, without
  editing the file (and check first what that invocation would do):
    amele explain agent.yaml -w /srv/logs --set limits.max_turns=5
    amele run agent.yaml -w /srv/logs --set limits.max_turns=5 "sweep"

  Treat an overlapping locked run (exit 7) as success in a cron wrapper:
    amele run sentry.yaml "hourly sweep" || [ $? -eq 7 ]

  A cron job that mails only when something goes wrong:
    amele run sentry.yaml -q "hourly sweep"

  Watch what the agent is doing while it does it:
    amele run agent.yaml -v "triage the failing tests"

  Continue a run the provider killed halfway, then push it further:
    amele run agent.yaml --resume sessions/20260907T101500Z-4f2a.jsonl
    amele run agent.yaml --resume sessions/20260907T101500Z-4f2a.jsonl "now open a ticket"
`

const helpChat = `amele chat - interactive conversation with the agent

SYNOPSIS
  amele chat <config.yaml|dir> [--model MODEL] [--set key=value] [-w DIR] [-q|-v]

DESCRIPTION
  An interactive REPL over the same config, tools and permissions as
  amele run. One user message per line; the model answers, and the exchange
  is carried into the next line so the conversation keeps its context.

  Free-form arguments are rejected (exit 2) with a hint to use run - a chat
  reads its input from stdin.

  A directory argument is shorthand for <dir>/agent.yaml inside it, and a
  bare name that names nothing on disk is looked up as a saved agent under
  $XDG_CONFIG_HOME/amele (~/.config/amele when unset).

  The whole session is one entry in the session log: one run_start (recorded
  with the task "interactive chat") and one run_end with the session totals.

FLAGS
  --model MODEL   Shortcut for --set model=MODEL, for this session. Default:
                  the model set in the config.
  --set KEY=VALUE Override one config field before validation, exactly as in
                  run - same closed key list, same command-line path
                  resolution, same "last one wins" rule. Repeatable. Run
                  "amele help run" for the key list. Default: nothing
                  overridden.
  -w, --workspace DIR
                  Shortcut for --set workspace=DIR. Default: the config's.
  -q, --quiet     Drop the closing summary and the non-error notes (for
                  example the output.schema note). The prompt, the errors and
                  the permission questions stay - they are the conversation.
                  Default: off.
  -v, --verbose   Print one progress line per loop event to stderr; the turn
                  numbers continue across the whole session. Default: off.
  -h, --help      Print this page to stdout and exit 0.

  -q and -v ask for opposite things: giving both is a usage error (exit 2).

STDIN
  One user message per line; a line is capped at 1 MB (the excess is
  discarded, never re-served as the next line). Empty lines cost nothing - no
  provider call, no turn, no tokens. EOF (Ctrl-D, or the end of a piped
  script) ends the session with exit 0. The REPL and the ask-policy approval
  prompt share one reader, so answering a question never eats the next chat
  line.

STDOUT
  The model's answers only, each followed by a newline. It is a stream, not a
  record format: answers routinely span several lines and there is no
  delimiter. A scripted consumer that needs a parseable boundary should use
  amele run (one answer per process).

STDERR
  The "> " prompt, approval questions, notes (for example that output.schema
  is ignored in chat), errors, the -v progress lines, and the cumulative
  session summary at the end.

BUDGETS
  limits.max_turns and limits.max_tokens form one pool for the whole session;
  an exhausted pool is exit 3. limits.timeout bounds a single exchange, not
  the session, so thinking time at the prompt costs nothing.

  output.schema is not enforced in chat - it constrains a one-shot answer. A
  note is printed to stderr and the conversation proceeds; use amele run to
  enforce it.

EXIT CODES
  0  the session ended at EOF (Ctrl-D, or the end of a piped script)
  1  interrupted at the prompt (SIGINT), or stdin broke
  2  usage or config error, including task arguments (use run for those)
  3  the session budget pool is spent, or one exchange hit limits.timeout
  4  a permission denial aborted the session
  5  provider or network failure after the client's retries were exhausted
  8  a required MCP server (mcp.servers[].required: true) could not be started
     or reached

EXAMPLES
  Talk to the agent a config defines:
    amele chat agent.yaml

  Try the same agent against a different model:
    amele chat agent.yaml --model gpt-4o

  Talk to the same agent about another directory:
    amele chat agent.yaml -w /srv/logs

  Replay a scripted conversation - stdin is not a terminal, so every ask
  permission auto-denies:
    amele chat agent.yaml < script.txt
`

const helpValidate = `amele validate - check a config without spending anything

SYNOPSIS
  amele validate <config.yaml|dir> [--set key=value] [-w DIR]

DESCRIPTION
  Loads and validates the config and, when one is present, compiles
  output.schema - so a config that validates cannot fail configuration under
  run. Every ${ENV_VAR} reference is interpolated exactly as a run would
  interpolate it, so an unset variable is reported here instead of surfacing
  much later as a confusing provider error.

  A directory argument is shorthand for <dir>/agent.yaml inside it, and a
  bare name that names nothing on disk is looked up as a saved agent under
  $XDG_CONFIG_HOME/amele (~/.config/amele when unset).

  Violations are collected and reported together: one invocation names
  everything that is wrong with the file, not just the first problem.

  No network, no tokens, no session file - this is the command for a
  pre-commit hook or a CI step.

FLAGS
  --set KEY=VALUE Apply a config override before validating, exactly as run
                  would apply it: same closed key list, same command-line path
                  resolution, same "last one wins" rule. Repeatable. This is
                  how a parametrized invocation gets checked as the invocation
                  it will be - validating the bare file would answer a
                  question nobody asked. Run "amele help run" for the key list.
  -w, --workspace DIR
                  Shortcut for --set workspace=DIR.
  -h, --help      Print this page to stdout and exit 0. It is honored only as
                  the SOLE argument, so a wrong argument count is never
                  answered with a page and an exit 0.

  validate takes exactly one positional argument (the config path); flags
  follow it, as in run. --model is not accepted here: use --set model=MODEL.

STDIN
  Not read.

STDOUT
  <config.yaml>: OK on success, nothing otherwise.

STDERR
  Every violation, reported together.

EXIT CODES
  0  the config is valid
  2  usage error, or the config failed to load, validate or compile

EXAMPLES
  Check a config before installing it in cron:
    amele validate agent.yaml

  Gate a commit in CI:
    amele validate agent.yaml && echo config ok

  Check every agent config in a directory:
    for f in agents/*.yaml; do amele validate "$f" || exit 1; done

  Check the config as the parametrized cron line will actually run it:
    amele validate agent.yaml --set model=gpt-4o-mini -w /srv/logs
`

const helpExplain = `amele explain - dry-run report on what a config would do

SYNOPSIS
  amele explain <config.yaml|dir> [--set key=value] [-w DIR]

DESCRIPTION
  Prints a report of what the agent may touch, spend and emit, plus warnings
  for valid-but-suspicious settings. It performs everything a run would do up
  to, but not including, provider construction: load, validate, compile
  output.schema, build the real tool registry.

  A directory argument is shorthand for <dir>/agent.yaml inside it, and a
  bare name that names nothing on disk is looked up as a saved agent under
  $XDG_CONFIG_HOME/amele (~/.config/amele when unset).

  explain reports; run gates. A config that cannot run yet - unset ${VAR}s, a
  workspace that does not exist, a schema that will not compile - is still
  described in full: the reasons open the report in a PROBLEMS block and the
  command exits 0. Exit 2 is what survives from loading the file, when there
  is no config to describe at all; see EXIT CODES. "amele run" and
  "amele validate" are unchanged: they refuse a config with any problem.

  The report has one section per grant - PROBLEMS (only when there are any),
  MODEL & PROVIDER, TOOLS (workspace, fs builtins, shell, subprocess tools),
  requirements (${VAR}s and subprocess executables the host must provide, each
  marked set/found or MISSING, plus each tool's env allowlist), PERMISSIONS
  (default policy and per-tool overrides), BUDGETS, CONCURRENCY, OUTPUT,
  SESSION - and closes with WARNINGS.

  Because the tool registry is built for real, the report names the tools a
  run would actually hold. This is the review step for a config someone else
  wrote, and the thing to read before granting an agent a shell.

  Interpolated ${VAR} values are shown so a parametrized pack can be
  pre-flighted (which model will this cron line buy?) EXCEPT credentials:
  whatever feeds provider.api_key, and any variable whose name contains key,
  token, secret, passw or cred, prints as [REDACTED]. Name credential
  variables accordingly.

  No network, no tokens, no session file.

FLAGS
  --set KEY=VALUE Apply a config override before reporting, exactly as run
                  would apply it. Repeatable. The report then describes the
                  parametrized run: an OVERRIDES block echoes what the command
                  line contributed, and every line whose value came from there
                  is marked "(overridden via --set)" - so no command-line value
                  is ever mistaken for something the YAML says. Run
                  "amele help run" for the key list.
  -w, --workspace DIR
                  Shortcut for --set workspace=DIR.
  -h, --help      Print this page to stdout and exit 0. It is honored only as
                  the SOLE argument.

  explain takes exactly one positional argument (the config path); flags follow
  it, as in run. --model is not accepted here: use --set model=MODEL.

STDIN
  Not read.

STDOUT
  The report, including any PROBLEMS.

STDERR
  Errors only - empty whenever the report was printed.

EXIT CODES
  0  the report was printed, whether or not the config could actually run
  2  usage error (including a malformed --set), or the loader rejected the
     file: unreadable, unparseable YAML, an unknown key or a wrong type, a
     literal provider.api_key, or an unusable system_prompt_file

EXAMPLES
  Review a config before trusting it:
    amele explain agent.yaml

  Pre-flight a pack on a fresh host - what must I set up?
    amele explain ./log-sentry | sed -n '/^PROBLEMS/,/^$/p'

  Read the warnings alone:
    amele explain agent.yaml | sed -n '/^WARNINGS/,$p'

  Diff what two agents are allowed to do:
    diff <(amele explain a.yaml) <(amele explain b.yaml)

  See exactly what a parametrized run would do before running it:
    amele explain agent.yaml -w /srv/logs --set limits.max_turns=5
`

const helpDoctor = `amele doctor - pre-flight checks: will this config run on this host?

SYNOPSIS
  amele doctor <config.yaml|dir> [--set key=value] [-w DIR]

DESCRIPTION
  Runs a fixed list of checks against a config and prints one line per
  check with a PASS, WARN or FAIL verdict:

    config       the file loads and validates (every violation is a FAIL)
    env          every ${VAR} the config references is set
    provider     the primary endpoint answers and accepts the key - and the
                 same for every provider.fallback entry (one line each)
    workspace    the directory exists and takes a write
    session_dir  the directory can be created and written (when set)
    tool <name>  a subprocess tool's command[0] is on PATH (one line each)
    mcp <name>   a stdio server's command is on PATH; an http server's URL
                 names a host (it is not dialled - amele explain connects)
    tty          whether a terminal is attached, related to the permission
                 profile: an ask policy auto-denies without one
    lock         the lock file's directory takes a write (lock: true)

  The provider check sends one GET to the wire's models listing with the
  configured credential: 2xx is a PASS, 401/403 is a FAIL (the key was
  refused), 404/405 is a WARN (the endpoint exposes no listing, so the key
  could not be checked - gateways and self-hosted servers), anything else or
  no answer is a FAIL. Vertex AI targets are not probed (their credential
  flow is the run's own) and print a WARN saying so. No token is spent.

  Unlike explain, doctor GATES: any FAIL exits 1, so a cron line can run it
  before the real run and a deploy script can branch on it. WARN never
  changes the exit code.

  A directory argument is shorthand for <dir>/agent.yaml inside it, and a
  bare name that names nothing on disk is looked up as a saved agent under
  $XDG_CONFIG_HOME/amele (~/.config/amele when unset).

FLAGS
  --set KEY=VALUE Apply a config override before checking, exactly as run
                  would apply it. Repeatable. Run "amele help run" for the
                  key list.
  -w, --workspace DIR
                  Shortcut for --set workspace=DIR.
  -h, --help      Print this page to stdout and exit 0. It is honored only as
                  the SOLE argument.

  doctor takes exactly one positional argument (the config path); flags
  follow it. --model is not accepted here: use --set model=MODEL.

STDIN
  Not read; only its terminal state is inspected (the tty check).

STDOUT
  The report: one line per check, then a closing count.

STDERR
  Errors only - empty whenever the report was printed.

EXIT CODES
  0  every check passed (warnings allowed)
  1  at least one check failed
  2  usage error (including a malformed --set), or the loader rejected the
     file: unreadable, unparseable YAML, an unknown key or a wrong type, a
     literal provider.api_key, or an unusable system_prompt_file

EXAMPLES
  Check a config before its first run:
    amele doctor agent.yaml

  Guard a cron line with it:
    amele doctor agent.yaml >/dev/null && amele run agent.yaml -q "scan the logs"

  Pre-flight a saved agent under a different workspace:
    amele doctor sentry -w /srv/logs
`

const helpSchema = `amele schema - print the config JSON Schema

SYNOPSIS
  amele schema

DESCRIPTION
  Prints the JSON Schema for the config file that this binary embeds - the
  same document as docs/contracts/config.schema.json. Editors consume it for
  autocomplete and inline validation, and it is the machine-readable half of
  the config contract.

  The schema travels inside the binary, so a machine that has amele needs no
  source checkout to get it.

FLAGS
  -h, --help   Print this page to stdout and exit 0.

  schema takes no arguments and no other flags; any argument is a usage error
  (exit 2), so a misremembered "amele schema config.yaml" fails loudly instead
  of silently ignoring the file.

STDIN
  Not read.

STDOUT
  Exactly the schema document plus a trailing newline - a valid JSON file
  as-is.

STDERR
  The usage error, if any.

EXIT CODES
  0  the schema was printed
  2  usage error (any argument)

EXAMPLES
  Save it next to your configs for editor autocomplete:
    amele schema > config.schema.json

  Look up what one section accepts:
    amele schema | jq .properties.limits

  Validate a config with an external JSON Schema tool:
    amele schema > s.json && check-jsonschema --schemafile s.json agent.yaml
`

const helpInit = `amele init - write an annotated starter config

SYNOPSIS
  amele init [path]

DESCRIPTION
  Writes a commented starter config to path (default agent.yaml). The
  generated file passes amele validate exactly as written once AMELE_API_KEY
  is set, so the shortest road from nothing to a working agent is init,
  export, validate, run.

  What it enables is deliberately conservative: sandboxed fs tools, every
  budget armed, session logging on. Everything riskier or optional - the
  native Anthropic provider, the shell tool, permission profiles,
  output.schema - ships as accurate commented examples, because a scaffold
  should show the doors without opening them.

  An existing file is never overwritten. init creates a starting point, and a
  tool that can destroy the config you have been editing is worse than no
  tool.

FLAGS
  -h, --help   Print this page to stdout and exit 0.

  init takes at most one argument (the path) and no other flags.

STDIN
  Not read.

STDOUT
  Nothing - init composes in scripts like every other command.

STDERR
  The next-step hint (amele: wrote agent.yaml - next: set AMELE_API_KEY and
  run: amele validate agent.yaml), or the error.

EXIT CODES
  0  the file was written
  2  usage error, the target already exists, or the write failed

EXAMPLES
  The five-minute start:
    amele init agent.yaml
    export AMELE_API_KEY=sk-...
    amele validate agent.yaml
    amele run agent.yaml "summarize the files in this directory"

  Scaffold a second agent under its own name:
    amele init log-sentry.yaml
`

const helpVersion = `amele version - print this binary's build identity

SYNOPSIS
  amele version
  amele --version
  amele -V

DESCRIPTION
  Prints one line naming the version, the commit it was built from, the build
  date, the Go toolchain that compiled it and the target platform. The three
  spellings above are the same command.

  A source checkout (go build or go run without the release ldflags) reports
  amele dev (commit unknown, built unknown, ...); a released binary carries
  the real version, commit and build date baked in by the Makefile. This is
  the line to paste into a bug report.

FLAGS
  -h, --help   Print this page to stdout and exit 0.

  version takes no arguments and no other flags; any argument is a usage error
  (exit 2).

STDIN
  Not read.

STDOUT
  Exactly one line, followed by a single newline:
    amele <version> (commit <commit>, built <date>, <go version>, <os>/<arch>)

STDERR
  The usage error, if any.

EXIT CODES
  0  the version line was printed
  2  usage error (any argument)

EXAMPLES
  Check what is installed:
    amele version

  Record the exact binary in a build log:
    amele --version >> build-provenance.txt

  Extract the version field alone:
    amele version | cut -d' ' -f2
`

const helpCompletion = `amele completion - print a shell completion script

SYNOPSIS
  amele completion bash|zsh|fish

DESCRIPTION
  Prints a static completion script for the named shell to stdout. The
  scripts are hand-written against each shell's own completion builtins
  (bash's compgen/complete, zsh's compsys, fish's complete) - no generator,
  no shared framework, so the static binary carries no rendering layer for
  them.

  Each script completes the subcommands (run, chat, validate, explain,
  schema, init, version, completion, help), the flags each subcommand
  accepts, YAML files in the config-path slot, and the shell names accepted
  by completion itself.

FLAGS
  -h, --help   Print this page to stdout and exit 0.

  completion takes exactly one argument, the shell name, and no other flags;
  any other argument count is a usage error (exit 2).

STDIN
  Not read.

STDOUT
  The completion script for the named shell, newline-terminated. Nothing
  else - the output is meant to be redirected straight into the shell's
  completion directory or sourced from a startup file.

STDERR
  The usage error, if any: no shell name, an unrecognized one, or extra
  arguments.

EXIT CODES
  0  the script was printed
  2  usage error: missing or unrecognized shell name, or extra arguments

EXAMPLES
  Install for bash (system-wide, if writable):
    amele completion bash > /etc/bash_completion.d/amele

  Install for zsh (a directory already on fpath):
    amele completion zsh > "${fpath[1]}/_amele"

  Install for fish:
    amele completion fish > ~/.config/fish/completions/amele.fish

  Try it in the current shell without installing anything:
    source <(amele completion bash)
`

const helpMCP = `amele mcp - log in to, inspect and log out of MCP OAuth credentials

SYNOPSIS
  amele mcp login <config.yaml|dir> [server]
  amele mcp status <config.yaml|dir>
  amele mcp logout <config.yaml|dir> [server]

DESCRIPTION
  These commands manage the OAuth credentials of the MCP servers a config
  declares with an auth block. They act on the config as written: there are no
  --set overrides, because no mcp.* key is overridable.

  A credential is stored per authorization server, per resource and per client
  id, under ${XDG_STATE_HOME:-$HOME/.local/state}/amele/mcp - one 0600 file per
  credential in a 0700 directory. The file format is subject to change until
  v0.3.

  login runs the browser flow, one server at a time in config order, and needs
  a real terminal on stdin: it asks before it opens anything, and a run with a
  pipe or /dev/null on stdin is refused rather than left waiting. With no
  server named, every server that declares oauth is logged into. All of its
  output - the question, the URL, the result - goes to stderr.

  status is a report and never changes anything: it does not refresh, open a
  browser or contact any server. It prints one row per stored credential, with
  the expiry, whether a refresh token is present, the granted scopes and the
  issuer. A token value is never printed. Its exit code is 0 even when nothing
  is stored - "no token" is an answer, not a failure.

  logout deletes the credential locally and, when the authorization server
  advertised an RFC 7009 revocation endpoint at login, asks it to invalidate
  the token first. The revocation is best effort: if it fails, the local delete
  still happens and the line says "local only".

FLAGS
  -h, --help   Print this page to stdout and exit 0.

  The mcp commands take no other flags.

STDIN
  login reads the y/N answers to its confirmation questions and must be a
  terminal. status and logout do not read stdin.

STDOUT
  status writes the credential table. login and logout write nothing to
  stdout, so they are safe to run with stdout redirected.

STDERR
  login writes the confirmation question, the authorization URL and one
  "mcp login ok: <server> (expires <ts>)" line per server. logout writes one
  "mcp logout: <server> (revoked|local only|no token)" line per server, plus a
  warning when a revocation failed.

EXIT CODES
  0  the command completed (status: always, when the config loaded)
  1  a login did not complete (declined, or the flow failed)
  2  usage error, config error, an unknown server name, a server without an
     auth block, or a login without an interactive terminal

EXAMPLES
  Log in to every OAuth server in a config:
    amele mcp login agent.yaml

  Log in to one of them:
    amele mcp login agent.yaml github

  See what is stored, without touching anything:
    amele mcp status agent.yaml

  Hand the token back and forget it:
    amele mcp logout agent.yaml github
`

const helpHelp = `amele help - the command list, or a detailed page per command

SYNOPSIS
  amele help [command]
  amele -h | --help
  amele <command> -h | --help

DESCRIPTION
  With no argument, prints the short usage: every command with a one-line
  description, the synopsis block and the exit-code table. With a command
  name, prints that command's detailed page - the same page
  amele <command> --help prints.

  Commands with a page: run, chat, validate, explain, doctor, schema, init,
  version, completion, mcp, help. The alternate spellings resolve too, so
  amele help --version reaches the version page.

  For run and chat the help flag is recognized only where a flag is
  recognized. Flag parsing stops at the first non-flag argument, so a -h that
  appears after the task text is part of the task, not a help request.

  For the commands with a fixed argument count - validate, explain, doctor,
  schema, init, version, completion - the flag is recognized only as the sole
  argument. A -h next to other arguments leaves the invocation a usage error
  (exit 2), so a wrong argument count is never answered with a page and an
  exit 0.

FLAGS
  -h, --help   Print this page to stdout and exit 0.

  help takes at most one command name.

STDIN
  Not read.

STDOUT
  The short usage, or the requested command page.

STDERR
  For an unknown command name: the error plus the short usage. For more than
  one argument: the usage line.

EXIT CODES
  0  a help text was printed
  2  unknown command name, or more than one argument

EXAMPLES
  The command list and the exit codes:
    amele help

  Everything about the one-shot runner:
    amele help run

  The same page, reached from the command itself:
    amele run --help
`

// helpPages maps a command name to its detailed page. A map (not a switch) so
// the set of documented commands is one greppable list that a test can iterate
// over: a command added to the dispatch switch without a page fails that test
// instead of shipping undocumented.
var helpPages = map[string]string{
	"run":        helpRun,
	"chat":       helpChat,
	"validate":   helpValidate,
	"explain":    helpExplain,
	"doctor":     helpDoctor,
	"schema":     helpSchema,
	"init":       helpInit,
	"mcp":        helpMCP,
	"version":    helpVersion,
	"completion": helpCompletion,
	"help":       helpHelp,
}

// canonicalCommand resolves a command's alternate spellings to the name
// helpPages is keyed by, so `amele help --version` finds the version page
// instead of reporting an unknown command. A user who learned a command as
// `--version` must be able to ask for help using the name they know.
func canonicalCommand(name string) string {
	switch name {
	case "--version", "-V":
		return "version"
	case "-h", "--help":
		return "help"
	default:
		return name
	}
}

// printHelp writes the detailed page for cmd to stdout and returns ExitOK, or
// - for a name no command answers to - writes the error plus the short usage
// to stderr and returns ExitConfigError.
//
// CONTRACT: help requested successfully is stdout + exit 0, whichever spelling
// asked for it (`amele help run`, `amele run -h`, `amele run --help`); help
// requested for something that is not a command is a usage error like every
// other (exit 2), and the short usage rides along so a mistyped name still
// teaches the reader the real ones.
func printHelp(cmd string, stdout, stderr io.Writer) int {
	page, ok := helpPages[canonicalCommand(cmd)]
	if !ok {
		_, _ = fmt.Fprintf(stderr, "unknown command %q\n\n%s", cmd, usageText)
		return ExitConfigError
	}
	_, _ = fmt.Fprint(stdout, page)
	return ExitOK
}

// hasHelpFlag reports whether args is exactly one bare -h or --help, i.e. the
// command was invoked to ask for its page and nothing else.
//
// It is used ONLY by the fixed-arity commands that take no flags at all
// (schema, init, version). run and chat must NOT use it: their tail is
// free-form task text where a literal -h has to survive, so they let the flag
// package draw the boundary (see parseAgentArgs). validate and explain take
// flags, so they draw the same boundary in parseInspectArgs - which keeps this
// rule: the flag is honored only as the SOLE argument.
//
// CONTRACT: the flag is honored only as the SOLE argument, never scanned out
// of an arbitrary position. docs/contracts/cli.md freezes "every usage error
// (wrong argument count, bad flag) is exit 2"; a positional scan made
// `amele validate a.yaml b.yaml -h` exit 0, converting a wrong argument count
// into success and hiding the mistake from any script that checks $?. The
// arity check downstream then reports the real error, as it does for every
// other malformed invocation.
func hasHelpFlag(args []string) bool {
	return len(args) == 1 && (args[0] == "-h" || args[0] == "--help")
}

func main() {
	// Ctrl-C / SIGTERM cancel the run context so tools and HTTP calls stop
	// promptly; the deferred cleanup in run() still writes run_end.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	os.Exit(run(ctx, os.Args[1:], os.Stdin, os.Stdout, os.Stderr, os.LookupEnv))
}

// run is the testable entry point: all process I/O and the environment are
// injected, and the return value is the process exit code.
func run(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer, env config.LookupEnv) int {
	if len(args) < 1 {
		_, _ = fmt.Fprint(stderr, usageText)
		return ExitConfigError
	}

	switch args[0] {
	case "run":
		return cmdRun(ctx, args[1:], stdin, stdout, stderr, env)
	case "chat":
		return cmdChat(ctx, args[1:], stdin, stdout, stderr, env)
	case "validate":
		return cmdValidate(args[1:], stdout, stderr, env)
	case "explain":
		return cmdExplain(ctx, args[1:], stdout, stderr, env)
	case "doctor":
		return cmdDoctor(ctx, args[1:], stdin, stdout, stderr, env)
	case "schema":
		return cmdSchema(args[1:], stdout, stderr)
	case "init":
		return cmdInit(args[1:], stdout, stderr)
	case "mcp":
		return cmdMCP(ctx, args[1:], stdin, stdout, stderr, env)
	case "version", "--version", "-V":
		return cmdVersion(args[1:], stdout, stderr)
	case "completion":
		return cmdCompletion(args[1:], stdout, stderr)
	case "help", "-h", "--help":
		return cmdHelp(args[1:], stdout, stderr)
	default:
		_, _ = fmt.Fprintf(stderr, "unknown command %q\n\n%s", args[0], usageText)
		return ExitConfigError
	}
}

// cmdHelp implements `amele help [command]` (and the -h/--help spellings that
// dispatch to it): the short usage with no argument, one detailed page with a
// command name.
//
// CONTRACT: bare `amele help` keeps its frozen behavior - the short usage on
// stdout, exit 0. Naming a command is the additive part. A second argument is
// a usage error rather than being ignored, the same stance `amele schema
// anything` takes: a caller who wrote something amele cannot act on should
// hear about it.
func cmdHelp(args []string, stdout, stderr io.Writer) int {
	switch len(args) {
	case 0:
		_, _ = fmt.Fprint(stdout, usageText)
		return ExitOK
	case 1:
		return printHelp(args[0], stdout, stderr)
	default:
		_, _ = fmt.Fprintln(stderr, "usage: amele help [command]")
		return ExitConfigError
	}
}

// cmdValidate loads and validates a config, printing either OK or every
// violation at once. --set overrides are applied exactly as `run` applies
// them, so the command answers "will THIS invocation work?" rather than a
// question about a file nobody runs bare.
func cmdValidate(args []string, stdout, stderr io.Writer, env config.LookupEnv) int {
	parsed, ok := parseInspectArgs(context.Background(), env, "validate", usageValidate, args, stderr)
	if !ok {
		return ExitConfigError
	}
	if parsed.help {
		return printHelp("validate", stdout, stderr)
	}
	cfg, err := config.Load(parsed.configPath, env)
	if err == nil {
		// Inspection has no deadline to honor, so a background context.
		err = applyCLIOverrides(context.Background(), cfg, parsed.overrides)
	}
	if err == nil {
		err = cfg.Validate()
	}
	if err == nil {
		// output.schema is only checkable by compiling it, which config
		// deliberately does not do (it never imports internal/schema). Doing
		// it here keeps `validate` honest: a config it calls OK must not blow
		// up with exit 2 on the next run.
		_, err = compileOutputSchema(cfg)
	}
	if err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return ExitConfigError
	}
	_, _ = fmt.Fprintf(stdout, "%s: OK\n", parsed.configPath)
	return ExitOK
}

// cmdExplain prints the dry-run report for a config: what the agent may
// touch, spend and emit, plus warnings for valid-but-suspicious settings.
//
// It performs everything a run would do up to - but not including - provider
// construction: load, validate, compile output.schema, build the tool
// registry. No provider call, no tokens, no session file.
//
// It does contact the configured MCP servers, which is the one network the
// report needs and the one an operator expects it to make: a server's toolset
// is not in the YAML, so a dry run that did not ask for it would be silent
// about the largest unreviewed surface in the config. A server that cannot be
// reached is REPORTED, never fatal - even a `required: true` one, whose
// failure would abort `amele run` with exit 8.
//
// CONTRACT: explain REPORTS, run GATES. Every config problem it finds after
// the file LOADS - unset ${VAR}s, validation violations, an uncompilable
// output.schema, a registry that cannot be built - is printed in the report's
// PROBLEMS section, and the command still exits 0. Exit 2 is what survives
// from the load itself, where there is no *Config to describe: an unreadable
// file, unparseable YAML, an unknown key or wrong type, a literal
// provider.api_key (rejected on the raw bytes, so it fires on a file that
// reads and parses fine) or an unusable system_prompt_file - plus usage
// errors, including a malformed --set. docs/contracts/cli.md enumerates the
// set; TestExplainExitTwoCases pins it. Refusing to describe a broken config
// was the wrong trade - the
// operator pre-flighting somebody else's pack on a fresh host is precisely
// the reader who has unset variables and no workspace yet, and `run` still
// refuses to touch such a config (exit 2), which is where that judgement
// belongs.
func cmdExplain(ctx context.Context, args []string, stdout, stderr io.Writer, env config.LookupEnv) int {
	parsed, ok := parseInspectArgs(context.Background(), env, "explain", usageExplain, args, stderr)
	if !ok {
		return ExitConfigError
	}
	if parsed.help {
		return printHelp("explain", stdout, stderr)
	}
	cfg, err := config.LoadTolerant(parsed.configPath, env)
	if err != nil {
		// Unreadable or unparseable: there is no config to describe.
		_, _ = fmt.Fprintln(stderr, err)
		return ExitConfigError
	}
	// Applied before anything is inspected: explain describes the invocation
	// it was given, overrides and all. A malformed --set is a usage error
	// (the command line, not the file), so it keeps failing loudly.
	if err := applyCLIOverrides(ctx, cfg, parsed.overrides); err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return ExitConfigError
	}
	problems, registry := explainProblems(cfg)
	// The servers are dialled for real, and closed before the report is
	// printed: a stdio server is a child process, and `explain` must not leave
	// one behind. WithoutCancel for the close so a Ctrl-C that ended the dial
	// still gets an orderly shutdown. A registry that could not be built is
	// already a problem line; without one there is nothing to register into,
	// so the servers are not contacted at all.
	var mcpReports []explain.MCPServerReport
	if registry != nil {
		reports, mcpProblems, set := explainMCP(ctx, cfg, parsed.configPath, registry, env, version)
		set.close(context.WithoutCancel(ctx))
		mcpReports = reports
		problems = append(problems, mcpProblems...)
	}
	_, _ = fmt.Fprint(stdout, explain.Render(cfg, registry, parsed.overrides, problems, nil, mcpReports))
	return ExitOK
}

// explainProblems collects everything that would make `amele run` refuse this
// config, in the order a run would hit it, and returns the tool registry when
// one could be built (nil otherwise - Render then skips the warnings that
// need it).
//
// The registry is built for real (fs sandbox checks included) so the report
// reflects the tools a run would actually hold, and so the
// unknown-permission-entry warning checks against the truth.
func explainProblems(cfg *config.Config) (problems []string, registry *tools.Registry) {
	if missing := cfg.EnvMissing(); len(missing) > 0 {
		problems = append(problems, fmt.Sprintf(
			"undefined environment variable(s): %s - set them before running (see REQUIREMENTS below)",
			strings.Join(missing, ", ")))
	}
	problems = append(problems, cfg.Violations()...)
	// Same reasoning as cmdValidate: a schema that cannot compile is a config
	// error the reader must be told about, even though explain no longer
	// refuses the report over it.
	if _, err := compileOutputSchema(cfg); err != nil {
		problems = append(problems, err.Error())
	}
	registry, err := buildRegistry(cfg)
	if err != nil {
		problems = append(problems, err.Error())
		registry = nil
	}
	return problems, registry
}

// cmdDoctor runs the pre-flight checks (internal/doctor) and gates on them.
//
// CONTRACT (docs/contracts/cli.md): exit 0 when every check passed, 1 when
// any failed, 2 when there was no config to check at all - unreadable file,
// malformed --set. A config that loads but does not validate is checked
// anyway, with its violations as the config check's FAIL: the operator asked
// what is wrong with this host and this file, and the answer is the whole
// list, not the first item.
func cmdDoctor(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer, env config.LookupEnv) int {
	parsed, ok := parseInspectArgs(context.Background(), env, "doctor", usageDoctor, args, stderr)
	if !ok {
		return ExitConfigError
	}
	if parsed.help {
		return printHelp("doctor", stdout, stderr)
	}
	// Tolerant, like explain: an unset ${VAR} is a finding for the env check,
	// not a reason to print nothing.
	cfg, err := config.LoadTolerant(parsed.configPath, env)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return ExitConfigError
	}
	if err := applyCLIOverrides(ctx, cfg, parsed.overrides); err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return ExitConfigError
	}
	problems := cfg.Violations()
	if _, err := compileOutputSchema(cfg); err != nil {
		problems = append(problems, err.Error())
	}
	opts := doctor.Options{
		Problems: problems,
		IsTTY:    func() bool { return stdinIsTerminal(stdin) },
	}
	if cfg.Lock {
		if lock, err := lockFilePath(parsed.configPath); err == nil {
			opts.LockPath = lock
		}
	}
	report := doctor.Run(ctx, cfg, opts)
	// SECURITY: a base_url can carry a credential and a workspace path can
	// carry an interpolated secret; the report is redacted as a whole, the
	// way explain's is.
	_, _ = fmt.Fprint(stdout, session.Redactor(agentSecrets(cfg))(report.Render()))
	if report.Failed() {
		return ExitTaskFailed
	}
	return ExitOK
}

// cmdSchema prints the embedded config JSON Schema (docs/contracts/
// config.schema.json) so editors and tooling can consume it without a source
// checkout: `amele schema > config.schema.json`.
//
// CONTRACT: stdout carries exactly the schema document plus a trailing
// newline, nothing else - the output must be a valid JSON file as-is. The
// command takes no arguments; any argument is a usage error (exit 2), so a
// misremembered `amele schema config.yaml` fails loudly instead of silently
// ignoring the file.
func cmdSchema(args []string, stdout, stderr io.Writer) int {
	if hasHelpFlag(args) {
		return printHelp("schema", stdout, stderr)
	}
	if len(args) != 0 {
		_, _ = fmt.Fprintln(stderr, "usage: amele schema")
		return ExitConfigError
	}
	raw := config.SchemaJSONBytes()
	if !bytes.HasSuffix(raw, []byte("\n")) {
		raw = append(raw, '\n')
	}
	_, _ = stdout.Write(raw)
	return ExitOK
}

// cmdVersion prints this binary's build identity - version, commit, build
// date and the Go/platform triple that compiled it - as ONE line on stdout.
//
// CONTRACT: exit 0, stdout carries exactly that line plus a trailing newline
// and nothing else. `amele version`, `amele --version` and `amele -V` are
// three spellings of the same command (dispatched from the same switch case
// in run); this function is the single implementation all three share, so
// the wording cannot drift between them. The command takes no arguments -
// any argument is a usage error (exit 2), the same stance `schema` and
// `init` already take, so a typo like `amele version --json` fails loudly
// instead of silently being ignored.
func cmdVersion(args []string, stdout, stderr io.Writer) int {
	if hasHelpFlag(args) {
		return printHelp("version", stdout, stderr)
	}
	if len(args) != 0 {
		_, _ = fmt.Fprintln(stderr, "usage: amele version")
		return ExitConfigError
	}
	_, _ = fmt.Fprintf(stdout, "amele %s (commit %s, built %s, %s, %s/%s)\n",
		version, commit, date, runtime.Version(), runtime.GOOS, runtime.GOARCH)
	return ExitOK
}

// cmdCompletion prints the static completion script for one shell
// (completions.go) to stdout.
//
// CONTRACT: exactly one argument, the shell name - no shell name or more than
// one argument is a usage error (exit 2), and so is a shell name completion
// does not know, which names the shells it DOES know so the typo is
// immediately actionable. `-h`/`--help` is honored only as the sole argument,
// the same fixed-arity stance `schema`, `init` and `version` take.
func cmdCompletion(args []string, stdout, stderr io.Writer) int {
	if hasHelpFlag(args) {
		return printHelp("completion", stdout, stderr)
	}
	if len(args) != 1 {
		_, _ = fmt.Fprintln(stderr, usageCompletion)
		return ExitConfigError
	}
	script, ok := completionScripts[args[0]]
	if !ok {
		_, _ = fmt.Fprintf(stderr, "amele completion: unknown shell %q (want bash, zsh or fish)\n%s\n", args[0], usageCompletion)
		return ExitConfigError
	}
	_, _ = fmt.Fprint(stdout, script)
	return ExitOK
}

// starterConfig is the annotated YAML written by `amele init`. It must pass
// `amele validate` exactly as written once AMELE_API_KEY is set (a test enforces
// this), so the enabled fields stay conservative: fs tools only, every budget
// armed, session logging on. Everything riskier or optional - anthropic
// provider, shell, permissions, output.schema - ships as accurate commented
// examples, because a scaffold should show the doors without opening them.
// A const (not go:embed) on purpose: one file, and the template stays greppable
// next to the command that writes it.
const starterConfig = `# amele starter config - generated by amele init.
# Any ${VAR} is read from the environment when the config loads.
# Check it:      amele validate agent.yaml
# Full schema:   amele schema

# Model identifier sent to the provider. Override per run with --model.
model: gpt-4o-mini

provider:
  # OpenAI-compatible endpoint (the default protocol). Point base_url at any
  # compatible gateway: OpenRouter, Ollama, vLLM, ...
  base_url: https://api.openai.com/v1
  # Secrets never live in YAML; only ${ENV_VAR} references are accepted.
  api_key: ${AMELE_API_KEY}
  # Native Anthropic API instead: set type and remove base_url (the client
  # defaults to the official endpoint; a custom one must NOT end with /v1).
  # type: anthropic

system_prompt: You are a helpful assistant. Be concise and precise.

tools:
  # Sandboxed filesystem tools (fs_read/fs_write/fs_list). They cannot reach
  # outside the workspace, which defaults to this file's directory.
  fs: true
  # Builtin shell tool - off by default. allow/deny are glob patterns matched
  # against the whole command (deny wins); they prevent accidents, they are
  # not a security boundary.
  # shell:
  #   enabled: true
  #   allow: ["git *", "ls*"]
  #   deny: ["git push*"]

# Budgets - the kill switches that make unattended runs safe. Exceeding any
# of them ends the run with exit code 3.
limits:
  max_turns: 20      # provider round-trips
  max_tokens: 200000 # cumulative input+output tokens (the primary budget)
  timeout: 5m        # wall clock for the whole run

# Append-only JSONL session log, one file per run (relative to this file).
session_dir: sessions

# Single-flight guard for cron: with lock: true, a run that starts while
# another run of THIS config is still going exits 7 instead of interleaving
# with it. Off by default so the same config can be run concurrently with
# different tasks.
# lock: true

# Per-tool approval profile: allow | ask | deny. "ask" prompts on the
# terminal and degrades to a logged deny when no TTY is attached (cron-safe).
# permissions:
#   default: allow
#   tools:
#     fs_write: ask
#     shell: deny

# Constrain the final answer to a JSON Schema: stdout then carries only JSON
# that validated against it, and an unmet schema exits with code 6.
# output:
#   schema:
#     type: object
#     required: [summary]
#     properties:
#       summary: {type: string}
#   max_schema_retries: 2
`

// cmdInit writes the starter config to path (default agent.yaml) and points
// the user at `amele validate` on stderr. stdout stays empty - init composes
// in scripts like every other command. It takes stdout only so `amele init -h`
// can print the help page there, like every other command's help flag.
//
// CONTRACT: an existing file is never overwritten (exit 2). init is a
// scaffold: it creates a starting point, and a tool that can destroy the
// config a user has been editing is worse than no tool.
func cmdInit(args []string, stdout, stderr io.Writer) int {
	if hasHelpFlag(args) {
		return printHelp("init", stdout, stderr)
	}
	if len(args) > 1 {
		_, _ = fmt.Fprintln(stderr, "usage: amele init [path]")
		return ExitConfigError
	}
	path := "agent.yaml"
	if len(args) == 1 {
		path = args[0]
	}
	if err := writeStarterConfig(path, createExclusive); err != nil {
		if errors.Is(err, os.ErrExist) {
			_, _ = fmt.Fprintf(stderr, "amele init: %s already exists; refusing to overwrite (pick another path or remove it first)\n", path)
		} else {
			_, _ = fmt.Fprintf(stderr, "amele init: %v\n", err)
		}
		return ExitConfigError
	}
	_, _ = fmt.Fprintf(stderr, "amele: wrote %s - next: set AMELE_API_KEY and run: amele validate %s\n", path, path)
	return ExitOK
}

// fileCreator creates the file `init` is about to fill. It is a parameter of
// writeStarterConfig (rather than a call to os.OpenFile inside it) so the
// write-failure path - a full disk, a quota, an I/O error - is reachable from
// a hermetic test; production always passes createExclusive.
type fileCreator func(path string) (io.WriteCloser, error)

// createExclusive is the production fileCreator: create-or-fail, never
// truncate. O_EXCL makes `init`'s no-overwrite guarantee atomic - a separate
// existence check would race against anything else creating the file - and
// 0600 keeps a config that will grow secrets private from the first byte.
func createExclusive(path string) (io.WriteCloser, error) {
	return os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600) //nolint:gosec // G304: creating the user-named file is this command's purpose.
}

// writeStarterConfig creates path through create and writes starterConfig into
// it. A create failure is returned unwrapped, so the caller can still match
// os.ErrExist; a write or close failure is wrapped with the path.
//
// CONTRACT: on a write or close failure the just-created file is REMOVED. It
// was created by this call and can only be empty or half-written, and leaving
// it behind turns a transient disk error into a permanent one: the retry the
// user is about to type would hit "already exists" and refuse to write the
// config they never got.
func writeStarterConfig(path string, create fileCreator) error {
	f, err := create(path)
	if err != nil {
		return err
	}
	_, err = io.WriteString(f, starterConfig)
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		// The removal's own error is deliberately not reported: the write
		// failure is the actionable one, and a failed cleanup leaves exactly
		// the state that existed before this fix.
		_ = os.Remove(path) //nolint:gosec // G703: this removes the very file the create above just made at the user-named path; that path IS the command's argument.
		return fmt.Errorf("writing %s: %w", path, err)
	}
	return nil
}

// overrideFlag is the flag.Value behind --set and its sugar spellings
// (--model, -w/--workspace). Every one of them appends to the SAME ordered
// list, which is what makes the merge rule statable in one sentence: the
// entries apply in the order they were written and the last one for a key
// wins. A precedence between the spellings would instead make the effective
// value depend on a rule the command line cannot show.
type overrideFlag struct {
	// list is the shared, ordered override list every flag appends to.
	list *[]string
	// key is empty for --set, whose argument is already a key=value pair, and
	// the config key for a sugar flag ("model", "workspace").
	key string
	// skipEmpty drops an empty value instead of recording it. Only --model
	// sets it: `--model "$MODEL"` with an unset variable has meant "no
	// override" since Phase 1, and a wrapper script relying on that must not
	// start failing with "model is required". `--set model=` is how one asks
	// for the empty value deliberately.
	skipEmpty bool
}

// String implements flag.Value. It reports the empty string (never a value
// read back from the list) so the flag package prints no default for a flag
// whose whole point is to be repeated; a nil receiver is safe, which is how
// the flag package probes for zero values.
func (f *overrideFlag) String() string { return "" }

// Set implements flag.Value by appending this occurrence to the shared list.
// It never fails: the pair's shape and the key's admissibility are
// config.ApplyOverrides's job, and reporting them here would split one error
// message across two layers.
func (f *overrideFlag) Set(value string) error {
	if f.key == "" {
		*f.list = append(*f.list, value)
		return nil
	}
	if value == "" && f.skipEmpty {
		return nil
	}
	*f.list = append(*f.list, f.key+"="+value)
	return nil
}

// registerOverrideFlags defines --set and the -w/--workspace shortcut on fs,
// all appending to list. withModel adds the --model shortcut, which only `run`
// and `chat` carry (validate and explain take `--set model=...`, so the frozen
// flag stays exactly where it already was).
func registerOverrideFlags(fs *flag.FlagSet, list *[]string, withModel bool) {
	fs.Var(&overrideFlag{list: list}, "set", "override one config field: key=value (repeatable; see 'amele help run')")
	// One value registered under both names: the flag package treats -w and
	// --w alike, so this is purely about the long spelling a script should use.
	workspace := &overrideFlag{list: list, key: "workspace"}
	fs.Var(workspace, "w", "shortcut for --set workspace=DIR")
	fs.Var(workspace, "workspace", "shortcut for --set workspace=DIR")
	if withModel {
		fs.Var(&overrideFlag{list: list, key: "model", skipEmpty: true}, "model", "shortcut for --set model=MODEL")
	}
}

// agentArgs is the argument shape `run` and `chat` share: a config path, the
// config overrides collected from the flags, and whatever the caller wrote
// after them.
type agentArgs struct {
	configPath string
	// overrides are the `key=value` pairs --set, --model and -w produced, in
	// command-line order (config.ApplyOverrides applies them in that order).
	overrides []string
	// help reports that -h/--help was given in a flag position. The caller
	// prints its page and returns before anything is loaded, so a help request
	// never depends on the config path being real.
	help bool
	// quiet drops the summary line and the non-error notes; verbose adds a
	// progress line per loop event. Both are stderr-only and mutually
	// exclusive (parseAgentArgs rejects the combination).
	quiet   bool
	verbose bool
	// resume is the session log --resume named, empty when the flag was not
	// given. It is `run`-only in meaning but parsed for both commands, so
	// `chat --resume x` can be answered with what is actually wrong instead
	// of an unknown-flag error (cmdChat rejects a non-empty value).
	//
	// CONTRACT: the string is the path exactly as the operator typed it. It is
	// what run_start.resumed_from records, so the log names the file the
	// operator can find rather than a resolved form they never wrote.
	resume string
	// resumeSet records that --resume was given at all, which the value alone
	// cannot say: `--resume ""` and no flag both leave resume empty. It is
	// what lets an empty path be refused (issue #30) instead of read as "start
	// from scratch" - in a pipeline an empty value is a broken `$(...)`
	// substitution far more often than an intent.
	resumeSet bool
	// rest is the free-form remainder: task text for `run`; for `chat` any
	// remainder is a usage error, because a chat reads its input from stdin.
	rest []string
}

// rejectFlagInConfigPathSlot reports the argument-order mistake behind a
// flag-shaped first argument: `amele run --set model=x agent.yaml`. All four
// config-taking commands share it, so the four cannot disagree about an
// argument order there is only one of.
//
// CONTRACT: exit 2, like every usage error. The diagnosis exists because the
// alternative is a lie about the filesystem - taking "--set" as the config
// path made the command report "open --set: no such file or directory", which
// sends the reader looking for a missing file instead of showing them that
// amele puts the config path first and the flags after it (flags-first is the
// GNU habit, so this is the invocation a new user tries). "-" has never been
// a usable config path, so diagnosing it costs nothing.
func rejectFlagInConfigPathSlot(name, usage, arg string, stderr io.Writer) {
	_, _ = fmt.Fprintf(stderr,
		"amele %s: %q is a flag, but the first argument is the config path - write the flags after it\n%s\n",
		name, arg, usage)
}

// flagsAgree applies the two rules a parsed flag set can still break after
// the flag package accepted every flag on its own. It writes the reason to
// stderr and reports false on a violation. CONTRACT: both are usage errors
// like any other - exit 2, nothing loaded, no session log created.
func (a agentArgs) flagsAgree(name, usage string, stderr io.Writer) bool {
	if a.quiet && a.verbose {
		// They ask for opposite things. Letting one silently win would hide a
		// mistake in a script that means to change how noisy a cron job is.
		_, _ = fmt.Fprintf(stderr, "amele %s: -q/--quiet and -v/--verbose cannot be combined\n", name)
		return false
	}
	if a.resumeSet && a.resume == "" && name == "run" {
		// The flag was given with nothing to read. Treating it as "no resume"
		// would start a fresh run and exit 0 on what is almost certainly a
		// script's empty substitution (issue #30). chat is exempt only
		// because it refuses the flag at every value with its own sentence.
		_, _ = fmt.Fprintf(stderr, "amele %s: --resume needs a path\n%s\n", name, usage)
		return false
	}
	return true
}

// setFlag is a string flag that remembers whether it was given at all. The
// flag package hands a *string flag its default for an absent flag and "" for
// an explicitly empty one, and the two are indistinguishable afterwards; this
// Value keeps the distinction so a flag that NEEDS a value can refuse an empty
// one (--resume, issue #30) without turning "" into a sentinel path.
type setFlag struct {
	value string
	set   bool
}

// String implements flag.Value. It returns the value as given.
func (f *setFlag) String() string { return f.value }

// Set implements flag.Value: it records the value and that the flag was seen.
func (f *setFlag) Set(v string) error {
	f.value, f.set = v, true
	return nil
}

// parseAgentArgs parses that shared shape for the named command. It returns
// the exit code the caller should end with when parsing did not succeed: on a
// usage error it writes the reason to stderr and returns ExitConfigError; a
// config-path lookup ended by the run's context is reported as an interrupted
// run instead (reportLoadError). ExitOK means the arguments are usable.
//
// The config path comes first, then flags, then the free-form remainder.
// Parsed in this order because the flag package stops at the first non-flag
// argument - flags after the task text would be task text.
//
// CONTRACT: that flag-stop boundary is frozen behavior and it also bounds the
// help flag. -h/--help is registered on the FlagSet rather than scanned for by
// hand, so the flag package itself decides where the flag region ends: a -h
// written after the task text stays task text (`amele run cfg.yaml "explain
// the -h flag"` must run, not print help). The only hand-checked position is
// args[0], the config path slot, because the FlagSet never sees it - and "-h"
// was never a usable config path, so reading it as a help request is additive.
// The same slot rejects any other flag-shaped argument outright
// (rejectFlagInConfigPathSlot): no flag is a config path either.
func parseAgentArgs(ctx context.Context, env config.LookupEnv, name, usage string, args []string, stderr io.Writer) (agentArgs, int) {
	if len(args) < 1 {
		_, _ = fmt.Fprintln(stderr, usage)
		return agentArgs{}, ExitConfigError
	}
	if args[0] == "-h" || args[0] == "--help" {
		return agentArgs{help: true}, ExitOK
	}
	if strings.HasPrefix(args[0], "-") {
		rejectFlagInConfigPathSlot(name, usage, args[0], stderr)
		return agentArgs{}, ExitConfigError
	}
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	// The flag package's own error output dumps the defaults block, whose
	// single-dash spellings (-model, -set) appear in no example and no page;
	// an operator shown them writes a second wrong invocation. Discarded here
	// and reprinted below in the one format every usage error uses.
	fs.SetOutput(io.Discard)
	var overrides []string
	registerOverrideFlags(fs, &overrides, true)
	// Defining these takes -h away from the flag package's own ErrHelp path,
	// which would print the flag defaults to stderr and exit 2. Both spellings
	// are registered because the flag package treats -x and --x alike, so
	// "help" alone would already answer `-help` but not read as documentation.
	helpShort := fs.Bool("h", false, "print the detailed help page for this command")
	helpLong := fs.Bool("help", false, "print the detailed help page for this command")
	// Each verbosity level gets both spellings for the same reason: the short
	// one is what a human types, the long one is what a script should carry.
	quietShort := fs.Bool("q", false, "suppress the summary line and non-error notes")
	quietLong := fs.Bool("quiet", false, "suppress the summary line and non-error notes")
	verboseShort := fs.Bool("v", false, "print a progress line per loop event to stderr")
	verboseLong := fs.Bool("verbose", false, "print a progress line per loop event to stderr")
	// Registered for BOTH commands although only `run` can act on it: an
	// unknown-flag error would tell a chat user that the flag does not exist,
	// which is not the useful half of the truth. cmdChat refuses a non-empty
	// value with the reason.
	var resumeFlag setFlag
	fs.Var(&resumeFlag, "resume", "continue the run recorded in this session log")
	if err := fs.Parse(args[1:]); err != nil {
		_, _ = fmt.Fprintf(stderr, "amele %s: %v\n%s\n", name, err, usage)
		return agentArgs{}, ExitConfigError
	}
	parsed := agentArgs{
		configPath: args[0],
		overrides:  overrides,
		help:       *helpShort || *helpLong,
		quiet:      *quietShort || *quietLong,
		verbose:    *verboseShort || *verboseLong,
		resume:     resumeFlag.value,
		resumeSet:  resumeFlag.set,
		rest:       fs.Args(),
	}
	// Help wins over the conflicts below: someone who asked for the manual
	// gets the manual, and the page is where the flags are explained.
	if parsed.help {
		return agentArgs{help: true}, ExitOK
	}
	if !parsed.flagsAgree(name, usage, stderr) {
		return agentArgs{}, ExitConfigError
	}
	resolved, err := resolveConfigArg(ctx, env, parsed.configPath)
	if err != nil {
		if ctx.Err() != nil {
			// Not a usage error: a signal arrived while the path was being
			// stat'ed (issue #29), so the run is reported as interrupted.
			return agentArgs{}, reportLoadError(err, stderr)
		}
		_, _ = fmt.Fprintf(stderr, "amele %s: %v\n", name, err)
		return agentArgs{}, ExitConfigError
	}
	parsed.configPath = resolved
	return parsed, ExitOK
}

// resolveConfigArg maps the config argument to the file `Load` should open:
// a directory to its canonical entry point <dir>/agent.yaml, and a bare name
// that names nothing on disk to a saved agent under the user's config
// directory (issue #12). Files pass through untouched, and so does anything
// that is neither a directory nor a resolvable name, so Load keeps reporting
// them with its own errors.
//
// CONTRACT: resolution happens at parse time, BEFORE the run lock is derived,
// so `run pack/`, `run pack/agent.yaml` and `run <name>` for a saved pack all
// contend on the same lock file. An existing path always wins over a saved
// agent of the same spelling: the shell's own view of the filesystem is never
// second-guessed, and every invocation written before names existed resolves
// exactly as it did.
//
// Both lookups go through the context (issue #29): a stat on a hung mount
// blocks like a read, and this is the first thing the binary does with the
// path. A context error is returned as-is so the caller can report an
// interrupted run rather than a usage error.
func resolveConfigArg(ctx context.Context, env config.LookupEnv, arg string) (string, error) {
	info, err := ctxfile.Stat(ctx, arg)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return "", ctxErr
		}
		// Nothing on disk: a bare name may be a saved agent. Anything else
		// passes through so Load reports the missing file.
		return resolveAgentName(ctx, env, arg)
	}
	if !info.IsDir() {
		return arg, nil
	}
	candidate := filepath.Join(arg, "agent.yaml")
	if _, err := ctxfile.Stat(ctx, candidate); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return "", ctxErr
		}
		return "", fmt.Errorf("no agent.yaml in %s", arg)
	}
	return candidate, nil
}

// resolveAgentName looks a bare name up under the agents directory:
// <dir>/<name>.yaml first, then <dir>/<name>/agent.yaml (a saved pack). An
// argument that is not a bare name, or a host with no config directory at all,
// is handed back unchanged; a bare name that resolves to nothing is an error
// naming both places the lookup went, so the operator learns where a saved
// agent is expected to live without reading the docs.
func resolveAgentName(ctx context.Context, env config.LookupEnv, arg string) (string, error) {
	dir := agentsDir(env)
	if !isAgentName(arg) || dir == "" {
		return arg, nil
	}
	for _, candidate := range []string{filepath.Join(dir, arg+".yaml"), filepath.Join(dir, arg, "agent.yaml")} {
		if _, err := ctxfile.Stat(ctx, candidate); err == nil {
			return candidate, nil
		} else if ctxErr := ctx.Err(); ctxErr != nil {
			return "", ctxErr
		}
	}
	return "", fmt.Errorf("no such file %q and no saved agent named %q (looked in %s)", arg, arg, dir)
}

// isAgentName reports whether arg is spelled like a saved agent's name rather
// than like a path: one bare component, no directory separator, no YAML
// extension. `sentry` is a name; `./sentry`, `sentry.yaml` and `packs/sentry`
// are paths and are never looked up - a path an operator typed points where it
// points.
func isAgentName(arg string) bool {
	if arg == "" || arg == "." || arg == ".." || filepath.Base(arg) != arg {
		return false
	}
	if strings.ContainsAny(arg, `/\`) {
		return false
	}
	ext := strings.ToLower(filepath.Ext(arg))
	return ext != ".yaml" && ext != ".yml"
}

// agentsDir is where saved agents live: the amele directory under the XDG
// config home - $XDG_CONFIG_HOME/amele, or $HOME/.config/amele when the
// variable is unset (on Windows %AppData%\amele). Empty when the host names
// no such place, in which case there is nothing to look in and a bare name
// is treated as the file it would be anywhere else.
//
// The environment is the injected LookupEnv rather than os.Getenv
// (docs/engineering.md §5.4), which is also what makes the lookup testable
// without touching the developer's own config directory.
func agentsDir(env config.LookupEnv) string {
	if xdg, ok := env("XDG_CONFIG_HOME"); ok && xdg != "" {
		return filepath.Join(xdg, "amele")
	}
	if runtime.GOOS == "windows" {
		if appData, ok := env("AppData"); ok && appData != "" {
			return filepath.Join(appData, "amele")
		}
		return ""
	}
	if home, ok := env("HOME"); ok && home != "" {
		return filepath.Join(home, ".config", "amele")
	}
	return ""
}

// reportLoadError prints a config-load failure and maps it to an exit code.
//
// CONTRACT: a config that cannot be loaded is exit 2 - EXCEPT when what ended
// the load was the run's own context: a SIGINT/SIGTERM that arrived while the
// config file or a prompt file was being read (issue #29). That is an
// interrupted run, reported as the Signals contract says (`run interrupted:
// context canceled`, exit 1), not a broken config. No session log exists yet,
// so there is no run_end to write.
func reportLoadError(err error, stderr io.Writer) int {
	for _, cause := range []error{context.Canceled, context.DeadlineExceeded} {
		if errors.Is(err, cause) {
			err = interruptedError(cause)
			_, _ = fmt.Fprintln(stderr, err)
			return exitCodeFor(err)
		}
	}
	_, _ = fmt.Fprintln(stderr, err)
	return ExitConfigError
}

// loadAgentConfig loads, overrides, validates and compiles one agent config -
// everything both commands must do before anything can block or contact a
// provider. The returned validator is nil when the config declares no
// output.schema.
//
// CONTRACT: every error here is exit 2 (config error) at the call site, and it
// must be reported without spending a single token.
func loadAgentConfig(ctx context.Context, parsed agentArgs, env config.LookupEnv) (*config.Config, *schema.Validator, error) {
	// Through the context: the config file and a prompt file are
	// operator-named paths, and a read of one that blocks must end when a
	// signal arrives rather than hold the process past it (issue #29). No
	// deadline exists yet - limits.timeout is IN the file being read - so
	// only a signal can end the read here.
	cfg, err := config.LoadContext(ctx, parsed.configPath, env)
	if err != nil {
		return nil, nil, err
	}
	// Overrides are applied BEFORE Validate, so they participate in it: "no
	// model in YAML but --set model=X given" is valid, "still no model" is
	// caught, and a nonsense override (a negative budget, an unreachable
	// workspace) is an exit-2 config error rather than a mid-run surprise.
	if err := applyCLIOverrides(ctx, cfg, parsed.overrides); err != nil {
		return nil, nil, err
	}
	if err := cfg.Validate(); err != nil {
		return nil, nil, err
	}
	// A schema that cannot compile is the user's config being wrong, so it is
	// caught here rather than mid-run.
	validator, err := compileOutputSchema(cfg)
	if err != nil {
		return nil, nil, err
	}
	return cfg, validator, nil
}

// applyCLIOverrides applies the `key=value` pairs collected from --set and its
// shortcuts. Every command that loads a config goes through it, so the four
// commands cannot disagree about what an override means.
//
// The working directory is read HERE rather than inside internal/config
// because process state belongs to cmd (docs/engineering.md §5.4: library code takes it
// injected). It is the base for CLI-given paths: a path typed in a shell means
// what it means in that shell, while the same field written in YAML stays
// relative to the YAML.
func applyCLIOverrides(ctx context.Context, cfg *config.Config, overrides []string) error {
	if len(overrides) == 0 {
		return nil
	}
	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("resolving the working directory for --set paths: %w", err)
	}
	return config.ApplyOverridesContext(ctx, cfg, overrides, cwd)
}

// inspectArgs is the argument shape `validate` and `explain` share: one config
// path, plus the overrides that describe WHICH invocation is being inspected.
type inspectArgs struct {
	configPath string
	overrides  []string
	// help reports that the command was invoked as `amele <cmd> -h` and
	// nothing else.
	help bool
}

// parseInspectArgs parses that shape for the named command, writing the reason
// to stderr and reporting ok=false on a usage error.
//
// CONTRACT: the arity rule is unchanged - exactly one positional argument, the
// config path, and it comes FIRST, as in run and chat. A --set pair is a flag,
// not a positional, so it is not an arity violation. -h/--help stays honored
// only as the SOLE argument (docs/contracts/cli.md): alongside anything else
// the invocation remains a usage error, so a wrong argument count is never
// answered with a help page and an exit 0.
func parseInspectArgs(ctx context.Context, env config.LookupEnv, name, usage string, args []string, stderr io.Writer) (inspectArgs, bool) {
	if len(args) < 1 {
		_, _ = fmt.Fprintln(stderr, usage)
		return inspectArgs{}, false
	}
	if args[0] == "-h" || args[0] == "--help" {
		// Only as the SOLE argument, exactly as in every flag position below:
		// `amele validate -h extra.yaml` named a file too, so it is a wrong
		// argument count - answering it with a page and an exit 0 would hide
		// the mistake from a script that checks $?.
		if len(args) == 1 {
			return inspectArgs{help: true}, true
		}
		_, _ = fmt.Fprintf(stderr, "amele %s: -h/--help is honored only as the sole argument\n%s\n", name, usage)
		return inspectArgs{}, false
	}
	if strings.HasPrefix(args[0], "-") {
		rejectFlagInConfigPathSlot(name, usage, args[0], stderr)
		return inspectArgs{}, false
	}

	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	// The flag package's own error output dumps the defaults block; these
	// commands answer a usage error with the same one-line usage as every
	// other command, so its output is discarded and reprinted below.
	fs.SetOutput(io.Discard)
	var overrides []string
	registerOverrideFlags(fs, &overrides, false)
	// Registered (rather than left to the flag package's ErrHelp path) so the
	// arity violation can be reported as what it is.
	helpShort := fs.Bool("h", false, "print the detailed help page for this command")
	helpLong := fs.Bool("help", false, "print the detailed help page for this command")
	if err := fs.Parse(args[1:]); err != nil {
		_, _ = fmt.Fprintf(stderr, "amele %s: %v\n%s\n", name, err, usage)
		return inspectArgs{}, false
	}
	if *helpShort || *helpLong {
		_, _ = fmt.Fprintf(stderr, "amele %s: -h/--help is honored only as the sole argument\n%s\n", name, usage)
		return inspectArgs{}, false
	}
	if len(fs.Args()) > 0 {
		_, _ = fmt.Fprintln(stderr, usage)
		return inspectArgs{}, false
	}
	resolved, err := resolveConfigArg(ctx, env, args[0])
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "amele %s: %v\n", name, err)
		return inspectArgs{}, false
	}
	return inspectArgs{configPath: resolved, overrides: overrides}, true
}

// cmdRun executes a one-shot agent run and maps every failure to the exit
// code contract.
func cmdRun(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer, env config.LookupEnv) int {
	parsed, parseCode := parseAgentArgs(ctx, env, "run", usageRun, args, stderr)
	if parseCode != ExitOK {
		return parseCode
	}
	if parsed.help {
		return printHelp("run", stdout, stderr)
	}
	taskArgs := strings.Join(parsed.rest, " ")

	cfg, validator, err := loadAgentConfig(ctx, parsed, env)
	if err != nil {
		return reportLoadError(err, stderr)
	}

	// ONE registry for this run, built before the first sink exists so every
	// sink below shares it (see runSecrets).
	secrets := runSecrets(cfg)

	// One reader for the whole run: the OAuth question in startRun and the
	// permission prompter consume the SAME stdin, so they must share one
	// buffer - two independent bufio readers would let one swallow bytes the
	// other needed. It is created here rather than at buildAgent because the
	// gate asks first; wrapping reads nothing by itself, so buildTask's stdin
	// contract is untouched (and the two never both read: the gate only asks
	// on a terminal, and readPipedInput never reads a terminal).
	lines := newLineReader(stdin)

	release, startCode := startRun(ctx, cfg, validator, parsed, taskArgs, lines, stderr, env, secrets)
	if startCode != ExitOK {
		return startCode
	}
	defer release()

	// The run timeout is armed BEFORE anything that can block - most
	// importantly the stdin read below. An open pipe that never delivers
	// data must be interruptible by limits.timeout (live-test finding B3),
	// and so must the --resume log's read (issue #29): both go through ctx.
	if cfg.Limits.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, cfg.Limits.Timeout.Std())
		defer cancel()
	}

	task, history, replay, taskErr := prepareRun(ctx, cfg, parsed, taskArgs, stdin)
	// Reading stdin is already part of the run, so an interruption there is an
	// interrupted RUN, not a config error: it is carried past the agent
	// construction below and reported through the normal ending path.
	// CONTRACT (docs/contracts/cli.md "Signals"): run_end in the session log,
	// the cause and the summary on stderr, exit 1 - or exit 3 when the cause
	// was the configured limits.timeout.
	interrupted := errors.Is(taskErr, context.Canceled) || errors.Is(taskErr, context.DeadlineExceeded)
	if taskErr != nil && !interrupted {
		// Nothing started: no task was given at all, stdin itself failed, or
		// the log --resume named cannot be continued. CONTRACT: exit 2 for all
		// three, and a resume failure is printed as internal/resume phrased it
		// - those messages are written for the operator and already name both
		// the file and the config key that would make it resumable, so
		// wrapping them here would only add a prefix in front of the advice.
		_, _ = fmt.Fprintln(stderr, taskErr)
		return ExitConfigError
	}

	agent, answer, hints, err := buildAgent(cfg, validator, lines, stderr, secrets, parsed.quiet)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return ExitConfigError
	}
	if parsed.verbose {
		agent.Progress = progressLogger(stderr, secrets)
	}

	if interrupted {
		// Nothing will run, so no MCP server is started: spending a connect
		// attempt on a run that is already over would only add a failure to
		// the log.
		return reportInterruptedRead(agent, cfg, taskArgs, taskErr, validator != nil, parsed.quiet, stderr, secrets)
	}

	// run_start is written here rather than by loop.Run because the MCP
	// connects must land BETWEEN run_start and the first llm_response
	// (docs/contracts/jsonl-events.md, Ordering). loop.Run's contract already
	// hands that choice to callers driving their own history - `chat` has
	// always done it - so the run below goes through RunMessages.
	logResumeNote(stderr, parsed, replay, secrets)
	logRunStart(agent, cfg, task, parsed.resume, resumeInstruction(taskArgs), replay)

	set, mcpErr := connectMCP(ctx, cfg, agent.Registry, agent.Session, stderr, env, parsed.quiet, version, secrets)
	maps.Copy(hints, set.hints)
	// CONTRACT: mcp_disconnect precedes run_end, and run_end.mcp_errors is
	// final. WithoutCancel so an orderly close still happens after a SIGTERM
	// cancelled the run context.
	finish := func() {
		set.close(context.WithoutCancel(ctx))
		agent.Session.SetMCPErrors(set.errors())
	}
	if mcpErr != nil {
		// An interruption reaches the servers before it reaches the loop: a
		// SIGTERM during the connect fails every attempt. CONTRACT: that is an
		// interrupted RUN (exit 1), not a missing dependency (exit 8) - a cron
		// job that was told to stop must not page anyone about a server that
		// is perfectly healthy.
		if ctx.Err() != nil {
			mcpErr = interruptedError(ctx.Err())
		}
		code := exitCodeFor(mcpErr)
		finish()
		reportRun(agent, &loop.Result{}, mcpErr, code, validator != nil, parsed.quiet, stderr, secrets.Redact)
		return code
	}
	// After the MCP tools joined the registry: their schemas are the ones most
	// likely to lose a keyword, and they do not exist until this point.
	warnSanitizedToolSchemas(cfg, agent.Registry, stderr, parsed.quiet, secrets)

	res, runErr := agent.RunMessages(ctx, history)
	code := exitCodeFor(runErr)

	finish()
	reportRun(agent, res, runErr, code, validator != nil, parsed.quiet, stderr, secrets.Redact)

	if runErr == nil {
		// CONTRACT: stdout carries only the agent's final answer, so runs
		// compose in pipes (`amele run ... | jq`). A failed run - including a
		// schema failure (exit 6) - writes nothing here.
		_, _ = fmt.Fprintln(stdout, answer(res))
	}
	return code
}

// warnSanitizedToolSchemas prints the one line that says which JSON Schema
// keywords the gemini wire's sanitizer removed from which tool, or nothing at
// all when there was nothing to remove (and on every other wire).
//
// CONTRACT (design doc §"Gemini-specific mechanics" 1): nothing is dropped
// silently. Gemini's FunctionDeclaration.parameters is an OpenAPI-3.0 subset
// whose unknown keywords are hard 400s, so amele strips them rather than
// failing the run - which costs the model a constraint it can no longer see
// ("pattern", "additionalProperties"), and that trade has to be visible to the
// operator. `amele explain` lists it per tool before a token is spent; this is
// the same fact for the run that was never explained, which for an MCP toolset
// is the common case: those schemas arrive from the other side and can change
// under a config nobody edited.
//
// It is called AFTER the MCP servers joined the registry, so the line covers
// the definitions actually sent. quiet drops it like every other note (-q is
// errors only, docs/contracts/cli.md), and the text goes through the run's
// secret registry because a tool schema can carry an interpolated value.
//
// SECURITY: names and key PATHS only, each quoted with %q - no schema values,
// and no unescaped newline from a remote key that could forge a line.
func warnSanitizedToolSchemas(cfg *config.Config, reg *tools.Registry, stderr io.Writer,
	quiet bool, secrets *session.SecretSet) {
	if quiet || cfg.Provider.Type != config.ProviderTypeGemini || reg == nil {
		return
	}
	var stripped []string
	for _, def := range reg.Defs() {
		_, keys := llm.SanitizeGeminiSchema(def.Parameters)
		for _, key := range keys {
			stripped = append(stripped, fmt.Sprintf("%q: %q", def.Name, key))
		}
	}
	if len(stripped) == 0 {
		return
	}
	if len(stripped) > maxSanitizedWarnEntries {
		rest := len(stripped) - maxSanitizedWarnEntries
		stripped = append(stripped[:maxSanitizedWarnEntries],
			fmt.Sprintf("and %d more (run `amele explain` for the full list)", rest))
	}
	_, _ = fmt.Fprintln(stderr, secrets.Redact(
		"warning: tool schemas sanitized for the gemini wire (unsupported JSON Schema keywords and shapes removed): "+
			strings.Join(stripped, ", ")))
}

// maxSanitizedWarnEntries caps the tool:key pairs the one-line sanitizer
// warning lists. A large MCP toolset can strip hundreds of keys, and this
// line lands in cron mail; `amele explain` already lists every pair, so the
// warning only has to prove the stripping happened and name where the full
// list lives.
const maxSanitizedWarnEntries = 8

// interruptedError renders an interruption of a run that had already started,
// mapping it onto the exit code contract.
//
// CONTRACT (docs/contracts/cli.md "Signals"): an operator interrupt is exit 1,
// but exit 3 is reserved for configured budgets - and limits.timeout is the
// only thing that can produce a deadline here, the same split
// loop.wrapContextErr makes mid-run.
func interruptedError(cause error) error {
	if errors.Is(cause, context.DeadlineExceeded) {
		return fmt.Errorf("%w: %v", loop.ErrBudgetExceeded, cause)
	}
	return fmt.Errorf("run interrupted: %w", cause)
}

// prepareRun decides what conversation this run starts from, and reports
// whether it continues an earlier one.
//
// There are exactly two openings. A normal run renders its task
// (buildTask, which is the only thing that may read stdin) and opens with the
// system prompt plus that task. A --resume run rebuilds the conversation of an
// earlier run from its session log and opens with THAT, optionally followed by
// a new instruction; it does not call buildTask at all, so stdin is untouched
// and the prompt template never runs.
//
// It returns the task the session log should record, the history the loop is
// driven with, the rebuilt log for a resumed run (nil for a normal one - it is
// what the run_start origin fields and the -v note are derived from), and the
// error that stops the run. CONTRACT: every error here is exit 2 at the call
// site, EXCEPT a context error from the stdin read, which is an interrupted
// run - cmdRun makes that split, exactly as it did when it called buildTask
// itself.
func prepareRun(ctx context.Context, cfg *config.Config, parsed agentArgs, taskArgs string,
	stdin io.Reader) (string, []llm.Message, *resume.Replay, error) {
	if parsed.resume == "" {
		task, err := buildTask(ctx, cfg, taskArgs, stdin)
		if err != nil {
			return "", nil, nil, err
		}
		return task, openingHistory(cfg, task), nil, nil
	}
	// The provider identity AND the model decide whether the log's reasoning
	// payloads come back: they are signed or hash-checked by the backend that
	// produced them, for the model that produced them (internal/resume
	// Options), so replaying one into a different provider - or into another
	// model on the same one, which is what `--resume --model x` asks for - is
	// at best rejected. cfg.Model is the EFFECTIVE model, after --model and
	// --set.
	replay, err := resume.Read(ctx, parsed.resume,
		resume.Options{Provider: cfg.Provider.Identity(), Model: cfg.Model})
	if err != nil {
		return "", nil, nil, err
	}
	instruction := resumeInstruction(taskArgs)
	if replay.Completed && instruction == "" {
		return "", nil, nil, errResumeCompleted
	}
	// The task is the OLD run's task: it is what the conversation is about,
	// and repeating it in the new log is what lets the two be read as one
	// story. The origin fields say the rest.
	return replay.Task, resumeHistory(cfg, replay, instruction), replay, nil
}

// resumeInstruction is the follow-up a --resume run sends as its last user
// message: the task text as typed, or nothing. Whitespace-only arguments are
// no instruction at all - the same rule buildTask applies to a one-shot task,
// so `--resume log " "` cannot buy a round trip that asks the model nothing.
// Anything that survives is sent exactly as typed (resumeHistory) and recorded
// exactly as typed (run_start.resumed_instruction), so the two cannot differ.
func resumeInstruction(taskArgs string) string {
	if strings.TrimSpace(taskArgs) == "" {
		return ""
	}
	return taskArgs
}

// errResumeCompleted is the refusal for a log whose run already answered.
//
// CONTRACT: exit 2, before a token is spent. Such a log has nothing left to
// do on its own - resending it would ask the model to repeat an answer it
// already gave - so continuing it takes a new instruction from the operator,
// and the message says so rather than reporting an empty success.
var errResumeCompleted = errors.New("run already produced a final answer; pass an instruction to continue")

// resumeHistory builds the conversation a --resume run starts from: the
// CURRENT config's system prompt, the rebuilt history of the run being
// continued, and the operator's follow-up instruction when there is one.
//
// The system prompt is taken from the config rather than from the log because
// no log records one (internal/resume never yields a system message): the
// prompt is the current config's business, so editing it between the two runs
// is how an operator steers the continuation.
//
// CONTRACT: the instruction is sent VERBATIM as the last user message. The
// config's `prompt` template is deliberately NOT applied to it - the template
// shaped the ORIGINAL task, which the log already carries as the first user
// message, and re-rendering it around a follow-up would send the model a
// second copy of the framing it has been reading all along.
func resumeHistory(cfg *config.Config, replay *resume.Replay, instruction string) []llm.Message {
	history := make([]llm.Message, 0, len(replay.Messages)+2)
	if cfg.SystemPrompt != "" {
		history = append(history, llm.Message{Role: llm.RoleSystem, Content: cfg.SystemPrompt})
	}
	history = append(history, replay.Messages...)
	if instruction != "" {
		history = append(history, llm.Message{Role: llm.RoleUser, Content: instruction})
	}
	return history
}

// logRunStart writes the run's opening event, naming the log it continued when
// there was one.
//
// CONTRACT (docs/contracts/jsonl-events.md): a resumed run writes run_start
// with the three v1.9 origin fields plus the v1.11 instruction, and a normal
// run writes none of them, so absence keeps meaning "this run started from
// nothing" - including in every log written before the fields existed. The
// path is recorded exactly as the operator typed it, because that is the
// string that names the file again; resumed_turn counts the turns of the
// whole chain the path leads to, since that is how much conversation precedes
// this run's turn 1.
func logRunStart(agent *loop.Loop, cfg *config.Config, task, from, instruction string, replay *resume.Replay) {
	if replay == nil {
		agent.Session.RunStart(cfg.Model, cfg.Provider.Identity(), task)
		return
	}
	agent.Session.RunStartResumed(cfg.Model, cfg.Provider.Identity(), task, session.Resumed{
		From: from, Turn: replay.LastTurn, Pending: replay.Pending, Instruction: instruction,
	})
}

// logResumeNote prints the one -v line that says what a resumed run starts
// from, before the first turn is spent on it: how much conversation came back,
// which model and backend produced it, how many tool calls it carries no
// result for, and whether the reasoning carriers survived the provider/model
// gate (internal/resume Options). None of that is visible from the summary
// line, and the carrier verdict in particular explains a signature 400 that
// would otherwise look like a bug.
//
// Both guards live here rather than at the call site: a helper that decides
// whether it has anything to say keeps cmdRun's branch count where it is.
//
// SECURITY: the same two hazards progressLogger documents. The model, the
// provider identity and the path all come from operator-owned input that gets
// persisted (a cron job's stderr), so the line is redacted through the run's
// live secret registry and stripped of terminal control bytes exactly like a
// progress line.
func logResumeNote(stderr io.Writer, parsed agentArgs, replay *resume.Replay, secrets *session.SecretSet) {
	if !parsed.verbose || replay == nil {
		return
	}
	carriers := "not restored"
	if replay.Carriers {
		carriers = "restored"
	}
	note := fmt.Sprintf("resuming %s: %d %s of %s on %s; %d pending %s; reasoning carriers %s",
		parsed.resume, replay.LastTurn, pluralNoun(replay.LastTurn, "turn"), replay.Model, replay.Provider,
		len(replay.Pending), pluralNoun(len(replay.Pending), "tool call"), carriers)
	if replay.Links > 1 {
		// The turns above span every log the named one continues; say so, or
		// "5 turns" of a 1-turn file reads like a lie.
		note += fmt.Sprintf("; a chain of %d logs", replay.Links)
	}
	_, _ = fmt.Fprintf(stderr, "amele: %s\n", safeForTerminal(secrets.Redact(note), maxProgressLine))
}

// pluralNoun returns noun for a count of one and noun+"s" otherwise, so the
// human-facing lines read "1 turn" and "2 turns" like the summary line does.
func pluralNoun(n int, noun string) string {
	if n == 1 {
		return noun
	}
	return noun + "s"
}

// openingHistory builds the one-shot conversation `run` starts from: the
// system prompt, when there is one, then the task as the user message.
//
// It mirrors loop.Run, which cmdRun no longer calls: run_start has to be
// written before the MCP servers connect (see cmdRun), and loop.Run writes its
// own. Keeping the two lines here rather than adding a flag to the loop keeps
// the loop's contract - "callers driving their own history log their own
// opening event" - exactly as it was.
func openingHistory(cfg *config.Config, task string) []llm.Message {
	messages := make([]llm.Message, 0, 2)
	if cfg.SystemPrompt != "" {
		messages = append(messages, llm.Message{Role: llm.RoleSystem, Content: cfg.SystemPrompt})
	}
	return append(messages, llm.Message{Role: llm.RoleUser, Content: task})
}

// reportInterruptedRead closes out a run that was interrupted while its task
// input was still being read from stdin, and returns the process exit code.
//
// CONTRACT (docs/contracts/cli.md "Signals"): the stdin read happens inside the
// run - after the lock is taken, with the run timeout already armed - so an
// interruption there must leave exactly the evidence a mid-turn interruption
// leaves: run_start/run_end in the session log, the cause on stderr, the
// summary unless -q, and an empty stdout. Returning the raw read error instead
// (the original behavior) skipped the session ending entirely, so a cron job
// that was SIGTERMed at the wrong moment left a log nobody could audit.
//
// run_start is written here rather than by loop.Run, which is never reached:
// a session file whose only event is run_end would describe no run at all. The
// task recorded is what the operator typed, since the piped part never arrived.
func reportInterruptedRead(agent *loop.Loop, cfg *config.Config, taskArgs string, cause error, schemaMode, quiet bool,
	stderr io.Writer, secrets *session.SecretSet) int {
	err := interruptedError(cause)
	code := exitCodeFor(err)
	agent.Session.RunStart(cfg.Model, cfg.Provider.Identity(), taskArgs)
	// Zero accounting: nothing was spent, and the summary must not invent turns.
	reportRun(agent, &loop.Result{}, err, code, schemaMode, quiet, stderr, secrets.Redact)
	return code
}

// reportRun closes one `run` out on stderr and in the session log: the error
// if there was one, run_end always, then the summary and the notes.
//
// CONTRACT: run_end and the error are unconditional - the session must carry a
// truthful ending and a failure must always say why. quiet drops only the
// summary and the notes (docs/contracts/cli.md), so a `-q` cron job is silent
// exactly while it is healthy. schemaMode says whether output.schema was in
// play; without it a provider flagging unenforced responses has nothing worth
// warning about.
func reportRun(agent *loop.Loop, res *loop.Result, runErr error, code int, schemaMode, quiet bool, stderr io.Writer, redact func(string) string) {
	status := "success"
	if runErr != nil {
		status = "error"
		// SECURITY: the error line can quote remote text - an MCP connect
		// failure echoes whatever the server sent, which may include a header
		// value this run interpolated - so it is redacted like every other
		// operator-facing line.
		_, _ = fmt.Fprintln(stderr, redact(runErr.Error()))
	}
	// CONTRACT: Usage.Total() is input+output with the cached share already
	// inside input; the cache-read count rides beside it as a subset, never as
	// an addition (docs/contracts/jsonl-events.md).
	agent.Session.RunEnd(session.RunEnd{
		Status: status, ExitCode: code,
		Turns: res.Turns, ToolCalls: res.ToolCalls,
		TotalTokens: res.Usage.Total(), CacheReadTokens: res.Usage.CacheReadTokens,
		Fallbacks: res.Fallbacks,
		Duration:  res.Duration,
	})
	if quiet {
		return
	}
	_, _ = fmt.Fprintln(stderr, session.Summary(runErr == nil, session.Stats{
		Turns: res.Turns, ToolCalls: res.ToolCalls,
		TotalTokens: res.Usage.Total(), CachedTokens: res.Usage.CacheReadTokens,
		Fallbacks: res.Fallbacks, Duration: res.Duration,
	}))
	// The native-downgrade warning follows the summary, once per run: in
	// schema mode the operator must learn when provider-native enforcement was
	// unavailable and the validate+retry layer carried output.schema alone. It
	// is printed regardless of runErr - it is most valuable on an exit-6
	// failure, where it names a likely contributing cause. stderr only: stdout
	// is the answer channel (pipe contract).
	if schemaMode && res.SchemaEnforcementDropped {
		_, _ = fmt.Fprintln(stderr, "warning: provider did not enforce output.schema natively; the validate+retry layer was the only enforcement")
	}
}

// startRun is everything a run does before it may cost anything: the run lock,
// then the pre-connect OAuth phase.
//
// CONTRACT (spec §3.1, docs/contracts/cli.md): the order is lock -> login
// phase -> (caller) deadline -> run_start -> connect. The lock is taken before
// ANYTHING else - before stdin is read, before the session file is created,
// before the first token is bought - so a blocked run costs nothing and leaves
// no trace. The OAuth phase follows it, so a browser flow cannot race a second
// run of the same config, and precedes the `limits.timeout` deadline the
// caller arms next, because the minutes a human spends at an authorization
// server are not the agent's budget.
//
// It returns the lock's release function and ExitOK, or nil and the exit code
// to return - having already released the lock and left the run's evidence.
func startRun(ctx context.Context, cfg *config.Config, validator *schema.Validator, parsed agentArgs,
	taskArgs string, lines *lineReader, stderr io.Writer, env config.LookupEnv,
	secrets *session.SecretSet) (func(), int) {
	release, code := acquireRunLock(cfg, parsed.configPath, stderr)
	if code != ExitOK {
		return nil, code
	}
	if err := mcpCredentialGate(ctx, cfg, parsed.configPath, lines, stderr, env, secrets, parsed.quiet); err != nil {
		// The lock is dropped here rather than by the caller: the run never
		// started, and a config whose credential is missing must not keep the
		// next attempt out while the operator logs in.
		defer release()
		return nil, reportGateFailure(cfg, validator, parsed, taskArgs, err, lines, stderr, secrets)
	}
	return release, ExitOK
}

// reportGateFailure ends a run that the OAuth phase refused, leaving exactly
// the evidence a connect failure would have left.
//
// CONTRACT (docs/contracts/jsonl-events.md): an exit-8 run writes run_start and
// run_end with mcp_errors, whether the missing dependency was discovered by the
// pre-connect phase or by the connect itself. Without this the credential gate
// would be the one silent failure in the binary - the same reason
// reportInterruptedRead exists for a run interrupted while reading stdin.
//
// The session is opened HERE, after the refusal: building the agent is what
// creates the session file, and a run that is about to be refused must not
// leave one behind unless it also leaves the ending that explains it.
func reportGateFailure(cfg *config.Config, validator *schema.Validator, parsed agentArgs, taskArgs string,
	gateErr error, lines *lineReader, stderr io.Writer, secrets *session.SecretSet) int {
	code := exitCodeFor(gateErr)
	agent, _, _, err := buildAgent(cfg, validator, lines, stderr, secrets, parsed.quiet)
	if err != nil {
		// No session could be opened at all (a bad session_dir, a broken tool
		// definition). Both failures are reported: the one that ended the run
		// first, then the one that stopped it being recorded.
		_, _ = fmt.Fprintln(stderr, secrets.Redact(gateErr.Error()))
		_, _ = fmt.Fprintln(stderr, err)
		return ExitConfigError
	}
	// The task recorded is what the operator typed: nothing was rendered, and
	// stdin was never read.
	agent.Session.RunStart(cfg.Model, cfg.Provider.Identity(), taskArgs)
	agent.Session.SetMCPErrors(gateMCPErrors(gateErr))
	// Zero accounting: nothing was spent, and the summary must not invent turns.
	reportRun(agent, &loop.Result{}, gateErr, code, validator != nil, parsed.quiet, stderr, secrets.Redact)
	return code
}

// gateMCPErrors counts what the OAuth phase cost the run: the one declared
// dependency it could not equip. An INTERRUPTED phase counts nothing - a
// SIGTERM is not a server's unavailability, the same rule connectFailed
// applies.
func gateMCPErrors(gateErr error) int {
	if errors.Is(gateErr, mcp.ErrUnavailable) {
		return 1
	}
	return 0
}

// acquireRunLock enforces the single-flight contract of `lock: true`. It
// returns the release function and ExitOK when the run may proceed, or a
// no-op release and the exit code to return when it may not.
//
// CONTRACT: only `amele run` calls this. `validate`, `explain` and `chat`
// deliberately never lock - inspecting a config while a run is in progress is
// exactly when an operator needs those, and an interactive chat is not the
// unattended overlap this guards against.
func acquireRunLock(cfg *config.Config, configPath string, stderr io.Writer) (release func(), code int) {
	noop := func() {}
	if !cfg.Lock {
		return noop, ExitOK
	}
	lockPath, err := lockFilePath(configPath)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return noop, ExitConfigError
	}
	release, err = runlock.Acquire(lockPath)
	if err != nil {
		if errors.Is(err, runlock.ErrHeld) {
			// CONTRACT: exit 7, and a message a cron wrapper can act on. The
			// wording says "another run", not "error": for a cron job that
			// overran its interval this is the guard working as designed.
			_, _ = fmt.Fprintf(stderr, "another run holds the lock for this config (lock file: %s)\n", lockPath)
			return noop, ExitLockHeld
		}
		// A lock path that cannot be opened at all is a broken setup, not
		// contention: exit 2, so nothing reports a concurrent run that does
		// not exist.
		_, _ = fmt.Fprintln(stderr, err)
		return noop, ExitConfigError
	}
	return release, ExitOK
}

// lockFilePath returns the run lock file for a config: the config's absolute
// path plus a ".lock" suffix, i.e. a sibling of the config file.
//
// Absolute because cron and CI invoke amele from arbitrary working
// directories, and `amele run ./agent.yaml` must contend with
// `amele run /etc/amele/agent.yaml` when they are the same file. Next to the
// config rather than in a temp or state directory because it needs no
// environment lookup, needs no directory to exist first, and gives exactly
// the per-config granularity the lock is scoped to.
func lockFilePath(configPath string) (string, error) {
	abs, err := filepath.Abs(configPath)
	if err != nil {
		return "", fmt.Errorf("resolving config path %s: %w", configPath, err)
	}
	return abs + ".lock", nil
}

// chatPrompt is written to stderr before every input line. It lives on stderr
// so stdout stays the answer channel: `amele chat cfg.yaml < script.txt` still
// pipes cleanly.
const chatPrompt = "> "

// chatTaskLabel is the run_start task recorded for an interactive session.
// A chat has no single task string, but the session log's contract wants one
// per run - this makes chat sessions greppable in a session directory.
const chatTaskLabel = "interactive chat"

// cmdChat runs the interactive REPL: read a line, answer it, repeat until EOF
// (Ctrl-D). Config loading and validation are identical to `run` - the same
// YAML describes both modes - and so is the exit code contract.
func cmdChat(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer, env config.LookupEnv) int {
	parsed, parseCode := parseAgentArgs(ctx, env, "chat", usageChat, args, stderr)
	if parseCode != ExitOK {
		return parseCode
	}
	if parsed.help {
		return printHelp("chat", stdout, stderr)
	}
	if len(parsed.rest) > 0 {
		// A chat takes its input from stdin; free-form arguments are almost
		// certainly a `run` invocation that lost its verb, so say so instead
		// of silently ignoring them. Checked before the config is loaded so
		// the reported problem is the one the operator can act on.
		_, _ = fmt.Fprintf(stderr, "amele chat takes no task arguments (got %q); use `amele run` for a one-shot task\n", strings.Join(parsed.rest, " "))
		return ExitConfigError
	}
	if parsed.resumeSet {
		// The flag exists on this command only to make this sentence
		// possible. A conversation is continued by having it - the REPL keeps
		// its own history - and a chat log is refused by internal/resume
		// anyway, since "interactive chat" is not a task to finish.
		// CONTRACT: exit 2, like every other usage error, before anything is
		// loaded.
		_, _ = fmt.Fprintln(stderr, "chat has no --resume")
		return ExitConfigError
	}

	cfg, validator, err := loadAgentConfig(ctx, parsed, env)
	if err != nil {
		return reportLoadError(err, stderr)
	}
	// The compiled schema is deliberately NOT enforced, only reported.
	//
	// Why: output.schema is a one-shot contract - it exists so `amele run ... |
	// jq` can rely on stdout being exactly one JSON document. A conversation
	// has no single output to constrain, and enforcing the schema per line
	// would turn every "hi" into a validate-and-retry loop that spends the
	// session's budget arguing about JSON. Compiling it anyway (loadAgentConfig
	// always does) keeps `chat` honest about a broken config: the same YAML must
	// not pass here and fail under `run`. CONTRACT: exit 2 for a schema that
	// cannot compile.
	if validator != nil && !parsed.quiet {
		_, _ = fmt.Fprintln(stderr, "amele: output.schema is ignored in chat (it constrains a one-shot answer); use `amele run` to enforce it")
	}

	// One reader for the whole session. SECURITY-adjacent correctness point:
	// the REPL and the permission prompter consume the SAME stdin, so they
	// must share one buffer - two independent bufio readers would let an
	// approval read swallow the user's next chat line (or vice versa).
	lines := newLineReader(stdin)

	// ONE registry for the whole conversation (see runSecrets).
	secrets := runSecrets(cfg)

	agent, _, hints, err := buildAgent(cfg, nil, lines, stderr, secrets, parsed.quiet)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return ExitConfigError
	}
	if parsed.verbose {
		agent.Progress = progressLogger(stderr, secrets)
	}

	// The same pre-connect OAuth phase `run` applies, in the same place - before
	// run_start - and for the same reason: a conversation whose tools never
	// came up would mislead the human at the keyboard for its whole length,
	// and the question belongs before the REPL rather than in the middle of
	// it. A refusal still ends through the normal path, so a chat that never
	// opened is as auditable as a run that never started.
	if gateErr := mcpCredentialGate(ctx, cfg, parsed.configPath, lines, stderr, env, secrets, parsed.quiet); gateErr != nil {
		agent.Session.RunStart(cfg.Model, cfg.Provider.Identity(), chatTaskLabel)
		s := &chatSession{cfg: cfg, agent: agent, quiet: parsed.quiet,
			mcp: &mcpSet{failed: gateMCPErrors(gateErr)}, runCtx: ctx, secrets: secrets}
		return s.finish(stderr, exitCodeFor(gateErr), gateErr)
	}

	// A chat writes ONE run_start for the whole session, and the MCP servers
	// belong inside it like every other event (docs/contracts/jsonl-events.md).
	agent.Session.RunStart(cfg.Model, cfg.Provider.Identity(), chatTaskLabel)

	set, mcpErr := connectMCP(ctx, cfg, agent.Registry, agent.Session, stderr, env, parsed.quiet, version, secrets)
	maps.Copy(hints, set.hints)
	s := &chatSession{cfg: cfg, agent: agent, quiet: parsed.quiet, mcp: set, runCtx: ctx, secrets: secrets}
	if mcpErr != nil {
		// The conversation never starts: a chat whose tools are missing would
		// mislead the human at the keyboard for its whole length. A Ctrl-C
		// during the connect is reported as the interruption it is, exactly
		// like in `run`.
		if ctx.Err() != nil {
			mcpErr = interruptedError(ctx.Err())
		}
		return s.finish(stderr, exitCodeFor(mcpErr), mcpErr)
	}
	warnSanitizedToolSchemas(cfg, agent.Registry, stderr, parsed.quiet, secrets)

	return s.repl(ctx, lines, stdout, stderr)
}

// maxProgressLine bounds one rendered progress line, in bytes. The loop
// already clips the model-controlled fragments it embeds (currently 512 runes
// per argument, raised from 120 so by-value secret redaction can still match a
// full-length secret), so this is a backstop on the whole line rather than the
// primary cap: it keeps one event to roughly one screen line however the loop
// tunes its own bounds. Neither number is a promise to the user - the contract
// says events are clipped for readability, not clipped to N
// (docs/contracts/cli.md).
const maxProgressLine = 600

// progressLogger renders loop progress events to stderr for -v: one line per
// event, prefixed like every other note this binary writes. secrets is the
// run's shared secret registry (runSecrets).
//
// SECURITY, two independent hazards in the same model-chosen text:
//
//   - Secrets. -v is a persisted sink - a cron job's stderr lands in journald
//     or a mail spool - carrying tool arguments the model may have filled with
//     a credential it read. The session log redacts by value before writing;
//     this writer redacts through the SAME live registry, so adding an output
//     channel did not quietly add a leak channel. Redaction runs first,
//     before any clipping, so a secret cannot survive by being cut in two.
//   - Terminal control bytes. The event is routed through safeForTerminal
//     exactly like the approval question - an escape sequence in a tool
//     argument could otherwise erase the real question and redraw a
//     harmless-looking one (see safeForTerminal).
//
// stderr only: stdout is the answer channel, and -v must never change what a
// pipe receives.
func progressLogger(stderr io.Writer, secrets *session.SecretSet) func(string) {
	redact := secrets.Redact
	return func(event string) {
		_, _ = fmt.Fprintf(stderr, "amele: %s\n", safeForTerminal(redact(event), maxProgressLine))
	}
}

// chatSession owns one interactive conversation: the history handed to the
// loop on every line, and the cumulative accounting behind the shared budget
// pool and the closing summary.
type chatSession struct {
	cfg   *config.Config
	agent *loop.Loop
	// quiet drops the closing summary; the prompt, the errors and the
	// permission questions stay, because those are the conversation itself.
	quiet bool
	// mcp are the session's MCP servers, closed by finish before run_end.
	mcp *mcpSet
	// secrets is the conversation's shared secret registry - the same live set
	// the session log, the -v feed and the MCP relays redact through.
	secrets *session.SecretSet
	// runCtx is the session's context, kept only so finish can derive an
	// uncancellable one for the orderly MCP shutdown after a signal.
	runCtx context.Context //nolint:containedctx // session lifetime, not request scope

	// history is the conversation the caller owns. loop.RunMessages never
	// mutates it, so every turn is appended here explicitly.
	history []llm.Message

	turns     int
	toolCalls int
	tokens    int
	// cached is the cumulative prompt-cache read count across the session's
	// exchanges. It is a subset of tokens, tracked separately only so the
	// summary and run_end can report what the cache saved.
	cached int
	// fallbacks is the cumulative count of provider switches across the
	// session's exchanges. Each exchange reports only the switches IT made,
	// and the loop's position is sticky across them (loop.Loop.active), so the
	// sum is the number of moves the conversation made down the chain - which
	// is what run_end reports, like every other number here.
	fallbacks int
	duration  time.Duration
}

// repl drives the conversation until EOF, an error, or an exhausted budget,
// and returns the process exit code. It writes exactly one run_start/run_end
// pair and one summary line for the whole session.
func (s *chatSession) repl(ctx context.Context, lines *lineReader, stdout, stderr io.Writer) int {
	if s.cfg.SystemPrompt != "" {
		s.history = append(s.history, llm.Message{Role: llm.RoleSystem, Content: s.cfg.SystemPrompt})
	}

	for {
		_, _ = fmt.Fprint(stderr, chatPrompt)
		line, readErr := readAsync(ctx, lines.ReadLine)
		if readErr != nil && !errors.Is(readErr, io.EOF) {
			// Ctrl-C at the prompt, a deadline inherited from the caller, or a
			// stdin that broke. None of them can be answered by asking again.
			// CONTRACT: an operator interrupt is exit 1, not the budget code -
			// monitoring must not read a Ctrl-C as a budget overrun.
			_, _ = fmt.Fprintln(stderr)
			code := ExitTaskFailed
			if errors.Is(readErr, context.DeadlineExceeded) {
				code = ExitBudgetExceeded
			}
			return s.finish(stderr, code, fmt.Errorf("chat interrupted: %w", readErr))
		}

		// A bare Enter costs nothing: no provider call, no turn, no tokens.
		// CONTRACT (docs/contracts/cli.md, chat): only a whitespace-ONLY line is
		// free. The emptiness test therefore runs on a trimmed COPY and the line
		// is sent as typed - trimming it in place silently reindented pasted
		// code, and leading whitespace can be the whole point of a message.
		if strings.TrimSpace(line) != "" {
			answer, err := s.nextTurn(ctx, line)
			if err != nil {
				return s.finish(stderr, exitCodeFor(err), err)
			}
			// CONTRACT: stdout carries only the model's answers - each one
			// followed by a newline, nothing else. It is a STREAM, not a
			// record format: a final answer routinely spans several lines, so
			// a consumer must not assume one line per answer (there is
			// deliberately no delimiter; use `amele run` when a scripted
			// consumer needs a parseable boundary).
			_, _ = fmt.Fprintln(stdout, answer)
		}

		if readErr != nil { // io.EOF: Ctrl-D or the end of a scripted session.
			_, _ = fmt.Fprintln(stderr)
			return s.finish(stderr, ExitOK, nil)
		}
	}
}

// nextTurn sends one user line through the loop and appends the exchange to
// the history. The returned string is the model's final answer.
func (s *chatSession) nextTurn(ctx context.Context, line string) (string, error) {
	if err := s.applyBudget(); err != nil {
		return "", err
	}

	// limits.timeout bounds ONE exchange here, not the session: a human
	// thinking at the prompt must not burn the run timeout, and a chat that
	// self-destructs after `timeout` of wall clock would be useless.
	if s.cfg.Limits.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, s.cfg.Limits.Timeout.Std())
		defer cancel()
	}

	s.history = append(s.history, llm.Message{Role: llm.RoleUser, Content: line})
	res, err := s.agent.RunMessages(ctx, s.history)

	// Accounting first, unconditionally: the summary and run_end must never
	// under-report what a failed exchange actually spent.
	s.turns += res.Turns
	s.toolCalls += res.ToolCalls
	s.tokens += res.Usage.Total()
	s.cached += res.Usage.CacheReadTokens
	s.fallbacks += res.Fallbacks
	s.duration += res.Duration
	if err != nil {
		return "", err
	}

	// Only the final answer re-enters the history: the tool-call rounds that
	// happened inside RunMessages are dropped, because RunMessages does not
	// hand back the messages it appended. The model therefore sees its own
	// conclusions but not its scratch work - enough for a coherent
	// conversation, and it keeps the context small. Full transcript
	// continuity is a Phase 3 candidate (RunMessages would have to return the
	// grown history).
	s.history = append(s.history, llm.Message{Role: llm.RoleAssistant, Content: res.FinalText})
	return res.FinalText, nil
}

// applyBudget charges the session's cumulative spend against the configured
// limits and arms the loop with what is left.
//
// CONTRACT: a chat session shares ONE budget pool across all lines. limits are
// per-RunMessages-call in the loop, so "remaining = configured - consumed" is
// what turns them into a session-wide ceiling; an exhausted pool is exit 3
// exactly like a one-shot overrun.
func (s *chatSession) applyBudget() error {
	turns := s.cfg.Limits.MaxTurns - s.turns
	if turns <= 0 {
		return fmt.Errorf("%w: max_turns (%d) is spent for this chat session", loop.ErrBudgetExceeded, s.cfg.Limits.MaxTurns)
	}
	// 0 means "no token limit" both in the config and in loop.Limits, so the
	// remaining budget stays 0 when none was configured - subtracting into a
	// negative number would silently disable the check.
	tokens := 0
	if s.cfg.Limits.MaxTokens > 0 {
		tokens = s.cfg.Limits.MaxTokens - s.tokens
		if tokens <= 0 {
			return fmt.Errorf("%w: max_tokens (%d) is spent for this chat session", loop.ErrBudgetExceeded, s.cfg.Limits.MaxTokens)
		}
	}
	s.agent.Limits = loop.Limits{MaxTurns: turns, MaxTokens: tokens}
	// CONTRACT: the session log numbers turns continuously across the whole
	// chat. The loop counts this call's turns from 1 (that is what the budget
	// above bounds), so the already-consumed turns become the logging offset -
	// otherwise every line would restart at turn 1 inside one run_start/run_end
	// pair and run_end.turns would disagree with the events.
	s.agent.TurnBase = s.turns
	return nil
}

// finish closes the session log and prints the one-line cumulative summary,
// returning the exit code it was given.
func (s *chatSession) finish(stderr io.Writer, code int, err error) int {
	// CONTRACT: the mcp_disconnect events precede run_end. WithoutCancel so a
	// Ctrl-C at the prompt still buys the servers their orderly close.
	s.mcp.close(context.WithoutCancel(s.runCtx))
	s.agent.Session.SetMCPErrors(s.mcp.errors())
	status := "success"
	if err != nil {
		status = "error"
		// SECURITY: same rule as reportRun - the error may quote remote text.
		_, _ = fmt.Fprintln(stderr, s.secrets.Redact(err.Error()))
	}
	s.agent.Session.RunEnd(session.RunEnd{
		Status: status, ExitCode: code,
		Turns: s.turns, ToolCalls: s.toolCalls,
		TotalTokens: s.tokens, CacheReadTokens: s.cached,
		Fallbacks: s.fallbacks,
		Duration:  s.duration,
	})
	if !s.quiet {
		_, _ = fmt.Fprintln(stderr, session.Summary(err == nil, session.Stats{
			Turns: s.turns, ToolCalls: s.toolCalls, TotalTokens: s.tokens,
			CachedTokens: s.cached, Fallbacks: s.fallbacks, Duration: s.duration,
		}))
	}
	return code
}

// readAsync runs read in a goroutine and returns its result, or ctx's error if
// the context ends first.
//
// It exists because an io.Reader blocked on a terminal or an open pipe cannot
// be interrupted: without this, Ctrl-C would cancel the context and then sit at
// the chat prompt until the user pressed Enter anyway, and a pipe that never
// delivers data would outlive limits.timeout. On cancellation the goroutine
// stays blocked until the process exits, which is imminent - every caller turns
// a ctx error into an immediate exit code. That owned-and-bounded trade-off is
// stated once here rather than re-argued at each read.
func readAsync[T any](ctx context.Context, read func() (T, error)) (T, error) {
	type result struct {
		value T
		err   error
	}
	// Buffered so the abandoned goroutine can always deliver and exit.
	ch := make(chan result, 1)
	go func() {
		value, err := read()
		ch <- result{value, err}
	}()

	select {
	case <-ctx.Done():
		var zero T
		return zero, ctx.Err()
	case r := <-ch:
		return r.value, r.err
	}
}

// maxChatLineBytes caps one input line. A pasted megabyte is a plausible chat
// turn (a stack trace, a config file); a gigabyte is a runaway process piping
// into amele, and the excess is dropped rather than allocated - same stance as
// the 10MB cap on piped `run` input.
const maxChatLineBytes = 1 << 20

// lineReader is the single line-oriented view of this process's stdin.
//
// It exists because two consumers read the same stream: the chat REPL and the
// permission prompter. Each holding its own bufio.Reader would be a data-loss
// bug - whichever read first would buffer bytes the other one needed. The
// original reader is kept so the TTY check still inspects the real file.
type lineReader struct {
	src io.Reader
	buf *bufio.Reader
}

// newLineReader wraps r. The wrapper must be created once per stream.
func newLineReader(r io.Reader) *lineReader {
	return &lineReader{src: r, buf: bufio.NewReader(r)}
}

// IsTerminal reports whether the underlying stream is an interactive terminal.
// It goes through the stdinIsTerminal seam so the interactive paths can be
// exercised without a pty; the production value is isTerminal itself.
func (l *lineReader) IsTerminal() bool { return stdinIsTerminal(l.src) }

// ReadLine returns the next line without its trailing newline (CRLF tolerated).
// A final line with no newline is returned together with io.EOF, so callers
// must handle the value before the error. Lines longer than maxChatLineBytes
// are truncated and the remainder is discarded - never re-served as if it were
// the next line the user typed.
func (l *lineReader) ReadLine() (string, error) {
	var b strings.Builder
	for {
		// ReadSlice (not ReadString) is what makes the cap real: it returns
		// at most one buffer at a time, so an unbounded line is consumed in
		// pieces we are free to drop instead of accumulating.
		chunk, err := l.buf.ReadSlice('\n')
		if room := maxChatLineBytes - b.Len(); room > 0 {
			if len(chunk) > room {
				chunk = chunk[:room]
			}
			b.Write(chunk)
		}
		if errors.Is(err, bufio.ErrBufferFull) {
			continue // the line is longer than the buffer; keep consuming
		}
		return strings.TrimRight(b.String(), "\r\n"), err
	}
}

// agentSecrets lists every value this run must never emit verbatim.
//
// SECURITY: every interpolated environment value (not just the API key - DB
// passwords in prompts count too) is registered, because a secret reaches an
// output channel through arbitrary paths: tool output, model echoes, or the
// model putting it back into the next call's arguments. Deliberate trade-off:
// interpolating a non-secret like ${HOME} redacts every path in the log -
// documented in docs/session-logging.md.
//
// It is the run's STARTING list; values a run mints later (an OAuth access
// token) are registered on the same live set through runSecrets below.
func agentSecrets(cfg *config.Config) []string {
	secrets := append(cfg.InterpolatedSecrets(), cfg.Provider.APIKey)
	// SECURITY: a fallback's credential is a credential. Every entry of the
	// chain is registered even though the run may never reach it: the list is
	// built before the first call, and a target the run DOES switch to would
	// otherwise echo its own key back through an error body into an
	// unprotected log. The ${VAR} form is already covered by
	// InterpolatedSecrets above; this covers the literal, exactly as the
	// primary's own key is covered on the line above.
	for i := range cfg.Provider.Fallback {
		secrets = append(secrets, cfg.Provider.Fallback[i].APIKey)
	}
	// An MCP header is a COMPOSED value ("Bearer " + ${TOKEN}): the
	// environment value alone is already in the list above, but the assembled
	// header is what a server echoes back in an error, so it is registered as
	// a secret in its own right.
	return append(secrets, cfg.MCPHeaderSecrets()...)
}

// runSecrets builds the one secret registry a single `run`, `chat` or
// `explain` invocation uses.
//
// SECURITY: exactly one *session.SecretSet per invocation, seeded with
// agentSecrets and shared by every sink (session JSONL, the -v progress feed,
// the MCP stderr relays, the error lines). Two sets would mean two answers to
// "is this value a secret", and the sink holding the shorter one prints the
// credential. Because the set is live rather than a snapshot, a token minted
// mid-run is scrubbed from every sink the moment it is registered.
func runSecrets(cfg *config.Config) *session.SecretSet {
	return session.NewSecretSet(agentSecrets(cfg))
}

// sessionOptions translates the config's session-logging keys into the
// writer's options. It exists as its own function because the two layers spell
// "no bound" differently and the translation must live in exactly one place:
//
//	config (YAML)                     session.Options
//	limits.max_logged_field absent    0   -> the package default (8192)
//	limits.max_logged_field: 0        -1  -> unbounded, nothing is clipped
//	limits.max_logged_field: n        n   -> n bytes
//
// CONTRACT: the middle row is the whole reason this is not an assignment.
// Copying the number across (opts.MaxLoggedField = *cfg...) would turn the
// operator's explicit "log everything" into the 8 KiB default, silently and
// with every unit test still green - the config says 0 means unbounded
// (docs/contracts/config.schema.json), while a zero-valued Options field can
// only mean "this caller never heard of the option".
func sessionOptions(cfg *config.Config, secrets *session.SecretSet) session.Options {
	opts := session.Options{SecretSource: secrets, LogReasoning: cfg.LogReasoning}
	if cfg.Limits.MaxLoggedField != nil {
		if v := *cfg.Limits.MaxLoggedField; v == 0 {
			opts.MaxLoggedField = -1
		} else {
			opts.MaxLoggedField = v
		}
	}
	return opts
}

// buildAgent assembles the loop for one run: tools, permissions, session
// logging, provider and - when validator is non-nil - structured output
// enforcement. lines and stderr are the terminal the permission approver asks
// on; they wrap the process streams in production and buffers in tests.
//
// It also returns the function that turns the finished Result into the text
// for stdout. That indirection is the point: in schema mode stdout must carry
// the CANONICAL JSON the validator accepted, which only the validator closure
// below ever sees. Result.FinalText is the raw model reply and may be wrapped
// in a ```json fence or padded with prose, so it is not printable.
//
// quiet is only consulted for the print_session_path note: it is a note like
// every other, and -q means errors only (docs/contracts/cli.md).
func buildAgent(cfg *config.Config, validator *schema.Validator, lines *lineReader, stderr io.Writer,
	secrets *session.SecretSet, quiet bool) (*loop.Loop, func(*loop.Result) string, map[string]string, error) {
	registry, err := buildRegistry(cfg)
	if err != nil {
		return nil, nil, nil, err
	}

	// The hint map is handed out EMPTY and filled later by connectMCP: the MCP
	// servers may only be started after run_start is in the session log
	// (docs/contracts/jsonl-events.md ordering), which is after this function
	// returns. Nothing reads it until the first tool call, well after that.
	hints := map[string]string{}
	approve, err := buildApprover(cfg, lines, stderr, hints)
	if err != nil {
		return nil, nil, nil, err
	}

	var sess *session.Writer
	if cfg.SessionDir != "" {
		sess, err = session.New(cfg.SessionDir, sessionOptions(cfg, secrets))
		if err != nil {
			return nil, nil, nil, err
		}
		// The note names the file this run is writing, so a human who did not
		// choose the timestamped name can `tail -f` it. It is a note, so -q
		// (errors only) drops it like every other.
		//
		// SECURITY: printed verbatim, deliberately. The path is session_dir
		// from the operator's own config plus a timestamp this process
		// generated - nothing the model or a tool produced can reach this
		// line. A session_dir written as ${VAR} is registered as a secret and
		// so IS redacted inside the log; redacting it here too would print
		// "[REDACTED]/run-....jsonl", which is exactly the one thing this note
		// exists to avoid. The operator asked, in their own config, to be
		// shown their own directory on their own terminal.
		if cfg.PrintSessionPath && !quiet {
			_, _ = fmt.Fprintf(stderr, "session log: %s\n", sess.Path())
		}
	}

	provider, err := buildProviderFrom(&cfg.Provider, secrets.Add)
	if err != nil {
		return nil, nil, nil, err
	}
	tuning, err := providerTuningFrom(&cfg.Provider)
	if err != nil {
		return nil, nil, nil, err
	}
	backends, err := buildFallbacks(cfg, secrets.Add)
	if err != nil {
		return nil, nil, nil, err
	}

	agent := &loop.Loop{
		Provider: provider,
		Registry: registry,
		Session:  sess,
		Approve:  approve,
		// The concurrency gate: the loop may only overlap calls it knows
		// nobody will be asked about (docs/features.md, "Parallel tool calls").
		AutoApprove:   perm.AutoApproves(cfg.Permissions),
		ParallelTools: cfg.Tools.IsParallel(),
		Limits:        loop.Limits{MaxTurns: cfg.Limits.MaxTurns, MaxTokens: cfg.Limits.MaxTokens},
		// The last cut before the transcript: each family already caps what
		// it produces, and this bounds the finished result - after the
		// family's own cap and after whatever framing it added, which is
		// what a per-stream cap alone cannot bound. Its reach is the output
		// of a tool that RAN, MCP included; a dispatch that FAILED is
		// reported to the model as the loop's own `error: <message>` text
		// and does not pass through the ceiling (internal/loop.runCall).
		MaxToolResultBytes: toolResultCap(cfg),
		Model:              cfg.Model,
		Identity:           cfg.Provider.Identity(),
		SystemPrompt:       cfg.SystemPrompt,
		Tuning:             tuning,
		Fallbacks:          backends,
	}

	if validator == nil {
		// Plain-text mode: whatever the model said is the answer.
		return agent, func(res *loop.Result) string { return res.FinalText }, hints, nil
	}

	// Native structured output where the provider supports it (the client
	// falls back on its own when it does not); validate+retry is the actual
	// enforcement either way.
	agent.ResponseFormat = &llm.ResponseFormat{Name: responseFormatName, Schema: validator.JSON()}

	var canonical string
	agent.FinalValidator = func(text string) (string, bool) {
		extracted, feedback, ok := validator.Validate(text)
		if ok {
			canonical = extracted
		}
		return feedback, ok
	}
	agent.MaxFinalRejections = cfg.Output.MaxSchemaRetries
	if agent.MaxFinalRejections == 0 {
		agent.MaxFinalRejections = defaultMaxSchemaRetries
	}
	// CONTRACT: in schema mode stdout carries ONLY JSON that passed the
	// schema. The loop returns a nil error exactly when the validator accepted
	// an answer, so canonical is always set by the time this is called.
	return agent, func(*loop.Result) string { return canonical }, hints, nil
}

// buildFallbacks builds one backend per provider.fallback entry, in the order
// the file lists them. Nil for a config without the key, which is what the loop
// reads as "there is no chain".
//
// Every entry is built HERE, before the run starts, rather than lazily at the
// moment of a switch: a mistyped dialect in the third entry must fail the run
// while it has cost nothing, not two minutes in when the primary has already
// gone down and the fallback is the only thing left. That also guarantees the
// loop never sees a backend with a nil Provider - the one shape that would turn
// a failover into a panic.
//
// registerSecret is the run's live redactor sink, passed to each entry for the
// same reason the primary gets it: a target that mints a credential (the Vertex
// path) must register it in the one set every sink reads.
func buildFallbacks(cfg *config.Config, registerSecret func(...string)) ([]loop.Backend, error) {
	if len(cfg.Provider.Fallback) == 0 {
		return nil, nil
	}
	backends := make([]loop.Backend, 0, len(cfg.Provider.Fallback))
	for i := range cfg.Provider.Fallback {
		entry := &cfg.Provider.Fallback[i]
		provider, err := buildProviderFrom(&entry.ProviderConfig, registerSecret)
		if err != nil {
			return nil, fmt.Errorf("provider.fallback[%d]: %w", i, err)
		}
		tuning, err := providerTuningFrom(&entry.ProviderConfig)
		if err != nil {
			return nil, fmt.Errorf("provider.fallback[%d]: %w", i, err)
		}
		backends = append(backends, loop.Backend{
			Provider: provider,
			// The entry's own model, never the top-level one: a backup
			// endpoint rarely serves the primary's model
			// (config.FallbackTarget.Model).
			Model:    entry.Model,
			Tuning:   tuning,
			Identity: entry.Identity(),
		})
	}
	return backends, nil
}

// buildProviderFrom constructs the LLM client selected by ONE target's
// provider.type - the primary block or one entry of provider.fallback, which
// is a complete provider block of its own (config.FallbackTarget embeds
// ProviderConfig). It takes the target rather than the whole config so the
// chain's entries are built by exactly the code that builds the primary: a
// backup that got a different client than the file describes would only be
// discovered during an outage.
//
// Validate has already constrained the type, so anything that is neither
// anthropic nor gemini is the OpenAI-compatible default - including "", which
// is what every pre-Type config carries.
//
// ResponseFormat is deliberately NOT decided here: buildAgent sets it on the
// loop for every provider, and each client maps it to its own wire spelling -
// response_format:json_schema on the OpenAI-compatible path,
// output_config.format on the Anthropic one, both GA. A client whose endpoint
// rejects that field repeats the call once without it and reports the
// degradation through Response.SchemaEnforcementDropped, which run surfaces as
// a warning. Special-casing it per provider in cmd would duplicate a
// capability decision the clients already own.
//
// The dialect is parsed here rather than stored parsed on the config: config
// validates the spelling and keeps the file's own string, so this is the one
// place that turns it into the wire mapping. A parse failure is impossible for
// a validated config, which is why it is WRAPPED and returned rather than
// panicked on or ignored - falling back to the openai mapping would silently
// reshape every request of the run.
//
// registerSecret is the run's live redactor sink (secrets.Add). It is a
// parameter rather than a package-level hook because only ONE set exists per
// invocation (runSecrets) and a client that minted credentials into a second
// one would be writing them into a registry no sink reads. nil is allowed for
// callers that keep no log; only the Vertex credential path uses it today.
func buildProviderFrom(p *config.ProviderConfig, registerSecret func(...string)) (llm.Provider, error) {
	maxAttempts, initialBackoff := retryPolicy(p.Retry)
	if p.Type == config.ProviderTypeAnthropic {
		// The dialect names a variation of the OpenAI-compatible wire and is
		// documented as ignored here (config.schema.json), so it is not parsed
		// on this path: a leftover dialect must not fail a run that never
		// speaks it.
		return &llm.AnthropicClient{
			BaseURL:         p.BaseURL,
			APIKey:          p.APIKey,
			RequestTimeout:  p.RequestTimeout.Std(),
			MaxAttempts:     maxAttempts,
			InitialBackoff:  initialBackoff,
			MaxOutputTokens: p.MaxOutputTokens,
			PromptCache:     promptCacheOn(p),
		}, nil
	}
	if p.Type == config.ProviderTypeGemini {
		// The dialect is not parsed here either, for a stronger reason than on
		// the anthropic path: a dialect with type gemini is a validate ERROR
		// (internal/config.tuningDialect), so this line is unreachable with one
		// - and parsing it would trade that clear message for a vocabulary
		// complaint about a key that has to go.
		//
		// max_output_tokens, reasoning, sampling and params are absent on
		// purpose: they travel per request through loop.Tuning, exactly as on
		// the openai wire. Only the Messages API needs its cap on the client,
		// because it requires the field on every request.
		return &llm.GeminiClient{
			BaseURL: p.BaseURL,
			APIKey:  p.APIKey,
			// The vertex block travels as the client's target so the request is
			// addressed to the endpoint the config names, and as the credential
			// source that authenticates it. Both are nil without the block,
			// which is what keeps the AI Studio path a keyed one.
			Vertex:         vertexTarget(p.Vertex),
			TokenSource:    vertexTokenSource(p.Vertex, registerSecret),
			RequestTimeout: p.RequestTimeout.Std(),
			MaxAttempts:    maxAttempts,
			InitialBackoff: initialBackoff,
		}, nil
	}
	dialect, err := llm.ParseDialect(p.Dialect)
	if err != nil {
		return nil, fmt.Errorf("provider.dialect: %w", err)
	}
	return &llm.OpenAIClient{
		BaseURL:        p.BaseURL,
		APIKey:         p.APIKey,
		Dialect:        dialect,
		RequestTimeout: p.RequestTimeout.Std(),
		MaxAttempts:    maxAttempts,
		InitialBackoff: initialBackoff,
		// Opt-in here, unlike the anthropic wire: only an explicit true asks
		// the gateway for its caching (config.ProviderConfig.PromptCache).
		// The client applies it on the openrouter dialect alone.
		PromptCache: p.PromptCache != nil && *p.PromptCache,
	}, nil
}

// promptCacheOn resolves provider.prompt_cache into the client's plain bool.
//
// CONTRACT: an absent key means ON. The pointer exists so that "the file said
// nothing" and "the file said false" stay distinguishable in the config, and
// this is the single place the distinction is spent: nil and true both ask for
// the cache_control markers, false asks for the pre-v0.3 request bytes.
//
// Called only on the anthropic branch of buildProviderFrom, for whichever
// target that branch is building. The other wires cache on their own and
// validate refuses the key there, so translating it for them would describe a
// request field neither client writes.
func promptCacheOn(p *config.ProviderConfig) bool {
	return p.PromptCache == nil || *p.PromptCache
}

// vertexTarget translates the optional provider.vertex block into the client's
// target. A nil block means the AI Studio backend, which is what every gemini
// config written before the block carries.
//
// The two types are separate on purpose: the config block is the operator's
// YAML surface (and carries the credentials PATH, which the auth layer reads),
// while the target is the pair of coordinates the endpoint is built from.
func vertexTarget(v *config.VertexConfig) *llm.VertexTarget {
	if v == nil {
		return nil
	}
	return &llm.VertexTarget{Project: v.Project, Location: v.Location}
}

// vertexTokenSource builds the credential source for a vertex config: the
// service-account key file when provider.vertex.credentials names one, the
// Application Default Credentials chain otherwise.
//
// The return type is the INTERFACE and the nil case returns a literal nil, not
// a typed nil pointer: assigning a (*llm.GoogleTokenSource)(nil) to the
// client's field would produce a non-nil interface holding a nil pointer, and
// the AI Studio path - which checks that field for nil - would then call
// through it.
//
// SECURITY: registerSecret is the run's live SecretSet.Add, so a token minted
// during turn seven is scrubbed from the session log, the -v feed and the error
// lines from the moment it exists. This is the same wiring the MCP OAuth
// handler uses (mcp.Deps.RegisterSecret).
func vertexTokenSource(v *config.VertexConfig, registerSecret func(...string)) llm.GeminiTokenSource {
	if v == nil {
		return nil
	}
	return &llm.GoogleTokenSource{
		CredentialsFile: v.Credentials,
		Project:         v.Project,
		Register:        registerSecret,
	}
}

// retryPolicy unpacks the optional provider.retry block for both clients. An
// absent block (and a block that leaves a knob at zero) yields zero values,
// which each client reads as "my default": the wiring never invents a number,
// so the defaults live in exactly one place - the llm package.
func retryPolicy(r *config.RetryConfig) (maxAttempts int, initialBackoff time.Duration) {
	if r == nil {
		return 0, 0
	}
	return r.MaxAttempts, r.InitialBackoff.Std()
}

// providerTuningFrom translates ONE target's provider knobs into the neutral
// request fields the loop forwards on every turn. Each entry of the fallback
// chain carries its own - a backup on a different vendor needs its own
// reasoning level and its own params - so the tuning travels with the backend
// rather than being read off the primary once.
//
// CONTRACT: this is the ONE place where provider.params (arbitrary YAML)
// becomes JSON. Validate already proved the map is serializable and collides
// with no field amele owns, so a failure here is not reachable through a
// validated config - it is wrapped rather than ignored because a silently
// dropped params map would leave the run missing a knob the file asked for.
func providerTuningFrom(p *config.ProviderConfig) (loop.Tuning, error) {
	extra, err := paramsJSON(p.Params)
	if err != nil {
		return loop.Tuning{}, fmt.Errorf("provider.params: %w", err)
	}
	return loop.Tuning{
		MaxOutputTokens: p.MaxOutputTokens,
		Reasoning:       reasoningSpec(p.Reasoning),
		Temperature:     p.Temperature,
		TopP:            p.TopP,
		Extra:           extra,
	}, nil
}

// reasoningSpec converts the config's reasoning block into the neutral spec,
// or nil when the config asks for no reasoning knob at all.
//
// An EMPTY block counts as no block. `--set provider.reasoning.effort=` on a
// config whose YAML carried a reasoning block leaves exactly that shape behind
// (internal/config.overrideReasoningEffort), and it means "back to the provider
// default" - so it must produce no ReasoningSpec, not one the clients would
// have to interpret as "unset" a second time.
func reasoningSpec(r *config.ReasoningConfig) *llm.ReasoningSpec {
	if r == nil || (r.Effort == "" && r.BudgetTokens == 0) {
		return nil
	}
	return &llm.ReasoningSpec{Effort: r.Effort, BudgetTokens: r.BudgetTokens}
}

// paramsJSON pre-serializes provider.params so the llm package never re-encodes
// user YAML: the clients merge these bytes into the request body root verbatim.
// A nil or empty map yields nil - "there is nothing to merge" - rather than an
// empty map the clients would have to special-case.
func paramsJSON(params map[string]any) (map[string]json.RawMessage, error) {
	if len(params) == 0 {
		return nil, nil
	}
	out := make(map[string]json.RawMessage, len(params))
	for key, value := range params {
		encoded, err := json.Marshal(value)
		if err != nil {
			return nil, fmt.Errorf("key %q: %w", key, err)
		}
		out[key] = encoded
	}
	return out, nil
}

// maxPromptArgs caps how much of a tool call's JSON arguments is shown in the
// approval question. Long enough to recognize the call (a path, a URL, the
// head of a command), short enough that a 100KB fs_write payload cannot scroll
// the question itself off the screen - the human must still see what they are
// approving.
const maxPromptArgs = 200

// maxToolName caps the tool name shown to the operator. Honest names are
// already bounded to 64 characters by the config schema (toolNameRe), but the
// name in a tool call comes from the model, not from the config - the loop
// consults the approver BEFORE the registry lookup, so an unregistered name
// reaches the terminal.
const maxToolName = 64

// maxPromptHint caps the annotation shown next to the approval question. It is
// server-controlled prose, so it must never be long enough to push the "[y/N]"
// off the line the human is reading.
const maxPromptHint = 120

// clipMarker marks text that safeForTerminal shortened.
const clipMarker = "... (clipped)"

// buildApprover wires the permission profile to this process's terminal.
//
// The policy itself lives in internal/perm; everything terminal-shaped is
// injected from here, so perm stays deterministic and testable and the TTY
// detection has exactly one implementation (isTerminal).
func buildApprover(cfg *config.Config, lines *lineReader, stderr io.Writer, hints map[string]string) (loop.Approver, error) {
	return perm.NewApprover(cfg.Permissions, perm.Options{
		// What an MCP server said about the tool it published (read-only,
		// destructive). SECURITY: advisory only - it is shown to the human,
		// never consulted by the policy (docs/threat-model.md S9).
		Hint: func(toolName string) string { return hints[toolName] },
		// Evaluated per call rather than once: nothing here caches a fact
		// about the process that a caller might have changed.
		IsTTY:  lines.IsTerminal,
		Prompt: newPrompter(lines, stderr),
		// The loop already reports the denial to the model as a tool result;
		// this note tells the *operator* why it happened. It goes to stderr
		// because stdout is reserved for the final answer (pipe contract).
		// SECURITY: the tool name is attacker-influenced text (see
		// safeForTerminal) and the audit note is written to the same terminal
		// as the approval question, so it goes through the same sanitizer -
		// a note is otherwise just as good a place to forge a question.
		Log: func(toolName, decision string) {
			_, _ = fmt.Fprintf(stderr, "amele: tool %s: %s\n", safeForTerminal(toolName, maxToolName), decision)
		},
	})
}

// newPrompter returns the interactive approval question for an "ask" policy.
// The reader is injected (rather than os.Stdin captured here) so tests can
// script the answers, and so the interactive `chat` command hands it the very
// same lineReader it reads user turns from - one buffer, no stolen bytes.
//
// SECURITY: only an explicit "y"/"yes" approves. Everything else - a blank
// line, an unrecognized word, or EOF (Ctrl-D, or a closed stdin) - is a
// refusal, so an accidental Enter never grants a tool.
func newPrompter(lines *lineReader, stderr io.Writer) func(toolName, args, hint string) (bool, error) {
	return func(toolName, args, hint string) (bool, error) {
		// SECURITY: the hint is remote text (an MCP server's annotation), so
		// it goes through the same sanitizer as the name and the arguments -
		// otherwise it would be the easiest place to forge a second question.
		if hint != "" {
			_, _ = fmt.Fprintf(stderr, "amele: allow tool %s with %s? (%s) [y/N] ",
				safeForTerminal(toolName, maxToolName), safeForTerminal(args, maxPromptArgs),
				safeForTerminal(hint, maxPromptHint))
		} else {
			_, _ = fmt.Fprintf(stderr, "amele: allow tool %s with %s? [y/N] ",
				safeForTerminal(toolName, maxToolName), safeForTerminal(args, maxPromptArgs))
		}
		line, err := lines.ReadLine()
		// EOF is an answer ("no human left to ask"), not a failure; any other
		// read error means we never got one, and perm turns that into an abort.
		if err != nil && !errors.Is(err, io.EOF) {
			return false, fmt.Errorf("reading approval answer: %w", err)
		}
		switch strings.ToLower(strings.TrimSpace(line)) {
		case "y", "yes":
			return true, nil
		default:
			return false, nil
		}
	}
}

// safeForTerminal renders provider-controlled text - tool names and tool call
// arguments, both decoded straight out of the model's JSON - for the
// operator's terminal.
//
// SECURITY: without this, a prompt-injected model can forge the approval
// dialog. A tool named
//
//	"fs_write\x1b[2K\ramele: allow tool fs_read with {\"path\":\"README.md\"}? [y/N] "
//
// erases the real question with an escape sequence and redraws a harmless
// looking one, so the operator types "y" believing they are allowing a read
// and actually approves the write. A merely enormous name is the same attack
// in blunt form: it scrolls the real question off the screen. Both render
// paths (the question and the stderr audit note) go through this single
// helper so they cannot drift apart.
//
// The strip is deliberately total - every byte below 0x20 (newline and tab
// included) and DEL - because nothing in a tool name or a JSON argument
// string needs a control byte to be readable, and a partial allowlist is how
// this class of bug comes back.
//
// C0 alone was not the whole class, so two more groups go with it:
//
//   - C1 controls, U+0080–U+009F. A terminal in an 8-bit mode reads U+009B as
//     CSI - the single character form of "\x1b[" - so the spoof above works
//     verbatim without ever containing an ESC byte.
//   - Bidi formatting, U+202A–U+202E (embeddings and overrides) and
//     U+2066–U+2069 (isolates). They do not move the cursor; they reorder what
//     the operator READS, so "fs_read" followed by an override can render as a
//     completely different call than the one about to be approved. Same lie,
//     told by layout instead of by escape sequence.
//
// Everything else printable is kept: an honest name or path may legitimately
// be non-ASCII, and mangling it would train operators to ignore the question.
func safeForTerminal(s string, max int) string {
	stripped := strings.Map(func(r rune) rune {
		switch {
		case r < 0x20, r == 0x7f: // C0 and DEL
			return -1
		case r >= 0x80 && r <= 0x9f: // C1, U+009B (CSI) included
			return -1
		case r >= 0x202a && r <= 0x202e, // bidi embeddings and overrides
			r >= 0x2066 && r <= 0x2069: // bidi isolates
			return -1
		}
		return r
	}, s)
	// Clip AFTER stripping so the cap bounds what is actually displayed.
	return clip(stripped, max)
}

// clip shortens s to at most max bytes, marking the cut. ToValidUTF8 drops a
// rune the cut may have split in half, so the terminal never gets a mangled
// byte sequence.
func clip(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return strings.ToValidUTF8(s[:max], "") + clipMarker
}

// compileOutputSchema compiles cfg's output.schema, returning (nil, nil) when
// the config declares no schema. Errors are the caller's exit-2 path.
func compileOutputSchema(cfg *config.Config) (*schema.Validator, error) {
	raw, err := cfg.SchemaJSON()
	if err != nil || raw == nil {
		return nil, err
	}
	return schema.Compile(raw)
}

// exitCodeFor maps run errors onto the frozen exit code table.
func exitCodeFor(err error) int {
	switch {
	case err == nil:
		return ExitOK
	case errors.Is(err, loop.ErrBudgetExceeded):
		return ExitBudgetExceeded
	case errors.Is(err, loop.ErrPermissionDenied):
		return ExitPermissionDenied
	case errors.Is(err, llm.ErrProvider):
		return ExitProviderError
	case errors.Is(err, mcp.ErrUnavailable):
		// CONTRACT: exit 8 - a declared dependency is missing. Checked before
		// the default so it can never be reported as a plain task failure.
		return ExitMCPUnavailable
	case errors.Is(err, mcp.ErrToolset):
		// A tool name collision is a config mistake, not a runtime failure:
		// the same YAML would fail identically on every retry.
		return ExitConfigError
	case errors.Is(err, loop.ErrOutputRejected):
		// CONTRACT: the model answered, but never within output.schema -
		// exit 6, distinct from a task failure (1) so pipelines can tell
		// "wrong shape" from "could not do the job".
		return ExitSchemaUnmet
	default:
		return ExitTaskFailed
	}
}

// buildTask renders the user message from CLI args, stdin and the optional
// prompt template ({{args}} / {{input}}). ctx bounds the stdin read: the
// caller arms the run timeout first, so even a never-closing pipe cannot
// hang the process beyond limits.timeout.
func buildTask(ctx context.Context, cfg *config.Config, taskArgs string, stdin io.Reader) (string, error) {
	// Read stdin ONLY when it is actually needed: the template references
	// {{input}}, or no task text was given so stdin is the sole source.
	// CONTRACT: `amele run cfg "task"` must never touch stdin. Reading it
	// unconditionally hangs forever when stdin is an open pipe with no data
	// (a backgrounded run, a systemd socket, an orchestrator spawning amele).
	// Live-test finding B3.
	//
	// "No task text" is decided on the TRIMMED args, matching the emptiness
	// rule the switch below applies: whitespace-only args are no task at all,
	// so treating them as one suppressed the pipe and then refused with
	// "pass task text as arguments or pipe input on stdin" - advice the
	// operator had already followed.
	argsGiven := strings.TrimSpace(taskArgs) != ""
	needStdin := strings.Contains(cfg.Prompt, "{{input}}") || (cfg.Prompt == "" && !argsGiven)

	var input string
	if needStdin {
		var err error
		input, err = readPipedInput(ctx, stdin)
		if err != nil {
			return "", fmt.Errorf("reading stdin: %w", err)
		}
	}

	if cfg.Prompt != "" {
		rendered := strings.ReplaceAll(cfg.Prompt, "{{args}}", taskArgs)
		rendered = strings.ReplaceAll(rendered, "{{input}}", input)
		if strings.TrimSpace(rendered) == "" {
			// CONTRACT: exit 2, and nothing is sent to the provider - see
			// errEmptyPrompt.
			return "", errEmptyPrompt(cfg.Prompt)
		}
		return rendered, nil
	}

	// No template: the message is the task text, or - when there is none -
	// the piped input. The two can never both be present here, because
	// needStdin is false whenever taskArgs carries text and no template asked
	// for {{input}}; merging them is the prompt template's job.
	switch {
	case argsGiven:
		return taskArgs, nil
	case strings.TrimSpace(input) != "":
		return input, nil
	default:
		return "", errNoTask
	}
}

// errNoTask is the refusal when a run was given nothing to do at all.
var errNoTask = errors.New("no task given: pass task text as arguments or pipe input on stdin")

// errEmptyPrompt is the refusal for a prompt TEMPLATE that rendered to nothing
// but whitespace - typically `prompt: "{{input}}"` with an empty stdin, which
// is exactly what a cron job hands amele on the day its input file is empty.
//
// CONTRACT: this is a config-level refusal (exit 2, via cmdRun) raised BEFORE
// the provider is contacted. Sending it would buy a billable round trip that
// asks the model nothing, and the answer to a content-free prompt is noise the
// operator then has to read (live-test finding B-A03). Whitespace-only counts
// as empty, but whitespace AROUND content does not: the fixed text of a
// template is content, so `prompt: "Summarize:\n{{input}}"` still runs with an
// empty stdin - the model has an instruction even when it has no data.
//
// The template is quoted back because the operator wrote it once and is now
// reading a cron mail: naming it turns "why did this fail" into "which
// placeholder was empty". The message states only what the check knows - the
// rendered text is empty - and points at the two sources a template can draw
// from. It used to assert that "both {{args}} and {{input}} were empty", which
// is a guess: a template with no placeholders at all (prompt: "   ") renders to
// whitespace whatever the operator passed, and the sentence then blamed input
// that was never read.
func errEmptyPrompt(template string) error {
	return fmt.Errorf("empty prompt: the prompt template %q rendered to nothing but whitespace, so the model would be asked nothing; check the template and the text its placeholders read ({{args}} from the command line, {{input}} from stdin)", template)
}

// maxStdinBytes caps piped input so an accidental `cat hugefile | amele` does
// not balloon memory; the cut is marked so the model knows data is missing.
const maxStdinBytes = 10 * 1024 * 1024

// stdinTruncationMarker is appended when piped input hits maxStdinBytes.
const stdinTruncationMarker = "\n[input truncated at 10MB by amele]"

// isTerminal reports whether r is an interactive terminal, i.e. whether a
// human is sitting in front of this process. Two callers depend on it:
// readPipedInput (a terminal means "nothing was piped") and the permission
// approver (no terminal means an `ask` policy cannot be answered, docs/engineering.md
// §5.5).
//
// SECURITY: it fails safe towards "not a terminal". A non-*os.File reader, a
// file whose Stat fails, or the null device cannot be PROVEN interactive - and
// treating an unknown as headless only ever turns an `ask` into a deny, never
// the reverse.
//
// The character-device test alone was not enough: /dev/null is a character
// device too, and it is exactly what cron and systemd hand a job as stdin. An
// `ask` policy then prompted into the void, read the immediate EOF back, and
// logged it as a human's refusal - a deny either way, but one that hid the
// real reason from the operator reading the audit note. Detecting a TTY
// properly needs an isatty ioctl (golang.org/x/term or x/sys/unix), and the
// runner deliberately carries no dependency for it (docs/engineering.md §2), so the one
// impostor that actually occurs in production is excluded by identity instead.
func isTerminal(r io.Reader) bool {
	f, ok := r.(*os.File)
	if !ok {
		return false
	}
	info, err := f.Stat()
	if err != nil {
		return false
	}
	if info.Mode()&os.ModeCharDevice == 0 {
		return false
	}
	return !isNullDevice(info)
}

// isNullDevice reports whether info describes the same file as os.DevNull,
// comparing device and inode (os.SameFile) rather than a path - the caller's
// stdin arrives as an open descriptor with no name attached.
//
// SECURITY: every doubt answers "yes, it is the null device", so isTerminal
// returns false and an `ask` policy auto-denies. A null device that cannot be
// opened or stat'ed is a system too broken to be trusted with an approval.
func isNullDevice(info os.FileInfo) bool {
	null, err := os.Open(os.DevNull)
	if err != nil {
		return true
	}
	defer func() { _ = null.Close() }()
	nullInfo, err := null.Stat()
	if err != nil {
		return true
	}
	return os.SameFile(info, nullInfo)
}

// readPipedInput returns stdin's content when data is actually piped in, ""
// when stdin is an interactive terminal (a cron/CI run never blocks waiting
// for a human to type). Read is bounded by ctx so the armed run timeout can
// interrupt a pipe that never delivers.
func readPipedInput(ctx context.Context, stdin io.Reader) (string, error) {
	if isTerminal(stdin) {
		return "", nil // tty, nothing piped
	}

	// One byte past the cap is read so a full-but-not-over input is
	// distinguishable from a truncated one.
	data, err := readAsync(ctx, func() ([]byte, error) {
		return io.ReadAll(io.LimitReader(stdin, maxStdinBytes+1))
	})
	if err != nil {
		return "", err
	}
	if len(data) > maxStdinBytes {
		// The tail is gone; say so instead of letting the model reason
		// over silently incomplete data.
		return string(data[:maxStdinBytes]) + stdinTruncationMarker, nil
	}
	return string(data), nil
}

// toolResultCap resolves limits.max_tool_result_bytes for the tool constructors
// and the loop: 0 keeps every family's built-in cap and disables the loop
// ceiling, which is what a config without the key has always had.
func toolResultCap(cfg *config.Config) int {
	if cfg.Limits.MaxToolResultBytes == nil {
		return 0
	}
	return *cfg.Limits.MaxToolResultBytes
}

// buildRegistry assembles the tool registry from the validated config.
func buildRegistry(cfg *config.Config) (*tools.Registry, error) {
	registry := tools.NewRegistry()
	// CONTRACT: one number governs every tool family. Each constructor reads
	// it as "0 means your own legacy cap", so an absent key leaves the pre-v1.6
	// behaviour of all three families untouched.
	capBytes := toolResultCap(cfg)
	if cfg.Tools.FS {
		fsTools, err := tools.NewFSTools(cfg.Workspace, tools.FSOptions{MaxReadBytes: capBytes, MaxListBytes: capBytes})
		if err != nil {
			return nil, fmt.Errorf("initializing fs tools: %w", err)
		}
		for _, t := range fsTools {
			if err := registry.Register(t); err != nil {
				return nil, err
			}
		}
	}
	// SECURITY: the builtin shell exists ONLY when the config says so. This is
	// the single place the default-off contract is enforced, and it is a plain
	// `if` on an explicit YAML flag on purpose - nothing else (a permission
	// profile, a tool list, a flag) can turn the shell on by accident. The
	// allow/deny patterns inside the block are accident prevention, not a
	// security boundary; the boundary is the OS/container (docs/threat-model.md).
	if cfg.Tools.Shell.Enabled {
		shell, err := tools.NewShell(cfg.Tools.Shell, cfg.Workspace, tools.ShellOptions{MaxOutputBytes: capBytes})
		if err != nil {
			return nil, fmt.Errorf("initializing shell tool: %w", err)
		}
		if err := registry.Register(shell); err != nil {
			return nil, err
		}
	}
	for _, def := range cfg.Tools.Subprocess {
		if err := registry.Register(tools.NewSubprocess(def, cfg.Workspace, tools.SubprocessOptions{MaxOutputBytes: capBytes})); err != nil {
			return nil, err
		}
	}
	return registry, nil
}
