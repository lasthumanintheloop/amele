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
	"github.com/lasthumanintheloop/amele/internal/explain"
	"github.com/lasthumanintheloop/amele/internal/llm"
	"github.com/lasthumanintheloop/amele/internal/loop"
	"github.com/lasthumanintheloop/amele/internal/mcp"
	"github.com/lasthumanintheloop/amele/internal/perm"
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
	usageRun        = "usage: amele run <config.yaml|dir> [--model MODEL] [--set key=value] [-w DIR] [-q|-v] [task...]"
	usageChat       = "usage: amele chat <config.yaml|dir> [--model MODEL] [--set key=value] [-w DIR] [-q|-v]"
	usageValidate   = "usage: amele validate <config.yaml|dir> [--set key=value] [-w DIR]"
	usageExplain    = "usage: amele explain <config.yaml|dir> [--set key=value] [-w DIR]"
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
  amele run <config.yaml|dir> [--model MODEL] [--set key=value] [-w DIR] [-q|-v] [task...]

DESCRIPTION
  Loads the config, runs the agent on the task, prints the final answer and
  exits per the exit-code contract. This is the headless mode: it needs no
  terminal, so it composes in cron jobs, CI steps and shell pipes.

  The config path comes first, then flags, then free-form task text. Flag
  parsing stops at the first non-flag argument, so anything after the task
  text - including --model - is task text.

  A directory argument is shorthand for <dir>/agent.yaml inside it; the run
  lock is derived from the resolved file, so both spellings contend on the
  same lock.

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
                    output.max_schema_retries, provider.max_output_tokens,
                    provider.reasoning.effort, provider.temperature,
                    provider.top_p
                  Tools, permissions, the provider's identity (type, base_url,
                  api_key) and the run lock are deliberately NOT settable: the
                  YAML file stays the audited grant of authority, so what
                  "amele explain agent.yaml" reports cannot be widened - or, in
                  the lock's case, weakened - by a flag on the cron line
                  (docs/threat-model.md §2). The provider tuning knobs above
                  are settable because they only change what a run spends, not
                  what it may do.
                  workspace, session_dir and system_prompt_file resolve against
                  the CURRENT DIRECTORY, not the config's - a path typed in a
                  shell means what it means in that shell. system_prompt_file
                  is re-read and replaces whatever prompt the config carried.
                  An empty session_dir (--set session_dir=) turns session
                  logging off. Default: nothing overridden.
  -w, --workspace DIR
                  Shortcut for --set workspace=DIR. Default: the config's
                  workspace (its own directory unless the YAML says otherwise).
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
  references {{input}}, or there is no prompt and no task text.
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
  the one-line summary: ✓ 8 turns, 3 tool calls, 41.0k tokens, 34.2s (✗ on
  failure).

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

  A directory argument is shorthand for <dir>/agent.yaml inside it.

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

  A directory argument is shorthand for <dir>/agent.yaml inside it.

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

  A directory argument is shorthand for <dir>/agent.yaml inside it.

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

  Commands with a page: run, chat, validate, explain, schema, init, version,
  completion, help. The alternate spellings resolve too, so
  amele help --version reaches the version page.

  For run and chat the help flag is recognized only where a flag is
  recognized. Flag parsing stops at the first non-flag argument, so a -h that
  appears after the task text is part of the task, not a help request.

  For the commands with a fixed argument count - validate, explain, schema,
  init, version, completion - the flag is recognized only as the sole
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
	parsed, ok := parseInspectArgs("validate", usageValidate, args, stderr)
	if !ok {
		return ExitConfigError
	}
	if parsed.help {
		return printHelp("validate", stdout, stderr)
	}
	cfg, err := config.Load(parsed.configPath, env)
	if err == nil {
		err = applyCLIOverrides(cfg, parsed.overrides)
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
	parsed, ok := parseInspectArgs("explain", usageExplain, args, stderr)
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
	if err := applyCLIOverrides(cfg, parsed.overrides); err != nil {
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

// parseAgentArgs parses that shared shape for the named command. On a usage
// error it writes the reason to stderr and reports ok=false; the caller turns
// that into ExitConfigError.
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
func parseAgentArgs(name, usage string, args []string, stderr io.Writer) (agentArgs, bool) {
	if len(args) < 1 {
		_, _ = fmt.Fprintln(stderr, usage)
		return agentArgs{}, false
	}
	if args[0] == "-h" || args[0] == "--help" {
		return agentArgs{help: true}, true
	}
	if strings.HasPrefix(args[0], "-") {
		rejectFlagInConfigPathSlot(name, usage, args[0], stderr)
		return agentArgs{}, false
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
	if err := fs.Parse(args[1:]); err != nil {
		_, _ = fmt.Fprintf(stderr, "amele %s: %v\n%s\n", name, err, usage)
		return agentArgs{}, false
	}
	parsed := agentArgs{
		configPath: args[0],
		overrides:  overrides,
		help:       *helpShort || *helpLong,
		quiet:      *quietShort || *quietLong,
		verbose:    *verboseShort || *verboseLong,
		rest:       fs.Args(),
	}
	// Help wins over the conflict below: someone who asked for the manual gets
	// the manual, and the page is where the two flags are explained.
	if parsed.help {
		return agentArgs{help: true}, true
	}
	if parsed.quiet && parsed.verbose {
		// They ask for opposite things. Letting one silently win would hide a
		// mistake in a script that means to change how noisy a cron job is.
		// CONTRACT: a usage error like any other - exit 2, nothing loaded.
		_, _ = fmt.Fprintf(stderr, "amele %s: -q/--quiet and -v/--verbose cannot be combined\n", name)
		return agentArgs{}, false
	}
	resolved, err := resolveConfigArg(parsed.configPath)
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "amele %s: %v\n", name, err)
		return agentArgs{}, false
	}
	parsed.configPath = resolved
	return parsed, true
}

// resolveConfigArg maps a directory argument to its canonical entry point
// <dir>/agent.yaml. Files and non-existent paths pass through untouched so
// Load keeps reporting them with its own errors. CONTRACT: resolution
// happens at parse time, BEFORE the run lock is derived, so `run pack/`
// and `run pack/agent.yaml` contend on the same lock file.
func resolveConfigArg(path string) (string, error) {
	info, err := os.Stat(path) //nolint:gosec // G703: the path is the operator's own config argument; statting it is this command's purpose
	if err != nil || !info.IsDir() {
		return path, nil
	}
	candidate := filepath.Join(path, "agent.yaml")
	if _, err := os.Stat(candidate); err != nil { //nolint:gosec // G703: same operator-supplied path, joined with a fixed name
		return "", fmt.Errorf("no agent.yaml in %s", path)
	}
	return candidate, nil
}

// loadAgentConfig loads, overrides, validates and compiles one agent config -
// everything both commands must do before anything can block or contact a
// provider. The returned validator is nil when the config declares no
// output.schema.
//
// CONTRACT: every error here is exit 2 (config error) at the call site, and it
// must be reported without spending a single token.
func loadAgentConfig(parsed agentArgs, env config.LookupEnv) (*config.Config, *schema.Validator, error) {
	cfg, err := config.Load(parsed.configPath, env)
	if err != nil {
		return nil, nil, err
	}
	// Overrides are applied BEFORE Validate, so they participate in it: "no
	// model in YAML but --set model=X given" is valid, "still no model" is
	// caught, and a nonsense override (a negative budget, an unreachable
	// workspace) is an exit-2 config error rather than a mid-run surprise.
	if err := applyCLIOverrides(cfg, parsed.overrides); err != nil {
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
func applyCLIOverrides(cfg *config.Config, overrides []string) error {
	if len(overrides) == 0 {
		return nil
	}
	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Errorf("resolving the working directory for --set paths: %w", err)
	}
	return config.ApplyOverrides(cfg, overrides, cwd)
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
func parseInspectArgs(name, usage string, args []string, stderr io.Writer) (inspectArgs, bool) {
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
	resolved, err := resolveConfigArg(args[0])
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "amele %s: %v\n", name, err)
		return inspectArgs{}, false
	}
	return inspectArgs{configPath: resolved, overrides: overrides}, true
}

// cmdRun executes a one-shot agent run and maps every failure to the exit
// code contract.
func cmdRun(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer, env config.LookupEnv) int {
	parsed, ok := parseAgentArgs("run", usageRun, args, stderr)
	if !ok {
		return ExitConfigError
	}
	if parsed.help {
		return printHelp("run", stdout, stderr)
	}
	taskArgs := strings.Join(parsed.rest, " ")

	cfg, validator, err := loadAgentConfig(parsed, env)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return ExitConfigError
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
	// data must be interruptible by limits.timeout (live-test finding B3).
	if cfg.Limits.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, cfg.Limits.Timeout.Std())
		defer cancel()
	}

	task, taskErr := buildTask(ctx, cfg, taskArgs, stdin)
	// Reading stdin is already part of the run, so an interruption there is an
	// interrupted RUN, not a config error: it is carried past the agent
	// construction below and reported through the normal ending path.
	// CONTRACT (docs/contracts/cli.md "Signals"): run_end in the session log,
	// the cause and the summary on stderr, exit 1 - or exit 3 when the cause
	// was the configured limits.timeout.
	interrupted := errors.Is(taskErr, context.Canceled) || errors.Is(taskErr, context.DeadlineExceeded)
	if taskErr != nil && !interrupted {
		// Nothing started: no task was given at all, or stdin itself failed.
		_, _ = fmt.Fprintln(stderr, taskErr)
		return ExitConfigError
	}

	agent, answer, hints, err := buildAgent(cfg, validator, lines, stderr, secrets)
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
	agent.Session.RunStart(cfg.Model, task)

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

	res, runErr := agent.RunMessages(ctx, openingHistory(cfg, task))
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
	_, _ = fmt.Fprintln(stderr, secrets.Redact(
		"warning: tool schemas sanitized for the gemini wire (unsupported JSON Schema keywords removed): "+
			strings.Join(stripped, ", ")))
}

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
	agent.Session.RunStart(cfg.Model, taskArgs)
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
	agent.Session.RunEnd(status, code, res.Turns, res.ToolCalls, res.Usage.Total(), res.Duration)
	if quiet {
		return
	}
	_, _ = fmt.Fprintln(stderr, session.Summary(runErr == nil, res.Turns, res.ToolCalls, res.Usage.Total(), res.Duration))
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
	agent, _, _, err := buildAgent(cfg, validator, lines, stderr, secrets)
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
	agent.Session.RunStart(cfg.Model, taskArgs)
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
	parsed, ok := parseAgentArgs("chat", usageChat, args, stderr)
	if !ok {
		return ExitConfigError
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

	cfg, validator, err := loadAgentConfig(parsed, env)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return ExitConfigError
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

	agent, _, hints, err := buildAgent(cfg, nil, lines, stderr, secrets)
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
		agent.Session.RunStart(cfg.Model, chatTaskLabel)
		s := &chatSession{cfg: cfg, agent: agent, quiet: parsed.quiet,
			mcp: &mcpSet{failed: gateMCPErrors(gateErr)}, runCtx: ctx, secrets: secrets}
		return s.finish(stderr, exitCodeFor(gateErr), gateErr)
	}

	// A chat writes ONE run_start for the whole session, and the MCP servers
	// belong inside it like every other event (docs/contracts/jsonl-events.md).
	agent.Session.RunStart(cfg.Model, chatTaskLabel)

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
	s.agent.Session.RunEnd(status, code, s.turns, s.toolCalls, s.tokens, s.duration)
	if !s.quiet {
		_, _ = fmt.Fprintln(stderr, session.Summary(err == nil, s.turns, s.toolCalls, s.tokens, s.duration))
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
func buildAgent(cfg *config.Config, validator *schema.Validator, lines *lineReader, stderr io.Writer,
	secrets *session.SecretSet) (*loop.Loop, func(*loop.Result) string, map[string]string, error) {
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
		sess, err = session.New(cfg.SessionDir, session.Options{SecretSource: secrets})
		if err != nil {
			return nil, nil, nil, err
		}
	}

	provider, err := buildProvider(cfg, secrets.Add)
	if err != nil {
		return nil, nil, nil, err
	}
	tuning, err := providerTuning(cfg)
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
		Model:         cfg.Model,
		SystemPrompt:  cfg.SystemPrompt,
		Tuning:        tuning,
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

// buildProvider constructs the LLM client selected by provider.type. Validate
// has already constrained the type, so anything that is neither anthropic nor
// gemini is the OpenAI-compatible default - including "", which is what every
// pre-Type config carries.
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
func buildProvider(cfg *config.Config, registerSecret func(...string)) (llm.Provider, error) {
	maxAttempts, initialBackoff := retryPolicy(cfg.Provider.Retry)
	if cfg.Provider.Type == config.ProviderTypeAnthropic {
		// The dialect names a variation of the OpenAI-compatible wire and is
		// documented as ignored here (config.schema.json), so it is not parsed
		// on this path: a leftover dialect must not fail a run that never
		// speaks it.
		return &llm.AnthropicClient{
			BaseURL:         cfg.Provider.BaseURL,
			APIKey:          cfg.Provider.APIKey,
			RequestTimeout:  cfg.Provider.RequestTimeout.Std(),
			MaxAttempts:     maxAttempts,
			InitialBackoff:  initialBackoff,
			MaxOutputTokens: cfg.Provider.MaxOutputTokens,
		}, nil
	}
	if cfg.Provider.Type == config.ProviderTypeGemini {
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
			BaseURL: cfg.Provider.BaseURL,
			APIKey:  cfg.Provider.APIKey,
			// The vertex block travels as the client's target so the request is
			// addressed to the endpoint the config names, and as the credential
			// source that authenticates it. Both are nil without the block,
			// which is what keeps the AI Studio path a keyed one.
			Vertex:         vertexTarget(cfg.Provider.Vertex),
			TokenSource:    vertexTokenSource(cfg.Provider.Vertex, registerSecret),
			RequestTimeout: cfg.Provider.RequestTimeout.Std(),
			MaxAttempts:    maxAttempts,
			InitialBackoff: initialBackoff,
		}, nil
	}
	dialect, err := llm.ParseDialect(cfg.Provider.Dialect)
	if err != nil {
		return nil, fmt.Errorf("provider.dialect: %w", err)
	}
	return &llm.OpenAIClient{
		BaseURL:        cfg.Provider.BaseURL,
		APIKey:         cfg.Provider.APIKey,
		Dialect:        dialect,
		RequestTimeout: cfg.Provider.RequestTimeout.Std(),
		MaxAttempts:    maxAttempts,
		InitialBackoff: initialBackoff,
	}, nil
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

// providerTuning translates the config's provider knobs into the neutral
// request fields the loop forwards on every turn.
//
// CONTRACT: this is the ONE place where provider.params (arbitrary YAML)
// becomes JSON. Validate already proved the map is serializable and collides
// with no field amele owns, so a failure here is not reachable through a
// validated config - it is wrapped rather than ignored because a silently
// dropped params map would leave the run missing a knob the file asked for.
func providerTuning(cfg *config.Config) (loop.Tuning, error) {
	extra, err := paramsJSON(cfg.Provider.Params)
	if err != nil {
		return loop.Tuning{}, fmt.Errorf("provider.params: %w", err)
	}
	return loop.Tuning{
		MaxOutputTokens: cfg.Provider.MaxOutputTokens,
		Reasoning:       reasoningSpec(cfg.Provider.Reasoning),
		Temperature:     cfg.Provider.Temperature,
		TopP:            cfg.Provider.TopP,
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

// buildRegistry assembles the tool registry from the validated config.
func buildRegistry(cfg *config.Config) (*tools.Registry, error) {
	registry := tools.NewRegistry()
	if cfg.Tools.FS {
		fsTools, err := tools.NewFSTools(cfg.Workspace, tools.FSOptions{})
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
		shell, err := tools.NewShell(cfg.Tools.Shell, cfg.Workspace)
		if err != nil {
			return nil, fmt.Errorf("initializing shell tool: %w", err)
		}
		if err := registry.Register(shell); err != nil {
			return nil, err
		}
	}
	for _, def := range cfg.Tools.Subprocess {
		if err := registry.Register(tools.NewSubprocess(def, cfg.Workspace)); err != nil {
			return nil, err
		}
	}
	return registry, nil
}
