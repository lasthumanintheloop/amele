# CLI contract

**v1 - FROZEN as of v0.1.** Command names, flag names, argument shapes and
the stdout/stderr/exit-code behavior below are public API. The config file
itself has its own contract: [config.schema.json](config.schema.json) (also
printed by `amele schema`). Exit codes are specified in
[exit-codes.md](exit-codes.md); this page only cross-references them.

```
amele run <config.yaml|dir> [--model MODEL] [--set key=value] [-w DIR] [-q|-v] [task...]
amele chat <config.yaml|dir> [--model MODEL] [--set key=value] [-w DIR] [-q|-v]
amele validate <config.yaml|dir> [--set key=value] [-w DIR]
amele explain <config.yaml|dir> [--set key=value] [-w DIR]
amele mcp login|status|logout <config.yaml|dir> [server]
amele schema
amele init [path]
amele version
amele completion bash|zsh|fish
amele help [command]
```

Global behavior:

- `amele` with no arguments prints usage to **stderr** and exits 2.
- `amele help` (also `-h`, `--help`) prints usage to **stdout** and exits 0.
- `amele help <command>` and `amele <command> -h|--help` print that command's
  detailed page to **stdout** and exit 0 (see
  [`amele help`](#amele-help-command) below).
- An unknown command prints an error plus usage to stderr and exits 2.
- Every usage error (wrong argument count, bad flag) is exit 2. What stderr
  carries depends on the kind of mistake: a wrong argument count prints that
  command's `usage:` line, which is the answer to it, while a mistake that
  needs explaining prints the explanation (`amele run: -q/--quiet and
  -v/--verbose cannot be combined`).
- **Rejected flags** are uniform across `run`, `chat`, `validate` and
  `explain`: an unknown flag, or a flag written where the config path belongs
  (`amele run --set model=x agent.yaml`), is reported as
  `amele <command>: <reason>` followed by that command's `usage:` line. The Go
  `flag` package's own defaults dump never reaches stderr - its single-dash
  spellings (`-model`, `-set`) are not the documented flags - and a flag in the
  config-path slot is diagnosed as the argument-order mistake it is, not as a
  missing file. (Message wording is not frozen; the exit code is.)

The pipe rule that everything below follows: **stdout carries the product,
stderr carries everything meant for a human** - prompts, progress, errors, the
run summary.

## `amele run <config.yaml|dir> [--model MODEL] [--set key=value] [-w DIR] [-q|-v] [task...]`

One-shot run: load the config, run the agent on the task, print the final
answer, exit per the contract.

**Arguments.** The config path comes first, then flags, then free-form task
text. Flag parsing stops at the first non-flag argument, so anything after the
task text - including `--model` - is task text.

**`--model`** overrides the config's `model` for this run and participates in
validation (a config with no `model` plus `--model X` is valid). An empty value
(`--model ""`, an unset variable in a wrapper script) means *no override*, not
"no model"; `--set model=` is how the empty value is asked for deliberately.

### Config overrides: `--set key=value` and `-w` / `--workspace`

Added additively after the v1 freeze. Available
on `run`, `chat`, `validate` and `explain`.

- **`--set key=value`** overrides one config field for this invocation. It is
  repeatable and is split on the **first** `=`, so the value may contain more.
- **`-w DIR` / `--workspace DIR`** is sugar for `--set workspace=DIR`;
  **`--model M`** is sugar for `--set model=M` (on `run` and `chat`, where the
  flag already existed - `validate` and `explain` take `--set model=M`).
- **Merge order.** `--set`, `--model` and `-w` append to **one ordered list**
  in the order they appear on the command line, and the last entry for a key
  wins. There is no precedence between the spellings:
  `--model a --set model=b` sends `b`, `--set model=b --model a` sends `a`.
- **Ordering vs. validation.** Overrides are applied after loading and
  **before** validation, so they participate in it: an override can supply a
  missing required field, and a nonsense value is an exit-2 config error before
  anything is spent.
- **Settable keys (closed list).** `model`, `prompt`, `system_prompt_file`,
  `workspace`, `session_dir`, `limits.max_turns`, `limits.max_tokens`,
  `limits.timeout`, `output.max_schema_retries`. Any other key - including a
  typo - is exit 2 with
  `cannot override "X" from the command line; settable keys: ...`.
- **What is deliberately NOT settable**, and why: `tools.*`, `mcp.*`,
  `permissions.*` and `provider.*` grant capability - connecting an MCP server
  hands the run a new set of tools and the credential they travel with - and
  `lock` guards single-flight. The YAML
  file is the operator's audited grant of authority
  ([threat model §2](../threat-model.md)), so what
  `amele explain agent.yaml` reports about a config cannot be widened - or, for
  the lock, weakened - by a flag appended to the cron line that runs it.
- **Migration (2026-08-12): the allowlist shrank.** `lock` was settable when
  overrides shipped and no longer is: `--set lock=true|false` is now exit 2
  like any other non-settable key. It was the one entry that could *weaken* a
  run - `--set lock=false` disarmed the guard an audited `lock: true` had
  armed, from the invocation, invisibly to the file and its `explain` report.
  Set the field in YAML instead; nothing about `lock:` in the config changed.
  This is the only key ever removed from the list.
- **Values.** Integers via `strconv` (`limits.max_turns`, `limits.max_tokens`,
  `output.max_schema_retries`); `limits.timeout` takes a Go duration
  (`30s`, `5m`); the rest are strings taken verbatim. An
  empty value is accepted where it means something: `--set session_dir=`
  disables session logging. It is refused for `workspace` and
  `system_prompt_file`, which name nothing readable when empty.
- **Paths.** `workspace`, `session_dir` and `system_prompt_file` given via an
  override resolve against the **current working directory**, not the config
  file's directory (which is what the same fields written in YAML resolve
  against): a path typed in a shell means what it means in that shell.
  `system_prompt_file` is **re-read**, replacing whatever system prompt the
  config carried.
- Overrides never change stdout's contract, the exit codes or the session log
  format.

**`-q` / `--quiet`** and **`-v` / `--verbose`** - added additively after the v1 freeze.
Both are stderr-only: neither ever changes stdout, the exit code, or the
session log.

- `-q` suppresses the one-line summary and the non-error notes (the
  `output.schema` native-downgrade warning). It does **not** suppress errors,
  permission questions or permission decisions - a quiet run is silent when it
  works and still explains itself when it does not.
- `-v` adds one line per loop event, as it happens:

  ```
  amele: turn 3: model requested fs_read {"path":"app.log"}
  amele: turn 3: fs_read ok (1.2s)
  amele: turn 3: shell exit 3 (0.4s)
  amele: turn 3: shell timed out (30.0s)
  amele: turn 3: shell rejected (0.0s)
  amele: turn 3: fs_read error: <message>
  amele: turn 4: final answer (312 tokens)
  ```

  The turn number is the one the session log records; the token count is what
  the model produced in that turn. A tool call that ran but did not work is
  named as such instead of `ok`: `exit N` (the command failed), `timed out`
  (the tool's own timeout fired), `aborted` (the run ended under the command)
  and `rejected` (the shell policy refused the command). These are the endings
  the model receives as ordinary result text - the session log records that
  text, and the wording of the `-v` line is human-facing, not a parsing
  contract. Tool names, arguments and tool error text
  are model-controlled, so before they reach the terminal they are stripped of
  control and bidi-formatting characters, every value the config interpolated
  from the environment (`${VAR}`, including the API key) is replaced with
  `[REDACTED]` - the same by-value redaction the session log applies, because a
  cron job's stderr is persisted just as durably - and the result is **clipped**
  so one event stays one readable line. Redaction happens before clipping, so a
  long secret cannot get through by being cut in half. The clip bounds are an
  implementation detail chosen for readability, not a fixed promise, and they
  have already been raised once (from 120 to 512 runes per argument, plus a
  whole-line backstop) for exactly that redaction reason: a consumer that needs
  untruncated text should read the session log, not `-v` output.
- Giving both is a usage error: exit 2, before anything is loaded.
- Both obey the flag-stop rule: written after the task text they are task text.

**`-h` / `--help`** prints `run`'s detailed page to stdout and exits 0 without
loading anything - accepted in the config-path slot (`amele run -h`) and in the
flag positions (`amele run cfg.yaml --help`). It obeys the same flag-stop rule
as every other flag: a `-h` written after the task text is task text.

**stdin** is read only when it is actually needed: the config's `prompt`
template references `{{input}}`, or there is no `prompt` and no task text.
`amele run cfg.yaml "task"` never touches stdin, so it cannot hang on an open
pipe. When stdin is an interactive terminal, nothing is read (a run never
blocks waiting for typing). Piped input is capped at 10 MB; the cut is marked
with `[input truncated at 10MB by amele]` so the model knows data is missing.

A run whose final user message would be **empty or whitespace-only** - no task
text, nothing piped, or a `prompt` template whose placeholders all rendered
empty - is refused with exit 2 before the provider is contacted: a content-free
prompt costs tokens and answers nothing. Whitespace around content is content,
so a template that carries its own instruction (`prompt: "Summarize:\n{{input}}"`)
still runs with an empty stdin.

**stdout**: on success, exactly the agent's final answer followed by one
newline - nothing else, so runs compose in pipes. With `output.schema` set,
stdout carries the **canonical JSON** the validator accepted (the model's
fencing/prose framing is stripped). On any failure - including exit 6 -
stdout gets nothing.

**stderr**: config/run errors, permission questions
(`amele: allow tool X with {...}? [y/N]`) and audit notes, the `-v` progress
lines, and - unless `-q` - the one-line summary:
`✓ 8 turns, 3 tool calls, 41.0k tokens, 34.2s` (`✗` on failure; the two nouns
turn singular at a count of exactly 1: `✓ 1 turn, 1 tool call, ...`).

**Run lock**: with `lock: true` in the config, `run` takes a non-blocking
advisory lock on `<absolute config path>.lock` (created 0600) before reading
stdin or contacting the provider, and releases it when the run ends - normally
or not. A run that finds the lock held prints
`another run holds the lock for this config (lock file: ...)` to stderr and
exits **7**, having spent nothing and written nothing. The lock file is never
deleted. Default is off, so the same config can still be run concurrently with
different tasks. Only `run` locks, and the switch is settable **only in YAML**
(no `--set lock=`), so an invocation cannot disarm the guard a reviewed config
armed. For the cron/systemd angle on this flag -
and why an embedding that wants concurrent runs should leave it off - see
[docs/deployment.md](../deployment.md).

### Directory arguments

If the config argument is a directory, `<dir>/agent.yaml` is used. A
directory without `agent.yaml` is a config error (exit 2:
`no agent.yaml in <dir>`). The lock file (`lock: true`) is derived from
the resolved path, so `run pack/` and `run pack/agent.yaml` are the same
run for single-flight purposes. Applies to `run`, `chat`, `validate`,
`explain`. (Additive, 2026-08-12.)

**Exit codes**: the full [table](exit-codes.md) - 0 success, 1 task failed /
interrupted, 2 config error, 3 budget, 4 aborting permission denial,
5 provider, 6 schema unmet, 7 run lock held, 8 required MCP server
unavailable. Ctrl-C and SIGTERM land on 1; see [Signals](#signals).

## `amele chat <config.yaml|dir> [--model MODEL] [--set key=value] [-w DIR] [-q|-v]`

Interactive REPL over the same config, tools and permissions as `run`.

**Arguments.** Free-form arguments are rejected (exit 2) with a hint to use
`run` - a chat reads its input from stdin. `--model`, `--set` and
`-w` / `--workspace` behave as in `run` (same closed key list, same
command-line path resolution, same merge order), and `-h` / `--help` prints
`chat`'s detailed page to stdout (exit 0).

**`-q` / `--quiet`** and **`-v` / `--verbose`** behave as in `run`. In chat,
`-q` suppresses the closing session summary and the `output.schema is ignored
in chat` note, and keeps the `> ` prompt, the errors and the permission
questions - those are the conversation, not noise. `-v` progress lines carry
the session's continuous turn numbers, so they keep counting across lines.

**stdin**: one user message per line; a line is capped at 1 MB (the excess is
discarded, never re-served as the next line). Empty lines cost nothing - no
provider call, no turn, no tokens. EOF (Ctrl-D, or the end of a piped script)
ends the session with exit 0. The REPL and the `ask`-policy approval prompt
share one reader, so answering a question never eats the next chat line.

**stdout**: the model's answers only, each followed by a newline. It is a
*stream*, not a record format: answers routinely span several lines and there
is no delimiter. A scripted consumer that needs a parseable boundary should
use `amele run` (one answer per process).

**stderr**: the `> ` prompt, approval questions, notes (e.g.
`output.schema is ignored in chat`), errors, the `-v` progress lines, and -
unless `-q` - the cumulative session summary at the end.

**Budgets**: `limits.max_turns` and `limits.max_tokens` form one pool for the
whole session (exhausted → exit 3); `limits.timeout` bounds a single exchange,
not the session. One session file, one `run_start` (task `interactive chat`),
one `run_end` with the session totals.

**`output.schema`** is not enforced in chat; a note is printed to stderr and
the conversation proceeds.

## `amele validate <config.yaml|dir> [--set key=value] [-w DIR]`

Loads, validates and - when present - compiles `output.schema`, so a config
that validates cannot fail configuration under `run`.

- **Arguments**: exactly one positional argument, the config path, and it comes
  first; flags follow it, as in `run`. `--set` and `-w` are applied exactly as
  `run` applies them, so a parametrized invocation can be checked as the
  invocation it will be. A `--set` pair is a flag, not a positional, so it is
  not an argument-count violation. `--model` is not accepted here (use
  `--set model=M`).
- **stdout**: `<config.yaml>: OK` on success, nothing otherwise.
- **stderr**: every violation, reported together.
- **Exit**: 0 or 2. No network, no tokens, no session file.

## `amele explain <config.yaml|dir> [--set key=value] [-w DIR]`

Dry-run report: what the agent may touch, spend and emit, plus warnings for
valid-but-suspicious settings. It performs everything a run would do up to,
but not including, provider construction - load, validate, compile
`output.schema`, build the real tool registry. It also **connects to the
configured MCP servers** (see the `MCP SERVERS` section below): a server's
toolset lives on the far side of the connection, not in the YAML, so a report
that skipped it would be silent about the config's largest unreviewed surface.

**explain reports; run gates** (changed 2026-08-12). Every config problem
found after the file **loads** - undefined `${VAR}`s, validation violations, an
`output.schema` that will not compile, a tool registry that cannot be built -
is printed in a `PROBLEMS` block at the top of the report, and the command
still **exits 0**. Exit 2 is what survives from the load itself, where there
is no `*Config` to report on. Exactly these cases, and nothing else:

- a usage error (wrong argument count, unknown flag, malformed `--set`);
- the config file could not be read (missing, unreadable);
- the YAML could not be parsed: syntax error, empty file, or more than one
  document;
- strict decoding rejected the document: an unknown key (typically a typo like
  `max_token`) or a value of the wrong type;
- `provider.api_key` holds a literal secret rather than an `${ENV_VAR}`
  reference - the ban is enforced on the raw file, before anything is
  interpolated, so it fires even though the file itself is perfectly readable
  and parseable;
- the `system_prompt_file` could not be read (only when no `${VAR}` is
  missing - a path built from an unset variable is reported as a missing
  variable instead). Setting `system_prompt` and `system_prompt_file` together
  is **not** in this list (changed 2026-08-12): it is an ordinary validation
  violation now, so it appears in `PROBLEMS` alongside the file's other
  violations instead of aborting the load and hiding them.

`run`, `chat` and `validate` are unchanged: they still refuse a config with
any problem, at exit 2. The previous `explain` behaviour - exit 2 plus a
truncated report - withheld the report from exactly the reader who needed it,
the operator pre-flighting someone else's pack on a host where nothing is set
up yet.

- **Arguments**: as `validate` above.
- **Overrides in the report**: with `--set` (or `-w`) in play, the report opens
  with an `OVERRIDES` block echoing every pair the command line contributed -
  sugar spellings shown desugared, values quoted - and every line whose value
  came from there carries the suffix ` (overridden via --set)`. Without any
  override the report is byte-identical to what it was before the flag existed.
  `PROBLEMS`, when present, precedes `OVERRIDES`.
- **stdout**: the report, including any `PROBLEMS`.
- **stderr**: errors only; empty whenever the report was printed.
- **Exit**: 0 whenever a report was printed; 2 for a usage error or a config
  the loader rejected (the list above). No provider call, no tokens, no session
  file. The only network `explain` makes is to the declared MCP servers.
- **MCP servers section** (additive, 2026-08-19): when the config declares
  `mcp.servers`, the report carries an `MCP SERVERS` block - one row per
  server with its transport and target (the URL **without its query string**,
  or `command[0]` for stdio), then either `✓ connected (<ms> ms, proto <v>,
  <server name> <version>)` or `✗ FAILED (<class>): <error>`, then one line
  per tool (`✓ kept`, with `(destructive)` / `(read-only)` when the server
  annotates it and `(was "<original>")` when the name had to be rewritten, or
  `- <reason>` for a tool a `tools.include`/`exclude` filter or a size cap
  left out), and a `definitions: N tools, B bytes ≈ T tokens` summary. A
  server with an `auth` block also gets a credential line directly under its
  row - `auth: oauth (token valid until <RFC3339>, refresh: yes)`, or
  `auth: oauth (no token - run 'amele mcp login <config> <server>')` - which
  states facts ABOUT the stored credential and never the credential itself.
  Note that `explain` **uses** that credential: it connects for real, so a
  stale token may be refreshed (and rotated) by an `explain`.
  A connect failure is **reported, never fatal**: `explain` still exits 0 even
  for a `required: true` server whose failure would abort `amele run` with
  exit 8. A tool-name collision is different - it is a mistake in the config,
  so it appears in `PROBLEMS`. Definitions are re-sent on every request, so a
  toolset estimated above 4000 tokens earns a `WARNINGS` line suggesting
  `tools.include`. Servers are disconnected (and stdio child processes
  reaped) before the report is printed.
- **Requirements section** (additive, 2026-08-12): the report carries a
  `REQUIREMENTS` block listing every `${VAR}` the config references (✓ set /
  ✗ MISSING), every executable the config needs on `PATH` (✓ found /
  ✗ MISSING): each subprocess tool's `command[0]` and each stdio MCP server's,
  and each subprocess/shell tool's `env:` allowlist plus each stdio MCP
  server's (rows labelled `mcp:<name>`; variable names only, never values) - the answer to "what does this config need from the
  host, and what will it read from my environment?".
- **Interpolated values** (changed 2026-08-12): the report shows the values
  `${VAR}` interpolation substituted, so a parametrized pack can be
  pre-flighted, EXCEPT credentials, which print as `[REDACTED]`: whatever
  feeds `provider.api_key` (regardless of the variable's name) and any
  variable whose name contains `key`, `token`, `secret`, `passw` or `cred`
  (case-insensitive, anywhere in the name). This display rule is explain's
  alone - session-log redaction (`docs/contracts/jsonl-events.md`) stays
  unconditional by value.

## `amele mcp login|status|logout <config.yaml|dir> [server]`

Credential management for the MCP servers a config declares with
`auth: {type: oauth}` ([docs/mcp.md](../mcp.md#oauth)). Added additively
2026-08-19.

- **Arguments**: the subcommand first, then the config path (a directory is
  resolved exactly as `run` resolves one, see
  [Directory arguments](#directory-arguments)). `login` and `logout` take an
  optional server name after it; with none, they act on **every** server in
  the config that declares oauth, in config order. `status` takes **no** server
  argument - naming one would invite the reader to believe the others had been
  checked and found fine. Anything else is a usage error (exit 2).
- **Flags**: none, ever, apart from `-h`/`--help`. `--set` and `-w` are not
  accepted here: no `mcp.*` key is overridable
  (see "Config overrides" under `amele run`), so an override could only point a
  credential command at a server the run would never use.
- **The config is loaded and validated first**, exactly as `run` would (minus
  overrides). A config `run` would refuse is refused here too, at exit 2: the
  URL a credential is about to be keyed by is only trustworthy once `Validate`
  has seen it.

### `login`

Runs the interactive browser flow, one server at a time in config order - three
servers must not race three browser windows.

- **stdin**: must be a **real terminal**. A pipe, a closed stdin or
  `/dev/null` is refused with exit 2 rather than left waiting; the permission
  system's "EOF is a refusal" tolerance is deliberately not reused, because a
  cron job must hear that a login needs a human.
- **stdout**: nothing. Safe to redirect.
- **stderr**: the confirmation questions, the authorization URL, and one
  `mcp login ok: <server> (expires <RFC3339>)` line per server.
- **Exit**: 0 when every selected login completed; **1** when one did not -
  declined at either question, or the flow failed (`login` is a command that
  did not do its job, not a gated run: a run gated on a missing credential is
  [exit 8](exit-codes.md#8---required-mcp-server-unavailable)); **2** for a
  usage error, a config error, an unknown server name, a server with no `auth`
  block, or a non-interactive stdin.

### `status`

A **report**, like `explain`: it never refreshes, never opens a browser,
never contacts any server and never changes a byte on disk.

- **stdout**: one row per stored credential (a server can hold more than one),
  or a `no token` row for a server with none, followed by a `problem: ...` line
  for each unreadable file. The row carries the state (`ok`/`expired`), the
  expiry, whether a refresh token is present, the granted scopes and the
  issuer. **A token value is never printed.**
- **The table's layout is INFORMATIONAL and NOT frozen**: column order,
  spacing and wording may change without a contract bump. What is frozen is
  the stream (stdout), the exit code, and that no credential appears in it.
  Scripts must not parse it.
- **stderr**: empty.
- **Exit**: **0** whenever the config loaded - including when nothing at all is
  stored ("no token" is an answer, not a failure) and when a token file could
  not be read. 2 only for a usage or config error.

### `logout`

Hands the credential back to the authorization server (RFC 7009) when one was
advertised at login, then deletes it locally.

- The revocation is **best effort**: if it fails, the local delete still
  happens and the line says so. An operator who asked to be rid of a
  credential must not keep it because a server is down.
- Revocation invalidates the **whole grant**, not this machine's copy of it:
  any other machine holding a copy of that credential is logged out too. Do
  not use `logout` to clean up after copying a credential elsewhere - move or
  `rm` the file instead.
- **stdout**: nothing.
- **stderr**: one `mcp logout: <server> (<label>)` line per server, where the
  label is `revoked` (the server confirmed), `local only` (no revocation
  endpoint, or telling it failed - the credential is gone here and may still be
  live there) or `no token` (there was nothing to delete). A failed revocation
  also prints an `amele: warning: ...` line.
- **Exit**: 0 normally; 1 when a credential file could not be deleted; 2 for a
  usage or config error.

## `amele schema`

Prints the embedded config JSON Schema (the same document as
[config.schema.json](config.schema.json)).

- **Arguments**: none; any argument is a usage error (exit 2), so a
  misremembered `amele schema config.yaml` fails loudly. `-h` / `--help` is
  the one exception, and prints the `schema` page.
- **stdout**: exactly the schema document plus a trailing newline - valid JSON
  as-is (`amele schema > config.schema.json`).
- **Exit**: 0 or 2.

## `amele init [path]`

Writes an annotated starter config to `path` (default `agent.yaml`). The
generated file passes `amele validate` as written once `AMELE_API_KEY` is set.

- **stdout**: nothing - init composes in scripts like every other command.
- **stderr**: the next-step hint
  (`amele: wrote agent.yaml - next: set AMELE_API_KEY and run: amele validate agent.yaml`),
  or the error.
- An existing file is **never overwritten**: exit 2 with an explanation.
- **Exit**: 0 or 2. At most one argument.

## `amele version` (also `amele --version` / `amele -V`)

Prints this binary's build identity - added additively after the v1 freeze.

- **stdout**: exactly one line -
  `amele <version> (commit <commit>, built <date>, <go runtime version>, <GOOS>/<GOARCH>)`
  - followed by a single newline. A source checkout (`go build`/`go run`
  without the release `-ldflags`) reports `amele dev (commit unknown, built
  unknown, ...)`; a released binary carries the real version, commit and
  build date baked in by the Makefile.
- **Arguments**: none; any argument is a usage error (exit 2). `-h` /
  `--help` is the one exception, and prints the `version` page.
- **Exit**: 0 or 2. No network, no tokens, no session file.

## `amele completion bash|zsh|fish`

Prints a static shell completion script to stdout - added additively after the v1 freeze.
The scripts are hand-written against each shell's own completion builtins
(bash's `compgen`/`complete`, zsh's `compsys`, fish's `complete`); there is no
generator and no shared completion framework, in keeping with the
single-static-binary, no-runtime-dependency rule ([docs/engineering.md](../../docs/engineering.md) §2).
Each script completes the subcommands, the flags each subcommand accepts,
config paths (YAML files or pack directories) in the config-path slot, and
the shell names accepted by `completion` itself.

- **Arguments**: exactly one, the shell name (`bash`, `zsh` or `fish`); no
  argument, an unrecognized shell, or more than one argument is a usage error
  (exit 2). `-h` / `--help` is the one exception, honored only as the sole
  argument, and prints the `completion` page.
- **stdout**: the completion script for the named shell, newline-terminated.
- **stderr**: the usage error, if any.
- **Exit**: 0 or 2.
- **Install hints**:
  - bash: `amele completion bash > /etc/bash_completion.d/amele`
  - zsh: `amele completion zsh > "${fpath[1]}/_amele"`
  - fish: `amele completion fish > ~/.config/fish/completions/amele.fish`

## `amele help [command]`

The built-in manual - added additively after the v1 freeze. Bare
`amele help` (and `-h` / `--help`) keeps its frozen behavior: the **short
usage** on stdout, exit 0. Naming a command is the new part.

- **`amele help <command>`** prints that command's detailed page to stdout and
  exits 0. Every page carries the same sections: SYNOPSIS, DESCRIPTION, FLAGS
  (every flag with its default), STDIN / STDOUT / STDERR, EXIT CODES for that
  command, and runnable EXAMPLES. The pages restate this contract; where they
  disagree with this document, this document wins.
- **`amele <command> -h|--help`** prints the same page. Additive: `-h` was
  previously a usage error for every command and was never a valid config
  path. For `run` and `chat` the flag is honored only where a flag is honored -
  flag parsing stops at the first non-flag argument, so a `-h` after the task
  text stays task text. For the commands with a fixed positional count
  (`validate`, `explain`, `schema`, `init`, `version`, `completion`) it is
  honored only as the command's **sole** argument; alongside other arguments
  (`--set` pairs included) the invocation stays a usage error (exit 2), per
  the global rule above.
- Documented commands: `run`, `chat`, `validate`, `explain`, `mcp`, `schema`,
  `init`, `version`, `completion`, `help`. Alternate spellings resolve to their
  command, so `amele help --version` reaches the `version` page.
- **`amele help <not-a-command>`** prints `unknown command "..."` plus the
  short usage to stderr and exits 2.
- **Arguments**: at most one command name; more is a usage error (exit 2).
- **stdout**: the short usage or the requested page. **stderr**: the usage
  error only.
- **Exit**: 0 or 2. No network, no tokens, no session file.

## Signals

`SIGINT` (Ctrl-C) and `SIGTERM` **cancel the run** rather than killing the
process. The cancellation reaches everything the run is doing - the in-flight
provider HTTP call, a running subprocess or shell tool, a blocked stdin read -
so a run stops promptly instead of at the next convenient moment. Then the
normal ending path runs:

- the session log gets its `run_end` event (`status: error`, `exit_code: 1`),
  so an interrupted run is as auditable as a failed one;
- stderr gets the error (`run interrupted: context canceled`) and - unless
  `-q` - the closing summary line;
- stdout stays empty: an interrupted run produces no answer, so nothing
  malformed reaches a pipe;
- the process exits **1** (task failed; see [exit-codes.md](exit-codes.md)).

**Exit 1, not 128+N.** A shell reports a process *killed by* an uncaught signal
as `128+N` - 130 for SIGINT, 143 for SIGTERM. amele catches both, so a script
that interrupts it sees plain **1**. This is deliberate and part of contract
v1: an interrupted run is a run that did not deliver, which is exactly what
exit 1 already means, and a cron wrapper needs no second failure convention.
`SIGKILL` is not catchable - it leaves no `run_end` and the shell reports 137;
it is the escape hatch, not the normal stop. amele stays in charge of both
catchable signals for its whole life, so a second Ctrl-C does not escalate.

`chat` follows the same contract from either position. Interrupted **at the
`> ` prompt** (the common case - a blocked stdin read cannot see a signal on
its own, so the REPL races the read against the context) it prints
`chat interrupted: context canceled`, writes the session's single `run_end`
with the cumulative totals, prints the summary unless `-q`, and exits 1.
Interrupted **mid-exchange** the loop unwinds the same way and the session ends
identically. Ctrl-D (EOF) remains the clean way out: exit 0.

For what this means running under a supervisor - systemd sending
`TimeoutStartSec`'s SIGTERM, or `systemctl stop` - see
[docs/deployment.md](../deployment.md#2-systemd-service--timer).

## Change policy

Renaming or removing a command or flag, changing an argument shape, or
changing any stdout/stderr/exit behavior above is breaking: semver major,
`contract:`-titled PR, migration note in `docs/contracts/`. New commands and
new optional flags are additive and allowed within v1.

History: the binary was named `agnt` while the project was unreleased and was
renamed to `amele` before the first public release. Since no version had ever
shipped, nothing could depend on the old name - the rename is **not** a
breaking contract change and does not bump v1.
