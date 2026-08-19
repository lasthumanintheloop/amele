# Exit code contract

**v1.2 - FROZEN as of v0.1; codes 7 and 8 added additively (0–6 unchanged).**
Scripts, cron jobs and CI pipelines branch on these values; changing the
meaning of any code is a breaking change (see
[Change policy](#change-policy)). Adding a code for a new outcome - as v1.1
did for the run lock and v1.2 for the MCP dependency - is additive and stays
within v1.

Every `amele` command maps every outcome onto this table. The mapping lives in
`cmd/amele/main.go` (`exitCodeFor` plus the per-command handlers) and is grep-able
via `// CONTRACT:` markers.

| Code | Name | Meaning |
|------|------|---------|
| 0 | success | The command completed. For `run`: the agent finished and its final answer was printed to stdout. For `chat`: the session ended at EOF (Ctrl-D). For `validate` / `explain` / `schema` / `init` / `help`: the command did its job. |
| 1 | task failed | The agent could not complete the task. |
| 2 | config error | The command never got as far as a run: bad usage or a bad config. |
| 3 | budget exceeded | A configured limit (`limits.max_turns`, `limits.max_tokens`, `limits.timeout`) killed the run. |
| 4 | permission denied | A tool call was denied **and the run aborted** because of it. |
| 5 | provider error | The LLM provider or the network failed after the client's retries were exhausted. |
| 6 | output schema unmet | `output.schema` is set, the model produced answers, but none validated within the retry budget. |
| 7 | lock held | `lock: true` is set and another run of this config is in progress; this run did nothing. |
| 8 | MCP unavailable | a required MCP server failed to start, connect or authenticate (including a missing or refused OAuth credential) |

## Per-code semantics

### 0 - success

`run`: the loop reached a clean final answer (and, in schema mode, one that
validated); stdout carries the answer and nothing else. `chat`: stdin reached
EOF. `validate`/`explain`/`schema`: report printed to stdout. `init`: file
written. `amele help` (also `-h`/`--help`): usage printed to stdout.

### 1 - task failed

The catch-all for "the agent ran but did not deliver":

- the model's tool-call-free turn was not a clean completion - output truncated
  at the token limit (`finish_reason: length`), blocked by a content filter
  (`content_filter`), a provider-reported mid-generation error (`error`), or an
  empty final answer;
- the run was interrupted by the operator: SIGINT (Ctrl-C) or SIGTERM cancels
  the run context, and a cancellation is deliberately exit 1, **not** 3 -
  monitoring must not read a Ctrl-C as a budget overrun (see
  [Signals](#signals) below);
- `chat` was interrupted at the prompt or its stdin broke;
- any run error that matches none of the typed cases below.

### 2 - config error

Everything that must be reported **before a single token is spent**:

- usage errors: no command, unknown command, missing/extra arguments, bad
  flags (`amele chat cfg.yaml some task` and `amele schema anything` are usage
  errors on purpose);
- the YAML failed to load or validate (unknown keys, missing model, literal
  API key, invalid permission policy, ...);
- `output.schema` does not compile (`validate`, `run` and `chat` all compile
  it, so a config one command blesses cannot fail on the next);
- the tool registry could not be built (e.g. the fs workspace does not exist)
  or the session file could not be created;
- `amele init`: the target file already exists (never overwritten) or could not
  be written.

`explain` is the exception, by design: it **reports**, and `run` **gates**. It
uses exit 2 only for a usage error or a config the loader itself rejected -
unreadable file, unparseable YAML, unknown key or wrong type, a literal
`provider.api_key`, an unusable `system_prompt_file` - because there is then no
config to describe. Every problem found after the load (undefined `${VAR}`s,
validation violations, an uncompilable `output.schema`, an unbuildable
registry) is printed in the report's `PROBLEMS` block with exit 0. The code's
meaning is unchanged - this narrows when `explain` reaches for it. The `amele
explain` section of [cli.md](cli.md) enumerates the surviving cases.

### 3 - budget exceeded

Reserved for **configured** limits:

- `limits.max_turns` reached without a final answer;
- `limits.max_tokens` exceeded by cumulative provider-reported usage - and,
  failing closed, when `max_tokens` is set but the endpoint reports no usage
  at all;
- `limits.timeout` (the run deadline) fired - anywhere, including while
  reading piped stdin before the first provider call, and including a deadline
  that cuts a provider request short (a timeout during an HTTP call is exit 3,
  not 5);
- `chat`: the session-wide turn/token pool is spent, or the per-exchange
  timeout fired.

### 4 - permission denied

The run **aborted** on a denial (`loop.ErrPermissionDenied`). Note the
asymmetry: under the built-in permission profiles an ordinary `deny` policy, a
refused `ask`, or a headless auto-deny does *not* end the run - the model
receives `permission denied` as a tool result and continues with the tools it
still has. Exit 4 is the aborting path: a denial the loop cannot continue
past. See [docs/features.md](../features.md#permission-profiles-permissions).

In practice this code is currently **reserved**: no built-in permission
profile produces exit 4 today. The profile's aborting paths (a broken or
erroring approver) pair the abort with an ordinary error, which the loop
reports as an approval-check failure and maps to exit 1 (task failed). The
code stays in the table because the abort-on-denial path exists in the loop
and future policies may use it; scripts should treat 4 as "denial killed the
run" whenever they see it.

### 5 - provider error

Any `llm.ErrProvider`: transport failures, non-2xx API responses, undecodable
replies - after the HTTP client's own retries are exhausted. Applies to both
the OpenAI-compatible and the native Anthropic client.

### 6 - output schema unmet

`output.schema` is configured, the model kept producing clean final answers,
and every one of them failed validation through `max_schema_retries` feedback
rounds (default 2). Distinct from exit 1 so pipelines can tell "wrong shape"
from "could not do the job". stdout stays empty - a consumer never sees
partial or invalid JSON.

### 7 - lock held

The config sets `lock: true` and another `amele run` of the **same config file**
already holds its lock, so this run exited immediately without contacting the
provider, without writing a session file and without touching the workspace.
stdout stays empty; stderr names the lock file.

This is the cron overlap guard: a run that hangs past its interval no longer
gets a second copy of itself interleaved into the same workspace. The lock is
advisory and per config file (`<config>.lock`), so `lock: true` is opt-in and
only excludes other cooperating `amele` processes. Only `run` locks -
`validate`, `explain`, `chat`, `schema` and `init` never do.

A cron wrapper that treats overlap as normal should special-case 7:
`amele run cfg.yaml "task"; [ $? -eq 7 ] && exit 0`.

### 8 - required MCP server unavailable

An MCP server declared with `required: true` could not be brought up: the
process failed to spawn, the transport never connected, the `initialize`
handshake failed or was rejected, the peer spoke something other than the
expected protocol, or a **required OAuth credential was missing, declined or
unrefreshable** - either at the pre-connect credential gate or at the connect
itself. The run stops before the agent loop starts - no provider call is made.
stdout stays empty; stderr names the server and the error class:

```
amele: mcp server "github": auth: 401 Unauthorized
mcp server "github": no oauth credential: mcp server unavailable (run 'amele mcp login agent.yaml github' first)
```

**The credential gate is still audited.** A run that ends at the gate - no
token and no terminal to ask on, or a login the operator declined - writes its
`run_start` and `run_end` (with `exit_code: 8` and `mcp_errors`) exactly like a
run whose connect failed later. The two are the same incident to whoever reads
the log afterwards, and an exit-8 run with no session line at all would be the
one case an operator could not reconstruct.

**Mid-run is different.** A credential that dies *during* a run - a refresh
refused, a token revoked under a live session - is **degradation, never a
mid-run exit 8**: the affected calls come back to the model as tool errors, the
loss is counted in `run_end.mcp_errors`, and the run ends on its own merits.
Exit 8 is a statement about **startup**: the toolset the config declared was
not there when the run began.

An interactive `amele mcp login` that does not complete - declined, or the flow
failed - is **exit 1**, not 8: nothing was gated, a command simply did not do
its job (see [cli.md](cli.md#amele-mcp-loginstatuslogout-configyamldir-server)).

Distinct from exit 5 on purpose. 5 means "the provider or the network was
flaky, retry the same command later"; 8 means "a dependency this config
declares is not there - a token expired, a binary is missing, an endpoint
moved - and retrying will keep failing until a human fixes it".

A server with `required: false` never produces this code: its failure is a
warning on stderr plus an `mcp_connect` event with `ok: false` in the session
log, and the run continues without that server's tools.

A cron wrapper that wants a soft-fail on flakiness but a page on a missing
dependency:

```sh
amele run agent.yaml || case $? in 5) exit 75;; 8) mail -s "amele: dependency down" ops@example.com </dev/null;; esac
```

## Signals

SIGINT (Ctrl-C) and SIGTERM cancel the run context; they do not kill the
process. Tools and the in-flight provider HTTP call stop promptly, the session
log still receives its `run_end` event (`status: error`, `exit_code: 1`), the
closing summary is still printed to stderr unless `-q`, stdout stays empty, and
the process exits **1**.

A shell reports a process *killed by* an uncaught signal as `128+N` - 130 for
SIGINT, 143 for SIGTERM. Because amele catches both, a script that interrupts a
run sees plain **1**, never 130 or 143. That is deliberate and part of
contract v1: an interrupted run is a run that did not deliver, which is what
exit 1 already means. SIGKILL cannot be caught: it leaves no `run_end` and the
shell reports 137.

`chat` behaves the same way from either position - interrupted at the `> `
prompt or mid-exchange, it writes its single `run_end`, prints
`chat interrupted: ...` and the summary, and exits 1. EOF (Ctrl-D) stays the
clean exit 0. Full wording in [cli.md](cli.md#signals).

## Change policy

This table is public API (project constitution §7). A breaking change - reusing
a code, changing a code's meaning, remapping an outcome - requires:

- a **semver major** bump,
- a PR titled `contract: ...` containing only the contract change,
- a migration note versioned in `docs/contracts/`.

Adding a *new* code for a *new* outcome is additive and allowed within v1;
consumers should treat unknown non-zero codes as failure.
