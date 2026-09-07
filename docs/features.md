# Structured output, permissions, shell, chat and parallel tool calls

Beyond the core `run` + `validate`, five features round out the agent: structured
output, permission profiles, a builtin `shell` tool, an interactive `chat`
mode, and parallel tool calls. Everything below is configured in the same single
YAML file.

Exit codes are unchanged and frozen
([contract](contracts/exit-codes.md)):

```
0 success · 1 task failed · 2 config error · 3 budget exceeded
4 permission denied · 5 provider error · 6 output schema unmet
7 run lock held · 8 required MCP server unavailable
```

## Structured output (`output.schema`)

Declare a JSON Schema and stdout becomes machine-readable: `amele run` prints
**only** a JSON document that satisfies the schema, or it fails.

```yaml
model: openai/gpt-4o-mini
provider:
  base_url: https://openrouter.ai/api/v1
  api_key: ${OPENROUTER_API_KEY}
system_prompt: "You are a strict code reviewer."
output:
  schema:
    type: object
    additionalProperties: false
    required: [score, summary]
    properties:
      score:   {type: integer, minimum: 0, maximum: 10}
      summary: {type: string}
  max_schema_retries: 2   # default: 2
```

```console
$ amele run judge.yaml < diff.txt | jq .score
7
```

With no task text on the command line, the piped diff *is* the task (stdin is
only read when it is needed: passing task text as an argument instead would
leave the pipe untouched - add a `prompt:` template with `{{args}}` and
`{{input}}` to combine both).

How it is enforced:

1. Providers with native structured output get `response_format:
   {type: json_schema, ...}`, so the reply is constrained while decoding. A
   dialect whose endpoint has no `json_schema` sends the JSON mode it does have
   (`{type: json_object}`) instead, and a provider that rejects the schema
   outright is detected and retried without it - automatically, no config
   needed.
2. The answer is validated locally either way. A fenced answer
   (```` ```json … ``` ````) or one padded with prose is unwrapped; stdout gets
   the extracted JSON, never the model's framing.
3. A violation is fed back to the model as a repair message, up to
   `max_schema_retries` times. Retries are ordinary turns, so `max_turns` and
   `max_tokens` still bound them.
4. If no answer ever validates: **exit 6**, and stdout stays empty. A pipeline
   never sees partial or invalid JSON.

An `output.schema` that cannot compile is a config error: `amele validate`
catches it, and `amele run` fails with exit 2 before spending a token.

Which providers enforce a schema natively and which leave it to steps 2-4 is in
[providers.md](providers.md#structured-output), together with a warning line on
stderr whenever the native path was unavailable.

**A schema constrains shape, not content.** stdout is printed raw: unlike the
session log, it is **not** secret-redacted. If untrusted content can reach the
model (scraped pages, log lines, issue bodies) and stdout feeds a cron mail or
a CI log, `output.schema` is the right tool for bounding *what shape* lands
there - but it cannot stop a secret sitting inside a string field, because a
secret is a perfectly valid string. Keep secrets out of what the agent can read
in the first place (the `env` allowlist on shell and subprocess tools, see
[shell-tool.md](shell-tool.md)) and treat every stdout sink as somewhere the
model can write (see [threat-model.md](threat-model.md) §4.6).

## Permission profiles (`permissions`)

Per-tool approval policy with one fallback:

```yaml
permissions:
  default: ask            # allow | ask | deny - absent block means allow
  tools:
    fs_read: allow
    fs_write: ask
    shell: deny
```

- `allow` - the call runs.
- `ask` - the operator is asked on stderr (`amele: allow tool fs_write with
  {"path":"out.txt"}? [y/N]`). Only `y`/`yes` approves; a blank line, anything
  else, or EOF is a refusal.
- `deny` - the call never runs. The model is told "permission denied" as a tool
  result and keeps going with the tools it still has, so a denial does not kill
  the run.

A key may contain `*` and then matches tool names by pattern - useful when one
server contributes many tools (`github__*`). Precedence is fixed and does not
depend on the order you write the entries: an exact key wins over any pattern,
and among matching patterns the most restrictive policy wins (`deny` > `ask` >
`allow`), so a narrow `"*_delete*": deny` is never overridden by a broad
`"github__*": allow`.

```yaml
permissions:
  default: allow
  tools:
    "github__*": ask      # every tool of the github server
    "*_delete*": deny     # ...except deleting anything, ever
```

The `<server>__<tool>` names come from MCP servers declared in the same file;
[docs/mcp.md](mcp.md) covers connecting them, and why a server's own
`destructiveHint` annotation is shown to you but never changes a ruling.

**TTY fail-safe (headless-first):** when stdin is not a terminal - cron, CI, a
pipe - `ask` degrades to `deny` automatically and the reason is logged to
stderr. A headless agent can never block waiting for a human that is not there,
and it can never silently grant itself the tool either.

Tool names and arguments printed in the question come from the model, so they
are stripped of control bytes and clipped before they reach the terminal: a
prompt-injected model cannot redraw the approval dialog.

An invalid policy value is a config error (exit 2) at load time.

## The `shell` tool

Disabled by default; it exists only when the config says so.

```yaml
tools:
  shell:
    enabled: true
    allow: ["git status", "git diff*", "ls*"]
    deny:  ["rm *", "git push*"]
    timeout: 60s
```

Patterns are matched per line of the command (`*` matches any substring), deny
first. Each call is a fresh `sh -c` in the workspace: `cd` and variables do not
persist.

**The allow/deny lists are accident prevention, not a security boundary.** A
pattern like `git *` still reaches arbitrary code through `git -c
core.fsmonitor=cmd`, aliases and hooks. The boundary is the OS/container the
agent runs in. Read [docs/shell-tool.md](shell-tool.md) before enabling it -
it covers the matching rules and the full security model, including the
environment the command inherits.

Combine with permissions for a second pair of eyes:

```yaml
permissions:
  tools:
    shell: ask
```

## Interactive mode (`amele chat`)

```console
$ amele chat assistant.yaml
> summarize the last 5 commits
The last five commits ...
> and who wrote them?
Alice wrote three ...
> ^D
✓ 4 turns, 2 tool calls, 12.4k tokens, 9.8s
```

Same YAML, same tools, same permissions as `run`. Details that matter:

- **stdout is still the answer channel.** The `> ` prompt, approval questions
  and the summary go to stderr, so `amele chat cfg.yaml < script.txt >
  answers.txt` captures the answers and nothing else. Treat that file as a
  *stream*: each answer is followed by a newline, but an answer routinely spans
  several lines, so splitting stdout by line only recovers answer boundaries
  when every answer happens to be single-line. There is no delimiter - when a
  script needs a parseable boundary, use `amele run` (one answer per process,
  and `output.schema` if it must be JSON).
- **Ctrl-D (EOF) exits 0** and prints the cumulative summary. Empty lines cost
  nothing - no provider call, no turn, no tokens.
- **One budget pool for the whole session.** `limits.max_turns` and
  `limits.max_tokens` are cumulative across every line, not per line: when the
  pool is empty the session ends with exit 3. `limits.timeout`, by contrast,
  bounds **one exchange** - a human thinking at the prompt must not burn the
  run timeout.
- **One session file per chat**, opened with a `run_start` whose task is
  `interactive chat` and closed with a single `run_end` carrying the session
  totals.
- **`ask` policies work interactively** here, because stdin is a terminal. The
  REPL and the approval prompt share one reader, so answering a question never
  eats your next chat line.
- Any error ends the session with its usual exit code (5 provider, 4 permission
  abort, 3 budget, 1 interrupted/failed) - the exit code contract is identical
  to `run`.

### Two deliberate limitations

**`output.schema` is ignored in chat.** It constrains a one-shot answer; there
is no single output in a conversation, and validating every line would spend
the session's budget arguing about JSON. amele says so on stderr and continues.
Use `amele run` when you want the schema enforced.

**Tool-call rounds are not kept in the transcript.** After each line the
history keeps your message and the model's *final answer*; the intermediate
tool calls and their results are dropped. The model sees its own conclusions
but not its scratch work - coherent for conversation, and it keeps the context
small. Full transcript continuity is on the roadmap.

## Parallel tool calls (`tools.parallel`)

Models ask for several tools in one turn - three file reads, two searches, a
fetch and a grep. amele runs them **at the same time**, so the turn costs the
slowest call instead of the sum of all of them.

It is on by default. Turn it off per config:

```yaml
tools:
  parallel: false   # one tool call at a time (default: true)
```

**The recorded order never changes.** Whatever order the tools finish in, the
session log, the message history sent back to the model and the `-v` progress
lines all appear in the order the *model asked for the calls*
([JSONL contract](contracts/jsonl-events.md#ordering-guarantees)). Two runs of
the same recorded conversation therefore produce their events in the same
ORDER - the timestamps still differ, so the files are not byte-identical -
and concurrency buys latency, never a different transcript.

**When amele falls back to one at a time**, automatically:

- the turn has only one tool call (nothing to overlap);
- `tools.parallel: false`;
- **any** call in the turn is governed by an `ask` permission. Two approval
  questions racing for one terminal would be unanswerable: you could not tell
  which call you just granted. One `ask` in the turn puts the whole turn back
  on the sequential path, and the questions arrive one after the other, in
  call order.

Per-tool timeouts are unchanged: each subprocess, shell command and MCP call
keeps its own clock, and `limits.timeout` still bounds the whole run.

**At most 8 calls run at once.** Providers ask for a handful of tools per turn,
so this ceiling is invisible in practice - but the length of that list is
model output, and a broken or hostile response could otherwise start a thousand
subprocesses at once. A wider turn runs in waves instead: every call still
runs, and the recorded order is still the model's call order.

**What to check before leaving it on.** amele's own tools are independent by
construction (a separate process or request per call), but *your* tools might
not be: two subprocess tools appending to the same file, a script with a lock
file, an MCP server that serializes requests badly. Those are the cases
`parallel: false` exists for.

## Session logs

All of the above is recorded when `session_dir` is set (one JSONL file per run
or chat session). Note the redaction caveat before interpolating non-secret
variables like `${HOME}`: see
[docs/session-logging.md](session-logging.md).

A `tool_result` whose text was cut before the model read it carries
`truncated: true`, and the text itself contains `[output truncated by amele]`
(`fs_list` appends its entry counts to the same marker:
`[output truncated by amele: N of M entries shown]`) - at the end of the
result for most cuts, but mid-result where framing follows the cut stream: a
failed command whose stdout was cut renders its `stderr:` section after the
marker. The cap is per tool family - `fs_read` 256 KiB, subprocess and shell
64 KiB per stream, `fs_list` 64 KiB of directory entries, MCP 64 KiB - unless
`limits.max_tool_result_bytes` sets one number for all of them and for the
framed result the loop hands back:

```yaml
limits:
  max_tool_result_bytes: 16384   # every tool family; minimum 1024
```

One number covers both cuts, and on a failed command the two compete: stdout is
rendered first and may spend the whole budget, after which the loop's ceiling
can cut the `stderr:` section away entirely. Raise the number when a failing
command's stderr is the part you need.

There is deliberately no unbounded setting - a tool result with no bound is
the one thing that can spend a whole context window in a single call - so the
minimum is 1024 and omitting the key keeps the built-in caps byte for byte.
`amele explain` prints the effective value either way. Do not confuse this cut
with the log's own per-field clip (`limits.max_logged_field`): that one
shortens what the FILE stores, after the model has already read the text, and
never sets `truncated`.

Each `llm_response` also records what the provider's prompt cache did for that
turn: `cache_read_tokens` (input served from the cache) and
`cache_write_tokens` (input billed to populate it), both a share of the turn's
`input_tokens` rather than an addition to it, and both absent when nothing was
cached, the endpoint reports no such count, or the log predates v1.7.
`run_end.cache_read_tokens` is the run's total. The same fact
reaches the terminal: when a run read anything back from a cache, the summary
line's token figure carries a parenthetical -
`✓ 8 turns, 3 tool calls, 41.0k tokens (28.0k cached), 34.2s` - and a run with
no cache reads prints the line exactly as before. Where the caching comes from
depends on the wire: amele places the markers itself on the anthropic wire
(`provider.prompt_cache`, on by default), while every other endpoint decides on
its own - see [docs/providers.md](providers.md#prompt-caching).
