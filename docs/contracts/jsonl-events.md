# JSONL event schema

**v1.5 - FROZEN as of v0.1; `tool_result`'s `outcome`, `exit_code` and
`result_bytes` (v1.1), the MCP events plus `run_end.mcp_errors` (v1.2),
`mcp_connect.auth` (v1.3), `llm_response.reasoning_bytes` (v1.4) and the
opt-in `llm_response.reasoning` (v1.5) added
additively (every v1 field unchanged, and the
on-the-wire `v` stays `1`).** This is the format of the session log: one append-only JSONL
file per run or chat session, written when `session_dir` is set. Log, session
and (future) replay input are deliberately the same format. Source of truth:
`session.Event` in `internal/session/session.go`.

## File

- Path: `<session_dir>/run-<UTC timestamp>-<pid>.jsonl`
  (e.g. `run-20260811T090412Z-31337.jsonl`); filenames sort chronologically
  with plain `ls`, the PID disambiguates same-second starts.
- Permissions: directory `0750`, file `0600` - session logs can carry
  sensitive tool output.
- The file is opened with `O_EXCL` (a name collision fails loudly rather than
  interleaving two runs) and `O_APPEND`: events are never mutated, and the
  kernel positions every write at the current end of file, so amele cannot
  overwrite bytes it did not write even if something else grows the file.
- Write errors after the file is open are swallowed: a full disk degrades
  observability, it never aborts a running agent. The flip side: a hard kill
  (SIGKILL, power loss) can leave a file without its `run_end`.

## Envelope

Every line is one JSON object with three always-present fields:

| Field | Type | Meaning |
|-------|------|---------|
| `v` | int | Wire schema version. Always `1` for this document - the `v1.5` above is this document's revision, and additive changes deliberately leave `v` alone (a bump means a consumer must be rewritten). |
| `type` | string | Event type: `run_start`, `llm_response`, `tool_call`, `tool_result`, `mcp_connect`, `mcp_tools_listed`, `mcp_disconnect`, `run_end`. |
| `ts` | string | Event time, RFC 3339 UTC (Go `time.Time` JSON encoding). |

All other fields are declared with `omitempty`: **a zero value is omitted**.
Consumers must treat an absent numeric field as `0`, an absent boolean as
`false`, an absent string as `""`, an absent array as empty.

## Event types

### `run_start` - first event of every file

| Field | Type | Meaning |
|-------|------|---------|
| `model` | string | Model identifier the run was started with (after any `--model` override). |
| `task` | string | The rendered user task (clipped + redacted, see below). For a chat session this is the fixed label `interactive chat`. |

### `llm_response` - one per provider round-trip

| Field | Type | Meaning |
|-------|------|---------|
| `turn` | int | 1-based turn number, strictly increasing within the run (see [Turn numbering](#turn-numbering)). |
| `content` | string | The assistant's text for this turn (clipped + redacted). Absent when the model sent only tool calls. |
| `tool_call_ids` | string[] | IDs of the tool calls requested in this same message; absent when there are none. Each ID reappears as `tool_call_id` on the matching `tool_call`/`tool_result` events. |
| `input_tokens` | int | Provider-reported input tokens for **this** round-trip (not cumulative). |
| `output_tokens` | int | Provider-reported output tokens for this round-trip. |
| `finish_reason` | string | The provider's finish reason verbatim (`stop`, `length`, `tool_calls`, `content_filter`, `error`, ...); may be absent when the provider omits it. |
| `reasoning_bytes` | int | Byte length of the provider's reasoning payload for this turn (a DeepSeek `reasoning_content`, Anthropic thinking blocks, an OpenRouter `reasoning_details` array), as amele stored it. Absent means the turn carried no reasoning - or the log predates v1.4. **The reasoning CONTENT is not logged by default**: it is the model's unfiltered scratchpad, it can restate a secret in words the value redactor never sees, and replay does not need it. `log_reasoning: true` opts it in (v1.5, the `reasoning` field below) through the same redact+clip path as every other free-text field; the rationale above is why that is a deliberate data-governance decision rather than a default. This number is what answers "why did that turn cost so much?" - echoed reasoning is billed as input on every later turn, and it answers it whether or not the content is logged. Since v1.4. |
| `reasoning` | string | The turn's reasoning payload, as the provider sent it, rendered as text (clipped + redacted like every other free-text field). Written **only** when the config sets `log_reasoning: true` **and** the turn carried a payload, so absence has three readings - the run did not opt in, the turn did no thinking, or the log predates v1.5 - and `reasoning_bytes` is what separates them: a positive `reasoning_bytes` with no `reasoning` means the content was not written (opted out, or a pre-v1.5 log), both absent means the turn carried no reasoning at all. The value is the provider's RAW payload, not prose: on a wire whose payload is a JSON string (a DeepSeek/GLM/Kimi `reasoning_content`) the logged text INCLUDES that JSON's own quoting and escapes - a `reasoning_content` of `first I considered...` is logged as the text `"first I considered..."`, quotes and all - while the anthropic wire logs the raw content-blocks JSON array and the gemini wire the raw parts JSON array. A consumer must parse it as the provider's payload for that wire and never read it as plain text - but a **clipped** value is a byte prefix plus the marker and will not parse at all (most visibly for the two array-shaped wires), so a consumer that needs to parse every turn sets `limits.max_logged_field: 0` and keeps the payload whole. Since v1.5. |

### `tool_call` - the model requested a tool

| Field | Type | Meaning |
|-------|------|---------|
| `tool_call_id` | string | Links this event (and its result) back to the requesting `llm_response.tool_call_ids` entry. |
| `tool` | string | Tool name as the model wrote it - it may name a tool that does not exist. |
| `args` | string | The raw JSON argument string (clipped + redacted). |

A `tool_call` is logged **before** the permission check, so denied and
unknown-tool calls appear in the log too - that is the audit trail.

### `tool_result` - the outcome of that call

| Field | Type | Meaning |
|-------|------|---------|
| `tool_call_id` | string | Same ID as the `tool_call` it answers. |
| `tool` | string | Tool name, repeated for greppability. |
| `result` | string | The text handed back to the model (clipped + redacted): tool output, an `error: ...` message, or a permission-denial notice. |
| `is_error` | bool | Present (`true`) for **harness dispatch failures only**: the call never produced a tool's own output. Absent otherwise. See below. |
| `outcome` | string | How the call ended, from the fixed enum below. Since v0.1.0; absent in logs written before it. |
| `exit_code` | int | The command's exit status. Present **only** when a subprocess/shell tool actually ran and its status is known - i.e. together with `outcome: nonzero_exit`. A child killed by a signal reports `-1` here (Go's exit status for "died on a signal"), which is still `nonzero_exit`: the command ran, it just has no clean status. Since v0.1.0. |
| `result_bytes` | int | Byte length of the **full** result text before session clipping (and before redaction, which rewrites lengths). Compare with `result`: the log keeps `limits.max_logged_field` bytes per field (8 KiB unless the config says otherwise) while the model may have read considerably more - the subprocess/shell cap is 64 KiB **per stream**, so a failed command's result carries up to 64 KiB of stdout plus 64 KiB of stderr plus the `exit status`/`stdout:`/`stderr:` framing. This field is what makes that loss visible. Absent for an empty result. Since v0.1.0. |

#### `is_error` vs `outcome`

`is_error` is **frozen** and narrow: it marks a call the harness could not
dispatch - an unknown tool, unusable arguments, a tool invocation error, a
denied call, or a broken approval check. A tool that *ran* and *failed* is not
an error here: `grep` exiting 1 because it matched nothing is ordinary task
information the model reacts to, and the result text says so in the words the
model read.

`outcome` is the observability field. It answers "what actually happened",
which `is_error` cannot: before it existed, a rejected command, a timeout and a
non-zero exit were all indistinguishable from a clean run in the log.

| `outcome` | Meaning | `is_error` |
|-----------|---------|------------|
| `ok` | The tool did its job. | absent |
| `timeout` | The **tool's own** timeout killed the command. | absent |
| `nonzero_exit` | The command ran and exited non-zero; `exit_code` carries the status, or `-1` when a signal killed the child instead of it exiting on its own. | absent |
| `aborted` | The **run** ended under the command - its overall timeout, or SIGINT/SIGTERM - not the tool's own budget. | absent |
| `denied_policy` | An operator policy refused the call before it ran. | see below |
| `denied_no_tty` | An `ask` policy auto-denied because no TTY was attached to ask a human on (the headless fail-safe). | `true` |
| `ask_refused` | A human was asked and said no. | `true` |
| `error` | The harness could not dispatch the call at all. | `true` |
| `tool_error` | An MCP tool ran and reported its own failure (`isError` in the MCP result). Like `nonzero_exit`, this is the tool's answer, not a harness failure. Since v1.2. | absent |
| `indeterminate` | The response was lost after the request was sent; the side effect may or may not have happened; amele never retries. Since v1.2. | absent |

`denied_policy` covers the **two** operator policies that refuse a call before
it runs, because both make the same statement about the call: the permission
profile's `deny` (including the default policy for an unlisted tool), and the
`shell` tool's allow/deny patterns. They stay distinguishable in the event
itself - the permission gate sets `is_error: true` and writes a
`permission denied` result, while a shell rejection is content the model reads
and adapts to (`is_error` absent, `tool: shell`, result starting
`command rejected by shell policy`).

Consumers must treat an **unknown** `outcome` value as "something else
happened", never as a failure to parse: new values may be added additively.

### `mcp_connect` - one connection attempt to one MCP server

| Field | Type | Meaning |
|-------|------|---------|
| `server` | string | The server's name from the config. |
| `transport` | string | How it was reached: `stdio` or `http`. |
| `ok` | bool | Whether the handshake completed. **Always present**, including `false` (like `run_end.exit_code`, it is exempt from omit-zero). |
| `error_class` | string | Failure group for aggregation; one of `spawn`, `network`, `auth`, `protocol`, `timeout`. Absent on success. |
| `error` | string | Human-readable failure text (clipped + redacted). Absent on success. |
| `duration_ms` | int | How long the attempt took, in milliseconds. |
| `protocol_version` | string | MCP protocol version agreed on. Absent on failure. |
| `server_name` | string | The server's self-reported name (may differ from `server`, which is the config's name). |
| `server_version` | string | The server's self-reported version. |
| `session_fp` | string | Short SHA-256 fingerprint of the MCP session id, for correlating events across a run. The raw session id is a bearer credential and is **never** written to the log. |
| `tool_count` | int | How many tools this server contributes **after** filtering - the count of definitions actually exposed to the model, not the raw advertised count. |
| `auth` | string | The credential mechanism the attempt used: `oauth`, or **absent** when the server needed none (a static `Authorization` header counts as none - nothing about it is amele's to manage). Present on failed attempts too, so "the oauth server did not come up" is distinguishable from "the plain server did not come up". SECURITY: the mechanism only - no token, issuer, scope or expiry is ever written here. Since v1.3. |

### `mcp_tools_listed` - the tool inventory taken from one server

| Field | Type | Meaning |
|-------|------|---------|
| `server` | string | The server's name from the config. |
| `tools` | object[] | The definitions actually exposed to the model (see below). |
| `total_bytes` | int | Summed size of **every** definition the server sent, skipped and filtered ones included - the bytes that crossed the wire, not the tokens the model is charged for. |
| `skipped` | object[] | Advertised tools that were **not** exposed: `{ "name": <server's name>, "reason": <one of "not included", "excluded", "definition too large", "invalid output schema", "input schema not an object"> }`. Absent when nothing was skipped. (A name conflict is never skipped - it is a fatal config error, exit 2.) |

Each `tools` entry:

| Field | Type | Meaning |
|-------|------|---------|
| `name` | string | The tool name as amele exposed it to the model (server-prefixed, normalized to the provider's allowed character set when needed). |
| `original_name` | string | The server's own name, present **only** when normalization rewrote it. |
| `sha256` | string | Hex SHA-256 of the tool definition amele sent to the model - proof after the fact of *which* definition the model was shown, since a server may change a description between runs. |
| `bytes` | int | Size of that definition. |
| `annotations` | object | MCP tool hints amele understood, as `name -> bool`: `readOnly`, `destructive`, `openWorld`, `idempotent`. Absent entirely when the server sent no annotations object at all. |

Presence in `annotations` does not mean the same thing for all four keys,
because the SDK models two of them as plain booleans:

- `readOnly` and `idempotent` are present whenever the server sent **any**
  annotations object - their value is the SDK's zero (`false`) when the server
  said nothing about them, so presence proves only that annotations exist.
- `destructive` and `openWorld` are present **only** when the server actually
  stated that hint; an absent key means "the server said nothing", not `false`.

### `mcp_disconnect` - a server connection ended

| Field | Type | Meaning |
|-------|------|---------|
| `server` | string | The server's name from the config. |
| `reason` | string | `run_end` (the run finished) or `reconnect` (amele is re-establishing the connection). |

### `run_end` - last event of every file

| Field | Type | Meaning |
|-------|------|---------|
| `status` | string | `success` or `error`. |
| `exit_code` | int | The process exit code per the [exit code contract](exit-codes.md). **Always present**, including `0` (it is the one field exempt from omit-zero). |
| `turns` | int | **Attempted** provider round-trips. An attempt is counted before its provider call, so when the final attempt fails (e.g. a provider error) `turns` can exceed the highest logged `turn` by one. |
| `tool_calls` | int | Tool calls dispatched - equals the number of `tool_result` events. Denied-and-continued calls, unknown-tool recoveries and erroring tools all count; only a call whose denial aborts the run is logged (as a `tool_result`) but not counted. |
| `total_tokens` | int | Cumulative input + output tokens. |
| `mcp_errors` | int | MCP-attributable failures over the whole run (failed connects, lost responses, tool-listing failures). Absent means **0**. Since v1.2. |
| `duration_ms` | int | Loop time in milliseconds, **not** process wall clock: `run` measures the agent loop only (config loading and the stdin read are excluded), and `chat` sums the per-exchange loop durations - time idle at the prompt is excluded. |

`run_end` is written for failed runs too - especially for failed runs - with
truthful partial accounting.

## Ordering guarantees

- Events are written in strict chronological order; `ts` is non-decreasing.
- Exactly one `run_start` (first line) and at most one `run_end` (last line;
  absent only after a hard kill) per file.
- All MCP events (`mcp_connect`, `mcp_tools_listed`, `mcp_disconnect`) occur
  strictly between `run_start` and `run_end`. The **initial** connects and
  tool listings happen before the first `llm_response` of the run. After that
  a lost session may add further `mcp_disconnect` (reason `reconnect`) /
  `mcp_connect` pairs at any point between `run_start` and `run_end` - with
  `ok` false when the reconnect itself failed. The
  orderly shutdown emits a `mcp_disconnect` (reason `run_end`) for every
  still-connected server before `run_end`.
- Within a turn: the `llm_response` comes first, then its tool calls. Both
  `tool_call` and `tool_result` events appear in the order the model requested
  the calls in that `llm_response` (the same order as its `tool_call_ids`),
  whether the calls ran one after the other or side by side. Correlation is by
  `tool_call_id`, not by position.
- Two shapes are possible for the tool events of one turn, and a consumer must
  accept both:
  - **sequential dispatch** - each `tool_call` is immediately followed by its
    own `tool_result` (`call c1, result c1, call c2, result c2`). This is what
    a single-call turn, `tools.parallel: false`, or any turn containing an
    `ask`-governed call produces;
  - **parallel dispatch** (default since v0.2, `tools.parallel`) - every
    `tool_call` of the turn is logged first, in call order, then every
    `tool_result`, also in call order (`call c1, call c2, result c1,
    result c2`).

  In both shapes every `tool_call` is answered by exactly one `tool_result`,
  even on the turn that aborts the run. What changes is only the interleaving;
  the order *within* each group is the model's call order, never completion
  order, so a log does not record which call happened to finish first. `ts`
  stays non-decreasing in both shapes: the events are written when they are
  published, not when the tool returned.

## Turn numbering

`turn` is strictly monotonic within one file. A one-shot `run` numbers from 1.
A `chat` session drives many exchanges inside **one** `run_start`/`run_end`
pair; the loop offsets each exchange's turns by the turns already consumed
(`loop.TurnBase`), so numbering continues - line one logs turns 1..n, line two
continues at n+1. `run_end.turns` counts **attempted** round-trips in the same
numbering: on a successful run it equals the highest `turn` in the file, but
when the final attempt failed before its `llm_response` was logged (a dead
endpoint logs a `run_start`/`run_end` pair with `turns: 1` and no
`llm_response` at all) it exceeds the highest logged `turn` by one.

The counter alone therefore cannot tell the two cases apart: within the file,
an attempted-but-unanswered turn is distinguishable from an answered one only
by the **absence of its `llm_response` event**. A consumer that needs answered
round-trips must count `llm_response` events rather than read `run_end.turns`.

## Clipping and redaction

Every free-text field is bounded and scrubbed before it hits disk - `task`,
`content`, `args`, `result`, `error` and the opt-in `reasoning`. The bound is
per field and applies to all of them, not to `result` alone:

- **Clipping:** the payload is bounded to `limits.max_logged_field` bytes per
  field (default 8192; `0` disables the bound and writes every field whole),
  cut at a UTF-8 rune boundary; when text is clipped the marker `...[clipped]`
  is appended on top, so a clipped field value is up to the bound plus that
  12-byte marker.
- **Redaction:** every value interpolated into the config via `${VAR}` - plus
  `provider.api_key` - is replaced by `[REDACTED]` wherever it appears, by
  value. This catches secrets arriving through any channel (tool output, model
  echoes), and it means interpolating a broad non-secret like `${HOME}` will
  redact every absolute path in the log - see
  [docs/session-logging.md](../session-logging.md).

Redaction runs before clipping, so a secret can never survive by sitting on
the clip boundary - and it runs unconditionally, before the bound is even
consulted, so `limits.max_logged_field: 0` widens the record without weakening
the scrubbing.

## Change policy

Within v1, changes are **additive only**: new event types and new optional
(omitempty) fields may appear, and consumers must ignore unknown fields and
unknown event types. Removing a field, renaming one, changing a field's type
or meaning, or changing the guarantees above bumps `v` and requires a semver
major, a `contract:`-titled PR and a migration note in `docs/contracts/`.

## Change log

### v1.1 (amele v0.1.0) - `tool_result` observability (additive, `v` stays `1`)

Added three optional fields to `tool_result`: `outcome`, `exit_code` and
`result_bytes`. Nothing was removed, renamed or re-typed, no other event type
changed a byte, and `is_error` keeps the exact meaning it always had - the
prose above only states it precisely, it does not narrow it.

**Migration:** none required. A consumer written against the earlier schema
keeps working (the fields are additional keys it ignores); a consumer that
wants them must tolerate their absence, which is also how it reads a log
written by an older amele. Concretely:

- absent `outcome` means the log predates v0.1.0 - it does **not** mean the
  call succeeded, and a reader must not infer one;
- absent `exit_code` on a `tool_result` means no process ran or amele never
  collected a status (a command stopped by its own `timeout` or by the run
  ending is reported as `timeout`/`aborted`, not as an exit);
- absent `result_bytes` means an empty result (or a pre-v0.1.0 log).

### v1.2 (amele v0.2.0) - MCP observability (additive, `v` stays `1`)

Added three event types - `mcp_connect`, `mcp_tools_listed`,
`mcp_disconnect` - one optional field to `run_end` (`mcp_errors`), and two
values to the `tool_result.outcome` enum (`tool_error`, `indeterminate`).
Nothing was removed, renamed or re-typed, and no existing event type changed a
byte.

**Migration:** none required. A consumer written against v1.0/v1.1 keeps
working: it must already ignore unknown event types and unknown fields, which
is exactly what the additions are. A consumer that wants them must tolerate
their absence, which is also how it reads a log written by an older amele.
Concretely:

- absent `mcp_errors` on `run_end` means **0** MCP-attributable failures (and
  is also what every pre-v0.2.0 log says);
- absent MCP events mean the run configured no MCP servers - or predates
  v0.2.0;
- the two new `outcome` values follow the standing rule: an unknown `outcome`
  is "something else happened", never a parse failure.

### v1.3 (amele v0.2.0) - MCP OAuth (additive, `v` stays `1`)

Added one optional field to `mcp_connect`: `auth`. Nothing was removed,
renamed or re-typed, no other event type changed a byte, and no new event type
appeared - an OAuth login is deliberately **not** an event: it happens before
`run_start` (the run's session file does not exist yet), and the only thing
worth recording afterwards is which mechanism authenticated the connect.

**Migration:** none required. Concretely:

- absent `auth` means the connect used no credential amele manages - or the
  log predates v0.2.0. It never means "unknown";
- the value is a mechanism from a closed set (`oauth` today). A consumer must
  treat an unknown value as "some other mechanism", never as a parse failure;
- a token value is not in the log and never will be: a reader that needs the
  expiry or the issuer reads `amele mcp status`, not the session file.

### v1.4 (amele v0.2.0) - reasoning observability (additive, `v` stays `1`)

Added one optional field to `llm_response`: `reasoning_bytes`. Nothing was
removed, renamed or re-typed, no other event type changed a byte, and no new
event type appeared.

**Migration:** none required. Concretely:

- absent `reasoning_bytes` means the turn carried no reasoning payload - or
  the log predates v1.4. It never means "unknown";
- it is a SIZE and stays one. It is not the length of the `reasoning` string
  v1.5 added: it is measured on the payload amele stored, before redaction
  rewrites lengths and before the clip bound shortens anything, so the two
  numbers differ whenever either applies;
- the reasoning content is not in the log by default and the size is what a
  consumer can always count on: `reasoning` appears only under
  `log_reasoning: true` (v1.5).

### v1.5 (amele v0.2.x) - opt-in reasoning content (additive, `v` stays `1`)

Added one optional field to `llm_response`: `reasoning`. Nothing was removed,
renamed or re-typed, no other event type changed a byte, and no new event type
appeared. The default is unchanged: without `log_reasoning: true` in the
config, a v1.5 amele writes exactly the bytes a v1.4 one did.

The same revision makes the clip bound configurable
(`limits.max_logged_field`, default 8192, `0` unbounded). That is not a schema
change - no field's type or meaning moved - but it does change what a reader
may assume about a field's length: a value is no longer guaranteed to be at
most 8204 bytes, and a consumer that sized a buffer on the old constant must
stop.

**Migration:** none required. Concretely:

- absent `reasoning` is a disjunction, not a verdict: the run did not opt in,
  the turn carried no reasoning, or the log predates v1.5. Read it with
  `reasoning_bytes`, which is written whether or not the content is - positive
  `reasoning_bytes` and no `reasoning` means the content was withheld, both
  absent means there was none. (An opted-out run and a pre-v1.5 log look the
  same from inside the file; the config that produced the run is what tells
  them apart);
- the value is the provider's raw payload rendered as text, not prose. Parse it
  as that wire's shape (see the field's note above) - reading it as a plain
  sentence will hand you the payload's own JSON quoting;
- it is clipped and redacted like every other free-text field, so a logged
  payload can be a prefix of what the model produced - and a clipped value is a
  byte prefix with a marker glued on, which no JSON parser will accept.
  `limits.max_logged_field: 0` is how a consumer that must parse every payload
  asks for whole ones; `reasoning_bytes` remains the honest size either way;
- a consumer that must not touch model scratchpads should ignore the key. Its
  presence is the operator's explicit decision, recorded in the config that
  produced the run.
