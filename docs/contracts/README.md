# Frozen contracts

This directory versions amele's public API. **Frozen** means: scripts, cron
jobs, editors and pipelines may depend on these surfaces, and a breaking
change to any of them requires a **semver major** bump, a PR titled
`contract: ...` that contains only the contract change, and a migration note
in this directory (project constitution §7/§9).

The four contract artifacts:

| Artifact | Surface |
|----------|---------|
| [exit-codes.md](exit-codes.md) | The 0..8 process exit code table (v1.2). |
| [jsonl-events.md](jsonl-events.md) | The session log event schema (`v: 1`, doc revision v1.7). |
| [cli.md](cli.md) | Commands, flags, stdin/stdout/stderr behavior. |
| [config.schema.json](config.schema.json) | The YAML config schema (JSON Schema, also printed by `amele schema`). |

## Additive-change policy

Within a frozen version, changes must be additive and backwards-compatible:

- **exit codes** - new codes for new outcomes may be added; existing codes
  never change meaning. Treat unknown non-zero codes as failure.
- **JSONL events** - new event types and new optional (`omitempty`) fields
  may be added; consumers must ignore what they do not know. Removals,
  renames and meaning changes bump `v`.
- **CLI** - new commands and new optional flags may be added; existing names
  and behavior stay.
- **config schema** - new optional keys may be added; existing keys keep
  their type and meaning. (Unknown keys are rejected at load time, so an old
  binary fails loudly on a newer config rather than misreading it.)

## Pre-v0.1 behavior changes

- **2026-08-12 - subprocess `command[0]` resolution.** A relative
  `command[0]` containing a path separator now resolves against the config
  file's directory (previously: against the workspace via the child's
  working directory). Bare names still resolve from PATH. Migration: if a
  config relied on a workspace-relative script path, write the path as
  absolute or move the script next to the config file. Recorded here
  because v0.1 has not shipped; no semver impact.
- **2026-08-12 - `tool_result` observability fields.** The JSONL
  `tool_result` event gained `outcome`, `exit_code` and `result_bytes`
  (jsonl-events.md v1.1). Additive: the wire `v` stays `1`, nothing was
  removed, renamed or re-typed, and a consumer that ignores unknown keys is
  unaffected. Migration: treat all three as optional - absent `outcome` means
  the log predates them, not that the call succeeded.
- **2026-08-12 - schema encodes the non-empty executable rule.** The
  published schema now rejects `command: [""]` (`prefixItems` +
  `minLength: 1`), matching what `amele validate` already enforced at
  runtime. Documents the existing runtime contract; no behavior change.
- **2026-08-18 - schema encodes the env-names-not-assignments rule.** The
  `env` allowlists (`tools.shell.env`, `tools.subprocess[].env`) now carry
  `pattern: "^[^=]+$"`, so `env: ["PATH=/usr/bin"]` fails schema validation
  the same way `amele validate` already rejected it. Documents the existing
  runtime contract; no behavior change.
- **2026-08-19 - exit code 8, required MCP server unavailable.** The exit-code
  contract moves to **v1.2**: code `8` is added for "an MCP server declared
  with `required: true` could not be started, connected to or authenticated
  against". Additive - codes 0-7 are unchanged, and a script that does not
  know 8 sees a non-zero exit exactly as before. Migration: none required;
  callers that already special-case 5 as "retry" may want to route 8 to a
  human instead.
- **2026-08-19 - JSONL v1.2, MCP observability.** The session log gains three
  event types (`mcp_connect`, `mcp_tools_listed`, `mcp_disconnect`), the
  optional `run_end.mcp_errors` field, and two `tool_result.outcome` values
  (`tool_error`, `indeterminate`) - jsonl-events.md v1.2. Additive: the wire
  `v` stays `1`, nothing was removed, renamed or re-typed. Migration: none
  required; absent `mcp_errors` means 0, and an unknown `outcome` stays
  "something else happened".

- **2026-08-19 - JSONL v1.3, MCP OAuth.** `mcp_connect` gains the optional
  `auth` field, naming the credential mechanism a connect used (`oauth`, or
  absent) - jsonl-events.md v1.3. Additive: the wire `v` stays `1`, nothing was
  removed, renamed or re-typed, and no credential is written. Migration: none
  required; absent `auth` means "no mechanism amele manages", never "unknown".
- **2026-08-19 - `amele mcp login|status|logout`.** A new command for the
  OAuth credentials of MCP servers, plus an `auth:` row per OAuth server in
  the `amele explain` report - cli.md. Additive: no existing command, flag or
  report line changed. The `status` table's layout is explicitly
  informational and NOT frozen; the streams and exit codes are.

- **2026-08-24 - JSONL v1.4, reasoning observability.** `llm_response` gains
  the optional `reasoning_bytes` field: the SIZE of the provider's reasoning
  payload for that turn, not its content by default (v1.5 added the opt-in) -
  jsonl-events.md v1.4. Additive:
  the wire `v` stays `1`, nothing was removed, renamed or re-typed. Migration:
  none required; absent `reasoning_bytes` means the turn carried no reasoning.
- **2026-08-24 - config schema: the provider tuning surface.** `provider` gains
  five optional keys - `dialect` (enum: openai, deepseek, glm, kimi, groq,
  openrouter; omitted means openai), `reasoning` (`effort` enum + optional
  `budget_tokens`), `temperature`, `top_p` and the free-form `params` mapping.
  Additive: every key is optional, `additionalProperties: false` still holds,
  and a config that sets none of them behaves exactly as before.
  **One meaning changed:** `provider.max_output_tokens` was documented as
  "ignored by the OpenAI path" and is now honored there too, with the dialect
  picking the field name (`max_completion_tokens` or `max_tokens`). Migration:
  a config that set `max_output_tokens` while using the OpenAI path got the
  provider's server-side default and now gets the cap it asked for - review the
  value before upgrading, and remember that reasoning tokens are billed against
  it. Setting nothing keeps the old behavior (no cap is sent).
  The `--set` allowlist gains `provider.max_output_tokens`,
  `provider.reasoning.effort`, `provider.temperature` and `provider.top_p`
  (cli.md); no key was removed.
- **2026-08-24 - config schema: `provider.retry`.** `provider` gains one more
  optional block, `retry`, with `max_attempts` (0..10; 0 or omitted means the
  default 3, 1 disables retrying) and `initial_backoff` (a duration; empty or
  omitted means the default 1s, accepted range 100ms..60s). Additive: both keys
  are optional, `additionalProperties: false` still holds, and a config without
  the block retries exactly as every previous release did (3 attempts, 1s/2s
  backoff, `Retry-After` honored up to 60s). Migration: none required. The
  retryable failure classes (429, 5xx, network) are NOT configurable, and the
  `--set` allowlist is unchanged.

- **2026-08-25 - config schema: the Gemini wire family.** `provider.type`
  gains the enum value `gemini` (the native Google `generateContent` API,
  alongside `openai` and `anthropic`), and `provider` gains one optional block,
  `vertex` (`project` and `location`, both required inside the block, plus an
  optional `credentials` path), which points that wire at Vertex AI. Two
  cross-key rules are enforced at load time rather than left to the server:
  `dialect` is rejected with type `gemini` (a different wire family, so a
  dialect there would name a mapping the request never takes), and
  `reasoning.budget_tokens` and `reasoning.effort` are mutually exclusive on
  that wire (the API refuses a request carrying both). `params` reserves the
  gemini wire's own body keys on the active target (`contents`,
  `systemInstruction`, `tools`, `toolConfig`, `generationConfig`,
  `safetySettings`, `cachedContent`, in either spelling), and `vertex` is
  refused together with `api_key`. Additive: the new enum value and the new
  block are optional, `additionalProperties: false` still holds, and a config
  that names neither behaves exactly as before - every new rule fires only on
  a config that opts into the new wire. Migration: none required. The
  `--set` allowlist is unchanged. The same change documented two facts on the
  other surfaces without changing them: exit code 5 covers the third client
  (exit-codes.md), and `amele explain` gains the Gemini wire rows and the two
  Vertex rows (cli.md, additive).
- **2026-08-31 - JSONL v1.5, opt-in reasoning content.** `llm_response` gains
  the optional `reasoning` field: the provider's reasoning payload itself,
  written only when the config sets `log_reasoning: true` and the turn carried
  one, through the same redact+clip path as every other free-text field -
  jsonl-events.md v1.5. The same revision makes the per-field clip bound
  configurable (`limits.max_logged_field`), so a logged value is no longer
  guaranteed to be at most 8204 bytes. Additive: the wire `v` stays `1`,
  nothing was removed, renamed or re-typed, and a run that sets neither key
  writes exactly the bytes the previous release did. Migration: none required;
  absent `reasoning` is a disjunction - not opted in, no reasoning that turn,
  or a pre-v1.5 log - and `reasoning_bytes` is what separates them. The value
  is the provider's RAW payload rendered as text (a JSON-string carrier keeps
  its own quoting), and a clipped one is a byte prefix that will not parse.
- **2026-08-31 - config schema: session logging keys.** Three optional keys:
  `limits.max_logged_field` (integer or a whole-value `${VAR}`, minimum 0;
  omitted means the built-in 8192, `0` means unbounded), `log_reasoning`
  (boolean, default false) and `print_session_path` (boolean, default false -
  one `session log: <path>` note on stderr, suppressed by `-q`). Additive:
  every key is optional, `additionalProperties: false` still holds, and a
  config that sets none of them behaves exactly as before. Migration: none
  required. The `--set` allowlist gains `limits.max_logged_field` only
  (cli.md); `log_reasoning` and `print_session_path` are deliberately NOT
  settable - what a run persists is a data-governance decision the audited
  YAML owns - and no key was removed.
- **2026-09-06 - JSONL v1.6, tool-result truncation.** `tool_result` gains
  the optional `truncated` boolean: the text the model was handed was cut to
  a byte cap - by the tool family's built-in cap (fs_read 256 KiB,
  fs_list/subprocess/shell 64 KiB per stream, MCP 64 KiB) or by the new
  `limits.max_tool_result_bytes` - jsonl-events.md v1.6. The marker in the
  text said so to the model already; the field says so to a reader.
  Additive: the wire `v` stays `1`, nothing was removed, renamed or re-typed,
  and a run that never truncates writes exactly the bytes v1.5 wrote.
  Migration: none required; absent means not cut, or a pre-v1.6 log. The
  log's own per-field clip (`limits.max_logged_field`) is a different cut and
  never sets the flag.
- **2026-09-06 - config schema: `limits.max_tool_result_bytes`.** One optional
  key (integer or a whole-value `${VAR}`, minimum 1024): the byte cap on any
  single tool result the model reads, applied to every tool family and to
  the framed result the loop hands back. Omitted keeps each family's
  built-in cap byte-identically; there is deliberately no unbounded setting.
  Additive: the key is optional and `additionalProperties: false` still
  holds. Migration: none required. The `--set` allowlist gains
  `limits.max_tool_result_bytes` (cli.md) - a budget, like
  `limits.max_logged_field`: the worst an operator can do with it is make
  the model read more or less of a result; no key was removed.
- **2026-09-07 - JSONL v1.7, prompt-cache accounting.** `llm_response` gains
  `cache_read_tokens` and `cache_write_tokens` (both optional; the cached and
  cache-written shares of that turn's input, as the provider reported them),
  and `run_end` gains `cache_read_tokens` (the run's total) - jsonl-events.md
  v1.7. `input_tokens` keeps its meaning, the turn's total billed input; on the
  anthropic wire that total now includes the cached share the API reports
  separately, which the pre-v1.7 client left out (no request carried cache
  markers then, so nothing was left out in practice). Additive: the wire `v`
  stays `1`, nothing was removed, renamed or re-typed, and a run with no cache
  traffic writes exactly the bytes v1.6 wrote. Migration: none required;
  absent means 0.
- **2026-09-07 - config schema: `provider.prompt_cache`.** One optional boolean
  for the anthropic wire: when omitted or true the client places
  `cache_control: {type: ephemeral}` markers on the last tool definition,
  the system prompt (sent as a one-block array) and the last content block
  of the last message; false sends the pre-v0.3 request byte-for-byte. An
  explicit value with any other `provider.type` is a config error - caching
  is automatic on those wires and the key would be a silently dropped field.
  Additive to the schema: the key is optional and `additionalProperties:
  false` still holds. **One behavior changed:** every anthropic config that
  does not set the key now sends the markers, and its
  `llm_response.input_tokens` counts the cached share it never reported
  before (jsonl-events.md v1.7). Migration: none required for a multi-turn
  agent, where the second turn already pays back the 1.25x write premium; a
  single-turn config over the model's minimum cacheable prefix pays that
  premium with no read to recoup it - set `provider.prompt_cache: false`
  there. Not on the `--set` allowlist: it reshapes every request rather than
  retuning one.

## Schema versioning note (`$id`)

`config.schema.json` is **v1** even though it does not yet embed a `$id` /
version marker: its version is carried by this directory and by the release
the binary shipped in, not by the document itself. Embedding a canonical `$id`
URI (and with it an in-band version) changes the published document and is
deferred as a future `contract:` change.
