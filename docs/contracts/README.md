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
| [jsonl-events.md](jsonl-events.md) | The session log event schema (`v: 1`, doc revision v1.4). |
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
  payload for that turn, never its content - jsonl-events.md v1.4. Additive:
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

## Schema versioning note (`$id`)

`config.schema.json` is **v1** even though it does not yet embed a `$id` /
version marker: its version is carried by this directory and by the release
the binary shipped in, not by the document itself. Embedding a canonical `$id`
URI (and with it an in-band version) changes the published document and is
deferred as a future `contract:` change.
