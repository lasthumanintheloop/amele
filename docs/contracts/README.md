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
| [jsonl-events.md](jsonl-events.md) | The session log event schema (`v: 1`, doc revision v1.2). |
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

## Schema versioning note (`$id`)

`config.schema.json` is **v1** even though it does not yet embed a `$id` /
version marker: its version is carried by this directory and by the release
the binary shipped in, not by the document itself. Embedding a canonical `$id`
URI (and with it an in-band version) changes the published document and is
deferred as a future `contract:` change.
