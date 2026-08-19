# MCP client: connecting an agent to MCP servers

An amele agent can borrow tools from [Model Context
Protocol](https://modelcontextprotocol.io) servers. Declare the servers in the
same single YAML file; their tools join the builtin and subprocess tools in the
registry, under the same permissions, the same budgets and the same session
log.

The design centre is unchanged: **headless-first**. A run either starts with
exactly the toolset the config declared, or it fails loudly with an exit code a
scheduler can act on.

## What works today

- **Tools.** `tools/list` at startup, `tools/call` during the run. Nothing else
  from the protocol - see [Supported / not yet supported](#supported--not-yet-supported).
- **Two transports.** `stdio` (amele spawns the server as a child process) and
  Streamable HTTP.
- **Static authentication.** Extra HTTP headers, whose credential-shaped values
  must come from the environment (`${VAR}`), never from the file.
- **Tool filtering.** `include` / `exclude` globs per server, applied before the
  model ever sees a definition.
- **Deterministic naming.** Every tool is `<server>__<tool>`, rewritten by a
  documented rule when a provider would reject the raw name.
- **Permissions.** One mechanism for every tool: `permissions.tools`, with glob
  keys and a fixed precedence.
- **Pre-flight.** `amele explain` really connects, lists what each server would
  contribute, and estimates the token bill of the definitions.
- **Observability.** `mcp_connect`, `mcp_tools_listed`, `mcp_disconnect` events
  in the [JSONL session](contracts/jsonl-events.md), plus `run_end.mcp_errors`.
- **A dedicated failure code.** A `required` server that cannot be brought up is
  [exit 8](contracts/exit-codes.md#8---required-mcp-server-unavailable), distinct
  from a config error (2) and from a provider/network error (5).

## Config reference

```yaml
mcp:
  servers:
    - name: github                    # ^[a-z0-9_-]{1,32}$, unique
      transport:
        type: http                    # required: stdio | http
        url: https://api.example.com/mcp/
        headers:
          Authorization: "Bearer ${GITHUB_TOKEN}"
          X-Client-Name: amele
      tools:
        include: ["issue_*", "search_*"]
        exclude: ["*_delete*"]
      call_timeout: 60s               # default 60s
      required: true                  # default true

    - name: files
      transport:
        type: stdio
        command: ["/usr/local/bin/mcp-server-filesystem", "."]
        env: ["TZ"]                   # allowlist; empty = PATH/HOME/LANG only
```

| Key | Default | Meaning |
| --- | --- | --- |
| `mcp.servers[].name` | - | Required. `^[a-z0-9_-]{1,32}$`, unique, and it may not collide with a builtin (`fs_*`, `shell`) or subprocess tool name. It prefixes every tool the server contributes. |
| `transport.type` | - | Required: `stdio` or `http`. There is no default, because guessing from the other keys would make a typo mean something. |
| `transport.url` | - | Required for `http`. Must be `https://`, with an exception for loopback hosts (`localhost`, `127.0.0.1`, `::1`) so a local server can be developed against. Plain `http://` to any other host is rejected, not warned about. Userinfo in the URL (`https://user:pw@host/`) is rejected. |
| `transport.headers` | none | Extra request headers for `http`. Values of credential-shaped names (`authorization`, `cookie`, or a name containing `key`, `token`, `secret`, `passw`, `cred`, case-insensitive) must contain at least one `${VAR}` reference; a literal is a config error. Headers the transport manages itself (`Host`, `Content-Length`, `Content-Type`, `Accept`, `Mcp-Session-Id`, `MCP-Protocol-Version`, `Last-Event-ID`), duplicate names differing only in case, and values containing CR/LF are rejected. |
| `transport.command` | - | Required for `stdio`. An **argv vector**, never a shell string. A relative, path-like `command[0]` resolves against the config file's directory, so a [pack](packs.md) can ship its own server. |
| `transport.env` | `[]` | Environment allowlist for the stdio child. **An empty list means the minimal environment** - `PATH`, `HOME`, `LANG` and nothing else. This differs from subprocess tools, where an empty allowlist means full inheritance: MCP is a new surface, so it got the safe default. Your provider API key does not reach an MCP server unless you name it. |
| `tools.include` | all | Globs (only `*`, matching any substring; case-sensitive) matched against the **server-side** tool name, before the `<server>__` prefix. Empty means every tool. |
| `tools.exclude` | none | Globs removing tools that `include` let through. Applied second. |
| `call_timeout` | `60s` | Bounds one `tools/call`. A timeout is a tool error the model sees, not a run failure. |
| `required` | `true` | `true`: a connect failure aborts the run with exit 8. `false`: a warning on stderr (silenced by `-q`), an `mcp_connect{ok:false}` event, and the run continues without that server's tools. |

Two values are deliberately not configurable: the connect timeout (30 s,
covering at most two jittered attempts) and the wire caps below. An operator
who could raise them would be removing his own safety net.

`mcp.*` is **not** in the `--set`/`-w` override allowlist: connecting a server
grants capability, and CLI overrides may never do that
([cli.md](contracts/cli.md#set)). Audit the YAML, and what it grants is what
runs.

### Wire caps

Charged before the payload is decoded:

- one JSON-RPC message: **8 MiB**;
- one server's whole tool inventory: **4 MiB** and **512 tools** - past either,
  the connection is a protocol error rather than a silent truncation;
- one tool definition (name + description + schema): **32 KiB** - past this the
  single tool is skipped and reported, so one bloated tool does not cost the run
  its other tools.

## Naming and normalization

The model sees `<server>__<tool>`. If that joined name already satisfies the
provider-side rule `^[A-Za-z0-9_-]{1,64}$` it is used verbatim. Otherwise it is
rewritten, deterministically:

1. every character outside the class becomes `_`;
2. the tool half is cut to fit the 64-character ceiling (at most 48 characters);
3. `_` plus the first 8 hex characters of `sha256(<original tool name>)` are
   appended.

```
server "files", tool "read.file"  ->  files__read_file_dd32cdf5
```

The hash is taken over the *original* name, so `read.file` and `read-file`
never collapse onto one name. Nothing is silently truncated: every rewritten
name appears in `amele explain` and in the `mcp_tools_listed` event next to the
name the server published, and `tools/call` always sends the original name back
on the wire.

If a name still collides with an existing tool (a builtin, a subprocess tool, or
a tool from another server), the server is rejected as a **static config error**
(exit 2), not as an unavailable dependency - a name clash is a mistake in the
file, and it is reproducible without a network.

## Permissions

MCP tools go through the one permission mechanism amele has; there is no
per-server policy shortcut:

```yaml
permissions:
  default: allow
  tools:
    "github__*": ask      # every tool of the github server
    "*_delete*": deny     # ...except deleting anything, ever
```

Precedence does not depend on the order the keys are written: an exact name
wins over any pattern, and among matching patterns **the most restrictive wins**
(`deny` > `ask` > `allow`). With no match, `permissions.default` applies. See
[features.md](features.md#permission-profiles-permissions) for the full
semantics, including the TTY fail-safe that turns `ask` into a logged `deny`
when no human is at the keyboard.

**Caveat: globs see the model-facing name.** A pattern like `"*_delete*"` is
matched against the effective name amele exposed - after the server's own
naming and after normalization, which truncates a long name to a cleaned
prefix plus a hash suffix. A server that renames its tool, or a name long
enough to be truncated, can therefore produce a name the deny-glob no longer
matches, and the call falls through to `permissions.default`. For
security-relevant exclusions do not rely on the glob alone: use `tools.exclude`
/ `tools.include`, which are matched against the **original** server-side name,
or set `permissions.default: deny` and allow the tools you want explicitly.

Servers can annotate tools (`readOnlyHint`, `destructiveHint`, `openWorldHint`).
amele shows those annotations in `explain` and in the interactive approval
prompt, and they **never** change a ruling: the annotation is written by the
same party as the description, so trusting it to widen a permission would be
trusting the untrusted side of the boundary
([threat-model.md](threat-model.md) S9).

## Failure semantics

**At startup.** All servers connect in parallel, before the first model call,
with at most two jittered attempts inside a 30 s window.

| Situation | Result |
| --- | --- |
| `required: true` server fails to spawn/connect/handshake, speaks a broken protocol, or answers 401/403 | **exit 8**, before any token is spent |
| `required: false` server fails | warning on stderr, `mcp_connect{ok:false}`, run continues |
| Bad name, name collision, invalid transport block | **exit 2** (static config error) |
| Ctrl-C during connect | exit 1, like any interrupted run |

**During the run.** A tool error the server reports (`isError: true`) comes back
to the model as a tool result, not as a run failure - the model can try
something else. So does a `call_timeout`. The run's own budgets
(`limits.max_turns` / `max_tokens` / `timeout`) still bound everything.

**The toolset is frozen for the whole run.** `tools/list_changed` is not
subscribed to and is ignored if it arrives, and no standalone SSE stream is
opened. The registry a run starts with is the registry it ends with: that is
what makes a run deterministic, replayable and safe to parallelize. New tools
are discovered by the next run.

### Indeterminate results: read this before writing a tool

If the connection drops while a `tools/call` is in flight, amele returns an
**indeterminate** result to the model, worded plainly: the response was lost and
the action may or may not have happened.

amele **never** retries that call automatically. The request may already have
been committed on the server, and MCP has no idempotency key with which a second
attempt could be recognized as the same request - so an automatic retry is a
coin flip between "the mail was not sent" and "the mail was sent twice". The
honest move is to tell the model what is known and let the run decide.

The consequence for server authors and operators: **design MCP tools to be
idempotent**, and prefer tools that can be asked "did this already happen?".

On the *next* call to that server, amele reconnects: a single jittered attempt,
coalesced per server so concurrent calls do not stampede, with the SDK's own
retry loop turned off. Over Streamable HTTP, a `404` to a request carrying an
`Mcp-Session-Id` means the session is gone, and amele re-initializes from
scratch without the stale id. If reconnecting fails,
the tool keeps returning errors and the run continues - the budgets are the kill
switch, and `run_end.mcp_errors` makes the degradation visible afterwards.

**Shutdown** happens at the end of every run, success or failure or signal: HTTP
sessions get a best-effort `DELETE` (2 s), and stdio servers are killed as a
process group (SIGTERM to the group, SIGKILL after a 5 s grace) so a server's own
children do not survive it.

## Running a fleet

amele is built to be a worker: hundreds of processes, each with its own YAML.
The MCP client is tuned for that shape.

- **Connect attempts are jittered** so a timer that fires a hundred agents at
  03:00 does not become a hundred simultaneous handshakes.
- **No idle connections.** No standalone SSE GET stream is opened, so a
  long-running fleet does not park an open connection per worker per server.
- **Still stagger your schedules.** `RandomizedDelaySec=` on a systemd timer, or
  a random sleep in the crontab line, remains the cheapest protection for a
  remote server's rate limit.
- **`run_end.mcp_errors` is the "exit 0 but degraded" signal.** With
  `required: false`, a run that lost a server still succeeds; alert on a rising
  `mcp_errors` rather than on exit codes alone.
- **`lock: true`** keeps a slow run from overlapping the next tick (exit 7),
  which matters more once tools have side effects on a shared remote.
- **The session log is per run.** `session_fp` in `mcp_connect` is a short hash
  of the MCP session id, never the id itself, so logs can be correlated without
  leaking a credential-equivalent value.

## Production guidance

**Pin the server, preinstall the binary.**

```yaml
command: ["/usr/local/bin/mcp-server-filesystem", "."]
```

**Never `npx -y ...` (or `uvx`, or `pipx run`) in production.** It is convenient
in a README and wrong everywhere else: it resolves a version at run time, so the
code your agent trusts changes without a deploy; it needs a network and a
package registry to be up at 03:00; it puts a package manager's full
dependency-resolution surface inside your agent's trust boundary; and it makes
`amele explain` a statement about the past rather than the future. Install the
server with your normal deployment mechanism (a package, a container layer, a
checked-in binary), pin its version, and give the config the absolute path. That
is why no example in this repository uses `npx -y`.

The rest of the hardening advice is the ordinary amele advice:

- keep `transport.env` minimal - the default already is;
- put credentials in the environment (`${VAR}`) and the environment in a
  `0600` file, never in the YAML;
- narrow the toolset with `include`/`exclude` rather than trusting the model to
  avoid the dangerous tool;
- run `amele explain cfg.yaml` after a server upgrade: the tool list, the
  annotations and the definition size are printed, and a changed toolset is
  something a human should see.

## Supported / not yet supported

| Capability | Status |
| --- | --- |
| Tools (`tools/list`, `tools/call`) | ✓ |
| stdio transport | ✓ |
| Streamable HTTP transport | ✓ |
| Static header auth | ✓ |
| Tool annotations (shown, never authoritative) | ✓ |
| Cancellation (`notifications/cancelled`) | ✓ |
| Resources | ✗ not yet |
| Prompts | ✗ not yet |
| Sampling (server asks amele for a completion) | ✗ not yet |
| Elicitation | ✗ not yet |
| Roots | ✗ not yet |
| Tasks | ✗ not yet |
| OAuth 2.1 | ✗ next slice of v0.2 |
| Legacy HTTP+SSE transport | ✗ not planned for now - use a bridge |
| `tools/list_changed` mid-run | ✗ by design (frozen toolset) |
| `Last-Event-ID` resumption | ✗ meaningless without the SSE stream amele does not open |
| Image / audio content in results | ✗ rendered as `[image: image/png, 12345 bytes]` placeholders |

Unsupported capabilities are not merely unused: amele does not **declare** them
at `initialize`, so a well-behaved server never asks for them. The server's
`instructions` field is ignored on purpose - it is prompt content from an
untrusted party. `_meta` on results is ignored.

### SSE-only servers

A server that only speaks the deprecated HTTP+SSE transport cannot be
configured directly. Put a transport bridge in front of it (any of the
community `mcp-proxy`-style tools speaks SSE upstream and stdio or Streamable
HTTP downstream) and point amele at the bridge - as always, with a pinned,
preinstalled binary.

## See also

- [examples/mcp-filesystem/](../examples/mcp-filesystem/) - a runnable pack.
- [contracts/exit-codes.md](contracts/exit-codes.md) - exit 8.
- [contracts/jsonl-events.md](contracts/jsonl-events.md) - the MCP events.
- [contracts/cli.md](contracts/cli.md) - the `MCP SERVERS` section of `explain`.
- [threat-model.md](threat-model.md) - S9, tool poisoning via MCP definitions.
