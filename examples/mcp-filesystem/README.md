# mcp-filesystem

An agent whose only filesystem access comes from an **MCP server**. It is the
smallest complete example of the `mcp:` block: one stdio server, a read-only
include filter, and a permission entry that governs the whole server.

Full reference: [docs/mcp.md](../../docs/mcp.md).

## Requirements

- `AMELE_API_KEY` - a key for the endpoint in `provider.base_url` (any
  OpenAI-compatible one; the config ships with OpenAI's).
- `AMELE_MCP_ROOT` - the absolute path of the directory the server may expose.
  It is interpolated into the server's argv, so the model never chooses the
  root.
- An MCP filesystem server installed at `/usr/local/bin/mcp-server-filesystem`.

Check all three with:

    amele explain mcp-filesystem/

`explain` actually connects: it prints the tools the server would contribute,
which ones the `include` filter kept, any name that had to be rewritten, and
what those definitions cost in tokens. Run it after every server upgrade - a
toolset that changed under your feet is exactly what you want to see.

## Installing the server

Any MCP server that exposes filesystem tools works; what matters is *how* you
install it:

- Install it the way you install everything else on that host - a distribution
  package, a container layer, a release binary you unpack, a `go install` /
  `npm install -g` into an image you build - and **pin the version**.
- Put the resulting binary at an absolute path and write that path into
  `transport.command`. If the server is a Node or Python program, install it
  once and point at the wrapper script it created; if it ships no wrapper, a
  two-line launcher script of your own is still a pinned, preinstalled binary.
- Alternatively drop the binary into this folder (`tools/mcp-server-filesystem`)
  and reference it as `command: ["./tools/mcp-server-filesystem", "..."]`: a
  relative `command[0]` resolves against this pack's directory. Restore execute
  bits after unpacking a zip: `chmod +x tools/*`.

**Do not use `npx -y ...`, `uvx` or `pipx run` here.** They resolve a version at
run time, so the code your agent trusts can change between two cron ticks; they
need a package registry to be reachable at 03:00; and they make `amele explain`
a statement about the past instead of the future.

## Run

    export AMELE_API_KEY=sk-...
    export AMELE_MCP_ROOT=/srv/notes
    amele run mcp-filesystem/ "summarize what changed in the notes this week"

Exit codes worth knowing here: **8** means the server could not be started or
did not answer (`required: true`), **2** means the config itself is wrong - a
bad name, or a server tool colliding with an existing one.

## What the config is demonstrating

- `tools.fs: false` - the MCP server is the *only* road to the filesystem, so
  the server's own root is the sandbox.
- `tools.include` - the write tools are filtered out before the model ever sees
  a definition. Removing a capability beats asking the model not to use it.
- `permissions.tools: {"files__*": allow}` with `default: ask` - one glob
  governs a whole server. An exact key beats a pattern, and among patterns the
  most restrictive wins, so adding `"*_delete*": deny` later cannot be undone by
  the broader `files__*` line.
- `env: ["TZ"]` - a stdio MCP server starts from the minimal environment
  (`PATH`, `HOME`, `LANG`); named variables are added, never subtracted. The
  provider API key is not among them.
