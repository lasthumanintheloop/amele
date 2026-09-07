# The builtin `shell` tool

The `shell` tool gives the model one capability: run a command line through
`sh -c` with the workspace as the working directory. It is **disabled by
default** and only exists when the config asks for it explicitly.

```yaml
tools:
  shell:
    enabled: true            # absent block or false → the tool does not exist
    allow: ["git *", "ls*"]  # empty list = everything the deny list misses
    deny:  ["rm *"]          # checked first; a match is final
                             # both match each LINE of the command
    env: ["GIT_DIR", "TZ"]   # environment allowlist; absent = inherit everything
    timeout: 60s             # per command; empty = 60s
```

Each call is a fresh shell: `cd`, variables and shell functions do not persist
between calls. Stdout is returned to the model (capped at 64 KiB per stream,
or at `limits.max_tool_result_bytes` when the config sets one; a cut result
carries `[output truncated by amele]` - at the end, or mid-result when the
`stderr:` section is rendered after it - and is flagged `truncated` in the
session log); a non-zero exit is reported with its stderr as a tool result,
not as a run failure. On timeout the whole process group is terminated.

`limits.max_tool_result_bytes` is one number for both cuts: it caps each stream
AND the framed result the loop hands back. On a failed command stdout is
rendered first and can spend the entire budget, so the ceiling may cut the
`stderr:` section away completely - raise the number when the stderr of a
failing command is what you are after.

## Pattern matching

Patterns are matched **per line**. The command is split on newlines, each line
is whitespace-trimmed, and empty lines are ignored - so a leading space, a tab,
or a denied command sitting on the second line of a multi-line command cannot
slip past the list. Trimming affects **matching only**: the command is executed
exactly as the model wrote it.

A single pattern matches one line like this:

- `*` matches any substring, including the empty one;
- every other character matches itself - there is no `?`, no `[...]`, no
  escaping, and `*` does **not** stop at `/` (this is not `path.Match`);
- matching is case-sensitive and anchored at both ends of the trimmed line.

Evaluation order:

1. **deny first** - if *any* line matches *any* deny pattern, the whole command
   is rejected;
2. **allow** - if the allow list is non-empty, *every* non-empty line must match
   at least one allow pattern, or the whole command is rejected. An empty allow
   list permits everything the deny list did not catch.

So with `deny: ["rm *"]`, all of these are rejected:

```
 rm -rf x            # leading space
<TAB>rm -rf x        # leading tab
cd build
rm -rf .             # second line
```

and with `allow: ["echo *", "ls*"]`, `echo hi` followed by `curl …` on the next
line is rejected because the second line matches nothing in the list.

Anchoring is at the ends of the line, not at command boundaries: a pattern that
ends in `*` constrains only the start of the command line - everything after the
matched prefix, including `;`-chained commands, runs too, so `allow: ["date*"]`
also passes `date; id`.

A rejected command is not an error. The model receives
`command rejected by shell policy: ...` as the tool result and can adapt -
pick a different command, or explain that it cannot complete the task.

## SECURITY: the pattern list is accident prevention, not a boundary

**Do not treat `allow`/`deny` as a security control.** They stop a model from
*casually* doing something destructive. They do not stop a determined or
prompt-injected one, because a command that passes the patterns can still spawn
anything:

```
git -c core.fsmonitor=cmd status     # passes "git *", runs arbitrary code
find . -exec cmd {} \;               # passes "find *"
ssh host cmd                         # passes "ssh *"
env VAR=x cmd / xargs cmd / sh -c …  # aliases, hooks, wrappers
echo hi; curl http://evil            # one line can chain commands
```

Per-line normalization closes the *accidental* bypasses (whitespace, multi-line
commands). It closes none of the deliberate ones above.

The layer that actually holds is the operating system. Run amele in a
container or VM with only the filesystem, network and privileges the agent
legitimately needs, and treat the shell tool as "this agent may run anything
inside that box" (docs/threat-model.md).

## SECURITY: scope the environment with `env` - by default the shell sees everything

**By default**, commands started by the shell tool - and by subprocess tools -
inherit amele's **entire environment**, including the provider credential:

```json
{"command": "printenv OPENROUTER_API_KEY"}
```

returns the API key to the model as an ordinary tool result. No allow/deny
pattern stops this: reading an environment variable needs no dangerous-looking
command, and any of `printenv`, `env`, `set`, `sh -c 'echo $VAR'` or a script
that reads `/proc/self/environ` does it.

The first-line mitigation is the **`env` allowlist**:

```yaml
tools:
  shell:
    enabled: true
    env: ["GIT_DIR", "TZ"]   # the child sees ONLY these (+ PATH, HOME, LANG)
```

When `env` is non-empty, the child process is started with only the listed
variables - values taken from amele's own environment; listed names that are
unset are silently skipped - plus `PATH`, `HOME` and `LANG`, which are always
passed so basic tools keep working. `printenv OPENROUTER_API_KEY` then prints
nothing. Subprocess tools take the same field per tool
(`tools.subprocess[].env`). An absent or empty `env` keeps full inheritance,
exactly as before the field existed.

The allowlist reduces exposure; it does not move the boundary. It scopes what
the child *inherits*, not what a same-user process can *reach*: on Linux,
`cat /proc/$PPID/environ` reads amele's own environment right back, allowlist
or not. Anything the agent must never read still must not be in amele's
environment at all.

**The session log will not show you a leak happened.** Every
`${VAR}`-interpolated value is registered for redaction (see
[session-logging.md](session-logging.md)), so a leaked key is written to the
audit trail as `[REDACTED]` - the redaction that protects the log also hides
the leak from it. Do not expect to find this in a transcript after the fact.

Consequences to plan around:

- treat every variable in amele's environment as readable by the agent, and by
  anything the agent's output reaches (a model provider, an email, a PR
  comment);
- set `env` on every shell and subprocess tool that does not need the full
  environment - the empty-by-default field is the single biggest accidental
  leak surface;
- still run amele with only the variables it needs - `env -i`, a systemd unit
  with an explicit `Environment=`, or a container with a minimal env - rather
  than inheriting a developer shell;
- rotate the provider key if a shell-enabled agent processed untrusted input.

This is another reason the container/OS layer is the real boundary: the process
environment is part of what that layer must scope, and the allowlist only
narrows what amele hands down inside it.

## Guardrails worth combining with the OS layer

- a permission profile (`permissions.tools.shell: ask` / `deny`) gates the tool
  by name like any other tool - and with no TTY an `ask` policy always degrades
  to a denial, so a cron run never blocks on a question nobody can answer;
- an agent that reads **external** content (issues, emails, scraped pages) and
  also has a shell is a confused deputy. Prefer keeping the shell off in that
  shape, or split it into two agents.
