# Session logging and secret redaction

When `session_dir` is set, every run appends a JSONL event log (one file per
run). The log doubles as the observability trail and the future replay/resume
source, so it records the full conversation: task, assistant text, tool calls
and results, token accounting.

## Reading what a tool did

Each `tool_result` line carries an `outcome` - `ok`, `timeout`,
`nonzero_exit` (with `exit_code`), `aborted`, `denied_policy`,
`denied_no_tty`, `ask_refused` or `error` - so a morning-after read of a cron
run tells a working tool from a refused or failing one:

```sh
jq -r 'select(.type=="tool_result") | "\(.tool)\t\(.outcome)"' run-*.jsonl
```

Note that a failing command is not an `error`: `grep` exiting 1 is
`nonzero_exit`, and `is_error` is reserved for calls amele could not dispatch
at all. `result_bytes` reports the size of what the model actually read, which
is how you see when the per-field log clip (`limits.max_logged_field`, 8 KiB by
default; `0` writes every field whole) dropped part of it from the file.
The full field list is in
[docs/contracts/jsonl-events.md](contracts/jsonl-events.md).

There are two different cuts here, and only one of them reaches the model. A
result that hit the tool-result byte cap is flagged `truncated` and ends in
`[output truncated by amele]` (`fs_list` appends its entry counts to the same
marker: `[output truncated by amele: N of M entries shown]`):

```sh
jq -r 'select(.type=="tool_result" and .truncated) | .tool' run-*.jsonl
```

That names the tools whose answer the MODEL was handed short - the ones to
give a narrower query, or a larger `limits.max_tool_result_bytes`. The log
clip is the other cut: it happens after the model has read the text and it
never sets `truncated`. A `result` shorter than `result_bytes` is the log clip
and/or redaction - redaction runs first and unconditionally, and a `[REDACTED]`
shorter than the value it replaced shrinks the field on its own - so the
trailing `...[clipped]` marker is what tells you it was the clip.

## What gets redacted

Every environment value substituted into the config via `${VAR}` interpolation
is registered as a secret and replaced with `[REDACTED]` wherever it appears
in the log - tool output, model echoes, anywhere. This is deliberate:
`${DB_PASSWORD}` inside a prompt is just as much a credential as
`provider.api_key`, and amele cannot know which interpolated values are
sensitive, so it redacts all of them.

## Caveat: interpolating non-secret variables

The flip side: if the config interpolates a **non-secret** variable whose
value appears all over normal output, the log becomes hard to read. The
classic case is `${HOME}` or `${USER}`:

```yaml
workspace: ${HOME}/data
```

Here the value of `$HOME` (e.g. `/home/alice`) is registered as a secret, so
**every absolute path** in tool output is logged as `[REDACTED]/data/...`.

Recommendation: avoid interpolating broad, non-secret values like `${HOME}`,
`${USER}` or `${PWD}`. Prefer relative paths (they resolve against the config
file's directory) or literal absolute paths:

```yaml
workspace: data          # relative to the config file - no interpolation
```

Reserve `${VAR}` interpolation for values that actually are secrets.
