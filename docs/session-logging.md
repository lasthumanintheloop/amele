# Session logging and secret redaction

When `session_dir` is set, every run appends a JSONL event log (one file per
run). The log doubles as the observability trail and the resume source -
`amele run --resume` reads it back and continues the run it recorded - so it
records the full conversation: task, assistant text, tool calls and results,
token accounting.

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
result that hit the tool-result byte cap is flagged `truncated` and carries
`[output truncated by amele]` in the text - at the end for most cuts, but
mid-result when framing follows the cut stream (a failed subprocess whose
stdout was cut renders its `stderr:` section after the marker), so a search
for the marker is safer than a suffix test (`fs_list` appends its entry
counts to the same marker: `[output truncated by amele: N of M entries
shown]`):

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

## Resuming a run

`amele run agent.yaml --resume <log> [instruction]` rebuilds the conversation
a session log recorded and carries on from it, so a run that died three tool
calls in does not pay for those turns again. The full rules - what is refused,
and with which message - are in the
[CLI contract](contracts/cli.md#resuming-a-run---resume-path); what matters
here is what the *logging* config has to say for a log to be resumable at all.

**`limits.max_logged_field: 0` is the prerequisite.** By default every
free-text field is clipped to 8 KiB, and a conversation rebuilt from clipped
text is not the conversation the model had - so a log carrying the clip marker
in a field the history needs is refused (exit 2) rather than silently resumed
from, and the message names the key:

```
session log is clipped: result in turn 3 ends in the clip marker; write the log with limits.max_logged_field: 0 to make it resumable
```

Decide this before the run, not after it: nothing can put back bytes the log
never stored.

```yaml
limits:
  max_logged_field: 0   # the whole record, so the run can be resumed
```

**`log_reasoning: true` is what keeps a reasoning payload replayable.** With
it the payloads go back verbatim - but only when the log's
`run_start.provider` is still the current config's provider identity, its
`run_start.model` is still the model being called, and the old run never fell
back to another backend: a provider signs or hash-checks its own reasoning
bytes, for the model that produced them.

Without it the resumed conversation is carrier-less, and that is not something
every provider accepts: a thinking-enabled `anthropic` or `gemini` run whose
turns are replayed without the payloads they were produced with can be refused
outright (a 400, exit 5 - see [docs/providers.md](providers.md) on
signatures). For those configs the resumable combination is `log_reasoning:
true` plus the same model, the same provider identity and no fallback in the
log; a config that does not enable thinking has no carriers to lose. Remember
what the key persists (see the caveat above and
[docs/deployment.md](deployment.md) §4): a reasoning trace is where a model
paraphrases, and redaction works by value.

**Redaction is not a fidelity gate.** A `[REDACTED]` inside a logged tool
result is replayed to the model exactly as it stands, because that is the text
the run being continued was reading - resuming shows the model nothing it was
not already shown. The one exception is a reasoning carrier: a payload
containing `[REDACTED]` is no longer the bytes the provider signed, so it is
dropped instead of echoed back.

**The resumed run writes its own file.** The log named by `--resume` is opened
read-only and never appended to, turn numbering starts at 1 again, and the new
file's `run_start` carries `resumed_from`, `resumed_turn` and (when the old
run was killed mid-tool-call) `resumed_pending` - the call ids whose results
never reached the log, which nothing re-executed. `resumed_from` is the one
logged field that is redacted but never clipped, so a secret value inside the
path is replaced there too and the logged string is then no longer the path to
feed back to `--resume`.
