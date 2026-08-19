# Deployment: cron, systemd, log cleanup, embedding

amele is one static binary and one YAML file - there is no server to run and
nothing to install besides the binary itself (see the [README](../README.md)).
This document covers the four things that come up once an agent config leaves
a terminal and starts running unattended: scheduling it with cron or systemd,
what happens to session logs over time, and calling amele from another
program instead of a shell. Everything here composes with the frozen
[CLI contract](contracts/cli.md) and [exit-code contract](contracts/exit-codes.md);
nothing below introduces new behavior.

## 1. Cron

The simplest deployment is the one in [`examples/log-sentry/agent.yaml`](../examples/log-sentry/agent.yaml):
one YAML file plus one crontab line.

```
0 3 * * *  cd /srv/myapp && amele run log-sentry/ "daily log triage" >> /var/log/amele/cron.log 2>&1
```

This variant keeps everything - the run's stderr notes and, on success, the
final answer on stdout - appended to a log file, which is the simplest way to
get a durable record without touching `session_dir`.

**`lock: true`.** `log-sentry/` sets `lock: true`, so if a run ever takes
longer than the interval between cron ticks, the next tick's `amele run` finds
the lock held, prints `another run holds the lock for this config (lock
file: <path>)` to stderr,
and exits **7** immediately - instead of a second triage starting against the
same log directory and sending a duplicate email (exit code
[7 - lock held](contracts/exit-codes.md#7--lock-held)).
A wrapper that wants to treat overlap as a non-event rather than a cron
failure can special-case it (the [exit-code
contract](contracts/exit-codes.md#7--lock-held) shows an equivalent
snippet):

```
amele run log-sentry/ "daily log triage" || [ $? -eq 7 ]
```

**`MAILTO` and `-q`.** Classic cron mails the job owner (or the `MAILTO`
address, if set) whenever a run produces *any* output on stdout or stderr,
independent of its exit code. `amele run` always writes the final answer to
stdout on success, so a plain crontab line - like the one above without
`-q` - mails a "success" message every single day. `-q`/`--quiet` does **not**
change that: per the [CLI contract](contracts/cli.md) it is stderr-only -
it never changes stdout, the exit code, or the session log. To get mail only
when something actually goes wrong, discard stdout yourself
and let `-q` empty out stderr on the success path:

```
MAILTO=ops@example.com
0 3 * * *  cd /srv/myapp && amele run log-sentry/ "daily log triage" -q >/dev/null
```

With this line: a clean run writes its answer to `/dev/null` and nothing to
stderr, so cron sends no mail. A failure - provider error, budget exceeded, a
denied permission, an interrupted run, or the lock-held case above - still
writes to stderr (`-q` never suppresses errors), so `MAILTO` fires exactly
when there is something to look at. Setting `MAILTO=""` is the cron-level
equivalent for operators who route notifications through the agent's own
`send_email` tool instead (as `log-sentry/` does) and want cron itself to
stay silent unconditionally.

## 2. systemd: service + timer

A `systemd` timer replaces the crontab line with a unit that gets
`journalctl`, `systemctl status`, and dependency ordering (`After=`,
`Wants=`) for free. `amele run` is `Type=oneshot`: it starts, does one run,
and exits - exactly what a oneshot service expects.

`/etc/systemd/system/amele-agent.service`:

```ini
[Unit]
Description=amele log-sentry agent run
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=amele
WorkingDirectory=/srv/myapp
EnvironmentFile=/etc/amele/log-sentry.env
ExecStart=/usr/local/bin/amele run /srv/myapp/log-sentry/ "daily log triage"
TimeoutStartSec=310
KillMode=control-group
```

`/etc/systemd/system/amele-agent.timer`:

```ini
[Unit]
Description=Run amele log-sentry agent daily

[Timer]
OnCalendar=*-*-* 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

Notes on the fields that matter:

- **`User=`** should be a dedicated, unprivileged system user whose home and
  file permissions match the workspace the config declares - do not run this
  as root.
- **`EnvironmentFile=`** is where the provider API key lives, e.g.
  `AMELE_API_KEY=sk-...` in `/etc/amele/log-sentry.env`, mode `0600` and
  owned by that user. This is the systemd equivalent of exporting the
  variable before a cron run; the YAML still only ever contains
  `${AMELE_API_KEY}` (docs/engineering.md §5.5 / [threat model §2](threat-model.md#2-system-model-and-trust-boundaries)).
- **`ExecStart=`** takes absolute paths for both the binary and the config -
  systemd services do not run through a login shell, so there is no `PATH`
  lookup or `cd` the way the crontab line above has one.
- **`TimeoutStartSec=`** is a backstop above the config's own
  `limits.timeout`; it should be a little larger than
  `limits.timeout` (`5m` in the example config → `310s` here) so amele's own
  budget enforcement fires first (exit 3) in the normal case, and systemd's
  timeout is only the safety net for a hang amele's own accounting missed.
  When it does fire, systemd sends **SIGTERM**, which amele's [signal
  handling](contracts/cli.md#signals) turns into: the run context is
  canceled, the session gets its `run_end` event, and the process exits
  **1** - the same thing that happens on `systemctl stop
  amele-agent.service` or an operator's Ctrl-C. systemd then reports the unit
  as failed (a oneshot service exiting non-zero is a failure), which shows up
  in `systemctl status` and can drive `OnFailure=` alerting.
- **`KillMode=control-group`** (systemd's default, but write it down once an
  agent spawns children) makes systemd kill everything in the unit's cgroup,
  not just the main process. amele already kills a `stdio` MCP server as a
  process group at the end of every run, including on SIGTERM
  ([docs/mcp.md](mcp.md#failure-semantics)) - but an amele that is itself
  SIGKILLed never runs that cleanup, and its grandchildren would survive. The
  cgroup is the only thing that reliably reaps them. `KillMode=process` on a
  unit that runs MCP servers leaks a process per run.
- **`OnCalendar=*-*-* 03:00:00`** is systemd's calendar syntax for "every
  day at 03:00"; `Persistent=true` means a run that was missed because the
  machine was off at 03:00 fires once at the next boot instead of being
  silently skipped - useful for laptops and VMs that aren't always up.

Install and enable:

```console
$ sudo cp amele-agent.service amele-agent.timer /etc/systemd/system/
$ sudo systemctl daemon-reload
$ sudo systemctl enable --now amele-agent.timer
$ systemctl list-timers amele-agent.timer
$ sudo systemctl start amele-agent.service   # trigger one run now, to test
$ journalctl -u amele-agent.service -e
```

**Where the output goes.** Unlike the cron pattern above, a systemd service's
stdout *and* stderr both land in the journal by default - there is no
separate "discard stdout" step, so the daily answer and every stderr note are
both in `journalctl -u amele-agent.service` unless the unit sets
`StandardOutput=`/`StandardError=` otherwise. `-q` still trims the stderr
side (the summary line, the non-error notes) if a quieter journal is
preferred; it has no effect on the answer that reaches stdout.

Both unit files above were checked with `systemd-analyze verify` against a
locally built `amele` binary (no other tool required beyond `systemd-analyze`,
which ships with systemd) and parse cleanly; `User=amele` and the
`/srv/myapp` paths are illustrative and need to exist on the target host.

## 3. Session log cleanup

**Design position: amele never deletes its own session logs.** Every run with
`session_dir` set appends a new JSONL file; nothing in amele ever removes,
truncates, or rotates one. This is deliberate, not an oversight - it follows
the same Unix-tool philosophy as the rest of the CLI (docs/engineering.md §2): amele's
job is to produce an accurate audit trail (§4.9 of the
[threat model](threat-model.md#49-audit-trail)), and the system that owns the
*schedule* - cron, systemd, whatever decides how often amele runs - is also
the system that knows how much retention that schedule implies. Baking a
retention policy into amele would mean guessing at a number (7 days? 30? 90?)
that has nothing to do with the agent and everything to do with the
operator's disk, compliance requirements, and how often they actually look at
old runs. Deleting old files is a job standard Unix tools already do well;
amele does not need to reimplement `find` or `logrotate` badly.

**cron**, deleting anything older than 30 days, run weekly:

```
0 4 * * 0  find /var/lib/amele/sessions -name '*.jsonl' -mtime +30 -delete
```

(`-mtime +30` selects files whose content was last modified more than 30
days ago; verified against a directory containing a 40-day-old and a 5-day-old
file - only the older one is selected/deleted.)

**`systemd-tmpfiles`**, the systemd-native equivalent, in
`/etc/tmpfiles.d/amele-sessions.conf`:

```
# Type Path                       Mode UID  GID  Age
e      /var/lib/amele/sessions    -    -    -    30d
```

Type `e` cleans a directory's *contents* on the age threshold without
touching or recreating the directory itself - appropriate here since amele
creates `session_dir` itself (`0750`, per the threat model). Most
distributions already run `systemd-tmpfiles-clean.timer` periodically, so
dropping this file is enough; `sudo systemd-tmpfiles --clean
/etc/tmpfiles.d/amele-sessions.conf` runs it immediately for testing. (Syntax
verified with `systemd-tmpfiles`; note if testing this yourself with `touch
-d` to fake old files, the default age check also considers change time
(`ctime`), which `touch -d` resets to "now" - a real 30-day-old, untouched
session file does not have this problem.)

**Why not `logrotate`.** `logrotate` rotates a single file that keeps
growing (`/var/log/nginx/access.log` today, `.1` tomorrow). Session logs are
already one immutable file per run - there is nothing to rotate, only files
to eventually delete. `find -mtime -delete` or a tmpfiles.d age rule is the
tool that matches the shape of the data; reach for `logrotate` only if
something else in the deployment funnels amele's stderr into a single
growing file (e.g. the `>> cron.log 2>&1` pattern in §1), where rotating
*that* file is the ordinary logrotate use case.

## 4. Embedding amele as an agent core from another program

Because amele is one binary that speaks stdin/stdout/exit-codes, calling it
from PHP or Python is a subprocess call, not an SDK integration. The pattern
that keeps concurrent callers from stepping on each other:

- pass `-q` so a successful call's stderr is empty and only real errors show
  up;
- give each request its own `--set session_dir=<path>` so concurrent
  invocations don't interleave in one session file (or `--set session_dir=`
  to disable logging for that call entirely);
- do **not** set `lock: true` in a config used this way - the lock is
  per-config-file and is for single-flight *scheduled* runs (§1); a config
  meant to serve concurrent requests should leave it off so overlapping
  calls run side by side instead of exit-7-ing each other;
- set `output.schema` on the config so stdout is parseable JSON on success
  and empty on any failure (exit 6 included) - no prose-scraping;
- branch on the exit code using the [exit-code table](contracts/exit-codes.md)
  rather than parsing stderr text.

Python (`subprocess`):

```python
import json
import subprocess

def amele_run(config_path: str, task: str, session_dir: str) -> dict:
    """Run one amele task and return the parsed JSON answer.

    Raises subprocess.CalledProcessError on any non-zero exit - see
    docs/contracts/exit-codes.md for what each code means.
    """
    result = subprocess.run(
        [
            "amele", "run", config_path,
            "-q",
            "--set", f"session_dir={session_dir}",
            task,
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    return json.loads(result.stdout)
```

PHP (`proc_open`, array form - no shell, so task text with spaces or quotes
needs no escaping):

```php
<?php

function amele_run(string $configPath, string $task, string $sessionDir): array
{
    $cmd = [
        'amele', 'run', $configPath,
        '-q',
        '--set', "session_dir=$sessionDir",
        $task,
    ];

    $descriptorSpec = [
        0 => ['pipe', 'r'],
        1 => ['pipe', 'w'],
        2 => ['pipe', 'w'],
    ];

    $process = proc_open($cmd, $descriptorSpec, $pipes);
    if (!is_resource($process)) {
        throw new RuntimeException('failed to start amele');
    }

    fclose($pipes[0]);
    $stdout = stream_get_contents($pipes[1]);
    $stderr = stream_get_contents($pipes[2]);
    fclose($pipes[1]);
    fclose($pipes[2]);

    $exitCode = proc_close($process);
    if ($exitCode !== 0) {
        throw new RuntimeException("amele exited $exitCode: $stderr", $exitCode);
    }

    return json_decode($stdout, true, 512, JSON_THROW_ON_ERROR);
}
```

Both sketches use array-form subprocess APIs (`subprocess.run([...])`,
`proc_open($cmd, ...)` with `$cmd` as an array), which - same as amele's own
subprocess tools (§4.2 of the threat model) - never invoke a shell, so task
text is passed as one argument regardless of spaces, quotes, or shell
metacharacters. Both were checked for syntax (`python3 -m py_compile`,
`php -l`) and use only standard-library APIs.
