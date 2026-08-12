# log-sentry

Scans application logs once a day and emails a triage summary. This folder
is an amele *pack*: `agent.yaml` is the whole agent; run it from anywhere.

## Requirements

- `AMELE_API_KEY` - a key for the endpoint in `provider.base_url` (any
  OpenAI-compatible one; the config ships with OpenAI's).
- `msmtp` (or another sendmail-compatible binary) on PATH for the
  `send_email` tool.

Check both with: `amele explain log-sentry/`

## Run

    amele run log-sentry/ "daily log triage"

Deploy with one crontab line:

    0 3 * * *  cd /srv/myapp && amele run log-sentry/ "daily log triage"

`lock: true` in agent.yaml makes overlapping runs exit 7 instead of
double-mailing (see docs/deployment.md).

## Layout notes

- `workspace: ./logs` - the agent reads only the log directory.
- `sessions/` - run logs land here; gitignored.
- If you add scripts, put them in `tools/` and reference them as
  `command: ["./tools/x.sh"]` - the path resolves against this folder.
  After unpacking from a zip, restore execute bits: `chmod +x tools/*`.
