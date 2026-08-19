<div align="center">

# amele

*/ah-meh-leh/ - an old word for the one who quietly does the work,
from the Semitic root ʿ-m-l, "to labor":<br>
Hebrew ʿamal (עָמָל, toil) · Arabic ʿāmil (عامل, worker) · Turkish amele (laborer)*

### Your agent is a single YAML file.<br>Its runtime is a single static binary.<br>Together they run anywhere.

no Python · no Node · no platform · no account

[![CI](https://github.com/lasthumanintheloop/amele/actions/workflows/ci.yml/badge.svg)](https://github.com/lasthumanintheloop/amele/actions/workflows/ci.yml)
![Go version](https://img.shields.io/github/go-mod/go-version/lasthumanintheloop/amele)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/lasthumanintheloop/amele)](https://github.com/lasthumanintheloop/amele/releases/latest)

</div>

- **Agent as an artifact.** Your agent *is* a file: version it, diff it,
  review it in a pull request, share it as a folder. Deleting an agent is
  `rm`.
- **Build your own.** Your prompt, your tools, your budgets - on any
  OpenAI-compatible endpoint (OpenAI, OpenRouter, vLLM, Ollama) or the native
  Anthropic API. With a local model it is yours end to end, offline included.
- **A core to build around.** amele owns the loop, the budgets, the sandbox
  and the log; capabilities come from outside, in whatever language you
  like - a tool is any executable, and your own app can drive amele the
  same way. The integration API is the process boundary, which every
  language already speaks. Nobody has to write Go.
- **For everyone.** If you can edit a config file, you can author an agent.
  Nothing to install around it, nobody to sign up with.
- **Zero dependencies, runs everywhere.** One static ~9.3 MB binary; the
  answer to `pip install` is `scp`.
- **A well-mannered process.** Meaningful exit codes, schema-guaranteed JSON
  on stdout, one JSONL log per run - your scripts get a contract, not a
  chatbot.

![an agent in a pipe: log in, JSON out, then a crontab line](docs/demos/hero.gif)

And it is one file in many seats - the *same* agent, unchanged:

| runs as | looks like |
|---|---|
| a cron job | `0 3 * * * amele run log-sentry/ "daily triage"` |
| a CI gate | the exit code fails the build; `output.schema` feeds the next step |
| a GitHub Action | `uses: lasthumanintheloop/amele@main` - answer comes back as a step output |
| a pipe stage | `amele run judge.yaml < diff.txt \| jq .score` |
| part of your app | spawn `amele run` from PHP, Python, anything; read stdout |
| a terminal chat | `amele chat agent.yaml` - same tools, same budgets, interactive |

**Get started in a minute:** [download a binary](#install), then
[build your own agent](#build-your-own-in-five-minutes) - `amele init`,
set `AMELE_API_KEY`, `amele run`.

## Runs everywhere

<div align="center">

**One binary. No runtime. Every box you already have.**

![Linux](https://img.shields.io/badge/Linux-x86--64%20%C2%B7%20ARM64%20%C2%B7%20ARMv6%20%C2%B7%20RISC--V%20%C2%B7%20s390x%20%C2%B7%20ppc64le-2d3748?logo=linux&logoColor=white)
![macOS](https://img.shields.io/badge/macOS-Intel%20%C2%B7%20Apple%20Silicon-2d3748?logo=apple&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-x64%20%C2%B7%20ARM64%20%C2%B7%20x86-2d3748?logo=windows&logoColor=white)
<br>
![FreeBSD](https://img.shields.io/badge/FreeBSD-x86--64%20%C2%B7%20ARM64-2d3748?logo=freebsd&logoColor=white)
![OpenBSD](https://img.shields.io/badge/OpenBSD-x86--64%20%C2%B7%20ARM64-2d3748?logo=openbsd&logoColor=white)
![NetBSD](https://img.shields.io/badge/NetBSD-x86--64-2d3748?logo=netbsd&logoColor=white)
![illumos](https://img.shields.io/badge/illumos-x86--64-2d3748?logo=oracle&logoColor=white)
![Android](https://img.shields.io/badge/Android%20%2F%20Termux-ARM64-2d3748?logo=android&logoColor=white)
<br>
![Docker](https://img.shields.io/badge/scratch%20%C2%B7%20distroless%20%C2%B7%20Alpine-container%20ready-2d3748?logo=docker&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-Zero%20to%205-2d3748?logo=raspberrypi&logoColor=white)
![OpenWrt](https://img.shields.io/badge/OpenWrt-routers-2d3748?logo=openwrt&logoColor=white)

</div>

`CGO_ENABLED=0` and no dependency outside the Go standard library, so the
same source becomes a static executable for **21 OS/arch pairs** - every
one of them ships as a release archive (`make dist` builds them all). The
platform-specific surface is tiny and on purpose: `flock(2)` for the run
lock, a process group so a timed-out tool cannot leave orphans, `SIGTERM`
handling. Everything else is portable by construction.

| tier | platforms | what it means |
|---|---|---|
| **tested** | Linux (amd64, arm64) | CI runs the full suite here; this is where amele lives in production |
| **expected to work** | macOS, Windows, FreeBSD, OpenBSD, NetBSD, DragonFly, illumos, Android/Termux, every other Linux arch | built from the same tree; the unix code path is byte-identical to Linux. Windows: `lock:` says "unsupported" instead of pretending; everything else works |
| **build it yourself** | anything in `go tool dist list` with an OS that has processes and signals | `GOOS=… GOARCH=… go build` - and open an issue if it does not |

Put it in a `FROM scratch` image, `scp` it to a Raspberry Pi, drop it on a
NAS, run it from a router's cron - the agent YAML travels with it, unchanged.

## Why this exists

amele started as an internal tool. Doing harness engineering, we needed
hundreds of agent variations - which only works when an agent is a file you
can version, diff and regenerate, not a codebase. And we kept reaching for
the same small, dependable core to build around, from model benchmarks to
daily chores. At some point it was quietly doing its job well enough that
keeping it to ourselves felt wrong. I hope it is useful to you too.

It is early, and it is shaped by our own use - so help is genuinely
welcome, at every level: an idea, a "this part confused me", a bug report,
a pull request. All of it counts.

## Show me

The canonical demo, [examples/log-sentry/](examples/log-sentry/): an agent
that reads application logs and emails a triage summary. The entire agent is
one YAML file:

```yaml
model: gpt-4.1-mini
provider:
  base_url: https://api.openai.com/v1
  api_key: ${AMELE_API_KEY}          # secrets never live in YAML

system_prompt: |
  You are a log triage agent. Read the logs in the workspace, identify
  errors from the last 24 hours, and email a short report.

workspace: ./logs                    # the agent cannot read outside this
tools:
  fs: true
  subprocess:
    - name: send_email
      description: Send an email. Pipe the RFC-822 message to stdin.
      command: ["msmtp", "${AMELE_MAIL_TO}"]   # argv is fixed; the model
                                     # writes the mail, never the recipient
      timeout: 30s

limits: {max_turns: 15, max_tokens: 150000, timeout: 5m}
session_dir: ./sessions
lock: true                           # overlapping cron ticks exit 7, not twice
```

And the entire deployment is one crontab line:

```
0 3 * * *  cd /srv/myapp && amele run log-sentry/ "daily log triage"
```

```console
$ amele run log-sentry/ "daily log triage"
✓ 4 turns, 6 tool calls, 21.3k tokens, 27.8s
```

Here it is live - an 11-line agent finding the root cause in last night's
logs (and dismissing a red herring on the way):

![log triage live: the agent names the incident and its deploy](docs/demos/logsentry.gif)

That folder is a *pack*: config, prompt and helper scripts travelling
together, runnable from anywhere by path. Sharing an agent is sending a
folder. See [docs/packs.md](docs/packs.md).

## Build your own in five minutes

```console
$ amele init agent.yaml
amele: wrote agent.yaml - next: set AMELE_API_KEY and run: amele validate agent.yaml
$ export AMELE_API_KEY=sk-...
$ amele validate agent.yaml
agent.yaml: OK
$ amele run agent.yaml "summarize the files in this directory"
```

The generated `agent.yaml` is annotated: model + endpoint, a system prompt,
sandboxed filesystem tools, budgets (`max_turns`, `max_tokens`, `timeout` -
exceeding any of them is exit code 3), and a JSONL session log. Only
`${ENV_VAR}` references are accepted where secrets go. A longer system
prompt can live in its own file via `system_prompt_file` and be swapped per
invocation with `--set system_prompt_file=...`.

Before spending a token, `amele explain agent.yaml` prints what the agent
may touch and spend - model and endpoint, tool inventory, permission
profile, budgets, which environment variables it needs and which are
missing. Explain reports; run gates.

Want the agent fully self-owned? Point `base_url` at a local server:

```yaml
provider:
  base_url: http://localhost:11434/v1   # Ollama; api_key not needed locally
```

The same YAML also works interactively: `amele chat agent.yaml` is a REPL
over the same prompt, tools and budgets - your own terminal agent from the
same artifact you deploy.

## Machines are users too

Declare a JSON Schema and stdout becomes machine-readable - either JSON that
validated against the schema, or exit code 6 and an empty stdout. No prose,
no fences, no surprises:

```yaml
output:
  schema:
    type: object
    required: [score, summary]
    properties:
      score:   {type: integer, minimum: 0, maximum: 10}
      summary: {type: string}
```

```console
$ amele run judge.yaml < diff.txt | jq .score
7
```

(With no task text on the command line, the piped input *is* the task. Task
text and piped input combine via a `prompt:` template with `{{args}}` and
`{{input}}` - see the [CLI contract](docs/contracts/cli.md).)

Providers with native structured output are used when available; the answer
is validated locally either way, with violations fed back to the model for
up to `max_schema_retries` repair rounds. And because an agent is just a
process, agents compose like processes: a config whose subprocess tool runs
`amele run` is an LLM judge with different models for worker and judge - no
framework feature required. Details: [docs/features.md](docs/features.md).

## Bring the tools you already have

Any executable becomes a tool: name it, describe it, give the argv. No
wrapper, no SDK, no rewrite - here is a 2007 Perl script answering
questions as an agent tool, six YAML lines later:

![a 2007 Perl CLI called live as an agent tool](docs/demos/perltool.gif)

**Or borrow someone else's tools.** amele is an MCP client: declare the
servers in the same YAML - `stdio` (a local binary) or Streamable HTTP with
static header auth or OAuth (`amele mcp login`, silent refresh in the run) -
and their tools join the registry under the same
permissions, budgets and session log, named `<server>__<tool>`. The toolset is
frozen for the run, a missing `required` server is exit 8 rather than a
half-equipped agent, and `amele explain` connects and shows you exactly what a
server would contribute before you run it: [docs/mcp.md](docs/mcp.md).

More demos: [build an agent in 40 seconds](docs/demos/d1-build.gif) ·
[the full pipe demo](docs/demos/pipe.gif) ·
[it fails like a program - exit 2, 3, 7](docs/demos/failmodes.gif) ·
[audit an agent someone sent you](docs/demos/audit.gif)

## GitHub Action

The repo doubles as a composite GitHub Action: point it at a config, get the
agent's answer as a step output, with `amele run`'s exit code deciding
whether the step passes:

```yaml
jobs:
  triage:
    runs-on: ubuntu-latest
    env:   # job-level: inherited by every step, the action included
      AMELE_API_KEY: ${{ secrets.AMELE_API_KEY }}
    steps:
      - uses: actions/checkout@v4
      - uses: lasthumanintheloop/amele@main
        with:
          config: agent.yaml
          task: "triage yesterday's logs"
```

Inputs, the `answer` output, secrets handling and the security notes:
[docs/github-action.md](docs/github-action.md).

## What amele is not

No GUI, no embedded vector store, no workflow DSL, no plugin system, no
scheduler, no multi-tenant server. amele does one thing: run one agent from
one file, well, unattended. Everything else composes from the outside - your
scheduler schedules, your shell pipes, and new capabilities arrive as
subprocess tools in the language of your choice, or as MCP servers amele
connects to, not as core features.

## Roadmap

Direction, not dates. Details and discussion live in
[issues](https://github.com/lasthumanintheloop/amele/issues) and
[milestones](https://github.com/lasthumanintheloop/amele/milestones).

- **v0.2 - connected and fast.** MCP client - **done**: stdio and Streamable
  HTTP with static header auth, plus OAuth 2.1 with `amele mcp
  login|status|logout` ([docs/mcp.md](docs/mcp.md)).
  Then: parallel tool calls within a turn; sampling parameters (`temperature`, `top_p`) and a
  provider-normalized `reasoning` setting (effort / thinking budget) that
  survives tool loops across OpenAI, Anthropic, DeepSeek and OpenRouter;
  configurable retry policy.
- **v0.3 - cheap and durable.** Prompt caching (Anthropic `cache_control`,
  cache-hit reporting for every dialect); provider fallback; guards against
  oversized tool results; `--resume` for interrupted runs.
- **v0.4 - a person at the keyboard.** Streaming output; a nicer `chat`
  (history, multi-line input); default config discovery; `amele doctor`.
- **v1.0** - not a feature: the contracts have held long enough to promise
  they will keep holding.

Not planned, on purpose: GUI/TUI, embedded RAG, workflow DSL, plugin
system, built-in scheduler, multi-agent orchestration (fallback is not
orchestration), Authenticode/notarization.

## Built for nobody watching

Headless operation is the design center, not an afterthought:

- **Permissions** - per-tool `allow` / `ask` / `deny` profiles; with no TTY
  an `ask` degrades to a logged deny, so cron runs never hang on a question.
- **Budgets are hard limits** - turns, tokens and wall clock; crossing any
  one is exit 3, not a warning.
- **Run lock** - `lock: true` makes a run single-flight per config: a cron
  tick firing mid-run exits 7 instead of interleaving.
- **Signals** - SIGINT/SIGTERM still write the session log's final record;
  systemd timers get a clean story ([docs/deployment.md](docs/deployment.md)).
- **Quiet mode** - `-q` prints errors only, keeping cron mail meaningful;
  `-v` narrates every turn and tool outcome to stderr.
- **Custom tools** - any executable becomes a tool via `tools.subprocess`:
  a Python script, a bash one-liner, a compiled binary - executable + argv,
  never a shell string. The shell tool exists, is off by default, and wants
  you to read [docs/shell-tool.md](docs/shell-tool.md) first.

## Session logs

With `session_dir` set, every run appends one JSONL file: the task, every
model turn, every tool call with its outcome and exit code, token
accounting, final status - the observability trail and future replay source
in one format ([contract](docs/contracts/jsonl-events.md)). Every
`${VAR}`-interpolated value is redacted from the log by value; see the
caveat about interpolating broad non-secrets like `${HOME}` in
[docs/session-logging.md](docs/session-logging.md).

## Security model

amele's own guardrails - the workspace-sandboxed fs tools, shell allow/deny
patterns, permission profiles - are **accident prevention, not a security
boundary**. A determined or prompt-injected model can route around a command
pattern, and by default subprocesses inherit amele's entire environment,
credentials included (a per-tool `env` allowlist narrows what the child
inherits, but not what a same-user process can reach). The boundary is the
OS: run amele in a container or VM holding only the files, network and
environment variables the agent legitimately needs. The full argument, with
concrete bypasses: [docs/shell-tool.md](docs/shell-tool.md).

The complete prompt-injection threat model - what amele defends, what it
deliberately only contains, and least-privilege config patterns with a
hardened log-watcher example - lives in
[docs/threat-model.md](docs/threat-model.md). Read it before pointing an
agent at attacker-influenced data (which, for a log watcher, is the logs).

## Contracts

Exit codes, the JSONL session-event schema, the CLI surface and the YAML
config schema are frozen public API, versioned in
[docs/contracts/](docs/contracts/). Scripts can rely on them: breaking
changes require a semver major and a migration note.

| exit | meaning |
|---|---|
| 0 | success |
| 1 | task failed or run interrupted |
| 2 | config or usage error |
| 3 | budget exceeded |
| 4 | permission denied, could not continue |
| 5 | provider/network error, retries exhausted |
| 6 | output schema unmet |
| 7 | run lock held by another run |
| 8 | a required MCP server was unavailable |

## Install

Prebuilt static binaries for Linux, macOS, Windows, the BSDs and more (see
[Runs everywhere](#runs-everywhere)) are on the
[releases page](https://github.com/lasthumanintheloop/amele/releases/latest).
Unpack, put `amele` on your `PATH`; that is the whole installation.

```console
$ v=0.1.0; os=linux; arch=amd64        # or darwin/arm64, windows/amd64 (.zip), ...
$ curl -fsSLO https://github.com/lasthumanintheloop/amele/releases/download/v$v/amele_${v}_${os}_${arch}.tar.gz
$ tar xzf amele_${v}_${os}_${arch}.tar.gz amele && sudo install amele /usr/local/bin/
$ amele version
```

Or with Go 1.25+, straight into `$GOBIN`:

```console
$ go install github.com/lasthumanintheloop/amele/cmd/amele@latest
```

Or from a checkout: `CGO_ENABLED=0 go build -o amele ./cmd/amele` (`make
dist` builds every platform's archive). Pure Go, no CGO - it cross-compiles
to everything Go supports. Shell completions: `amele completion
bash|zsh|fish` prints a static script.

### Verify a download

Every release ships a `SHA256SUMS` file, and `SHA256SUMS.sigstore.json`, a
[Sigstore](https://www.sigstore.dev/) keyless signature over it made by the
release workflow's own GitHub identity - no long-lived key anyone could
leak. Releases built by the workflow (tag identity, `@refs/tags/v...`) also
carry SLSA build provenance (`multiple.intoto.jsonl`); v0.1.0 predates the
workflow and was signed after the fact from `main` (`@refs/heads/main`),
without provenance.

```console
$ sha256sum -c --ignore-missing SHA256SUMS
amele_0.1.0_linux_amd64.tar.gz: OK
$ cosign verify-blob SHA256SUMS --bundle SHA256SUMS.sigstore.json \
    --certificate-identity-regexp '^https://github.com/lasthumanintheloop/amele/\.github/workflows/release\.yml@refs/tags/v' \
    --certificate-oidc-issuer https://token.actions.githubusercontent.com
Verified OK
```

The binaries are **not** Authenticode-signed or Apple-notarized (that takes
a paid certificate; not planned for now), so Windows SmartScreen and macOS
Gatekeeper will warn on first launch. The checksum plus the Sigstore
signature is the integrity check; the warning is about the absence of a
vendor certificate, not about the file. On macOS `xattr -d
com.apple.quarantine amele` clears it once.
