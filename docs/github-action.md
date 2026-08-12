# amele as a GitHub Action

The repository root ships an [`action.yml`](../action.yml): a composite
action that builds amele from the action's own bundled source with
`actions/setup-go`, runs `amele run` on your config, and exposes the agent's
stdout as a step output. Nothing beyond what `ubuntu-latest` ships is
required - Go comes from `setup-go`, amele is one static binary.

There is no `version` input: the action lives *in* the amele repository and
builds the source it ships with, so the `uses: lasthumanintheloop/amele@<ref>` line is
the version pin. `@main` tracks the tip; pin a tag or commit SHA for
reproducible runs.

## Inputs and outputs

| Input | Required | Meaning |
|---|---|---|
| `config` | yes | Path to the agent YAML, relative to the workspace |
| `task` | no | Task text for `amele run` |
| `model` | no | Override the config's `model` (`--model`) |

| Output | Meaning |
|---|---|
| `answer` | The agent's final answer - `amele run` stdout: exactly the answer on success, empty on any failure |

The step's exit code is `amele run`'s exit code, unchanged - the
[exit-code contract](contracts/exit-codes.md) applies, so a failed agent
fails the workflow step, and `continue-on-error` / `if:` conditions can
branch on it like any other step.

## Minimal workflow

```yaml
name: nightly-triage
on:
  schedule:
    - cron: "0 6 * * *"

jobs:
  triage:
    runs-on: ubuntu-latest
    # Job-level env: it reaches every step in the job, this action included.
    env:
      AMELE_API_KEY: ${{ secrets.AMELE_API_KEY }}
    steps:
      # The action itself needs no checkout - it carries its own source.
      # You almost always want one anyway: the agent's config and workspace
      # files live in *your* repository.
      - uses: actions/checkout@v4

      - uses: lasthumanintheloop/amele@main   # pin a tag or SHA in real use
        with:
          config: agent.yaml
          task: "triage yesterday's logs"
```

The API key never appears in the YAML config or the workflow file's plain
text: the config references `${AMELE_API_KEY}` (the only form amele accepts -
plain keys in YAML are rejected), the workflow supplies it from repository
secrets via `env`, and amele redacts the interpolated value from its session
log.

Declare that `env:` on the **job**, as above, rather than on the amele step.
Job-level environment is inherited by every step of the job, so the variable
is there when the action's own steps run; it also keeps a second amele step
from needing its own copy. The env name is amele's convention, not a
requirement - the config decides which variable it interpolates, so a config
that reads `${OPENAI_API_KEY}` wants that name in the job's `env:` instead.

## Consuming the answer

With `output.schema` set in the config, stdout is canonical JSON that
validated against your schema - or the step fails with exit 6 and an empty
`answer`. That makes the output safe to feed straight into later steps:

```yaml
      - id: judge
        uses: lasthumanintheloop/amele@main
        with:
          config: judge.yaml
          task: "review the diff in review.patch"
        # No step `env:` - the job's AMELE_API_KEY above covers this step too.

      - name: Fail below threshold
        run: |
          score="$(jq .score <<< "$ANSWER")"
          echo "review score: $score"
          test "$score" -ge 7
        env:
          ANSWER: ${{ steps.judge.outputs.answer }}
```

(Passing the output through `env` rather than splicing
`${{ steps.judge.outputs.answer }}` into the script keeps model-produced text
from ever becoming shell syntax.)

## Security

The agent runs with the workflow's full permissions: its `GITHUB_TOKEN`, every
secret handed to the step, network access, the runner's filesystem. amele's own
sandboxing is [accident prevention, not a security boundary](shell-tool.md) -
a prompt-injected model can act with everything the step can reach. Scope
accordingly:

- Give the job the minimum [`permissions:`](https://docs.github.com/en/actions/security-for-github-actions/security-guides/automatic-token-authentication#permissions-for-the-github_token)
  block; default to `contents: read`.
- Pass only the secrets the agent needs - an API key, usually nothing else.
- Treat workflow-triggering content (PR titles, issue bodies) reaching the
  task or workspace as untrusted model input.
- Think twice before enabling the [shell tool](shell-tool.md) in CI configs.
