# Packs: sharing an agent as a folder

A *pack* is a folder that carries a complete amele agent: the YAML, its
tool scripts, its prompt files, its README. There is no package format, no
manifest and no registry - a pack is a convention over a plain directory,
distributed however folders are distributed (git clone, zip, rsync).

## Layout

    my-agent/
      agent.yaml     # canonical entry point (the only required file)
      README.md      # what it does, what it needs, how to run it
      tools/         # executables the pack ships (shell, PHP, anything)
      prompts/       # system_prompt_file targets (optional)
      sessions/      # run logs; keep it in .gitignore
      .gitignore     # at minimum: sessions/

A single `agent.yaml` with nothing else is a valid pack. Extra agents may
live beside it (`judge.yaml`); `agent.yaml` is what directory shorthand
resolves to.

## Running a pack

    git clone https://example.com/some-pack && cd some-pack
    amele explain .      # required env vars, host executables, env allowlists
    amele run . "task"

A directory argument is shorthand for `<dir>/agent.yaml` (run, chat,
validate, explain). The run lock derives from the resolved file, so
`amele run pack/` and `amele run pack/agent.yaml` are the same run.

## Path rules

Path-like relative commands resolve against the config file's directory;
bare names resolve from PATH:

    tools:
      subprocess:
        - name: scan
          description: Scan the logs
          command: ["./tools/scan.sh"]   # resolves against this folder
        - name: send_email
          description: Send a mail via msmtp
          command: ["msmtp", "${AMELE_MAIL_TO}"]   # bare name: PATH lookup;
                                                   # recipient pinned in argv

`workspace`, `session_dir` and `system_prompt_file` also resolve against
the config file's directory, so the folder works from any cwd.

Only `command[0]` gets this resolution - later elements are arguments and
reach the tool untouched. So run pack scripts through their shebang, not
through an interpreter prefix:

    command: ["./tools/analyze.py"]            # right: resolves, shebang picks python3
    command: ["python3", "./tools/analyze.py"] # wrong: the script path is an
                                               # argument and resolves against
                                               # the WORKSPACE at run time

**SECURITY: the interpreter-prefixed form is more than a path bug.** Tools run
with the *workspace* as their working directory, so `["python3",
"./tools/analyze.py"]` executes `<workspace>/tools/analyze.py` - a path the
model can create itself whenever `tools.fs: true` is set - and no execute bit
is needed, since the interpreter only has to read the file. The pack then runs
model-authored code under the pack's own tool name, and the tool call looks
exactly like the intended one in the session log. The shebang form resolves
against the config directory before the run starts and cannot be shadowed this
way.

Give every pack script a shebang line and the execute bit; `amele explain`
then also probes the actual script instead of just the interpreter.

## Shell `allow` patterns bound the prefix, not the command

If a pack enables the builtin [shell tool](shell-tool.md), remember that an
allow pattern constrains only the start of the command line - everything after
the matched prefix, including `;`-chained commands, runs too. `allow: ["date*"]`
passes `date; id`. Patterns are accident prevention, not a boundary; the OS
layer is (see [shell-tool.md](shell-tool.md)).

## Requirements are derived, not declared

`agent.yaml` is the machine-readable manifest: `amele explain <pack>`
derives required environment variables from `${VAR}` references and
required host executables from subprocess commands, marks what is MISSING
on this host, and lists each tool's `env:` allowlist. Keep the
human-readable story in README.md.

explain reports; run gates. On a fresh host - nothing exported, no
workspace yet - `amele explain <pack>` still prints the whole report and
exits 0, with the reasons it could not run in a `PROBLEMS` block at the
top. `amele run` remains the one that refuses (exit 2).

## Secrets

`provider.api_key` must be an `${ENV_VAR}` reference - amele rejects
literal keys there. For every OTHER field this is a convention you must
follow yourself: never commit a literal secret in prompts, tool arguments
or URLs; pass them all as `${ENV_VAR}`.

Name credential variables like credentials. `amele explain` prints
interpolated `${VAR}` values so an operator can pre-flight a parametrized
pack (`model: ${MODEL}`), and withholds only two kinds: whatever feeds
`provider.api_key`, and any variable whose name contains `key`, `token`,
`secret`, `passw` or `cred`. A credential in `${MY_THING}` will be printed
by the pre-flight report; `${MY_THING_TOKEN}` will not.

## Distribution notes

- Zip archives may drop execute bits; tell users to run `chmod +x tools/*`
  (put it in your README, as examples/log-sentry does).
- Scripts run with the interpreters their shebangs name; list interpreter
  requirements (php, python, …) in your README.
- Cron/systemd-targeted packs should set `lock: true` (see
  docs/deployment.md).
