# Engineering standards

The rules this codebase is built and reviewed against. Code comments cite
sections of this file (e.g. "§5.5"); section numbers are stable.

## 1. Identity

One static Go binary plus one YAML file is a working agent. The measure is
usefulness, not surface: put an agent where deploying one is currently hard,
with two files.

## 2. Invariants

1. **Single binary** - the core runner has zero runtime dependencies (no
   Python, no Node).
2. **Pure Go, no CGO** - `CGO_ENABLED=0`, static linking, free
   cross-compilation.
3. **YAML is the declarative entry point** - behavior is config, not code.
4. **Headless-first** - every feature is designed to be safe under cron/CI
   first; interactive use is secondary.
5. **Anti-features** (requests we decline): a GUI, an embedded RAG/vector
   store, a workflow DSL or graph engine, a Go plugin system, a scheduler, a
   multi-tenant server. The extension path is subprocess tools and, later,
   MCP.

## 3. Language

Code, comments, commit messages, godoc and error messages are English. CLI
output is English (machine parseability first). The occasional Turkish
working note in a comment is a deliberate exception, not an invitation.

## 4. Repository layout

```
cmd/amele/         CLI entry point (flag parsing + wiring only, no business logic)
internal/config/   YAML schema, validation, env interpolation, JSON Schema
internal/llm/      provider abstraction + OpenAI-compatible and Anthropic clients + fake
internal/tools/    tool registry + builtins (fs, subprocess, shell)
internal/loop/     agent loop: turns, budget enforcement, stop conditions
internal/session/  append-only JSONL: log = session = replay, one format
docs/              user-facing docs + frozen contracts (docs/contracts/)
testdata/          golden files, example YAML, recorded provider responses
```

Packages under `internal/` connect through interfaces only. `cmd/` holds no
business logic. A package answers one question.

## 5. Code standards

### 5.1 Style and static analysis

`gofmt` + `goimports` are mandatory. `golangci-lint` (see `.golangci.yml`)
is a merge gate: govet, errcheck, staticcheck, revive, gosec, ineffassign,
misspell, gocyclo (threshold 15). A `//nolint` must be single-line, linter-
specific, and carry a justification. No global mutable state; dependencies
are injected through constructors. Interfaces are declared in the consuming
package.

### 5.2 Comment policy

Uncommented code does not merge. Every package has a package comment; every
exported symbol has a godoc that starts with its name and states the
contract (behavior, error cases, concurrency). Non-obvious blocks carry a
"why" comment: the decision taken, the alternative rejected, the constraint
it rests on. Budget/security/contract enforcement points are marked
`// CONTRACT:` or `// SECURITY:` so they stay greppable. TODOs carry an
issue reference or they do not exist.

### 5.3 Errors

Errors are wrapped with context (`fmt.Errorf("loading config: %w", err)`).
Programmatic distinctions use sentinels or typed errors, never string
matching. No `panic` in library code. Every error path maps to the frozen
exit-code contract (§7).

### 5.4 Determinism and testability

`time.Now`, `rand` and `os.Getenv` are never called directly; a Clock, Rand
or Env lookup is injected. Every long operation takes a `context.Context`
as its first parameter. No goroutine leaks: every goroutine has an owner
and a shutdown path.

### 5.5 Security rules

Secrets are never logged; known secret values are redacted by value before
anything reaches the JSONL. Plain API keys in YAML are rejected; only
`${ENV_VAR}` interpolation. Filesystem tools are workspace-sandboxed via a
directory handle, not string prefixing. Tool definitions take an executable
plus argv vector, never a shell string. With no TTY, every `ask` policy
degrades to a logged deny.

## 6. Testing

TDD: the failing test comes first; behavior without a test does not exist.
Unit tests touch no network and call no LLM (the fake provider plays
recorded responses). Table-driven is the default shape. Coverage across
`internal/` stays >= 80% and `go test -race` runs in every CI pass. JSONL
and `validate`/`explain` output are golden-file tested; goldens change only
through the `-update` flag, reviewed by eye. The config parser has a fuzz
test. Live-API smoke tests sit behind `//go:build integration`. Every bugfix
ships with a regression test that reproduces the bug first.

## 7. Frozen contracts (public API)

Breaking any of these requires a semver major and a migration note in
`docs/contracts/`:

1. **Exit codes** (`docs/contracts/exit-codes.md`): 0 success, 1 task
   failed/interrupted, 2 config error, 3 budget exceeded, 4 permission
   denied, 5 provider error, 6 output schema unmet, 7 run lock held,
   8 required MCP server unavailable. New codes are additive.
2. **The YAML config schema**, published as JSON Schema.
3. **The JSONL event schema** (`docs/contracts/jsonl-events.md`).
4. **The CLI surface**: command and flag names.

Budget enforcement uses the token counts the provider reports as the
primary unit; USD is informational only.

## 8. CI budgets

Measured on every PR; exceeding one fails the build. Binary size <= 14 MB
(target 12 MB, `-ldflags="-s -w" -trimpath`; raised from 10 MB on 2026-08-19
for the MCP go-sdk. Measured with the MCP client in: **9,777,336 bytes**
(9.3 MB) on linux/amd64 - the +4.6 MB the SDK adds to a bare binary is partly
absorbed by dependencies amele already carried). Harness token load (system
prompt + builtin tool definitions) <= 1500 tokens. Coverage >= 80% across
`internal/`. Builds are reproducible: `-trimpath`, pinned Go version,
verified `go.sum`.

## 9. Commits and PRs

Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`,
`chore:`). Small, single-purpose commits whose body explains the why. One PR
is one logical change; a frozen-contract change (§7) ships in its own PR
titled `contract:`.
