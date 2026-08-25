# Providers: one config, many endpoints

amele speaks three wire formats: OpenAI-compatible `chat/completions` (the
default), the native Anthropic Messages API (`provider.type: anthropic`) and
the native Google Gemini `generateContent` API (`provider.type: gemini`). The
first one is nominally a single protocol and is in practice a family of them -
the providers speaking it disagree about which field carries the output cap,
how the reasoning knob is spelled, and whether the model's own reasoning has to
be handed back to it inside a tool loop. `provider.dialect` names which
variation your endpoint speaks, and one config can then say what it *wants*
once and be run against any of them. The two native wires are their own
families, not variations: `dialect` is ignored on one and refused on the other.

The promise is not "every feature works everywhere". It is:

> Nothing amele sends is silently dropped, and nothing a provider requires -
> reasoning echo-back above all - is silently violated. Every mapping and
> every degradation is visible in `amele explain` and in the session log.

**How to read this page.** Every behavior described here is either pinned by a
test in this repository or verified against the provider's official
documentation on 2026-08-24. Where neither is true - a provider documents
nothing, or documents two contradictory things - the claim is marked
**unverified** and says what amele does about it. Model ids and base URLs are
the parts that rot fastest; copy them from your provider's own docs if they
have moved.

## The tuning surface

```yaml
provider:
  type: openai                # openai (default) | anthropic | gemini - the WIRE
  dialect: deepseek           # openai (default) | deepseek | glm | kimi
                              #                  | groq | openrouter
  base_url: https://api.deepseek.com
  api_key: ${DEEPSEEK_API_KEY}
  max_output_tokens: 8192     # the per-request output ceiling
  reasoning:
    effort: high              # none | low | medium | high | xhigh | max
    budget_tokens: 8192       # a token count instead of a level; anthropic
                              # wire / gemini wire / openrouter only
  temperature: 0.2
  top_p: 0.9
  retry:
    max_attempts: 5           # total tries per call, 1 disables retrying
    initial_backoff: 2s       # first wait; doubles for each attempt after
  params:                     # escape hatch: merged verbatim into the body
    provider: {require_parameters: true}   # e.g. an OpenRouter routing rule
```

| Key | Default | Meaning |
| --- | --- | --- |
| `type` | `openai` | The wire format. `anthropic` selects the native Messages API, where `dialect` is not consulted at all; `gemini` selects the native `generateContent` API, where a `dialect` is a config error (exit 2) rather than an ignored key. |
| `dialect` | `openai` | Which variation of the OpenAI-compatible format this endpoint speaks. **Explicit, never sniffed** from `base_url`: a silently chosen dialect would reshape every request in a way the YAML file does not show. `explain` prints a hint when the host is one it recognizes. |
| `max_output_tokens` | none (openai and gemini wires) / `8192` (anthropic wire) | The per-request output ceiling. The dialect picks the field name (`max_completion_tokens` or `max_tokens`; `generationConfig.maxOutputTokens` on the gemini wire); the Messages API requires the field, so that wire always sends one. **Reasoning tokens are billed against this ceiling** on every provider that reports them. |
| `reasoning.effort` | none | The reasoning depth, in a vocabulary that is the UNION of what the providers accept - no single provider takes all six values. A dialect with a smaller vocabulary rounds **up** to the nearest level it has and says so in `explain`: the config asked for at least this much thinking, and quietly buying less is the failure mode you cannot see in the output. |
| `reasoning.budget_tokens` | none | A token budget instead of a level. Only the anthropic wire, the gemini wire and the `openrouter` dialect can carry one; anywhere else it is a config error (exit 2) rather than a silently dropped field. On the gemini wire it is an *alternative* to `effort`, never a companion - setting both is exit 2. |
| `temperature` | none | `[0, 2]`, narrowed to `[0, 1]` on the anthropic wire and on the `glm` dialect. Unset means the provider decides - which is not the same as `0`. |
| `top_p` | none | `(0, 1]`. `0` is rejected rather than clamped: an empty nucleus is a 400, not greedy decoding. |
| `retry.max_attempts` | `3` | Total tries for one provider call (1 initial + retries), so `1` disables retrying. Accepted range 1..10; `0` (or omitted) means the default. |
| `retry.initial_backoff` | `1s` | The wait before the second attempt; each further attempt doubles it, up to a 60s ceiling per wait. Accepted range `100ms`..`60s`; empty (or omitted) means the default. |
| `params` | none | Arbitrary keys merged verbatim into the request body root, for provider extras amele has no neutral field for. Keys amele writes itself **on the active target** are a config error, so `params` can extend a request but never rewrite one - see [What `params` may not carry](#what-params-may-not-carry). |

`provider.max_output_tokens`, `provider.reasoning.effort`,
`provider.temperature` and `provider.top_p` are in the `--set` override
allowlist, so one config can be swept over several settings from a shell loop.
`provider.dialect`, `provider.reasoning.budget_tokens`, `provider.retry.*` and
`provider.params` are deliberately not ([cli.md](contracts/cli.md#set)).

### What `params` may not carry

The refused set is **per target**: a key is a collision only where amele
actually writes it.

| target | keys amele writes itself |
| --- | --- |
| every openai-wire dialect | `model`, `messages`, `tools`, `response_format`, `temperature`, `top_p` |
| + `openai`, `groq` | `max_completion_tokens`, `reasoning_effort` |
| + `deepseek`, `glm` | `max_tokens`, `reasoning_effort`, `thinking` |
| + `kimi` | `max_completion_tokens`, `reasoning_effort` |
| + `openrouter` | `max_tokens`, `reasoning` |
| anthropic wire | `model`, `max_tokens`, `system`, `messages`, `tools`, `thinking`, `output_config`, `temperature`, `top_p` |
| gemini wire | `contents`, `systemInstruction`, `tools`, `toolConfig`, `generationConfig`, `safetySettings`, `cachedContent` - each in **both** spellings (protobuf JSON accepts the snake_case form too) |

Two more are refused everywhere, for a different reason - not because amele
writes them, but because its own machinery cannot survive them: `stream` (the
clients read a single JSON body; an SSE stream is a parse error) and
`tool_choice` (the loop stops when the model answers without calling a tool, so
a pinned `required` turns every run into a `max_turns` failure).

Everything else is yours. That is what makes a provider-specific control
reachable even when amele has no neutral field for it: `thinking` is a config
error on `deepseek`, where amele emits one, and perfectly legal on `kimi`, where
it does not. Change the dialect and `amele validate` re-answers the question
before the run.

On the gemini wire `params` merges into the request body **root**, which is the
only level it reaches: `generationConfig` sub-keys (`topK`, `seed`,
`stopSequences`) are not addressable through `params` in this slice, because
merging into a nested object amele also writes would need a merge rule this
strict endpoint gives no room to get wrong. Ask for a neutral field if you need
one.

### Retries

A 429, a 5xx or a dropped connection is retried; everything else is returned to
you as it is. **Which** failures are retried is not configurable, because that
list is not a matter of taste: a 400 is a request the provider will refuse just
as firmly on the next attempt, and retrying it would only spend your budget
slower. What you can set is the rhythm - `retry.max_attempts` tries in total
(`1` means "never retry") with `retry.initial_backoff` before the second one,
doubled for each attempt after that: `2s` gives 2s, 4s, 8s. A `Retry-After`
header from the provider **stretches** an individual wait - never shrinks it -
because retrying earlier than the rate limiter allows burns an attempt for
nothing. **No single wait exceeds 60s**, whichever produced it: the doubling
flattens at the same ceiling, so the longest ladder you can configure
(`max_attempts: 10`) spends at most 9 x 60s - roughly 9 minutes - asleep,
rather than the hours plain doubling would reach on the last rungs. All three
wires share the policy: the same block applies whichever `type` you set. The
gemini wire is the one that carries the provider's wish somewhere else - a 429
there has **no `Retry-After` header**, and the delay arrives in the error body
as a `google.rpc.RetryInfo` detail; amele reads it and feeds the same
stretch-never-shrink mechanism. When the attempts run out, the run ends as a provider error
(exit 5), so a longer ladder trades latency for surviving a rate-limit window.
`limits.timeout` stays the wall-clock kill switch above all of it: a backoff
wait is cut short when the run deadline fires, and the run then ends as a
budget timeout (exit 3), not as a provider error.

## The dialect table

What each dialect makes of the same config, on the OpenAI-compatible wire:

| dialect | `reasoning.effort` becomes | output cap field | reasoning returned / echoed back | sampling | `output.schema` | unknown request fields |
| --- | --- | --- | --- | --- | --- | --- |
| `openai` | `reasoning_effort` (verbatim) | `max_completion_tokens` | nothing returned on this API | passed through | native `response_format: json_schema` | rejected (400) |
| `deepseek` | `thinking: {"type":"enabled"}` + `reasoning_effort` (medium->high, xhigh->max); `none` -> `thinking: {"type":"disabled"}` | `max_tokens` | `reasoning_content`, echoed back on **every** later request | passed through (ignored while thinking) | `json_object` + validate + retry | ignored (undocumented, observed) |
| `glm` | same as `deepseek` | `max_tokens` | `reasoning_content`, echoed back | passed through; `temperature` outside `[0, 1]` is a validate error (exit 2) | `json_object` + validate + retry | not documented; assume rejected (400) |
| `kimi` | `reasoning_effort` (medium->high, xhigh->max), no thinking object | `max_completion_tokens` | `reasoning_content`, echoed back | **config error**: the K-series fixes `temperature`/`top_p` | attempted, then fallback | not documented; assume rejected (400) |
| `groq` | `reasoning_effort` (verbatim) | `max_completion_tokens` | `reasoning_content` echoed back; a bare `reasoning` is captured for the log only | passed through | `json_schema` sent, then fallback - native support **unverified** | not documented; assume rejected (400) |
| `openrouter` | `reasoning: {"effort": ...}` (verbatim; `budget_tokens` -> `reasoning: {"max_tokens": N}`) | `max_tokens` | `reasoning_details` array, echoed back verbatim and in order | passed through (the gateway drops what the model cannot take) | native `json_schema` - but see [OpenRouter](#openrouter) | passed through to the upstream provider |

Three rules are worth stating outside the table.

**Behavior keys on the dialect, never on the model name.** Model ids churn
within months - `deepseek-chat` and `deepseek-reasoner` were retired in July
2026, `kimi-k2-thinking` in May 2026 - so a model-behavior table in amele would
be wrong before your next upgrade. Model-level quirks are recognized from the
provider's own error text instead and answered with advice (see [When a
provider says no](#when-a-provider-says-no)).

**The reasoning carrier is unconditional.** DeepSeek thinks by default; Kimi's
K-series and GLM-5.3 cannot be switched off at all. So capturing the reasoning
payload and echoing it back is dialect behavior that runs even when your config
never mentions reasoning. `provider.reasoning` is opt-in; the carrier is not.

**The payload is opaque.** amele stores what the wire returned as raw bytes and
writes those same bytes back in the dialect's own shape - never parsed, never
reordered, never truncated. It has to be byte-exact: Anthropic signs its
thinking blocks, and OpenRouter requires the block sequence to match "the
outputs generated by the model during the original request".

## The Anthropic wire

`provider.type: anthropic` selects the native Messages API. The dialect is not
consulted there (a leftover `dialect:` is reported as ignored, not as an
error), and the reasoning knob takes a different shape:

| config | request |
| --- | --- |
| `reasoning.effort: high` | `thinking: {"type":"adaptive"}` + `output_config: {"effort":"high"}` - the current models' shape |
| `reasoning.effort: none` | `thinking: {"type":"disabled"}` |
| `reasoning.budget_tokens: 8192` | `thinking: {"type":"enabled","budget_tokens":8192}` - the legacy shape, Haiku 4.5 and older. A budget **wins** over an effort: the two target different model generations and cannot be combined. |

- The effort vocabulary needs no rounding: Anthropic's own levels are
  low/medium/high/xhigh/max, which is the neutral union minus `none`.
- `max_output_tokens` is required on every request; without one amele sends
  **8192**. On this wire a `budget_tokens` is drawn from the same ceiling, so a
  budget at or above the ceiling leaves nothing for the answer and is a config
  error (exit 2) rather than a 400 on your first unattended run. Leaving
  `max_output_tokens` out does not lift the ceiling - the check then runs
  against that 8192 default, because that is the number the request will carry.
- `output.schema` is enforced natively (`output_config.format`), with the
  validate+retry layer still behind it.
- `temperature` above 1 is a config error; current models reject any
  non-default sampling value outright, which amele can only recognize at run
  time (see below).
- The two thinking shapes are **not** interchangeable: `adaptive` is a 400 on
  Claude 4.5 and older, and the legacy `enabled` shape is a 400 on 4.7+. amele
  sends what your config asked for and lets the provider's own error name the
  mismatch - it does not guess a model generation.

DeepSeek, GLM and Kimi all publish Anthropic-compatible endpoints
(`api.deepseek.com/anthropic`, `api.z.ai/api/anthropic`,
`api.moonshot.ai/anthropic`). They work through this same client via
`base_url`, with the provider's own deviations - DeepSeek ignores
`budget_tokens` there, for instance. Those are documentation facts, not amele
behavior: nothing in the code special-cases them.

## The Gemini wire

`provider.type: gemini` selects the native `generateContent` API. It is a third
family rather than a variation: a `dialect:` next to it is a **config error**
(exit 2), not an ignored key, because nothing was ever written against this wire
and strictness costs no working config while buying you the certainty that no
knob is quietly dropped.

Today amele speaks the AI Studio half of this API: `api_key` is the
`x-goog-api-key` header and is **required** (`gemini needs api_key` at exit 2).
Vertex AI - Google credentials, a project and a region - is a later slice and
arrives as a `vertex:` block; until then there is no way to reach it, which the
validate message says out loud rather than leaving you to read a 401.

`base_url` is optional (the client knows `generativelanguage.googleapis.com`)
and must **not** carry the version segment: amele appends
`/v1beta/models/{model}:generateContent` itself, so a `base_url` ending in
`/v1beta` or `/v1` is exit 2 rather than a 404 at 03:00.

| config | request |
| --- | --- |
| `reasoning.effort: low\|medium\|high` | `generationConfig.thinkingConfig.thinkingLevel` with the same word |
| `reasoning.effort: xhigh\|max` | `thinkingLevel: high` - this wire has nothing above it, and `explain` prints the rounding |
| `reasoning.effort: none` | `thinkingConfig.thinkingBudget: 0`. Gemini 3 models **cannot** stop thinking and answer this with a 400; amele does not guess a model generation from its name, so that stays a runtime error with advice |
| `reasoning.budget_tokens: 8192` | `thinkingConfig.thinkingBudget: 8192` (the 2.5-era spelling). A level and a budget are **alternatives** - both together is exit 2, because the API refuses the pair |
| `max_output_tokens: 4096` | `generationConfig.maxOutputTokens` |
| `temperature` / `top_p` | `generationConfig.temperature` / `topP`, sent as given |
| `output.schema` | `generationConfig.responseJsonSchema` + `responseMimeType: application/json` |
| `params` | merged into the body **root**, never into `generationConfig` |

Five things behave differently enough here to state on their own.

**Thought signatures round-trip untouched.** Gemini 3 signs the steps it
produces, and a tool loop that sends a step back without its signature is a 400.
amele stores the model turn's **entire raw `parts` array** and re-emits it
verbatim as the next request's model content - never parsed, never reordered,
never rebuilt - so the signature stays on the part it belongs to by
construction. If you ever see a `missing a thought_signature` 400 from amele,
that is a bug in amele: it says so in the error advice, and the session log is
what to attach to the report.

**Tool schemas are sanitized, and the strip is announced.** A
`FunctionDeclaration.parameters` is an OpenAPI-3.0 **subset**, not JSON Schema,
and an unknown keyword is a hard 400 that fails the *whole* request - every
tool, not just the offending one. amele therefore strips what the subset cannot
carry (`additionalProperties`, `$schema`, `$ref`, `pattern`, ... - it is an
allowlist, so a keyword invented after this release goes too) from its own
builtins and from whatever schemas your MCP servers publish. The cost is a
constraint the model no longer sees, so nothing is stripped silently:
`amele explain` lists the removed paths per tool, and a run prints one warning
line naming them. Values are never printed - only key paths.

**A malformed function call ends the turn.** `MALFORMED_FUNCTION_CALL` is
reported as a provider error (exit 5) with the advice to simplify the tool's
parameter schema, rather than being handed to the loop as an empty answer that
would exit 0 on a broken turn. Like every error turn on every wire, it carries
**no usage** into the run's accounting: the tokens it burned are real but
unreported, so a `limits.max_tokens` budget cannot see them.

**Thinking is billed as output.** `usage.output_tokens` is
`candidatesTokenCount` **plus** `thoughtsTokenCount`, because that is what
Google bills - a run whose budget ignored the thinking half would be off by the
most expensive part of the turn.

**Sampling has a recommendation, not a rule.** Google documents `1.0` as the
Gemini 3 default and recommends leaving it alone. amele sends whatever your
config says - second-guessing it would be the silent degradation this page
exists to prevent - and `explain` prints a note next to the value:
`google recommends the default 1.0 on gemini 3 models; non-default may degrade output`.

Two claims on this wire are **unverified** until the live smoke test in this
milestone runs, and both are recorded here rather than smoothed over:

- **The structured-output field form.** The SDK documents
  `generationConfig.responseJsonSchema`, which is what amele sends; a newer
  nested `responseFormat` shape appears in other Google documentation. If the
  endpoint answers with a 400 naming the field, amele repeats that one request
  without the schema - immediately, costing no retry budget - warns that native
  enforcement was lost, and enforces `output.schema` itself. So the *contract*
  holds either way; which spelling actually travels is the open question.
- **Whether thinking debits `maxOutputTokens`.** On every other provider
  reasoning is drawn from the output ceiling. Assume it does here too and leave
  room in `max_output_tokens`; the alternative failure mode is a `MAX_TOKENS`
  finish (exit 1) in the middle of an answer.

## Quickstarts

Copy one, set the key in your environment, run `amele explain` before you spend
a token.

The `base_url` and model values below are what each provider publishes today
and are **not pinned by any test here** - they are the fastest-rotting part of
this page. Note that the OpenAI-compatible endpoint sits at a different path
per provider (`/openai/v1` at Groq, `/api/paas/v4` at Z.ai, the bare root at
DeepSeek); take the exact one from your provider's own documentation.

### OpenAI

```yaml
model: gpt-5.5
provider:
  base_url: https://api.openai.com/v1
  api_key: ${OPENAI_API_KEY}
  max_output_tokens: 32768        # reserve room: reasoning is billed here too
  reasoning: {effort: medium}
```

### Anthropic

```yaml
model: claude-haiku-4-5
provider:
  type: anthropic                 # native Messages API; no /v1 in base_url
  api_key: ${ANTHROPIC_API_KEY}
  max_output_tokens: 16384
  reasoning: {budget_tokens: 4096}   # Haiku 4.5 and older; newer: effort
```

### Gemini (AI Studio)

```yaml
model: gemini-3-pro
provider:
  type: gemini                    # native generateContent; no version in base_url
  api_key: ${GEMINI_API_KEY}      # required today - Vertex needs the vertex block
  max_output_tokens: 8192         # leave room: thinking is billed here too
  reasoning: {effort: medium}     # -> thinkingConfig.thinkingLevel: medium
```

`api_key` is not optional on this wire, and `base_url` is: the client knows the
first-party host and appends the version itself. Swap `effort` for
`budget_tokens: 8192` on a 2.5-era model - one or the other, never both. If your
agent has tools, run `amele explain` once and read the `tool schemas:` rows:
they list every JSON Schema keyword this API cannot carry, per tool, before the
first request pays for the discovery.

### DeepSeek (native)

```yaml
model: deepseek-v4-flash
provider:
  dialect: deepseek
  base_url: https://api.deepseek.com
  api_key: ${DEEPSEEK_API_KEY}
  max_output_tokens: 8192
  reasoning: {effort: low}
```

DeepSeek thinks by default, so a config with no `reasoning` block still gets
reasoning back - and still has to echo it. Set `effort: none` to turn thinking
off. That default also decides what your sampling knobs do: `temperature` and
`top_p` are accepted and then **silently ignored while thinking**, so `explain`
prints `temperature/top_p: sent but ignored by deepseek in thinking mode` next
to the values rather than promising an effect the run will not have. With
`effort: none` they take effect and the caveat disappears. `output.schema`
travels as `response_format: {"type":"json_object"}` here - there is no
`json_schema` on this API - and amele enforces the schema itself.

### GLM (Z.ai)

```yaml
model: glm-5.3
provider:
  dialect: glm
  base_url: https://api.z.ai/api/paas/v4
  api_key: ${ZAI_API_KEY}
  max_output_tokens: 16384
  reasoning: {effort: high}
  temperature: 0.3                # GLM's range is 0..1, not 0..2
```

### Kimi (Moonshot)

```yaml
model: kimi-k3
provider:
  dialect: kimi
  base_url: https://api.moonshot.ai/v1
  api_key: ${MOONSHOT_API_KEY}
  max_output_tokens: 32768        # Kimi advises >= 16000 when tools are in play
  reasoning: {effort: high}
```

`temperature` and `top_p` are a **config error** on this dialect: the K-series
pins them and answers any other value with a 400, so amele refuses at validate
instead of letting the run die at 03:00. `reasoning.effort: none` is refused
for the same reason - these models cannot stop thinking.

amele sends no `thinking` object on this dialect: K3 has no such parameter, and
emitting one would be a guess at which model generation is behind the name.
Because amele writes no `thinking` here, `params` may: the older K2.x controls
(`thinking: {type, keep}`) are reachable through the escape hatch, and
`reasoning.effort` remains the neutral knob for everything else.

```yaml
provider:
  dialect: kimi
  params:
    thinking: {type: enabled, keep: true}   # K2.x only; amele writes none
```

### Groq

```yaml
model: openai/gpt-oss-20b
provider:
  dialect: groq
  base_url: https://api.groq.com/openai/v1
  api_key: ${GROQ_API_KEY}
  max_output_tokens: 8192
  reasoning: {effort: low}
```

Groq takes `reasoning_effort` and amele passes the value through verbatim.
**Which values a given Groq-hosted model accepts is model-dependent and not
verified here** - the vocabulary differs per model family, and amele keeps no
per-model table. If the model does not know your value, Groq's own 400 names
it. `output.schema` is in the same position: whether a Groq-hosted model
enforces a schema natively is unverified here, so amele sends it and falls back
- see [Structured output](#structured-output).

Reasoning comes back under `reasoning_content` if the model uses that spelling,
and otherwise under the bare `reasoning` field Groq's documentation describes
(also unverified here). A payload read from `reasoning` is **captured but not
echoed**: it counts in the session log's `reasoning_bytes`, so the turn's cost
is visible, but nothing sends it back - no source establishes a request-side
spelling for that key, and this dialect's unknown-field policy is "assume
rejected". A payload read from `reasoning_content` is echoed as on every other
dialect.

### OpenRouter

```yaml
model: anthropic/claude-sonnet-5
provider:
  dialect: openrouter
  base_url: https://openrouter.ai/api/v1
  api_key: ${OPENROUTER_API_KEY}
  max_output_tokens: 16384
  reasoning: {effort: high}       # the gateway maps it per upstream provider
  params:
    provider: {require_parameters: true}
```

OpenRouter's `json_schema` support is a **soft preference**: it is silently
ignored when the endpoint serving your request cannot honor it, and support
varies per endpoint rather than per model. `require_parameters: true` turns
that into routing - the gateway only picks providers that support every
parameter you sent. Without it, a dropped `json_schema` is the one degradation
amele cannot report: an ignored field produces no 400, so there is nothing to
warn about, and the run's only enforcement is the local validate+retry layer.
That still holds the exit-code contract - stdout is schema-valid JSON or the
run is exit 6 - it just costs retries no one asked for.

Because the reasoning payload carries upstream signatures, a session's
reasoning cannot be replayed against a different model. That is a property of
the gateway, not of amele - amele only sends back what it received, for the
model that produced it.

## Reasoning costs tokens twice

Every provider that returns reasoning charges it against the **output** cap of
the turn that produced it. Then, in a tool loop, that same reasoning is sent
back with the next request - where it is billed as **input**, again, on every
later turn of the run.

That has three practical consequences.

1. **Leave room in `max_output_tokens`.** A cap sized for the answer alone is
   how a run ends with `finish_reason: length` in the middle of a JSON
   document. OpenAI advises reserving 25k for reasoning; Kimi advises at least
   16000 when tools are involved. A config that sends no cap at all inherits
   the provider's default, which may be much smaller than you think.
2. **Size `limits.max_tokens` for the echo.** The run budget counts input
   tokens, and a long tool loop re-sends the accumulated reasoning every turn.
   Crossing the budget is exit 3, not a warning.
3. **Watch `reasoning_bytes`.** Every `llm_response` event in the
   [session log](contracts/jsonl-events.md#llm_response---one-per-provider-round-trip)
   carries the byte size of that turn's reasoning payload (never its content -
   the model's scratchpad is not written to disk). It is what answers "why did
   that turn cost so much?".

If a run does not need the depth, `reasoning: {effort: low}` or `none` is the
cheapest change available - and on the dialects that cannot turn thinking off,
the answer is a smaller model rather than a knob.

## Structured output

[`output.schema`](features.md#structured-output-outputschema) works on every
provider; what differs is who enforces it.

- **Natively** on `openai`, `openrouter`, the anthropic wire and the gemini
  wire: the schema travels in the request (`response_format: json_schema`,
  `output_config.format` on the Messages API, or
  `generationConfig.responseJsonSchema` + `responseMimeType` on
  `generateContent` - the last one **unverified**, see [The Gemini
  wire](#the-gemini-wire)).
- **By `json_object` + local enforcement** on `deepseek` and `glm`. Neither has
  `json_schema` on this wire, so amele sends the JSON mode they do have -
  `response_format: {"type":"json_object"}` - in the first request and enforces
  the schema itself. No capability probe is involved: sending a `json_schema`
  those endpoints cannot accept would buy a guaranteed 400 and a second
  round-trip on **every turn**, and the repeat would then carry no JSON
  instruction at all. An endpoint that refuses `response_format` *entirely*
  still degrades the same way the fallback below does - one immediate repeat
  without the field - so a strict proxy costs the JSON hint, not the run.
- **By fallback** on `kimi` and `groq`, whose support is unverified (below).
  When the endpoint answers a schema-carrying request with a 400 naming the
  field, amele repeats that one request without it - once, immediately, costing
  no retry budget - and enforces the schema itself. The gemini wire carries the
  same fallback behind its native attempt, for the field-form question the
  section above records.

"Enforces the schema itself" means the same thing in both cases: the answer is
validated against `output.schema` and violations are fed back to the model for
up to `output.max_schema_retries` repair rounds.

Local enforcement is never silent. Whenever the schema did not travel with the
request - a `json_object` dialect, or a fallback that fired - the run prints a
warning on stderr (`provider did not enforce output.schema natively; the
validate+retry layer was the only enforcement`). The exit-code contract is
unchanged either way: stdout is a schema-valid JSON document, or the run fails
with exit 6.

For two dialects the endpoint's own support is **unverified** here - no test in
this repository pins it and the 2026-08-24 documentation sweep did not
establish it:

- **`kimi`.** Its API reference lists `json_schema` while its guide documents
  `json_object`, and the interaction with thinking is undocumented.
- **`groq`.** Groq's structured-output support is not verified here and varies
  per hosted model, the same way its `reasoning_effort` vocabulary does.

For both, amele sends the schema and takes the fallback if the endpoint refuses
it - the right behavior whichever way that documentation settles. The case
worth knowing about is the third one: an endpoint that **accepts the field and
ignores it** answers with no 400, so there is nothing to warn about and the
local validate+retry layer is the only enforcement (the same silent
degradation described under [OpenRouter](#openrouter)). The exit-code contract
holds regardless.

## When a provider says no

Some failures are configuration mistakes wearing a 400. amele recognizes a
small, fixed set of them and appends advice in the vocabulary of the YAML file
you are holding. The detection is a string match on the provider's error text
by necessity, and it changes **nothing** but the message - no retry, no
downgrade, no rewritten request. A provider that rewords its error costs you a
hint, never correctness.

| The provider says | amele adds |
| --- | --- |
| `Function tools with reasoning_effort are not supported for ...` | set `provider.reasoning.effort: none` for this model on chat/completions, or use a different model |
| `'max_tokens' is not supported ... use 'max_completion_tokens'` | this model requires `max_completion_tokens`; set `provider.dialect` to a dialect that maps it (`openai`/`groq`/`kimi`) |
| `Unsupported value: 'temperature' does not support ...` | this model rejects non-default sampling; remove `provider.temperature`/`top_p` |
| `temperature may only be set to 1 when thinking is enabled` (anthropic) | this model rejects non-default sampling; remove `provider.temperature`/`top_p` |
| a 400 naming both `thinkingLevel` and `thinkingBudget` (gemini) | set only one of `provider.reasoning.effort` or `budget_tokens` |
| `... budget 0 is invalid` / `cannot disable thinking` (gemini) | this model cannot disable thinking; remove `reasoning.effort: none` |
| `Unknown name ...` (gemini) | the gemini API rejects unknown fields; if this key came from `provider.params`, remove it |
| a 400 saying a part is missing a `thought_signature` (gemini) | amele must echo signatures automatically - this is a bug, please report it with the session log |

### The gpt-5.6 chat/completions restriction

On OpenAI's `chat/completions`, the gpt-5.6 family rejects a request that
carries **function tools together with any `reasoning_effort` other than
`none`**. The default effort is `medium`, so an agent with tools breaks out of
the box on those models. gpt-5.5 is not affected. (The restriction is in no
official document; it is confirmed by the API's own error text and by several
independent bug trackers - the error string is the only detection contract
there is.)

Two workarounds, both one line:

```yaml
provider:
  reasoning: {effort: none}   # tools, no reasoning
```

```yaml
model: gpt-5.5                # reasoning, tools, older family
```

amele does **not** silently downgrade the effort for you. Choosing between "no
thinking" and "another model" is a decision about the agent's quality, and an
agent framework that made it behind your back would be lying about what it
sent.

## What amele does not do

- **No dialect auto-detection.** `explain` prints
  `hint: base_url looks like api.deepseek.com; consider dialect: deepseek` when
  it recognizes a host (`api.deepseek.com`, `api.z.ai`, `open.bigmodel.cn`,
  `api.moonshot.ai`, `api.moonshot.cn`, `api.groq.com`, `openrouter.ai`,
  `api.openrouter.ai`) and your config picked something else. It is a hint. You
  pick.
- **No model-capability database.** Nothing in amele maps a model id to a
  behavior, because such a table is wrong within a release cycle.
- **No auto-downgrade.** amele never removes a parameter you set to make a
  request succeed. The one place it drops a field - `response_format` (or
  `output_config`) on an endpoint that refuses it - is announced as a warning
  and the schema is enforced anyway.
- **No OpenAI Responses API.** `chat/completions` is the only OpenAI transport,
  and streaming is a later slice ([roadmap](../README.md#roadmap)).

## Verify it before you trust it

```console
$ amele explain agent.yaml
MODEL & PROVIDER
  model:           "deepseek-v4-flash"
  provider type:   "openai"
  base_url:        "https://api.deepseek.com"
  request_timeout: default (120s)
  retry:           3 attempts (default), 1s initial backoff (default)
  dialect:         "deepseek"
  max_output_tokens: 8192
  provider mapping (the wire fields this config will send):
    max_output_tokens: 8192 -> max_tokens: 8192
    reasoning.effort: low -> thinking: {"type":"enabled"}
    reasoning.effort: low -> reasoning_effort: low
```

The mapping rows are produced by the same functions the clients call, so the
report cannot promise a request amele will not send - including every rounding
and every value that is **not** sent. `provider.params` keys are listed;
their values never are, because a routing key can be a credential.

The acceptance fixture for all of this is
[testdata/marina/](../testdata/marina/): a style-guide checker that has to
think, call a tool and still answer in schema. One parametrized config, run
against every dialect you have a key for:

```console
$ MARINA_DIALECT=deepseek MARINA_BASE_URL=https://api.deepseek.com \
  MARINA_MODEL=deepseek-v4-flash MARINA_API_KEY=$DEEPSEEK_API_KEY \
  amele run testdata/marina/style-guide.yaml "check violating.md"
```

Its automated form is `cmd/amele/marina_integration_test.go`, behind the
`integration` build tag: it asserts exit 0, one schema-valid JSON document on
stdout, and no truncated turn in the session log.

## See also

- [features.md](features.md) - structured output, permissions, `chat`.
- [contracts/cli.md](contracts/cli.md) - the `explain` report and the `--set`
  allowlist.
- [contracts/jsonl-events.md](contracts/jsonl-events.md) - `reasoning_bytes`.
- [contracts/exit-codes.md](contracts/exit-codes.md) - exit 3 (budgets), 5
  (provider errors), 6 (schema).
- [mcp.md](mcp.md) - borrowing tools from MCP servers, which is where the tool
  loops that make echo-back load-bearing usually come from.
