package llm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// Dialect names one variation of the OpenAI-compatible wire format. The
// protocol is nominally shared, but the providers speaking it disagree on the
// details that matter to an agent: which field carries the output cap, how the
// reasoning knob is spelled, whether reasoning must be echoed back inside a
// tool loop, and which sampling parameters are accepted at all.
//
// CONTRACT: behavior keys on the DIALECT, never on the model name. Model ids
// churn within months (docs/superpowers/specs/2026-08-24-provider-dialects-research.md
// §"Load-bearing quirks"), so a model-name-keyed table would rot; model-level
// quirks are detected from provider error signatures instead.
//
// The zero value is not a dialect: use ParseDialect, which maps the empty
// config value to DialectOpenAI.
type Dialect string

// The dialects amele knows. The string values are the config file's
// vocabulary (provider.dialect) and are enumerated in the published JSON
// Schema, so they are public API.
const (
	// DialectOpenAI is the OpenAI chat/completions baseline and the meaning
	// of an omitted provider.dialect: every config written before dialects
	// existed keeps its behavior.
	DialectOpenAI Dialect = "openai"
	// DialectDeepSeek is DeepSeek's native API (api.deepseek.com).
	DialectDeepSeek Dialect = "deepseek"
	// DialectGLM is Z.ai's GLM API.
	DialectGLM Dialect = "glm"
	// DialectKimi is Moonshot's Kimi API.
	DialectKimi Dialect = "kimi"
	// DialectGroq is Groq's OpenAI-compatible endpoint.
	DialectGroq Dialect = "groq"
	// DialectOpenRouter is the OpenRouter gateway, which normalizes the
	// reasoning knob across the models it fronts.
	DialectOpenRouter Dialect = "openrouter"
)

// dialects lists every accepted value in the order ParseDialect names them,
// with the default first. One list backs both the lookup and the error
// message, so a dialect can never be accepted without being advertised.
var dialects = []Dialect{
	DialectOpenAI,
	DialectDeepSeek,
	DialectGLM,
	DialectKimi,
	DialectGroq,
	DialectOpenRouter,
}

// ParseDialect maps a config value to a Dialect. The empty string yields
// DialectOpenAI. Any other value must match a known dialect EXACTLY: matching
// is neither case-folded nor trimmed, because a value silently "corrected"
// from "OpenAI " would pick a wire mapping the author never named, and the
// consequence (a dropped reasoning field, a rejected cap) would only surface
// as a provider error much later. The returned error names the offending
// value and every accepted spelling, and the returned Dialect is the zero
// value on failure.
func ParseDialect(s string) (Dialect, error) {
	if s == "" {
		return DialectOpenAI, nil
	}
	for _, d := range dialects {
		if string(d) == s {
			return d, nil
		}
	}
	return "", fmt.Errorf("unknown dialect %q (valid: %s; omit for openai)", s, dialectValues())
}

// dialectValues renders the accepted spellings as one comma-separated list for
// ParseDialect's error, derived from the same slice the lookup walks.
func dialectValues() string {
	names := make([]string, len(dialects))
	for i, d := range dialects {
		names[i] = string(d)
	}
	return strings.Join(names, ", ")
}

// effortNone is the one effort value that means "turn thinking OFF" rather
// than "think this deeply". Dialects that carry a thinking switch map it to
// that switch instead of to a level.
const effortNone = "none"

// The thinking-object fragments the CN-native dialects (deepseek, glm) take.
// They are string constants rather than package-level json.RawMessage vars
// because a []byte var is mutable shared state (docs/engineering.md §5.1) and
// these bytes are a contract with the provider.
const (
	thinkingEnabledJSON  = `{"type":"enabled"}`
	thinkingDisabledJSON = `{"type":"disabled"}`
)

// CapField returns the request-body key that carries the output-token cap for
// this dialect. The zero-value Dialect answers like DialectOpenAI, so a client
// that was never told a dialect keeps the pre-dialect behavior.
//
// CONTRACT: the split is not cosmetic. The reasoning-era OpenAI family answers
// max_tokens with a 400, and the CN natives plus OpenRouter only know
// max_tokens (research §matrix "Output cap field"). Sending the wrong name is
// either a hard failure or - worse - a silently uncapped run.
//
// It is exported so `amele explain` names the field the request will actually
// carry instead of maintaining a second copy of this table.
func CapField(d Dialect) string {
	switch d {
	case DialectDeepSeek, DialectGLM, DialectOpenRouter:
		return "max_tokens"
	case DialectOpenAI, DialectGroq, DialectKimi:
		return "max_completion_tokens"
	default:
		// Unreachable through ParseDialect; the zero value lands here and must
		// behave like openai.
		return "max_completion_tokens"
	}
}

// The two response_format variants of the OpenAI-compatible wire.
// json_schema constrains decoding to the caller's schema; json_object only
// promises "some JSON object" and leaves the schema to the validate+retry
// layer above.
const (
	responseFormatJSONSchema = "json_schema"
	responseFormatJSONObject = "json_object"
)

// ResponseFormatType returns the response_format variant to send for this
// dialect when the config asked for structured output. The zero-value Dialect
// answers like DialectOpenAI.
//
// CONTRACT: json_object is not a downgrade decided at runtime - it is the only
// JSON mode deepseek and glm HAVE on this wire (research §matrix
// "response_format"), so sending json_schema there bought a guaranteed 400 and
// a schema-less repeat on every single Chat call. The caller must therefore
// treat a json_object response as "native schema enforcement did not happen"
// (Response.SchemaEnforcementDropped): the schema never reached the provider.
//
// Every other dialect keeps json_schema plus the 400-probe fallback, which is
// the right behavior for the ones whose support is documented (openai,
// openrouter) and for the ones whose support is ambiguous (kimi, groq -
// docs/providers.md §"Structured output").
func ResponseFormatType(d Dialect) string {
	switch d {
	case DialectDeepSeek, DialectGLM:
		return responseFormatJSONObject
	case DialectOpenAI, DialectGroq, DialectKimi, DialectOpenRouter:
		return responseFormatJSONSchema
	default:
		// Unreachable through ParseDialect; the zero value lands here and must
		// behave like openai.
		return responseFormatJSONSchema
	}
}

// The message keys that can carry a reasoning payload on the OpenAI wire.
// The first two are the response field a client reads AND the request field it
// writes back, so they double as the json tags on oaMessage - keep the two in
// step. fieldReasoning is RESPONSE-ONLY (see capturesPlainReasoning).
const (
	fieldReasoningContent = "reasoning_content"
	fieldReasoningDetails = "reasoning_details"
	fieldReasoning        = "reasoning"
)

// capturesPlainReasoning reports whether this dialect may also read a bare
// `reasoning` field off a response when the dialect's own carrier is absent.
//
// Only groq does. Groq's documentation puts the model's reasoning in
// `message.reasoning` - a claim this repository has NOT verified against a live
// endpoint, which is exactly why the field is read as a FALLBACK behind
// reasoning_content rather than replacing it: if the spelling is wrong, nothing
// changes, and if it is right, the run stops under-reporting its own cost.
//
// CONTRACT: capture only. The payload is observable (reasoning_bytes in the
// session log) but never echoed, because no source establishes a request-side
// spelling for this field and this dialect's unknown-field policy is "assume
// rejected" (docs/providers.md §"The dialect table"). Sending an unknown key
// back to buy an unproven echo contract is the wrong side of that trade.
//
// OpenRouter also returns a plaintext `reasoning` beside its typed array, and
// is deliberately NOT here: there the field is documented display sugar next to
// the signed carrier that does round-trip.
func capturesPlainReasoning(d Dialect) bool {
	return d == DialectGroq
}

// echoesReasoningFrom reports whether a payload captured from capturedFrom may
// travel back to this dialect. An empty capturedFrom means the message did not
// come from a wire capture (a caller-built message, a fake provider), which is
// answered like the dialect's own key.
//
// CONTRACT: store-and-echo stays SYMMETRIC - the echo only ever uses the key
// the payload was captured from. Echoing bytes under a different key is how a
// signed or hash-checked carrier turns into a 400 that names a field the config
// never mentions.
func echoesReasoningFrom(d Dialect, capturedFrom string) bool {
	return capturedFrom == "" || capturedFrom == reasoningField(d)
}

// reasoningField returns the message key that carries this dialect's reasoning
// payload. The zero-value Dialect answers like DialectOpenAI.
//
// CONTRACT: one function answers for BOTH directions - the field captured off
// a response and the field echoed back on the next request. A round-trip that
// captured one spelling and echoed another would look correct in isolation and
// fail only against a live provider, so the two ends cannot be decided apart.
//
// The split is between OpenRouter and everyone else. OpenRouter returns a
// typed `reasoning_details` array (text/summary/encrypted blocks, each with an
// optional signature) beside a plaintext `reasoning` summary; only the array
// round-trips - it is what carries the signatures the upstream provider checks
// (research §OpenRouter). The rest of the wire, CN natives included, uses the
// single `reasoning_content` field (research §"Load-bearing quirks" #6).
func reasoningField(d Dialect) string {
	switch d {
	case DialectOpenRouter:
		return fieldReasoningDetails
	case DialectOpenAI, DialectGroq, DialectDeepSeek, DialectGLM, DialectKimi:
		return fieldReasoningContent
	default:
		// Unreachable through ParseDialect; the zero value lands here and must
		// behave like openai.
		return fieldReasoningContent
	}
}

// reasoningWireFields lists the body-root keys MapReasoning can emit for this
// dialect. It is the one part of the owned-field answer that is not visible in
// the request struct, because the KEY itself is dialect data.
//
// CONTRACT: it must cover every key MapReasoning writes for this dialect -
// TestOwnedWireFieldsCoverEveryMappedKey walks the whole vocabulary against the
// mapper and fails if a key is missing, so the two cannot drift.
func reasoningWireFields(d Dialect) []string {
	switch d {
	case DialectDeepSeek, DialectGLM:
		return []string{"thinking", "reasoning_effort"}
	case DialectOpenRouter:
		return []string{"reasoning"}
	case DialectOpenAI, DialectGroq, DialectKimi:
		return []string{"reasoning_effort"}
	default:
		// Unreachable through ParseDialect; the zero value answers like openai.
		return []string{"reasoning_effort"}
	}
}

// OwnedWireFields returns the request-body keys the OpenAI-compatible client
// WILL write for this dialect: the struct-encoded fields of oaRequest plus the
// cap spelling and the reasoning keys this dialect maps. The zero-value Dialect
// answers like DialectOpenAI.
//
// CONTRACT: it is the authority for provider.params collision checking
// (config.validateParams), and it is deliberately DIALECT-SCOPED rather than a
// union across every wire. A union blocked keys nothing was going to send - it
// made `params: {thinking: ...}` a config error on kimi, whose mapper emits no
// thinking object, and so put the K2.x thinking controls out of reach of every
// config. Ownership means "amele writes this key here"; a key this target never
// writes cannot be clobbered by it, and switching the dialect re-answers the
// question at validate time (exit 2), visibly.
//
// Keys no target writes but that amele still reserves (stream, tool_choice)
// are the caller's business: they are not owned, they are incompatible with
// amele's transport and loop, and config names them separately.
func OwnedWireFields(d Dialect) []string {
	owned := []string{"model", "messages", "tools", "response_format", "temperature", "top_p", CapField(d)}
	return append(owned, reasoningWireFields(d)...)
}

// ReasoningMapping is what one dialect makes of a ReasoningSpec: the request
// body fragments to merge at the body root, and the human-readable lines that
// describe the mapping.
//
// CONTRACT: this is the single source of truth for the reasoning knob. The
// wire encoder merges Fields and `amele explain` prints Notes, so what the
// report promises and what the request carries cannot drift apart.
type ReasoningMapping struct {
	// Fields maps body-root keys to pre-serialized JSON values. Nil means the
	// dialect sends no reasoning field at all.
	Fields map[string]json.RawMessage
	// Notes describes the mapping in the config's vocabulary, one line per
	// decision, including every rounding and every value that is NOT sent.
	// Order is deterministic (emission order), never map order.
	Notes []string
}

// field records one body-root fragment, allocating the map on first use so an
// empty mapping stays nil.
func (m *ReasoningMapping) field(key string, value json.RawMessage) {
	if m.Fields == nil {
		m.Fields = make(map[string]json.RawMessage, 2)
	}
	m.Fields[key] = value
}

// note appends one human-readable mapping line.
func (m *ReasoningMapping) note(format string, args ...any) {
	m.Notes = append(m.Notes, fmt.Sprintf(format, args...))
}

// MapReasoning translates the neutral reasoning knob into one dialect's wire
// fields. It is a pure function: the same spec always yields the same fields
// and the same notes, which is what lets `explain` promise before a run
// exactly what the request will carry.
//
// The effort vocabulary (none/low/medium/high/xhigh/max) is the UNION of what
// the providers accept; a dialect with a smaller vocabulary rounds UP to the
// nearest level it has and says so in Notes. Rounding up rather than down is
// deliberate: the config asked for at least this much thinking, and quietly
// buying less is the failure mode a user cannot see in the output.
//
// An empty Effort with no BudgetTokens yields an empty mapping - "the config
// said nothing, so the provider default stands". BudgetTokens is mapped only
// by the openrouter dialect; on the rest of the openai wire it is a validation
// error (config.validateReasoning), never a silent drop here.
//
// Values outside the vocabulary are passed through or rounded by the same
// rules rather than rejected: validation owns the vocabulary, and a client
// that second-guessed it would just fail later with less context.
func MapReasoning(d Dialect, spec ReasoningSpec) ReasoningMapping {
	switch d {
	case DialectDeepSeek, DialectGLM:
		return mapThinkingObjectDialect(d, spec)
	case DialectKimi:
		return mapKimi(spec)
	case DialectOpenRouter:
		return mapOpenRouter(spec)
	case DialectOpenAI, DialectGroq:
		return mapEffortPassthrough(spec)
	default:
		// The zero value behaves like openai.
		return mapEffortPassthrough(spec)
	}
}

// mapEffortPassthrough is the OpenAI baseline, shared by groq: a top-level
// reasoning_effort carrying the value verbatim. Both accept the whole
// vocabulary, "none" included, so nothing is rounded.
func mapEffortPassthrough(spec ReasoningSpec) ReasoningMapping {
	var m ReasoningMapping
	if spec.Effort == "" {
		return m
	}
	m.field("reasoning_effort", jsonString(spec.Effort))
	m.note("reasoning.effort: %s -> reasoning_effort: %s", spec.Effort, spec.Effort)
	return m
}

// mapThinkingObjectDialect covers deepseek and glm, which share a shape: a
// thinking object that switches thinking on or off, plus the level.
//
// CONTRACT: reasoning_effort goes at the BODY ROOT, not inside the thinking
// object. DeepSeek documents both placements (API reference: inside thinking;
// thinking-mode guide: top-level - research §matrix "Reasoning knob"), so one
// had to be pinned; the top-level spelling is the one GLM and Kimi also use,
// which keeps a single shape across the CN trio. An integration smoke test
// guards it, and a server-side change would surface there rather than as a
// silently ignored knob in production.
func mapThinkingObjectDialect(d Dialect, spec ReasoningSpec) ReasoningMapping {
	var m ReasoningMapping
	if spec.Effort == "" {
		return m
	}
	if spec.Effort == effortNone {
		// Thinking off: a level would be contradictory, and these APIs default
		// thinking ON, so the switch is the whole instruction.
		m.field("thinking", json.RawMessage(thinkingDisabledJSON))
		m.note(`reasoning.effort: none -> thinking: %s (%s sends no reasoning_effort with thinking off)`, thinkingDisabledJSON, d)
		return m
	}
	// The switch is sent explicitly even though these providers think by
	// default: a config that named a depth should not depend on a default
	// staying put.
	m.field("thinking", json.RawMessage(thinkingEnabledJSON))
	m.note(`reasoning.effort: %s -> thinking: %s`, spec.Effort, thinkingEnabledJSON)
	m.roundedEffort(d, spec.Effort)
	return m
}

// mapKimi maps the level for the K-series: a top-level reasoning_effort and
// never a thinking object. K3 has no thinking parameter at all (research
// §matrix "Reasoning knob"), so amele emitting one would be a guess at the
// model behind the name. Because nothing here writes that key, it is NOT in
// this dialect's OwnedWireFields: a K2.x user reaches the older
// thinking: {type, keep} controls through provider.params.
func mapKimi(spec ReasoningSpec) ReasoningMapping {
	var m ReasoningMapping
	if spec.Effort == "" {
		return m
	}
	if spec.Effort == effortNone {
		// Unreachable through validate, which rejects kimi + none (exit 2).
		// Reported rather than mapped: the K-series thinks unconditionally, so
		// there is no field that would honor the request.
		m.note("reasoning.effort: none not sent: kimi models cannot disable thinking")
		return m
	}
	m.roundedEffort(DialectKimi, spec.Effort)
	return m
}

// mapOpenRouter maps onto the gateway's unified reasoning object. effort and
// max_tokens are mutually exclusive there (research §OpenRouter), so a spec
// carrying both sends the budget - the more specific instruction - and the
// dropped effort is reported instead of being silently ignored.
func mapOpenRouter(spec ReasoningSpec) ReasoningMapping {
	var m ReasoningMapping
	if spec.BudgetTokens > 0 {
		budget := fmt.Sprintf(`{"max_tokens":%d}`, spec.BudgetTokens)
		m.field("reasoning", json.RawMessage(budget))
		m.note("reasoning.budget_tokens: %d -> reasoning: %s", spec.BudgetTokens, budget)
		if spec.Effort != "" {
			m.note("reasoning.effort: %s not sent: openrouter takes effort or max_tokens, not both", spec.Effort)
		}
		return m
	}
	if spec.Effort == "" {
		return m
	}
	// OpenRouter accepts the full effort union and maps it per underlying
	// provider itself, so amele passes the value through unrounded.
	object := fmt.Sprintf(`{"effort":%s}`, jsonString(spec.Effort))
	m.field("reasoning", json.RawMessage(object))
	m.note("reasoning.effort: %s -> reasoning: %s", spec.Effort, object)
	return m
}

// roundedEffort emits the level for the dialects whose vocabulary is
// low/high/max (the CN trio) and notes the rounding when one happened.
func (m *ReasoningMapping) roundedEffort(d Dialect, effort string) {
	level := roundEffortLowHighMax(effort)
	m.field("reasoning_effort", jsonString(level))
	if level == effort {
		m.note("reasoning.effort: %s -> reasoning_effort: %s", effort, level)
		return
	}
	m.note("reasoning.effort: %s -> reasoning_effort: %s (%s has no %s)", effort, level, d, effort)
}

// roundEffortLowHighMax rounds the neutral vocabulary onto the low/high/max
// subset DeepSeek, GLM and Kimi share (research §"Load-bearing quirks" #5).
// Unknown values pass through untouched: the vocabulary is validated in
// config, and inventing a level for an unrecognized one would hide a typo the
// provider's own 400 names precisely.
func roundEffortLowHighMax(effort string) string {
	switch effort {
	case "medium":
		return "high"
	case "xhigh":
		return "max"
	default:
		return effort
	}
}

// jsonString renders one string as a JSON string fragment. json.Marshal cannot
// fail for a string (invalid UTF-8 is replaced, not rejected), so the error
// path is a defensive fallback rather than a reachable branch.
func jsonString(s string) json.RawMessage {
	encoded, err := json.Marshal(s)
	if err != nil {
		return json.RawMessage(`""`)
	}
	return encoded
}

// errorSignature pairs a recognizer for one provider failure with the advice
// that names the config knob to turn.
type errorSignature struct {
	// match reports whether this failure is the one the entry describes. It
	// receives the whole typed error, not just the text, so an entry may key on
	// the status code too.
	match func(*statusError) bool
	// advice is appended to the provider error message after an em dash. It is
	// written in the CONFIG's vocabulary (provider.reasoning.effort,
	// provider.dialect), because the reader is holding a YAML file, not a
	// request body.
	advice string
}

// errorSignatures is the ordered table consulted for a non-retryable 400 on the
// OpenAI-compatible wire (the Anthropic client has its own, different table:
// anthropicErrorSignatures). First match wins, so more specific entries come
// first. The slice is read-only after initialization (like dialects above);
// nothing mutates it.
//
// CONTRACT: these are STRING HEURISTICS by necessity and the fixtures in
// openai_errsig_test.go are what pins them. docs/engineering.md §5.3 bans
// deciding control flow by matching error strings; this table is the documented
// exception the design approved (design doc §"Error-signature detection"),
// alongside the older rejectsResponseFormat precedent. It is safe in a way a
// general string match is not, because it changes NOTHING but the human-facing
// text: no retry, no downgrade, no request rewrite. A signature that stops
// matching (a provider reworded its 400) costs a hint, never correctness - which
// is also why detection was chosen over auto-downgrading the offending field.
//
// Every match runs against statusError.snippet, already capped to maxErrorBody
// bytes; a proxy that buries the signature past that cut-off simply gets no
// hint.
var errorSignatures = []errorSignature{
	{
		// gpt-5.6 on chat/completions: function tools plus any reasoning_effort
		// other than "none" is a hard 400, and medium is the DEFAULT - so tools
		// break out of the box on that family (research §"Load-bearing quirks"
		// #1, which also records that the error string is the only detection
		// contract; the quirk is in no official doc).
		match:  func(e *statusError) bool { return strings.Contains(e.snippet, "Function tools with reasoning_effort") },
		advice: "set provider.reasoning.effort: none for this model on chat/completions, or use a different model",
	},
	{
		// The output-cap mistake, checked BEFORE the sampling entry: OpenAI
		// phrases it "'max_tokens' is not supported ... Use
		// 'max_completion_tokens' instead", which shares wording with the
		// sampling family. Requiring BOTH field names keeps it from matching a
		// 400 that merely mentions a cap.
		match: func(e *statusError) bool {
			return strings.Contains(e.snippet, "max_tokens") && strings.Contains(e.snippet, "max_completion_tokens")
		},
		// Both doors are named. The dialect is the usual cause, but the same 400
		// is reachable when the operator wrote a cap key into provider.params
		// themselves - legal on a dialect that does not write that key, so
		// validate cannot refuse it - and "change the dialect" would then be
		// the wrong door, on a config where changing it fixes nothing.
		advice: "this model requires max_completion_tokens; set provider.dialect to a dialect that maps it (openai/groq/kimi), or remove that key from provider.params",
	},
	{
		// Reasoning models on OpenAI accept only the default temperature, and
		// the K-series fixes both temperature and top_p (research §matrix
		// "temperature/top_p"). Two spellings are in the wild: OpenAI's
		// "Unsupported value: 'temperature' does not support ..." and the
		// "'temperature' is not supported ..." form. The quotes are part of the
		// match so the word "temperature" in prose cannot trigger it.
		//
		// Both knobs are matched, in both spellings: the providers that fix one
		// fix the other, and a config may set top_p alone - which named nothing
		// in this table until 2026-08-24, so the operator got the raw 400 for
		// exactly the mistake the table exists to explain.
		match:  func(e *statusError) bool { return matchesFixedSampling(e.snippet) },
		advice: "this model rejects non-default sampling; remove provider.temperature/top_p",
	},
}

// matchesFixedSampling reports whether body is a "this model does not let you
// set that sampling knob" 400, for either knob.
//
// The field name is quoted in the match so the word "temperature" appearing in
// unrelated prose cannot trigger the hint - the same reason the table's other
// entries key on distinctive fragments rather than on a bare field name.
func matchesFixedSampling(body string) bool {
	for _, field := range []string{"'temperature'", "'top_p'"} {
		if strings.Contains(body, field+" does not support") ||
			strings.Contains(body, field+" is not supported") {
			return true
		}
	}
	return false
}

// adviceFor returns the actionable hint for a recognized provider failure, or
// "" when nothing in the given table matches - in which case the error message
// stays exactly what it was before this table existed.
//
// The table is a parameter because each wire family has its own signatures:
// errorSignatures for the OpenAI-compatible clients, anthropicErrorSignatures
// for the Messages API. One matcher serves both so the 400 gate below cannot
// drift between them.
//
// Only a 400 is inspected. A 429 or 5xx is a transient condition the client
// retries; the same body text there says nothing about the request being
// wrong, and advising a config change over a rate limit would be a wrong hint
// at the worst moment.
func adviceFor(signatures []errorSignature, e *statusError) string {
	if e == nil || e.code != http.StatusBadRequest {
		return ""
	}
	for _, sig := range signatures {
		if sig.match(e) {
			return sig.advice
		}
	}
	return ""
}
