// Package schema turns a user-supplied JSON Schema into a compiled validator
// for an agent's final answer.
//
// It answers exactly one question: "is this blob of model text an acceptable
// final answer, and if not, what do I tell the model?" That is why it returns
// three things at once - the extracted JSON, a model-facing feedback string,
// and a verdict - instead of a plain error: the caller feeds the feedback
// straight back into the loop as the next user message.
//
// The package is a leaf on purpose. It imports nothing from config, llm, loop
// or session, so the schema contract can be tested without a provider, a
// workspace or a session file; cmd wires it to the loop's FinalValidator hook.
//
// Extraction is deliberately dumb (whole text, then the first fenced block)
// rather than a scan for balanced braces: a deterministic, explainable rule is
// worth more here than a clever one that occasionally guesses the wrong span
// and silently ships the wrong answer to stdout.
package schema

import (
	"bytes"
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/santhosh-tekuri/jsonschema/v6"
)

// MaxFeedbackErrors caps how many validation errors a single feedback message
// lists. CONTRACT: feedback is re-sent to the model on every retry, so its
// cost is bounded rather than proportional to how wrong the output was.
const MaxFeedbackErrors = 3

// MaxFeedbackBytes caps the total size of a feedback message. Longer messages
// are truncated on a UTF-8 boundary with a marker.
const MaxFeedbackBytes = 1024

// resourceURL is the opaque identifier the schema is compiled under. A
// relative name like "schema.json" would be resolved against the process
// working directory and leak that path into compile and validation error
// messages, which then differ per machine and per invocation.
const resourceURL = "mem:///amele/output-schema"

// truncationMarker is appended when feedback is cut at MaxFeedbackBytes.
const truncationMarker = "… (truncated)"

// Validator is a compiled output schema. It is immutable after Compile and
// safe for concurrent use by multiple goroutines.
type Validator struct {
	// raw is a private copy of the schema source, handed to providers that
	// support native structured output (response_format: json_schema).
	raw json.RawMessage
	sch *jsonschema.Schema
}

// Compile parses and compiles schemaJSON as a JSON Schema (draft 2020-12 by
// default; an explicit $schema selects another draft). It returns an error if
// schemaJSON is not a single JSON document or is not a valid schema - the
// caller surfaces that at config-load time, before any provider is contacted.
//
// Compilation is hermetic: internal "#/..." refs resolve, but a $ref to any
// external document (http, file, or relative) is rejected rather than fetched.
//
// The returned Validator keeps its own copy of schemaJSON; later mutations of
// the caller's slice do not affect it.
func Compile(schemaJSON []byte) (*Validator, error) {
	doc, err := jsonschema.UnmarshalJSON(bytes.NewReader(schemaJSON))
	if err != nil {
		return nil, fmt.Errorf("parsing output schema: %w", err)
	}

	c := jsonschema.NewCompiler()
	// SECURITY: compilation is hermetic. The library's default loader resolves
	// an unknown $ref by reading a file off disk, so a schema could turn config
	// loading into an arbitrary local-file read. Everything a schema needs must
	// be inline ($defs and internal "#/..." refs still work); the bundled
	// metaschemas are resolved before the loader is ever consulted.
	c.UseLoader(deniedLoader{})
	if err := c.AddResource(resourceURL, doc); err != nil {
		return nil, fmt.Errorf("loading output schema: %w", err)
	}
	sch, err := c.Compile(resourceURL)
	if err != nil {
		return nil, fmt.Errorf("compiling output schema: %w", err)
	}

	return &Validator{raw: bytes.Clone(schemaJSON), sch: sch}, nil
}

// deniedLoader refuses every external schema reference. It exists so Compile
// performs no I/O at all: same input, same result, on any machine.
type deniedLoader struct{}

// Load always fails, naming the fix rather than the mechanism.
func (deniedLoader) Load(url string) (any, error) {
	return nil, fmt.Errorf("external schema reference %q is not supported; inline it with $defs", url)
}

// JSON returns a copy of the schema source exactly as it was given to Compile.
// Providers with native structured-output support send it verbatim; callers
// may modify the returned slice without affecting the Validator.
func (v *Validator) JSON() json.RawMessage {
	return bytes.Clone(v.raw)
}

// Validate extracts a JSON value from text and checks it against the schema.
//
// Extraction tries the whole text first, then the first fenced code block
// (```json, or a bare ``` fence) whose contents parse as JSON. Nothing else:
// no brace balancing, no partial repair.
//
// On success it reports ok=true and canonical - the extracted JSON exactly as
// the model wrote it, minus surrounding whitespace. It is never re-marshaled,
// so key order, spacing and number precision survive to stdout untouched.
//
// On failure it reports ok=false, an empty canonical and feedback: a short,
// English, model-facing message naming at most MaxFeedbackErrors problems.
// Feedback is deterministic - identical text always yields identical bytes.
func (v *Validator) Validate(text string) (canonical string, feedback string, ok bool) {
	candidate, doc, err := extract(text)
	if err != nil {
		return "", noJSONFeedback(err), false
	}

	if err := v.sch.Validate(doc); err != nil {
		var verr *jsonschema.ValidationError
		if !errors.As(err, &verr) {
			// Defensive: *ValidationError is Validate's only documented
			// failure mode, but an unexpected one must not crash the loop.
			return "", "Your reply was rejected: the JSON does not match the required schema.", false
		}
		return "", schemaFeedback(verr), false
	}

	return candidate, "", true
}

// extract returns the JSON span found in text along with its parsed form.
// The parsed form comes from jsonschema.UnmarshalJSON so numbers keep full
// precision (json.Number), which the schema's numeric keywords rely on.
func extract(text string) (candidate string, doc any, err error) {
	trimmed := strings.TrimSpace(text)

	// wholeErr is the error worth reporting when nothing parses: a reply with
	// no fence at all is the common failure, and its parse error points at the
	// text the model actually produced.
	doc, wholeErr := parseJSON(trimmed)
	if wholeErr == nil {
		return trimmed, doc, nil
	}

	for _, block := range fencedBlocks(text) {
		if doc, err := parseJSON(block); err == nil {
			return block, doc, nil
		}
	}

	return "", nil, wholeErr
}

// parseJSON accepts only a complete, single JSON document; trailing content
// after the first value is an error (jsonschema.UnmarshalJSON enforces this).
func parseJSON(s string) (any, error) {
	if s == "" {
		return nil, fmt.Errorf("empty output")
	}
	return jsonschema.UnmarshalJSON(strings.NewReader(s))
}

// fencedBlocks returns the trimmed contents of every closed ``` block whose
// info string is empty or "json" (case-insensitive), in source order - so when
// a bare fence and a ```json fence both hold valid JSON, the one that appears
// first in the text wins. An unterminated fence yields nothing: a truncated
// block is not a JSON document, and guessing where it ends would make the
// result depend on the truncation.
//
// Every ``` line toggles the scanner, including one that opens a block this
// function will discard. Skipping the toggle for a rejected info string
// desynchronizes fence parity: the rejected block's closing ``` then reads as
// an opening bare fence, which both hides later ```json blocks and turns the
// prose between two blocks into a "block" of its own.
func fencedBlocks(text string) []string {
	var blocks []string
	var current []string
	open, accepted := false, false

	for _, line := range strings.Split(text, "\n") {
		marker := strings.TrimSpace(strings.TrimSuffix(line, "\r"))
		if !strings.HasPrefix(marker, "```") {
			if open && accepted {
				current = append(current, line)
			}
			continue
		}
		if open {
			if accepted {
				blocks = append(blocks, strings.TrimSpace(strings.Join(current, "\n")))
			}
			current, open, accepted = nil, false, false
			continue
		}
		info := strings.ToLower(strings.TrimSpace(marker[3:]))
		open, accepted = true, info == "" || info == "json"
	}

	return blocks
}

// noJSONFeedback tells the model that nothing parseable was found. It names
// both accepted shapes so the retry has a concrete target.
func noJSONFeedback(err error) string {
	return truncate(fmt.Sprintf(
		"Your reply was rejected: it must contain a single JSON value matching the required schema, "+
			"but no JSON could be parsed from it (parse error: %v). "+
			"Reply with the JSON value alone, or wrap it in one ```json fenced block. Do not add prose around it.",
		err))
}

// schemaFeedback renders the first MaxFeedbackErrors validation failures.
func schemaFeedback(verr *jsonschema.ValidationError) string {
	var b strings.Builder
	b.WriteString("Your reply was rejected: the JSON must match the required schema.")

	leaves := leafErrors(verr)
	if len(leaves) == 0 {
		b.WriteString(" It does not validate against the schema.")
	} else {
		b.WriteString(" Validation errors:")
		for i, leaf := range leaves {
			if i == MaxFeedbackErrors {
				fmt.Fprintf(&b, "\n  (and %d more)", len(leaves)-MaxFeedbackErrors)
				break
			}
			fmt.Fprintf(&b, "\n  - at %s: %s", leaf.location, leaf.message)
		}
	}

	b.WriteString("\nReply with the corrected JSON value alone, matching the schema exactly.")
	return truncate(b.String())
}

// leafError is one concrete complaint: the innermost failure, with the
// instance location that caused it.
type leafError struct {
	location string
	message  string
}

// leafErrors flattens the detailed output to its innermost failures and sorts
// them. Sorting is not cosmetic: the library walks instance objects with a Go
// map range, so unsorted errors arrive in a different order on every call and
// feedback would stop being reproducible.
func leafErrors(verr *jsonschema.ValidationError) []leafError {
	units := collectLeaves(*verr.DetailedOutput(), nil)

	leaves := make([]leafError, 0, len(units))
	for _, u := range units {
		location := u.InstanceLocation
		if location == "" {
			location = "(root)"
		}
		leaves = append(leaves, leafError{location: location, message: u.Error.String()})
	}

	slices.SortFunc(leaves, func(a, b leafError) int {
		return cmp.Or(cmp.Compare(a.location, b.location), cmp.Compare(a.message, b.message))
	})
	return leaves
}

// collectLeaves walks the detailed output tree and keeps only units with no
// children. Intermediate units restate their children ("validation failed"),
// which would waste the small error budget on nothing actionable.
func collectLeaves(u jsonschema.OutputUnit, out []jsonschema.OutputUnit) []jsonschema.OutputUnit {
	if len(u.Errors) == 0 {
		if u.Error != nil {
			out = append(out, u)
		}
		return out
	}
	for _, child := range u.Errors {
		out = collectLeaves(child, out)
	}
	return out
}

// truncate clips s to MaxFeedbackBytes on a UTF-8 boundary.
func truncate(s string) string {
	if len(s) <= MaxFeedbackBytes {
		return s
	}
	cut := MaxFeedbackBytes - len(truncationMarker)
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + truncationMarker
}
