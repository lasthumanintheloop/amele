package llm

// This file holds the wire-neutral HTTP/JSON helpers shared by the three wire
// families (OpenAI-compatible, Anthropic native, Gemini native).

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"slices"
	"strings"
	"time"
)

// decodeResponseBody decodes a provider's success body under maxResponseBody.
// A body that exceeds the cap is cut mid-JSON and surfaces as a decode error,
// which is exactly how the caller should treat an endpoint it cannot parse.
func decodeResponseBody(body io.Reader, into any) error {
	return json.NewDecoder(io.LimitReader(body, maxResponseBody)).Decode(into)
}

// backoffDelay returns how long to wait before attempt (>= 2): an exponential
// ladder rooted at initial (0 means defaultInitialBackoff), stretched - never
// shrunk - to the provider's Retry-After wish when one was sent. No single wait
// exceeds maxRetryAfter, whether it came from the ladder or from the header.
//
// Shared by all three clients on purpose: a retry rhythm that differed between
// the OpenAI-compatible, the native Anthropic and the native Gemini wire would
// be a trap for a config that switches wires, and provider.retry configures
// exactly one behavior.
func backoffDelay(initial time.Duration, attempt int, retryAfter time.Duration) time.Duration {
	if initial <= 0 {
		initial = defaultInitialBackoff
	}
	// The ladder starts at initial on the first retry (attempt 2). The shift is
	// clamped rather than trusting the caller: a negative shift count is a
	// runtime panic in Go, and library code must not panic
	// (docs/engineering.md §5.3).
	shift := max(attempt-2, 0)
	// CONTRACT: the ladder is capped by the same ceiling as a Retry-After wish.
	// A wait is a wait: whether the number came from a provider header or from
	// doubling initial_backoff, one that runs into the tens of minutes reads as
	// a hung run, and the config's accepted maximum (60s) reaches 4h16m by the
	// last rung of a max_attempts: 10 policy. Capping here bounds the whole
	// ladder's worst case to (attempts-1) x 60s.
	delay := min(initial<<shift, maxRetryAfter)
	if retryAfter > delay {
		delay = min(retryAfter, maxRetryAfter)
	}
	return delay
}

// statusError carries the HTTP status and body snippet of a non-200 reply so
// Chat can route on them programmatically. docs/engineering.md §5.3 bans deciding
// control flow by matching error strings; this typed error is how the
// response_format fallback inspects the failure without re-parsing text.
type statusError struct {
	code    int
	snippet string
}

func (e *statusError) Error() string {
	return fmt.Sprintf("status %d: %s", e.code, e.snippet)
}

// statusFailure turns a non-200 reply into the typed provider error, reporting
// whether the failure is worth retrying and the provider's Retry-After wish.
// The response body is read here (bounded to maxErrorBody) and not by the
// caller, so the two cannot disagree about who consumed it.
//
// signatures is the caller's error-signature table: all three clients share the
// retry/Retry-After/typed-error handling and differ only in which 400 bodies
// they recognize.
func statusFailure(httpResp *http.Response, signatures []errorSignature) (retryable bool, retryAfter time.Duration, err error) {
	snippet, _ := io.ReadAll(io.LimitReader(httpResp.Body, maxErrorBody))
	retryable = httpResp.StatusCode == http.StatusTooManyRequests || httpResp.StatusCode >= 500
	// Double %w: the message is unchanged ("provider error: status N: …")
	// while callers keep both errors.Is(ErrProvider) and errors.As on the
	// typed status.
	statusErr := &statusError{code: httpResp.StatusCode, snippet: strings.TrimSpace(string(snippet))}
	err = fmt.Errorf("%w: %w", ErrProvider, statusErr)
	// A recognized 400 keeps its message and gains a hint at the end
	// ("… — set provider.reasoning.effort: none …"). The advice is appended
	// HERE rather than inside statusError.Error() so that the anthropic client,
	// which shares the type, keeps its own (different) signatures; and so the
	// typed error stays a pure carrier of what the wire said. Nothing else
	// changes: retryable, Retry-After and the errors.Is/As behavior are the
	// same with or without a match.
	if advice := adviceFor(signatures, statusErr); advice != "" {
		err = fmt.Errorf("%w — %s", err, advice)
	}
	return retryable, parseRetryAfter(httpResp.Header.Get("Retry-After")), err
}

// shouldFallback reports whether a failed attempt must be repeated once with
// the offending field stripped. rejected recognizes the failure that warrants
// it - response_format on the OpenAI wire, output_config on the Anthropic one -
// and fallback is nil when there is nothing to strip or when the single
// permitted fallback has already been spent, which is what bounds it to one
// extra round-trip per Chat call.
func shouldFallback(err error, fallback []byte, rejected func(*statusError) bool) bool {
	if err == nil || fallback == nil {
		return false
	}
	var se *statusError
	return errors.As(err, &se) && rejected(se)
}

// encodeBody renders one request body: the struct-encoded fields first, in
// declaration order, then the merged fragments. wire is any request struct
// (oaRequest, anRequest, gemRequest) - all three wire families need the same
// two-stage encoding, because the KEY of a merged fragment is data rather than
// a Go field.
func encodeBody(wire any, fields map[string]json.RawMessage) ([]byte, error) {
	body, err := json.Marshal(wire)
	if err != nil {
		return nil, fmt.Errorf("encoding request: %w", err)
	}
	return mergeBodyFields(body, fields)
}

// mergeBodyFields appends pre-serialized fragments to the root of an already
// encoded JSON object.
//
// Hand-editing JSON is normally a smell; it is the right tool here because the
// KEY of a dialect's reasoning field and of every provider.params entry is
// data, not a Go type, and re-encoding the whole body through a map would give
// up the stable key order the goldens (and reviewable request diffs) depend
// on. The keys are merged in sorted order for that same reason - Go map
// iteration is randomized - and every value is passed through json.Compact, so
// an unparseable fragment fails the request here instead of reaching the
// provider as a malformed body it answers with an opaque 400.
func mergeBodyFields(body []byte, fields map[string]json.RawMessage) ([]byte, error) {
	if len(fields) == 0 {
		return body, nil
	}
	if len(body) < 2 || body[0] != '{' || body[len(body)-1] != '}' {
		return nil, fmt.Errorf("merging request fields: body is not a JSON object")
	}
	out := make([]byte, 0, len(body)+64)
	out = append(out, body[:len(body)-1]...)
	for _, key := range slices.Sorted(maps.Keys(fields)) {
		var compact bytes.Buffer
		if err := json.Compact(&compact, fields[key]); err != nil {
			return nil, fmt.Errorf("merging request field %q: %w", key, err)
		}
		// An object that is still empty ends with '{' and takes no separator.
		if out[len(out)-1] != '{' {
			out = append(out, ',')
		}
		encodedKey, err := json.Marshal(key)
		if err != nil {
			return nil, fmt.Errorf("merging request field %q: %w", key, err)
		}
		out = append(out, encodedKey...)
		out = append(out, ':')
		out = append(out, compact.Bytes()...)
	}
	return append(out, '}'), nil
}

// compactJSONObject normalizes a tool_use input into the compact JSON string
// the neutral ToolCall carries. Absent or null input becomes "{}" so tools
// always receive a parseable object.
func compactJSONObject(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return "{}"
	}
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		// Malformed provider JSON is deferred to the tool layer, mirroring
		// the OpenAI client: parsing tool arguments is the tool's job so the
		// model can recover from its own bad output.
		return string(raw)
	}
	return buf.String()
}

// extraFields copies the caller's raw provider.params into a fresh map, so the
// fragments merged into one request body can never leak into another.
func extraFields(extra map[string]json.RawMessage) map[string]json.RawMessage {
	if len(extra) == 0 {
		return nil
	}
	fields := make(map[string]json.RawMessage, len(extra))
	for key, value := range extra {
		fields[key] = value
	}
	return fields
}

// isJSONArray reports whether a carrier holds a JSON array, i.e. whether it can
// be a content array of this wire at all. nil, a JSON null and a payload from
// another provider's wire format all answer false.
func isJSONArray(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) > 0 && trimmed[0] == '['
}
