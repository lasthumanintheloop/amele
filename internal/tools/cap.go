package tools

import "unicode/utf8"

// TruncationMarker is appended to any tool result that was cut to a byte cap.
// The model sees it and can narrow its request instead of assuming it read
// everything; every tool family uses this one wording so the model learns one
// signal, not four.
const TruncationMarker = "\n[output truncated by amele]"

// CapText bounds s to at most max bytes of content plus TruncationMarker. The
// cut moves back to the start of a rune so a multi-byte character is never
// split - half a rune is invalid UTF-8 and some providers reject the whole
// request over it. max <= 0 means no cap. The boolean reports whether anything
// was dropped, which is the fact the session log records (Outcome.Truncated);
// it is driven by bytes actually removed, so a result of exactly max bytes is
// returned whole and unmarked.
func CapText(s string, max int) (string, bool) {
	if max <= 0 || len(s) <= max {
		return s, false
	}
	// end can reach 0 only for a string whose first rune is already longer
	// than max, and an empty body plus the marker is the honest answer there.
	return s[:runeStart(s, max)] + TruncationMarker, true
}

// markTruncated appends TruncationMarker to s when dropped is true, cutting a
// trailing partial rune first.
//
// It exists because CapText cannot state this case: a stream limiter never
// stores more than its cap, so its buffer is always at or under max and
// CapText - which returns a result of exactly max whole and unmarked - would
// leave a cut stream unmarked. The dropped flag is the fact here; the rune
// back-off is CapText's rule applied to the same bytes. Every captured stream
// (stdout, and the trimmed stderr of a failed command) goes through this one
// function so the two can never drift apart.
func markTruncated(s string, dropped bool) string {
	if !dropped {
		return s
	}
	// A cap that fell inside a multi-byte character leaves a trailing
	// sequence that decodes as a one-byte RuneError; only that sequence is
	// removed. Bytes further back may be the command's own non-UTF-8 output,
	// and amele reports what the command wrote rather than rewriting it.
	if r, size := utf8.DecodeLastRuneInString(s); r == utf8.RuneError && size == 1 {
		s = s[:runeStart(s, len(s)-1)]
	}
	return s + TruncationMarker
}

// runeStart walks end back to the nearest index that starts a rune, so a cut
// at end never splits a multi-byte character. utf8.RuneStart is a byte-level
// test, so this walks back at most three continuation bytes.
func runeStart(s string, end int) int {
	for end > 0 && !utf8.RuneStart(s[end]) {
		end--
	}
	return end
}
