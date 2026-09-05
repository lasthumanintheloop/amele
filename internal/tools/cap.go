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
	end := max
	// utf8.RuneStart is a byte-level test, so this walks back at most three
	// continuation bytes; end can reach 0 only for a string whose first rune
	// is already longer than max, and an empty body plus the marker is the
	// honest answer there.
	for end > 0 && !utf8.RuneStart(s[end]) {
		end--
	}
	return s[:end] + TruncationMarker, true
}
