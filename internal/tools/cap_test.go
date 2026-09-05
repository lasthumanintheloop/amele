package tools

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestCapText(t *testing.T) {
	tests := []struct {
		name      string
		in        string
		max       int
		wantLen   int // bytes before the marker
		wantTrunc bool
	}{
		{"under", "abc", 10, 3, false},
		{"exact", "abcdefghij", 10, 10, false},
		{"over by one", "abcdefghijk", 10, 10, true},
		{"no cap", strings.Repeat("x", 100), 0, 100, false},
		// "é" is 2 bytes; a cap that lands inside it must back off to 4.
		{"rune boundary", "abcdé", 5, 4, true},
		{"only multibyte", "ééé", 3, 2, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out, trunc := CapText(tc.in, tc.max)
			if trunc != tc.wantTrunc {
				t.Fatalf("truncated = %v, want %v", trunc, tc.wantTrunc)
			}
			body := strings.TrimSuffix(out, TruncationMarker)
			if trunc && body == out {
				t.Fatalf("truncated output lacks marker: %q", out)
			}
			if !trunc && body != tc.in {
				t.Fatalf("untruncated output changed: %q", out)
			}
			if len(body) != tc.wantLen {
				t.Fatalf("kept %d bytes, want %d (%q)", len(body), tc.wantLen, body)
			}
			if !utf8.ValidString(out) {
				t.Fatalf("output is not valid UTF-8: %q", out)
			}
		})
	}
}
