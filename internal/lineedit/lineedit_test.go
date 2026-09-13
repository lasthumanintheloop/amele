package lineedit

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"strings"
	"sync"
	"testing"
)

// fakeTerm records raw-mode transitions and reports a fixed width. The
// counters are guarded because the abandoned-read test drives MakeRaw from
// another goroutine; entered is signalled once per MakeRaw so that test can
// wait without polling.
type fakeTerm struct {
	width   int
	entered chan struct{}

	mu       sync.Mutex
	raw      bool
	enters   int
	restores int
}

func (f *fakeTerm) MakeRaw() (func() error, error) {
	f.mu.Lock()
	f.enters++
	f.raw = true
	f.mu.Unlock()
	if f.entered != nil {
		f.entered <- struct{}{}
	}
	return func() error {
		f.mu.Lock()
		defer f.mu.Unlock()
		f.restores++
		f.raw = false
		return nil
	}, nil
}

func (f *fakeTerm) Width() int { return f.width }

// snapshot returns the guarded counters.
func (f *fakeTerm) snapshot() (raw bool, enters, restores int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.raw, f.enters, f.restores
}

// editor builds an Editor over the given key bytes.
func editor(input string, width int) (*Editor, *bytes.Buffer, *fakeTerm) {
	var out bytes.Buffer
	term := &fakeTerm{width: width}
	return New(bufio.NewReader(strings.NewReader(input)), &out, term), &out, term
}

func TestReadLineKeys(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{"plain", "hello\r", "hello"},
		{"newline also submits", "hello\n", "hello"},
		{"backspace", "helpo\x7f\x7flo\r", "hello"},
		{"left and insert", "hllo\x1b[D\x1b[D\x1b[De\r", "hello"},
		{"home and end", "llo\x1b[Hhe\x1b[F!\r", "hello!"},
		{"ctrl-a ctrl-e", "llo\x01he\x05!\r", "hello!"},
		{"ctrl-u kills to start", "junk \x15hello\r", "hello"},
		{"ctrl-k kills to end", "hello junk\x1b[D\x1b[D\x1b[D\x1b[D\x1b[D\x0b\r", "hello"},
		{"ctrl-w kills a word", "hello junk\x17\r", "hello "},
		{"delete key", "hxello\x1b[H\x1b[C\x1b[3~\r", "hello"},
		{"utf-8 runes", "günaydın\r", "günaydın"},
		{"tab inserts a space", "a\tb\r", "a b"},
		{"unknown escape is swallowed", "he\x1b[15~llo\r", "hello"},
		{"a bare escape is dropped and the key after it kept", "he\x1bxllo\x1b\r", "hexllo"},
		{"escape then a non-ascii key keeps the rune whole", "caf\x1bé\r", "café"},
		{"a pasted block keeps its newlines", "\x1b[200~line one\r\nline two\x1b[201~\r", "line one\nline two"},
		{"ctrl-c discards the line and re-prompts", "junk\x03hello\r", "hello"},
		{"control bytes are ignored", "he\x1fllo\r", "hello"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			e, _, term := editor(tc.input, 80)
			got, err := e.ReadLine("> ")
			if err != nil || got != tc.want {
				t.Fatalf("ReadLine = %q, %v; want %q", got, err, tc.want)
			}
			if raw, enters, restores := term.snapshot(); enters != 1 || restores != 1 || raw {
				t.Errorf("raw mode: enters %d restores %d raw %v; want one round trip", enters, restores, raw)
			}
		})
	}
}

func TestReadLineEnds(t *testing.T) {
	t.Run("ctrl-d on an empty line is EOF", func(t *testing.T) {
		e, _, _ := editor("\x04", 80)
		if _, err := e.ReadLine("> "); !errors.Is(err, io.EOF) {
			t.Fatalf("err = %v, want io.EOF", err)
		}
	})
	t.Run("ctrl-d on a non-empty line deletes under the cursor", func(t *testing.T) {
		e, _, _ := editor("hxello\x1b[H\x1b[C\x04\r", 80)
		if got, err := e.ReadLine("> "); err != nil || got != "hello" {
			t.Fatalf("ReadLine = %q, %v", got, err)
		}
	})
	t.Run("ctrl-c on an empty line interrupts", func(t *testing.T) {
		e, _, term := editor("\x03", 80)
		if _, err := e.ReadLine("> "); !errors.Is(err, ErrInterrupt) {
			t.Fatalf("err = %v, want ErrInterrupt", err)
		}
		if raw, _, _ := term.snapshot(); raw {
			t.Error("the terminal was left raw")
		}
	})
	t.Run("input ending without a newline delivers the text, then EOF", func(t *testing.T) {
		e, _, _ := editor("tail", 80)
		got, err := e.ReadLine("> ")
		if err != nil || got != "tail" {
			t.Fatalf("first ReadLine = %q, %v", got, err)
		}
		if _, err := e.ReadLine("> "); !errors.Is(err, io.EOF) {
			t.Fatalf("second ReadLine err = %v, want io.EOF", err)
		}
	})
}

// TestHistory: Up recalls earlier entries newest first, Down walks back to
// the line being typed, blank and consecutively repeated entries are not
// recorded (a recalled entry that differs from the last one is, as in
// readline), and ctrl-p/ctrl-n are the same keys. A blank line is still
// delivered as typed: the REPL owns the "whitespace costs nothing" rule.
func TestHistory(t *testing.T) {
	input := "first\r" + "second\r" + "   \r" + "second\r" +
		"typed\x1b[A\x1b[A\x1b[B\x1b[B\r" + // up twice (first), down twice back to "typed"
		"\x1b[A\x1b[A\x1b[A\x1b[A\r" + // past the oldest stays at the oldest: "first" (recorded again)
		"\x10\x10\x0e\r" // ctrl-p twice (first, typed), ctrl-n once (first)
	e, _, _ := editor(input, 80)
	var got []string
	for {
		line, err := e.ReadLine("> ")
		if err != nil {
			break
		}
		got = append(got, line)
	}
	want := []string{"first", "second", "   ", "second", "typed", "first", "first"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("lines = %q, want %q", got, want)
	}
	if strings.Join(e.history, "|") != "first|second|typed|first" {
		t.Errorf("history = %q", e.history)
	}
}

// TestDraw pins the one-row repaint: carriage return, prompt, text, clear to
// end of line, cursor moved back; a newline in the entry is shown as a glyph;
// a line wider than the terminal scrolls instead of wrapping.
func TestDraw(t *testing.T) {
	t.Run("cursor in the middle", func(t *testing.T) {
		e, out, _ := editor("ab\x1b[D\r", 80)
		if _, err := e.ReadLine("> "); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(out.String(), "\r> ab\x1b[K\x1b[1D") {
			t.Errorf("output lacks the repaint with the cursor moved back one cell:\n%q", out.String())
		}
	})
	t.Run("a pasted newline is drawn as a glyph", func(t *testing.T) {
		e, out, _ := editor("\x1b[200~a\nb\x1b[201~\r", 80)
		if _, err := e.ReadLine("> "); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(out.String(), "\r> a↵b\x1b[K") {
			t.Errorf("output does not draw the newline as a glyph:\n%q", out.String())
		}
	})
	t.Run("a long line scrolls to keep the cursor visible", func(t *testing.T) {
		e, out, _ := editor("abcdefghij\r", 8) // room for 5 cells after the prompt
		if _, err := e.ReadLine("> "); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(out.String(), "\r> fghij\x1b[K") || strings.Contains(out.String(), "\r> abcdefghij") {
			t.Errorf("the window did not scroll:\n%q", out.String())
		}
	})
	t.Run("bracketed paste is switched on for the read and off after", func(t *testing.T) {
		e, out, _ := editor("x\r", 80)
		if _, err := e.ReadLine("> "); err != nil {
			t.Fatal(err)
		}
		s := out.String()
		if !strings.HasPrefix(s, enablePaste) || !strings.HasSuffix(s, disablePast) {
			t.Errorf("paste mode bracketing missing:\n%q", s)
		}
	})
}

// TestCloseRestoresAnAbandonedRead: when the REPL abandons a ReadLine (its
// context ended while the read was blocked), Close must still hand the
// terminal back cooked.
func TestCloseRestoresAnAbandonedRead(t *testing.T) {
	pr, pw := io.Pipe()
	defer func() { _ = pw.Close() }()
	var out syncBuffer
	term := &fakeTerm{width: 80, entered: make(chan struct{}, 1)}
	e := New(bufio.NewReader(pr), &out, term)
	go func() { _, _ = e.ReadLine("> ") }()
	<-term.entered // the goroutine is inside ReadLine, blocked on the pipe
	e.Close()
	if raw, _, restores := term.snapshot(); raw || restores != 1 {
		t.Fatalf("Close left the terminal raw (raw %v, restores %d)", raw, restores)
	}
	if !strings.HasSuffix(out.String(), disablePast) {
		t.Errorf("Close did not switch bracketed paste off:\n%q", out.String())
	}
}

// syncBuffer is a bytes.Buffer safe for the two goroutines of the abandoned
// read test.
type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

// TestCloseIsPermanent: a ReadLine that starts after Close - the abandoned
// goroutine of a cancelled read arriving late - must not touch the terminal.
func TestCloseIsPermanent(t *testing.T) {
	e, out, term := editor("late\r", 80)
	e.Close()
	if _, err := e.ReadLine("> "); !errors.Is(err, ErrClosed) {
		t.Fatalf("err = %v, want ErrClosed", err)
	}
	if _, enters, _ := term.snapshot(); enters != 0 || strings.Contains(out.String(), enablePaste) {
		t.Errorf("a closed editor entered raw mode or paste mode: enters %d, out %q", enters, out.String())
	}
}

// TestEscapeThenKey: Escape followed by Enter submits and Escape followed by
// Ctrl-C interrupts; the escape itself is dropped.
func TestEscapeThenKey(t *testing.T) {
	e, _, _ := editor("hi\x1b\r\x1b\x03", 80)
	if got, err := e.ReadLine("> "); err != nil || got != "hi" {
		t.Fatalf("ReadLine = %q, %v", got, err)
	}
	if _, err := e.ReadLine("> "); !errors.Is(err, ErrInterrupt) {
		t.Fatalf("err = %v, want ErrInterrupt", err)
	}
}

// TestPasteIsBounded: a paste past the line cap is read and dropped, not
// retained.
func TestPasteIsBounded(t *testing.T) {
	input := "\x1b[200~" + strings.Repeat("x", maxLineRunes+100) + "\x1b[201~\r"
	e, _, _ := editor(input, 80)
	got, err := e.ReadLine("> ")
	if err != nil || len(got) != maxLineRunes {
		t.Fatalf("len = %d, err %v; want the cap", len(got), err)
	}
}

// TestNoTerminal: with no Terminal the editor still works on the bytes it is
// given, which is what makes it drivable from a test or a pipe.
func TestNoTerminal(t *testing.T) {
	var out bytes.Buffer
	e := New(bufio.NewReader(strings.NewReader("hi\r")), &out, nil)
	if got, err := e.ReadLine("> "); err != nil || got != "hi" {
		t.Fatalf("ReadLine = %q, %v", got, err)
	}
}
