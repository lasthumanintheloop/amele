// Package lineedit is the chat prompt's line editor: a readline-shaped input
// for a terminal, with in-session history (Up/Down), cursor movement and
// editing keys, and bracketed paste, so a pasted block arrives as one entry
// with its line breaks intact (issue #11). It is deliberately not a TUI: it
// draws one prompt line and nothing else, keeps no screen state beyond that
// line, and hands every other byte of the terminal to whoever owns it.
//
// The editor reads raw bytes and interprets a small, fixed key vocabulary
// (see the constants below); it never owns the terminal outside a ReadLine
// call - raw mode is entered when a line is asked for and left when it is
// delivered, so an approval question asked between two prompts reads a
// cooked line from the same stream as before. What makes that safe is the
// shared reader: the editor reads through the caller's *bufio.Reader, the same
// one the approval prompt reads, so bytes typed ahead of a prompt cannot be
// buffered by one consumer and lost to the other.
//
// History lives in memory for the session only. Persisting it would write
// whatever the operator typed - including the secrets people paste into a
// chat - to disk under no redaction, so it is a deliberate non-feature.
package lineedit

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"unicode/utf8"
)

// ErrInterrupt is returned by ReadLine when Ctrl-C is pressed on an empty
// line. In raw mode Ctrl-C no longer raises SIGINT, so the editor reports it
// and the caller ends the session the way a signal would have.
var ErrInterrupt = errors.New("interrupted")

// Terminal is what the editor needs from the terminal it draws on: raw mode
// in and out, and the width for horizontal scrolling. It is an interface so
// the editor's key handling is tested with a fake rather than a pty.
type Terminal interface {
	// MakeRaw puts the terminal into raw mode and returns the function that
	// restores it. Calling restore more than once must be harmless.
	MakeRaw() (restore func() error, err error)
	// Width is the terminal's column count, or 0 when unknown.
	Width() int
}

// Editor reads edited lines from a terminal.
//
// CONCURRENCY: ReadLine is called from one goroutine at a time - the REPL's -
// but that goroutine may be abandoned mid-read by a cancelled context
// (readAsync in cmd), so the raw-mode state is guarded and Close, called from
// the REPL's own goroutine at session end, restores the terminal whether or
// not the abandoned read ever returns.
type Editor struct {
	in   *bufio.Reader
	out  io.Writer
	term Terminal

	// history is every entry delivered so far, oldest first. maxHistory
	// bounds it; the oldest entries fall off.
	history []string

	mu      sync.Mutex
	restore func() error
	// closed is set by Close and never cleared: a ReadLine that starts after
	// it (an abandoned goroutine reaching the call late) must not put the
	// terminal back into raw mode behind the session's back.
	closed bool
}

// ErrClosed is returned by ReadLine once Close has been called.
var ErrClosed = errors.New("line editor closed")

// The bounds. A history that grew without limit would only matter in a
// session that outlives its usefulness; a line that grew without limit is the
// 1 MB cap the cooked reader already enforces, applied here in runes for the
// same reason (a runaway paste must not allocate unboundedly - it is read and
// dropped past the cap, never retained).
const (
	maxHistory   = 1000
	maxLineRunes = 1 << 20
)

// New returns an editor reading from in and drawing on out. in MUST be the
// same buffered reader every other consumer of the stream uses (see the
// package comment).
func New(in *bufio.Reader, out io.Writer, term Terminal) *Editor {
	return &Editor{in: in, out: out, term: term}
}

// The key bytes the editor acts on. Everything else printable is inserted;
// everything else non-printable is ignored.
const (
	keyCtrlA     = 0x01
	keyCtrlB     = 0x02
	keyCtrlC     = 0x03
	keyCtrlD     = 0x04
	keyCtrlE     = 0x05
	keyCtrlF     = 0x06
	keyBackspace = 0x08
	keyTab       = 0x09
	keyEnter     = 0x0d
	keyNewline   = 0x0a
	keyCtrlK     = 0x0b
	keyCtrlL     = 0x0c
	keyCtrlN     = 0x0e
	keyCtrlP     = 0x10
	keyCtrlU     = 0x15
	keyCtrlW     = 0x17
	keyEscape    = 0x1b
	keyDelete    = 0x7f
)

// The escape sequences the terminal sends for the keys the editor handles,
// and the two bracketed-paste markers. Home/End come in three spellings
// depending on the terminal's mode; all three are recognized.
const (
	pasteBegin  = "[200~"
	pasteEnd    = "[201~"
	enablePaste = "\x1b[?2004h"
	disablePast = "\x1b[?2004l"
)

// state is one ReadLine's editing state: the line as runes, the cursor as a
// rune index, and the history position while browsing (len(history) means
// "the line being typed", which is stashed so Down returns to it).
type state struct {
	prompt  string
	line    []rune
	cursor  int
	histPos int
	stash   []rune
}

// ReadLine prompts, reads one edited entry and returns it without a trailing
// newline. A pasted block is one entry, its own newlines included. Ctrl-D on
// an empty line is io.EOF; Ctrl-C on an empty line is ErrInterrupt, and on a
// non-empty line it discards the line and prompts again, like a shell. The
// entry is added to the history unless it is blank or repeats the last one.
func (e *Editor) ReadLine(prompt string) (string, error) {
	if err := e.enterRaw(); err != nil {
		return "", err
	}
	defer e.leaveRaw()
	defer func() { _, _ = io.WriteString(e.out, disablePast) }()

	st := &state{prompt: prompt, histPos: len(e.history)}
	e.draw(st)
	for {
		r, _, err := e.in.ReadRune()
		if err != nil {
			_, _ = io.WriteString(e.out, "\r\n")
			if errors.Is(err, io.EOF) && len(st.line) > 0 {
				// A final line with no newline, as the cooked reader also
				// delivers it: the text first, EOF on the next call.
				return e.deliver(st), nil
			}
			return "", err
		}
		done, deliver, err := e.key(st, r)
		if err != nil {
			return "", err
		}
		if done {
			if deliver {
				return e.deliver(st), nil
			}
			return "", nil
		}
		e.draw(st)
	}
}

// deliver records the finished line in the history and returns it.
func (e *Editor) deliver(st *state) string {
	text := string(st.line)
	if strings.TrimSpace(text) != "" && (len(e.history) == 0 || e.history[len(e.history)-1] != text) {
		e.history = append(e.history, text)
		if len(e.history) > maxHistory {
			e.history = e.history[len(e.history)-maxHistory:]
		}
	}
	return text
}

// key applies one key. done means the read is over; deliver says whether
// the line is the result (false: Ctrl-C discarded it and the caller prompts
// again with an empty line - which is what "" with done means to ReadLine).
// The keys that end or redirect the read are decided here; every key that
// only changes the line is edit's.
func (e *Editor) key(st *state, r rune) (done, deliver bool, err error) {
	switch r {
	case keyEnter, keyNewline:
		_, _ = io.WriteString(e.out, "\r\n")
		return true, true, nil
	case keyCtrlC:
		_, _ = io.WriteString(e.out, "^C\r\n")
		if len(st.line) == 0 {
			return true, false, ErrInterrupt
		}
		// Discard and re-prompt: the line is gone, the session is not.
		st.line, st.cursor, st.histPos = nil, 0, len(e.history)
		return false, false, nil
	case keyCtrlD:
		if len(st.line) == 0 {
			_, _ = io.WriteString(e.out, "\r\n")
			return true, false, io.EOF
		}
		st.deleteAt()
	case keyEscape:
		return e.escape(st)
	case keyCtrlP:
		e.older(st)
	case keyCtrlN:
		e.newer(st)
	case keyCtrlL:
		// Clear the screen and redraw the prompt: the one screen-level act,
		// because a garbled terminal is not something a line editor can
		// otherwise recover from.
		_, _ = io.WriteString(e.out, "\x1b[H\x1b[2J")
	default:
		st.edit(r)
	}
	return false, false, nil
}

// edit applies one editing key to the line: the cursor and kill keys, and
// insertion for every printable rune. Non-printable bytes it does not know
// are ignored.
func (st *state) edit(r rune) {
	switch r {
	case keyBackspace, keyDelete:
		st.backspace()
	case keyCtrlA:
		st.cursor = 0
	case keyCtrlE:
		st.cursor = len(st.line)
	case keyCtrlB:
		st.left()
	case keyCtrlF:
		st.right()
	case keyCtrlU:
		st.line = st.line[st.cursor:]
		st.cursor = 0
	case keyCtrlK:
		st.line = st.line[:st.cursor]
	case keyCtrlW:
		st.killWord()
	case keyTab:
		st.insert(' ')
	default:
		if r >= 0x20 && r != utf8.RuneError {
			st.insert(r)
		}
	}
}

// escape reads the rest of an escape sequence and applies it. An unknown
// sequence is consumed and ignored: leaking its bytes into the line as text
// is the failure mode this exists to prevent. A bare Escape - one not
// followed by a sequence introducer - is dropped and the byte that followed
// it is handled as the key it is, so Escape then Enter still submits and
// Escape then Ctrl-C still interrupts. The returned flags are key's.
func (e *Editor) escape(st *state) (done, deliver bool, err error) {
	// The byte after Escape decides: a sequence introducer, or a key of its
	// own - read as a rune, so Escape then a non-ASCII character keeps that
	// character whole.
	r, _, err := e.in.ReadRune()
	if err != nil {
		return false, false, err
	}
	if r != '[' && r != 'O' {
		return e.key(st, r)
	}
	seq, err := e.readSequence(byte(r))
	if err != nil {
		return false, false, err
	}
	switch seq {
	case "[A", "OA":
		e.older(st)
	case "[B", "OB":
		e.newer(st)
	case "[C", "OC":
		st.right()
	case "[D", "OD":
		st.left()
	case "[H", "[1~", "OH":
		st.cursor = 0
	case "[F", "[4~", "OF":
		st.cursor = len(st.line)
	case "[3~":
		st.deleteAt()
	case pasteBegin:
		return false, false, e.paste(st)
	}
	return false, false, nil
}

// readSequence reads the rest of one CSI/SS3 sequence whose introducer ([ or
// O) has already been read: parameter bytes up to and including the final
// byte in 0x40-0x7e, bounded so an unknown sequence cannot eat the line.
func (e *Editor) readSequence(introducer byte) (string, error) {
	var b strings.Builder
	b.WriteByte(introducer)
	for {
		c, err := e.in.ReadByte()
		if err != nil {
			return b.String(), err
		}
		b.WriteByte(c)
		if c >= 0x40 && c <= 0x7e {
			return b.String(), nil
		}
		if b.Len() > 16 {
			return b.String(), nil // not a sequence this editor knows; stop eating
		}
	}
}

// paste inserts everything up to the end marker verbatim, newlines included:
// a pasted block is one entry. CR is normalized to LF so a block copied from
// a CRLF source does not carry stray carriage returns into the message. The
// block is inserted as it is read, so the line cap applies to it as it grows
// and the excess of a runaway paste is read and dropped, never retained.
func (e *Editor) paste(st *state) error {
	var prev rune
	for {
		r, _, err := e.in.ReadRune()
		if err != nil {
			return err
		}
		if r == keyEscape {
			intro, err := e.in.ReadByte()
			if err != nil {
				return err
			}
			if intro != '[' && intro != 'O' {
				continue // an escape inside a paste is not a key
			}
			seq, err := e.readSequence(intro)
			if err != nil {
				return err
			}
			if seq == pasteEnd {
				break
			}
			continue
		}
		// CRLF and a bare CR both become one LF; the LF of a CRLF pair is
		// the one byte skipped.
		if r == '\n' && prev == '\r' {
			prev = r
			continue
		}
		prev = r
		if r == '\r' {
			r = '\n'
		}
		st.insert(r)
	}
	return nil
}

// older moves one entry back in the history, stashing the line being typed
// on the first step so Down can bring it back.
func (e *Editor) older(st *state) {
	if st.histPos == 0 {
		return
	}
	if st.histPos == len(e.history) {
		st.stash = append([]rune(nil), st.line...)
	}
	st.histPos--
	st.line = []rune(e.history[st.histPos])
	st.cursor = len(st.line)
}

// newer moves one entry forward, back to the stashed line at the end.
func (e *Editor) newer(st *state) {
	if st.histPos >= len(e.history) {
		return
	}
	st.histPos++
	if st.histPos == len(e.history) {
		st.line = append([]rune(nil), st.stash...)
	} else {
		st.line = []rune(e.history[st.histPos])
	}
	st.cursor = len(st.line)
}

func (st *state) insert(r rune) {
	if len(st.line) >= maxLineRunes {
		return
	}
	st.line = append(st.line, 0)
	copy(st.line[st.cursor+1:], st.line[st.cursor:])
	st.line[st.cursor] = r
	st.cursor++
}

func (st *state) backspace() {
	if st.cursor == 0 {
		return
	}
	st.line = append(st.line[:st.cursor-1], st.line[st.cursor:]...)
	st.cursor--
}

func (st *state) deleteAt() {
	if st.cursor >= len(st.line) {
		return
	}
	st.line = append(st.line[:st.cursor], st.line[st.cursor+1:]...)
}

func (st *state) left() {
	if st.cursor > 0 {
		st.cursor--
	}
}

func (st *state) right() {
	if st.cursor < len(st.line) {
		st.cursor++
	}
}

// killWord deletes the word before the cursor and the spaces after it, the
// way Ctrl-W does in a shell.
func (st *state) killWord() {
	i := st.cursor
	for i > 0 && st.line[i-1] == ' ' {
		i--
	}
	for i > 0 && st.line[i-1] != ' ' {
		i--
	}
	st.line = append(st.line[:i], st.line[st.cursor:]...)
	st.cursor = i
}

// draw repaints the prompt line: carriage return, prompt, the visible window
// of the line, clear to end of line, then the cursor moved back into place.
//
// Only one row is ever drawn. When the line is wider than the terminal the
// window scrolls horizontally around the cursor instead of wrapping, because
// a wrapped line would need row bookkeeping this editor does not keep (that
// is where a TUI starts). Newlines inside the line - a pasted block - are
// shown as a return glyph so the entry stays one row; the entry itself keeps
// the real newlines. Every rune is assumed one cell wide; a double-width
// character misplaces the cursor by one cell until the line is sent, which
// is a known limit rather than a bug to work around here.
func (e *Editor) draw(st *state) {
	width := 0
	if e.term != nil {
		width = e.term.Width()
	}
	room := width - len([]rune(st.prompt)) - 1
	start := 0
	if width > 0 && room > 0 && len(st.line) > room {
		// Keep the cursor inside the window, with the window as far left as
		// that allows.
		if st.cursor > room {
			start = st.cursor - room
		}
	}
	end := len(st.line)
	if width > 0 && room > 0 && end-start > room {
		end = start + room
	}
	var b strings.Builder
	b.WriteString("\r")
	b.WriteString(st.prompt)
	for _, r := range st.line[start:end] {
		if r == '\n' {
			r = '↵'
		}
		b.WriteRune(r)
	}
	b.WriteString("\x1b[K")
	if back := end - st.cursor; back > 0 {
		fmt.Fprintf(&b, "\x1b[%dD", back)
	}
	_, _ = io.WriteString(e.out, b.String())
}

// enterRaw puts the terminal into raw mode for one ReadLine and switches
// bracketed paste on, both under the lock so neither can happen after Close.
func (e *Editor) enterRaw() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return ErrClosed
	}
	if e.term != nil && e.restore == nil {
		restore, err := e.term.MakeRaw()
		if err != nil {
			return fmt.Errorf("entering raw mode: %w", err)
		}
		e.restore = restore
	}
	_, _ = io.WriteString(e.out, enablePaste)
	return nil
}

// leaveRaw restores the terminal. Safe to call any number of times, from
// ReadLine's own defer and from Close.
func (e *Editor) leaveRaw() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.restore != nil {
		_ = e.restore()
		e.restore = nil
	}
}

// Close restores the terminal if a ReadLine left it raw - the case when the
// REPL abandoned a read on a cancelled context and the goroutine holding it
// never returned. It also switches bracketed paste off, for the same reason.
func (e *Editor) Close() {
	e.mu.Lock()
	raw := e.restore != nil
	e.closed = true
	e.mu.Unlock()
	if raw {
		_, _ = io.WriteString(e.out, disablePast)
	}
	e.leaveRaw()
}
