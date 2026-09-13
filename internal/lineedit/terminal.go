package lineedit

import (
	"os"

	"golang.org/x/term"
)

// TTY is the real Terminal over a file descriptor, backed by x/term.
type TTY struct {
	fd int
}

// NewTTY returns the Terminal for f, which the caller has already established
// to be a terminal (the editor is only built for one).
func NewTTY(f *os.File) *TTY {
	return &TTY{fd: int(f.Fd())}
}

// MakeRaw implements Terminal.
func (t *TTY) MakeRaw() (func() error, error) {
	saved, err := term.MakeRaw(t.fd)
	if err != nil {
		return nil, err
	}
	return func() error { return term.Restore(t.fd, saved) }, nil
}

// Width implements Terminal; 0 when the size cannot be read.
func (t *TTY) Width() int {
	w, _, err := term.GetSize(t.fd)
	if err != nil {
		return 0
	}
	return w
}
