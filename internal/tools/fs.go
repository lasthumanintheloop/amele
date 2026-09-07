package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/lasthumanintheloop/amele/internal/llm"
)

// DefaultMaxReadBytes bounds fs_read output so a single large file cannot
// blow the model's context (and the token budget with it).
const DefaultMaxReadBytes = 256 * 1024

// DefaultMaxListBytes bounds fs_list output for the same reason: a directory
// of rotated logs can hold tens of thousands of entries. It is smaller than
// the read cap because a listing is an index, not content - a model that
// needs more should narrow the path. It is defined as the subprocess stdout
// cap rather than repeating the number: both answer "how much incidental tool
// output may one call spend", and two copies of 64 KiB would drift.
const DefaultMaxListBytes = DefaultMaxOutputBytes

// sandbox confines all filesystem access to a workspace root.
//
// SECURITY: built on os.Root, which resolves every path component inside the
// kernel-side directory handle (openat2/RESOLVE_BENEATH semantics). This
// blocks "..", absolute paths and symlink escapes race-free: a parent
// directory swapped for an outward symlink between check and use is caught
// by the kernel, not by a stale userspace prefix check. It still is not a
// full security boundary for the whole agent - subprocess tools are outside
// its reach (docs/threat-model.md).
type sandbox struct {
	root *os.Root
}

func newSandbox(workspace string) (*sandbox, error) {
	root, err := os.OpenRoot(workspace)
	if err != nil {
		return nil, fmt.Errorf("opening workspace %q: %w", workspace, err)
	}
	// The root handle intentionally lives for the process lifetime: amele is
	// a one-shot CLI and every fs tool call needs it.
	return &sandbox{root: root}, nil
}

// clean normalizes a model-supplied path and rejects the shapes os.Root
// would refuse anyway, with friendlier messages.
func (s *sandbox) clean(rel string) (string, error) {
	if rel == "" {
		return ".", nil
	}
	if filepath.IsAbs(rel) {
		return "", fmt.Errorf("path %q must be relative to the workspace", rel)
	}
	cleaned := filepath.Clean(rel)
	if cleaned == ".." || strings.HasPrefix(cleaned, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path %q escapes the workspace", rel)
	}
	return cleaned, nil
}

// escapeErr rewraps os.Root escape errors into the model-friendly wording
// used across the fs tools. The kernel-side resolver (openat2 with
// RESOLVE_BENEATH) reports escapes as EXDEV, which is the robust signal; the
// text match covers Go's userspace fallback resolver, whose error carries no
// sentinel - fragile across Go versions, but pinned by TestSandboxEscapes.
func escapeErr(rel string, err error) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, syscall.EXDEV) || strings.Contains(err.Error(), "escapes from parent") {
		return fmt.Errorf("path %q escapes the workspace", rel)
	}
	return err
}

// FSOptions tunes the builtin filesystem tools.
type FSOptions struct {
	// MaxReadBytes overrides DefaultMaxReadBytes when > 0.
	MaxReadBytes int
	// MaxListBytes overrides DefaultMaxListBytes when > 0.
	MaxListBytes int
}

// NewFSTools builds the three builtin filesystem tools bound to workspace.
func NewFSTools(workspace string, opts FSOptions) ([]Tool, error) {
	sb, err := newSandbox(workspace)
	if err != nil {
		return nil, err
	}
	maxRead := opts.MaxReadBytes
	if maxRead <= 0 {
		maxRead = DefaultMaxReadBytes
	}
	maxList := opts.MaxListBytes
	if maxList <= 0 {
		maxList = DefaultMaxListBytes
	}
	return []Tool{
		&fsRead{sb: sb, maxBytes: maxRead},
		&fsWrite{sb: sb},
		&fsList{sb: sb, maxBytes: maxList},
	}, nil
}

// pathArgs is the argument shape shared by fs_read and fs_list.
type pathArgs struct {
	Path string `json:"path"`
}

// decodeArgs parses model-supplied JSON arguments strictly, so mis-spelled
// argument names surface as immediate tool errors instead of being ignored.
//
// SECURITY: strictness here is a safety boundary, not tidiness - the argument
// string is model-controlled and every tool acts on what it yields. Three
// things are rejected:
//   - unknown fields (a typo must not silently become a default),
//   - anything after the first JSON value: `{"path":"x"} rm -rf /` used to run
//     as `{"path":"x"}`, hiding the rest of the payload from every log that
//     records what the tool was asked to do,
//   - a literal null, which decodes into a ZERO struct without error and so
//     turned an argument-less `fs_list null` into "list the workspace root".
//     A null argument object is never a meaningful request for any tool.
//
// The empty string is the caller's business, not this function's: tools that
// take no required argument skip decodeArgs entirely for it.
func decodeArgs(raw string, into any) error {
	if strings.TrimSpace(raw) == "null" {
		return fmt.Errorf("invalid arguments %q: expected a JSON object", raw)
	}
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(into); err != nil {
		return fmt.Errorf("invalid arguments %q: %w", raw, err)
	}
	// A clean end of input is the only acceptable remainder; Token reports
	// io.EOF there and an error (or another value) for anything else.
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return fmt.Errorf("invalid arguments %q: unexpected data after the JSON object", raw)
	}
	return nil
}

type fsRead struct {
	sb       *sandbox
	maxBytes int
}

func (t *fsRead) Def() llm.ToolDef {
	return llm.ToolDef{
		Name:        "fs_read",
		Description: "Read a text file inside the workspace. Path is relative to the workspace root.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {"path": {"type": "string", "description": "Relative file path"}},
			"required": ["path"],
			"additionalProperties": false
		}`),
	}
}

func (t *fsRead) Invoke(ctx context.Context, rawArgs string) (string, error) {
	out, _, err := t.InvokeOutcome(ctx, rawArgs)
	return out, err
}

// InvokeOutcome is Invoke plus the truncation fact for the session log: the
// marker tells the model its read was cut, and Outcome.Truncated tells the
// operator the same thing without parsing the result text.
func (t *fsRead) InvokeOutcome(ctx context.Context, rawArgs string) (string, Outcome, error) {
	var args pathArgs
	if err := decodeArgs(rawArgs, &args); err != nil {
		return "", Outcome{}, err
	}
	rel, err := t.sb.clean(args.Path)
	if err != nil {
		return "", Outcome{}, err
	}
	if err := ctx.Err(); err != nil {
		return "", Outcome{}, err
	}

	f, err := t.sb.root.Open(rel)
	if err != nil {
		return "", Outcome{}, escapeErr(args.Path, fmt.Errorf("reading %q: %w", args.Path, err))
	}
	defer func() { _ = f.Close() }()

	// Only regular files are readable: a FIFO would block past the run
	// deadline and a device file is never agent business.
	info, err := f.Stat()
	if err != nil {
		return "", Outcome{}, fmt.Errorf("reading %q: %w", args.Path, err)
	}
	if !info.Mode().IsRegular() {
		return "", Outcome{}, fmt.Errorf("%q is not a regular file", args.Path)
	}

	// Read at most maxBytes+1: the extra byte detects truncation without
	// ever loading an arbitrarily large file into memory. The +1 is
	// saturated because it overflows at the top of the int range, and a
	// negative io.LimitReader reads NOTHING - the largest possible cap, the
	// one that can never truncate anything, used to return every file empty.
	limit := int64(math.MaxInt64)
	if int64(t.maxBytes) < math.MaxInt64 {
		limit = int64(t.maxBytes) + 1
	}
	data, err := io.ReadAll(io.LimitReader(f, limit))
	if err != nil {
		return "", Outcome{}, fmt.Errorf("reading %q: %w", args.Path, err)
	}
	out, truncated := CapText(string(data), t.maxBytes)
	return out, Outcome{Truncated: truncated}, nil
}

type fsWriteArgs struct {
	Path    string `json:"path"`
	Content string `json:"content"`
}

type fsWrite struct {
	sb *sandbox
}

func (t *fsWrite) Def() llm.ToolDef {
	return llm.ToolDef{
		Name:        "fs_write",
		Description: "Write a text file inside the workspace, creating parent directories. Overwrites existing content.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {"type": "string", "description": "Relative file path"},
				"content": {"type": "string", "description": "Full file content"}
			},
			"required": ["path", "content"],
			"additionalProperties": false
		}`),
	}
}

func (t *fsWrite) Invoke(ctx context.Context, rawArgs string) (string, error) {
	var args fsWriteArgs
	if err := decodeArgs(rawArgs, &args); err != nil {
		return "", err
	}
	rel, err := t.sb.clean(args.Path)
	if err != nil {
		return "", err
	}
	if rel == "." {
		return "", errors.New("path must name a file, not the workspace root")
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}

	// SECURITY: MkdirAll and OpenFile both run through the os.Root handle,
	// so directory creation and the write resolve inside the sandbox even
	// if the tree is being mutated concurrently.
	if dir := filepath.Dir(rel); dir != "." {
		if err := t.sb.root.MkdirAll(dir, 0o750); err != nil {
			return "", escapeErr(args.Path, fmt.Errorf("creating parent directories for %q: %w", args.Path, err))
		}
	}
	f, err := t.sb.root.OpenFile(rel, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return "", escapeErr(args.Path, fmt.Errorf("writing %q: %w", args.Path, err))
	}
	if _, err := f.Write([]byte(args.Content)); err != nil {
		_ = f.Close()
		return "", fmt.Errorf("writing %q: %w", args.Path, err)
	}
	if err := f.Close(); err != nil {
		return "", fmt.Errorf("writing %q: %w", args.Path, err)
	}
	return fmt.Sprintf("wrote %d bytes to %s", len(args.Content), args.Path), nil
}

type fsList struct {
	sb       *sandbox
	maxBytes int
}

func (t *fsList) Def() llm.ToolDef {
	return llm.ToolDef{
		Name:        "fs_list",
		Description: "List a directory inside the workspace. Directories are suffixed with '/'. Path defaults to the workspace root.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {"path": {"type": "string", "description": "Relative directory path (default: '.')"}},
			"additionalProperties": false
		}`),
	}
}

func (t *fsList) Invoke(ctx context.Context, rawArgs string) (string, error) {
	out, _, err := t.InvokeOutcome(ctx, rawArgs)
	return out, err
}

// InvokeOutcome is Invoke plus the truncation fact for the session log. The
// listing is cut on an entry boundary rather than through CapText: half a
// filename is worse than a missing one, and the count in the marker tells the
// model exactly how much it did not see.
func (t *fsList) InvokeOutcome(ctx context.Context, rawArgs string) (string, Outcome, error) {
	var args pathArgs
	if rawArgs != "" {
		if err := decodeArgs(rawArgs, &args); err != nil {
			return "", Outcome{}, err
		}
	}
	rel, err := t.sb.clean(args.Path)
	if err != nil {
		return "", Outcome{}, err
	}
	if err := ctx.Err(); err != nil {
		return "", Outcome{}, err
	}

	entries, err := fs.ReadDir(t.sb.root.FS(), rel)
	if err != nil {
		return "", Outcome{}, escapeErr(args.Path, fmt.Errorf("listing %q: %w", args.Path, err))
	}
	if len(entries) == 0 {
		return "(empty directory)", Outcome{}, nil
	}
	// The listing is bounded like every other tool output (fs_read,
	// subprocess): a directory of rotated logs can hold tens of thousands
	// of entries, and one unbounded result would flood the model's context.
	//
	// The marker is part of the budget, not an extra on top of it: room for
	// it is reserved BEFORE any entry is written. When
	// limits.max_tool_result_bytes is set, the loop ceiling is the same
	// number as this budget, and a listing that overshot here would be
	// re-cut there by CapText - which knows nothing about entry counts and
	// would leave the model half a marker followed by the plain one. The
	// reservation uses the total for both counts, an upper bound on the
	// digits `shown` can take. A budget under the marker's own length cannot
	// be honored at all (CapText has the same floor); the counts win there,
	// because a silently empty listing reads as an empty directory.
	reserve := len(fmt.Sprintf("[output truncated by amele: %d of %d entries shown]", len(entries), len(entries)))
	var b strings.Builder
	shown := 0
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() {
			name += "/"
		}
		if b.Len()+len(name)+1+reserve > t.maxBytes {
			break
		}
		b.WriteString(name)
		b.WriteString("\n")
		shown++
	}
	if shown < len(entries) {
		// Stating the totals lets the model narrow its request instead of
		// assuming it saw everything.
		fmt.Fprintf(&b, "[output truncated by amele: %d of %d entries shown]", shown, len(entries))
		return b.String(), Outcome{Truncated: true}, nil
	}
	return strings.TrimSuffix(b.String(), "\n"), Outcome{}, nil
}
