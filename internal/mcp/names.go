package mcp

import (
	"crypto/sha256"
	"encoding/hex"
	"regexp"
	"strconv"
	"strings"

	"github.com/lasthumanintheloop/amele/internal/tools"
)

const (
	// NameSeparator joins the server name and the server-side tool name into
	// the single flat name a provider sees. Two underscores (rather than one,
	// or a dot) keep the boundary readable while staying inside the character
	// class every provider accepts.
	NameSeparator = "__"
	// MaxToolNameLen is the provider-side ceiling on a tool name, mirroring
	// config.toolNameRe. CONTRACT: no name this package hands to the loop may
	// exceed it, whatever the server called its tool.
	MaxToolNameLen = 64
	// hashLen is how many hex characters of sha256(original) are appended when
	// a name has to be rewritten. Four bytes is enough to keep two tools of one
	// server apart while costing little of the name budget.
	hashLen = 8
	// maxCleanedLen is the preferred cap on the tool half of a rewritten name.
	// It is a readability limit, not a correctness one - the real bound is the
	// remaining budget computed in EffectiveName, which always wins.
	maxCleanedLen = 48
	// suffixLen is the cost of the disambiguating tail: '_' plus hashLen hex
	// characters.
	suffixLen = 1 + hashLen
	// maxServerLen is the longest server half that still leaves room for the
	// separator and the suffix inside MaxToolNameLen. A server name longer than
	// this (config forbids it: mcpServerNameRe caps it at 32) is truncated so
	// the hard ceiling is never breached.
	maxServerLen = MaxToolNameLen - len(NameSeparator) - suffixLen
)

// toolNameRe is the provider-side rule a model-facing tool name must satisfy.
// It is duplicated from config on purpose: this package must be able to check
// its own output without importing the config package.
var toolNameRe = regexp.MustCompile(`^[A-Za-z0-9_-]{1,` + strconv.Itoa(MaxToolNameLen) + `}$`)

// NamedTool is one server-side tool name and the model-facing name derived
// from it. The pair is kept together because every later step needs both: the
// loop dispatches on Effective, the wire call uses Original.
type NamedTool struct {
	// Effective is the model-facing name: <server>__<tool>, rewritten if the
	// raw join would break the provider-side rule.
	Effective string
	// Original is the server-side name, sent back verbatim in tools/call.
	Original string
	// Normalized reports whether Effective differs from the plain join. It is
	// logged so an operator can see that a name was rewritten rather than
	// wondering why the model sees a name the server never published.
	Normalized bool
}

// EffectiveName builds the model-facing name for tool published by server.
//
// If <server>__<tool> already satisfies ^[A-Za-z0-9_-]{1,64}$ it is used
// verbatim and Normalized is false. Otherwise the name is rewritten: every
// rune outside that class becomes '_', the cleaned tool half is cut to fit,
// and '_' plus the first 8 hex characters of sha256 of the ORIGINAL tool name
// are appended. Hashing the original (not the cleaned form) is what stops
// "a.b" and "a-b" from collapsing onto one name.
//
// CONTRACT: the result is at most MaxToolNameLen bytes and always satisfies
// the provider-side rule, for ANY input. The tool half is cut to whatever the
// hard ceiling leaves - 64 - len(server) - len(NameSeparator) - 1 - 8 - capped
// at 48 for readability. The ceiling always wins over readability, never the
// other way round. Config limits a server name to 32 characters, which leaves
// 21 characters of tool name; if a caller ever passes a server name longer
// than maxServerLen (53) the server half itself is truncated and the tool half
// becomes empty, so the name degrades to <cut server>___<hash>. That case
// cannot distinguish two such servers - it is a deterministic last resort that
// keeps the contract, not a supported configuration.
//
// An empty tool name is always rewritten, so a nameless tool cannot silently
// present itself to the model as the bare server prefix.
//
// EffectiveName is pure and deterministic: same inputs, same bytes, forever.
func EffectiveName(server, tool string) NamedTool {
	joined := server + NameSeparator + tool
	if tool != "" && toolNameRe.MatchString(joined) {
		return NamedTool{Effective: joined, Original: tool}
	}

	// The server half is cleaned too. Config already guarantees it is valid,
	// but this package must be able to keep its own contract without trusting
	// its caller.
	cleanedServer := cut(clean(server), maxServerLen)

	// budget is what the hard ceiling leaves for the tool half after the server,
	// the separator and the suffix. It is derived from MaxToolNameLen on every
	// call - never floored at a constant - so the ceiling cannot be breached by
	// an unusually long server name.
	budget := min(MaxToolNameLen-len(cleanedServer)-len(NameSeparator)-suffixLen, maxCleanedLen)
	cleanedTool := ""
	if budget > 0 {
		cleanedTool = cut(clean(tool), budget)
	}

	sum := sha256.Sum256([]byte(tool))
	return NamedTool{
		Effective:  cleanedServer + NameSeparator + cleanedTool + "_" + hex.EncodeToString(sum[:hashLen/2]),
		Original:   tool,
		Normalized: true,
	}
}

// clean replaces every rune that may not appear in a tool name with '_'.
func clean(s string) string {
	return strings.Map(func(r rune) rune {
		if validRune(r) {
			return r
		}
		return '_'
	}, s)
}

// cut shortens s to at most n runes. Cutting by runes (not bytes) cannot split
// a character: clean has already replaced every non-ASCII rune with '_', so
// runes are bytes here, but the rune form survives a future widening of
// validRune.
func cut(s string, n int) string {
	if r := []rune(s); len(r) > n {
		return string(r[:n])
	}
	return s
}

// validRune reports whether r may appear in a model-facing tool name. It is a
// hand-rolled test rather than a regexp per rune because it runs once per rune
// of every discovered tool name.
func validRune(r rune) bool {
	return r == '_' || r == '-' ||
		(r >= '0' && r <= '9') ||
		(r >= 'a' && r <= 'z') ||
		(r >= 'A' && r <= 'Z')
}

// Keep applies a server's include/exclude filter to one server-side tool name.
//
// An empty include list means "everything"; a non-empty one is a whitelist.
// exclude is applied afterwards and always wins, so an operator can write
// include: ["*"] with a short exclude list. Matching uses tools.GlobMatch (the
// same one-rule glob as the shell allow/deny lists) against the ORIGINAL name,
// because that is the name the operator read in the server's documentation.
//
// reason is "" when the tool is kept and otherwise a short, loggable phrase:
// "not included" or "excluded".
func Keep(original string, include, exclude []string) (keep bool, reason string) {
	for _, pattern := range exclude {
		if tools.GlobMatch(pattern, original) {
			return false, "excluded"
		}
	}
	if len(include) == 0 {
		return true, ""
	}
	for _, pattern := range include {
		if tools.GlobMatch(pattern, original) {
			return true, ""
		}
	}
	return false, "not included"
}
