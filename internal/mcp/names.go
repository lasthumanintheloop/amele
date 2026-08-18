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
	// remaining budget computed in EffectiveName.
	maxCleanedLen = 48
	// minCleanedLen keeps a rewritten name from losing the tool half entirely
	// if a server name ever grows past what config allows (32 characters).
	minCleanedLen = 8
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
// The cut is sized so the whole result stays within MaxToolNameLen:
// 64 - len(server) - len(NameSeparator) - 1 - 8, capped at 48 for readability
// and floored at 8. Since config limits a server name to 32 characters the
// floor is unreachable in practice; it exists so a hand-built call still
// returns a usable name rather than an empty tool half.
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

	cleaned := strings.Map(func(r rune) rune {
		if validRune(r) {
			return r
		}
		return '_'
	}, tool)

	budget := MaxToolNameLen - len(server) - len(NameSeparator) - 1 - hashLen
	budget = min(budget, maxCleanedLen)
	budget = max(budget, minCleanedLen)
	// Cutting by runes (not bytes) cannot split a character: strings.Map has
	// already replaced every non-ASCII rune with '_', so runes are bytes here,
	// but the rune form survives a future widening of validRune.
	if r := []rune(cleaned); len(r) > budget {
		cleaned = string(r[:budget])
	}

	sum := sha256.Sum256([]byte(tool))
	return NamedTool{
		Effective:  server + NameSeparator + cleaned + "_" + hex.EncodeToString(sum[:hashLen/2]),
		Original:   tool,
		Normalized: true,
	}
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
