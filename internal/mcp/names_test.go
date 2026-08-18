package mcp

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
)

// hex8 mirrors the suffix the implementation must produce, computed here from
// first principles so the test fails if the implementation changes the recipe.
func hex8(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:4])
}

func TestEffectiveName(t *testing.T) {
	const server32 = "abcdefghijklmnopqrstuvwxyz012345" // 32 chars: the config maximum
	long := strings.Repeat("x", 70)

	for _, tc := range []struct {
		name       string
		server     string
		tool       string
		want       string
		normalized bool
	}{
		{"already valid", "github", "create_issue", "github__create_issue", false},
		{"dash is valid", "gh", "create-issue", "gh__create-issue", false},
		{"dot is not allowed", "gh", "a.b", "gh__a_b_" + hex8("a.b"), true},
		{"too long", "gh", long, "gh__" + strings.Repeat("x", 48) + "_" + hex8(long), true},
		{"non-ascii", "gh", "ünïcode", "gh___n_code_" + hex8("ünïcode"), true},
		{"empty tool", "gh", "", "gh___" + hex8(""), true},
		{
			"long server shrinks the budget",
			server32, long,
			server32 + "__" + strings.Repeat("x", 21) + "_" + hex8(long),
			true,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := EffectiveName(tc.server, tc.tool)
			if got.Effective != tc.want {
				t.Errorf("Effective = %q, want %q", got.Effective, tc.want)
			}
			if got.Normalized != tc.normalized {
				t.Errorf("Normalized = %v, want %v", got.Normalized, tc.normalized)
			}
			if got.Original != tc.tool {
				t.Errorf("Original = %q, want %q", got.Original, tc.tool)
			}
			if len(got.Effective) > MaxToolNameLen {
				t.Errorf("Effective is %d bytes, want <= %d", len(got.Effective), MaxToolNameLen)
			}
			if !toolNameRe.MatchString(got.Effective) {
				t.Errorf("Effective %q does not satisfy the provider-side rule", got.Effective)
			}
		})
	}
}

// TestEffectiveNameRespectsTheCeiling is the contract test for the hard
// MaxToolNameLen bound: it must hold for every server length, including ones
// config would never allow, not just for the 32-character maximum.
func TestEffectiveNameRespectsTheCeiling(t *testing.T) {
	tools := []string{"", "x", "a.b", "create_issue", strings.Repeat("x", 70), strings.Repeat("ü", 90)}
	for length := 1; length <= 64; length++ {
		server := strings.Repeat("s", length)
		for _, tool := range tools {
			got := EffectiveName(server, tool)
			if len(got.Effective) > MaxToolNameLen {
				t.Errorf("server len %d, tool %q: Effective is %d bytes (%q), want <= %d",
					length, tool, len(got.Effective), got.Effective, MaxToolNameLen)
			}
			if !toolNameRe.MatchString(got.Effective) {
				t.Errorf("server len %d, tool %q: %q does not satisfy the provider-side rule",
					length, tool, got.Effective)
			}
		}
	}
}

// TestEffectiveNameWithAnOverlongServer pins the deterministic last resort: a
// server name longer than the name budget itself is truncated and the tool
// half disappears, rather than the ceiling being breached.
func TestEffectiveNameWithAnOverlongServer(t *testing.T) {
	server := strings.Repeat("s", 60)
	got := EffectiveName(server, "create_issue")
	want := strings.Repeat("s", 53) + "___" + hex8("create_issue")
	if got.Effective != want {
		t.Errorf("Effective = %q, want %q", got.Effective, want)
	}
	if len(got.Effective) != MaxToolNameLen {
		t.Errorf("Effective is %d bytes, want exactly %d", len(got.Effective), MaxToolNameLen)
	}
	if !got.Normalized {
		t.Error("Normalized = false, want true")
	}
}

func TestEffectiveNameDisambiguates(t *testing.T) {
	a := EffectiveName("s", "a.b")
	b := EffectiveName("s", "a-b")
	if a.Effective == b.Effective {
		t.Fatalf("distinct originals collided on %q", a.Effective)
	}
}

func TestEffectiveNameIsDeterministic(t *testing.T) {
	for _, tool := range []string{"create_issue", "a.b", strings.Repeat("ü", 90)} {
		first := EffectiveName("gh", tool)
		second := EffectiveName("gh", tool)
		if first != second {
			t.Errorf("EffectiveName(%q) is not deterministic: %+v vs %+v", tool, first, second)
		}
	}
}

func TestKeep(t *testing.T) {
	for _, tc := range []struct {
		name       string
		tool       string
		include    []string
		exclude    []string
		wantKeep   bool
		wantReason string
	}{
		{"no filters", "create_issue", nil, nil, true, ""},
		{"include hit", "create_issue", []string{"create_*"}, nil, true, ""},
		{"include exact hit", "create_issue", []string{"create_issue"}, nil, true, ""},
		{"include miss", "delete_repo", []string{"create_*"}, nil, false, "not included"},
		{"exclude hit", "delete_repo", nil, []string{"delete_*"}, false, "excluded"},
		{"exclude miss", "create_issue", nil, []string{"delete_*"}, true, ""},
		{"exclude beats include", "delete_repo", []string{"*"}, []string{"delete_*"}, false, "excluded"},
		{"empty include list is all", "anything", []string{}, nil, true, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			keep, reason := Keep(tc.tool, tc.include, tc.exclude)
			if keep != tc.wantKeep || reason != tc.wantReason {
				t.Errorf("Keep = (%v, %q), want (%v, %q)", keep, reason, tc.wantKeep, tc.wantReason)
			}
		})
	}
}
