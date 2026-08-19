package oauthtoken

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fixedClock returns a clock function pinned to ts, so no test depends on the
// wall clock.
func fixedClock(ts time.Time) func() time.Time { return func() time.Time { return ts } }

func testKey() Key {
	return Key{
		Issuer:   "https://as.example.com",
		Resource: "https://mcp.example.com/mcp",
		ClientID: "cid",
	}
}

func testRecord(k Key) *Record {
	//nolint:gosec // G101: "at"/"rt" are two-letter test fixtures, not credentials.
	return &Record{
		Version:       1,
		Issuer:        k.Issuer,
		Resource:      k.Resource,
		ClientID:      k.ClientID,
		TokenEndpoint: "https://as.example.com/token",
		AccessToken:   "at",
		RefreshToken:  "rt",
		ExpiresAt:     time.Unix(2000, 0).UTC(),
	}
}

func TestSaveLoadRoundtrip(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0).UTC()))
	k := testKey()
	r := testRecord(k)
	if err := s.Save(k, r); err != nil {
		t.Fatal(err)
	}
	got, err := s.Load(k)
	if err != nil || got.AccessToken != "at" {
		t.Fatalf("Load = %+v, %v", got, err)
	}
	if got.RefreshToken != "rt" || !got.ExpiresAt.Equal(r.ExpiresAt) || got.Version != 1 {
		t.Fatalf("Load lost fields: %+v", got)
	}
}

func TestLoadMissingIsErrNotFound(t *testing.T) {
	s := NewStore(t.TempDir(), fixedClock(time.Unix(1000, 0).UTC()))
	if _, err := s.Load(testKey()); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Load on empty dir = %v, want ErrNotFound", err)
	}
}

func TestFilePermissions(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "mcp")
	s := NewStore(dir, fixedClock(time.Unix(1000, 0).UTC()))
	k := testKey()
	if err := s.Save(k, testRecord(k)); err != nil {
		t.Fatal(err)
	}
	di, err := os.Stat(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got := di.Mode() & os.ModePerm; got != 0o700 {
		t.Errorf("dir mode = %o, want 700", got)
	}
	files := listJSON(t, dir)
	if len(files) != 1 {
		t.Fatalf("files = %v, want 1", files)
	}
	fi, err := os.Stat(files[0])
	if err != nil {
		t.Fatal(err)
	}
	if got := fi.Mode() & os.ModePerm; got != 0o600 {
		t.Errorf("file mode = %o, want 600", got)
	}
}

func TestKeyFieldVerification(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0).UTC()))
	k := testKey()
	if err := s.Save(k, testRecord(k)); err != nil {
		t.Fatal(err)
	}
	files := listJSON(t, dir)
	if len(files) != 1 {
		t.Fatalf("files = %v, want 1", files)
	}
	var raw map[string]any
	data, err := os.ReadFile(files[0])
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatal(err)
	}
	raw["issuer"] = "https://evil.example.com"
	tampered, err := json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(files[0], tampered, 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = s.Load(k)
	if err == nil {
		t.Fatal("Load of tampered record = nil error, want identity error")
	}
	if errors.Is(err, ErrNotFound) {
		t.Fatalf("Load of tampered record = ErrNotFound, want identity error: %v", err)
	}
	if !strings.Contains(err.Error(), "identity") {
		t.Fatalf("error %q does not mention identity", err)
	}
}

func TestDifferentClientIDsAreDifferentFiles(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0).UTC()))
	a := Key{Issuer: "https://as.example.com", Resource: "https://mcp.example.com/mcp", ClientID: "a"}
	b := Key{Issuer: "https://as.example.com", Resource: "https://mcp.example.com/mcp", ClientID: "b"}
	ra, rb := testRecord(a), testRecord(b)
	ra.AccessToken, rb.AccessToken = "token-a", "token-b"
	if err := s.Save(a, ra); err != nil {
		t.Fatal(err)
	}
	if err := s.Save(b, rb); err != nil {
		t.Fatal(err)
	}
	if files := listJSON(t, dir); len(files) != 2 {
		t.Fatalf("files = %v, want 2", files)
	}
	got, err := s.Load(a)
	if err != nil || got.AccessToken != "token-a" {
		t.Fatalf("Load(a) = %+v, %v", got, err)
	}
}

func TestDeleteIsIdempotent(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0).UTC()))
	k := testKey()
	if err := s.Delete(k); err != nil {
		t.Fatalf("Delete on empty dir = %v, want nil", err)
	}
	if err := s.Save(k, testRecord(k)); err != nil {
		t.Fatal(err)
	}
	if err := s.Delete(k); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Load(k); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Load after Delete = %v, want ErrNotFound", err)
	}
	if err := s.Delete(k); err != nil {
		t.Fatalf("second Delete = %v, want nil", err)
	}
}

func TestListReturnsEntriesAndSkipsCorrupt(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0).UTC()))
	k := testKey()
	if err := s.Save(k, testRecord(k)); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "broken-0123456789abcdef0123456789abcdef.json"), []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	entries, err := s.List()
	if err == nil {
		t.Error("List = nil error, want a report of the corrupt file")
	}
	if len(entries) != 1 {
		t.Fatalf("entries = %+v, want the one good record", entries)
	}
	if entries[0].Key != k || entries[0].Record.AccessToken != "at" {
		t.Fatalf("entry = %+v, want key %+v", entries[0], k)
	}
}

func TestListOnMissingDirIsEmpty(t *testing.T) {
	s := NewStore(filepath.Join(t.TempDir(), "absent"), fixedClock(time.Unix(1000, 0).UTC()))
	entries, err := s.List()
	if err != nil || len(entries) != 0 {
		t.Fatalf("List = %+v, %v; want empty, nil", entries, err)
	}
}

func TestCanonicalResource(t *testing.T) {
	cases := []struct {
		in, want string
		wantErr  bool
	}{
		{"https://x.com/mcp/", "https://x.com/mcp", false},
		{"https://x.com/mcp//", "https://x.com/mcp", false},
		{"https://x.com/mcp?k=v", "https://x.com/mcp", false},
		{"https://u:p@x.com/mcp", "https://x.com/mcp", false},
		{"https://x.com/mcp#frag", "", true}, // fragment rejected per spec
		{"://bad", "", true},
		{"https://X.com/mcp", "https://x.com/mcp", false},
		{"HTTPS://x.com:8443/mcp", "https://x.com:8443/mcp", false},
		{"https://x.com/", "https://x.com", false},
		{"/mcp", "", true},
		{"ftp://x.com/mcp", "", true},
		{"https:///mcp", "", true},
		{"", "", true},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			got, err := CanonicalResource(tc.in)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("CanonicalResource(%q) = %q, want error", tc.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("CanonicalResource(%q) = %v", tc.in, err)
			}
			if got != tc.want {
				t.Fatalf("CanonicalResource(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestFresh(t *testing.T) {
	now := time.Unix(1000, 0).UTC()
	cases := []struct {
		name      string
		expiresAt time.Time
		margin    time.Duration
		want      bool
	}{
		{"expires inside margin", now.Add(30 * time.Second), time.Minute, false},
		{"expires beyond margin", now.Add(120 * time.Second), time.Minute, true},
		{"zero expiry", time.Time{}, time.Minute, false},
		{"already expired", now.Add(-time.Second), 0, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := &Record{ExpiresAt: tc.expiresAt}
			if got := r.Fresh(now, tc.margin); got != tc.want {
				t.Fatalf("Fresh = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestDefaultDir(t *testing.T) {
	cases := []struct {
		name string
		env  map[string]string
		want string
	}{
		{"xdg set", map[string]string{"XDG_STATE_HOME": "/xdg"}, filepath.Join("/xdg", "amele", "mcp")},
		{"xdg empty falls back to home", map[string]string{"XDG_STATE_HOME": "", "HOME": "/home/u"}, filepath.Join("/home/u", ".local", "state", "amele", "mcp")},
		{"home only", map[string]string{"HOME": "/home/u"}, filepath.Join("/home/u", ".local", "state", "amele", "mcp")},
		{"nothing set", map[string]string{}, filepath.Join(".local", "state", "amele", "mcp")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			env := func(k string) (string, bool) { v, ok := tc.env[k]; return v, ok }
			if got := DefaultDir(env); got != tc.want {
				t.Fatalf("DefaultDir = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestSaveOverwritesInPlace(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0).UTC()))
	k := testKey()
	r := testRecord(k)
	if err := s.Save(k, r); err != nil {
		t.Fatal(err)
	}
	r.AccessToken = "rotated"
	if err := s.Save(k, r); err != nil {
		t.Fatal(err)
	}
	if files := listJSON(t, dir); len(files) != 1 {
		t.Fatalf("files = %v, want 1 (no temp leftovers)", files)
	}
	got, err := s.Load(k)
	if err != nil || got.AccessToken != "rotated" {
		t.Fatalf("Load = %+v, %v", got, err)
	}
}

func TestFilenameContainsResourceHost(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0).UTC()))
	k := Key{Issuer: "https://as.example.com", Resource: "https://mcp.example.com:8443/mcp", ClientID: "cid"}
	if err := s.Save(k, testRecord(k)); err != nil {
		t.Fatal(err)
	}
	files := listJSON(t, dir)
	if len(files) != 1 {
		t.Fatalf("files = %v, want 1", files)
	}
	name := filepath.Base(files[0])
	if !strings.HasPrefix(name, "mcp.example.com_8443-") {
		t.Fatalf("filename %q does not start with the sanitized host", name)
	}
	if strings.ContainsAny(name, `:/\`) {
		t.Fatalf("filename %q contains a path-unsafe character", name)
	}
}

// listJSON returns the absolute paths of the *.json files in dir.
func listJSON(t *testing.T, dir string) []string {
	t.Helper()
	matches, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		t.Fatal(err)
	}
	return matches
}

func TestNowUsesInjectedClock(t *testing.T) {
	ts := time.Unix(4242, 0).UTC()
	s := NewStore(t.TempDir(), fixedClock(ts))
	if got := s.Now(); !got.Equal(ts) {
		t.Fatalf("Now = %v, want %v", got, ts)
	}
}

func TestLoadRejectsNewerSchemaVersion(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0).UTC()))
	k := testKey()
	r := testRecord(k)
	r.Version = Version + 1
	if err := s.Save(k, r); err != nil {
		t.Fatal(err)
	}
	_, err := s.Load(k)
	if err == nil || errors.Is(err, ErrNotFound) {
		t.Fatalf("Load = %v, want a schema-version error", err)
	}
	if !strings.Contains(err.Error(), "version") {
		t.Fatalf("error %q does not mention the version", err)
	}
}

func TestSaveFailsWhenDirCannotBeCreated(t *testing.T) {
	root := t.TempDir()
	blocker := filepath.Join(root, "file")
	if err := os.WriteFile(blocker, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	s := NewStore(filepath.Join(blocker, "mcp"), fixedClock(time.Unix(1000, 0).UTC()))
	k := testKey()
	if err := s.Save(k, testRecord(k)); err == nil {
		t.Fatal("Save into a path blocked by a file = nil, want error")
	}
}

// TestPathOccupiedByDirectory covers the three operations that must fail loudly
// (not silently) when something other than a token file sits at the record's
// path: Save cannot rename over it, Load cannot read it, Delete cannot remove it.
func TestPathOccupiedByDirectory(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0).UTC()))
	k := testKey()
	occupied := s.path(k)
	if err := os.MkdirAll(filepath.Join(occupied, "child"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := s.Save(k, testRecord(k)); err == nil {
		t.Error("Save over a directory = nil, want error")
	}
	if _, err := s.Load(k); err == nil || errors.Is(err, ErrNotFound) {
		t.Errorf("Load of a directory = %v, want a read error", err)
	}
	if err := s.Delete(k); err == nil {
		t.Error("Delete of a non-empty directory = nil, want error")
	}
}

func TestListReportsMisfiledRecord(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0).UTC()))
	k := testKey()
	//nolint:gosec // G117: encoding a fixture record in a test; nothing here leaves the temp dir.
	data, err := json.Marshal(testRecord(k))
	if err != nil {
		t.Fatal(err)
	}
	// A valid record under a name that does not match its own identity: the
	// filename says one server, the contents another.
	if err := os.WriteFile(filepath.Join(dir, "wrong-0123456789abcdef0123456789abcdef.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	entries, err := s.List()
	if len(entries) != 0 {
		t.Fatalf("entries = %+v, want none", entries)
	}
	if err == nil || !strings.Contains(err.Error(), "wrong name") {
		t.Fatalf("List error = %v, want a misfiled-name report", err)
	}
}

func TestListFailsOnUnreadableDir(t *testing.T) {
	root := t.TempDir()
	blocker := filepath.Join(root, "file")
	if err := os.WriteFile(blocker, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	s := NewStore(blocker, fixedClock(time.Unix(1000, 0).UTC()))
	if _, err := s.List(); err == nil {
		t.Fatal("List over a file = nil, want error")
	}
}

func TestFilenameWithoutHostFallsBack(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0).UTC()))
	k := Key{Issuer: "https://as.example.com", Resource: "", ClientID: "cid"}
	if err := s.Save(k, testRecord(k)); err != nil {
		t.Fatal(err)
	}
	files := listJSON(t, dir)
	if len(files) != 1 || !strings.HasPrefix(filepath.Base(files[0]), "unknown-") {
		t.Fatalf("files = %v, want one unknown-*.json", files)
	}
}

// TestSaveTightensPreExistingDir guards the promise doc.go makes about the
// directory mode: MkdirAll only applies its mode to directories it creates, so a
// directory left behind by a stray mkdir or a looser umask must be tightened
// before a token is written into it.
func TestSaveTightensPreExistingDir(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "mcp")
	//nolint:gosec // G301: the loose mode is the point of this regression test; Save must tighten it.
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	// Mkdir applies the umask, so assert the starting point is actually loose.
	before, err := os.Stat(dir)
	if err != nil {
		t.Fatal(err)
	}
	if before.Mode()&0o077 == 0 {
		t.Skipf("umask already restricts the dir to %o; nothing to tighten", before.Mode()&os.ModePerm)
	}
	s := NewStore(dir, fixedClock(time.Unix(1000, 0).UTC()))
	k := testKey()
	if err := s.Save(k, testRecord(k)); err != nil {
		t.Fatal(err)
	}
	after, err := os.Stat(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got := after.Mode() & os.ModePerm; got != 0o700 {
		t.Fatalf("dir mode after Save = %o, want 700", got)
	}
}
