package oauthtoken

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"
)

// Version is the schema version written into every record. A reader that finds
// a larger value must refuse the file rather than guess at its meaning.
const Version = 1

// ErrNotFound reports that no credential is stored for the requested Key.
// Callers distinguish it with errors.Is; every other error means the store was
// reachable but the record could not be trusted or read.
var ErrNotFound = errors.New("oauth token not found")

// Key identifies one credential: all three fields, always.
//
// The triple is the identity because the same authorization server can issue
// tokens for several resources, and the same resource can be reached with more
// than one client registration; collapsing any pair of them would let one
// server's token be presented to another.
type Key struct{ Issuer, Resource, ClientID string }

// Record is the stored credential (schema Version = 1).
//
// It carries the token endpoint alongside the tokens so a refresh needs no
// second discovery round-trip, and the granted scopes so a caller can tell
// whether a cached credential still covers what it needs.
type Record struct {
	Version       int       `json:"v"`
	Issuer        string    `json:"issuer"`
	Resource      string    `json:"resource"`
	ClientID      string    `json:"client_id"`
	TokenEndpoint string    `json:"token_endpoint"`
	AccessToken   string    `json:"access_token"`
	RefreshToken  string    `json:"refresh_token,omitempty"`
	ExpiresAt     time.Time `json:"expires_at"`
	Scopes        []string  `json:"scopes,omitempty"`
}

// Entry pairs a stored Record with the Key it is filed under. List returns
// these so a caller can report what is on disk without re-deriving keys.
type Entry struct {
	Key    Key
	Record *Record
}

// Store reads and writes credential files inside one directory. The zero value
// is not usable; construct it with NewStore. A Store holds no state beyond its
// directory and clock, so it is safe to share between goroutines, but see the
// package comment for what concurrent writers do and do not get.
type Store struct {
	dir string
	// clock is injected rather than read from time.Now so tests (and later the
	// refresh path, which decides freshness) share one controllable source of
	// time; docs/engineering.md §5.4 forbids calling time.Now directly here.
	clock func() time.Time
}

// NewStore returns a Store over dir, using clock as its source of time. The
// directory is not touched until the first Save; a missing directory reads as
// an empty store.
func NewStore(dir string, clock func() time.Time) *Store {
	return &Store{dir: dir, clock: clock}
}

// Now returns the store's current time. It exists so callers that already hold
// a Store evaluate Record.Fresh against the same injected clock instead of
// reaching for time.Now.
func (s *Store) Now() time.Time { return s.clock() }

// DefaultDir returns the directory credentials live in by default:
// ${XDG_STATE_HOME:-$HOME/.local/state}/amele/mcp. Environment lookup is a
// parameter so the choice is testable and no global state is read.
//
// XDG_STATE_HOME is used only when it is set and non-empty (an empty variable
// means "unset" in the XDG spec). With neither variable set the result is a
// relative path under the working directory: that is a visibly wrong location
// rather than a silent write into an unrelated absolute directory, and the CLI
// layer is expected to report the missing HOME.
func DefaultDir(env func(string) (string, bool)) string {
	if v, ok := env("XDG_STATE_HOME"); ok && v != "" {
		return filepath.Join(v, "amele", "mcp")
	}
	home, _ := env("HOME")
	return filepath.Join(home, ".local", "state", "amele", "mcp")
}

// Load returns the credential stored under k, or an error wrapping ErrNotFound
// when there is none.
//
// A file that decodes but names a different issuer, resource or client id is an
// error, never ErrNotFound: the difference matters because ErrNotFound sends
// the caller into a fresh login, while a mismatch means the file on disk is not
// what it claims to be and must be surfaced to a human.
func (s *Store) Load(k Key) (*Record, error) {
	path := s.path(k)
	//nolint:gosec // G304: the path is derived from the store directory and a hash of the key, never from user-supplied file names.
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("loading token for %s: %w", k.Resource, ErrNotFound)
		}
		return nil, fmt.Errorf("reading token file %s: %w", path, err)
	}
	var r Record
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("decoding token file %s: %w", path, err)
	}
	if err := verify(k, &r); err != nil {
		return nil, fmt.Errorf("token file %s: %w", path, err)
	}
	return &r, nil
}

// verify re-checks the identity fields inside a decoded record against the key
// that was asked for.
//
// SECURITY: the file name carries only 128 bits of a hash, and files can be
// edited or copied by hand. Without this check a collision or a swapped file
// would cause one server's access token to be sent to a different server, which
// is exactly the credential leak the per-key file layout exists to prevent.
func verify(k Key, r *Record) error {
	if r.Version > Version {
		return fmt.Errorf("schema version %d is newer than supported version %d", r.Version, Version)
	}
	if r.Issuer != k.Issuer || r.Resource != k.Resource || r.ClientID != k.ClientID {
		return fmt.Errorf("identity mismatch: file holds issuer=%q resource=%q client_id=%q, want issuer=%q resource=%q client_id=%q",
			r.Issuer, r.Resource, r.ClientID, k.Issuer, k.Resource, k.ClientID)
	}
	return nil
}

// Save writes r as the credential for k, replacing any previous one.
//
// SECURITY: the directory is created 0700 and the file is written 0600 before
// any token byte reaches it, so a credential is never briefly world-readable.
// The write is atomic (temp file, fsync, rename, directory fsync): a crash
// during refresh-token rotation must leave a usable credential, not a truncated
// file, because the rotated refresh token may be the only copy that still works.
func (s *Store) Save(k Key, r *Record) error {
	if err := s.ensureDir(); err != nil {
		return err
	}
	//nolint:gosec // G117: writing the access token to disk is the whole point of this package; the bytes land in a 0600 file inside a 0700 directory.
	data, err := json.Marshal(r)
	if err != nil {
		return fmt.Errorf("encoding token record: %w", err)
	}
	path := s.path(k)
	tmp, err := os.CreateTemp(s.dir, ".tmp-*")
	if err != nil {
		return fmt.Errorf("creating temp token file: %w", err)
	}
	defer func() { _ = os.Remove(tmp.Name()) }() // no-op after a successful rename
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("restricting token file: %w", err)
	}
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("writing token file: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("syncing token file: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("closing token file: %w", err)
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		return fmt.Errorf("installing token file: %w", err)
	}
	// fsync the directory so the rename survives a crash (spec: refresh-token
	// rotation must not lose the only copy of the new token). A directory that
	// cannot be opened or synced is not fatal: the data is already durable in
	// the file itself, only the rename's ordering guarantee is lost.
	if d, err := os.Open(s.dir); err == nil {
		_ = d.Sync()
		_ = d.Close()
	}
	return nil
}

// ensureDir creates the store directory and guarantees its mode is 0700.
//
// SECURITY: os.MkdirAll applies its mode only to directories it creates, so a
// directory left behind by a stray mkdir, a looser umask or another tool would
// keep its old (possibly world-readable) mode and quietly host 0600 token
// files under a directory anyone can list. The chmod is therefore
// unconditional-on-mismatch rather than best-effort, and a directory that
// cannot be tightened aborts the Save: writing a credential into a readable
// directory is worse than failing to write it.
func (s *Store) ensureDir() error {
	if err := os.MkdirAll(s.dir, 0o700); err != nil {
		return fmt.Errorf("creating token dir: %w", err)
	}
	fi, err := os.Stat(s.dir)
	if err != nil {
		return fmt.Errorf("inspecting token dir: %w", err)
	}
	if fi.Mode()&os.ModePerm != 0o700 {
		//nolint:gosec // G302: 0700 is the intended directory mode; gosec reads the constant as a file mode.
		if err := os.Chmod(s.dir, 0o700); err != nil {
			return fmt.Errorf("restricting token dir: %w", err)
		}
	}
	return nil
}

// WithLock serializes access to one credential across processes.
//
// It takes an exclusive lock on the sibling lock file, RE-READS the record
// under that lock and passes it to fn (nil when there is no credential yet).
// The re-read is the whole contract: a caller that waited for the lock may have
// been waiting precisely because another process was refreshing the same
// credential, so anything read before the wait is stale — acting on it would
// spend a refresh token that has already been rotated away.
//
// If fn returns a non-nil record that differs from what was read, it is saved
// before the lock is released; a nil or unchanged record leaves the file
// untouched (no rewrite, no mtime change). WithLock returns the record now in
// effect. If fn returns an error, nothing is written and the error is returned
// wrapped, so errors.Is on the caller's own sentinel still works.
//
// CONTRACT: the lock is on <name>.lock, never on the record file itself. Save
// installs a record by rename, which swaps the inode, so a lock held on the
// record file would stop excluding anyone the moment a refresh completed.
//
// Waiting respects ctx: both flock(2) and LockFileEx block without a deadline,
// so acquisition runs on its own goroutine and this call races it against
// ctx.Done. When ctx wins, a small goroutine stays behind to release and close
// the lock file if and when acquisition eventually succeeds — it holds no
// caller state, cannot double-release (the acquiring goroutine reports exactly
// once over a buffered channel) and ends as soon as the current holder lets go.
// That is the only shutdown path a deadline-less lock allows; the alternative,
// abandoning the descriptor, would leak the lock itself.
func (s *Store) WithLock(ctx context.Context, k Key, fn func(current *Record) (*Record, error)) (*Record, error) {
	if err := ctx.Err(); err != nil {
		return nil, err // never take a lock a cancelled caller cannot use
	}
	if err := s.ensureDir(); err != nil {
		return nil, err
	}
	path := s.lockPath(k)
	// SECURITY: 0600 like the record itself. The lock file holds no secret, but
	// a world-writable one in a 0700 directory would still be an odd artifact,
	// and creating it with the same mode keeps one rule for the directory.
	//nolint:gosec // G304: the path is derived from the store directory and a hash of the key.
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, fmt.Errorf("opening lock file %s: %w", path, err)
	}

	acquired := make(chan error, 1) // buffered: the goroutine must never block on a caller that walked away
	go func() { acquired <- lockExclusive(f) }()

	select {
	case err := <-acquired:
		if err != nil {
			_ = f.Close()
			return nil, err
		}
	case <-ctx.Done():
		go func() {
			if err := <-acquired; err == nil {
				_ = unlockFile(f)
			}
			_ = f.Close()
		}()
		return nil, ctx.Err()
	}
	defer func() {
		_ = unlockFile(f)
		_ = f.Close()
	}()

	current, err := s.Load(k)
	if err != nil {
		if !errors.Is(err, ErrNotFound) {
			return nil, err
		}
		current = nil // an absent credential is a state fn handles, not an error
	}
	next, err := fn(current)
	if err != nil {
		return nil, fmt.Errorf("under credential lock for %s: %w", k.Resource, err)
	}
	if next == nil {
		return current, nil
	}
	// DeepEqual rather than a field-by-field compare: Record is small, acyclic
	// and grows over time, and a forgotten field here would mean a silently
	// dropped write. Skipping the equal case keeps a read-only refresh check
	// from rewriting (and re-fsyncing) the file on every run.
	if reflect.DeepEqual(current, next) {
		return current, nil
	}
	if err := s.Save(k, next); err != nil {
		return nil, err
	}
	return next, nil
}

// lockPath returns the sibling lock file for k: the record path with .json
// swapped for .lock. It is derived from path so the two can never drift apart,
// and List ignores it because it only reads .json entries.
func (s *Store) lockPath(k Key) string {
	return strings.TrimSuffix(s.path(k), ".json") + ".lock"
}

// Delete removes the credential stored under k. It is idempotent: deleting a
// credential that is not there succeeds, so "log out" needs no prior lookup.
func (s *Store) Delete(k Key) error {
	if err := os.Remove(s.path(k)); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("deleting token file: %w", err)
	}
	return nil
}

// List returns every credential in the store, in directory order.
//
// Files that cannot be read or decoded are skipped and reported: the returned
// slice always holds the records that were readable, and a non-nil error joins
// the problems found. Callers listing credentials for a human should print both
// — dropping the good entries because one file is corrupt would make a single
// stray file hide every working login.
func (s *Store) List() ([]Entry, error) {
	names, err := os.ReadDir(s.dir)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, nil // an absent directory is an empty store, not a failure
		}
		return nil, fmt.Errorf("reading token dir: %w", err)
	}
	var (
		entries []Entry
		probs   []error
	)
	for _, e := range names {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		path := filepath.Join(s.dir, e.Name())
		//nolint:gosec // G304: the path is a .json entry read back from the store's own directory listing.
		data, err := os.ReadFile(path)
		if err != nil {
			probs = append(probs, fmt.Errorf("reading token file %s: %w", path, err))
			continue
		}
		var r Record
		if err := json.Unmarshal(data, &r); err != nil {
			probs = append(probs, fmt.Errorf("decoding token file %s: %w", path, err))
			continue
		}
		k := Key{Issuer: r.Issuer, Resource: r.Resource, ClientID: r.ClientID}
		// SECURITY: a listed record must still be filed where its own identity
		// says it belongs, otherwise `list` would show a credential under a name
		// Load would never return.
		if want := s.path(k); want != path {
			probs = append(probs, fmt.Errorf("token file %s is filed under the wrong name for its identity", path))
			continue
		}
		entries = append(entries, Entry{Key: k, Record: &r})
	}
	return entries, errors.Join(probs...)
}

// Fresh reports whether the access token is still usable at now with margin to
// spare. A record with no expiry (zero ExpiresAt) is never fresh: an unknown
// lifetime is treated as expired so the caller refreshes rather than sends a
// token the server may already have rejected.
func (r *Record) Fresh(now time.Time, margin time.Duration) bool {
	if r.ExpiresAt.IsZero() {
		return false
	}
	return r.ExpiresAt.After(now.Add(margin))
}

// CanonicalResource normalizes an MCP server URL into the canonical resource
// identifier used in Key and in protected-resource metadata: lowercase scheme
// and host, no userinfo, no query, no fragment, no trailing slash.
//
// Normalization is not cosmetic: the resource string is half of the cache key
// and is echoed to the authorization server, so two spellings of one server
// must not produce two credentials — nor may a credential be filed under a
// string that carries a secret. Userinfo and query are dropped rather than
// rejected because they routinely appear in copy-pasted URLs; a fragment is
// rejected because the OAuth resource-indicator spec forbids one, and silently
// dropping it would hide a mistake in the config.
func CanonicalResource(raw string) (string, error) {
	u, err := url.Parse(raw)
	if err != nil {
		return "", fmt.Errorf("parsing resource %q: %w", raw, err)
	}
	if u.Fragment != "" || strings.Contains(raw, "#") {
		return "", fmt.Errorf("resource %q must not contain a fragment", raw)
	}
	scheme := strings.ToLower(u.Scheme)
	if scheme != "http" && scheme != "https" {
		return "", fmt.Errorf("resource %q must use http or https", raw)
	}
	if u.Host == "" {
		return "", fmt.Errorf("resource %q has no host", raw)
	}
	// TrimRight, not TrimSuffix: ".../mcp//" must canonicalize to ".../mcp"
	// too, otherwise two spellings of one server would key two credentials.
	path := strings.TrimRight(u.EscapedPath(), "/")
	return scheme + "://" + strings.ToLower(u.Host) + path, nil
}

// path returns the file a key is stored in: <host>-<hex32>.json.
//
// The host prefix is there for humans reading `ls`; identity comes from the
// hash, which covers all three key fields with a separator that cannot appear
// in a URL or client id, so no two distinct keys can be flattened into the same
// input string. 128 bits is far more than enough to make an accidental
// collision impossible, and Load re-verifies the identity anyway.
func (s *Store) path(k Key) string {
	sum := sha256.Sum256([]byte(k.Issuer + "\n" + k.Resource + "\n" + k.ClientID))
	return filepath.Join(s.dir, fmt.Sprintf("%s-%s.json", hostLabel(k.Resource), hex.EncodeToString(sum[:])[:32]))
}

// hostLabel renders the resource's host as a filename-safe label. Everything
// outside [A-Za-z0-9.-] (a port's colon, above all) becomes an underscore so
// the name is legal on every supported platform, including Windows.
func hostLabel(resource string) string {
	host := resource
	if u, err := url.Parse(resource); err == nil && u.Host != "" {
		host = u.Host
	}
	safe := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '.', r == '-':
			return r
		default:
			return '_'
		}
	}, host)
	if safe == "" {
		return "unknown"
	}
	return safe
}
