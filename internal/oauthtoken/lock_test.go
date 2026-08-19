package oauthtoken

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// noopFn is the identity body: it keeps whatever is on disk.
func noopFn(current *Record) (*Record, error) { return current, nil }

func TestWithLockRereadsUnderLock(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0)))
	k := testKey()
	v1 := testRecord(k)
	v1.AccessToken = "v1"
	if err := s.Save(k, v1); err != nil {
		t.Fatalf("Save: %v", err)
	}

	entered := make(chan struct{})
	release := make(chan struct{})
	var wg sync.WaitGroup

	// A holds the lock and replaces the record with v2 while B is queued behind
	// it. B must observe v2: the whole point of WithLock is that the record is
	// re-read after the lock is granted, not before the wait began.
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, err := s.WithLock(context.Background(), k, func(current *Record) (*Record, error) {
			close(entered)
			<-release
			if current.AccessToken != "v1" {
				t.Errorf("A saw access token %q, want v1", current.AccessToken)
			}
			v2 := *current
			v2.AccessToken = "v2"
			return &v2, nil
		})
		if err != nil {
			t.Errorf("A WithLock: %v", err)
		}
	}()

	<-entered
	var seenByB atomic.Value
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, err := s.WithLock(context.Background(), k, func(current *Record) (*Record, error) {
			seenByB.Store(current.AccessToken)
			return nil, nil
		})
		if err != nil {
			t.Errorf("B WithLock: %v", err)
		}
	}()
	close(release)
	wg.Wait()

	if got := seenByB.Load(); got != "v2" {
		t.Fatalf("B observed %v, want v2 (re-read under the lock)", got)
	}
}

func TestWithLockPersistsReturnedRecord(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0)))
	k := testKey()
	if err := s.Save(k, testRecord(k)); err != nil {
		t.Fatalf("Save: %v", err)
	}

	got, err := s.WithLock(context.Background(), k, func(current *Record) (*Record, error) {
		next := *current
		next.AccessToken = "rotated"
		return &next, nil
	})
	if err != nil {
		t.Fatalf("WithLock: %v", err)
	}
	if got.AccessToken != "rotated" {
		t.Fatalf("WithLock returned %q, want rotated", got.AccessToken)
	}
	loaded, err := s.Load(k)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.AccessToken != "rotated" {
		t.Fatalf("stored access token %q, want rotated", loaded.AccessToken)
	}
}

func TestWithLockAbsentRecordIsNil(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0)))
	k := testKey()

	got, err := s.WithLock(context.Background(), k, func(current *Record) (*Record, error) {
		if current != nil {
			t.Errorf("current = %+v, want nil for an absent credential", current)
		}
		return testRecord(k), nil
	})
	if err != nil {
		t.Fatalf("WithLock: %v", err)
	}
	if got == nil || got.AccessToken != "at" {
		t.Fatalf("WithLock returned %+v, want the freshly minted record", got)
	}
	if _, err := s.Load(k); err != nil {
		t.Fatalf("Load after WithLock: %v", err)
	}
}

func TestWithLockNilKeepsCurrent(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0)))
	k := testKey()
	if err := s.Save(k, testRecord(k)); err != nil {
		t.Fatalf("Save: %v", err)
	}
	path := s.path(k)
	before, err := os.ReadFile(path) //nolint:gosec // G304: a path this test just wrote.
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	// Backdate the file so "was it rewritten?" is decided by mtime rather than
	// by a sleep; a rewrite through Save would move it to now.
	old := time.Now().Add(-time.Hour).Truncate(time.Second)
	if err := os.Chtimes(path, old, old); err != nil {
		t.Fatalf("chtimes: %v", err)
	}

	for name, fn := range map[string]func(*Record) (*Record, error){
		"returns nil":       func(*Record) (*Record, error) { return nil, nil },
		"returns unchanged": noopFn,
	} {
		t.Run(name, func(t *testing.T) {
			got, err := s.WithLock(context.Background(), k, fn)
			if err != nil {
				t.Fatalf("WithLock: %v", err)
			}
			if got == nil || got.AccessToken != "at" {
				t.Fatalf("WithLock returned %+v, want the current record", got)
			}
			after, err := os.ReadFile(path) //nolint:gosec // G304: a path this test just wrote.
			if err != nil {
				t.Fatalf("read: %v", err)
			}
			if string(after) != string(before) {
				t.Fatalf("file content changed: %s", after)
			}
			fi, err := os.Stat(path)
			if err != nil {
				t.Fatalf("stat: %v", err)
			}
			if !fi.ModTime().Equal(old) {
				t.Fatalf("file was rewritten (mtime %v, want %v)", fi.ModTime(), old)
			}
		})
	}
}

func TestWithLockPropagatesFnError(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0)))
	k := testKey()
	if err := s.Save(k, testRecord(k)); err != nil {
		t.Fatalf("Save: %v", err)
	}
	sentinel := errors.New("refresh failed")

	_, err := s.WithLock(context.Background(), k, func(current *Record) (*Record, error) {
		next := *current
		next.AccessToken = "must-not-persist"
		return &next, sentinel
	})
	if !errors.Is(err, sentinel) {
		t.Fatalf("err = %v, want %v", err, sentinel)
	}
	loaded, err := s.Load(k)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.AccessToken != "at" {
		t.Fatalf("record was rewritten despite the error: %q", loaded.AccessToken)
	}
	// The lock must have been released even on the error path.
	done := make(chan struct{})
	go func() {
		defer close(done)
		if _, err := s.WithLock(context.Background(), k, noopFn); err != nil {
			t.Errorf("second WithLock: %v", err)
		}
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("lock was not released after fn returned an error")
	}
}

func TestWithLockCtxCancelWhileWaiting(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0)))
	k := testKey()
	if err := s.Save(k, testRecord(k)); err != nil {
		t.Fatalf("Save: %v", err)
	}

	entered := make(chan struct{})
	release := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_, err := s.WithLock(context.Background(), k, func(current *Record) (*Record, error) {
			close(entered)
			<-release
			return nil, nil
		})
		if err != nil {
			t.Errorf("holder WithLock: %v", err)
		}
	}()
	<-entered

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	start := time.Now()
	var called atomic.Bool
	_, err := s.WithLock(ctx, k, func(*Record) (*Record, error) {
		called.Store(true)
		return nil, nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if called.Load() {
		t.Fatal("fn ran even though the lock was never granted")
	}
	if d := time.Since(start); d > 2*time.Second {
		t.Fatalf("cancellation took %v, want prompt", d)
	}
	close(release)
	wg.Wait()
}

// TestWithLockSurvivesRename pins the reason the lock lives in a sibling file:
// Save replaces the record by rename, so a lock taken on the record's inode
// would stop excluding anyone the moment a refresh completed.
func TestWithLockSurvivesRename(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0)))
	k := testKey()
	if err := s.Save(k, testRecord(k)); err != nil {
		t.Fatalf("Save: %v", err)
	}
	// Two sequential calls, each of which rewrites the file (rename).
	for i, tok := range []string{"gen1", "gen2"} {
		tok := tok
		if _, err := s.WithLock(context.Background(), k, func(current *Record) (*Record, error) {
			next := *current
			next.AccessToken = tok
			return &next, nil
		}); err != nil {
			t.Fatalf("WithLock %d: %v", i, err)
		}
	}

	// A third call must still exclude a fourth, even though the record file has
	// been replaced twice since the lock file was created.
	entered := make(chan struct{})
	release := make(chan struct{})
	var inside atomic.Bool
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		if _, err := s.WithLock(context.Background(), k, func(current *Record) (*Record, error) {
			close(entered)
			<-release
			return nil, nil
		}); err != nil {
			t.Errorf("holder WithLock: %v", err)
		}
	}()
	<-entered

	waiterDone := make(chan struct{})
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(waiterDone)
		if _, err := s.WithLock(context.Background(), k, func(current *Record) (*Record, error) {
			inside.Store(true)
			return nil, nil
		}); err != nil {
			t.Errorf("waiter WithLock: %v", err)
		}
	}()

	select {
	case <-waiterDone:
		t.Fatal("waiter entered while the lock was held")
	case <-time.After(150 * time.Millisecond):
	}
	if inside.Load() {
		t.Fatal("waiter ran fn while the lock was held")
	}
	close(release)
	wg.Wait()
	if !inside.Load() {
		t.Fatal("waiter never ran")
	}
}

// TestWithLockAcrossProcesses is the only test that proves the lock is a
// kernel-level, cross-process one: an in-process test could pass on a purely
// in-memory mutex.
func TestWithLockAcrossProcesses(t *testing.T) {
	if testing.Short() {
		t.Skip("compiles and runs a helper binary")
	}
	if runtime.GOOS == "windows" {
		t.Skip("the helper locks with flock(2)")
	}
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0)))
	k := testKey()
	if err := s.Save(k, testRecord(k)); err != nil {
		t.Fatalf("Save: %v", err)
	}

	root, err := filepath.Abs("../..")
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	const hold = 500 * time.Millisecond
	cmd := exec.Command("go", "run", "./internal/oauthtoken/testdata/lockhelper", s.lockPath(k), hold.String()) //nolint:gosec // G204: fixed argv built from test-local paths.
	cmd.Dir = root
	cmd.Stderr = os.Stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("start helper: %v", err)
	}
	defer func() { _ = cmd.Wait() }()

	buf := make([]byte, 16)
	n, err := stdout.Read(buf)
	if err != nil {
		t.Fatalf("reading helper handshake: %v", err)
	}
	if got := strings.TrimSpace(string(buf[:n])); got != "held" {
		t.Fatalf("helper said %q, want held", got)
	}

	start := time.Now()
	if _, err := s.WithLock(context.Background(), k, noopFn); err != nil {
		t.Fatalf("WithLock: %v", err)
	}
	if waited := time.Since(start); waited < hold/2 {
		t.Fatalf("WithLock entered after %v, want to have waited for the helper's %v hold", waited, hold)
	}
}

// TestWithLockPersistsInPlaceMutation pins the aliasing trap: fn is handed the
// record it is expected to update, and mutating it in place then returning the
// same pointer is the idiom a refresh will reach for first. Comparing the
// returned record against that same (already mutated) pointer would report "no
// change" and silently drop a rotated refresh token.
func TestWithLockPersistsInPlaceMutation(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0)))
	k := testKey()
	if err := s.Save(k, testRecord(k)); err != nil {
		t.Fatalf("Save: %v", err)
	}

	got, err := s.WithLock(context.Background(), k, func(current *Record) (*Record, error) {
		current.AccessToken = "rotated-in-place"
		current.RefreshToken = "new-refresh"
		current.ExpiresAt = time.Unix(9000, 0).UTC()
		return current, nil // same pointer, deliberately
	})
	if err != nil {
		t.Fatalf("WithLock: %v", err)
	}
	if got.AccessToken != "rotated-in-place" {
		t.Fatalf("WithLock returned %q", got.AccessToken)
	}
	loaded, err := s.Load(k)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.AccessToken != "rotated-in-place" || loaded.RefreshToken != "new-refresh" {
		t.Fatalf("in-place mutation was not persisted: %+v", loaded)
	}
}

// TestWithLockScopeChangePersists guards the field-wise comparison: a change
// only in the scope slice still has to reach disk.
func TestWithLockScopeChangePersists(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0)))
	k := testKey()
	r := testRecord(k)
	r.Scopes = []string{"a"}
	if err := s.Save(k, r); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if _, err := s.WithLock(context.Background(), k, func(current *Record) (*Record, error) {
		next := *current
		next.Scopes = []string{"a", "b"}
		return &next, nil
	}); err != nil {
		t.Fatalf("WithLock: %v", err)
	}
	loaded, err := s.Load(k)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(loaded.Scopes) != 2 {
		t.Fatalf("scopes not persisted: %+v", loaded.Scopes)
	}
}

// TestWithLockMonotonicTimeIsNotAChange pins that an expiry compared with
// time.Equal (not reflect.DeepEqual) does not read as changed merely because
// one of the two values still carries a monotonic clock reading.
func TestWithLockMonotonicTimeIsNotAChange(t *testing.T) {
	dir := t.TempDir()
	s := NewStore(dir, fixedClock(time.Unix(1000, 0)))
	k := testKey()
	r := testRecord(k)
	now := time.Now() // carries a monotonic reading
	r.ExpiresAt = now
	if err := s.Save(k, r); err != nil {
		t.Fatalf("Save: %v", err)
	}
	path := s.path(k)
	old := time.Now().Add(-time.Hour).Truncate(time.Second)
	if err := os.Chtimes(path, old, old); err != nil {
		t.Fatalf("chtimes: %v", err)
	}
	if _, err := s.WithLock(context.Background(), k, func(current *Record) (*Record, error) {
		next := *current
		next.ExpiresAt = now // same instant, but with the monotonic reading attached
		return &next, nil
	}); err != nil {
		t.Fatalf("WithLock: %v", err)
	}
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if !fi.ModTime().Equal(old) {
		t.Fatal("file was rewritten for an unchanged expiry")
	}
}
