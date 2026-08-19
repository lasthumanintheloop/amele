// Package oauthtoken stores OAuth credentials for MCP servers on local disk.
//
// It exists because an MCP server reached over HTTP may require an OAuth
// access token, and a headless agent cannot ask a human for one on every run:
// the token (and the refresh token that renews it) has to outlive the process.
// The package is deliberately the whole persistence layer and nothing else —
// it never speaks HTTP, never mints or refreshes a token, and holds no policy
// about when a token should be renewed. Later slices layer discovery, the
// authorization-code flow and refresh on top of this file format.
//
// What it promises:
//
//   - Credentials live one-per-file under a 0700 directory with 0600 files, so
//     the operating system's own permission check is the access control.
//   - A file is replaced atomically (temp file, fsync, rename, directory
//     fsync). A crash mid-write leaves either the old record or the new one,
//     never a truncated file — refresh-token rotation must not destroy the only
//     usable copy of a credential.
//   - A record is only returned for the exact identity that was asked for: the
//     issuer, resource and client id inside the file are re-checked after
//     decoding, so a filename collision or a hand-edited file cannot hand one
//     server's token to another.
//   - Store.WithLock serializes access to one credential across processes,
//     using an OS lock on a sibling .lock file and re-reading the record after
//     the lock is granted. Two processes that decide to refresh the same
//     credential at the same time therefore perform one refresh, not two.
//
// The lock is opt-in and layered on top rather than hidden inside Load and
// Save: a plain read of a credential must not pay for (or block on) a lock, and
// only the read-modify-write of a refresh needs the exclusion. Bare Save keeps
// its old, weaker promise — a complete file, last rename wins.
package oauthtoken
