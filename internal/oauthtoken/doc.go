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
//
// What it deliberately does not promise: it is not a concurrency primitive.
// Two processes refreshing the same credential at the same time will each write
// a complete, valid file and the last rename wins — single concurrent refresh,
// not exactly-once. Serialization across processes is the job of the lock added
// by a later slice, layered on top of this store rather than hidden inside it.
package oauthtoken
