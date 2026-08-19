# OAuth client identity

This directory holds the **client-id metadata document** (CIMD) amele presents
to an MCP authorization server that supports RFC 7591-style client identity by
URL instead of dynamic registration.

- Source of truth in-repo: [`client-v1.json`](client-v1.json).
- Served at: `https://amele.work/oauth/client-v1.json`, which is the literal
  `client_id` amele sends (`mcp.CIMDDocumentURL` in
  `internal/mcp/login.go`).

## Two rules

1. **Deploying it is a manual release step.** Nothing in CI publishes this
   file. After changing it - which should be almost never, see below - a human
   copies it to the `amele.work` site and verifies that the URL serves it as
   `application/json`. A login against a CIMD-capable server fails while the
   URL is unreachable, so the deploy is part of shipping, not an afterthought.
2. **The URL is versioned and immutable.** A CIMD document *is* the client
   identity: an authorization server may cache it, and every stored refresh
   token is bound to the client it names. Editing what `client-v1.json` serves
   would silently change that identity under credentials already issued. A new
   client shape therefore gets a NEW file (`client-v2.json`) and a new
   `CIMDDocumentURL` constant, and the old URL keeps serving the old document
   for as long as tokens minted under it may still be refreshed.

## What it says

A public client (`token_endpoint_auth_method: none` - amele ships no client
secret, and a secret in a YAML recipe would be a secret in a git repository),
the authorization-code and refresh-token grants only, and a loopback redirect
(`http://127.0.0.1/callback`; the port is chosen per login and is not part of
the registered URI, per OAuth 2.1 §10.3.3 loopback matching).

Servers that do not support client-id metadata documents need a pre-registered
client instead: `mcp.servers[].auth.client_id` in the config
([docs/mcp.md](../mcp.md#oauth)).
