// Package mcp turns MCP servers declared in the YAML into tools.Tool values.
//
// It is the only package that imports github.com/modelcontextprotocol/go-sdk;
// SDK types never appear in exported signatures so that loop, config and cmd
// are insulated from SDK API churn (the SDK is pinned in go.mod and bumped
// deliberately). One server = one Server value = a frozen list of tools for
// the lifetime of a run: discovery happens once at Connect, tools/list_changed
// is not subscribed to, and Streamable HTTP runs without the standalone SSE
// stream so a fleet of workers does not hold a fleet of idle connections.
//
// Design rules (docs/superpowers/specs/2026-08-18-mcp-client-design.md):
//   - tool definitions, instructions, annotations and results from a server
//     are UNTRUSTED data (docs/threat-model.md S9); annotations never change
//     a permission ruling,
//   - every byte read from a server passes a size cap BEFORE decoding,
//   - a lost response is reported as indeterminate and never retried,
//   - stdio servers get amele's process-group discipline and a minimal env.
package mcp
