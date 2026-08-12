package config

import (
	_ "embed" // for the config schema below
)

// configSchemaJSON is the JSON Schema (draft 2020-12) describing this
// package's YAML surface. The file next to this source is the single source
// of truth; the published copy at docs/contracts/config.schema.json is kept
// byte-identical by a test, because go:embed cannot reach outside the package
// directory.
//
// CONTRACT: the schema is public API (docs/engineering.md §7/2) - editors and
// `amele schema` consumers depend on it, and a reflective test pins it to the
// Config struct tree so neither side can drift silently.
//
//go:embed config.schema.json
var configSchemaJSON []byte

// SchemaJSONBytes returns the config JSON Schema as published under
// docs/contracts and printed by `amele schema`. It returns a fresh copy on
// every call, so callers may modify the result freely; the function is safe
// for concurrent use.
func SchemaJSONBytes() []byte {
	return append([]byte(nil), configSchemaJSON...)
}
