// Package explain renders a human-readable dry-run report of a validated
// config: what the agent may touch, spend and emit - before a single token is
// bought. It answers the reviewer's question "what did I just sign?" for a
// YAML diff without contacting any provider.
//
// The report is a UI surface pinned by a golden test, so its output is
// deterministic by construction: the only maps in the config (permission
// entries) are iterated via sorted keys, and everything else follows the
// declaration order of the YAML.
//
// SECURITY: the report never contains secret values, by two mechanisms.
// provider.api_key is omitted entirely (not masked - omitted). Everything else
// the report echoes from the config (subprocess argv vectors, shell patterns,
// prompts embedded in paths) is printed for its review value, but every value
// ${VAR} interpolation substituted into the config is first classified by
// secretValues: credentials are replaced with "[REDACTED]", ordinary
// parameters are shown. This display rule is LOCAL to explain - internal/
// session's log redactor stays unconditional-by-value, because a session file
// is machine-written and cannot be eyeballed for a bad guess, whereas the
// report exists to be read: an operator pre-flighting a pack that takes its
// model, base_url or timezone from the environment must see which values the
// cron line will actually use, and "***" everywhere made that impossible. The
// residual risk of that trade is named in docs/threat-model.md §4.5: a
// credential in a variable named nothing like one reaches the report. The
// workspace and session_dir paths are printed by design: the operator wrote
// them, and a path review is half the point of a dry run.
package explain

import (
	"fmt"
	"maps"
	"os/exec"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/llm"
	"github.com/lasthumanintheloop/amele/internal/tools"
)

// defaultMaxSchemaRetries mirrors cmd/amele's constant of the same name: the
// feedback-retry budget applied when output.max_schema_retries is unset.
// Duplicated (not imported) because internal packages must not depend on cmd;
// the cmd constant's doc comment points back here.
const defaultMaxSchemaRetries = 2

// overrideMarker is appended to every report line whose value came from the
// command line instead of the file under review.
//
// It says "--set" even for the sugar spellings (--model, -w): they desugar to
// exactly that, and naming the general mechanism keeps one string to grep for
// instead of one per flag.
const overrideMarker = " (overridden via --set)"

// overrides answers "did the command line set this key?" for the line markers.
// The keys are config.SettableKeys() values ("limits.max_turns", ...).
type overrides map[string]bool

// mark returns the suffix for a line reporting key: the marker when the key was
// overridden, the empty string otherwise.
func (o overrides) mark(key string) string {
	if o[key] {
		return overrideMarker
	}
	return ""
}

// redactedMarker replaces a credential value wherever it would otherwise be
// printed. It is spelled out (rather than "***") because the report now shows
// non-secret interpolated values: the reader must be able to tell "this was
// withheld" from "this value happens to look like that".
const redactedMarker = "[REDACTED]"

// secretVarNameRe matches the NAME of an environment variable that names a
// credential.
//
// SECURITY: it is deliberately loose - it matches anywhere in the name and
// case-insensitively, so "MONKEY" is treated as a credential too. Over-
// redacting an ordinary variable costs a line of the report; under-redacting a
// credential costs the credential, in a report whose whole purpose is to be
// pasted into tickets and PR reviews.
var secretVarNameRe = regexp.MustCompile(`(?i)key|token|secret|passw|cred`)

// secretVarName reports whether a variable's name marks it as a credential.
func secretVarName(name string) bool { return secretVarNameRe.MatchString(name) }

// field renders a config-sourced string VALUE - a model name, a URL, a path, a
// policy - as a Go-quoted literal, or as placeholder (unquoted) when it is
// empty.
//
// SECURITY: this is the report's one rule for config-sourced values, and it
// exists because explain renders configs Validate rejected. A newline inside
// any such value forges report rows: `model: "m\n  base_url: https://evil/v1"`
// invented a base_url line, a newline in workspace invented a second
// fs-builtins line. Quoting also disposes of the quieter variants -
// terminal control bytes, leading and trailing whitespace, a value that
// merely looks like the next label. stripTerminalControls cannot cover any of
// this: the report is line-oriented, so it must preserve newlines. Composed
// sentences (problems, warnings) take the other rule, singleLine.
func field(value, placeholder string) string {
	if value == "" {
		return placeholder
	}
	return strconv.Quote(value)
}

// singleLine flattens a composed message - a PROBLEMS or WARNINGS line - onto
// the single line the report gives it, escaping any newline, carriage return
// or tab it carries.
//
// SECURITY: these messages are amele's own sentences, but they quote config
// text inside themselves (a workspace path in a stat error, a tool name in a
// registry error), so an embedded newline would forge a bullet. They cannot
// use field: quoting a whole sentence that already contains quoted fragments
// is unreadable, and the sentence is not a value.
func singleLine(msg string) string {
	return strings.NewReplacer("\n", `\n`, "\r", `\r`, "\t", `\t`).Replace(msg)
}

// MCPToolReport is one row of a server's tool list: what the model would see,
// or what was left out and why.
type MCPToolReport struct {
	// Name is the row's subject: the model-facing name ("<server>__<tool>")
	// for a kept tool, the server-side name for a skipped one.
	Name string
	// Original is the server-side name, set only when Normalized.
	Original string
	// Normalized reports that Name is a rewrite rather than a plain join, so
	// the report can show which server-side tool the model-facing name means.
	Normalized bool
	// Bytes is the size of the published definition (name, description,
	// input schema) - the number the token estimate is charged against.
	Bytes int
	// Hint is the short annotation phrase ("destructive", "read-only") or "".
	// SECURITY: annotations are advisory server-supplied claims; this is a
	// display hint, never a permission decision.
	Hint string
	// Kept says the tool was exposed to the model.
	Kept bool
	// Reason is why a not-Kept tool was left out ("excluded", "not
	// included", "definition too large", "invalid output schema").
	Reason string
}

// MCPServerReport is one declared MCP server as `explain` found it: whether it
// answered, what it called itself, and the toolset (and token bill) it would
// contribute to the run.
//
// CONTRACT: explain reports, run gates. A server that could not be reached
// produces a report with Connected false - never a non-zero exit, even when
// the server is `required: true` and `amele run` would abort on it.
type MCPServerReport struct {
	// Name is the configured server name, the prefix of its tool names.
	Name string
	// Transport is "stdio" or "http", as configured.
	Transport string
	// Target is what was dialled: the URL WITHOUT its query string for http
	// (a query can carry a token), command[0] for stdio.
	Target string
	// Connected says the handshake succeeded.
	Connected bool
	// Error and ErrorClass describe a failed connect (mcp.ErrorClass values:
	// "spawn", "auth", "timeout", ...). Both are empty when Connected.
	// SECURITY: Error is remote text; it is flattened onto one line and runs
	// through the report's redaction like everything else.
	Error, ErrorClass string
	// DurationMS is how long the connect took, successful or not.
	DurationMS int64
	// ProtocolVersion is the MCP version the two sides agreed on.
	ProtocolVersion string
	// ServerName and ServerVersion are what the server called itself -
	// UNTRUSTED display strings, quoted on output.
	ServerName, ServerVersion string
	// Auth is the credential mechanism the config declared for this server
	// ("oauth"), empty when it declared none. AuthStatus is the caller's
	// one-line summary of the stored credential - an expiry, whether a refresh
	// token is present, or the command that would obtain one.
	//
	// SECURITY: AuthStatus is FACTS ABOUT a credential, never the credential.
	// The report is the output an operator pastes into a ticket, so no token
	// value may ever reach it (see cmd/amele's authRow, which builds it).
	Auth, AuthStatus string
	// Tools lists the kept tools first (in registration order), then the
	// skipped ones.
	Tools []MCPToolReport
	// TotalBytes is the size of the KEPT definitions, and EstTokens the
	// estimate derived from it (see EstTokens).
	TotalBytes, EstTokens int
}

// bytesPerToken is the divisor behind the definition-token estimate. It is the
// same rough rule the harness budget uses (docs/engineering.md §8): four bytes
// of English-plus-JSON per token. The number is an ESTIMATE by construction -
// no tokenizer ships in the binary, and a per-provider one would make the
// report depend on the model - so it is printed with "≈".
const bytesPerToken = 4

// EstTokens converts a definition byte count into the token estimate the
// report prints. Exported so the caller assembling MCPServerReport fills the
// field with the same arithmetic the warning threshold is judged against.
func EstTokens(totalBytes int) int { return totalBytes / bytesPerToken }

// mcpTokenWarnThreshold is the estimated definition size, in tokens, above
// which the report tells the operator to trim the toolset.
//
// Tool definitions are re-sent on EVERY request, so an unfiltered server is a
// permanent per-turn tax the operator pays without ever seeing a line item for
// it. 4000 is chosen to sit well above a lean toolset (a handful of tools is a
// few hundred tokens) and below the point where the definitions rival the
// conversation itself.
const mcpTokenWarnThreshold = 4000

// ExecProbe checks whether an executable is invocable on this host: nil error
// means found. A nil ExecProbe passed to Render means
// the real exec.LookPath (defaultProbe). It is injected so the requirements
// report stays deterministic under test (docs/engineering.md §5.4: no direct host
// dependence) instead of depending on what happens to be on the test
// machine's PATH.
type ExecProbe func(name string) error

// defaultProbe is the real host check: exec.LookPath, which resolves both
// bare names (searched on PATH) and path-like values (stat plus exec-bit
// check), so one probe covers both forms a subprocess tool's command[0] can
// take.
func defaultProbe(name string) error {
	_, err := exec.LookPath(name)
	return err
}

// resolveProbe returns probe, or defaultProbe when probe is nil - the single
// place that applies the "nil means real LookPath" contract, so every caller
// of the requirements block agrees on it.
func resolveProbe(probe ExecProbe) ExecProbe {
	if probe == nil {
		return defaultProbe
	}
	return probe
}

// Render returns the dry-run report for a loaded cfg. The returned text always
// ends with a newline and is safe to print verbatim.
//
// CONTRACT: explain reports, run gates. Render describes whatever config it is
// given - including one that cannot run - so cfg need not be valid and reg may
// be nil (the registry could not be built). reg, when present, is the tool
// registry the run would use; it powers the "permission entry matches no tool"
// warning, the one check that needs to know which tools actually exist, and
// that warning is simply skipped without it.
//
// overridePairs is the list of `key=value` overrides config.ApplyOverrides
// applied (nil when there were none), in command-line order. They are echoed
// in an OVERRIDES section and mark the individual lines they changed -
// without that, a dry run of a parametrized invocation would attribute a
// command-line value to the YAML file the reviewer is reading.
//
// problems are the reasons this config would fail `amele run` (validation
// violations, an uncompilable output.schema, undefined ${VAR}s, a registry
// that could not be built), one message per line. They open the report so a
// reader cannot mistake a broken config for a working one; nil prints no
// section at all.
//
// probe checks host executables for the requirements section; nil uses the
// real exec.LookPath.
//
// mcpReports are the results of actually contacting the configured MCP servers
// (nil when there are none, or when the caller chose not to connect): the MCP
// SERVERS section is skipped entirely when the slice is empty. They are passed
// in rather than gathered here because dialling a server is I/O, and this
// package renders.
func Render(cfg *config.Config, reg *tools.Registry, overridePairs, problems []string, probe ExecProbe,
	mcpReports []MCPServerReport) string {
	set := overrides{}
	for _, pair := range overridePairs {
		if key, _, ok := config.SplitOverride(pair); ok {
			set[key] = true
		}
	}

	var b strings.Builder
	problemsSection(&b, problems)
	overridesSection(&b, overridePairs)
	providerSection(&b, cfg, reg, set)
	toolsSection(&b, cfg, set)
	mcpSection(&b, mcpReports)
	requirementsSection(&b, cfg, probe)
	permissionsSection(&b, cfg)
	budgetsSection(&b, cfg, set)
	concurrencySection(&b, cfg)
	outputSection(&b, cfg, set)
	sessionSection(&b, cfg, set)
	warningsSection(&b, cfg, reg, mcpReports)
	// SECURITY: redaction runs over the ASSEMBLED report, not per field, so a
	// section added later can never forget to redact - the same reasoning that
	// puts internal/session's redactor at the single write path.
	return redactSecrets(b.String(), secretValues(cfg))
}

// requirementsReport returns just the requirements block, for the package's
// own tests: it is the one section with enough rules of its own (found/missing
// marks, three independently omitted subsections) to be worth exercising
// without the noise of a full report.
//
// SECURITY: a subprocess command[0] can itself be the product of ${VAR}
// interpolation (e.g. command: ["${TOOL_PATH}"]), so this block goes through
// the same redactSecrets pass Render applies - a test that skipped it would
// stop pinning the leak it was written for.
func requirementsReport(cfg *config.Config, probe ExecProbe) string {
	var b strings.Builder
	requirementsSection(&b, cfg, probe)
	return redactSecrets(b.String(), secretValues(cfg))
}

// problemsSection opens the report with everything that would make `amele run`
// fail. It exists because explain is a REPORT, not a gate: refusing to
// describe a config with an unset ${VAR} or an unreachable workspace withheld
// the report exactly when it was most useful (pre-flighting someone else's
// pack on a fresh host). Stating the problems first keeps that honest - the
// sections below describe a config that cannot run yet, and the reader is told
// so before reading them.
func problemsSection(b *strings.Builder, problems []string) {
	if len(problems) == 0 {
		return
	}
	b.WriteString("PROBLEMS (this config would fail `amele run`)\n")
	for _, p := range problems {
		fmt.Fprintf(b, "  - %s\n", singleLine(p))
	}
	b.WriteString("\n")
}

// mcpSection reports what actually happened when explain dialled the
// configured MCP servers: whether each answered, what it calls itself, which
// of its tools the model would receive, and what those definitions cost.
//
// It exists because an MCP server is the one part of an amele config whose
// contents are NOT in the YAML: the operator writes a URL or a command line
// and the toolset arrives from the other side. A dry run that could not say
// "these 14 tools, one of them destructive, 3.1k tokens per request" would
// leave the largest unreviewed surface in the config unreviewed.
//
// The section is omitted entirely when there are no reports, so a config
// without an mcp: block renders exactly as it did before the feature.
func mcpSection(b *strings.Builder, reports []MCPServerReport) {
	if len(reports) == 0 {
		return
	}
	b.WriteString("MCP SERVERS\n")
	for _, r := range reports {
		// SECURITY: %q throughout. The name and transport come from a config
		// explain renders even when Validate rejected it, and the target,
		// server name and version are (or echo) REMOTE strings - a newline in
		// any of them would forge report rows, which stripTerminalControls
		// cannot prevent in a line-oriented report.
		fmt.Fprintf(b, "  %-14q %-5s %s: %s\n", r.Name, r.Transport,
			field(r.Target, "(unset)"), mcpStatus(r))
		// The credential line sits under the server it belongs to and above
		// its tools, connected or not: a server that failed WITH an auth class
		// is exactly where the reader wants to know whether a token is stored.
		if r.Auth != "" {
			fmt.Fprintf(b, "    auth: %s\n", mcpAuth(r))
		}
		for _, t := range r.Tools {
			fmt.Fprintf(b, "    %-32q %s\n", t.Name, mcpToolState(t))
		}
		if r.Connected {
			fmt.Fprintf(b, "    definitions: %d tools, %d bytes ≈ %d tokens\n",
				mcpKeptCount(r), r.TotalBytes, r.EstTokens)
		}
	}
	b.WriteString("\n")
}

// mcpStatus renders one server's connect outcome. A failure is upper-case for
// the same reason MISSING is: explain exits 0 on a dead server, so this mark
// is the only thing telling the reader the run would lose these tools.
func mcpStatus(r MCPServerReport) string {
	if !r.Connected {
		if r.ErrorClass != "" {
			return fmt.Sprintf("✗ FAILED (%s): %s", r.ErrorClass, singleLine(r.Error))
		}
		return fmt.Sprintf("✗ FAILED: %s", singleLine(r.Error))
	}
	return fmt.Sprintf("✓ connected (%d ms, proto %s, %s %s)", r.DurationMS,
		field(r.ProtocolVersion, "(unstated)"),
		field(r.ServerName, "(unnamed)"), field(r.ServerVersion, "(no version)"))
}

// mcpAuth renders the credential line of one server: the mechanism, plus the
// caller's summary of what is stored for it when there is one.
//
// SECURITY: the summary is flattened onto one line like every other value in
// this section - it is assembled from a stored record whose fields an
// authorization server chose.
func mcpAuth(r MCPServerReport) string {
	if r.AuthStatus == "" {
		return singleLine(r.Auth)
	}
	return fmt.Sprintf("%s (%s)", singleLine(r.Auth), singleLine(r.AuthStatus))
}

// mcpToolState renders one tool row's verdict: kept (with the server's
// advisory annotation and, for a rewritten name, the original) or the reason
// it was left out.
func mcpToolState(t MCPToolReport) string {
	if !t.Kept {
		// The reason is one of amele's own closed-set phrases ("excluded",
		// "definition too large"), not server text, so it reads as a word
		// rather than a quoted value - but it is still flattened, since a
		// caller could pass anything.
		if t.Reason == "" {
			return "- skipped (no reason given)"
		}
		return "- " + singleLine(t.Reason)
	}
	state := "✓ kept"
	if t.Hint != "" {
		state += fmt.Sprintf(" (%s)", t.Hint)
	}
	if t.Normalized {
		// The model-facing name is a rewrite (illegal characters replaced, a
		// hash suffix appended); an operator matching the report against the
		// server's own documentation needs the name it was made from.
		state += fmt.Sprintf(" (was %q)", t.Original)
	}
	return state
}

// mcpKeptCount counts the tools the model would actually receive - the same
// set TotalBytes is charged for.
func mcpKeptCount(r MCPServerReport) int {
	n := 0
	for _, t := range r.Tools {
		if t.Kept {
			n++
		}
	}
	return n
}

// mcpEstTokens totals the definition-token estimate across every connected
// server, the figure mcpTokenWarnThreshold is judged against.
func mcpEstTokens(reports []MCPServerReport) int {
	total := 0
	for _, r := range reports {
		total += r.EstTokens
	}
	return total
}

// requirementsSection reports what the host needs to provide: the
// environment variables the YAML references (by name only - a value may be a
// credential, and the section is a checklist, not a dump), the executables its
// subprocess tools invoke, and the per-tool env allowlists. The first two
// carry found/MISSING marks so the same section serves a clean config (a
// receipt) and a broken one (a checklist).
//
// Each subsection is omitted when empty, and the whole section when all are,
// so a config with none of these leaves the report unchanged by this feature -
// the same "silence when there is nothing to say" rule the rest of the report
// follows (see warningsSection, overridesSection).
func requirementsSection(b *strings.Builder, cfg *config.Config, probe ExecProbe) {
	envNames := cfg.EnvReferenced()
	exes := requiredExecutables(cfg)
	allowlists := envAllowlists(cfg)
	if len(envNames) == 0 && len(exes) == 0 && len(allowlists) == 0 {
		return
	}
	missing := make(map[string]bool, len(cfg.EnvMissing()))
	for _, name := range cfg.EnvMissing() {
		missing[name] = true
	}

	// Upper-case like every other section header: the lower-case spelling made
	// this section read as a subsection of TOOLS, which precedes it.
	b.WriteString("REQUIREMENTS\n")
	if len(envNames) > 0 {
		b.WriteString("  env:\n")
		for _, name := range envNames {
			// Upper-case MISSING for the same reason ENABLED and UNBOUNDED are
			// upper-case elsewhere: explain no longer refuses to report on a
			// config with unset variables, so this mark is the only thing
			// telling the reader the run cannot work yet.
			mark, state := "✓", "set"
			if missing[name] {
				mark, state = "✗", "MISSING"
			}
			// Env names need no quoting: the interpolation regex restricts
			// them to [A-Za-z0-9_].
			fmt.Fprintf(b, "    %-15s %s %s\n", name, mark, state)
		}
	}
	executablesSubsection(b, exes, resolveProbe(probe))
	envAllowlistSubsection(b, allowlists)
	b.WriteString("\n")
}

// requiredExecutables lists every program this config would have to find on
// the host: each subprocess tool's command[0], then each stdio MCP server's.
//
// The MCP servers belong here for the reason the whole subsection exists: an
// `npx`-style server command missing from a cron PATH is the single most
// common way a config that works in a terminal fails unattended, and before
// this the checklist stopped at subprocess tools and said nothing about it.
// Duplicates are kept rather than collapsed: the rows are a checklist, and
// two tools sharing an interpreter is not a fact worth hiding.
func requiredExecutables(cfg *config.Config) []string {
	var exes []string
	// Validate rejects an empty Command, but explain reports on configs
	// Validate rejected too, so both loops guard for it.
	for _, t := range cfg.Tools.Subprocess {
		if len(t.Command) > 0 && t.Command[0] != "" {
			exes = append(exes, t.Command[0])
		}
	}
	for _, s := range cfg.MCP.Servers {
		if s.Transport.Type != config.MCPTransportStdio {
			continue
		}
		if cmd := s.Transport.Command; len(cmd) > 0 && cmd[0] != "" {
			exes = append(exes, cmd[0])
		}
	}
	return exes
}

// executablesSubsection lists each required executable with a found/MISSING
// mark.
func executablesSubsection(b *strings.Builder, exes []string, probe ExecProbe) {
	if len(exes) == 0 {
		return
	}
	b.WriteString("  executables:\n")
	for _, exe := range exes {
		mark, state := "✓", "found"
		if probe(exe) != nil {
			mark, state = "✗", "MISSING"
		}
		// SECURITY: %q, not %s - command[0] is pack-author-controlled and
		// this report is the recommended pre-run audit of an untrusted
		// pack, so control bytes (OSC/ESC) must never reach the terminal
		// raw. Same rule as the TOOLS section's exec %q.
		fmt.Fprintf(b, "    %-25q %s %s\n", exe, mark, state)
	}
}

// envAllowlist pairs a tool name with the environment variables its child
// process is allowed to read.
type envAllowlist struct {
	tool string
	vars []string
}

// envAllowlists collects the declared allowlists in report order: the shell
// first (as in the TOOLS section), then subprocess tools in declaration order.
// Tools without an allowlist are absent - see envAllowlistSubsection.
func envAllowlists(cfg *config.Config) []envAllowlist {
	var out []envAllowlist
	if sh := cfg.Tools.Shell; sh.Enabled && len(sh.Env) > 0 {
		out = append(out, envAllowlist{tool: "shell", vars: sh.Env})
	}
	for _, t := range cfg.Tools.Subprocess {
		if len(t.Env) > 0 {
			out = append(out, envAllowlist{tool: t.Name, vars: t.Env})
		}
	}
	// A stdio MCP server is a child process with an env allowlist exactly like
	// a subprocess tool's, and it is a longer-lived one - it holds the
	// credentials for a whole remote API. The rows are labelled "mcp:<name>"
	// so a reader cannot mistake a server for a tool of the same name.
	for _, s := range cfg.MCP.Servers {
		if s.Transport.Type == config.MCPTransportStdio && len(s.Transport.Env) > 0 {
			out = append(out, envAllowlist{tool: "mcp:" + s.Name, vars: s.Transport.Env})
		}
	}
	return out
}

// envAllowlistSubsection prints which of amele's own environment variables
// each tool's process may read. That is a capability grant - the one control
// standing between a model-driven command and the operator's other
// credentials - so the pre-run audit must show it instead of sending the
// reviewer back to the YAML.
//
// SECURITY: names only, never values. A tool that declares no allowlist gets
// no row at all: an empty row would read as "this tool sees nothing", the
// exact opposite of the truth (an absent allowlist means the child inherits
// amele's whole environment).
func envAllowlistSubsection(b *strings.Builder, lists []envAllowlist) {
	if len(lists) == 0 {
		return
	}
	b.WriteString("  env allowlists (variables the tool's process may read):\n")
	for _, l := range lists {
		// Both the tool name and the variable names are quoted for the same
		// reason command[0] is: they come from the pack author, and a name
		// with spaces (or worse) must stay unambiguous on one line.
		fmt.Fprintf(b, "    %-15q %s\n", l.tool, patternList(l.vars))
	}
}

// secretValues returns the interpolated values the report must not display.
//
// SECURITY: this is explain's display rule - the place that decides a
// substituted value may be shown, subject to one further exemption redaction
// itself applies (minRedactableLen, for values too short to replace without
// corrupting the report). A value is withheld when either
//
//   - the variable fed provider.api_key - it is a credential no matter what it
//     is called, and the same value may appear again in an argv or a shell
//     pattern, which the report does print; or
//   - the variable's NAME says credential (secretVarName: key/token/secret/
//     passw/cred, case-insensitive, anywhere in the name).
//
// Everything else is shown, because a pre-flight that hides which model,
// endpoint or timezone a parametrized pack will use is not a pre-flight. The
// rule is a name heuristic and it can be wrong in one direction only: a
// credential in a variable named nothing like one still reaches the report if
// it is not the API key. Packs must name credential variables accordingly -
// docs/packs.md says so - and internal/session's JSONL redaction, which is not
// a display surface, stays unconditional regardless.
func secretValues(cfg *config.Config) []string {
	// SECURITY: an MCP header value is a COMPOSED string ("Bearer <token>"),
	// so registering the interpolated ${VAR} alone would leave the assembled
	// header readable wherever the report echoes one - today in a server's
	// error text, which is remote input and routinely quotes back the request.
	// config.SensitiveHeaderName decides which headers those are; explain's
	// own secretVarNameRe is deliberately NOT widened to "authorization" and
	// "cookie", because it classifies ENVIRONMENT VARIABLE names, where those
	// two words carry no credential meaning, and a looser rule there would
	// redact ordinary values out of a report whose job is to show them.
	values := cfg.MCPHeaderSecrets()
	for _, b := range cfg.EnvBindings() {
		if b.Value == "" {
			continue // nothing to leak, and an empty replacer pattern corrupts the text
		}
		if b.APIKey || secretVarName(b.Name) {
			values = append(values, b.Value)
		}
	}
	return values
}

// minRedactableLen is the shortest secret value by-value redaction will act
// on.
//
// SECURITY: redaction here is a blind substring replace, so a one- or
// two-character value ("x") rewrites every occurrence of that text anywhere in
// the report - `OPENAI_API_KEY=x` turned "explain exits 7" into
// "e[REDACTED]its 7", which corrupts the audit the operator is reading and
// hides nothing (a value under four characters is guessable offline in
// microseconds, so it is not a credential in any meaningful sense). Four is
// the smallest bound that kills the pathological cases while still covering
// anything a real token could be. This is explain-local: internal/session's
// redactor stays unconditional because a machine-written log is allowed to be
// noisy, while a report exists to be read. The variable's NAME is unaffected -
// requirements still lists it with its ✓ set / ✗ MISSING mark, which is how an
// operator learns what to configure.
const minRedactableLen = 4

// redactSecrets strips anything that could drive the reader's terminal from
// the report, then replaces every secret value in it with redactedMarker.
//
// SECURITY: stripping runs FIRST. Replacing first left the by-value match
// working on text that was about to change: a secret with a terminal-control
// rune wedged into it survives the match and is then stripped back into the
// plain credential, and a secret whose own value carries such a rune stops
// matching once the report is stripped. Matching against already-stripped text
// closes both.
//
// SECURITY: each secret is registered in every spelling the report can print
// it in. The report Go-quotes config-sourced values (field, the %q argv rows),
// so a credential containing a quote, a backslash or a non-printable rune
// appears ESCAPED - `sk-"quoted"` prints as sk-\"quoted\" and no longer equals
// the raw value. The interior of strconv.Quote is registered alongside the raw
// value (and the stripped form of the raw value, for the ordering above) so
// the quoting the report applies for safety cannot defeat the redaction.
//
// SECURITY: the replacements are ordered longest-secret-first (strings.Replacer
// resolves a position by argument order), so registration order cannot decide
// how much leaks - a short secret that prefixes a longer one would otherwise
// consume the prefix and leave the longer secret's tail in the report. The
// ordering holds across the combined raw/escaped/stripped set, not per secret.
func redactSecrets(report string, secrets []string) string {
	report = stripTerminalControls(report)
	ordered := redactionPatterns(secrets)
	pairs := make([]string, 0, len(ordered)*2)
	for _, s := range ordered {
		pairs = append(pairs, s, redactedMarker)
	}
	if len(pairs) > 0 {
		report = strings.NewReplacer(pairs...).Replace(report)
	}
	return report
}

// redactionPatterns expands each secret into every spelling the assembled
// report can carry it in - the raw value, its Go-quoted interior, and its
// terminal-control-stripped form - dropping duplicates and values below
// minRedactableLen, and returning them longest-first.
func redactionPatterns(secrets []string) []string {
	var ordered []string
	seen := make(map[string]bool, len(secrets)*3)
	add := func(s string) {
		// An empty replacer pattern corrupts the text, and minRedactableLen
		// already excludes it; the check is on length alone for that reason.
		if len(s) < minRedactableLen || seen[s] {
			return
		}
		seen[s] = true
		ordered = append(ordered, s)
	}
	for _, s := range secrets {
		add(s)
		// strconv.Quote wraps the escaped text in quotes the report supplies
		// itself, so only the interior is a pattern.
		if quoted := strconv.Quote(s); len(quoted) >= 2 {
			add(quoted[1 : len(quoted)-1])
		}
		add(stripTerminalControls(s))
	}
	// Stable so equal-length patterns keep registration order: the report is
	// golden-tested and must stay byte-identical run to run. Re-sorted on
	// every call, n is a handful of values, not worth caching.
	slices.SortStableFunc(ordered, func(a, b string) int { return len(b) - len(a) })
	return ordered
}

// stripTerminalControls removes every byte that could steer a terminal from
// the assembled report.
//
// SECURITY: explain is the recommended pre-run audit of an UNTRUSTED pack
// (docs/packs.md), and since it became a report rather than a gate it also
// renders configs that fail validation - including a subprocess tool whose
// name violates the tool-name rule precisely because it carries an escape
// sequence. `explain evil-pack/` must not let the pack erase and redraw the
// operator's screen (the attack cmd/amele's safeForTerminal documents for the
// permission prompt). Config-sourced VALUES are Go-quoted by field and the %q
// rows, which already neutralises them; this strip covers what quoting does
// not - the composed PROBLEMS and WARNINGS sentences, the section labels those
// sentences quote config text into, and every row a future section adds - and
// is the defence-in-depth layer under the quoting rather than a substitute for
// it. The strip is total - C0 except the report's own newlines, DEL, C1
// (U+009B is a one-character CSI) and the bidi reordering controls - and runs
// over the finished text, so a section added later cannot forget it. It runs
// BEFORE redactSecrets' replacements: see that function.
func stripTerminalControls(report string) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r == '\n':
			return r
		case r < 0x20, r == 0x7f: // C0 and DEL
			return -1
		case r >= 0x80 && r <= 0x9f: // C1, U+009B (CSI) included
			return -1
		case r >= 0x202a && r <= 0x202e, // bidi embeddings and overrides
			r >= 0x2066 && r <= 0x2069: // bidi isolates
			return -1
		}
		return r
	}, report)
}

// overridesSection heads the report with the command line's own contribution,
// in the order it was given, re-typable as written.
//
// It exists because the per-line markers cannot cover everything: `prompt` and
// `system_prompt_file` change what the model is told, and the report has no
// line for either. Printing the pairs verbatim keeps the audit complete
// ("exactly what would this parametrized run do?") without the report having
// to grow a section per field. Nothing is printed when nothing was overridden,
// so an ordinary `amele explain agent.yaml` is unchanged.
//
// The values pass through Render's redactor with everything else, so an
// interpolated secret that also appears here is replaced by "***".
func overridesSection(b *strings.Builder, pairs []string) {
	var written bool
	for _, pair := range pairs {
		key, value, ok := config.SplitOverride(pair)
		if !ok {
			continue // ApplyOverrides already rejected it; a report never panics
		}
		if !written {
			b.WriteString("OVERRIDES\n")
			written = true
		}
		// Quoted: a value with spaces, an empty value (session_dir=) or a
		// multi-line prompt must stay unambiguous on one line.
		fmt.Fprintf(b, "  --set %s=%q\n", key, value)
	}
	if written {
		b.WriteString("\n")
	}
}

// providerSection reports the model and endpoint. api_key is deliberately
// absent - not masked, absent - so no formatting bug can ever leak it.
func providerSection(b *strings.Builder, cfg *config.Config, reg *tools.Registry, set overrides) {
	b.WriteString("MODEL & PROVIDER\n")
	// "(unset)" is reachable now that explain reports on configs that fail
	// validation: an empty field would print as trailing whitespace, which
	// reads as a rendering bug rather than as the violation PROBLEMS names.
	fmt.Fprintf(b, "  model:           %s%s\n", field(cfg.Model, "(unset)"), set.mark("model"))
	ptype := cfg.Provider.Type
	if ptype == "" {
		// An empty type means openai (the pre-Type default); say so instead of
		// printing an empty field the reviewer would have to know the rule for.
		ptype = config.ProviderTypeOpenAI
	}
	fmt.Fprintf(b, "  provider type:   %s\n", field(ptype, "(unset)"))
	// base_url may stay empty on the two native wires, each of which has ONE
	// official host its client falls back to. Naming the wrong one would tell an
	// operator their run goes somewhere it does not.
	//
	// SECURITY: singleLine on the placeholder, not on the value - field quotes
	// the value, but a placeholder is bare prose and the vertex one embeds the
	// configured location (an unvalidated config reaches this report), so
	// without it a newline there could forge a row.
	fmt.Fprintf(b, "  base_url:        %s\n", field(cfg.Provider.BaseURL, singleLine(defaultHostNote(cfg))))
	vertexRows(b, cfg)
	fmt.Fprintf(b, "  request_timeout: %s\n", durationOrDefault(cfg.Provider.RequestTimeout, "120s"))
	retryRow(b, cfg)
	dialectRow(b, cfg)
	if cfg.Provider.MaxOutputTokens > 0 {
		fmt.Fprintf(b, "  max_output_tokens: %d%s\n",
			cfg.Provider.MaxOutputTokens, set.mark("provider.max_output_tokens"))
	}
	providerMapping(b, cfg, reg, set)
	b.WriteString("\n")
}

// vertexModelSentinel stands in for an unset model while the address is built.
// It is spelled out of characters the client's single-segment path escaping
// leaves untouched (letters and hyphens), so vertexRows can put it back as
// "{model}" by an exact match rather than by guessing at an encoding.
const vertexModelSentinel = "-model-unset-"

// vertexRows reports the two things a Vertex run cannot be reviewed without,
// and which no other row shows: WHERE the request goes, and WITH WHICH
// credential.
//
// Neither is readable from the YAML. The address is assembled from a project
// and a location that also decide the hostname (three different host shapes,
// see llm.VertexTarget), and the credential is either a file the config names
// or the ADC chain amele will walk on the host - a difference an operator
// discovers at 03:00 otherwise. Printing the full path rather than just the
// host also puts the project and the location where they are actually spent:
// the location is a data-residency decision amele will never reroute, so the
// report shows it in the address it commits to.
//
// SECURITY: the auth row prints a MODE and, for a key file, its PATH. Never a
// token, never a byte of the file's contents - `explain` output is what people
// paste into issues. Nothing here reads the file at all.
//
// Nothing is printed on the AI Studio half: no project, no location and no
// Google credential exist there, and rows describing a request this config will
// not send are worse than no rows.
func vertexRows(b *strings.Builder, cfg *config.Config) {
	v := cfg.Provider.Vertex
	if v == nil || !geminiWire(cfg) {
		return
	}
	target := llm.VertexTarget{Project: v.Project, Location: v.Location}
	// An unset model would leave "models/:generateContent", which reads as a
	// rendering bug rather than as the violation PROBLEMS already names. The
	// placeholder keeps the SHAPE of the address - the part this row exists to
	// show - honest about the one segment that is missing.
	model := cfg.Model
	if model == "" {
		model = vertexModelSentinel
	}
	endpoint, err := target.Endpoint(cfg.Provider.BaseURL, model)
	if err != nil {
		// explain reports on configs that FAIL validation (the contract on
		// Render), and an unaddressable project or location is one of them.
		// Saying so beats printing half a URL; PROBLEMS carries the violation
		// itself in the validator's own words.
		endpoint = "(unresolved: " + err.Error() + ")"
	} else if cfg.Model == "" {
		// The braces go on here rather than into Endpoint because the client
		// escapes the model as a SINGLE path segment, which would return them
		// as %7B/%7D. Guarded on the empty model so a config that literally
		// names the sentinel still reports its own address.
		endpoint = strings.Replace(endpoint, "/"+vertexModelSentinel+":", "/{model}:", 1)
	}
	// SECURITY: singleLine, because both values embed config text that reached
	// here without passing validation - a newline in a location would otherwise
	// forge a row in a report someone reads as amele's own words.
	fmt.Fprintf(b, "  vertex endpoint: %s\n", singleLine(endpoint))
	fmt.Fprintf(b, "  vertex auth:     %s\n", singleLine(vertexAuthMode(v.Credentials)))
}

// vertexAuthMode names the credential a Vertex run will authenticate with.
//
// The ADC wording lists the chain in the order the library walks it, because
// "application default credentials" alone does not tell an operator which of
// the three the host will actually answer with - and the fix for a 401 is
// different for each.
func vertexAuthMode(credentials string) string {
	if credentials == "" {
		return "application default credentials " +
			"(GOOGLE_APPLICATION_CREDENTIALS, then gcloud user credentials, then the metadata server)"
	}
	return "service account file " + strconv.Quote(credentials)
}

// retryRow reports the retry policy that will actually apply, the way
// request_timeout above reports its own default.
//
// It states the EFFECTIVE numbers rather than echoing the YAML because the
// zero value is not "no retries": `max_attempts: 0` is spelled "omitted" in
// RetryConfig and yields the client default of 3, which an operator who typed
// the 0 on purpose has no other way to discover. Each half is annotated
// separately, so a config that sets only one of them does not read as if both
// were defaults.
func retryRow(b *strings.Builder, cfg *config.Config) {
	attempts, backoff := "3 attempts (default)", "1s initial backoff (default)"
	if r := cfg.Provider.Retry; r != nil {
		if r.MaxAttempts > 0 {
			attempts = fmt.Sprintf("%d attempts", r.MaxAttempts)
		}
		if d := r.InitialBackoff.Std(); d > 0 {
			backoff = fmt.Sprintf("%s initial backoff", d)
		}
	}
	fmt.Fprintf(b, "  retry:           %s, %s\n", attempts, backoff)
}

// anthropicWire reports whether this config talks to the Anthropic Messages
// API. It is the first question every mapping row asks: on that wire the
// dialect is not consulted at all, and the reasoning knob takes a different
// shape entirely.
func anthropicWire(cfg *config.Config) bool {
	return cfg.Provider.Type == config.ProviderTypeAnthropic
}

// geminiWire reports whether this config talks to the native Gemini
// generateContent API - the third wire family, which like the anthropic one
// consults no dialect and spells every tuning knob its own way.
func geminiWire(cfg *config.Config) bool {
	return cfg.Provider.Type == config.ProviderTypeGemini
}

// defaultHostNote is the base_url placeholder: the host the selected client
// falls back to when the config names none.
//
// The openai-compatible path has no such host - OpenAI, OpenRouter, vLLM and
// Ollama all differ, which is why base_url is required there - so an empty
// value on that wire is a PROBLEMS entry, and this row keeps the phrasing it
// has always had rather than growing a fourth answer for a state validate
// refuses.
func defaultHostNote(cfg *config.Config) string {
	if geminiWire(cfg) {
		if v := cfg.Provider.Vertex; v != nil {
			// The gemini wire has two backends and the vertex one is addressed
			// by location, so naming the AI Studio host here would describe a
			// request this config will never send. This row answers only "what
			// does base_url default to"; the resolved path and the credential
			// are vertexRows' two rows, immediately below it.
			//
			// SECURITY: the location reaches this string unvalidated, so the
			// caller escapes the result - see providerSection's note on the
			// base_url row.
			return "(default: " + llm.VertexTarget{Location: v.Location}.Host() + ")"
		}
		return "(default: generativelanguage.googleapis.com)"
	}
	return "(default: api.anthropic.com)"
}

// dialectRow reports the resolved dialect and, when base_url names a provider
// whose dialect the config did not pick, the hint that says so.
//
// The row is printed only when it carries information: a config that names no
// dialect and points nowhere recognizable keeps the report it had before
// dialects existed.
func dialectRow(b *strings.Builder, cfg *config.Config) {
	if geminiWire(cfg) {
		// No row at all on this wire, in either direction. A dialect here is a
		// validate ERROR (PROBLEMS carries it, in the operator's own words), so
		// echoing the value next to the gemini mapping rows could only suggest
		// it applies to something - and there is no default to report either,
		// because this wire has no dialects to default to.
		return
	}
	if anthropicWire(cfg) {
		// A dialect left behind while switching wires is inert, not wrong.
		// Saying so beats both silence (the operator believes it applies) and
		// a validation error (which would blame a provider this config is not
		// talking to).
		if cfg.Provider.Dialect != "" {
			fmt.Fprintf(b, "  dialect:         %s (ignored: the anthropic wire has no dialects)\n",
				field(cfg.Provider.Dialect, ""))
		}
		return
	}
	hint, hinted := baseURLDialectHint(cfg)
	if cfg.Provider.Dialect == "" && !hinted {
		return
	}
	fmt.Fprintf(b, "  dialect:         %s\n", field(cfg.Provider.Dialect, `"openai" (default)`))
	if hinted {
		fmt.Fprintf(b, "  %s\n", singleLine(hint))
	}
}

// baseURLDialectHint returns the "your base_url looks like X" line, or "" when
// there is nothing to hint at.
//
// CONTRACT: a hint, never a decision. amele does NOT auto-detect the dialect
// from base_url (design doc §"No magic"): a silently chosen dialect would
// reshape every request in a way the YAML file does not show. The report names
// the host, and the operator picks.
func baseURLDialectHint(cfg *config.Config) (string, bool) {
	if anthropicWire(cfg) || geminiWire(cfg) {
		// The CN trio's anthropic-compatible endpoints are a documented setup
		// (docs/providers.md); hinting at a dialect that wire ignores would
		// send the operator to change a field that changes nothing. On the
		// gemini wire the same hint would be worse than useless: a dialect
		// there is refused at validate.
		return "", false
	}
	suggested, known := llm.DialectForBaseURL(cfg.Provider.BaseURL)
	if !known || string(suggested) == cfg.Provider.Dialect {
		return "", false
	}
	return fmt.Sprintf("hint: base_url looks like %s; consider dialect: %s",
		llm.BaseURLHost(cfg.Provider.BaseURL), suggested), true
}

// providerMapping prints what the tuning knobs become on the wire: which field
// carries the output cap, how the reasoning knob is spelled (and rounded), the
// sampling values, what the endpoint does with the raw params keys, and - on
// the gemini wire - which JSON Schema keywords leave each tool definition.
//
// It exists because every one of those answers depends on the target, and the
// operator reading a YAML file cannot see any of them. The block is omitted
// entirely when the config sets no tuning at all and the target has nothing of
// its own to report.
func providerMapping(b *strings.Builder, cfg *config.Config, reg *tools.Registry, set overrides) {
	lines := providerMappingLines(cfg, reg, set)
	if len(lines) == 0 {
		return
	}
	b.WriteString("  provider mapping (the wire fields this config will send):\n")
	for _, line := range lines {
		// SECURITY: every line embeds config text (an effort value, a params
		// key) and explain reports on configs that FAILED validation, so a
		// value carrying a newline must not be able to forge a row.
		fmt.Fprintf(b, "    %s\n", singleLine(line))
	}
}

// providerMappingLines builds the mapping rows in a fixed order: cap, then
// reasoning, then sampling, then the raw params, then the tool schemas.
//
// CONTRACT: not one mapping decision is made here. The cap field comes from
// llm.CapField, the reasoning lines from the same mapping functions the clients
// call (llm.MapReasoning / llm.AnthropicReasoningNotes / llm.GeminiReasoningNotes),
// the unknown-field policy from llm.UnknownFieldPolicy and its two per-wire
// counterparts, and the stripped schema keys from llm.SanitizeGeminiSchema -
// the same function the gemini client sanitizes with - so the report cannot
// promise a request the clients will not send.
//
// set marks the rows whose VALUE came from the command line. It matters more
// here than anywhere else in the report: reasoning.effort, temperature and
// top_p have no row of their own, so a mapping row is the only place their
// value is printed - unmarked, it would be read as the YAML file's.
func providerMappingLines(cfg *config.Config, reg *tools.Registry, set overrides) []string {
	var lines []string
	dialect, known := resolvedDialect(cfg)
	if !known {
		// Every dialect-dependent row is unanswerable. Guessing the default
		// would describe a request amele will never send - the run is refused
		// at validate - so the report says what it cannot say instead.
		lines = append(lines, "provider.dialect is not a known dialect: the wire mapping cannot be reported (see PROBLEMS)")
	}

	if capTokens := cfg.Provider.MaxOutputTokens; capTokens > 0 && known {
		lines = append(lines, fmt.Sprintf("max_output_tokens: %d -> %s: %d%s",
			capTokens, capFieldFor(cfg, dialect), capTokens, set.mark("provider.max_output_tokens")))
	}
	if known {
		lines = append(lines, reasoningMappingLines(cfg, dialect, set)...)
	}
	// Sampling is dialect-INdependent (every openai-wire dialect spells it
	// temperature/top_p and passes the value through), so these rows survive an
	// unknown dialect - but it is not WIRE-independent: the gemini fields live
	// under generationConfig and topP is not spelled top_p there at all.
	sampling := false
	if t := cfg.Provider.Temperature; t != nil {
		lines = append(lines, fmt.Sprintf("temperature: %g -> %s: %g%s",
			*t, samplingFieldFor(cfg, "temperature", "generationConfig.temperature"),
			*t, set.mark("provider.temperature")))
		sampling = true
	}
	if p := cfg.Provider.TopP; p != nil {
		lines = append(lines, fmt.Sprintf("top_p: %g -> %s: %g%s",
			*p, samplingFieldFor(cfg, "top_p", "generationConfig.topP"),
			*p, set.mark("provider.top_p")))
		sampling = true
	}
	// What the TARGET does with those values is not dialect-independent: a
	// value that is accepted and then ignored has to say so next to the row
	// that promises it, or the report promises an effect the run will not have.
	if note := samplingNote(cfg, dialect, sampling, known); note != "" {
		lines = append(lines, note)
	}
	if len(cfg.Provider.Params) > 0 {
		// SECURITY: KEYS only. A params value is arbitrary text from the YAML
		// file and a provider-specific routing key can be a credential; the
		// report is written for sharing, so nothing here may print a value.
		lines = append(lines, "params keys (merged verbatim, values not shown): "+quotedKeys(cfg.Provider.Params))
		if policy := unknownFieldPolicy(cfg, dialect, known); policy != "" {
			// The params rows' caveat, printed with them: the same raw key is
			// a hard 400 on one endpoint and a silent no-op on another.
			lines = append(lines, "unknown request fields: "+policy)
		}
	}
	return append(lines, geminiSchemaLines(cfg, reg)...)
}

// samplingFieldFor names the request field that will carry one sampling knob:
// openai on every dialect and on the Messages API, gemini on the third wire.
//
// It exists for the same reason capFieldFor does - the row must print the field
// the request will actually carry, not the config's own key. On this wire that
// is not cosmetic: `top_p` is not a spelling the Gemini API accepts anywhere,
// and the fields sit inside generationConfig rather than at the body root where
// provider.params merges. A report that echoed the config spelling would hand
// an operator a params key worth a 400 - which is exactly the failure the
// unknown-request-fields row two lines below warns about.
func samplingFieldFor(cfg *config.Config, openai, gemini string) string {
	if geminiWire(cfg) {
		return gemini
	}
	return openai
}

// samplingNote returns the caveat that belongs next to the temperature/top_p
// rows on this target, or "" when the values take effect as sent.
//
// The two wires ask different QUESTIONS of the llm package: the openai wire's
// answer is keyed on the dialect (deepseek ignores sampling while thinking),
// the gemini wire's on the temperature VALUE (Google recommends its default and
// honors everything else). Calling the dialect-keyed one on the gemini path
// would return "" every time - a silent wrong answer, which is why the split is
// made here rather than inside a single function with two meanings.
func samplingNote(cfg *config.Config, dialect llm.Dialect, sampling, known bool) string {
	if !sampling {
		return ""
	}
	if geminiWire(cfg) {
		return llm.GeminiSamplingNote(cfg.Provider.Temperature)
	}
	if !known || anthropicWire(cfg) {
		return ""
	}
	return llm.SamplingNote(dialect, reasoningSpec(cfg))
}

// geminiSchemaLines reports what the tool-schema sanitizer will remove, one row
// per tool, or nothing at all on the two wires that have no sanitizer.
//
// It exists because this is the one mapping amele performs on the MODEL's
// behalf rather than the operator's: Gemini's FunctionDeclaration.parameters is
// an OpenAPI-3.0 subset that answers an unknown keyword with a hard 400 for the
// WHOLE request, so amele strips them (llm.SanitizeGeminiSchema) - and a
// constraint that silently left a tool schema is exactly the kind of change a
// dry run exists to surface. CONTRACT (design doc §"Gemini-specific mechanics"
// 1): nothing is dropped silently.
//
// SECURITY: KEYS and PATHS only, quoted. A schema's values are operator text
// and, for an MCP server's tools, REMOTE text - a description, a default, an
// enum member - so none of them may reach a report written for sharing. The
// quoting also flattens a newline in a remote key, which would otherwise forge
// a row.
//
// A nil registry (one that could not be built - PROBLEMS says why) reports
// nothing rather than guessing at a toolset.
func geminiSchemaLines(cfg *config.Config, reg *tools.Registry) []string {
	if !geminiWire(cfg) || reg == nil {
		return nil
	}
	defs := reg.Defs()
	if len(defs) == 0 {
		return nil
	}
	// Registration order, which is the order the tools travel in: a report an
	// operator diffs between runs must not shuffle.
	lines := []string{"tool schemas: sanitized for the gemini wire (an unsupported JSON Schema keyword or shape is a 400 there)"}
	for _, def := range defs {
		_, stripped := llm.SanitizeGeminiSchema(def.Parameters)
		if len(stripped) == 0 {
			lines = append(lines, fmt.Sprintf("  %s: no keys stripped", field(def.Name, `""`)))
			continue
		}
		for i, path := range stripped {
			stripped[i] = field(path, `""`)
		}
		lines = append(lines, fmt.Sprintf("  %s: stripped %s", field(def.Name, `""`), strings.Join(stripped, ", ")))
	}
	return lines
}

// resolvedDialect parses the config's dialect, reporting whether it is usable.
// On the anthropic and gemini wires the dialect is not consulted, so it always
// resolves (to a value the callers ignore) rather than blocking those wires'
// rows on a leftover value - which on the gemini wire is refused at validate
// anyway, and must not take the mapping report down with it.
func resolvedDialect(cfg *config.Config) (llm.Dialect, bool) {
	if anthropicWire(cfg) || geminiWire(cfg) {
		return llm.DialectOpenAI, true
	}
	dialect, err := llm.ParseDialect(cfg.Provider.Dialect)
	return dialect, err == nil
}

// capFieldFor names the request field that will carry max_output_tokens.
func capFieldFor(cfg *config.Config, dialect llm.Dialect) string {
	switch {
	case anthropicWire(cfg):
		// The Messages API has exactly one spelling and requires it.
		return "max_tokens"
	case geminiWire(cfg):
		// Qualified with its parent object because that is where an operator
		// has to put a `params` key to reach a neighbour of it - and because
		// generationConfig sub-keys are NOT paramable in this slice.
		return "generationConfig.maxOutputTokens"
	}
	return llm.CapField(dialect)
}

// reasoningSpec is what the config asked of the reasoning knob, in the neutral
// shape the llm mapping functions take. A nil block yields the zero spec, which
// means "the provider default stands".
func reasoningSpec(cfg *config.Config) llm.ReasoningSpec {
	r := cfg.Provider.Reasoning
	if r == nil {
		return llm.ReasoningSpec{}
	}
	return llm.ReasoningSpec{Effort: r.Effort, BudgetTokens: r.BudgetTokens}
}

// reasoningMappingLines asks the client's own mapping function what this
// config's reasoning knob becomes, and returns its notes verbatim.
//
// One effort can produce several notes (deepseek sends a thinking object AND a
// rounded reasoning_effort), so the override marker goes on every note that
// actually prints the effort - marking only the first would leave the rest
// looking like they came from the file. Notes rooted in budget_tokens are left
// unmarked: that key is not settable, and a marker there would credit the
// command line with a value only the YAML can carry.
func reasoningMappingLines(cfg *config.Config, dialect llm.Dialect, set overrides) []string {
	r := cfg.Provider.Reasoning
	// An empty block is what `--set provider.reasoning.effort=` leaves behind
	// and means "the provider default" - the same as no block at all, so the
	// report must not claim a reasoning field is on the wire.
	if r == nil || (r.Effort == "" && r.BudgetTokens == 0) {
		return nil
	}
	spec := reasoningSpec(cfg)
	var notes []string
	switch {
	case anthropicWire(cfg):
		notes = llm.AnthropicReasoningNotes(spec)
	case geminiWire(cfg):
		notes = llm.GeminiReasoningNotes(spec)
	default:
		notes = llm.MapReasoning(dialect, spec).Notes
	}
	mark := set.mark("provider.reasoning.effort")
	if mark == "" {
		return notes
	}
	marked := make([]string, len(notes))
	for i, note := range notes {
		marked[i] = note
		if strings.HasPrefix(note, "reasoning.effort:") {
			marked[i] += mark
		}
	}
	return marked
}

// unknownFieldPolicy answers what this target does with a request field it does
// not recognize, or "" when the dialect did not parse and there is no answer.
func unknownFieldPolicy(cfg *config.Config, dialect llm.Dialect, known bool) string {
	if anthropicWire(cfg) {
		return llm.AnthropicUnknownFieldPolicy()
	}
	if geminiWire(cfg) {
		return llm.GeminiUnknownFieldPolicy()
	}
	if !known {
		return ""
	}
	return llm.UnknownFieldPolicy(dialect)
}

// quotedKeys renders a params map's keys as a sorted, quoted list. Sorted so
// the report is deterministic (Go map order would shuffle it between runs) and
// quoted because a YAML key is arbitrary text.
func quotedKeys(params map[string]any) string {
	keys := slices.Sorted(maps.Keys(params))
	for i, k := range keys {
		keys[i] = field(k, `""`)
	}
	return strings.Join(keys, ", ")
}

// toolsSection reports every capability the model would hold: the fs builtins,
// the shell (loudly, when enabled) and each subprocess tool with its full argv
// vector - the exact command line a reviewer must judge.
func toolsSection(b *strings.Builder, cfg *config.Config, set overrides) {
	b.WriteString("TOOLS\n")
	fmt.Fprintf(b, "  workspace: %s%s\n", field(cfg.Workspace, "(unset)"), set.mark("workspace"))
	state := "disabled"
	if cfg.Tools.FS {
		state = "enabled"
	}
	fmt.Fprintf(b, "  fs builtins (fs_read, fs_write, fs_list): %s\n", state)
	shellSubsection(b, cfg.Tools.Shell)
	if len(cfg.Tools.Subprocess) == 0 {
		b.WriteString("  subprocess tools: (none)\n")
	} else {
		b.WriteString("  subprocess tools:\n")
		for _, t := range cfg.Tools.Subprocess {
			args := "no"
			if t.AllowArgs {
				// Upper-case on purpose: allow_args hands the model the argv
				// tail, which changes what the command can do - the reviewer's
				// eye must snag on it.
				args = "YES"
			}
			// SECURITY: %q for the name, not %s. The name is
			// pack-author-controlled and validation no longer stands between
			// it and this renderer (explain reports on configs Validate
			// rejected), so a name carrying a NEWLINE would forge report rows
			// - an invented "shell: disabled" line, an extra tool entry.
			// stripTerminalControls cannot catch that: the report is
			// line-oriented, so it must keep newlines. Quoting is the same
			// rule command[0] and the shell patterns already follow.
			fmt.Fprintf(b, "    - %q: exec %q (timeout %s, model-supplied args: %s)\n",
				t.Name, t.Command, durationOrDefault(t.Timeout, "60s"), args)
		}
	}
	b.WriteString("\n")
}

// shellSubsection reports the shell tool. "ENABLED" is upper-case for the same
// reason allow_args is: it is the single most consequential line in the report.
func shellSubsection(b *strings.Builder, sh config.ShellConfig) {
	if !sh.Enabled {
		b.WriteString("  shell: disabled\n")
		return
	}
	b.WriteString("  shell: ENABLED (model-written command lines run via sh -c)\n")
	fmt.Fprintf(b, "    allow patterns: %s\n", patternList(sh.Allow))
	fmt.Fprintf(b, "    deny patterns:  %s\n", patternList(sh.Deny))
	fmt.Fprintf(b, "    timeout: %s\n", durationOrDefault(sh.Timeout, "60s"))
}

// permissionsSection reports the approval profile, ending with the headless
// fail-safe reminder: the same YAML behaves differently in cron than in a
// terminal, and a dry run is exactly where that must be said.
func permissionsSection(b *strings.Builder, cfg *config.Config) {
	b.WriteString("PERMISSIONS\n")
	fmt.Fprintf(b, "  default policy: %s\n", field(string(cfg.Permissions.Default), "allow (unset)"))
	names := sortedToolNames(cfg.Permissions.Tools)
	if len(names) == 0 {
		b.WriteString("  per-tool overrides: (none)\n")
	} else {
		b.WriteString("  per-tool overrides:\n")
		for _, name := range names {
			// SECURITY: %q for the same reason the subprocess rows quote their
			// name - a permission key is a YAML mapping key nothing validates
			// for newlines, and an unquoted one forges rows here. It also
			// matches how collectWarnings has always printed these keys.
			fmt.Fprintf(b, "    %q: %s\n", name, field(string(cfg.Permissions.Tools[name]), "(empty)"))
		}
	}
	b.WriteString("  note: without a TTY, every \"ask\" policy is auto-denied (headless fail-safe).\n\n")
}

// budgetsSection reports the kill switches. A missing budget is stated loudly
// rather than omitted: for an unattended run, "nothing bounds this" is the
// fact the reviewer most needs to see.
func budgetsSection(b *strings.Builder, cfg *config.Config, set overrides) {
	b.WriteString("BUDGETS\n")
	fmt.Fprintf(b, "  max_turns:  %d%s\n", cfg.Limits.MaxTurns, set.mark("limits.max_turns"))
	if cfg.Limits.MaxTokens > 0 {
		fmt.Fprintf(b, "  max_tokens: %d%s\n", cfg.Limits.MaxTokens, set.mark("limits.max_tokens"))
	} else {
		fmt.Fprintf(b, "  max_tokens: UNBOUNDED (no token budget)%s\n", set.mark("limits.max_tokens"))
	}
	if cfg.Limits.Timeout > 0 {
		fmt.Fprintf(b, "  timeout:    %s%s\n", cfg.Limits.Timeout.Std(), set.mark("limits.timeout"))
	} else {
		fmt.Fprintf(b, "  timeout:    none%s\n", set.mark("limits.timeout"))
	}
	// The guard belongs here rather than with the log: it bounds bytes the
	// MODEL reads, which is context and therefore tokens - a spend, not a
	// disk-space decision. The nil case names the built-in caps instead of
	// staying silent, because "unset" here does not mean "no cap" and a
	// reviewer must be able to read the real numbers off the report.
	if cfg.Limits.MaxToolResultBytes != nil {
		fmt.Fprintf(b, "  max_tool_result_bytes: %d (every tool family, and the framed result)%s\n",
			*cfg.Limits.MaxToolResultBytes, set.mark("limits.max_tool_result_bytes"))
	} else {
		fmt.Fprintf(b, "  max_tool_result_bytes: built-in per-tool caps (fs_read 256 KiB, subprocess/shell 64 KiB per stream, mcp 64 KiB)%s\n",
			set.mark("limits.max_tool_result_bytes"))
	}
	// limits.max_logged_field is deliberately NOT here: it bounds bytes on
	// disk, not what the run may spend, so it is reported with the log it
	// belongs to (sessionSection).
	b.WriteString("\n")
}

// concurrencySection reports what this config lets run at the same time: two
// whole runs of it (`lock`), and two tool calls inside one turn
// (`tools.parallel`). Both are stated in both directions - a disabled lock is
// reported as loudly as an enabled one - because "can this cron line run twice
// at once?" is a question a dry run must answer, and silence would read as "no".
//
// The lock file is named generically (<config>.lock) rather than resolved:
// Render is given the config's content, not the path it was loaded from.
//
// Alone among the reported settings these take no override marker: neither
// `lock` (it left the --set allowlist on 2026-08-12) nor `tools.parallel` (no
// tools.* key is settable, config.SettableKeys) can come from the command
// line, so the values on these lines can only have come from the YAML.
func concurrencySection(b *strings.Builder, cfg *config.Config) {
	b.WriteString("CONCURRENCY\n")
	if cfg.Lock {
		b.WriteString("  lock: enabled (a run started while another holds <config>.lock exits 7)\n")
	} else {
		b.WriteString("  lock: disabled (concurrent runs of this config are allowed)\n")
	}
	b.WriteString("  tool calls in a turn: " + parallelismNote(cfg.Tools) + "\n\n")
}

// parallelismNote describes what happens when one turn asks for several tool
// calls at once.
//
// It belongs in CONCURRENCY rather than TOOLS because it answers the same
// question `lock` does one level down: can two things this config authorizes
// be in flight at the same moment? An operator whose subprocess tool writes a
// shared file needs that answer, and before this row the only way to get it was
// to know that an omitted tools.parallel means true.
func parallelismNote(t config.ToolsConfig) string {
	// The word comes from IsParallel - the loop's own predicate - so the report
	// cannot drift from the behavior; only the parenthetical, which says where
	// the value came from, is decided here.
	if !t.IsParallel() {
		return "sequential (tools.parallel: false)"
	}
	if t.Parallel == nil {
		return "parallel (default)"
	}
	return "parallel (tools.parallel: true)"
}

// outputSection reports whether a schema constrains stdout. The schema body is
// not echoed - it can be large and it is already in the YAML under review;
// what the reviewer needs is the contract (JSON-only stdout) and the retry
// budget behind exit 6.
func outputSection(b *strings.Builder, cfg *config.Config, set overrides) {
	b.WriteString("OUTPUT\n")
	if len(cfg.Output.Schema) == 0 {
		b.WriteString("  schema: none (final answer is unconstrained text)\n\n")
		return
	}
	b.WriteString("  schema: present (stdout carries only schema-valid JSON)\n")
	if cfg.Output.MaxSchemaRetries == 0 {
		fmt.Fprintf(b, "  max_schema_retries: %d (default)%s\n", defaultMaxSchemaRetries, set.mark("output.max_schema_retries"))
	} else {
		fmt.Fprintf(b, "  max_schema_retries: %d%s\n", cfg.Output.MaxSchemaRetries, set.mark("output.max_schema_retries"))
	}
	b.WriteString("\n")
}

// defaultMaxLoggedField mirrors internal/session's clip bound so the dry run
// can name the number an omitted limits.max_logged_field resolves to.
//
// It is a copy, not an import: session does not export the constant, and
// explain must not gain a dependency on the writer package to print one
// number. TestExplainDefaultMatchesSessionDefault pins the two together.
const defaultMaxLoggedField = 8 * 1024

// sessionSection reports where the audit trail goes - or that there is none -
// and, when one is written, what it will contain and whether the run says
// where it went. max_logged_field is reported here rather than under BUDGETS
// because it is the LOG's budget: it bounds bytes on disk, not what the run
// may spend in turns, tokens or seconds.
//
// log_reasoning and print_session_path are reported only when they are on, and
// only when a log is actually written (cmd/amele consults both on the writer it
// just opened, so without session_dir neither does anything). They are the two
// keys `--set` cannot reach, precisely because what a run persists is a
// data-governance decision that belongs in the audited YAML - which makes this
// report the only place an operator pre-flighting someone else's pack can see
// them. Unlike `lock`, whose off state is worth stating out loud, these two
// default to off and off is what a reader assumes; printing "disabled" rows on
// every report would cost eleven goldens two lines each to say nothing.
func sessionSection(b *strings.Builder, cfg *config.Config, set overrides) {
	b.WriteString("SESSION\n")
	if cfg.SessionDir == "" {
		fmt.Fprintf(b, "  session_dir: none (no audit log)%s\n", set.mark("session_dir"))
		// The clip bound is still stated with no log to clip: it is the only
		// one of the three on the --set allowlist, and a dropped row would
		// drop its "(overridden via --set)" marker with it - the report would
		// go silent about a flag the cron line carries.
		maxLoggedFieldRow(b, cfg, set)
		b.WriteString("\n")
		return
	}
	fmt.Fprintf(b, "  session_dir: %s%s\n", field(cfg.SessionDir, "(unset)"), set.mark("session_dir"))
	maxLoggedFieldRow(b, cfg, set)
	if cfg.LogReasoning {
		b.WriteString("  log_reasoning: enabled (the provider's reasoning payload - the model's own " +
			"scratchpad - is written into the log)\n")
	}
	if cfg.PrintSessionPath {
		b.WriteString("  print_session_path: enabled (the run names its session file on stderr)\n")
	}
	b.WriteString("\n")
}

// maxLoggedFieldRow writes the log's own budget: how many bytes of each
// free-text field reach the session file.
//
// All three states are reported because none can be read off the YAML - an
// absent key means 8192, and an explicit 0 means the OPPOSITE of what a 0
// means on a budget line (no bound at all, not "no budget left").
func maxLoggedFieldRow(b *strings.Builder, cfg *config.Config, set overrides) {
	switch v := cfg.Limits.MaxLoggedField; {
	case v == nil:
		fmt.Fprintf(b, "  max_logged_field: %d (default)%s\n", defaultMaxLoggedField, set.mark("limits.max_logged_field"))
	case *v == 0:
		fmt.Fprintf(b, "  max_logged_field: UNBOUNDED (every logged field written whole)%s\n",
			set.mark("limits.max_logged_field"))
	default:
		fmt.Fprintf(b, "  max_logged_field: %d%s\n", *v, set.mark("limits.max_logged_field"))
	}
}

// warningsSection lists everything valid-but-suspicious. Validate deliberately
// accepts these configs (see internal/config's rationale on inert permission
// entries); explain is where they get said out loud instead of failing a cron
// run. The order is fixed - sorted permission entries, then shell, tokens,
// session, reasoning - so the section is golden-testable.
func warningsSection(b *strings.Builder, cfg *config.Config, reg *tools.Registry, mcpReports []MCPServerReport) {
	b.WriteString("WARNINGS\n")
	ws := collectWarnings(cfg, reg, mcpReports)
	if len(ws) == 0 {
		b.WriteString("  (none)\n")
		return
	}
	for _, w := range ws {
		fmt.Fprintf(b, "  - %s\n", singleLine(w))
	}
}

// collectWarnings computes the warning list in its fixed order.
func collectWarnings(cfg *config.Config, reg *tools.Registry, mcpReports []MCPServerReport) []string {
	var ws []string
	// Without a registry (it could not be built - that is one of the problems
	// reported at the top) there is nothing to check permission entries
	// against, and "matches no tool" would be a verdict on missing evidence.
	if reg != nil {
		registered := reg.Names()
		for _, name := range sortedToolNames(cfg.Permissions.Tools) {
			// Glob-aware, like the permission lookup itself: `github__*` is a
			// perfectly healthy entry when github tools are registered, and
			// warning "typo?" about it taught operators to ignore the warning.
			if !matchesAnyTool(name, registered) {
				ws = append(ws, fmt.Sprintf("permission entry %q matches no tool - typo?", name))
			}
		}
	}
	if sh := cfg.Tools.Shell; sh.Enabled && len(sh.Allow) == 0 && len(sh.Deny) == 0 {
		ws = append(ws, "tools.shell is enabled with no allow or deny patterns: any command the model writes will run")
	}
	if cfg.Limits.MaxTokens == 0 {
		ws = append(ws, "limits.max_tokens is 0: no token budget bounds this run")
	}
	if cfg.SessionDir == "" {
		ws = append(ws, "session_dir is not set: no session log (audit trail) will be written")
	}
	// SECURITY: the one config key that widens what lands on disk beyond what
	// the redactor can reason about. Redaction (internal/session) still runs
	// on the payload, but it matches known secret VALUES, and a scratchpad
	// paraphrases: "the key ends in 7f" survives every value match. Warned only
	// when a log is actually written - without session_dir the key is inert,
	// and the warning above already says nothing is recorded.
	if cfg.LogReasoning && cfg.SessionDir != "" {
		ws = append(ws, "log_reasoning is enabled: the session log will contain the model's reasoning; "+
			"redaction matches secret values and cannot catch a paraphrased one")
	}
	// Tool definitions ride along on every request, so an unfiltered server is
	// a recurring cost the operator never sees itemised. The threshold is
	// judged on the TOTAL across servers: three servers of 1500 tokens each
	// bill the same as one of 4500.
	if est := mcpEstTokens(mcpReports); est > mcpTokenWarnThreshold {
		ws = append(ws, fmt.Sprintf(
			"mcp definitions ≈ %d tokens; consider tools.include to trim", est))
	}
	return ws
}

// durationOrDefault renders a config duration, naming the consumer's default
// when the field is unset - "0" would read as "disabled", which is wrong for
// fields where zero means "use the default".
func durationOrDefault(d config.Duration, def string) string {
	if d == 0 {
		return fmt.Sprintf("default (%s)", def)
	}
	return d.Std().String()
}

// patternList renders shell allow/deny patterns, quoting each so a pattern
// containing spaces or commas stays unambiguous.
func patternList(ps []string) string {
	if len(ps) == 0 {
		return "(none)"
	}
	quoted := make([]string, len(ps))
	for i, p := range ps {
		quoted[i] = fmt.Sprintf("%q", p)
	}
	return strings.Join(quoted, ", ")
}

// matchesAnyTool reports whether a permission entry - exact name or glob,
// exactly the vocabulary internal/perm accepts - matches at least one
// registered tool.
func matchesAnyTool(entry string, registered []string) bool {
	for _, name := range registered {
		if tools.GlobMatch(entry, name) {
			return true
		}
	}
	return false
}

// sortedToolNames returns the permission map's keys sorted. Sorted iteration
// is what makes the per-tool list and the typo warnings deterministic
// (docs/engineering.md §6: golden files cannot tolerate map order).
func sortedToolNames(m map[string]config.Policy) []string {
	return slices.Sorted(maps.Keys(m))
}
