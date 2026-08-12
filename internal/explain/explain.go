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
func Render(cfg *config.Config, reg *tools.Registry, overridePairs, problems []string, probe ExecProbe) string {
	set := overrides{}
	for _, pair := range overridePairs {
		if key, _, ok := config.SplitOverride(pair); ok {
			set[key] = true
		}
	}

	var b strings.Builder
	problemsSection(&b, problems)
	overridesSection(&b, overridePairs)
	providerSection(&b, cfg, set)
	toolsSection(&b, cfg, set)
	requirementsSection(&b, cfg, probe)
	permissionsSection(&b, cfg)
	budgetsSection(&b, cfg, set)
	concurrencySection(&b, cfg)
	outputSection(&b, cfg, set)
	sessionSection(&b, cfg, set)
	warningsSection(&b, cfg, reg)
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
	subs := cfg.Tools.Subprocess
	allowlists := envAllowlists(cfg)
	if len(envNames) == 0 && len(subs) == 0 && len(allowlists) == 0 {
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
	executablesSubsection(b, subs, resolveProbe(probe))
	envAllowlistSubsection(b, allowlists)
	b.WriteString("\n")
}

// executablesSubsection lists each subprocess tool's command[0] with a
// found/MISSING mark.
func executablesSubsection(b *strings.Builder, subs []config.SubprocessTool, probe ExecProbe) {
	if len(subs) == 0 {
		return
	}
	b.WriteString("  executables:\n")
	for _, t := range subs {
		if len(t.Command) == 0 || t.Command[0] == "" {
			// Validate rejects an empty Command, but explain reports on
			// configs Validate rejected too, so the guard has to be here.
			continue
		}
		exe := t.Command[0]
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
	var values []string
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
func providerSection(b *strings.Builder, cfg *config.Config, set overrides) {
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
	// Validate only lets base_url stay empty on the anthropic path, where the
	// client falls back to the official endpoint.
	fmt.Fprintf(b, "  base_url:        %s\n", field(cfg.Provider.BaseURL, "(default: api.anthropic.com)"))
	fmt.Fprintf(b, "  request_timeout: %s\n", durationOrDefault(cfg.Provider.RequestTimeout, "120s"))
	if cfg.Provider.MaxOutputTokens > 0 {
		fmt.Fprintf(b, "  max_output_tokens: %d\n", cfg.Provider.MaxOutputTokens)
	}
	b.WriteString("\n")
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
	b.WriteString("\n")
}

// concurrencySection reports whether two runs of this config can overlap.
// It is stated in both directions - a disabled lock is reported as loudly as
// an enabled one - because "can this cron line run twice at once?" is a
// question a dry run must answer, and silence would read as "no".
//
// The lock file is named generically (<config>.lock) rather than resolved:
// Render is given the config's content, not the path it was loaded from.
//
// Alone among the reported settings this one takes no override marker: `lock`
// left the --set allowlist on 2026-08-12 (config.SettableKeys), so the value
// on this line can only have come from the YAML.
func concurrencySection(b *strings.Builder, cfg *config.Config) {
	b.WriteString("CONCURRENCY\n")
	if cfg.Lock {
		b.WriteString("  lock: enabled (a run started while another holds <config>.lock exits 7)\n\n")
		return
	}
	b.WriteString("  lock: disabled (concurrent runs of this config are allowed)\n\n")
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

// sessionSection reports where the audit trail goes - or that there is none.
func sessionSection(b *strings.Builder, cfg *config.Config, set overrides) {
	b.WriteString("SESSION\n")
	if cfg.SessionDir == "" {
		fmt.Fprintf(b, "  session_dir: none (no audit log)%s\n\n", set.mark("session_dir"))
		return
	}
	fmt.Fprintf(b, "  session_dir: %s%s\n\n", field(cfg.SessionDir, "(unset)"), set.mark("session_dir"))
}

// warningsSection lists everything valid-but-suspicious. Validate deliberately
// accepts these configs (see internal/config's rationale on inert permission
// entries); explain is where they get said out loud instead of failing a cron
// run. The order is fixed - sorted permission entries, then shell, tokens,
// session - so the section is golden-testable.
func warningsSection(b *strings.Builder, cfg *config.Config, reg *tools.Registry) {
	b.WriteString("WARNINGS\n")
	ws := collectWarnings(cfg, reg)
	if len(ws) == 0 {
		b.WriteString("  (none)\n")
		return
	}
	for _, w := range ws {
		fmt.Fprintf(b, "  - %s\n", singleLine(w))
	}
}

// collectWarnings computes the warning list in its fixed order.
func collectWarnings(cfg *config.Config, reg *tools.Registry) []string {
	var ws []string
	// Without a registry (it could not be built - that is one of the problems
	// reported at the top) there is nothing to check permission entries
	// against, and "matches no tool" would be a verdict on missing evidence.
	if reg != nil {
		registered := reg.Names()
		for _, name := range sortedToolNames(cfg.Permissions.Tools) {
			if !slices.Contains(registered, name) {
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

// sortedToolNames returns the permission map's keys sorted. Sorted iteration
// is what makes the per-tool list and the typo warnings deterministic
// (docs/engineering.md §6: golden files cannot tolerate map order).
func sortedToolNames(m map[string]config.Policy) []string {
	return slices.Sorted(maps.Keys(m))
}
