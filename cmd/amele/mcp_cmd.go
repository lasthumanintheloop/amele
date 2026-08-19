package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/mcp"
	"github.com/lasthumanintheloop/amele/internal/oauthtoken"
)

// The seams `amele mcp` is tested through. Each has exactly one production
// implementation and no configuration reaches it; they are package variables
// only because the alternative is a test that opens a browser, a real
// terminal, or a socket to an authorization server.
//
// stdinIsTerminal is the TTY gate. It cannot be faked through the execCLI
// harness - a *strings.Reader is not a terminal and no test can make it one -
// so the check is a seam rather than a direct isTerminal call. The production
// default is isTerminal itself, unchanged: the same detector `run` uses for
// the permission prompt.
var (
	stdinIsTerminal  = isTerminal
	loginToServer    = mcp.Login
	revokeCredential = mcp.Revoke
)

// mcpTimeFormat is how an expiry is printed: RFC 3339 in UTC, so two machines
// in different zones report the same credential the same way and the value
// sorts as text.
const mcpTimeFormat = time.RFC3339

// maxMCPReportField caps a server-controlled string (an issuer, a scope) on
// its way to the terminal. Same reason as the approval prompt: this text comes
// from an authorization server, and a report an operator reads must not be
// paintable by it.
const maxMCPReportField = 200

// cmdMCP implements the `amele mcp` credential commands: login, status and
// logout.
//
// CONTRACT (docs/contracts/cli.md):
//   - login is interactive by definition: it needs a REAL terminal on stdin
//     (exit 2 otherwise), it talks on stderr only, and it walks the OAuth
//     servers SEQUENTIALLY in config order - three servers must not race three
//     browser windows.
//   - status is a REPORT, like explain: the table goes to stdout, stderr stays
//     empty, and the exit code is 0 even when there is no credential at all.
//     It never refreshes, never opens a browser and never prints a token.
//   - logout deletes locally no matter what the authorization server says; the
//     RFC 7009 revocation is best effort and a failure is a warning.
//
// Every usage or config error is exit 2, like every other command.
func cmdMCP(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer, env config.LookupEnv) int {
	if hasHelpFlag(args) {
		return printHelp("mcp", stdout, stderr)
	}
	sub, rest, ok := parseMCPArgs(args, stderr)
	if !ok {
		return ExitConfigError
	}
	cfg, err := loadMCPConfig(rest.configPath, env)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return ExitConfigError
	}
	store, err := openTokenStore(env)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return ExitConfigError
	}
	servers, err := oauthServers(cfg, rest.server)
	if err != nil {
		_, _ = fmt.Fprintln(stderr, err)
		return ExitConfigError
	}
	switch sub {
	case "login":
		return mcpLogin(ctx, cfg, servers, store, stdin, stderr, env)
	case "status":
		return mcpStatus(servers, store, stdout)
	default: // "logout"; parseMCPArgs admits nothing else
		return mcpLogout(ctx, servers, store, stderr)
	}
}

// mcpArgs is one `amele mcp` invocation's positional arguments.
type mcpArgs struct {
	configPath string
	// server is the one server named on the command line, or "" for "every
	// server that declares oauth".
	server string
}

// parseMCPArgs validates the subcommand and its arity. It takes no flags at
// all: the credential commands act on the config as written, and the `--set`
// allowlist deliberately excludes every mcp.* key, so an override here could
// only make the CLI point at a server the run would never use.
func parseMCPArgs(args []string, stderr io.Writer) (string, mcpArgs, bool) {
	fail := func() (string, mcpArgs, bool) {
		_, _ = fmt.Fprintln(stderr, usageMCP)
		return "", mcpArgs{}, false
	}
	if len(args) < 2 {
		return fail()
	}
	sub, rest := args[0], args[1:]
	maxArgs := 2 // config plus an optional server name
	switch sub {
	case "login", "logout":
	case "status":
		// status reports on the whole config: naming one server would invite
		// the reader to believe the others were checked and found fine.
		maxArgs = 1
	default:
		return fail()
	}
	if len(rest) > maxArgs {
		return fail()
	}
	resolved, err := resolveConfigArg(rest[0])
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "amele mcp %s: %v\n", sub, err)
		return "", mcpArgs{}, false
	}
	parsed := mcpArgs{configPath: resolved}
	if len(rest) == 2 {
		parsed.server = rest[1]
	}
	return sub, parsed, true
}

// loadMCPConfig loads and validates the config exactly as `run` would, minus
// the overrides. A credential command must not act on a config a run would
// refuse: the server URL it is about to key a credential by is only trustworthy
// once Validate has seen it.
func loadMCPConfig(path string, env config.LookupEnv) (*config.Config, error) {
	cfg, err := config.Load(path, env)
	if err != nil {
		return nil, err
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

// openTokenStore builds the store over the XDG state directory.
//
// A missing HOME (and no XDG_STATE_HOME) is reported here rather than silently
// writing credentials into a relative path under the working directory -
// oauthtoken.DefaultDir documents that fallback as visibly wrong on purpose,
// and this is the layer that knows how to tell a human.
func openTokenStore(env config.LookupEnv) (*oauthtoken.Store, error) {
	if v, ok := env("XDG_STATE_HOME"); !ok || v == "" {
		if home, ok := env("HOME"); !ok || home == "" {
			return nil, errors.New("amele mcp: neither XDG_STATE_HOME nor HOME is set, so there is nowhere to keep credentials")
		}
	}
	return oauthtoken.NewStore(oauthtoken.DefaultDir(env), time.Now), nil
}

// oauthServers resolves which servers a command acts on: the one named, or
// every server that declares `auth: {type: oauth}`, in config order.
//
// Both refusals are config errors and both name what is wrong: a server the
// config does not declare, and a server that declares no oauth block (there is
// nothing to log into, and silently doing nothing would read as success).
func oauthServers(cfg *config.Config, name string) ([]config.MCPServer, error) {
	if name != "" {
		for _, s := range cfg.MCP.Servers {
			if s.Name != name {
				continue
			}
			if s.Auth == nil || s.Auth.Type != config.MCPAuthOAuth {
				return nil, fmt.Errorf("amele mcp: server %q has no auth block", name)
			}
			return []config.MCPServer{s}, nil
		}
		return nil, fmt.Errorf("amele mcp: no such mcp server %q", name)
	}
	var servers []config.MCPServer
	for _, s := range cfg.MCP.Servers {
		if s.Auth != nil && s.Auth.Type == config.MCPAuthOAuth {
			servers = append(servers, s)
		}
	}
	return servers, nil
}

// mcpLogin runs the interactive flow for every selected server, one at a time.
func mcpLogin(ctx context.Context, cfg *config.Config, servers []config.MCPServer,
	store *oauthtoken.Store, stdin io.Reader, stderr io.Writer, env config.LookupEnv) int {
	if len(servers) == 0 {
		_, _ = fmt.Fprintln(stderr, "amele mcp login: no server in this config declares `auth: {type: oauth}`")
		return ExitConfigError
	}
	// SECURITY: a real terminal, checked before anything is opened or spent.
	// The permission system's "EOF is a refusal" tolerance is deliberately NOT
	// reused here (spec §3.1): a cron job with /dev/null on stdin must hear
	// that a login needs a human, not wait for a browser nobody will see.
	if !stdinIsTerminal(stdin) {
		_, _ = fmt.Fprintln(stderr, "amele mcp login needs an interactive terminal (stdin is not a tty)")
		return ExitConfigError
	}
	// The run's redactor: Login mints tokens, and every line it writes to the
	// terminal goes through the same scrubber a run's session log uses, so a
	// token cannot reach the screen through an error message.
	secrets := runSecrets(cfg)
	out := &redactingWriter{w: stderr, redact: secrets.Redact}
	lines := newLineReader(stdin)
	deps := mcp.Deps{
		// cmd is the composition root: the one place allowed to name the real
		// clock and the real random source (docs/engineering.md §5.4).
		Clock:          time.Now,
		Rand:           rand.Float64,
		Env:            env,
		TokenStore:     store,
		RegisterSecret: secrets.Add,
	}
	for _, s := range servers {
		opts := mcp.LoginOptions{
			Stderr:  out,
			Confirm: mcpConfirm(s.Name, lines, stderr),
		}
		rec, err := loginToServer(ctx, s, deps, opts)
		if err != nil {
			_, _ = fmt.Fprintf(stderr, "amele: %s\n", secrets.Redact(err.Error()))
			// Not exit 2: the config was fine, the login did not complete.
			return ExitTaskFailed
		}
		_, _ = fmt.Fprintf(stderr, "mcp login ok: %s (expires %s)\n", s.Name, rec.ExpiresAt.UTC().Format(mcpTimeFormat))
	}
	return ExitOK
}

// mcpConfirm is the question asked before a browser opens. The issuer and the
// resource come from the authorization server's own metadata, so they are
// sanitized on the way to the terminal like every other remote string.
//
// SECURITY: only an explicit y/yes proceeds - a blank line, an unrecognized
// word or EOF declines, the same rule the tool approval prompt follows.
func mcpConfirm(server string, lines *lineReader, stderr io.Writer) func(issuer, resource string) bool {
	return func(issuer, resource string) bool {
		_, _ = fmt.Fprintf(stderr, "amele: log in to mcp server %s at %s for %s? open a browser [y/N] ",
			safeForTerminal(server, maxToolName),
			safeForTerminal(issuer, maxMCPReportField),
			safeForTerminal(resource, maxMCPReportField))
		line, err := lines.ReadLine()
		if err != nil && !errors.Is(err, io.EOF) {
			_, _ = fmt.Fprintf(stderr, "\namele: reading the answer: %v\n", err)
			return false
		}
		_, _ = fmt.Fprintln(stderr)
		switch strings.ToLower(strings.TrimSpace(line)) {
		case "y", "yes":
			return true
		default:
			return false
		}
	}
}

// mcpStatus prints the credential table.
//
// CONTRACT: stdout only, exit 0 always. Every field is a fact about the
// credential - never the credential: a token value must not be recoverable
// from a report an operator pastes into a ticket.
func mcpStatus(servers []config.MCPServer, store *oauthtoken.Store, stdout io.Writer) int {
	if len(servers) == 0 {
		// Said out loud rather than left as empty output: silence would read
		// as "the report failed" to someone who expected a row.
		_, _ = fmt.Fprintln(stdout, "no mcp server in this config declares `auth: {type: oauth}`")
		return ExitOK
	}
	tw := tabwriter.NewWriter(stdout, 0, 0, 2, ' ', 0)
	var problems []error
	now := store.Now()
	for _, s := range servers {
		matches, err := serverCredentials(store, s)
		if err != nil {
			problems = append(problems, err)
		}
		if len(matches) == 0 {
			_, _ = fmt.Fprintf(tw, "%s\tno token\texpires=-\trefresh=-\tscopes=-\tissuer=-\n", s.Name)
			continue
		}
		// One row per stored credential: the same server can hold more than
		// one (a pre-registered client and a discovered one), and folding them
		// into a single row would hide a credential from the person auditing.
		for _, m := range matches {
			_, _ = fmt.Fprintf(tw, "%s\t%s\texpires=%s\trefresh=%s\tscopes=%s\tissuer=%s\n",
				s.Name, credentialState(m.Record, now),
				m.Record.ExpiresAt.UTC().Format(mcpTimeFormat),
				yesNo(m.Record.RefreshToken != ""),
				reportField(strings.Join(m.Record.Scopes, ",")),
				reportField(m.Record.Issuer))
		}
	}
	_ = tw.Flush()
	// Unreadable files go under the table on STDOUT, not to stderr: status is
	// a report, and a report that splits its findings across two streams is
	// one an operator reads half of.
	for _, p := range problems {
		_, _ = fmt.Fprintf(stdout, "problem: %s\n", reportField(p.Error()))
	}
	return ExitOK
}

// credentialState is the one-word verdict a row leads with. It is decided on
// the raw expiry with NO refresh margin: status reports what is stored, and a
// run's jittered margin is a decision about when to refresh, not a fact about
// the credential.
func credentialState(rec *oauthtoken.Record, now time.Time) string {
	if rec.ExpiresAt.After(now) {
		return "ok"
	}
	return "expired"
}

// mcpLogout revokes (best effort) and deletes the credentials of every
// selected server. Messages go to stderr: nothing here is a report to pipe.
func mcpLogout(ctx context.Context, servers []config.MCPServer, store *oauthtoken.Store, stderr io.Writer) int {
	code := ExitOK
	for _, s := range servers {
		matches, err := serverCredentials(store, s)
		if err != nil {
			_, _ = fmt.Fprintf(stderr, "amele: warning: %s\n", reportField(err.Error()))
		}
		if len(matches) == 0 {
			_, _ = fmt.Fprintf(stderr, "mcp logout: %s (no token)\n", s.Name)
			continue
		}
		revoked := false
		for _, m := range matches {
			if m.Record.RevocationEndpoint != "" {
				if err := revokeCredential(ctx, m.Record); err != nil {
					// Best effort by contract: the operator asked to be rid of
					// the credential, and an authorization server that is down
					// or refusing must not be able to keep it on disk.
					_, _ = fmt.Fprintf(stderr, "amele: warning: %s: revoking: %s\n", s.Name, reportField(err.Error()))
				} else {
					revoked = true
				}
			}
			if err := store.Delete(m.Key); err != nil {
				_, _ = fmt.Fprintf(stderr, "amele: %s: %v\n", s.Name, err)
				code = ExitTaskFailed
			}
		}
		_, _ = fmt.Fprintf(stderr, "mcp logout: %s (%s)\n", s.Name, revokedLabel(revoked))
	}
	return code
}

// revokedLabel names what actually happened to the token at the far end.
// "local only" covers both "the server advertised no revocation endpoint" and
// "it did, and telling it failed" - in both cases the credential is gone here
// and may still be live there, which is the fact an operator needs.
func revokedLabel(revoked bool) string {
	if revoked {
		return "revoked"
	}
	return "local only"
}

// serverCredentials resolves the stored credentials for one configured server:
// every record filed under the server's canonical resource, narrowed to the
// configured client_id when the config names one.
//
// It shares MatchCredentials with the run path, so `status` and `logout` see
// exactly the credentials a run would consider - a report that disagreed with
// the runner about what is stored would be worse than no report.
func serverCredentials(store *oauthtoken.Store, s config.MCPServer) ([]oauthtoken.Entry, error) {
	resource, err := oauthtoken.CanonicalResource(s.Transport.URL)
	if err != nil {
		return nil, fmt.Errorf("mcp server %q: %w", s.Name, err)
	}
	matches, listErr := mcp.MatchCredentials(store, resource)
	if s.Auth != nil && s.Auth.ClientID != "" {
		kept := matches[:0]
		for _, m := range matches {
			if m.Key.ClientID == s.Auth.ClientID {
				kept = append(kept, m)
			}
		}
		matches = kept
	}
	return matches, listErr
}

// yesNo renders a boolean the way the status table spells it.
func yesNo(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

// reportField renders a stored, server-supplied string for the terminal: an
// empty value becomes "-" so a column is never blank, and everything else is
// sanitized and clipped.
func reportField(s string) string {
	if s == "" {
		return "-"
	}
	return safeForTerminal(s, maxMCPReportField)
}

// redactingWriter scrubs the run's secrets from everything internal/mcp writes
// to the terminal during a login.
//
// SECURITY: the login flow is the one place in amele where a fresh token comes
// into existence, and the store's own errors quote what they were writing. The
// registry is live (RegisterSecret feeds the same SecretSet), so a token minted
// mid-flow is redacted by the very next line.
type redactingWriter struct {
	w      io.Writer
	redact func(string) string
}

// Write implements io.Writer. It reports the input length on success: the
// redacted text is a different size, and a caller comparing counts would read
// a successful write as a short one.
func (rw *redactingWriter) Write(p []byte) (int, error) {
	if _, err := io.WriteString(rw.w, rw.redact(string(p))); err != nil {
		return 0, err
	}
	return len(p), nil
}
