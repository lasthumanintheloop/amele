// Package doctor answers one question before a token is spent: can this
// config run on this host, right now? It runs a fixed list of checks - the
// config itself, the environment it references, every provider target's
// reachability and credential, the workspace, the session directory, the
// executables and MCP servers it names, the permission profile against the
// terminal state, and the run lock - and renders one line per check with a
// PASS, WARN or FAIL verdict (issue #13).
//
// The package decides; the CLI wires and prints. Everything that touches the
// host or the network is injectable (Options), so the checks are exercised
// hermetically: an httptest endpoint stands in for a provider, a temp dir for
// the workspace, a func for the TTY.
//
// doctor is a GATE, unlike explain: a FAIL anywhere makes the command exit 1.
// It is the pre-flight for a cron line - "will tonight's run work?" - and a
// pre-flight that reports failure with exit 0 answers nothing a script can
// branch on.
package doctor

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
)

// Status is a check's verdict.
type Status string

// The three verdicts. FAIL is the only one that fails the command; WARN marks
// something the run will survive but the operator should know (an `ask`
// policy that will auto-deny headless, an endpoint that answers but exposes no
// listing to check the key against).
const (
	Pass Status = "PASS"
	Warn Status = "WARN"
	Fail Status = "FAIL"
)

// Check is one verdict with the fact behind it.
type Check struct {
	// Name is the check's short label (config, env, provider, ...).
	Name string
	// Status is the verdict.
	Status Status
	// Detail is the one-line fact: what was checked and what was found.
	Detail string
}

// Report is the ordered list of checks one Run produced.
type Report struct {
	Checks []Check
}

// Failed reports whether any check failed - the exit-1 condition.
func (r Report) Failed() bool {
	for _, c := range r.Checks {
		if c.Status == Fail {
			return true
		}
	}
	return false
}

// Render formats the report: one `  [VERDICT] name: detail` line per check,
// then a closing count. The verdict is padded to a fixed width so the column
// of names lines up, which is what makes a FAIL visible in a cron mail at a
// glance. The output always ends with a newline.
func (r Report) Render() string {
	var b strings.Builder
	var pass, warn, fail int
	for _, c := range r.Checks {
		fmt.Fprintf(&b, "  [%-4s] %s: %s\n", c.Status, c.Name, c.Detail)
		switch c.Status {
		case Pass:
			pass++
		case Warn:
			warn++
		case Fail:
			fail++
		}
	}
	fmt.Fprintf(&b, "%d checks: %d passed, %d warnings, %d failed\n", len(r.Checks), pass, warn, fail)
	return b.String()
}

// Options are the host and network seams of one Run.
type Options struct {
	// Problems are the reasons the config would fail `amele run` that the
	// caller already knows - validation violations, an uncompilable
	// output.schema - one per line. Each becomes a FAIL under the config
	// check; an empty list is the PASS.
	Problems []string
	// Probe checks whether an executable is invocable; nil means
	// exec.LookPath.
	Probe func(name string) error
	// HTTP performs the provider probes; nil means a client bounded by
	// Timeout.
	HTTP *http.Client
	// Timeout bounds one provider probe when HTTP is nil; zero means 10s.
	Timeout time.Duration
	// IsTTY reports whether a human is attached to stdin - the same question
	// the permission layer asks. nil means false, which is the headless
	// reading and the conservative one.
	IsTTY func() bool
	// LockPath is the lock file the run would take (`lock: true`), empty when
	// the config takes none.
	LockPath string
}

// defaultProbeTimeout bounds one provider probe when the caller set none.
const defaultProbeTimeout = 10 * time.Second

// Run performs every check against cfg and returns the report. cfg must have
// been loaded and had its overrides applied; it need not be valid - the
// caller passes what Validate said through Options.Problems, and every other
// check reads the fields as they stand.
func Run(ctx context.Context, cfg *config.Config, opts Options) Report {
	var r Report
	r.add(configCheck(opts.Problems))
	r.add(envCheck(cfg))
	r.Checks = append(r.Checks, providerChecks(ctx, cfg, opts)...)
	r.add(workspaceCheck(cfg))
	r.add(sessionDirCheck(cfg))
	r.Checks = append(r.Checks, executableChecks(cfg, opts.Probe)...)
	r.Checks = append(r.Checks, mcpHTTPChecks(cfg)...)
	r.add(ttyCheck(cfg, opts.IsTTY))
	r.add(lockCheck(cfg, opts.LockPath))
	return r
}

func (r *Report) add(c Check) { r.Checks = append(r.Checks, c) }

// configCheck reports what the loader and validator already found.
func configCheck(problems []string) Check {
	if len(problems) == 0 {
		return Check{"config", Pass, "loads and validates"}
	}
	return Check{"config", Fail, strings.Join(problems, "; ")}
}

// envCheck reports every ${VAR} the config references that is unset, and a
// primary credential that resolved to nothing.
func envCheck(cfg *config.Config) Check {
	if missing := cfg.EnvMissing(); len(missing) > 0 {
		return Check{"env", Fail, "unset: " + strings.Join(missing, ", ")}
	}
	if n := len(cfg.EnvReferenced()); n > 0 {
		return Check{"env", Pass, fmt.Sprintf("%d referenced variable(s) set", n)}
	}
	return Check{"env", Pass, "no environment variables referenced"}
}

// providerChecks probes the primary and every fallback target in chain order,
// one check each, named provider, provider.fallback[0], ...
func providerChecks(ctx context.Context, cfg *config.Config, opts Options) []Check {
	client := opts.HTTP
	if client == nil {
		timeout := opts.Timeout
		if timeout <= 0 {
			timeout = defaultProbeTimeout
		}
		// SECURITY: redirects are not followed. Go keeps x-api-key and
		// x-goog-api-key on a cross-host redirect (they are not Authorization),
		// so a probe that followed one could hand the credential to another
		// host - the same rule the run's clients apply (issue #21). A 3xx is
		// reported as the answer it is.
		client = &http.Client{Timeout: timeout, CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		}}
	}
	checks := []Check{probeTarget(ctx, client, "provider", &cfg.Provider)}
	for i := range cfg.Provider.Fallback {
		name := fmt.Sprintf("provider.fallback[%d]", i)
		checks = append(checks, probeTarget(ctx, client, name, &cfg.Provider.Fallback[i].ProviderConfig))
	}
	return checks
}

// probeTarget asks one target the cheapest question its wire answers with the
// credential: the models listing. What comes back decides the verdict:
//
//   - 2xx: reachable and the key was accepted;
//   - 401/403: reachable, the key was refused (FAIL - tonight's run would be
//     exit 5 at the first turn);
//   - 404/405: reachable, but this endpoint exposes no listing (WARN - a
//     gateway or a self-hosted server; the key could not be checked);
//   - anything else, or no answer at all: FAIL with the cause.
//
// SECURITY: the detail names the identity and the URL's host, never the
// credential; the caller's redactor covers a base_url that carries one.
func probeTarget(ctx context.Context, client *http.Client, name string, p *config.ProviderConfig) Check {
	identity := p.Identity()
	if p.Type == config.ProviderTypeGemini && p.Vertex != nil {
		// Vertex authenticates with a Google OAuth token minted from ADC or a
		// service-account file; obtaining one is the credential flow itself,
		// which this pre-flight does not run. Saying so is more honest than
		// a PASS that checked nothing.
		return Check{name, Warn, identity + ": not probed (the Google credential flow is exercised by the run itself)"}
	}
	if p.APIKey == "" {
		return Check{name, Fail, identity + ": api_key is empty"}
	}
	probeURL, headers, err := probeRequest(p)
	if err != nil {
		return Check{name, Fail, identity + ": " + err.Error()}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, probeURL, nil)
	if err != nil {
		return Check{name, Fail, identity + ": " + err.Error()}
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := client.Do(req)
	if err != nil {
		return Check{name, Fail, fmt.Sprintf("%s: %s unreachable: %v", identity, hostOf(probeURL), unwrapURLError(err))}
	}
	_ = resp.Body.Close()
	return verdict(name, identity, hostOf(probeURL), resp)
}

// verdict maps the probe's answer onto a check: what the status class says
// about the endpoint and the key.
func verdict(name, identity, host string, resp *http.Response) Check {
	switch {
	case resp.StatusCode >= 200 && resp.StatusCode < 300:
		return Check{name, Pass, fmt.Sprintf("%s: %s reachable, key accepted", identity, host)}
	case resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden:
		return Check{name, Fail, fmt.Sprintf("%s: %s reachable, key rejected (HTTP %d)", identity, host, resp.StatusCode)}
	case resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusMethodNotAllowed:
		return Check{name, Warn, fmt.Sprintf("%s: %s reachable; it exposes no models listing, so the key was not checked (HTTP %d)", identity, host, resp.StatusCode)}
	case resp.StatusCode >= 300 && resp.StatusCode < 400:
		return Check{name, Fail, fmt.Sprintf("%s: %s redirects the API (HTTP %d to %s); the credential is not sent on - point base_url at the final host", identity, host, resp.StatusCode, hostOf(resp.Header.Get("Location")))}
	default:
		return Check{name, Fail, fmt.Sprintf("%s: %s answered HTTP %d", identity, host, resp.StatusCode)}
	}
}

// The wire defaults the probe needs, spelled here rather than imported from
// the clients: doctor asks a different endpoint than a run does, and tying it
// to the clients' unexported constants would make a pre-flight package depend
// on request-building code it never calls.
const (
	anthropicDefaultBase = "https://api.anthropic.com"
	anthropicVersion     = "2023-06-01"
	geminiDefaultBase    = "https://generativelanguage.googleapis.com/v1beta"
)

// probeRequest builds the listing URL and the credential header for one
// target's wire.
func probeRequest(p *config.ProviderConfig) (string, map[string]string, error) {
	base := strings.TrimSuffix(p.BaseURL, "/")
	switch p.Type {
	case config.ProviderTypeAnthropic:
		if base == "" {
			base = anthropicDefaultBase
		}
		return base + "/v1/models", map[string]string{"x-api-key": p.APIKey, "anthropic-version": anthropicVersion}, nil
	case config.ProviderTypeGemini:
		if base == "" {
			base = geminiDefaultBase
		} else {
			// The run's client appends the version itself; the probe mirrors
			// that so a custom base_url is addressed the way the run will.
			base += "/v1beta"
		}
		return base + "/models", map[string]string{"x-goog-api-key": p.APIKey}, nil
	default:
		if base == "" {
			return "", nil, errors.New("base_url is empty")
		}
		return base + "/models", map[string]string{"Authorization": "Bearer " + p.APIKey}, nil
	}
}

// hostOf names the endpoint in a detail line: the host alone, never the path
// or the query, which is where a credential-bearing base_url keeps it.
func hostOf(raw string) string {
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" {
		return "endpoint"
	}
	return u.Host
}

// unwrapURLError strips the "Get \"https://...\": " prefix net/http puts on a
// transport error, which would otherwise print the full URL - path and query
// included - into a line that only wants the cause.
func unwrapURLError(err error) error {
	var ue *url.Error
	if errors.As(err, &ue) {
		return ue.Err
	}
	return err
}

// workspaceCheck verifies the sandbox exists and takes a write, by creating
// and removing an empty file: a permission bit can lie about an NFS export,
// and the run's fs_write would find out one turn in.
func workspaceCheck(cfg *config.Config) Check {
	info, err := os.Stat(cfg.Workspace)
	if err != nil {
		return Check{"workspace", Fail, fmt.Sprintf("%s: %v", cfg.Workspace, unwrapPathError(err))}
	}
	if !info.IsDir() {
		return Check{"workspace", Fail, cfg.Workspace + ": not a directory"}
	}
	if err := probeWrite(cfg.Workspace); err != nil {
		status := Fail
		if !cfg.Tools.FS {
			// Nothing the config enables writes there; the run would still
			// use it as the working directory, which needs no write.
			status = Warn
		}
		return Check{"workspace", status, fmt.Sprintf("%s: not writable: %v", cfg.Workspace, unwrapPathError(err))}
	}
	return Check{"workspace", Pass, cfg.Workspace + ": exists and is writable"}
}

// sessionDirCheck verifies the session directory can be created and written,
// creating it the way the run would (0750) when it does not exist yet.
func sessionDirCheck(cfg *config.Config) Check {
	if cfg.SessionDir == "" {
		return Check{"session_dir", Pass, "not set (no session log)"}
	}
	if err := os.MkdirAll(cfg.SessionDir, 0o750); err != nil {
		return Check{"session_dir", Fail, fmt.Sprintf("%s: %v", cfg.SessionDir, unwrapPathError(err))}
	}
	if err := probeWrite(cfg.SessionDir); err != nil {
		return Check{"session_dir", Fail, fmt.Sprintf("%s: not writable: %v", cfg.SessionDir, unwrapPathError(err))}
	}
	return Check{"session_dir", Pass, cfg.SessionDir + ": writable"}
}

// probeWrite creates and removes an empty file in dir. The name carries the
// pid so two doctors on one directory cannot collide, and O_EXCL so an
// existing file of that name is never truncated.
func probeWrite(dir string) error {
	path := filepath.Join(dir, fmt.Sprintf(".amele-doctor-%d", os.Getpid()))
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600) //nolint:gosec // G304: a fixed probe name inside the operator's own directory.
	if err != nil {
		return err
	}
	_ = f.Close()
	return os.Remove(path)
}

// unwrapPathError drops the "open <path>: " prefix so a detail that already
// names the path does not name it twice.
func unwrapPathError(err error) error {
	var pe *os.PathError
	if errors.As(err, &pe) {
		return pe.Err
	}
	return err
}

// executableChecks looks up every executable the config names: subprocess
// tools' command[0] and stdio MCP servers' command[0].
func executableChecks(cfg *config.Config, probe func(string) error) []Check {
	if probe == nil {
		probe = func(name string) error {
			_, err := exec.LookPath(name)
			return err
		}
	}
	var checks []Check
	for _, t := range cfg.Tools.Subprocess {
		if len(t.Command) > 0 && t.Command[0] != "" {
			checks = append(checks, executableCheck("tool "+t.Name, t.Command[0], probe))
		}
	}
	for _, s := range cfg.MCP.Servers {
		if s.Transport.Type == config.MCPTransportStdio && len(s.Transport.Command) > 0 && s.Transport.Command[0] != "" {
			checks = append(checks, executableCheck("mcp "+s.Name, s.Transport.Command[0], probe))
		}
	}
	return checks
}

func executableCheck(name, exe string, probe func(string) error) Check {
	if err := probe(exe); err != nil {
		return Check{name, Fail, fmt.Sprintf("%s: not found (%v)", exe, err)}
	}
	return Check{name, Pass, exe + ": found"}
}

// mcpHTTPChecks reports each HTTP MCP server's address. The server is not
// dialled: an MCP handshake may need an OAuth credential, and `amele explain`
// is the command that connects for real. What a pre-flight can say is whether
// the URL names a host at all.
func mcpHTTPChecks(cfg *config.Config) []Check {
	var checks []Check
	for _, s := range cfg.MCP.Servers {
		if s.Transport.Type != config.MCPTransportHTTP {
			continue
		}
		u, err := url.Parse(s.Transport.URL)
		if err != nil || u.Host == "" {
			checks = append(checks, Check{"mcp " + s.Name, Fail, "url names no host"})
			continue
		}
		checks = append(checks, Check{"mcp " + s.Name, Pass, u.Host + " (not dialled; amele explain connects)"})
	}
	return checks
}

// ttyCheck relates the permission profile to the terminal state: an `ask`
// policy is a question, and with no TTY there is nobody to answer it - the
// permission layer then auto-denies (docs/engineering.md §5.5), which is the
// surprise this check exists to announce before the run.
func ttyCheck(cfg *config.Config, isTTY func() bool) Check {
	asks := cfg.Permissions.Default == config.PolicyAsk
	for _, policy := range cfg.Permissions.Tools {
		asks = asks || policy == config.PolicyAsk
	}
	if !asks {
		return Check{"tty", Pass, "no ask policy; the terminal state does not matter"}
	}
	if isTTY != nil && isTTY() {
		return Check{"tty", Pass, "a terminal is attached; ask policies can be answered"}
	}
	return Check{"tty", Warn, "no terminal on stdin; every ask policy will auto-deny (run from cron, this is what happens)"}
}

// lockCheck verifies the lock file's directory takes a write when lock: true
// is set - the lock is taken before anything else, so an unwritable directory
// fails every run of this config at once.
func lockCheck(cfg *config.Config, lockPath string) Check {
	if !cfg.Lock || lockPath == "" {
		return Check{"lock", Pass, "lock: true is not set"}
	}
	dir := filepath.Dir(lockPath)
	if err := probeWrite(dir); err != nil {
		return Check{"lock", Fail, fmt.Sprintf("%s: cannot create the lock file: %v", dir, unwrapPathError(err))}
	}
	return Check{"lock", Pass, lockPath + ": directory writable"}
}
