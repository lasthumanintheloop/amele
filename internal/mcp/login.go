package mcp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os/exec"
	"runtime"
	"slices"
	"strings"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/oauthtoken"
	"github.com/modelcontextprotocol/go-sdk/auth"
	"github.com/modelcontextprotocol/go-sdk/oauthex"
	"golang.org/x/oauth2"
)

// CIMDDocumentURL is the versioned, immutable client-id metadata document
// amele publishes; docs/oauth/client-v1.json is its source of truth in-repo.
//
// CONTRACT: the URL is versioned and never edited in place. A client id
// metadata document IS the client identity, so changing what this URL serves
// would silently change the identity every stored refresh token is bound to;
// a new client shape gets client-v2.json and a new constant.
const CIMDDocumentURL = "https://amele.work/oauth/client-v1.json"

// probeBody is the request Login sends to the MCP endpoint to harvest the 401
// challenge. It is a well-formed JSON-RPC ping so a server that is NOT
// protected answers something recognizable instead of a parse error.
const probeBody = `{"jsonrpc":"2.0","id":1,"method":"ping"}`

// LoginOptions carries the interactive seams. Every message goes to Stderr;
// nothing here may write to stdout (pipe purity contract).
type LoginOptions struct {
	// OpenBrowser launches the operator's browser; nil uses the platform
	// default (xdg-open/open/start). A failure is reported and survivable:
	// the URL is already on Stderr.
	OpenBrowser func(url string) error
	// Stderr receives every human-facing line of the flow. nil discards.
	Stderr io.Writer
	// Confirm asks before the browser opens; shown the sanitized issuer and
	// resource. nil = auto-yes (amele mcp login already implies consent).
	Confirm func(issuer, resource string) bool
}

// Login runs the full OAuth authorization-code flow for one server and
// persists the credential. It REQUIRES interactivity by contract; callers
// gate on a real TTY before calling.
//
// The flow is: unauthenticated probe of the MCP endpoint -> discovery of the
// protected resource and its authorization server -> PKCE capability and
// client-identity checks -> operator confirmation -> loopback redirect
// listener -> the SDK's authorization code handler -> a record saved under the
// store's cross-process lock. Nothing is written before the token exchange
// succeeds, so a refused or abandoned login leaves the store untouched.
//
// SECURITY: discovery is performed with the hardened OAuth client (no
// redirects, bounded bodies, bounded time), and every server-supplied string
// that reaches the terminal passes through safeParam first.
func Login(ctx context.Context, cfg config.MCPServer, deps Deps, opts LoginOptions) (*oauthtoken.Record, error) {
	stderr := opts.Stderr
	if stderr == nil {
		stderr = io.Discard
	}
	if err := checkLoginTarget(cfg, deps); err != nil {
		return nil, err
	}
	rec, err := runLogin(ctx, cfg, deps, opts, stderr)
	if err != nil {
		// One wrapper for the whole flow: every step below names what it was
		// doing, and the server name is what the operator needs in front.
		return nil, fmt.Errorf("mcp login %s: %w", cfg.Name, err)
	}
	if rec.RefreshToken == "" {
		_, _ = fmt.Fprintf(stderr, "warning: %s issued no refresh token; this credential expires for good at %s\n",
			cfg.Name, rec.ExpiresAt.UTC().Format("2006-01-02 15:04:05Z"))
	}
	return rec, nil
}

// runLogin is the flow itself, split out so Login stays a readable contract
// (and within the cyclomatic budget).
func runLogin(ctx context.Context, cfg config.MCPServer, deps Deps, opts LoginOptions, stderr io.Writer) (*oauthtoken.Record, error) {
	resource, err := oauthtoken.CanonicalResource(cfg.Transport.URL)
	if err != nil {
		return nil, err
	}
	client := newOAuthClient()
	req, resp, err := probeUnauthorized(ctx, client, cfg.Transport.URL)
	if err != nil {
		return nil, err
	}
	// Authorize takes ownership of the body; closing twice is a no-op, and
	// this covers every path that returns before Authorize runs.
	defer func() { _ = resp.Body.Close() }()

	disc, err := discoverAuthServer(ctx, client, resp.Header, cfg.Transport.URL)
	if err != nil {
		return nil, err
	}
	if err := checkDiscovery(disc, cfg, resource); err != nil {
		return nil, err
	}
	if opts.Confirm != nil && !opts.Confirm(safeParam(disc.issuer), safeParam(disc.resource)) {
		return nil, errors.New("declined")
	}

	listener, err := newLoopbackListener()
	if err != nil {
		return nil, err
	}
	defer listener.close()

	flow := &loginFlow{cfg: cfg, stderr: stderr, listener: listener, open: opts.OpenBrowser}
	handler, err := flow.newHandler()
	if err != nil {
		return nil, err
	}
	if err := handler.Authorize(ctx, req, resp); err != nil {
		return nil, err
	}
	if flow.token == nil {
		// Authorize returning nil without ever building a token source would
		// mean the SDK changed shape under us; refusing beats persisting a
		// half-known credential.
		return nil, errors.New("the authorization flow produced no token")
	}
	rec, err := flow.record(disc, resource, deps.TokenStore.Now())
	if err != nil {
		return nil, err
	}
	if err := persistLogin(ctx, deps, rec); err != nil {
		return nil, err
	}
	return rec, nil
}

// checkLoginTarget rejects a server Login cannot act on before any network
// traffic happens.
func checkLoginTarget(cfg config.MCPServer, deps Deps) error {
	if cfg.Auth == nil || cfg.Auth.Type != config.MCPAuthOAuth {
		return fmt.Errorf("mcp login %s: server has no `auth: {type: oauth}` block", cfg.Name)
	}
	if cfg.Transport.Type != config.MCPTransportHTTP {
		return fmt.Errorf("mcp login %s: oauth is only available for http servers", cfg.Name)
	}
	if deps.TokenStore == nil {
		return fmt.Errorf("mcp login %s: %w", cfg.Name, errNoTokenStore)
	}
	return nil
}

// probeUnauthorized sends one unauthenticated request to the MCP endpoint and
// returns the 401 that carries the challenge.
//
// Login drives the handler directly rather than connecting an MCP session: a
// login must work even against a server amele could not otherwise talk to
// (protocol mismatch, an initialize that needs scopes we do not have yet), and
// the 401 is the only thing discovery actually needs.
func probeUnauthorized(ctx context.Context, client *http.Client, rawURL string) (*http.Request, *http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, rawURL, strings.NewReader(probeBody))
	if err != nil {
		return nil, nil, fmt.Errorf("building the probe request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	resp, err := client.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("probing %s: %w", rawURL, err)
	}
	if resp.StatusCode != http.StatusUnauthorized {
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		return nil, nil, fmt.Errorf("the server answered %s instead of 401; it is not asking for oauth", resp.Status)
	}
	return req, resp, nil
}

// asDiscovery is what Login learns about the authorization server before it
// hands the flow to the SDK. The SDK performs the same discovery internally
// but exposes none of it, and every field here is needed either for a
// spec-mandated refusal or for the stored record.
type asDiscovery struct {
	// issuer is the authorization server identifier; it is half the store key.
	issuer string
	// resource is the PRM's own `resource` value - the RFC 8707 identifier the
	// token will be bound to.
	resource string
	// tokenEndpoint is recorded so a run can refresh without discovery.
	tokenEndpoint string
	// revocationEndpoint reaches Record.RevocationEndpoint so `amele mcp
	// logout` can hand the credential back; an empty value means the server
	// advertises no RFC 7009 endpoint.
	revocationEndpoint string
	// codeChallengeMethods is what the PKCE refusal is decided on.
	codeChallengeMethods []string
	// cimdSupported reports whether the server accepts a client id metadata
	// document as the client identity.
	cimdSupported bool
}

// discoverAuthServer resolves the protected resource metadata and then the
// authorization server metadata, following the MCP discovery order.
func discoverAuthServer(ctx context.Context, client *http.Client, header http.Header, mcpURL string) (*asDiscovery, error) {
	challenges, err := oauthex.ParseWWWAuthenticate(header[http.CanonicalHeaderKey("WWW-Authenticate")])
	if err != nil {
		return nil, fmt.Errorf("parsing the WWW-Authenticate header: %w", err)
	}
	prm, err := fetchPRM(ctx, client, challenges, mcpURL)
	if err != nil {
		return nil, err
	}
	issuer, err := pickAuthServer(prm.AuthorizationServers)
	if err != nil {
		return nil, err
	}
	asm, err := auth.GetAuthServerMetadata(ctx, issuer, client)
	if err != nil {
		return nil, fmt.Errorf("fetching authorization server metadata for %s: %w", safeParam(issuer), err)
	}
	if asm == nil {
		// The SDK would fall back to guessed endpoints here. amele will not:
		// without metadata there is no way to confirm S256 PKCE support, and
		// the MCP authorization spec makes that a MUST.
		return nil, fmt.Errorf("authorization server %s publishes no metadata, so PKCE support cannot be confirmed", safeParam(issuer))
	}
	return &asDiscovery{
		issuer:               asm.Issuer,
		resource:             prm.Resource,
		tokenEndpoint:        asm.TokenEndpoint,
		revocationEndpoint:   asm.RevocationEndpoint,
		codeChallengeMethods: asm.CodeChallengeMethodsSupported,
		cimdSupported:        asm.ClientIDMetadataDocumentSupported,
	}, nil
}

// fetchPRM tries the protected resource metadata locations in the order the
// MCP specification mandates: the challenge's own pointer first, then the
// path-inserted well-known, then the root one.
func fetchPRM(ctx context.Context, client *http.Client, challenges []oauthex.Challenge, mcpURL string) (*oauthex.ProtectedResourceMetadata, error) {
	var last error
	for _, cand := range prmCandidates(challenges, mcpURL) {
		prm, err := oauthex.GetProtectedResourceMetadata(ctx, cand.metadataURL, cand.resource, client)
		if err != nil {
			last = err
			continue
		}
		if prm == nil || len(prm.AuthorizationServers) == 0 {
			last = errors.New("protected resource metadata names no authorization server")
			continue
		}
		return prm, nil
	}
	if last == nil {
		last = errors.New("no protected resource metadata found")
	}
	// The fallback the SDK offers (treat the MCP server's own root as the
	// authorization server) is deliberately NOT taken: it would send amele's
	// authorization request to a host that never claimed to be an
	// authorization server.
	return nil, fmt.Errorf("discovering protected resource metadata: %w", last)
}

// prmCandidate is one place protected resource metadata may live, paired with
// the resource identifier that document must claim.
type prmCandidate struct{ metadataURL, resource string }

// prmCandidates enumerates the discovery locations for one MCP URL.
func prmCandidates(challenges []oauthex.Challenge, mcpURL string) []prmCandidate {
	var out []prmCandidate
	for _, c := range challenges {
		if u := c.Params["resource_metadata"]; u != "" {
			out = append(out, prmCandidate{metadataURL: u, resource: mcpURL})
			break
		}
	}
	ru, err := url.Parse(mcpURL)
	if err != nil {
		return out
	}
	mu := *ru
	mu.Path = "/.well-known/oauth-protected-resource/" + strings.TrimLeft(ru.Path, "/")
	out = append(out, prmCandidate{metadataURL: mu.String(), resource: mcpURL})
	mu.Path = "/.well-known/oauth-protected-resource"
	root := *ru
	root.Path = ""
	out = append(out, prmCandidate{metadataURL: mu.String(), resource: root.String()})
	return out
}

// pickAuthServer resolves the authorization server a PRM points at.
//
// It takes entry [0] and nothing else, because that is exactly what the SDK
// does inside Authorize. Scanning past a non-https [0] for a usable entry
// would be worse than useless: amele's PKCE, CIMD and issuer checks - and the
// stored record - would describe one server while the flow itself ran against
// another, so a login could report success while filing a credential no run
// can ever use. A [0] that is not an absolute https URL is therefore a
// refusal, not a skip.
func pickAuthServer(servers []string) (string, error) {
	if len(servers) == 0 {
		return "", errors.New("protected resource metadata lists no authorization server")
	}
	first := servers[0]
	u, err := url.Parse(first)
	if err != nil || u.Scheme != "https" || u.Host == "" {
		return "", fmt.Errorf("protected resource metadata names %s as its authorization server, which is not an absolute https url",
			safeParam(first))
	}
	return first, nil
}

// checkDiscovery applies the refusals that must happen before a browser opens.
func checkDiscovery(disc *asDiscovery, cfg config.MCPServer, resource string) error {
	// Spec MUST: no silent downgrade to `plain`, and no hoping that a server
	// which advertises nothing supports S256 anyway.
	if !slices.Contains(disc.codeChallengeMethods, "S256") {
		return fmt.Errorf("authorization server %s does not advertise S256 PKCE support", safeParam(disc.issuer))
	}
	// The endpoint is checked with the run path's own rule, at login time,
	// while the operator is still watching: a credential whose token endpoint
	// a run would refuse is not worth storing.
	if err := checkTokenEndpoint(disc.tokenEndpoint); err != nil {
		return fmt.Errorf("authorization server %s: %w", safeParam(disc.issuer), err)
	}
	if cfg.Auth.ClientID == "" && !disc.cimdSupported {
		return fmt.Errorf("authorization server %s supports neither client id metadata documents nor anonymous clients; set auth.client_id for this server",
			safeParam(disc.issuer))
	}
	// The token is bound to the PRM's resource (RFC 8707), while a run looks
	// its credential up by the canonical form of the CONFIGURED url. If those
	// two disagree the credential would be unfindable, or refreshable only
	// with a resource the authorization server never issued it for, so the
	// mismatch is reported here rather than at the first refresh.
	prmResource, err := oauthtoken.CanonicalResource(disc.resource)
	if err != nil {
		return fmt.Errorf("protected resource metadata resource %q: %w", safeParam(disc.resource), err)
	}
	if prmResource != resource {
		return fmt.Errorf("the server declares resource %s but the config names %s; point transport.url at the declared resource",
			safeParam(prmResource), safeParam(resource))
	}
	return nil
}

// loginFlow holds the state one authorization run threads between the SDK's
// callbacks: the fetcher records what was actually requested, and the token
// source hook records what came back.
//
// Every field is written from the goroutine that calls Authorize (the SDK
// invokes both hooks synchronously), so no lock is needed.
type loginFlow struct {
	cfg      config.MCPServer
	stderr   io.Writer
	listener *loopbackListener
	open     func(string) error

	// requested is the scope list the authorization request finally carried,
	// configured extras included. It is the fallback when the token response
	// does not state the granted scopes.
	requested []string
	// clientID and tokenEndpoint are captured from the oauth2.Config the SDK
	// resolved; they are not otherwise readable from the handler.
	clientID, tokenEndpoint string
	// token is the exchanged credential.
	token *oauth2.Token
}

// newHandler builds the SDK handler for this flow.
//
// Exactly one client registration method is configured. Dynamic client
// registration is never configured: it is cut by design (it makes credentials
// non-portable and hammers authorization server rate limits), and the SDK
// would otherwise fall back to it.
func (f *loginFlow) newHandler() (*auth.AuthorizationCodeHandler, error) {
	acfg := &auth.AuthorizationCodeHandlerConfig{
		RedirectURL:              f.listener.URL(),
		AuthorizationCodeFetcher: f.fetch,
		RequestRefreshToken:      true,
		Client:                   newOAuthClient(),
		NewTokenSource:           f.captureToken,
	}
	if f.cfg.Auth.ClientID != "" {
		acfg.PreregisteredClient = &oauthex.ClientCredentials{ClientID: f.cfg.Auth.ClientID}
	} else {
		acfg.ClientIDMetadataDocumentConfig = &auth.ClientIDMetadataDocumentConfig{URL: CIMDDocumentURL}
	}
	h, err := auth.NewAuthorizationCodeHandler(acfg)
	if err != nil {
		return nil, fmt.Errorf("building the authorization handler: %w", err)
	}
	return h, nil
}

// fetch implements auth.AuthorizationCodeFetcher: it shows the operator the
// authorization URL, opens a browser, and waits for the loopback callback.
func (f *loginFlow) fetch(ctx context.Context, args *auth.AuthorizationArgs) (*auth.AuthorizationResult, error) {
	authURL, scopes, err := withExtraScopes(args.URL, f.cfg.Auth.Scopes)
	if err != nil {
		return nil, err
	}
	f.requested = scopes
	if err := checkBrowserURL(authURL); err != nil {
		return nil, err
	}
	// CONTRACT: stderr, always. stdout belongs to the run's output schema, and
	// a login that printed one byte there would corrupt a pipeline.
	_, _ = fmt.Fprintf(f.stderr, "Open this URL to authorize (server %q):\n  %s\n", f.cfg.Name, authURL)
	open := f.open
	if open == nil {
		open = openBrowserDefault
	}
	if err := open(authURL); err != nil {
		// Not fatal: the URL is already printed, and a headless-ish desktop
		// (no xdg-open, ssh session) is exactly the case the printed URL
		// exists for.
		_, _ = fmt.Fprintf(f.stderr, "could not open a browser (%v) - open the URL above by hand\n", err)
	}
	res, err := f.listener.wait(ctx)
	if err != nil {
		return nil, err
	}
	return &auth.AuthorizationResult{Code: res.code, State: res.state, Iss: res.iss}, nil
}

// captureToken implements the SDK's NewTokenSource hook. It is the ONLY place
// the resolved client id and token endpoint are visible: the handler keeps its
// oauth2.Config private and exposes neither.
//
// The returned source is static on purpose. The SDK calls Token() once more to
// read the granted scopes, and a refreshing source would spend a refresh token
// - and rotate it away - on that read.
func (f *loginFlow) captureToken(_ context.Context, cfg *oauth2.Config, tok *oauth2.Token) (oauth2.TokenSource, error) {
	f.clientID = cfg.ClientID
	f.tokenEndpoint = cfg.Endpoint.TokenURL
	f.token = tok
	return oauth2.StaticTokenSource(tok), nil
}

// record folds everything the flow learned into the credential to store.
func (f *loginFlow) record(disc *asDiscovery, resource string, now time.Time) (*oauthtoken.Record, error) {
	tok := f.token
	if tok.AccessToken == "" {
		return nil, errors.New("the token response carried no access token")
	}
	if tok.TokenType != "" && !strings.EqualFold(tok.TokenType, "bearer") {
		return nil, fmt.Errorf("the authorization server issued an unsupported token_type %q", safeParam(tok.TokenType))
	}
	endpoint := f.tokenEndpoint
	if endpoint == "" {
		endpoint = disc.tokenEndpoint
	}
	// The endpoint the exchange ACTUALLY used, re-checked: checkDiscovery saw
	// the advertised one, and these must not be allowed to diverge.
	if err := checkTokenEndpoint(endpoint); err != nil {
		return nil, fmt.Errorf("token endpoint: %w", err)
	}
	expiry := tok.Expiry
	if expiry.IsZero() {
		// RFC 6749 makes expires_in optional. An unknown lifetime is treated
		// as short rather than eternal, the same rule the refresh path uses.
		expiry = now.Add(defaultTokenLifetime)
	}
	return &oauthtoken.Record{
		Version:       oauthtoken.Version,
		Issuer:        disc.issuer,
		Resource:      resource,
		ClientID:      f.clientID,
		TokenEndpoint: endpoint,
		// Verbatim, not canonicalized: this is the literal string the
		// authorization server bound the token to, and every refresh has to
		// send it back byte for byte.
		AuthResource: disc.resource,
		// Stored at login because logout cannot rediscover it: discovery
		// starts from a 401 the protected resource may no longer be there to
		// send. Empty means the server advertises no RFC 7009 endpoint.
		RevocationEndpoint: disc.revocationEndpoint,
		AccessToken:        tok.AccessToken,
		RefreshToken:       tok.RefreshToken,
		ExpiresAt:          expiry,
		Scopes:             grantedScopes(tok, f.requested),
	}, nil
}

// grantedScopes prefers what the authorization server granted over what was
// asked for; RFC 6749 §5.1 only sends `scope` back when it differs, so an
// absent field means "as requested".
func grantedScopes(tok *oauth2.Token, requested []string) []string {
	if s, ok := tok.Extra("scope").(string); ok && strings.TrimSpace(s) != "" {
		return strings.Fields(s)
	}
	return requested
}

// persistLogin registers the fresh secrets and writes the record under the
// store's cross-process lock.
//
// The lock matters even for a first login: a run in another terminal may be
// refreshing the very credential this flow replaces, and the last writer must
// not be a torn one.
func persistLogin(ctx context.Context, deps Deps, rec *oauthtoken.Record) error {
	// SECURITY: registered BEFORE the write, so a value that lands in an error
	// message from the store itself is already redactable.
	if deps.RegisterSecret != nil {
		vals := []string{rec.AccessToken}
		if rec.RefreshToken != "" {
			vals = append(vals, rec.RefreshToken)
		}
		deps.RegisterSecret(vals...)
	}
	key := oauthtoken.Key{Issuer: rec.Issuer, Resource: rec.Resource, ClientID: rec.ClientID}
	if _, err := deps.TokenStore.WithLock(ctx, key, func(*oauthtoken.Record) (*oauthtoken.Record, error) {
		return rec, nil
	}); err != nil {
		return fmt.Errorf("storing the credential: %w", err)
	}
	return nil
}

// withExtraScopes unions the configured scopes into an authorization URL's
// `scope` parameter and reports the final list.
//
// The URL is rewritten rather than the SDK configured because the SDK derives
// the requested scopes entirely from the challenge and the protected resource
// metadata and offers no seam for extras. Union, never replacement: a recipe
// may ask for MORE than the server advertises, but must not be able to
// silently narrow what the flow needs (config contract).
func withExtraScopes(rawURL string, extra []string) (string, []string, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return "", nil, fmt.Errorf("parsing the authorization url: %w", err)
	}
	q := u.Query()
	scopes := strings.Fields(q.Get("scope"))
	for _, s := range extra {
		if s = strings.TrimSpace(s); s != "" && !slices.Contains(scopes, s) {
			scopes = append(scopes, s)
		}
	}
	if len(scopes) == 0 {
		return u.String(), nil, nil
	}
	q.Set("scope", strings.Join(scopes, " "))
	u.RawQuery = q.Encode()
	return u.String(), scopes, nil
}

// checkBrowserURL refuses an authorization URL amele will not hand to a
// browser command.
//
// SECURITY: the URL comes from server-supplied metadata. Requiring an absolute
// http(s) URL stops a `javascript:` or `file:` scheme from being launched, and
// stops a value beginning with `-` from being read as a flag by the opener.
func checkBrowserURL(raw string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("parsing the authorization url: %w", err)
	}
	if (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" {
		return fmt.Errorf("the authorization endpoint %s is not an absolute http(s) url", safeParam(raw))
	}
	return nil
}

// openBrowserDefault launches the platform's URL opener.
//
// The child is started and reaped in the background: xdg-open forks a browser
// and exits, and waiting for it would either block the flow or leave a zombie.
func openBrowserDefault(rawURL string) error {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", rawURL) //nolint:gosec // G204: the argv form takes no shell, and checkBrowserURL has already required an absolute http(s) url.
	case "windows":
		// `start` is a cmd builtin; the empty argument is the window title
		// slot, which start would otherwise take the URL for.
		cmd = exec.Command("cmd", "/c", "start", "", rawURL) //nolint:gosec // G204: see above.
	default:
		cmd = exec.Command("xdg-open", rawURL) //nolint:gosec // G204: see above.
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("launching a browser: %w", err)
	}
	go func() { _ = cmd.Wait() }()
	return nil
}
