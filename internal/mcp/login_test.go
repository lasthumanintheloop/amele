package mcp

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/oauthtoken"
)

// fakeAuthServer is one httptest server playing BOTH roles the login flow
// talks to: the protected MCP endpoint (401 + metadata) and the authorization
// server (/authorize, /token). Keeping them in one server keeps the issuer,
// the resource and the certificate consistent without a second fixture.
type fakeAuthServer struct {
	ts *httptest.Server

	// Knobs, set before the first request.
	//
	// asMetaOverride replaces the whole authorization-server metadata
	// document; nil serves the compliant default.
	asMetaOverride func(base string) map[string]any
	// noASMetadata makes every metadata well-known answer 404, which is how a
	// pre-2025-11 server that publishes nothing looks.
	noASMetadata bool
	// challengeScope is the `scope` parameter of the 401 challenge.
	challengeScope string
	// mcpPath is the path of the protected endpoint. A trailing slash makes
	// the declared resource literal differ from its canonical form.
	mcpPath string
	// prmScopes is the PRM scopes_supported list.
	prmScopes []string
	// authorizeError, when set, makes /authorize redirect with an error.
	authorizeError string
	// tokenScope is the scope string the token response carries; empty omits
	// the field entirely.
	tokenScope string

	mu sync.Mutex
	// authorizeQueries records every /authorize request's query.
	authorizeQueries []url.Values
	// tokenForms records every /token request's posted form.
	tokenForms []url.Values
	// codeChallenge is the PKCE challenge the last /authorize carried.
	codeChallenge string
}

// newFakeAuthServer starts the fixture and points the OAuth client seam at a
// client that trusts its certificate.
func newFakeAuthServer(t *testing.T) *fakeAuthServer {
	t.Helper()
	f := &fakeAuthServer{mcpPath: "/mcp"}
	f.ts = httptest.NewTLSServer(http.HandlerFunc(f.route))
	t.Cleanup(f.ts.Close)
	prev := newOAuthClient
	newOAuthClient = func() *http.Client { return f.ts.Client() }
	t.Cleanup(func() { newOAuthClient = prev })
	return f
}

// mcpURL is the protected MCP endpoint a config would name.
func (f *fakeAuthServer) mcpURL() string { return f.ts.URL + f.mcpPath }

// prmURL is where this fixture publishes its protected resource metadata.
func (f *fakeAuthServer) prmURL() string {
	return f.ts.URL + "/.well-known/oauth-protected-resource" + f.mcpPath
}

// issuer is the authorization server identifier this fixture publishes.
func (f *fakeAuthServer) issuer() string { return f.ts.URL }

func (f *fakeAuthServer) route(w http.ResponseWriter, r *http.Request) {
	base := f.ts.URL
	switch r.URL.Path {
	case f.mcpPath:
		params := fmt.Sprintf("resource_metadata=%q", f.prmURL())
		if f.challengeScope != "" {
			params += fmt.Sprintf(", scope=%q", f.challengeScope)
		}
		w.Header().Set("WWW-Authenticate", "Bearer "+params)
		http.Error(w, "unauthorized", http.StatusUnauthorized)
	case "/.well-known/oauth-protected-resource" + f.mcpPath:
		writeJSON(w, map[string]any{
			"resource":              f.mcpURL(),
			"authorization_servers": []string{base},
			"scopes_supported":      f.prmScopes,
		})
	case "/.well-known/oauth-authorization-server", "/.well-known/openid-configuration":
		if f.noASMetadata {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		meta := defaultASMeta(base)
		if f.asMetaOverride != nil {
			meta = f.asMetaOverride(base)
		}
		writeJSON(w, meta)
	case "/authorize":
		f.authorize(w, r)
	case "/token":
		f.token(w, r)
	default:
		http.Error(w, "not found", http.StatusNotFound)
	}
}

// defaultASMeta is a compliant authorization server metadata document.
func defaultASMeta(base string) map[string]any {
	return map[string]any{
		"issuer":                                base,
		"authorization_endpoint":                base + "/authorize",
		"token_endpoint":                        base + "/token",
		"revocation_endpoint":                   base + "/revoke",
		"response_types_supported":              []string{"code"},
		"code_challenge_methods_supported":      []string{"S256"},
		"client_id_metadata_document_supported": true,
		"scopes_supported":                      []string{"offline_access", "repo", "issues"},
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func (f *fakeAuthServer) authorize(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	f.mu.Lock()
	f.authorizeQueries = append(f.authorizeQueries, q)
	f.codeChallenge = q.Get("code_challenge")
	f.mu.Unlock()

	redirect, err := url.Parse(q.Get("redirect_uri"))
	if err != nil || redirect.Host == "" {
		http.Error(w, "bad redirect_uri", http.StatusBadRequest)
		return
	}
	rq := url.Values{"state": {q.Get("state")}}
	if f.authorizeError != "" {
		rq.Set("error", f.authorizeError)
	} else {
		rq.Set("code", "auth-code-1")
	}
	redirect.RawQuery = rq.Encode()
	//nolint:gosec // G710: this IS the fake authorization server; the redirect target is the loopback callback the test itself supplied.
	http.Redirect(w, r, redirect.String(), http.StatusFound)
}

func (f *fakeAuthServer) token(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad form", http.StatusBadRequest)
		return
	}
	f.mu.Lock()
	f.tokenForms = append(f.tokenForms, r.PostForm)
	challenge := f.codeChallenge
	scope := f.tokenScope
	f.mu.Unlock()

	sum := sha256.Sum256([]byte(r.PostForm.Get("code_verifier")))
	if got := base64.RawURLEncoding.EncodeToString(sum[:]); got != challenge {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, map[string]any{"error": "invalid_grant"})
		return
	}
	body := map[string]any{
		"access_token":  "access-token-1",
		"token_type":    "Bearer",
		"expires_in":    3600,
		"refresh_token": "refresh-token-1",
	}
	if scope != "" {
		body["scope"] = scope
	}
	writeJSON(w, body)
}

// lastAuthorizeQuery returns the query of the most recent /authorize request.
func (f *fakeAuthServer) lastAuthorizeQuery(t *testing.T) url.Values {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.authorizeQueries) == 0 {
		t.Fatal("no /authorize request was made")
	}
	return f.authorizeQueries[len(f.authorizeQueries)-1]
}

// lastTokenForm returns the form of the most recent /token request.
func (f *fakeAuthServer) lastTokenForm(t *testing.T) url.Values {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.tokenForms) == 0 {
		t.Fatal("no /token request was made")
	}
	return f.tokenForms[len(f.tokenForms)-1]
}

// loginFixture is one Login call's inputs: config, store, deps and the
// captured stderr.
type loginFixture struct {
	f       *fakeAuthServer
	cfg     config.MCPServer
	store   *oauthtoken.Store
	deps    Deps
	stderr  *strings.Builder
	now     time.Time
	secrets []string
	mu      sync.Mutex
	// visited records every URL the fake browser fetched.
	visited []string
}

// newLoginFixture wires a fixture whose browser follows the authorize
// redirect all the way to the loopback callback.
func newLoginFixture(t *testing.T, f *fakeAuthServer) *loginFixture {
	t.Helper()
	lf := &loginFixture{
		f:      f,
		stderr: &strings.Builder{},
		now:    time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC),
	}
	lf.store = oauthtoken.NewStore(t.TempDir(), func() time.Time { return lf.now })
	lf.cfg = config.MCPServer{
		Name:      "github",
		Transport: config.MCPTransport{Type: "http", URL: f.mcpURL()},
		Auth:      &config.MCPAuth{Type: config.MCPAuthOAuth},
	}
	lf.deps = Deps{
		TokenStore: lf.store,
		RegisterSecret: func(vals ...string) {
			lf.mu.Lock()
			defer lf.mu.Unlock()
			lf.secrets = append(lf.secrets, vals...)
		},
	}
	return lf
}

// browse is the OpenBrowser seam: it fetches the authorization URL with a
// client that trusts the fixture's certificate and follows the redirect to
// the loopback listener, exactly as a real browser would.
func (lf *loginFixture) browse(rawURL string) error {
	lf.mu.Lock()
	lf.visited = append(lf.visited, rawURL)
	lf.mu.Unlock()
	resp, err := lf.f.ts.Client().Get(rawURL) //nolint:noctx // the fake browser is a test seam
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	_, _ = io.Copy(io.Discard, resp.Body)
	return nil
}

// opts returns LoginOptions wired to this fixture.
func (lf *loginFixture) opts() LoginOptions {
	return LoginOptions{OpenBrowser: lf.browse, Stderr: lf.stderr}
}

// login runs Login with this fixture's inputs.
func (lf *loginFixture) login(t *testing.T) (*oauthtoken.Record, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return Login(ctx, lf.cfg, lf.deps, lf.opts())
}

func TestLoginEndToEnd(t *testing.T) {
	f := newFakeAuthServer(t)
	f.prmScopes = []string{"repo"}
	f.tokenScope = "repo offline_access"
	lf := newLoginFixture(t, f)

	rec, err := lf.login(t)
	if err != nil {
		t.Fatalf("Login: %v", err)
	}
	resource, err := oauthtoken.CanonicalResource(f.mcpURL())
	if err != nil {
		t.Fatalf("CanonicalResource: %v", err)
	}
	if rec.Issuer != f.issuer() {
		t.Errorf("issuer = %q, want %q", rec.Issuer, f.issuer())
	}
	if rec.Resource != resource {
		t.Errorf("resource = %q, want %q", rec.Resource, resource)
	}
	if rec.ClientID != CIMDDocumentURL {
		t.Errorf("client_id = %q, want the CIMD url %q", rec.ClientID, CIMDDocumentURL)
	}
	if rec.TokenEndpoint != f.issuer()+"/token" {
		t.Errorf("token_endpoint = %q", rec.TokenEndpoint)
	}
	if rec.AccessToken != "access-token-1" || rec.RefreshToken != "refresh-token-1" {
		t.Errorf("tokens = %q/%q", rec.AccessToken, rec.RefreshToken)
	}
	if got := strings.Join(rec.Scopes, " "); got != "repo offline_access" {
		t.Errorf("scopes = %q, want %q", got, "repo offline_access")
	}
	if rec.ExpiresAt.IsZero() {
		t.Error("expires_at is zero")
	}

	// The record is on disk under the key a run will look it up with.
	key := oauthtoken.Key{Issuer: rec.Issuer, Resource: rec.Resource, ClientID: rec.ClientID}
	stored, err := lf.store.Load(key)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if stored.AccessToken != rec.AccessToken {
		t.Errorf("stored access token = %q", stored.AccessToken)
	}
	// Both minted values reached the run's redactor.
	lf.mu.Lock()
	secrets := strings.Join(lf.secrets, " ")
	lf.mu.Unlock()
	for _, want := range []string{"access-token-1", "refresh-token-1"} {
		if !strings.Contains(secrets, want) {
			t.Errorf("RegisterSecret did not receive %q (got %q)", want, secrets)
		}
	}
}

func TestLoginKeepsListeningPastGarbage(t *testing.T) {
	f := newFakeAuthServer(t)
	lf := newLoginFixture(t, f)
	// The first thing the "browser" does is a favicon-shaped probe at the
	// callback path; only the real redirect must resolve the flow.
	var bogus struct {
		once   sync.Once
		status int
	}
	lf.deps.RegisterSecret = nil
	opts := lf.opts()
	opts.OpenBrowser = func(rawURL string) error {
		bogus.once.Do(func() {
			u, err := url.Parse(rawURL)
			if err != nil {
				t.Errorf("parsing authorize url: %v", err)
				return
			}
			redirect := u.Query().Get("redirect_uri")
			resp, err := http.Get(redirect + "?bogus=1") //nolint:noctx // test probe
			if err != nil {
				t.Errorf("probe: %v", err)
				return
			}
			bogus.status = resp.StatusCode
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
		})
		return lf.browse(rawURL)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if _, err := Login(ctx, lf.cfg, lf.deps, opts); err != nil {
		t.Fatalf("Login: %v", err)
	}
	if bogus.status != http.StatusBadRequest {
		t.Errorf("probe status = %d, want 400", bogus.status)
	}
}

func TestLoginAllOutputOnStderr(t *testing.T) {
	f := newFakeAuthServer(t)
	lf := newLoginFixture(t, f)
	if _, err := lf.login(t); err != nil {
		t.Fatalf("Login: %v", err)
	}
	out := lf.stderr.String()
	if !strings.Contains(out, "Open this URL to authorize") {
		t.Errorf("stderr lacks the instruction line: %q", out)
	}
	lf.mu.Lock()
	visited := lf.visited
	lf.mu.Unlock()
	if len(visited) == 0 {
		t.Fatal("the browser seam was never called")
	}
	if !strings.Contains(out, visited[0]) {
		t.Errorf("stderr does not carry the authorization url\nstderr: %q\nurl: %q", out, visited[0])
	}
	// Compile-time half of the contract: LoginOptions has no stdout seam at
	// all, so there is nowhere for Login to write to stdout by mistake.
	var _ = LoginOptions{OpenBrowser: nil, Stderr: nil, Confirm: nil}
}

func TestLoginRefusesASWithoutS256(t *testing.T) {
	for _, tc := range []struct {
		name    string
		methods []string
	}{
		{"omitted", nil},
		{"plain only", []string{"plain"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newFakeAuthServer(t)
			f.asMetaOverride = func(base string) map[string]any {
				meta := defaultASMeta(base)
				if tc.methods == nil {
					delete(meta, "code_challenge_methods_supported")
				} else {
					meta["code_challenge_methods_supported"] = tc.methods
				}
				return meta
			}
			lf := newLoginFixture(t, f)
			opts := lf.opts()
			opts.OpenBrowser = func(string) error {
				t.Error("browser opened despite a server without S256 PKCE")
				return nil
			}
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			_, err := Login(ctx, lf.cfg, lf.deps, opts)
			if err == nil {
				t.Fatal("Login succeeded against a server without S256 PKCE")
			}
			if !strings.Contains(strings.ToUpper(err.Error()), "PKCE") {
				t.Errorf("error does not name PKCE: %v", err)
			}
		})
	}
}

func TestLoginResourceParamOnTokenExchange(t *testing.T) {
	f := newFakeAuthServer(t)
	lf := newLoginFixture(t, f)
	if _, err := lf.login(t); err != nil {
		t.Fatalf("Login: %v", err)
	}
	resource, err := oauthtoken.CanonicalResource(f.mcpURL())
	if err != nil {
		t.Fatalf("CanonicalResource: %v", err)
	}
	if got := f.lastTokenForm(t).Get("resource"); got != resource {
		t.Errorf("token request resource = %q, want %q", got, resource)
	}
	if got := f.lastAuthorizeQuery(t).Get("resource"); got != resource {
		t.Errorf("authorize resource = %q, want %q", got, resource)
	}
}

func TestLoginConfiguredScopesRequested(t *testing.T) {
	f := newFakeAuthServer(t)
	f.challengeScope = "repo"
	lf := newLoginFixture(t, f)
	lf.cfg.Auth.Scopes = []string{"issues", "repo"}
	rec, err := lf.login(t)
	if err != nil {
		t.Fatalf("Login: %v", err)
	}
	got := strings.Fields(f.lastAuthorizeQuery(t).Get("scope"))
	for _, want := range []string{"repo", "issues"} {
		if !containsString(got, want) {
			t.Errorf("authorize scope %v lacks %q", got, want)
		}
	}
	// With no scope in the token response the granted set falls back to what
	// was requested, extras included.
	if !containsString(rec.Scopes, "issues") {
		t.Errorf("record scopes %v lack the configured extra", rec.Scopes)
	}
}

func containsString(list []string, want string) bool {
	for _, s := range list {
		if s == want {
			return true
		}
	}
	return false
}

func TestLoginErrorCallback(t *testing.T) {
	f := newFakeAuthServer(t)
	f.authorizeError = "access_denied"
	lf := newLoginFixture(t, f)
	_, err := lf.login(t)
	if err == nil {
		t.Fatal("Login succeeded after an access_denied callback")
	}
	if !strings.Contains(err.Error(), "access_denied") {
		t.Errorf("error does not name the refusal: %v", err)
	}
	if entries, lerr := lf.store.List(); lerr != nil || len(entries) != 0 {
		t.Errorf("store holds %d entries after a refused login (err %v)", len(entries), lerr)
	}
}

func TestLoginBrowserOpenFailureNotFatal(t *testing.T) {
	f := newFakeAuthServer(t)
	lf := newLoginFixture(t, f)
	opts := lf.opts()
	opts.OpenBrowser = func(rawURL string) error {
		// The opener fails; the operator reads the URL off stderr and drives
		// it by hand, which the test does on a goroutine.
		go func() { _ = lf.browse(rawURL) }()
		return fmt.Errorf("no xdg-open here")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	rec, err := Login(ctx, lf.cfg, lf.deps, opts)
	if err != nil {
		t.Fatalf("Login: %v", err)
	}
	if rec.AccessToken != "access-token-1" {
		t.Errorf("access token = %q", rec.AccessToken)
	}
	if !strings.Contains(lf.stderr.String(), "no xdg-open here") {
		t.Errorf("stderr does not report the failed browser launch: %q", lf.stderr.String())
	}
}

func TestLoginDeclinedByConfirm(t *testing.T) {
	f := newFakeAuthServer(t)
	lf := newLoginFixture(t, f)
	opts := lf.opts()
	var gotIssuer, gotResource string
	opts.Confirm = func(issuer, resource string) bool {
		gotIssuer, gotResource = issuer, resource
		return false
	}
	opts.OpenBrowser = func(string) error {
		t.Error("browser opened despite a declined confirmation")
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if _, err := Login(ctx, lf.cfg, lf.deps, opts); err == nil {
		t.Fatal("Login succeeded after the operator declined")
	}
	if gotIssuer != f.issuer() {
		t.Errorf("confirm issuer = %q, want %q", gotIssuer, f.issuer())
	}
	if !strings.Contains(gotResource, "/mcp") {
		t.Errorf("confirm resource = %q", gotResource)
	}
}

func TestLoginRequiresMetadata(t *testing.T) {
	f := newFakeAuthServer(t)
	f.noASMetadata = true
	lf := newLoginFixture(t, f)
	opts := lf.opts()
	opts.OpenBrowser = func(string) error {
		t.Error("browser opened against a server publishing no metadata")
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if _, err := Login(ctx, lf.cfg, lf.deps, opts); err == nil {
		t.Fatal("Login succeeded against an authorization server with no metadata")
	}
}

func TestLoginRequiresCIMDOrClientID(t *testing.T) {
	f := newFakeAuthServer(t)
	f.asMetaOverride = func(base string) map[string]any {
		meta := defaultASMeta(base)
		meta["client_id_metadata_document_supported"] = false
		return meta
	}
	lf := newLoginFixture(t, f)

	// Without a configured client id there is no registration route left.
	opts := lf.opts()
	opts.OpenBrowser = func(string) error {
		t.Error("browser opened without a usable client identity")
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, err := Login(ctx, lf.cfg, lf.deps, opts)
	if err == nil {
		t.Fatal("Login succeeded with no client registration method")
	}
	if !strings.Contains(err.Error(), "client_id") {
		t.Errorf("error does not point at auth.client_id: %v", err)
	}

	// With one, the same server logs in fine and the record keys on it.
	lf.cfg.Auth.ClientID = "Iv1.preregistered"
	rec, err := lf.login(t)
	if err != nil {
		t.Fatalf("Login with a pre-registered client: %v", err)
	}
	if rec.ClientID != "Iv1.preregistered" {
		t.Errorf("client_id = %q", rec.ClientID)
	}
	if got := f.lastAuthorizeQuery(t).Get("client_id"); got != "Iv1.preregistered" {
		t.Errorf("authorize client_id = %q", got)
	}
}

func TestLoginRejectsNonOAuthServer(t *testing.T) {
	cases := map[string]config.MCPServer{
		"no auth block": {Name: "x", Transport: config.MCPTransport{Type: "http", URL: "https://mcp.example/mcp"}},
		"stdio": {
			Name:      "x",
			Transport: config.MCPTransport{Type: "stdio", Command: []string{"true"}},
			Auth:      &config.MCPAuth{Type: config.MCPAuthOAuth},
		},
	}
	for name, cfg := range cases {
		t.Run(name, func(t *testing.T) {
			store := oauthtoken.NewStore(t.TempDir(), time.Now)
			_, err := Login(context.Background(), cfg, Deps{TokenStore: store}, LoginOptions{})
			if err == nil {
				t.Fatal("Login accepted a server it cannot log in")
			}
		})
	}
}

func TestLoginRequiresTokenStore(t *testing.T) {
	cfg := config.MCPServer{
		Name:      "x",
		Transport: config.MCPTransport{Type: "http", URL: "https://mcp.example/mcp"},
		Auth:      &config.MCPAuth{Type: config.MCPAuthOAuth},
	}
	if _, err := Login(context.Background(), cfg, Deps{}, LoginOptions{}); err == nil {
		t.Fatal("Login ran without a token store")
	}
}

func TestLoopbackListenerIgnoresGarbage(t *testing.T) {
	l, err := newLoopbackListener()
	if err != nil {
		t.Fatalf("newLoopbackListener: %v", err)
	}
	defer l.close()

	for _, tc := range []struct {
		name string
		req  func() (*http.Response, error)
		want int
	}{
		{"no params", func() (*http.Response, error) { return http.Get(l.URL()) }, http.StatusBadRequest},                                         //nolint:noctx // local test request
		{"code without state", func() (*http.Response, error) { return http.Get(l.URL() + "?code=x") }, http.StatusBadRequest},                    //nolint:noctx // local test request
		{"wrong path", func() (*http.Response, error) { return http.Get("http://" + l.addr() + "/favicon.ico") }, http.StatusNotFound},            //nolint:noctx // local test request
		{"post", func() (*http.Response, error) { return http.Post(l.URL(), "text/plain", strings.NewReader("x")) }, http.StatusMethodNotAllowed}, //nolint:noctx // local test request
	} {
		t.Run(tc.name, func(t *testing.T) {
			resp, err := tc.req()
			if err != nil {
				t.Fatalf("request: %v", err)
			}
			defer func() { _ = resp.Body.Close() }()
			if resp.StatusCode != tc.want {
				t.Errorf("status = %d, want %d", resp.StatusCode, tc.want)
			}
		})
	}

	// After all that noise the listener still resolves a real callback.
	resp, err := http.Get(l.URL() + "?code=c1&state=s1") //nolint:noctx // local test request
	if err != nil {
		t.Fatalf("callback: %v", err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	res, err := l.wait(context.Background())
	if err != nil {
		t.Fatalf("wait: %v", err)
	}
	if res.code != "c1" || res.state != "s1" {
		t.Errorf("result = %+v", res)
	}
}

func TestLoopbackListenerRespectsContext(t *testing.T) {
	l, err := newLoopbackListener()
	if err != nil {
		t.Fatalf("newLoopbackListener: %v", err)
	}
	defer l.close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := l.wait(ctx); err == nil {
		t.Fatal("wait ignored a cancelled context")
	}
}

func TestWithExtraScopes(t *testing.T) {
	for _, tc := range []struct {
		name  string
		url   string
		extra []string
		want  string
	}{
		{"no scope at all", "https://as.example/authorize?client_id=x", nil, ""},
		{"extras only", "https://as.example/authorize?client_id=x", []string{"repo", " "}, "repo"},
		{"union without duplicates", "https://as.example/authorize?scope=repo+issues", []string{"repo", "gist"}, "repo issues gist"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, scopes, err := withExtraScopes(tc.url, tc.extra)
			if err != nil {
				t.Fatalf("withExtraScopes: %v", err)
			}
			if joined := strings.Join(scopes, " "); joined != tc.want {
				t.Errorf("scopes = %q, want %q", joined, tc.want)
			}
			u, err := url.Parse(got)
			if err != nil {
				t.Fatalf("parsing result: %v", err)
			}
			if u.Query().Get("scope") != tc.want {
				t.Errorf("url scope = %q, want %q", u.Query().Get("scope"), tc.want)
			}
			if u.Query().Get("client_id") == "" && strings.Contains(tc.url, "client_id") {
				t.Error("the rewrite dropped an unrelated parameter")
			}
		})
	}
}

func TestCheckBrowserURL(t *testing.T) {
	for _, tc := range []struct {
		in string
		ok bool
	}{
		{"https://as.example/authorize?x=1", true},
		{"http://127.0.0.1:9/authorize", true},
		{"javascript:alert(1)", false},
		{"file:///etc/passwd", false},
		{"-flag-shaped", false},
		{"/relative", false},
	} {
		if err := checkBrowserURL(tc.in); (err == nil) != tc.ok {
			t.Errorf("checkBrowserURL(%q) error = %v, want ok=%v", tc.in, err, tc.ok)
		}
	}
}

func TestSafeParam(t *testing.T) {
	if got := safeParam("access_denied"); got != "access_denied" {
		t.Errorf("safeParam kept %q", got)
	}
	if got := safeParam("\x1b[31mred\x07"); got != "[31mred" {
		t.Errorf("safeParam did not strip control bytes: %q", got)
	}
	if got := safeParam("\x00\x01"); got != "unspecified" {
		t.Errorf("safeParam of an unprintable value = %q", got)
	}
	if got := safeParam(strings.Repeat("a", 200)); len(got) != 64 {
		t.Errorf("safeParam did not clip: %d runes", len(got))
	}
}

func TestPickAuthServer(t *testing.T) {
	for _, tc := range []struct {
		name    string
		servers []string
		want    string
	}{
		{"single https", []string{"https://as.example"}, "https://as.example"},
		// The SDK uses entry [0] unconditionally. Scanning past it for the
		// first https entry would run our checks against a different server
		// than the flow itself, so a plaintext [0] is a refusal, never a skip.
		{"plaintext first", []string{"http://as.local", "https://as.example"}, ""},
		{"relative", []string{"/authorize"}, ""},
		{"empty", nil, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := pickAuthServer(tc.servers)
			if tc.want == "" {
				if err == nil {
					t.Fatalf("pickAuthServer(%v) = %q, want a refusal", tc.servers, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("pickAuthServer(%v): %v", tc.servers, err)
			}
			if got != tc.want {
				t.Errorf("pickAuthServer(%v) = %q, want %q", tc.servers, got, tc.want)
			}
		})
	}
}

func TestLoginPersistsLiteralAuthResource(t *testing.T) {
	f := newFakeAuthServer(t)
	// A PRM that declares the resource WITH a trailing slash: it canonicalizes
	// to the config url, but the authorization server binds the token to this
	// literal string, so every later refresh must send it verbatim.
	f.mcpPath = "/mcp/"
	lf := newLoginFixture(t, f)
	rec, err := lf.login(t)
	if err != nil {
		t.Fatalf("Login: %v", err)
	}
	want := f.mcpURL()
	if rec.AuthResource != want {
		t.Errorf("AuthResource = %q, want %q", rec.AuthResource, want)
	}
	if got := f.lastTokenForm(t).Get("resource"); got != want {
		t.Errorf("token exchange resource = %q, want %q", got, want)
	}
	canonical, err := oauthtoken.CanonicalResource(f.mcpURL())
	if err != nil {
		t.Fatalf("CanonicalResource: %v", err)
	}
	if rec.Resource != canonical {
		t.Errorf("Resource = %q, want the canonical %q", rec.Resource, canonical)
	}
}

func TestRefreshGrantSendsAuthResource(t *testing.T) {
	as := newFakeAS(t, asReply{status: 200, body: `{"access_token":"at2","token_type":"Bearer","expires_in":3600}`})
	now := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	rec := &oauthtoken.Record{
		Version:       oauthtoken.Version,
		Issuer:        as.ts.URL,
		Resource:      "https://mcp.example/mcp",
		AuthResource:  "https://mcp.example/mcp/",
		ClientID:      "cid",
		TokenEndpoint: as.tokenURL(),
		AccessToken:   "at1",
		RefreshToken:  "rt1",
		ExpiresAt:     now,
	}
	if _, err := refreshGrant(context.Background(), as.ts.Client(), rec, rec.Resource, now); err != nil {
		t.Fatalf("refreshGrant: %v", err)
	}
	if got := as.lastForm().Get("resource"); got != rec.AuthResource {
		t.Errorf("refresh resource = %q, want the literal %q", got, rec.AuthResource)
	}
}
