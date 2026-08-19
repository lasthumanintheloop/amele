package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/oauthtoken"
	"github.com/lasthumanintheloop/amele/internal/tools"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// oauthResource is the MCP endpoint every OAuth test dials. It is never
// contacted: these tests exercise the handler, not the transport.
const oauthResource = "https://mcp.example/mcp"

// fakeAS is a stand-in authorization server: one /token endpoint, a scripted
// sequence of replies and a request counter.
type fakeAS struct {
	ts *httptest.Server

	mu       sync.Mutex
	requests int
	forms    []url.Values
	// replies are consumed in order; the last one repeats forever.
	replies []asReply
	// hook, when set, runs before each reply is written. It is how a test
	// simulates another process writing the store while this one waits.
	hook func(n int)
}

// asReply is one scripted token endpoint answer.
type asReply struct {
	status int
	body   string
}

// newFakeAS starts a TLS authorization server serving replies in order.
func newFakeAS(t *testing.T, replies ...asReply) *fakeAS {
	t.Helper()
	as := &fakeAS{replies: replies}
	as.ts = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			http.Error(w, "bad form", http.StatusBadRequest)
			return
		}
		as.mu.Lock()
		as.requests++
		n := as.requests
		as.forms = append(as.forms, r.PostForm)
		reply := as.replies[min(n, len(as.replies))-1]
		hook := as.hook
		as.mu.Unlock()
		if hook != nil {
			hook(n)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(reply.status)
		_, _ = w.Write([]byte(reply.body))
	}))
	t.Cleanup(as.ts.Close)
	// The hardened production client does not trust the test certificate, so
	// the seam hands the handler the httptest client instead.
	prev := newOAuthClient
	newOAuthClient = func() *http.Client { return as.ts.Client() }
	t.Cleanup(func() { newOAuthClient = prev })
	return as
}

// count returns how many token requests the fake server has served.
func (a *fakeAS) count() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.requests
}

// lastForm returns the most recent token request's form values.
func (a *fakeAS) lastForm() url.Values {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.forms) == 0 {
		return nil
	}
	return a.forms[len(a.forms)-1]
}

// tokenURL is the fake server's token endpoint.
func (a *fakeAS) tokenURL() string { return a.ts.URL + "/token" }

// oauthFixture is one handler under test plus everything it was built from.
type oauthFixture struct {
	store    *oauthtoken.Store
	key      oauthtoken.Key
	now      time.Time
	as       *fakeAS
	secrets  []string
	secretMu sync.Mutex
	dead     []error
}

// newOAuthFixture seeds a store with one record whose access token expires in
// `life` and returns the fixture. A negative life makes the record stale.
func newOAuthFixture(t *testing.T, as *fakeAS, life time.Duration) *oauthFixture {
	t.Helper()
	f := &oauthFixture{
		as:  as,
		now: time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC),
	}
	f.store = oauthtoken.NewStore(t.TempDir(), func() time.Time { return f.now })
	res, err := oauthtoken.CanonicalResource(oauthResource)
	if err != nil {
		t.Fatalf("CanonicalResource: %v", err)
	}
	f.key = oauthtoken.Key{Issuer: "https://as.example", Resource: res, ClientID: "cid"}
	f.save(t, "at1", "rt1", life)
	return f
}

// save writes a record with the given tokens and lifetime, bypassing the lock
// the way a second process would look to the handler.
func (f *oauthFixture) save(t *testing.T, access, refresh string, life time.Duration) {
	t.Helper()
	rec := &oauthtoken.Record{
		Version:       oauthtoken.Version,
		Issuer:        f.key.Issuer,
		Resource:      f.key.Resource,
		ClientID:      f.key.ClientID,
		TokenEndpoint: f.as.tokenURL(),
		AccessToken:   access,
		RefreshToken:  refresh,
		ExpiresAt:     f.now.Add(life),
		Scopes:        []string{"repo"},
	}
	if err := f.store.Save(f.key, rec); err != nil {
		t.Fatalf("Save: %v", err)
	}
}

// cfg is the server config the handler is built from.
func (f *oauthFixture) cfg() config.MCPServer {
	return config.MCPServer{
		Name:      "s",
		Transport: config.MCPTransport{Type: config.MCPTransportHTTP, URL: oauthResource},
		Auth:      &config.MCPAuth{Type: config.MCPAuthOAuth, ClientID: "cid"},
	}
}

// deps builds injected dependencies wired to this fixture.
func (f *oauthFixture) deps() Deps {
	return Deps{
		Clock: func() time.Time { return f.now },
		Rand:  func() float64 { return 0 },
		Env:   func(string) (string, bool) { return "", false },
		RegisterSecret: func(vals ...string) {
			f.secretMu.Lock()
			defer f.secretMu.Unlock()
			f.secrets = append(f.secrets, vals...)
		},
		TokenStore: f.store,
	}
}

// handler builds the handler under test.
func (f *oauthFixture) handler(t *testing.T) *runOAuthHandler {
	t.Helper()
	h, err := newRunOAuthHandler(f.cfg(), f.deps())
	if err != nil {
		t.Fatalf("newRunOAuthHandler: %v", err)
	}
	h.onDead = func(err error) { f.dead = append(f.dead, err) }
	return h
}

// registered returns the secret values collected so far.
func (f *oauthFixture) registered() []string {
	f.secretMu.Lock()
	defer f.secretMu.Unlock()
	return append([]string(nil), f.secrets...)
}

// tokenOf drives one TokenSource round trip.
func tokenOf(t *testing.T, h *runOAuthHandler) (string, error) {
	t.Helper()
	ts, err := h.TokenSource(context.Background())
	if err != nil {
		return "", err
	}
	tok, err := ts.Token()
	if err != nil {
		return "", err
	}
	return tok.AccessToken, nil
}

// okBody is a successful token response.
func okBody(access, refresh string, expiresIn int) string {
	if refresh == "" {
		return fmt.Sprintf(`{"access_token":%q,"token_type":"Bearer","expires_in":%d}`, access, expiresIn)
	}
	return fmt.Sprintf(`{"access_token":%q,"refresh_token":%q,"token_type":"bearer","expires_in":%d}`, access, refresh, expiresIn)
}

func TestTokenSourceServesFreshFromStore(t *testing.T) {
	as := newFakeAS(t, asReply{status: 200, body: okBody("nope", "", 3600)})
	f := newOAuthFixture(t, as, time.Hour)
	h := f.handler(t)

	got, err := tokenOf(t, h)
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if got != "at1" {
		t.Errorf("access token = %q, want at1", got)
	}
	if n := as.count(); n != 0 {
		t.Errorf("token endpoint hit %d times, want 0", n)
	}
}

func TestTokenSourceRefreshesStaleUnderLock(t *testing.T) {
	as := newFakeAS(t, asReply{status: 200, body: okBody("at2", "rt2", 3600)})
	f := newOAuthFixture(t, as, -time.Minute)
	h := f.handler(t)

	got, err := tokenOf(t, h)
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if got != "at2" {
		t.Errorf("access token = %q, want at2", got)
	}
	if n := as.count(); n != 1 {
		t.Errorf("token endpoint hit %d times, want 1", n)
	}
	form := as.lastForm()
	if form.Get("grant_type") != "refresh_token" || form.Get("refresh_token") != "rt1" ||
		form.Get("client_id") != "cid" || form.Get("resource") != f.key.Resource {
		t.Errorf("token request form = %v", form)
	}
	// The rotated refresh token must be on disk for the next process.
	rec, err := f.store.Load(f.key)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if rec.AccessToken != "at2" || rec.RefreshToken != "rt2" {
		t.Errorf("persisted record = %+v, want at2/rt2", rec)
	}
	if !rec.ExpiresAt.After(f.now) {
		t.Errorf("expiry %v not in the future of %v", rec.ExpiresAt, f.now)
	}
	// A second call is served from memory: one refresh per expiry, not per call.
	if _, err := tokenOf(t, h); err != nil {
		t.Fatalf("second Token: %v", err)
	}
	if n := as.count(); n != 1 {
		t.Errorf("token endpoint hit %d times after a second call, want 1", n)
	}
}

func TestTokenSourceAdoptsConcurrentRefresh(t *testing.T) {
	as := newFakeAS(t, asReply{status: 200, body: okBody("wrong", "", 3600)})
	f := newOAuthFixture(t, as, -time.Minute)
	h := f.handler(t)
	// Another worker refreshed between construction and the call.
	f.save(t, "at9", "rt9", time.Hour)

	got, err := tokenOf(t, h)
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if got != "at9" {
		t.Errorf("access token = %q, want at9 (adopted)", got)
	}
	if n := as.count(); n != 0 {
		t.Errorf("token endpoint hit %d times, want 0", n)
	}
}

func TestRefreshRotationKeepsOldWhenAbsent(t *testing.T) {
	as := newFakeAS(t, asReply{status: 200, body: okBody("at2", "", 3600)})
	f := newOAuthFixture(t, as, -time.Minute)
	h := f.handler(t)

	if _, err := tokenOf(t, h); err != nil {
		t.Fatalf("Token: %v", err)
	}
	rec, err := f.store.Load(f.key)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if rec.RefreshToken != "rt1" {
		t.Errorf("refresh token = %q, want the old rt1 kept", rec.RefreshToken)
	}
	if rec.AccessToken != "at2" {
		t.Errorf("access token = %q, want at2", rec.AccessToken)
	}
}

func TestInvalidGrantRereadsBeforeDeath(t *testing.T) {
	as := newFakeAS(t, asReply{status: 400, body: `{"error":"invalid_grant"}`})
	f := newOAuthFixture(t, as, -time.Minute)
	h := f.handler(t)
	// The "other process" finishes its own rotation while this refresh is in
	// flight: the record on disk changes before the re-read.
	as.hook = func(int) { f.save(t, "at7", "rt7", time.Hour) }

	got, err := tokenOf(t, h)
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if got != "at7" {
		t.Errorf("access token = %q, want at7 (adopted after re-read)", got)
	}
	if len(f.dead) != 0 {
		t.Errorf("credential declared dead: %v", f.dead)
	}
}

func TestTransientNotCachedNotFatal(t *testing.T) {
	as := newFakeAS(t,
		asReply{status: 503, body: `{"error":"temporarily_unavailable"}`},
		asReply{status: 200, body: okBody("at2", "rt2", 3600)},
	)
	f := newOAuthFixture(t, as, -time.Minute)
	h := f.handler(t)

	if _, err := tokenOf(t, h); !errors.Is(err, errTransientAuth) {
		t.Fatalf("first Token error = %v, want errTransientAuth", err)
	}
	if len(f.dead) != 0 {
		t.Errorf("a transient failure killed the credential: %v", f.dead)
	}
	got, err := tokenOf(t, h)
	if err != nil {
		t.Fatalf("second Token: %v", err)
	}
	if got != "at2" {
		t.Errorf("access token = %q, want at2", got)
	}
	if class := classify(fmt.Errorf("wrapped: %w", errTransientAuth)); class != ClassNetwork {
		t.Errorf("classify(errTransientAuth) = %q, want %q", class, ClassNetwork)
	}
}

func TestAuthorizeSecondFailureIsAuthDeath(t *testing.T) {
	as := newFakeAS(t, asReply{status: 400, body: `{"error":"invalid_grant"}`})
	f := newOAuthFixture(t, as, time.Hour)
	h := f.handler(t)

	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, oauthResource, nil)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Authorization", "Bearer at1")
	resp := &http.Response{StatusCode: http.StatusUnauthorized, Body: http.NoBody}

	if err := h.Authorize(context.Background(), req, resp); err == nil {
		t.Fatal("Authorize succeeded on a stable invalid_grant")
	}
	if n := as.count(); n != 1 {
		t.Fatalf("token endpoint hit %d times, want 1", n)
	}
	if len(f.dead) != 1 {
		t.Fatalf("onDead called %d times, want 1", len(f.dead))
	}
	// The cached verdict answers instantly, with no further network traffic.
	if err := h.Authorize(context.Background(), req, &http.Response{StatusCode: 401, Body: http.NoBody}); err == nil {
		t.Fatal("second Authorize succeeded")
	}
	if _, err := tokenOf(t, h); err == nil {
		t.Fatal("Token succeeded after auth death")
	}
	if n := as.count(); n != 1 {
		t.Errorf("token endpoint hit %d times after the death, want 1", n)
	}
	if len(f.dead) != 1 {
		t.Errorf("onDead called %d times, want exactly 1", len(f.dead))
	}
}

func TestNoTokenNamesLoginCommand(t *testing.T) {
	as := newFakeAS(t, asReply{status: 200, body: okBody("at2", "rt2", 3600)})
	f := newOAuthFixture(t, as, time.Hour)
	if err := f.store.Delete(f.key); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	_, err := newRunOAuthHandler(f.cfg(), f.deps())
	if err == nil {
		t.Fatal("newRunOAuthHandler succeeded with an empty store")
	}
	if !strings.Contains(err.Error(), "amele mcp login") {
		t.Errorf("error %q does not name the login command", err)
	}
	if class := classify(err); class != ClassAuth {
		t.Errorf("classify = %q, want %q", class, ClassAuth)
	}
}

func TestTokensRegisteredAsSecrets(t *testing.T) {
	as := newFakeAS(t, asReply{status: 200, body: okBody("at2", "rt2", 3600)})
	f := newOAuthFixture(t, as, -time.Minute)
	h := f.handler(t)

	if got := f.registered(); len(got) != 2 || got[0] != "at1" || got[1] != "rt1" {
		t.Fatalf("secrets after load = %v, want [at1 rt1]", got)
	}
	if _, err := tokenOf(t, h); err != nil {
		t.Fatalf("Token: %v", err)
	}
	want := []string{"at1", "rt1", "at2", "rt2"}
	got := f.registered()
	if len(got) != len(want) {
		t.Fatalf("secrets = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("secrets = %v, want %v", got, want)
		}
	}
	// A second call registers nothing new: the set has no de-duplication.
	if _, err := tokenOf(t, h); err != nil {
		t.Fatalf("second Token: %v", err)
	}
	if got := f.registered(); len(got) != len(want) {
		t.Errorf("secrets = %v, want no re-registration", got)
	}
}

func TestTransportGetsHandlerOnlyWithAuth(t *testing.T) {
	as := newFakeAS(t, asReply{status: 200, body: okBody("at2", "rt2", 3600)})
	f := newOAuthFixture(t, as, time.Hour)
	ctx := context.Background()

	plain := f.cfg()
	plain.Auth = nil
	tr, release, err := defaultTransport(ctx, plain, f.deps())
	if err != nil {
		t.Fatalf("defaultTransport: %v", err)
	}
	release()
	st, ok := tr.(*sdk.StreamableClientTransport)
	if !ok {
		t.Fatalf("transport is %T, want *sdk.StreamableClientTransport", tr)
	}
	if st.OAuthHandler != nil {
		t.Error("a server without auth got an OAuth handler")
	}

	tr, release, err = defaultTransport(ctx, f.cfg(), f.deps())
	if err != nil {
		t.Fatalf("defaultTransport with auth: %v", err)
	}
	release()
	st, ok = tr.(*sdk.StreamableClientTransport)
	if !ok {
		t.Fatalf("transport is %T, want *sdk.StreamableClientTransport", tr)
	}
	if st.OAuthHandler == nil {
		t.Error("a server with auth got no OAuth handler")
	}
}

func TestConnectRequiresTokenStoreForAuth(t *testing.T) {
	as := newFakeAS(t, asReply{status: 200, body: okBody("at2", "rt2", 3600)})
	f := newOAuthFixture(t, as, time.Hour)
	deps := f.deps()
	deps.TokenStore = nil

	_, err := Connect(context.Background(), f.cfg(), deps)
	if err == nil {
		t.Fatal("Connect succeeded without a token store")
	}
	if !strings.Contains(err.Error(), "TokenStore") {
		t.Errorf("error %q does not name the missing dependency", err)
	}
}

func TestAuthDeadShortCircuitsCalls(t *testing.T) {
	as := newFakeAS(t, asReply{status: 400, body: `{"error":"invalid_grant"}`})
	f := newOAuthFixture(t, as, time.Hour)
	srv := connectFake(t, &fakeConn{defs: callToolset()}, testCfg("s", config.MCPToolFilter{}), testDeps(nil))

	h := f.handler(t)
	h.onDead = srv.markAuthDead

	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, oauthResource, nil)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Authorization", "Bearer at1")
	if err := h.Authorize(context.Background(), req, &http.Response{StatusCode: 401, Body: http.NoBody}); err == nil {
		t.Fatal("Authorize succeeded on a stable invalid_grant")
	}

	tool := toolNamed(t, srv, "s__echo")
	for i := range 2 {
		text, out, err := tool.InvokeOutcome(context.Background(), `{"text":"hi"}`)
		if err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
		if out.Kind != tools.OutcomeToolError {
			t.Errorf("call %d outcome = %v, want tool_error", i, out.Kind)
		}
		if !strings.Contains(text, "authorization dead") {
			t.Errorf("call %d text = %q", i, text)
		}
	}
	if n := as.count(); n != 1 {
		t.Errorf("token endpoint hit %d times, want 1 (no traffic after the death)", n)
	}
	if got := srv.Errors(); got != 1 {
		t.Errorf("Errors() = %d, want the death counted exactly once", got)
	}
}

func TestPickCredential(t *testing.T) {
	entry := func(id string) oauthtoken.Entry {
		return oauthtoken.Entry{Key: oauthtoken.Key{Resource: oauthResource, ClientID: id}}
	}
	tests := []struct {
		name     string
		matches  []oauthtoken.Entry
		clientID string
		want     string
		wantErr  string
	}{
		{name: "single", matches: []oauthtoken.Entry{entry("a")}, want: "a"},
		{name: "config picks", matches: []oauthtoken.Entry{entry("a"), entry("b")}, clientID: "b", want: "b"},
		{name: "config misses", matches: []oauthtoken.Entry{entry("a")}, clientID: "b", wantErr: "client_id \"b\""},
		{name: "ambiguous", matches: []oauthtoken.Entry{entry("b"), entry("a")}, wantErr: "client ids: a, b"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := pickCredential(tc.matches, tc.clientID, oauthResource)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("error = %v, want one containing %q", err, tc.wantErr)
				}
				if !errors.Is(err, errAuthDenied) {
					t.Errorf("error does not wrap errAuthDenied: %v", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("pickCredential: %v", err)
			}
			if got.Key.ClientID != tc.want {
				t.Errorf("client id = %q, want %q", got.Key.ClientID, tc.want)
			}
		})
	}
}

func TestCheckTokenEndpointRefusesPlaintext(t *testing.T) {
	tests := []struct {
		name, raw string
		ok        bool
	}{
		{name: "https", raw: "https://as.example/token", ok: true},
		{name: "http", raw: "http://as.example/token"},
		{name: "loopback http", raw: "http://127.0.0.1:9000/token"},
		{name: "relative", raw: "/token"},
		{name: "empty", raw: ""},
		{name: "unparseable", raw: "://nope"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := checkTokenEndpoint(tc.raw)
			if tc.ok != (err == nil) {
				t.Fatalf("checkTokenEndpoint(%q) = %v, want ok=%v", tc.raw, err, tc.ok)
			}
		})
	}
}

func TestTokenLifetimeIsBounded(t *testing.T) {
	tests := []struct {
		in   int64
		want time.Duration
	}{
		{in: 0, want: defaultTokenLifetime},
		{in: -1, want: defaultTokenLifetime},
		{in: 3600, want: time.Hour},
		{in: 60 * 60 * 24 * 365, want: maxTokenLifetime},
		{in: 1 << 62, want: maxTokenLifetime}, // an absurd claim must not overflow into the past
	}
	for _, tc := range tests {
		if got := tokenLifetime(tc.in); got != tc.want {
			t.Errorf("tokenLifetime(%d) = %v, want %v", tc.in, got, tc.want)
		}
	}
}

func TestRefusalQuotesTheCodeOnly(t *testing.T) {
	err := refusal("400 Bad Request", []byte(`{"error":"invalid_client","error_description":"secret sauce"}`))
	if !errors.Is(err, errAuthDenied) {
		t.Errorf("error does not wrap errAuthDenied: %v", err)
	}
	if strings.Contains(err.Error(), "secret sauce") {
		t.Errorf("error quotes the server's description: %v", err)
	}
	if !strings.Contains(err.Error(), "invalid_client") {
		t.Errorf("error drops the code: %v", err)
	}
	if got := refusal("403 Forbidden", []byte("not json")); !strings.Contains(got.Error(), "403 Forbidden") {
		t.Errorf("unparseable body error = %v, want the status quoted", got)
	}
}

func TestApplyTokenResponse(t *testing.T) {
	base := &oauthtoken.Record{AccessToken: "at1", RefreshToken: "rt1", Scopes: []string{"repo"}}
	now := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	tests := []struct {
		name, body string
		wantErr    error
	}{
		{name: "html", body: "<html>captive portal</html>", wantErr: errTransientAuth},
		{name: "no access token", body: `{"token_type":"Bearer"}`, wantErr: errTransientAuth},
		{name: "dpop", body: `{"access_token":"a","token_type":"DPoP"}`, wantErr: errAuthDenied},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := applyTokenResponse(base, []byte(tc.body), now); !errors.Is(err, tc.wantErr) {
				t.Fatalf("error = %v, want %v", err, tc.wantErr)
			}
		})
	}
	got, err := applyTokenResponse(base, []byte(`{"access_token":"at2","scope":"repo issues"}`), now)
	if err != nil {
		t.Fatalf("applyTokenResponse: %v", err)
	}
	if got.RefreshToken != "rt1" || got.AccessToken != "at2" {
		t.Errorf("record = %+v, want at2 with the old refresh token", got)
	}
	if len(got.Scopes) != 2 || got.Scopes[1] != "issues" {
		t.Errorf("scopes = %v, want the granted set", got.Scopes)
	}
	if !got.ExpiresAt.Equal(now.Add(defaultTokenLifetime)) {
		t.Errorf("expiry = %v, want now+%v", got.ExpiresAt, defaultTokenLifetime)
	}
}

func TestConnectStopsAfterTerminalAuthDenial(t *testing.T) {
	var attempts int
	useTransport(t, func(context.Context, config.MCPServer, Deps) (sdk.Transport, func(), error) {
		attempts++
		return nil, nil, fmt.Errorf("%w: refresh token rejected", errAuthDenied)
	})
	as := newFakeAS(t, asReply{status: 400, body: `{"error":"invalid_grant"}`})
	f := newOAuthFixture(t, as, time.Hour)

	if _, err := Connect(context.Background(), f.cfg(), f.deps()); err == nil {
		t.Fatal("Connect succeeded on a terminal auth denial")
	}
	if attempts != 1 {
		t.Errorf("connect attempts = %d, want 1: a refused credential fails identically on every retry", attempts)
	}
}

func TestForcedRefreshRepeatedIsAuthDeath(t *testing.T) {
	// The token endpoint always answers; the MCP server is the one refusing
	// (a 403 on a token whose scopes do not cover the tool).
	as := newFakeAS(t,
		asReply{status: 200, body: okBody("at2", "rt2", 3600)},
		asReply{status: 200, body: okBody("at3", "rt3", 3600)},
	)
	f := newOAuthFixture(t, as, time.Hour)
	h := f.handler(t)

	forbid := func(token string) error {
		req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, oauthResource, nil)
		if err != nil {
			t.Fatalf("NewRequest: %v", err)
		}
		req.Header.Set("Authorization", "Bearer "+token)
		return h.Authorize(context.Background(), req, &http.Response{StatusCode: http.StatusForbidden, Body: http.NoBody})
	}

	// Call 1: the token is valid, the server refuses it anyway; one forced
	// refresh is fair.
	tok, err := tokenOf(t, h)
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if err := forbid(tok); err != nil {
		t.Fatalf("first Authorize: %v", err)
	}
	if n := as.count(); n != 1 {
		t.Fatalf("token endpoint hit %d times, want 1", n)
	}
	// Call 2: the freshly refreshed token is refused too. Refreshing again
	// would rotate the credential once per tool call for the rest of the run.
	tok, err = tokenOf(t, h)
	if err != nil {
		t.Fatalf("second Token: %v", err)
	}
	if err := forbid(tok); err == nil {
		t.Fatal("second Authorize succeeded on a server that keeps refusing")
	}
	if n := as.count(); n != 1 {
		t.Errorf("token endpoint hit %d times, want no second refresh", n)
	}
	if len(f.dead) != 1 {
		t.Errorf("onDead called %d times, want 1", len(f.dead))
	}
	if _, err := tokenOf(t, h); err == nil {
		t.Error("Token succeeded after the death")
	}
}

func TestStringExpiresInAccepted(t *testing.T) {
	as := newFakeAS(t, asReply{status: 200,
		body: `{"access_token":"at2","refresh_token":"rt2","token_type":"Bearer","expires_in":"3600"}`})
	f := newOAuthFixture(t, as, -time.Minute)
	h := f.handler(t)

	got, err := tokenOf(t, h)
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if got != "at2" {
		t.Errorf("access token = %q, want at2", got)
	}
	rec, err := f.store.Load(f.key)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if rec.RefreshToken != "rt2" {
		t.Errorf("refresh token = %q, want the rotated rt2 persisted", rec.RefreshToken)
	}
	if want := f.now.Add(time.Hour); !rec.ExpiresAt.Equal(want) {
		t.Errorf("expiry = %v, want %v", rec.ExpiresAt, want)
	}
}

func TestInvalidGrantAdoptsRefreshOnlyRotation(t *testing.T) {
	as := newFakeAS(t, asReply{status: 400, body: `{"error":"invalid_grant"}`})
	f := newOAuthFixture(t, as, -time.Minute)
	h := f.handler(t)
	// The other process rotated ONLY the refresh token (an authorization
	// server may hand back the same access token) and pushed the expiry out.
	as.hook = func(int) { f.save(t, "at1", "rt7", time.Hour) }

	if _, err := tokenOf(t, h); err != nil {
		t.Fatalf("Token: %v", err)
	}
	if len(f.dead) != 0 {
		t.Errorf("credential declared dead despite a concurrent rotation: %v", f.dead)
	}
}

func TestTokenSourceTerminalFailureSetsAuthDead(t *testing.T) {
	as := newFakeAS(t, asReply{status: 400, body: `{"error":"invalid_grant"}`})
	f := newOAuthFixture(t, as, -time.Minute)
	h := f.handler(t)

	// A terminal failure that never produced a 401 (the request was never
	// sent) must still end this run's authorization, or every call would
	// re-POST a doomed refresh.
	if _, err := tokenOf(t, h); err == nil {
		t.Fatal("Token succeeded on a stable invalid_grant")
	}
	if len(f.dead) != 1 {
		t.Fatalf("onDead called %d times, want 1", len(f.dead))
	}
	if _, err := tokenOf(t, h); err == nil {
		t.Fatal("second Token succeeded after the death")
	}
	if n := as.count(); n != 1 {
		t.Errorf("token endpoint hit %d times, want 1", n)
	}
}

func TestOAuthClientRefusesRedirect(t *testing.T) {
	// SECURITY: the refresh token travels in the request body; a redirect
	// would replay it to a location the stored token endpoint never named.
	if err := newOAuthHTTPClient().CheckRedirect(nil, nil); err == nil {
		t.Error("the token endpoint client follows redirects")
	}
}

func TestAuthDeathDuringReconnectCountsOnce(t *testing.T) {
	inner := (&fakeConn{defs: callToolset()}).factory()
	var calls int
	useTransport(t, func(ctx context.Context, cfg config.MCPServer, deps Deps) (sdk.Transport, func(), error) {
		calls++
		if calls == 1 {
			return inner(ctx, cfg, deps)
		}
		// What the OAuth handler does when the refresh is refused mid-run.
		err := fmt.Errorf("%w: refresh token rejected", errAuthDenied)
		if deps.authFailed != nil {
			deps.authFailed(err)
		}
		return nil, nil, err
	})
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	srv, err := Connect(ctx, testCfg("s", config.MCPToolFilter{}), testDeps(nil))
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() { _ = srv.Close(context.Background()) })

	// The session is condemned directly rather than through a dropped fake
	// connection: what this test is about starts at the RECONNECT, and driving
	// a real lost response first would make it depend on the scheduler.
	srv.mu.Lock()
	sess := srv.sess
	srv.mu.Unlock()
	srv.markDead(sess)
	srv.countError() // the lost response that condemned the session
	if got := srv.Errors(); got != 1 {
		t.Fatalf("Errors() = %d after the loss, want 1", got)
	}
	// The reconnect fails on a refused credential: ONE event, one count.
	if _, _, err := toolNamed(t, srv, "s__echo").InvokeOutcome(context.Background(), `{"text":"x"}`); err == nil {
		t.Fatal("call succeeded after a refused credential")
	}
	if got := srv.Errors(); got != 2 {
		t.Errorf("Errors() = %d, want 2 (the lost response plus one death)", got)
	}
	if srv.authDeadErr() == nil {
		t.Error("the server was not marked auth-dead")
	}
}

func TestFlexIntAcceptsTheSpellingsServersSend(t *testing.T) {
	tests := []struct {
		raw  string
		want flexInt
	}{
		{raw: `3600`, want: 3600},
		{raw: `"3600"`, want: 3600},
		{raw: `3599.5`, want: 3599},
		{raw: `null`, want: 0},
		{raw: `"soon"`, want: 0},
		{raw: `""`, want: 0},
	}
	for _, tc := range tests {
		var got flexInt
		if err := json.Unmarshal([]byte(tc.raw), &got); err != nil {
			t.Errorf("Unmarshal(%s): %v", tc.raw, err)
			continue
		}
		if got != tc.want {
			t.Errorf("Unmarshal(%s) = %d, want %d", tc.raw, got, tc.want)
		}
	}
}

func TestStaleRejectedAdoptsAfterForcedRefresh(t *testing.T) {
	as := newFakeAS(t,
		asReply{status: 200, body: okBody("at2", "rt2", 3600)},
		asReply{status: 200, body: okBody("at3", "rt3", 3600)},
	)
	f := newOAuthFixture(t, as, time.Hour)
	h := f.handler(t)

	authorize := func(rejected string) error {
		req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, oauthResource, nil)
		if err != nil {
			t.Fatalf("NewRequest: %v", err)
		}
		req.Header.Set("Authorization", "Bearer "+rejected)
		return h.Authorize(context.Background(), req, &http.Response{StatusCode: 401, Body: http.NoBody})
	}

	// One call's 401 buys the forced refresh at1 -> at2.
	if err := authorize("at1"); err != nil {
		t.Fatalf("first Authorize: %v", err)
	}
	// A SECOND call was in flight with the old token and is refused too. Its
	// rejection says nothing about at2: the credential was already replaced,
	// and the run must simply adopt it - not declare the server dead.
	if err := authorize("at1"); err != nil {
		t.Fatalf("stale-token Authorize: %v", err)
	}
	if n := as.count(); n != 1 {
		t.Errorf("token endpoint hit %d times, want 1: a stale rejection needs no refresh", n)
	}
	if len(f.dead) != 0 {
		t.Errorf("a stale rejection killed a healthy credential: %v", f.dead)
	}
	got, err := tokenOf(t, h)
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if got != "at2" {
		t.Errorf("access token = %q, want at2", got)
	}
}

// closedPortURL returns an https URL nothing listens on, so a request to it
// fails with a connection-refused *url.Error - the shape a token endpoint
// takes when the authorization server is unreachable.
func closedPortURL(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}
	addr := l.Addr().String()
	if err := l.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	return "https://" + addr + "/token"
}

// TestTransientRefreshMidRunIsPlainToolError drives a REAL SDK session whose
// token goes stale mid-run while the authorization server is unreachable.
//
// CONTRACT: a refresh that never left amele is a plain tool error. Reading it
// as a lost response would tell the model an action MAY have happened, tear
// down a healthy session and count an error per call.
func TestTransientRefreshMidRunIsPlainToolError(t *testing.T) {
	ts, _ := startHTTPServer(t)
	tokenEndpoint := closedPortURL(t)

	base := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	// The clock is read from the SDK's goroutines, so it must be race-free.
	var offset atomic.Int64
	now := func() time.Time { return base.Add(time.Duration(offset.Load())) }

	store := oauthtoken.NewStore(t.TempDir(), now)
	resource, err := oauthtoken.CanonicalResource(ts.URL)
	if err != nil {
		t.Fatalf("CanonicalResource: %v", err)
	}
	key := oauthtoken.Key{Issuer: "https://as.example", Resource: resource, ClientID: "cid"}
	if err := store.Save(key, &oauthtoken.Record{
		Version:       oauthtoken.Version,
		Issuer:        key.Issuer,
		Resource:      resource,
		ClientID:      "cid",
		TokenEndpoint: tokenEndpoint,
		AccessToken:   "at1",
		RefreshToken:  "rt1",
		ExpiresAt:     base.Add(time.Hour),
	}); err != nil {
		t.Fatalf("Save: %v", err)
	}

	cfg := config.MCPServer{
		Name:      "s",
		Transport: config.MCPTransport{Type: config.MCPTransportHTTP, URL: ts.URL},
		Auth:      &config.MCPAuth{Type: config.MCPAuthOAuth, ClientID: "cid"},
	}
	deps := testDeps(nil)
	deps.TokenStore = store
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	srv, err := Connect(ctx, cfg, deps)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = srv.Close(context.Background()) }()

	tool := toolNamed(t, srv, "s__echo")
	callOK := func(what, word string) {
		t.Helper()
		text, out, err := tool.InvokeOutcome(ctx, fmt.Sprintf(`{"text":%q}`, word))
		if err != nil || out.Kind != tools.OutcomeOK || text != word {
			t.Fatalf("%s = (%q, %v, %v), want (%s, ok, nil)", what, text, out.Kind, err, word)
		}
	}
	callOK("first call", "one")

	// The stored token ages out while the authorization server is down.
	offset.Store(int64(2 * time.Hour))
	text, out, err := tool.InvokeOutcome(ctx, `{"text":"two"}`)
	if err != nil {
		t.Fatalf("call during a transient refresh failure: %v", err)
	}
	if out.Kind != tools.OutcomeToolError {
		t.Errorf("outcome = %v (text %q), want tool_error: the request never left amele", out.Kind, text)
	}
	if strings.Contains(text, "response lost") {
		t.Errorf("text = %q, want no lost-response wording", text)
	}
	if srv.authDeadErr() != nil {
		t.Errorf("a transient refresh failure killed the credential: %v", srv.authDeadErr())
	}
	if got := srv.Errors(); got != 0 {
		t.Errorf("Errors() = %d, want 0: an unreachable token endpoint is not a transport failure", got)
	}
	srv.mu.Lock()
	dead := srv.dead
	srv.mu.Unlock()
	if dead {
		t.Error("a healthy session was marked dead")
	}

	// The authorization server comes back (here: the token is fresh again) and
	// the very next call goes through on the same session.
	offset.Store(0)
	callOK("retry after the transient failure", "three")
}

func TestClampMargin(t *testing.T) {
	for _, tc := range []struct {
		name     string
		margin   time.Duration
		lifetime time.Duration
		want     time.Duration
	}{
		{name: "lifetime not observed yet", margin: 90 * time.Second, want: 90 * time.Second},
		{name: "long lived token keeps the margin", margin: 90 * time.Second, lifetime: time.Hour, want: 90 * time.Second},
		{name: "short lived token is clamped", margin: 90 * time.Second, lifetime: 60 * time.Second, want: 15 * time.Second},
		{name: "exactly four margins", margin: 60 * time.Second, lifetime: 240 * time.Second, want: 60 * time.Second},
		{name: "negative lifetime is unknown", margin: 60 * time.Second, lifetime: -time.Hour, want: 60 * time.Second},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := clampMargin(tc.margin, tc.lifetime); got != tc.want {
				t.Errorf("clampMargin(%v, %v) = %v, want %v", tc.margin, tc.lifetime, got, tc.want)
			}
		})
	}
}

// TestShortLivedTokenRefreshedOnce proves the clamp on the live path: a
// 60-second token must not be refreshed by every call just because the margin
// is longer than the token's whole life.
func TestShortLivedTokenRefreshedOnce(t *testing.T) {
	as := newFakeAS(t, asReply{status: 200, body: okBody("at2", "rt2", 60)})
	f := newOAuthFixture(t, as, -time.Minute) // stale: the first call must refresh
	h := f.handler(t)

	for i := range 3 {
		got, err := tokenOf(t, h)
		if err != nil {
			t.Fatalf("call %d: %v", i, err)
		}
		if got != "at2" {
			t.Fatalf("call %d token = %q, want at2", i, got)
		}
	}
	if n := as.count(); n != 1 {
		t.Errorf("token endpoint hit %d times, want 1: a 60s token must not be refreshed per call", n)
	}
}
