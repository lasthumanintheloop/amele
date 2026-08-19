package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/oauthtoken"
	"golang.org/x/oauth2"
)

// Budgets and timings of the refresh path. Like the connection budgets above
// they are constants, not config: an operator who could raise them would be
// removing his own safety net.
const (
	// refreshMargin is how early a token is treated as expired. A token that
	// dies mid-flight costs a retry and a 401; refreshing a minute early costs
	// nothing. Deps.Rand adds up to the same amount again as jitter, so a
	// fleet of cron workers sharing one credential does not all refresh on the
	// same second (the same reasoning as the connect jitter).
	refreshMargin = 60 * time.Second
	// refreshTimeout bounds ONE token request. It is deliberately shorter than
	// ConnectTimeout: the token endpoint is a small JSON POST, and a run must
	// not hang on an authorization server that accepted the connection and
	// then went quiet.
	refreshTimeout = 30 * time.Second
	// maxTokenResponseBytes caps a token endpoint response. A token response
	// is a few hundred bytes; anything near this cap is a misdirected request
	// or an attack.
	maxTokenResponseBytes = 1 << 20
	// defaultTokenLifetime is assumed when the authorization server omits
	// expires_in (RFC 6749 makes it optional). Short on purpose: an unknown
	// lifetime must expire soon enough that the next refresh is cheap, and
	// never be treated as permanent.
	defaultTokenLifetime = 5 * time.Minute
	// maxTokenLifetime caps what a server may claim. A peer announcing a
	// ten-year access token must not stop amele from ever refreshing.
	maxTokenLifetime = 24 * time.Hour
)

var (
	// errTransientAuth marks a refresh failure that must NOT kill the
	// credential or be cached (authorization server 5xx, 429, network, a
	// response that is not a token response at all): the next call retries.
	// CONTRACT: classify reports it as ClassNetwork - it IS an availability
	// problem, not a rejected credential.
	errTransientAuth = errors.New("transient authorization failure")
	// errAuthDenied marks a TERMINAL authorization failure: the credential
	// cannot be used and no retry in this run can change that. Every error
	// that may end a server's authorization wraps it, which is what makes the
	// verdict greppable and classifiable without matching on text.
	errAuthDenied = errors.New("oauth authorization failed")
	// errInvalidGrant is the RFC 6749 §5.2 rejection of the refresh token
	// itself. It is separate from errAuthDenied because it is the one failure
	// that earns a second look at the store before the credential is declared
	// dead: a concurrent process may have rotated it a moment ago.
	errInvalidGrant = errors.New("invalid_grant")
	// errNoTokenStore is the wiring error for a server that declares auth in a
	// process that never built a token store. It is returned rather than
	// panicked for the same reason the required Deps are (see withDefaults).
	errNoTokenStore = errors.New("mcp: Deps.TokenStore is required for a server with auth")
)

// loginHint is appended to every "you have no usable credential" error. It is
// one sentence because it lands in a cron mail, where the operator needs the
// command and nothing else.
const loginHint = " - run 'amele mcp login'"

// newOAuthClient builds the HTTP client the refresh path uses. It is a package
// variable ONLY so tests can substitute a client that trusts an httptest
// certificate; production has exactly one implementation, and no configuration
// reaches it.
var newOAuthClient = newOAuthHTTPClient

// newOAuthHTTPClient returns the hardened client used for token requests: the
// same discipline as newHTTPTransport (response body cap, bounded time) minus
// everything that only makes sense for a streaming MCP session.
//
// SECURITY: redirects are refused outright. The request body carries the
// refresh token, and a redirect - even a same-origin one - would replay that
// credential to a location the stored token endpoint never named.
func newOAuthHTTPClient() *http.Client {
	// Comma-ok, never a bare assertion: a library must not panic because some
	// other package replaced http.DefaultTransport (CLAUDE.md 5.3).
	base, ok := http.DefaultTransport.(*http.Transport)
	if !ok {
		base = &http.Transport{Proxy: http.ProxyFromEnvironment}
	} else {
		base = base.Clone()
	}
	return &http.Client{
		Timeout:   refreshTimeout,
		Transport: &cappedBodyRoundTripper{next: base, max: maxTokenResponseBytes},
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return errors.New("token endpoint must not redirect")
		},
	}
}

// runOAuthHandler implements the SDK's auth.OAuthHandler for `run` and `chat`:
// it serves tokens from the store, refreshing them under the store's
// cross-process lock, and NEVER opens a browser. Acquiring a credential
// interactively is `amele mcp login`'s job; a headless run either has a usable
// refresh token or fails with an instruction.
//
// CONTRACT (spec §5): a mid-run 401 is an explicit refusal, not a lost
// response. Authorize gets exactly one forced re-read-and-refresh; a terminal
// failure is cached as auth death so the rest of the run neither spams the
// authorization server nor pays a round trip per call.
//
// A handler is safe for concurrent use: the SDK asks for a token on every
// outgoing request, from whichever goroutine is making the call.
type runOAuthHandler struct {
	store  *oauthtoken.Store
	key    oauthtoken.Key
	client *http.Client
	// margin is refreshMargin plus this handler's jitter, drawn once at
	// construction so repeated freshness checks inside one run agree with each
	// other while different processes still spread out.
	margin time.Duration
	// register receives every token value that leaves the store, so the run's
	// redactor scrubs it from the session log, the progress output and the MCP
	// stderr relay. nil is allowed (a caller that keeps no log).
	register func(...string)
	// onDead is called once, with the terminal error, when this credential
	// dies. It is how the Server learns to stop dispatching calls. nil is
	// allowed (the handler still caches the verdict for itself).
	onDead func(error)

	// mu serializes the whole token path, including the file lock. Holding it
	// across the lock is deliberate: it makes in-process refreshes
	// single-flight, so a turn with eight parallel tool calls performs one
	// refresh rather than eight that would each rotate the previous one's
	// refresh token.
	mu sync.Mutex
	// rec is the credential this handler last served or adopted.
	rec *oauthtoken.Record
	// authDead is the cached terminal verdict; non-nil means every further
	// request fails immediately with it.
	authDead error
	// lastAccess and lastRefresh are the values already handed to register;
	// the secret set has no de-duplication, so a token is registered once.
	lastAccess, lastRefresh string
}

// Compile-time proof that the handler is what the SDK transport accepts. The
// interface lives in the SDK's auth package; asserting it here means a
// signature drift breaks the build rather than a run.
var _ interface {
	TokenSource(context.Context) (oauth2.TokenSource, error)
	Authorize(context.Context, *http.Request, *http.Response) error
} = (*runOAuthHandler)(nil)

// newRunOAuthHandler builds the handler for one OAuth-enabled server.
//
// It resolves the credential WITHOUT contacting the authorization server: the
// stored record carries its own issuer and token endpoint, so a run needs no
// discovery round trip (and, on a headless box, no network at all until a
// refresh is actually due). The lookup is by canonical resource, because that
// is the only part of the key a config states; the issuer comes from whatever
// `amele mcp login` stored, and the client id disambiguates when one resource
// was logged in under more than one registration.
//
// An error here is terminal for the connection: it wraps errAuthDenied so
// classify reports `auth` and the operator's line names the login command.
func newRunOAuthHandler(cfg config.MCPServer, deps Deps) (*runOAuthHandler, error) {
	if cfg.Auth == nil {
		return nil, errors.New("mcp: newRunOAuthHandler called for a server without auth")
	}
	if deps.TokenStore == nil {
		return nil, errNoTokenStore
	}
	resource, err := oauthtoken.CanonicalResource(cfg.Transport.URL)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", errAuthDenied, err)
	}
	entries, listErr := deps.TokenStore.List()
	var matches []oauthtoken.Entry
	for _, e := range entries {
		if e.Key.Resource == resource {
			matches = append(matches, e)
		}
	}
	// listErr reports files that could not be read or decoded. It is only
	// fatal when nothing usable was found: one stray file in the token
	// directory must not take down a run whose credential is right there.
	if len(matches) == 0 {
		if listErr != nil {
			return nil, fmt.Errorf("%w: no oauth token for %q (%v)%s", errAuthDenied, resource, listErr, loginHint)
		}
		return nil, fmt.Errorf("%w: no oauth token for %q%s", errAuthDenied, resource, loginHint)
	}
	entry, err := pickCredential(matches, cfg.Auth.ClientID, resource)
	if err != nil {
		return nil, err
	}
	if err := checkTokenEndpoint(entry.Record.TokenEndpoint); err != nil {
		return nil, fmt.Errorf("%w: stored credential for %q: %w", errAuthDenied, resource, err)
	}
	h := &runOAuthHandler{
		store:    deps.TokenStore,
		key:      entry.Key,
		client:   newOAuthClient(),
		margin:   refreshMargin + time.Duration(deps.Rand()*float64(refreshMargin)),
		register: deps.RegisterSecret,
	}
	h.adopt(entry.Record)
	return h, nil
}

// pickCredential chooses among the credentials stored for one resource.
//
// Several are possible because the key includes the client id: the same server
// can be logged into with a pre-registered client and with a discovered one.
// The config's client_id decides; without one, an ambiguous store is an error
// rather than a guess, because guessing would silently send a run's requests
// under an identity the operator did not choose.
func pickCredential(matches []oauthtoken.Entry, clientID, resource string) (oauthtoken.Entry, error) {
	if clientID != "" {
		for _, e := range matches {
			if e.Key.ClientID == clientID {
				return e, nil
			}
		}
		return oauthtoken.Entry{}, fmt.Errorf("%w: no oauth token for %q with client_id %q%s",
			errAuthDenied, resource, clientID, loginHint)
	}
	if len(matches) == 1 {
		return matches[0], nil
	}
	ids := make([]string, 0, len(matches))
	for _, e := range matches {
		ids = append(ids, e.Key.ClientID)
	}
	slices.Sort(ids) // directory order must not decide what an error message says
	return oauthtoken.Entry{}, fmt.Errorf("%w: %d oauth tokens stored for %q (client ids: %s); set auth.client_id to choose one",
		errAuthDenied, len(matches), resource, strings.Join(ids, ", "))
}

// checkTokenEndpoint refuses a stored token endpoint amele will not POST to.
//
// SECURITY: https only, with no loopback exception. The request carries the
// refresh token, which is the long-lived half of the credential; a plaintext
// token endpoint would hand it to anyone on the path. A record with no
// endpoint at all cannot be refreshed and is refused here rather than at the
// first expiry, so the operator hears about it at connect time.
func checkTokenEndpoint(raw string) error {
	if raw == "" {
		return errors.New("no token endpoint recorded")
	}
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("parsing token endpoint: %w", err)
	}
	if u.Scheme != "https" || u.Host == "" {
		return fmt.Errorf("token endpoint %q must be an absolute https url", raw)
	}
	return nil
}

// TokenSource implements auth.OAuthHandler. The returned source reads through
// to the store on every call, so a token another process refreshed is picked
// up without reconnecting.
//
// ctx is the transport's context and bounds the refreshes the source performs;
// oauth2.TokenSource.Token takes none of its own, which is why it is captured.
func (h *runOAuthHandler) TokenSource(ctx context.Context) (oauth2.TokenSource, error) {
	return &storeTokenSource{h: h, ctx: ctx}, nil
}

// storeTokenSource adapts the handler to oauth2.TokenSource.
type storeTokenSource struct {
	h *runOAuthHandler
	//nolint:containedctx // oauth2.TokenSource.Token takes no context; the transport's is captured here on purpose.
	ctx context.Context
}

// Token implements oauth2.TokenSource.
func (s *storeTokenSource) Token() (*oauth2.Token, error) {
	return s.h.token(s.ctx, false, "")
}

// Authorize implements auth.OAuthHandler: the SDK calls it when a request came
// back 401 or 403, and retries the request once if it returns nil.
//
// CONTRACT (spec §5): this NEVER opens a browser. It forces one locked
// re-read-and-refresh; a terminal failure becomes auth death (cached here and
// reported to the Server, which stops dispatching), while a transient one is
// returned without being cached so the next call may try again.
func (h *runOAuthHandler) Authorize(ctx context.Context, req *http.Request, resp *http.Response) error {
	// The interface makes the handler responsible for the response body.
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	h.mu.Lock()
	dead := h.authDead
	h.mu.Unlock()
	if dead != nil {
		return dead
	}
	// The token the server just rejected. Under the lock it decides between
	// "another worker already replaced this token" (adopt) and "this
	// credential really was refused" (refresh).
	var rejected string
	if req != nil {
		if v := req.Header.Get("Authorization"); len(v) > len("Bearer ") && strings.EqualFold(v[:7], "bearer ") {
			rejected = v[7:]
		}
	}
	_, err := h.token(ctx, true, rejected)
	return err
}

// token returns a usable access token.
//
// force skips the in-memory fast path (Authorize's one forced refresh);
// rejected, when non-empty, names the access token the server just refused, so
// a record on disk that already carries a DIFFERENT token is adopted instead
// of triggering a needless refresh.
func (h *runOAuthHandler) token(ctx context.Context, force bool, rejected string) (*oauth2.Token, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.authDead != nil {
		return nil, h.authDead
	}
	if !force && h.rec != nil && h.rec.Fresh(h.store.Now(), h.margin) {
		return bearer(h.rec), nil
	}

	next, err := h.store.WithLock(ctx, h.key, func(cur *oauthtoken.Record) (*oauthtoken.Record, error) {
		return h.refreshLocked(ctx, cur, force, rejected)
	})
	if err != nil {
		// Only an explicit refusal is cached. A lock failure, a cancelled
		// context or an unreachable authorization server say nothing about
		// whether the credential is still good.
		if errors.Is(err, errAuthDenied) {
			h.die(err)
		}
		return nil, err
	}
	h.adopt(next)
	return bearer(next), nil
}

// refreshLocked is the body of the refresh protocol, run while this process
// holds the credential's cross-process lock and with cur re-read under it.
//
// It returns the record to serve: cur unchanged when someone else already did
// the work, a refreshed one otherwise. The store persists whatever comes back
// only if it differs, so the adopt paths cost no write.
func (h *runOAuthHandler) refreshLocked(ctx context.Context, cur *oauthtoken.Record, force bool, rejected string) (*oauthtoken.Record, error) {
	if cur == nil {
		// The credential was removed under us (`amele mcp logout`, or a wiped
		// state directory).
		return nil, fmt.Errorf("%w: no oauth token for %q%s", errAuthDenied, h.key.Resource, loginHint)
	}
	// Step 2 of the refresh protocol: someone else may have refreshed while
	// this call waited for the lock.
	if force {
		if rejected != "" && cur.AccessToken != rejected {
			return cur, nil
		}
	} else if cur.Fresh(h.store.Now(), h.margin) {
		return cur, nil
	}
	updated, err := refreshGrant(ctx, h.client, cur, h.key.Resource, h.store.Now())
	if err == nil {
		return updated, nil
	}
	if errors.Is(err, errInvalidGrant) {
		// Spec §4: read the file once more before condemning the credential -
		// a concurrent process may have completed its own rotation between
		// this call's read and its POST, in which case the refresh token amele
		// just sent was merely the OLD one. The record is never deleted here;
		// that is `amele mcp logout`'s job.
		if again, lerr := h.store.Load(h.key); lerr == nil && again.AccessToken != cur.AccessToken {
			return again, nil
		}
	}
	return nil, err
}

// die records the terminal verdict and tells the Server once.
func (h *runOAuthHandler) die(err error) {
	if h.authDead != nil {
		return
	}
	h.authDead = err
	if h.onDead != nil {
		h.onDead(err)
	}
}

// adopt takes rec as the current credential and registers any token value the
// run's redactor has not seen yet.
//
// SECURITY: every token that leaves the store passes through here, which is
// what keeps a refreshed access token out of the session log even though the
// redactor was built before the run started.
func (h *runOAuthHandler) adopt(rec *oauthtoken.Record) {
	h.rec = rec
	if h.register == nil {
		return
	}
	var fresh []string
	if rec.AccessToken != "" && rec.AccessToken != h.lastAccess {
		h.lastAccess = rec.AccessToken
		fresh = append(fresh, rec.AccessToken)
	}
	if rec.RefreshToken != "" && rec.RefreshToken != h.lastRefresh {
		h.lastRefresh = rec.RefreshToken
		fresh = append(fresh, rec.RefreshToken)
	}
	if len(fresh) > 0 {
		h.register(fresh...)
	}
}

// bearer renders a record as the token the transport puts in the header.
func bearer(rec *oauthtoken.Record) *oauth2.Token {
	return &oauth2.Token{AccessToken: rec.AccessToken, TokenType: "Bearer", Expiry: rec.ExpiresAt}
}

// tokenResponse is the RFC 6749 §5.1 success body. Unknown fields are ignored:
// authorization servers routinely add their own.
type tokenResponse struct {
	AccessToken  string `json:"access_token"`
	TokenType    string `json:"token_type"`
	ExpiresIn    int64  `json:"expires_in"`
	RefreshToken string `json:"refresh_token"`
	Scope        string `json:"scope"`
}

// refreshGrant performs one RFC 6749 §6 refresh and returns the updated record.
// It never writes: the caller persists the result under the store's lock.
//
// The POST is written out by hand rather than delegated to
// golang.org/x/oauth2, because the resource indicator (RFC 8707) must travel
// on the refresh request - the MCP authorization spec keys tokens to it - and
// the library's refresh path carries no extra parameters.
//
// Failures are split into two kinds, and the split is the point: an
// authorization server that is down (5xx, 429, network, a body that is not a
// token response) yields errTransientAuth and leaves the credential alone,
// while a refusal (RFC 6749 §5.2) yields an errAuthDenied error that ends this
// server's authorization for the run.
func refreshGrant(ctx context.Context, client *http.Client, rec *oauthtoken.Record, resource string, now time.Time) (*oauthtoken.Record, error) {
	if rec.RefreshToken == "" {
		return nil, fmt.Errorf("%w: stored credential for %q has no refresh token%s", errAuthDenied, resource, loginHint)
	}
	form := url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {rec.RefreshToken},
		// RFC 8707: the token stays bound to the MCP server it was issued for,
		// so a compromised peer cannot replay it against another resource.
		"resource": {resource},
	}
	if rec.ClientID != "" {
		form.Set("client_id", rec.ClientID)
	}
	// A bound of its own on top of the client's: the caller's context may be
	// the whole run's, and a refresh must not be allowed to consume it.
	rctx, cancel := context.WithTimeout(ctx, refreshTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(rctx, http.MethodPost, rec.TokenEndpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, fmt.Errorf("%w: building token request: %w", errAuthDenied, err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%w: token endpoint: %w", errTransientAuth, err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxTokenResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("%w: reading token response: %w", errTransientAuth, err)
	}
	if len(body) > maxTokenResponseBytes {
		return nil, fmt.Errorf("%w: token response over %d bytes", errTransientAuth, maxTokenResponseBytes)
	}

	switch {
	case resp.StatusCode == http.StatusOK:
	case resp.StatusCode >= 500 || resp.StatusCode == http.StatusTooManyRequests:
		return nil, fmt.Errorf("%w: token endpoint returned %s", errTransientAuth, resp.Status)
	default:
		return nil, refusal(resp.Status, body)
	}

	return applyTokenResponse(rec, body, now)
}

// applyTokenResponse folds a successful token response into a new record.
func applyTokenResponse(rec *oauthtoken.Record, body []byte, now time.Time) (*oauthtoken.Record, error) {
	var tr tokenResponse
	if err := json.Unmarshal(body, &tr); err != nil || tr.AccessToken == "" {
		// A 200 that is not a token response proves nothing about the
		// credential - a captive portal or a misrouted proxy answers exactly
		// like this - so it must not kill it.
		return nil, fmt.Errorf("%w: token endpoint returned a body that is not a token response", errTransientAuth)
	}
	if tr.TokenType != "" && !strings.EqualFold(tr.TokenType, "bearer") {
		// amele only knows how to present a bearer token; a DPoP or MAC token
		// would be sent as one and rejected on every call.
		return nil, fmt.Errorf("%w: token endpoint issued an unsupported token_type %q", errAuthDenied, tr.TokenType)
	}

	next := *rec
	next.AccessToken = tr.AccessToken
	next.ExpiresAt = now.Add(tokenLifetime(tr.ExpiresIn))
	// Rotation: a new refresh token replaces the old one, its ABSENCE keeps
	// the old one. Overwriting with an empty string here would lose the only
	// way this machine can ever refresh again.
	if tr.RefreshToken != "" {
		next.RefreshToken = tr.RefreshToken
	}
	if tr.Scope != "" {
		next.Scopes = strings.Fields(tr.Scope)
	}
	return &next, nil
}

// tokenLifetime turns an expires_in value into a bounded duration.
func tokenLifetime(expiresIn int64) time.Duration {
	if expiresIn <= 0 {
		return defaultTokenLifetime
	}
	life := time.Duration(expiresIn) * time.Second
	if life > maxTokenLifetime || life <= 0 { // <= 0 also catches the overflow of an absurd claim
		return maxTokenLifetime
	}
	return life
}

// refusal renders a non-2xx token response as a terminal error.
//
// SECURITY: only the RFC 6749 error CODE and the HTTP status are quoted. The
// error_description is authorization-server text of unbounded length that may
// echo parts of the request, and this string ends up in a cron mail.
func refusal(status string, body []byte) error {
	var parsed struct {
		Error string `json:"error"`
	}
	_ = json.Unmarshal(body, &parsed)
	switch {
	case parsed.Error == "invalid_grant":
		return fmt.Errorf("%w: refresh token rejected (%w)%s", errAuthDenied, errInvalidGrant, loginHint)
	case parsed.Error != "":
		return fmt.Errorf("%w: token endpoint refused the refresh (%s)%s", errAuthDenied, parsed.Error, loginHint)
	default:
		return fmt.Errorf("%w: token endpoint returned %s%s", errAuthDenied, status, loginHint)
	}
}
