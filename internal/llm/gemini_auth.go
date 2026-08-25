package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sync"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

// vertexScope is the only OAuth scope aiplatform.googleapis.com accepts for
// inference. The discovery document declares exactly two - cloud-platform and
// cloud-platform.read-only - and the read-only one cannot generate content, so
// there is nothing to choose between and no knob for it
// (docs/superpowers/specs/2026-08-25-vertex-adc-research.md §3.1).
const vertexScope = "https://www.googleapis.com/auth/cloud-platform"

// googleTokenRefreshMargin is how long before its stated expiry a cached token
// stops being handed out. Google's access tokens live an hour, and a request
// authorized with the last seconds of one would be refused mid-generation, so
// the margin buys the whole round-trip rather than a moment.
const googleTokenRefreshMargin = 60 * time.Second

// googleTokenTimeout bounds one token exchange.
//
// It exists because x/oauth2's JWT leg posts to the token endpoint WITHOUT a
// context (jwt.jwtSource.Token uses hc.PostForm), so the caller's cancellation
// cannot reach it; a client timeout is the only bound available there. The
// refresh-token and external-account legs do carry the context and honor
// cancellation as well.
const googleTokenTimeout = 30 * time.Second

// GoogleTokenSource obtains the OAuth access token Vertex AI requests are
// authenticated with. It implements GeminiTokenSource and GeminiQuotaProject.
//
// It covers the two credential shapes amele supports:
//
//   - CredentialsFile set - the service-account key file named by
//     provider.vertex.credentials. Its private key signs an RS256 JWT which is
//     exchanged for an access token.
//   - CredentialsFile empty - Application Default Credentials, the chain
//     Google documents: the GOOGLE_APPLICATION_CREDENTIALS variable, then
//     gcloud's user credentials file, then the metadata server (which also
//     covers workload identity federation, since a WIF configuration is just
//     another ADC file type).
//
// # Library boundary
//
// Everything below the two calls into golang.org/x/oauth2/google is that
// library's business: reading environment variables, probing the filesystem and
// the metadata server, signing the assertion, and reading the clock inside
// oauth2.Token. docs/engineering.md §5.4 requires amele's own code to take
// time, randomness and the environment through injected seams, and it does -
// Now is injectable here, and the freshness decision is made HERE rather than
// with oauth2.Token.Valid so the cache is testable without a real clock. The
// library's internals are exempt as a vendored dependency: ADC's variable names
// and file locations are Google's published contract, not amele's, and amele's
// knob for them is provider.vertex.credentials.
//
// # Concurrency and goroutines
//
// CONTRACT: safe for concurrent use, and it owns NO goroutine. Tokens are
// fetched on the calling goroutine, lazily, and cached under mu; there is no
// background refresh loop to outlive a run. The mutex is held across the
// exchange on purpose - that makes a turn with several parallel tool calls
// perform one exchange rather than several.
type GoogleTokenSource struct {
	// CredentialsFile is provider.vertex.credentials: the path of a
	// service-account key file. Empty selects the ADC chain.
	//
	// SECURITY: the file is loaded as a service_account key and nothing else.
	// The other ADC file types are reachable through GOOGLE_APPLICATION_
	// CREDENTIALS, which is Google's own contract; accepting them HERE would
	// mean a path in a YAML file could name an external_account configuration
	// whose credential_source runs an executable - the risk x/oauth2 documents
	// on that credential type.
	CredentialsFile string
	// Project is provider.vertex.project. It is used ONLY as the value of the
	// quota-project header, and only for the credential types that need one
	// (see QuotaProject); the request URL gets its project from VertexTarget.
	Project string
	// Register publishes every token value to the run's redactor before it is
	// returned. nil is allowed (a caller that keeps no log).
	//
	// SECURITY: this is what keeps a token minted mid-run out of the session
	// log, the -v progress feed and the error lines. A redactor frozen at
	// startup could not know it.
	Register func(...string)
	// HTTPClient performs the token exchange; nil means a client bounded by
	// googleTokenTimeout. Injectable for tests, and for a deployment that must
	// route the token endpoint through a proxy of its own.
	HTTPClient *http.Client
	// Now is injectable for tests; nil means time.Now. It decides only whether
	// a cached token is still fresh.
	//
	// Library boundary: the EXPIRY it is compared against is stamped by
	// x/oauth2 from that library's own clock (jwt.jwtSource.Token computes
	// time.Now().Add(expires_in) and exposes no way to supply one), so an
	// injected Now that is not close to the real clock compares two different
	// timelines. It exists to let a test cross the refresh margin without
	// sleeping, not to relocate the run in time.
	Now func() time.Time

	// find resolves the ADC chain. It is unexported because it is a test seam,
	// not a dependency a caller may choose: the chain's last leg probes for a
	// metadata server, which is a property of the machine and would otherwise
	// make the "nothing resolved" test depend on where CI runs.
	find func(ctx context.Context, scopes ...string) (*google.Credentials, error)

	// mu guards everything below it and serializes the exchange.
	mu sync.Mutex
	// tok is the cached token, nil until one has been obtained.
	tok *oauth2.Token
	// quota is the answer QuotaProject returns, decided when the credential
	// was resolved. Empty until then, and empty for credentials that need no
	// header.
	quota string
	// registered is the token value already handed to Register. The secret set
	// has no de-duplication, so a token is registered exactly once.
	registered string
}

// Compile-time proof that this type satisfies both halves of the auth seam.
var (
	_ GeminiTokenSource  = (*GoogleTokenSource)(nil)
	_ GeminiQuotaProject = (*GoogleTokenSource)(nil)
)

// Token returns a valid access token, obtaining one if the cached token is
// missing or within googleTokenRefreshMargin of expiry.
//
// CONTRACT (GeminiTokenSource): the returned token is valid now, the context
// bounds the exchange where x/oauth2 propagates it, and concurrent calls are
// safe. Every failure is an ErrProvider naming what to fix - which is what
// makes an unusable credential exit 5 rather than a bare 401 from the API.
func (s *GoogleTokenSource) Token(ctx context.Context) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.fresh() {
		return s.tok.AccessToken, nil
	}
	creds, err := s.credentials(ctx)
	if err != nil {
		return "", err
	}
	tok, err := creds.TokenSource.Token()
	if err != nil {
		return "", s.exchangeError(err)
	}
	if tok == nil || tok.AccessToken == "" {
		return "", fmt.Errorf("%w: %s produced no access token", ErrProvider, s.credentialName())
	}
	// SECURITY: registered BEFORE it is returned. Registering afterwards would
	// leave a window in which the first message quoting the token - a provider
	// error, a progress line - is written verbatim.
	if s.Register != nil && tok.AccessToken != s.registered {
		s.registered = tok.AccessToken
		s.Register(tok.AccessToken)
	}
	s.tok = tok
	s.quota = s.quotaProjectFor(creds)
	return tok.AccessToken, nil
}

// QuotaProject implements GeminiQuotaProject: it names the project for the
// x-goog-user-project header, or "" when this credential needs none.
//
// It is empty until the first Token call, because the credential type is not
// known before the chain has been resolved. The client asks for the token
// first, which is what makes that ordering work (see GeminiClient.authorize).
func (s *GoogleTokenSource) QuotaProject() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.quota
}

// fresh reports whether the cached token can still authorize a request.
// Callers hold mu.
//
// The decision is made here rather than with oauth2.Token.Valid because Valid
// reads time.Now directly: keeping it local is what lets Now be injected and
// the cache be tested without sleeping. A token with no stated expiry is taken
// at its word - that is what the metadata server's contract allows.
func (s *GoogleTokenSource) fresh() bool {
	if s.tok == nil || s.tok.AccessToken == "" {
		return false
	}
	if s.tok.Expiry.IsZero() {
		return true
	}
	return s.now().Before(s.tok.Expiry.Add(-googleTokenRefreshMargin))
}

func (s *GoogleTokenSource) now() time.Time {
	if s.Now != nil {
		return s.Now()
	}
	return time.Now()
}

// credentials resolves the credential this source authenticates with. Callers
// hold mu.
//
// The resolution is repeated per REFRESH rather than cached, so that each
// exchange runs under the context of the request that needed it. The cost is a
// file read (or, on GCE, a cached metadata probe) roughly once an hour, and
// what it buys is a token source that cannot be poisoned by the context of
// whichever request happened to be first - the wart x/oauth2 has by capturing
// a context at construction time.
func (s *GoogleTokenSource) credentials(ctx context.Context) (*google.Credentials, error) {
	// The client travels in the context because that is the only way to hand
	// x/oauth2 one: the OAuth legs read it back out with
	// internal.ContextClient. The metadata-server leg is the exception - it
	// uses cloud.google.com/go/compute/metadata's own client, which is already
	// bounded and only ever talks to a link-local address.
	ctx = context.WithValue(ctx, oauth2.HTTPClient, s.httpClient())

	if s.CredentialsFile != "" {
		// The path is the operator's own config value; naming a file to read
		// is the whole point of the knob.
		data, err := os.ReadFile(s.CredentialsFile)
		if err != nil {
			return nil, fmt.Errorf("%w: reading provider.vertex.credentials %q: %v", ErrProvider, s.CredentialsFile, err)
		}
		// CONTRACT: the token endpoint comes from the FILE. Google's own key
		// documentation contradicts itself about it - one example shows
		// https://accounts.google.com/o/oauth2/token and the next
		// https://oauth2.googleapis.com/token (research §3.2) - so hardcoding
		// either would be a guess. CredentialsFromJSONWithType honors the
		// file's token_uri and falls back to the modern endpoint when the
		// field is absent, which is exactly the documented-but-contradicted
		// behavior every Google library implements.
		//
		// SECURITY: the expected type is asserted, not read from the file. See
		// the note on CredentialsFile.
		creds, err := google.CredentialsFromJSONWithType(ctx, data, google.ServiceAccount, vertexScope)
		if err != nil {
			// SECURITY: the path, never the contents - this file holds a
			// private key. The library's error describes the shape of the
			// problem (wrong type, unparseable key) without quoting material.
			return nil, fmt.Errorf("%w: provider.vertex.credentials %q is not a usable service-account key file: %v",
				ErrProvider, s.CredentialsFile, err)
		}
		return creds, nil
	}

	find := s.find
	if find == nil {
		find = google.FindDefaultCredentials
	}
	creds, err := find(ctx, vertexScope)
	if err != nil {
		// The library's message is one sentence and a doc link. The operator
		// needs the list of places that were searched and the config key that
		// skips the search, so both are spelled out here.
		return nil, fmt.Errorf("%w: no google credentials for vertex: %v; "+
			"set provider.vertex.credentials to a service-account key file, export GOOGLE_APPLICATION_CREDENTIALS, "+
			"or run `gcloud auth application-default login` "+
			"(on GCE, GKE and Cloud Run the attached service account is read from the metadata server automatically)",
			ErrProvider, err)
	}
	return creds, nil
}

// httpClient returns the client the token exchange uses.
func (s *GoogleTokenSource) httpClient() *http.Client {
	if s.HTTPClient != nil {
		return s.HTTPClient
	}
	return &http.Client{Timeout: googleTokenTimeout}
}

// quotaProjectFor decides whether this credential needs the x-goog-user-project
// header, and with which project.
//
// The header is sent for USER credentials only. Google documents it as required
// when calling some APIs with `authorized_user` credentials - the gcloud leg of
// the ADC chain - and sending it demands serviceusage.services.use on the named
// project (research §3.1, "Quota project"). A service account that legitimately
// holds only roles/aiplatform.user does not have that permission, so an
// unconditional header would turn working configurations into 403s.
//
// The value is provider.vertex.project rather than the credential file's
// quota_project_id, deliberately: the request is already addressed to that
// project and billed to it, so quota follows billing, and the operator's YAML
// stays the single place that answers "which project is this run charged to".
func (s *GoogleTokenSource) quotaProjectFor(creds *google.Credentials) string {
	// An explicit service-account key is never a user credential.
	if s.CredentialsFile != "" || creds == nil || len(creds.JSON) == 0 {
		return ""
	}
	// Only the type is read out of the credential JSON; the rest of that
	// document is the credential itself and stays untouched.
	var f struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(creds.JSON, &f); err != nil {
		return ""
	}
	switch google.CredentialsType(f.Type) {
	case google.AuthorizedUser, google.ExternalAccountAuthorizedUser:
		return s.Project
	default:
		return ""
	}
}

// exchangeError wraps a failed token acquisition, naming the credential source
// so the operator knows which one to fix.
//
// SECURITY: the wrapped error can carry the token endpoint's response body (an
// OAuth error document), which is why the credentials PATH is added here and
// the file's contents never are.
func (s *GoogleTokenSource) exchangeError(err error) error {
	return fmt.Errorf("%w: obtaining a google access token from %s: %v", ErrProvider, s.credentialName(), err)
}

// credentialName names the credential source in operator vocabulary.
func (s *GoogleTokenSource) credentialName() string {
	if s.CredentialsFile != "" {
		return fmt.Sprintf("provider.vertex.credentials %q", s.CredentialsFile)
	}
	return "application default credentials"
}
