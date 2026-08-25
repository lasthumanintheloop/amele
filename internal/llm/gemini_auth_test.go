package llm

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/oauth2/google"
)

// The access token every fake credential server in this file mints. It is a
// literal, not a credential: nothing outside this process has ever seen it.
const fakeAccessToken = "ya29.fake-access-token" //nolint:gosec // G101: a fixture value, not a credential.

// tokenServer is a stand-in for oauth2.googleapis.com. Unit tests must never
// reach the real token endpoint, so every credential fixture below points its
// token_uri at one of these.
type tokenServer struct {
	*httptest.Server
	// hits counts exchanges. Atomic because the handler runs on the server's
	// own goroutine while the test reads the counter on its own - the server
	// is still open at that point (t.Cleanup closes it), so the read is not
	// ordered after the handler by anything but the request itself.
	hits atomic.Int64
	// form is the last exchange's POST body, which is what pins the grant type
	// and the signed assertion.
	form url.Values
	// expiresIn is the lifetime handed out; zero means one hour.
	expiresIn int
}

func newTokenServer(t *testing.T) *tokenServer {
	t.Helper()
	ts := &tokenServer{}
	ts.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		ts.hits.Add(1)
		ts.form = r.PostForm
		life := ts.expiresIn
		if life == 0 {
			life = 3600
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token": fakeAccessToken,
			"token_type":   "Bearer",
			"expires_in":   life,
		})
	}))
	t.Cleanup(ts.Close)
	return ts
}

// writeServiceAccountKey writes a service-account key file whose private key is
// generated here and thrown away with the test's temp directory. No real
// credential material is ever committed to the repository.
func writeServiceAccountKey(t *testing.T, tokenURI string) string {
	t.Helper()
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatalf("generating a throwaway key: %v", err)
	}
	der, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatalf("marshalling the throwaway key: %v", err)
	}
	doc := map[string]string{
		"type":           "service_account",
		"project_id":     "test-project",
		"private_key_id": "fake-key-id",
		"private_key":    string(pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})),
		"client_email":   "amele-test@test-project.iam.gserviceaccount.com",
		"token_uri":      tokenURI,
	}
	return writeCredentialFile(t, "sa.json", doc)
}

// writeAuthorizedUserFile writes the gcloud `application_default_credentials`
// shape: the ADC leg that needs the quota-project header.
func writeAuthorizedUserFile(t *testing.T, tokenURI string) string {
	t.Helper()
	return writeCredentialFile(t, "adc.json", map[string]string{ //nolint:gosec // G101: throwaway fixture values, not credentials.
		"type":          "authorized_user",
		"client_id":     "fake-client-id.apps.googleusercontent.com",
		"client_secret": "fake-client-secret",
		"refresh_token": "fake-refresh-token",
		"token_uri":     tokenURI,
	})
}

func writeCredentialFile(t *testing.T, name string, doc map[string]string) string {
	t.Helper()
	b, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("encoding the credential fixture: %v", err)
	}
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatalf("writing the credential fixture: %v", err)
	}
	return path
}

// jwtClaims decodes the claim set of a signed assertion. The signature is not
// verified - the test server is not Google - but the claims are what pin the
// scope, the issuer and the audience amele asks for.
func jwtClaims(t *testing.T, assertion string) map[string]any {
	t.Helper()
	parts := strings.Split(assertion, ".")
	if len(parts) != 3 {
		t.Fatalf("assertion is not a JWS: %d segments", len(parts))
	}
	raw, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		t.Fatalf("decoding the claim set: %v", err)
	}
	var claims map[string]any
	if err := json.Unmarshal(raw, &claims); err != nil {
		t.Fatalf("parsing the claim set: %v", err)
	}
	return claims
}

// TestServiceAccountKeyIsExchangedAtTheFilesTokenURI is the whole SA-key
// contract in one test: the key file is parsed, a JWT is signed with it, and
// the exchange goes to the token_uri THE FILE names.
//
// The last part is the one worth a test of its own: the IAM documentation
// contradicts itself about that URL (research §3.2 - its two examples disagree),
// so a hardcoded endpoint would be a coin flip. Pointing the fixture's token_uri
// at a local server proves amele reads it, and doubles as the guarantee that
// this unit test never reaches oauth2.googleapis.com.
func TestServiceAccountKeyIsExchangedAtTheFilesTokenURI(t *testing.T) {
	srv := newTokenServer(t)
	source := &GoogleTokenSource{CredentialsFile: writeServiceAccountKey(t, srv.URL+"/token")}

	got, err := source.Token(context.Background())
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if got != fakeAccessToken {
		t.Errorf("token = %q, want %q", got, fakeAccessToken)
	}
	if srv.hits.Load() != 1 {
		t.Fatalf("token endpoint hit %d times, want 1", srv.hits.Load())
	}
	if grant := srv.form.Get("grant_type"); grant != "urn:ietf:params:oauth:grant-type:jwt-bearer" {
		t.Errorf("grant_type = %q", grant)
	}
	claims := jwtClaims(t, srv.form.Get("assertion"))
	if claims["scope"] != vertexScope {
		t.Errorf("scope = %v, want %q", claims["scope"], vertexScope)
	}
	if claims["iss"] != "amele-test@test-project.iam.gserviceaccount.com" {
		t.Errorf("iss = %v", claims["iss"])
	}
	// The audience follows the token_uri, which is the second half of the
	// "read it from the file" contract.
	if want := srv.URL + "/token"; claims["aud"] != want {
		t.Errorf("aud = %v, want %q", claims["aud"], want)
	}
}

// TestTokenIsCachedUntilItNearsExpiry pins the pull-based contract: the client
// asks for a token on every attempt, and all but the first are answered from
// the cache. A source that exchanged per request would spend a network
// round-trip on every turn of the loop.
func TestTokenIsCachedUntilItNearsExpiry(t *testing.T) {
	srv := newTokenServer(t)
	// The injected clock starts at the real one on purpose: the EXPIRY it is
	// compared against is stamped by x/oauth2 from the library's own clock (see
	// the note on GoogleTokenSource.Now), so a fixed date unrelated to now
	// would be comparing two different timelines. What the injection buys is
	// crossing the refresh margin without sleeping through an hour.
	now := time.Now()
	source := &GoogleTokenSource{
		CredentialsFile: writeServiceAccountKey(t, srv.URL+"/token"),
		Now:             func() time.Time { return now },
	}

	for i := range 3 {
		if _, err := source.Token(context.Background()); err != nil {
			t.Fatalf("Token #%d: %v", i, err)
		}
	}
	if srv.hits.Load() != 1 {
		t.Errorf("token endpoint hit %d times, want 1", srv.hits.Load())
	}

	// Inside the refresh margin the cached token is no longer good enough: the
	// request it would authenticate could outlive it.
	now = now.Add(time.Hour - googleTokenRefreshMargin/2)
	if _, err := source.Token(context.Background()); err != nil {
		t.Fatalf("Token after expiry: %v", err)
	}
	if srv.hits.Load() != 2 {
		t.Errorf("token endpoint hit %d times after expiry, want 2", srv.hits.Load())
	}
}

// TestTokenIsRegisteredAsASecretBeforeItIsReturned is the SECURITY contract:
// the run's redactor learns the token BEFORE the caller can put it anywhere.
// Registering afterwards would leave a window in which the very first error
// mentioning it is written verbatim.
func TestTokenIsRegisteredAsASecretBeforeItIsReturned(t *testing.T) {
	srv := newTokenServer(t)
	var registered []string
	source := &GoogleTokenSource{
		CredentialsFile: writeServiceAccountKey(t, srv.URL+"/token"),
		Register: func(values ...string) {
			registered = append(registered, values...)
		},
	}

	for range 3 {
		if _, err := source.Token(context.Background()); err != nil {
			t.Fatalf("Token: %v", err)
		}
	}
	// Registered once, not once per call: the secret set has no
	// de-duplication, so a re-registration would grow it without bound over a
	// long run.
	if len(registered) != 1 || registered[0] != fakeAccessToken {
		t.Errorf("registered = %q, want exactly [%q]", registered, fakeAccessToken)
	}
}

// TestServiceAccountFileErrorsNameThePathOnly: a credential file that cannot be
// read or parsed produces a provider error naming the PATH. SECURITY: the file
// holds a private key, so its contents must never reach the message.
func TestServiceAccountFileErrorsNameThePathOnly(t *testing.T) {
	const secretMaterial = "-----BEGIN PRIVATE KEY-----super-secret-----END PRIVATE KEY-----"
	garbage := filepath.Join(t.TempDir(), "broken.json")
	if err := os.WriteFile(garbage, []byte("not json "+secretMaterial), 0o600); err != nil {
		t.Fatalf("writing the fixture: %v", err)
	}

	tests := []struct {
		name string
		path string
	}{
		{name: "missing file", path: filepath.Join(t.TempDir(), "absent.json")},
		{name: "unparseable file", path: garbage},
		{name: "wrong credential type", path: writeAuthorizedUserFile(t, "https://example.invalid/token")},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			source := &GoogleTokenSource{CredentialsFile: tc.path}
			_, err := source.Token(context.Background())
			if err == nil {
				t.Fatal("expected an error")
			}
			if !errors.Is(err, ErrProvider) {
				t.Errorf("error %v is not an ErrProvider", err)
			}
			if !strings.Contains(err.Error(), tc.path) {
				t.Errorf("error %q does not name the credentials path", err)
			}
			if !strings.Contains(err.Error(), "provider.vertex.credentials") {
				t.Errorf("error %q does not name the config key to fix", err)
			}
			if strings.Contains(err.Error(), secretMaterial) {
				t.Errorf("error %q leaked the file's contents", err)
			}
		})
	}
}

// TestADCUsesTheEnvironmentCredentialFile exercises the real ADC chain at its
// first step: with provider.vertex.credentials unset, GOOGLE_APPLICATION_
// CREDENTIALS decides. This is the one leg that can be driven hermetically -
// the gcloud well-known file and the metadata server are properties of the
// machine - and it is what proves amele delegates the whole chain to the
// library rather than reimplementing step 1.
func TestADCUsesTheEnvironmentCredentialFile(t *testing.T) {
	srv := newTokenServer(t)
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", writeServiceAccountKey(t, srv.URL+"/token"))

	source := &GoogleTokenSource{Project: "my-project"}
	got, err := source.Token(context.Background())
	if err != nil {
		t.Fatalf("Token: %v", err)
	}
	if got != fakeAccessToken {
		t.Errorf("token = %q", got)
	}
	// A service account carries its own quota project, so no header is asked
	// for. Sending one would demand serviceusage.services.use from a principal
	// that has no reason to hold it.
	if qp := source.QuotaProject(); qp != "" {
		t.Errorf("QuotaProject = %q, want empty for a service account", qp)
	}
}

// TestADCUserCredentialsAskForTheQuotaProjectHeader: gcloud user credentials
// are the ADC leg Google documents as needing x-goog-user-project. amele sends
// it for exactly that credential type, with the project from the vertex block -
// the same project the request URL is addressed to.
func TestADCUserCredentialsAskForTheQuotaProjectHeader(t *testing.T) {
	srv := newTokenServer(t)
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", writeAuthorizedUserFile(t, srv.URL+"/token"))

	source := &GoogleTokenSource{Project: "my-project"}
	if _, err := source.Token(context.Background()); err != nil {
		t.Fatalf("Token: %v", err)
	}
	if got := source.QuotaProject(); got != "my-project" {
		t.Errorf("QuotaProject = %q, want %q", got, "my-project")
	}
	// The refresh flow, not the JWT one: proof the type dispatch is the
	// library's and not a hardcoded assumption about the file.
	if grant := srv.form.Get("grant_type"); grant != "refresh_token" {
		t.Errorf("grant_type = %q, want refresh_token", grant)
	}
}

// TestQuotaProjectIsUnknownBeforeTheFirstToken pins the ordering the client
// depends on: a lazily resolved source cannot answer until it has produced a
// token, so authorize asks for the token first.
func TestQuotaProjectIsUnknownBeforeTheFirstToken(t *testing.T) {
	source := &GoogleTokenSource{Project: "my-project"}
	if got := source.QuotaProject(); got != "" {
		t.Errorf("QuotaProject before the first token = %q, want empty", got)
	}
}

// TestADCNotFoundNamesTheWholeChain: the message an operator sees when nothing
// in the ADC chain resolved must tell them every place amele looked and the
// config key that bypasses the search. A bare "could not find default
// credentials" sends them to a search engine.
func TestADCNotFoundNamesTheWholeChain(t *testing.T) {
	source := &GoogleTokenSource{
		Project: "my-project",
		// The chain is stubbed rather than driven: its last leg probes for a
		// metadata server, which is a property of the MACHINE running the test
		// and would make the outcome depend on where CI happens to run.
		find: func(context.Context, ...string) (*google.Credentials, error) {
			return nil, errors.New("google: could not find default credentials")
		},
	}
	_, err := source.Token(context.Background())
	if err == nil {
		t.Fatal("expected an error")
	}
	if !errors.Is(err, ErrProvider) {
		t.Errorf("error %v is not an ErrProvider", err)
	}
	for _, want := range []string{
		"provider.vertex.credentials",
		"GOOGLE_APPLICATION_CREDENTIALS",
		"gcloud auth application-default login",
		"metadata server",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

// TestTokenExchangeFailureIsAProviderError: the token endpoint refusing the
// assertion is a provider failure (exit 5), named as the exchange it was.
func TestTokenExchangeFailureIsAProviderError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error":"invalid_grant"}`, http.StatusBadRequest)
	}))
	defer srv.Close()

	path := writeServiceAccountKey(t, srv.URL+"/token")
	source := &GoogleTokenSource{CredentialsFile: path}
	_, err := source.Token(context.Background())
	if err == nil {
		t.Fatal("expected an error")
	}
	if !errors.Is(err, ErrProvider) {
		t.Errorf("error %v is not an ErrProvider", err)
	}
	if !strings.Contains(err.Error(), path) {
		t.Errorf("error %q does not name the credentials path", err)
	}
}

// TestTokenFetchHonoursContextCancellation: a cancelled run must not sit on a
// token exchange. The refresh-token leg is used because that is the one where
// x/oauth2 puts the context on the outgoing request; the JWT leg posts without
// one and is bounded by the client timeout instead (see GoogleTokenSource).
func TestTokenFetchHonoursContextCancellation(t *testing.T) {
	srv := newTokenServer(t)
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", writeAuthorizedUserFile(t, srv.URL+"/token"))
	source := &GoogleTokenSource{Project: "my-project"}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := source.Token(ctx)
	if err == nil {
		t.Fatal("expected an error from a cancelled context")
	}
	if !errors.Is(err, ErrProvider) {
		t.Errorf("error %v is not an ErrProvider", err)
	}
	if srv.hits.Load() != 0 {
		t.Errorf("the token endpoint was reached %d times despite cancellation", srv.hits.Load())
	}
}

// TestEmptyAccessTokenIsRefused: a token endpoint that answers 200 with no
// access_token would otherwise be handed on as a bare "Bearer ", and the 401 it
// earns describes the ENDPOINT rather than the credential. The JWT leg is used
// because it is the one x/oauth2 does not check for itself.
func TestEmptyAccessTokenIsRefused(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"access_token":"","token_type":"Bearer","expires_in":3600}`))
	}))
	defer srv.Close()

	path := writeServiceAccountKey(t, srv.URL+"/token")
	_, err := (&GoogleTokenSource{CredentialsFile: path}).Token(context.Background())
	if err == nil {
		t.Fatal("expected an error")
	}
	if !errors.Is(err, ErrProvider) {
		t.Errorf("error %v is not an ErrProvider", err)
	}
	if !strings.Contains(err.Error(), path) {
		t.Errorf("error %q does not name the credentials path", err)
	}
}

// TestInjectedHTTPClientPerformsTheExchange pins the seam a VPC-SC deployment
// needs: the token endpoint is reached through the client this source was
// given, not through a private one it built.
func TestInjectedHTTPClientPerformsTheExchange(t *testing.T) {
	srv := newTokenServer(t)
	var used bool
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		used = true
		return http.DefaultTransport.RoundTrip(r)
	})}

	source := &GoogleTokenSource{
		CredentialsFile: writeServiceAccountKey(t, srv.URL+"/token"),
		HTTPClient:      client,
	}
	if _, err := source.Token(context.Background()); err != nil {
		t.Fatalf("Token: %v", err)
	}
	if !used {
		t.Error("the injected client did not perform the exchange")
	}
}

// roundTripFunc adapts a function to http.RoundTripper.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
