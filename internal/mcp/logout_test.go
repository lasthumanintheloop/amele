package mcp

import (
	"context"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/lasthumanintheloop/amele/internal/oauthtoken"
)

// TestLoginStoresRevocationEndpoint pins the one field Task 7 adds to the
// record: without it a later `amele mcp logout` could only forget the
// credential locally, because discovery needs a 401 from a server that may by
// then be gone.
func TestLoginStoresRevocationEndpoint(t *testing.T) {
	f := newFakeAuthServer(t)
	lf := newLoginFixture(t, f)

	rec, err := lf.login(t)
	if err != nil {
		t.Fatalf("Login: %v", err)
	}
	want := f.issuer() + "/revoke"
	if rec.RevocationEndpoint != want {
		t.Errorf("revocation_endpoint = %q, want %q", rec.RevocationEndpoint, want)
	}
	key := oauthtoken.Key{Issuer: rec.Issuer, Resource: rec.Resource, ClientID: rec.ClientID}
	stored, err := lf.store.Load(key)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if stored.RevocationEndpoint != want {
		t.Errorf("stored revocation_endpoint = %q, want %q", stored.RevocationEndpoint, want)
	}
}

// TestRevokePostsRefreshToken is the RFC 7009 request itself: the refresh
// token is preferred (revoking it invalidates the whole grant) and the hint
// says which kind of token was sent.
func TestRevokePostsRefreshToken(t *testing.T) {
	f := newFakeAuthServer(t)
	rec := &oauthtoken.Record{
		Version:            oauthtoken.Version,
		Issuer:             f.issuer(),
		ClientID:           CIMDDocumentURL,
		RevocationEndpoint: f.issuer() + "/revoke",
		AccessToken:        "access-token-1",
		RefreshToken:       "refresh-token-1",
	}
	if err := Revoke(context.Background(), rec); err != nil {
		t.Fatalf("Revoke: %v", err)
	}
	form := f.lastRevokeForm(t)
	if form.Get("token") != "refresh-token-1" {
		t.Errorf("token = %q, want the refresh token", form.Get("token"))
	}
	if form.Get("token_type_hint") != "refresh_token" {
		t.Errorf("token_type_hint = %q", form.Get("token_type_hint"))
	}
	if form.Get("client_id") != CIMDDocumentURL {
		t.Errorf("client_id = %q", form.Get("client_id"))
	}
}

// TestRevokeFallsBackToAccessToken covers the credential an authorization
// server issued without a refresh token: there is still something to hand
// back, and the hint must not lie about what it is.
func TestRevokeFallsBackToAccessToken(t *testing.T) {
	f := newFakeAuthServer(t)
	rec := &oauthtoken.Record{
		RevocationEndpoint: f.issuer() + "/revoke",
		AccessToken:        "access-token-1",
	}
	if err := Revoke(context.Background(), rec); err != nil {
		t.Fatalf("Revoke: %v", err)
	}
	form := f.lastRevokeForm(t)
	if form.Get("token") != "access-token-1" || form.Get("token_type_hint") != "access_token" {
		t.Errorf("form = %v", form)
	}
}

// TestRevokeRefusesPlaintextEndpoint is the SECURITY rule: the request body
// carries the long-lived half of the credential, so a http:// endpoint is
// refused rather than handed the token in the clear. No loopback exception,
// the same stance checkTokenEndpoint takes.
func TestRevokeRefusesPlaintextEndpoint(t *testing.T) {
	rec := &oauthtoken.Record{RevocationEndpoint: "http://as.example/revoke", RefreshToken: "r"}
	err := Revoke(context.Background(), rec)
	if err == nil || !strings.Contains(err.Error(), "https") {
		t.Fatalf("err = %v, want an https refusal", err)
	}
}

// TestRevokeWithoutEndpointIsAnError keeps the caller honest: "the server
// advertised no revocation endpoint" is a decision the CLI reports as
// "local only", not something this function may silently succeed at.
func TestRevokeWithoutEndpointIsAnError(t *testing.T) {
	if err := Revoke(context.Background(), &oauthtoken.Record{RefreshToken: "r"}); err == nil {
		t.Fatal("Revoke succeeded without a revocation endpoint")
	}
	if err := Revoke(context.Background(), nil); err == nil {
		t.Fatal("Revoke succeeded on a nil record")
	}
}

// TestRevokeReportsRefusal pins that a non-2xx answer is an error the caller
// can print. It stays best-effort at the CLI layer: the local delete happens
// either way.
func TestRevokeReportsRefusal(t *testing.T) {
	f := newFakeAuthServer(t)
	f.revokeStatus = 400
	rec := &oauthtoken.Record{RevocationEndpoint: f.issuer() + "/revoke", RefreshToken: "r"}
	err := Revoke(context.Background(), rec)
	if err == nil || !strings.Contains(err.Error(), "400") {
		t.Fatalf("err = %v, want the status in the message", err)
	}
}

// TestRevokeAcceptsNoContent covers the servers that answer 204 instead of the
// 200 RFC 7009 names.
func TestRevokeAcceptsNoContent(t *testing.T) {
	f := newFakeAuthServer(t)
	f.revokeStatus = 204
	rec := &oauthtoken.Record{RevocationEndpoint: f.issuer() + "/revoke", RefreshToken: "r"}
	if err := Revoke(context.Background(), rec); err != nil {
		t.Fatalf("Revoke: %v", err)
	}
}

// TestMatchCredentials pins the lookup `status`, `logout` and the run path all
// share: canonical resource, store order, and every client id kept.
func TestMatchCredentials(t *testing.T) {
	store := oauthtoken.NewStore(t.TempDir(), func() time.Time { return time.Unix(0, 0) })
	const resource = "https://mcp.example/mcp"
	for _, id := range []string{"client-a", "client-b"} {
		rec := &oauthtoken.Record{
			Version: oauthtoken.Version, Issuer: "https://as.example",
			Resource: resource, ClientID: id, AccessToken: "t",
		}
		key := oauthtoken.Key{Issuer: rec.Issuer, Resource: rec.Resource, ClientID: id}
		if err := store.Save(key, rec); err != nil {
			t.Fatalf("Save: %v", err)
		}
	}
	other := &oauthtoken.Record{
		Version: oauthtoken.Version, Issuer: "https://as.example",
		Resource: "https://other.example/mcp", ClientID: "client-a", AccessToken: "t",
	}
	if err := store.Save(oauthtoken.Key{Issuer: other.Issuer, Resource: other.Resource, ClientID: other.ClientID}, other); err != nil {
		t.Fatalf("Save: %v", err)
	}

	matches, err := MatchCredentials(store, resource)
	if err != nil {
		t.Fatalf("MatchCredentials: %v", err)
	}
	if len(matches) != 2 {
		t.Fatalf("got %d matches, want 2: %+v", len(matches), matches)
	}
	for _, m := range matches {
		if m.Key.Resource != resource {
			t.Errorf("match for the wrong resource: %q", m.Key.Resource)
		}
	}
}

// lastRevokeForm returns the form of the most recent /revoke request.
func (f *fakeAuthServer) lastRevokeForm(t *testing.T) url.Values {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.revokeForms) == 0 {
		t.Fatal("no /revoke request was made")
	}
	return f.revokeForms[len(f.revokeForms)-1]
}
