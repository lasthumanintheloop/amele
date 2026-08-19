package mcp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/lasthumanintheloop/amele/internal/oauthtoken"
)

// revokeTimeout bounds ONE revocation request. Logging out must not hang on an
// authorization server that stopped answering: the local delete is what the
// operator actually asked for, and it happens whether this call lands or not.
const revokeTimeout = 10 * time.Second

// MatchCredentials returns every stored credential filed under resource, in
// store order, together with the problems Store.List reported.
//
// It is the ONE lookup `run`, `amele mcp status` and `amele mcp logout` share.
// Matching is by canonical resource (the caller passes
// oauthtoken.CanonicalResource of the server URL) because that is the only
// part of the key a config states; the issuer and the client id come from
// whatever the login stored. The list error is returned alongside the matches
// rather than instead of them: one unreadable file in the token directory must
// not hide the credential sitting next to it, and every caller decides for
// itself whether the problems are fatal.
func MatchCredentials(store *oauthtoken.Store, resource string) ([]oauthtoken.Entry, error) {
	entries, listErr := store.List()
	var matches []oauthtoken.Entry
	for _, e := range entries {
		if e.Key.Resource == resource {
			matches = append(matches, e)
		}
	}
	return matches, listErr
}

// Revoke hands a stored credential back to the authorization server (RFC 7009).
//
// CONTRACT: it is BEST EFFORT and says so by returning an error the caller is
// expected to print rather than to act on - `amele mcp logout` deletes the
// local record whether this succeeds or fails, because a credential the
// operator asked to be rid of must not survive an authorization server that is
// down, moved or refusing.
//
// The refresh token is preferred over the access token: RFC 7009 §2.1 lets a
// server invalidate the whole grant when it is given the refresh token, while
// revoking an access token need not touch anything else. A record with neither
// is an error - there would be nothing to send.
//
// SECURITY: https only, with no loopback exception (the same rule
// checkTokenEndpoint applies to refreshes), the hardened OAuth client (no
// redirects, capped body), and a timeout of its own.
func Revoke(ctx context.Context, rec *oauthtoken.Record) error {
	if rec == nil {
		return errors.New("no credential to revoke")
	}
	if err := checkHTTPSEndpoint(rec.RevocationEndpoint, "revocation endpoint"); err != nil {
		return err
	}
	token, hint := rec.RefreshToken, "refresh_token"
	if token == "" {
		token, hint = rec.AccessToken, "access_token"
	}
	if token == "" {
		return errors.New("the stored credential carries no token to revoke")
	}
	form := url.Values{"token": {token}, "token_type_hint": {hint}}
	if rec.ClientID != "" {
		form.Set("client_id", rec.ClientID)
	}
	rctx, cancel := context.WithTimeout(ctx, revokeTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(rctx, http.MethodPost, rec.RevocationEndpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return fmt.Errorf("building the revocation request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := newOAuthClient().Do(req)
	if err != nil {
		return fmt.Errorf("revocation endpoint: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	// The body is of no interest (RFC 7009 defines none for success), but it
	// is drained within the cap so the connection can be reused and a chatty
	// error page cannot be read without bound.
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxTokenResponseBytes))
	if resp.StatusCode/100 != 2 {
		// 2xx rather than exactly 200: RFC 7009 names 200, and servers in the
		// wild answer 204. Both mean the same thing to a caller that reads no
		// body.
		return fmt.Errorf("revocation endpoint returned %s", resp.Status)
	}
	return nil
}
