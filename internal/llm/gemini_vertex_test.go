package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"
)

// TestVertexEndpoint pins every shape of the Vertex generateContent URL.
//
// The rows are the contract, not examples: the regional host, the two hosts
// that are NOT "{location}-aiplatform" (global and the jurisdictional
// multi-regions), the base_url host override, and the invariant underneath all
// of them - the configured location always reaches the path, whatever the host
// ends up being.
func TestVertexEndpoint(t *testing.T) {
	tests := []struct {
		name   string
		base   string
		target VertexTarget
		model  string
		want   string
	}{
		{
			name:   "regional host carries the location prefix",
			target: VertexTarget{Project: "my-project", Location: "us-central1"},
			model:  "gemini-3-pro",
			want:   "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-3-pro:generateContent",
		},
		{
			// The host loses the region prefix AND the path keeps
			// locations/global: the two change together, they are not
			// alternatives.
			name:   "global drops the host prefix but keeps the path segment",
			target: VertexTarget{Project: "my-project", Location: "global"},
			model:  "gemini-3-pro",
			want:   "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/publishers/google/models/gemini-3-pro:generateContent",
		},
		{
			// The jurisdictional multi-regions are a third host shape. Building
			// "eu-aiplatform.googleapis.com" for them would be a DNS failure on
			// the one location an EU-residency deployment must be able to name.
			name:   "the eu multi-region has its own host",
			target: VertexTarget{Project: "my-project", Location: "eu"},
			model:  "gemini-3.5-flash",
			want:   "https://aiplatform.eu.rep.googleapis.com/v1/projects/my-project/locations/eu/publishers/google/models/gemini-3.5-flash:generateContent",
		},
		{
			name:   "the us multi-region has its own host",
			target: VertexTarget{Project: "my-project", Location: "us"},
			model:  "gemini-3.5-flash",
			want:   "https://aiplatform.us.rep.googleapis.com/v1/projects/my-project/locations/us/publishers/google/models/gemini-3.5-flash:generateContent",
		},
		{
			// SECURITY: base_url moves the HOST (a VPC-SC restricted VIP or a
			// Private Service Connect name) and nothing else. The location is
			// still us-central1 in the path - amele does not reroute processing
			// because the host moved.
			name:   "base_url overrides the host, not the location",
			base:   "https://vertex.restricted.example.com",
			target: VertexTarget{Project: "my-project", Location: "us-central1"},
			model:  "gemini-3-pro",
			want:   "https://vertex.restricted.example.com/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-3-pro:generateContent",
		},
		{
			name:   "a host override keeps its port",
			base:   "https://127.0.0.1:8443",
			target: VertexTarget{Project: "p", Location: "europe-west4"},
			model:  "gemini-3-pro",
			want:   "https://127.0.0.1:8443/v1/projects/p/locations/europe-west4/publishers/google/models/gemini-3-pro:generateContent",
		},
		{
			name:   "an http override stays http",
			base:   "http://localhost:9000",
			target: VertexTarget{Project: "p", Location: "europe-west4"},
			model:  "gemini-3-pro",
			want:   "http://localhost:9000/v1/projects/p/locations/europe-west4/publishers/google/models/gemini-3-pro:generateContent",
		},
		{
			name:   "a trailing slash does not double",
			base:   "https://vertex.restricted.example.com/",
			target: VertexTarget{Project: "p", Location: "us-central1"},
			model:  "gemini-3-pro",
			want:   "https://vertex.restricted.example.com/v1/projects/p/locations/us-central1/publishers/google/models/gemini-3-pro:generateContent",
		},
		{
			// SECURITY: the override is scheme+host, so a path written next to
			// it cannot reach the request. Validate refuses this config (a
			// dropped prefix is a request sent somewhere other than the file
			// says); this row pins that even so, nothing in base_url can move
			// the resource path.
			name:   "a path in base_url cannot reach the request",
			base:   "https://proxy.example.com/ignored/prefix",
			target: VertexTarget{Project: "p", Location: "us-central1"},
			model:  "gemini-3-pro",
			want:   "https://proxy.example.com/v1/projects/p/locations/us-central1/publishers/google/models/gemini-3-pro:generateContent",
		},
		{
			// Same single-segment escaping as the AI Studio path: the model is
			// a model, not a way to address another resource.
			name:   "the model is escaped as one segment",
			target: VertexTarget{Project: "p", Location: "us-central1"},
			model:  "models/../../foo",
			want:   "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/publishers/google/models/..%2F..%2Ffoo:generateContent",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			target := tt.target
			client := &GeminiClient{BaseURL: tt.base, Vertex: &target}
			got, err := client.endpoint(tt.model)
			if err != nil {
				t.Fatalf("endpoint: %v", err)
			}
			if got != tt.want {
				t.Errorf("endpoint:\n got %q\nwant %q", got, tt.want)
			}
		})
	}
}

// TestVertexEndpointRefusesUnaddressableTargets is the second half of the
// endpoint contract: what it will NOT build.
//
// Config validation already refuses every input here, so this is defense in
// depth on a security boundary - GeminiClient is an exported type any caller
// can construct - and the failure mode it prevents is the worst one available:
// a request addressed to a host nobody chose.
func TestVertexEndpointRefusesUnaddressableTargets(t *testing.T) {
	tests := []struct {
		name   string
		base   string
		target VertexTarget
		want   string
	}{
		{
			name:   "no project",
			target: VertexTarget{Location: "us-central1"},
			want:   "vertex project",
		},
		{
			name:   "no location",
			target: VertexTarget{Project: "p"},
			want:   "vertex location",
		},
		{
			// The location becomes part of the HOST, where percent-escaping
			// means nothing: it can only be refused.
			name:   "a location that would rewrite the host",
			target: VertexTarget{Project: "p", Location: "evil.example.com/v1"},
			want:   "vertex location",
		},
		{
			name:   "a project that would climb the path",
			target: VertexTarget{Project: "../other", Location: "us-central1"},
			want:   "vertex project",
		},
		{
			name:   "a base_url with no scheme",
			base:   "vertex.example.com",
			target: VertexTarget{Project: "p", Location: "us-central1"},
			want:   "base_url",
		},
		{
			name:   "a base_url with a scheme this client does not speak",
			base:   "file:///etc/passwd",
			target: VertexTarget{Project: "p", Location: "us-central1"},
			want:   "base_url",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			target := tt.target
			client := &GeminiClient{BaseURL: tt.base, Vertex: &target}
			got, err := client.endpoint("gemini-3-pro")
			if err == nil {
				t.Fatalf("endpoint must refuse this target, built %q", got)
			}
			if !errors.Is(err, ErrProvider) {
				t.Errorf("error %v is not an ErrProvider", err)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error %q does not name %q", err, tt.want)
			}
		})
	}
}

// TestGeminiAIStudioEndpointUnaffected: the AI Studio path keeps its own host,
// version and shape. Vertex is a mode, not a rewrite of the client.
func TestGeminiAIStudioEndpointUnaffected(t *testing.T) {
	client := &GeminiClient{}
	got, err := client.endpoint("gemini-3-pro")
	if err != nil {
		t.Fatalf("endpoint: %v", err)
	}
	const want = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro:generateContent"
	if got != want {
		t.Errorf("endpoint: got %q, want %q", got, want)
	}
}

// fakeTokenSource is the Task 8 seam under test: the client asks for a bearer
// token and never learns where it came from.
type fakeTokenSource struct {
	token string
	err   error
	calls int
	ctxOK bool
}

func (f *fakeTokenSource) Token(ctx context.Context) (string, error) {
	f.calls++
	f.ctxOK = ctx != nil && ctx.Err() == nil
	return f.token, f.err
}

// TestVertexRequestUsesBearerToken pins the auth swap: in vertex mode the
// credential is an Authorization bearer token from the injected source, and the
// AI Studio key header is absent from the request entirely.
func TestVertexRequestUsesBearerToken(t *testing.T) {
	var gotAuth, gotKey, gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotKey = r.Header.Get("x-goog-api-key")
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	}))
	defer srv.Close()

	source := &fakeTokenSource{token: "ya29.test-token"} //nolint:gosec // G101: a fake token that exists to be seen on the wire.
	client := &GeminiClient{
		BaseURL:     srv.URL,
		Vertex:      &VertexTarget{Project: "p", Location: "us-central1"},
		TokenSource: source,
	}
	if _, err := client.Chat(context.Background(), gemUserRequest()); err != nil {
		t.Fatalf("Chat: %v", err)
	}

	if gotAuth != "Bearer ya29.test-token" { //nolint:gosec // G101: the same fake token, asserted.
		t.Errorf("Authorization header = %q", gotAuth)
	}
	if gotKey != "" {
		t.Errorf("the AI Studio key header travelled to vertex: %q", gotKey)
	}
	const wantPath = "/v1/projects/p/locations/us-central1/publishers/google/models/gemini-3-pro:generateContent"
	if gotPath != wantPath {
		t.Errorf("path = %q, want %q", gotPath, wantPath)
	}
	if source.calls != 1 {
		t.Errorf("token fetched %d times, want 1", source.calls)
	}
	if !source.ctxOK {
		t.Error("the token source was called without the request context")
	}
}

// TestVertexAPIKeyNeverTravels: a client that somehow carries both credentials
// still sends only the bearer token. Config refuses the combination (exit 2);
// this pins that the wire could not leak the AI Studio key even if it did not.
func TestVertexAPIKeyNeverTravels(t *testing.T) {
	var gotKey string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get("x-goog-api-key")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
	}))
	defer srv.Close()

	client := &GeminiClient{
		BaseURL:     srv.URL,
		APIKey:      "ai-studio-key",
		Vertex:      &VertexTarget{Project: "p", Location: "global"},
		TokenSource: &fakeTokenSource{token: "t"},
	}
	if _, err := client.Chat(context.Background(), gemUserRequest()); err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if gotKey != "" {
		t.Errorf("the AI Studio key reached a vertex request: %q", gotKey)
	}
}

// TestVertexWithoutTokenSourceIsRefusedLocally: vertex mode with no credential
// source sends nothing at all. The prompt is not worth spending on a request
// the endpoint will refuse, and a local error can name the missing wiring while
// a 401 can only describe its symptom.
func TestVertexWithoutTokenSourceIsRefusedLocally(t *testing.T) {
	var called bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		called = true
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	client := &GeminiClient{BaseURL: srv.URL, Vertex: &VertexTarget{Project: "p", Location: "us-central1"}}
	_, err := client.Chat(context.Background(), gemUserRequest())
	if err == nil {
		t.Fatal("a vertex request without credentials must fail")
	}
	if !errors.Is(err, ErrProvider) {
		t.Errorf("error %v is not an ErrProvider", err)
	}
	if called {
		t.Error("the request was sent anyway")
	}
}

// TestVertexEmptyTokenIsRefused: a source that returns "" without an error is
// broken in a way only this client can name. Sent as-is it becomes a bare
// "Bearer " and comes back a 401 describing the endpoint, which points the
// operator at their IAM roles instead of at their credential wiring.
func TestVertexEmptyTokenIsRefused(t *testing.T) {
	var called bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		called = true
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	client := &GeminiClient{
		BaseURL:     srv.URL,
		Vertex:      &VertexTarget{Project: "p", Location: "us-central1"},
		TokenSource: &fakeTokenSource{},
	}
	_, err := client.Chat(context.Background(), gemUserRequest())
	if err == nil {
		t.Fatal("an empty access token must fail the call")
	}
	if !strings.Contains(err.Error(), "empty access token") {
		t.Errorf("error %q does not name the empty token", err)
	}
	if called {
		t.Error("the request was sent with an empty bearer token")
	}
}

// TestVertexTokenFailureIsNotRetried: a credential that cannot be obtained is
// not a transient provider failure - retrying it would burn the attempt budget
// and delay a message the operator has to act on.
func TestVertexTokenFailureIsNotRetried(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	source := &fakeTokenSource{err: errors.New("no credentials found")}
	client := &GeminiClient{
		BaseURL:     srv.URL,
		Vertex:      &VertexTarget{Project: "p", Location: "us-central1"},
		TokenSource: source,
		MaxAttempts: 3,
		Sleep:       func(context.Context, time.Duration) error { return nil },
	}
	_, err := client.Chat(context.Background(), gemUserRequest())
	if err == nil {
		t.Fatal("a token acquisition failure must fail the call")
	}
	if !strings.Contains(err.Error(), "no credentials found") {
		t.Errorf("error %q does not carry the source's own message", err)
	}
	if calls != 0 {
		t.Errorf("the request was sent %d times without a token", calls)
	}
	if source.calls != 1 {
		t.Errorf("the failing token source was called %d times, want 1", source.calls)
	}
}

// TestVertexBodyIsIdenticalToAIStudio is the vertex-mode body golden, and its
// result is the point: not one byte differs.
//
// The research names three body-level incompatibilities between the two
// backends (topK's type, responseFormat's shape, safetySettings.method) and two
// billing fields (labels, serviceTier). Every one of them is INERT for amele
// because amele does not send the field: the request this client owns is
// contents + systemInstruction + tools + generationConfig, and every field in
// it is shared byte-for-byte by both backends. The diff table therefore lives
// as a comment on gemRequest rather than as code, and this test is what keeps
// the claim honest.
func TestVertexBodyIsIdenticalToAIStudio(t *testing.T) {
	capture := func(client *GeminiClient) []byte {
		t.Helper()
		var body []byte
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			body, _ = io.ReadAll(r.Body)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
		}))
		defer srv.Close()
		client.BaseURL = srv.URL
		if _, err := client.Chat(context.Background(), gemTunedRequest()); err != nil {
			t.Fatalf("Chat: %v", err)
		}
		return body
	}

	studio := capture(&GeminiClient{APIKey: "k"})
	vertex := capture(&GeminiClient{
		Vertex:      &VertexTarget{Project: "p", Location: "us-central1"},
		TokenSource: &fakeTokenSource{token: "t"},
	})

	if string(studio) != string(vertex) {
		t.Errorf("vertex mode changed the request body:\n ai studio: %s\n    vertex: %s", studio, vertex)
	}
	// A guard against an empty comparison: the bodies must actually carry the
	// knobs whose spellings the diff table is about.
	for _, want := range []string{"generationConfig", "thinkingConfig", "topP"} {
		if !strings.Contains(string(studio), want) {
			t.Fatalf("the compared body does not exercise %q: %s", want, studio)
		}
	}
}

// TestGeminiWireFieldsPinTheVertexDiffTable is the tripwire under the claim
// above. The vertex body diffs are inert only for as long as this client sends
// none of the fields they concern, and that is a property of the wire structs -
// so the field set is pinned here.
//
// A new field appearing in this list fails the test on purpose: whoever adds it
// must check it against the diff table in
// docs/superpowers/specs/2026-08-25-vertex-adc-research.md §2 and decide
// whether vertex mode needs to spell it differently, rather than discovering
// the difference as a 400 from a live run.
func TestGeminiWireFieldsPinTheVertexDiffTable(t *testing.T) {
	tags := func(v any) []string {
		var out []string
		typ := reflect.TypeOf(v)
		for i := range typ.NumField() {
			tag, _, _ := strings.Cut(typ.Field(i).Tag.Get("json"), ",")
			if tag == "" || tag == "-" {
				continue
			}
			out = append(out, tag)
		}
		slices.Sort(out)
		return out
	}

	wantRequest := []string{"contents", "generationConfig", "systemInstruction", "tools"}
	if got := tags(gemRequest{}); !slices.Equal(got, wantRequest) {
		t.Errorf("the request body grew a field: got %v, want %v - check it against the vertex diff table", got, wantRequest)
	}

	// topK is deliberately absent: it is the field whose TYPE differs between
	// the backends (float on vertex, integer on AI Studio), and amele exposes
	// no knob for it. responseFormat is absent for the same reason - its SHAPE
	// differs (an array on vertex, an object on AI Studio) - and structured
	// output travels as responseJsonSchema, which is shared.
	wantConfig := []string{
		"maxOutputTokens", "responseJsonSchema", "responseMimeType",
		"stopSequences", "temperature", "thinkingConfig", "topP",
	}
	if got := tags(gemGenerationConfig{}); !slices.Equal(got, wantConfig) {
		t.Errorf("generationConfig grew a field: got %v, want %v - check it against the vertex diff table", got, wantConfig)
	}

	// safetySettings is owned (params cannot supply it) but never written, so
	// the vertex-only safetySettings.method has nothing to apply to.
	if slices.Contains(tags(gemRequest{}), "safetySettings") {
		t.Error("safetySettings is now written; vertex accepts an extra method field on it")
	}
}

// gemTunedRequest exercises every field this client writes: a system prompt, a
// tool, structured output, the cap, both sampling knobs and a reasoning level.
// It is the request the body comparison above needs - a bare user turn would
// prove only that two empty bodies match.
func gemTunedRequest() Request {
	temp, top := 0.2, 0.9
	return Request{
		Model: "gemini-3-pro",
		Messages: []Message{
			{Role: RoleSystem, Content: "be brief"},
			{Role: RoleUser, Content: "x"},
		},
		Tools: []ToolDef{{
			Name:        "fs_read",
			Description: "read a file",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`),
		}},
		ResponseFormat:  &ResponseFormat{Name: "amele_output", Schema: json.RawMessage(`{"type":"object"}`)},
		MaxOutputTokens: 4096,
		Reasoning:       &ReasoningSpec{Effort: "medium"},
		Temperature:     &temp,
		TopP:            &top,
	}
}
