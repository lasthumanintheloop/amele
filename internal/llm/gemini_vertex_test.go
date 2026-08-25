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
			// The exported reporting seam must answer with the very same URL:
			// `amele explain` prints it, and a report that could drift from
			// the request would be worse than no report.
			reported, err := tt.target.Endpoint(tt.base, tt.model)
			if err != nil {
				t.Fatalf("VertexTarget.Endpoint: %v", err)
			}
			if reported != got {
				t.Errorf("VertexTarget.Endpoint:\n got %q\nwant %q (the client's own)", reported, got)
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

// TestValidVertexID pins the charset itself, which is exported because
// internal/config validates against this same function rather than a copy of
// the rule (see its CONTRACT note). The rows are the boundary cases: what may
// lead, what may not, and every separator that would matter in a hostname or a
// path.
func TestValidVertexID(t *testing.T) {
	valid := []string{"my-project", "p", "p1", "123456789012", "global", "us", "europe-west4", "a-b-c-1"}
	for _, v := range valid {
		if !ValidVertexID(v) {
			t.Errorf("ValidVertexID(%q) = false, want true", v)
		}
	}

	invalid := []string{
		"",                 // no coordinate at all
		"-leading",         // cannot start a DNS label
		"My-Project", "US", // upper case: GCP has none, and the host is lowercase
		"under_score", "a b", // not host- or path-safe
		"dot.ted",         // would extend the hostname
		"a/b", "../other", // would climb the path
		"p@h", "p:1", "p?q", // userinfo, port, query
	}
	for _, v := range invalid {
		if ValidVertexID(v) {
			t.Errorf("ValidVertexID(%q) = true, want false", v)
		}
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
// so the field set of the WHOLE request tree is pinned here, not just its root.
//
// The tree matters because the research finds divergences at every level, not
// only at the top: Part is Vertex-only for audioTranscription and AI
// Studio-only for mediaProcessing/partMetadata/toolCall/toolResponse (§2.6),
// and Tool is Vertex-only for enterpriseWebSearch/retrieval and AI Studio-only
// for fileSearch/mcpServers (§2.7). A future `mcpServers` on gemTool would 400
// on Vertex; a root-only pin would not have noticed.
//
// A new field appearing in any of these lists fails the test on purpose:
// whoever adds it must check it against the diff table in
// docs/superpowers/specs/2026-08-25-vertex-adc-research.md §2 and decide
// whether vertex mode needs to spell it differently, rather than discovering
// the difference as a 400 from a live run.
//
// ONE EXCLUSION, deliberate: the walk reads json tags, so a field tagged `json:"-"`
// is invisible to it - gemContent.PartsRaw is the only one today, and it is the
// verbatim echo carrier, which by construction sends back bytes the PROVIDER
// produced rather than a field amele chose to spell. The wire goldens are the
// backstop for that path. Response types are out of scope here too: the client
// decodes only the shared fields, and a field it does not decode cannot be
// mis-sent.
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

	// Every struct the request body is assembled from, with the note that says
	// why the diff table does not reach it as it stands.
	pins := []struct {
		name string
		zero any
		want []string
		why  string
	}{
		{
			name: "gemRequest",
			zero: gemRequest{},
			want: []string{"contents", "generationConfig", "systemInstruction", "tools"},
			why:  "labels (vertex-only) and serviceTier/store/model (AI Studio-only) are all absent; safetySettings is owned but never written, so its vertex-only `method` key has nothing to attach to",
		},
		{
			name: "gemGenerationConfig",
			zero: gemGenerationConfig{},
			want: []string{
				"maxOutputTokens", "responseJsonSchema", "responseMimeType",
				"stopSequences", "temperature", "thinkingConfig", "topP",
			},
			why: "topK is absent (its TYPE differs: float on vertex, integer on AI Studio) and responseFormat is absent (its SHAPE differs: an array vs an object); structured output travels as responseJsonSchema, which is shared",
		},
		{
			name: "gemThinking",
			zero: gemThinking{},
			want: []string{"thinkingBudget", "thinkingLevel"},
			why:  "thinkingConfig has an identical field set on both backends (§2.4); only the enum ORDER differs, which is not a wire fact",
		},
		{
			name: "gemContent",
			zero: gemContent{},
			want: []string{"parts", "role"},
			why:  "Content is exactly identical on both backends (§2.7)",
		},
		{
			name: "gemPart",
			zero: gemPart{},
			want: []string{"functionCall", "functionResponse", "text", "thought", "thoughtSignature"},
			why:  "all five are shared; the diverging Part keys - audioTranscription on vertex, mediaProcessing/partMetadata/toolCall/toolResponse on AI Studio - are none of them, and thoughtSignature is byte-identical on both (§2.6)",
		},
		{
			name: "gemFunctionCall",
			zero: gemFunctionCall{},
			want: []string{"args", "id", "name"},
			why:  "FunctionCall is shared",
		},
		{
			name: "gemFunctionResponse",
			zero: gemFunctionResponse{},
			want: []string{"id", "name", "response"},
			why:  "FunctionResponse is shared",
		},
		{
			name: "gemTool",
			zero: gemTool{},
			want: []string{"functionDeclarations"},
			why:  "the ONE shared tool kind amele uses; the vertex-only kinds (enterpriseWebSearch, retrieval, ...) and the AI Studio-only ones (fileSearch, mcpServers) are all absent (§2.7)",
		},
		{
			name: "gemFunctionDecl",
			zero: gemFunctionDecl{},
			want: []string{"description", "name", "parameters"},
			why:  "FunctionDeclaration is exactly identical on both backends - the core tool-calling contract is portable (§2.7)",
		},
	}

	for _, pin := range pins {
		t.Run(pin.name, func(t *testing.T) {
			if got := tags(pin.zero); !slices.Equal(got, pin.want) {
				t.Errorf("%s changed shape: got %v, want %v\ninert because: %s\ncheck the new field against the vertex diff table before shipping it",
					pin.name, got, pin.want, pin.why)
			}
		})
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

// quotaTokenSource is a token source that also names a quota project - the
// optional half of the auth seam, implemented by the ADC leg that needs it.
type quotaTokenSource struct {
	fakeTokenSource
	project string
}

func (q *quotaTokenSource) QuotaProject() string { return q.project }

// TestVertexQuotaProjectHeader pins the x-goog-user-project behavior in both
// directions: the header travels only when the CREDENTIAL asks for it.
//
// It is not sent unconditionally on purpose. The header names the project
// billed for quota and requires serviceusage.services.use on it; a service
// account that legitimately holds only roles/aiplatform.user would start
// getting 403s from a header it never needed (research §3.1, "Quota project").
func TestVertexQuotaProjectHeader(t *testing.T) {
	tests := []struct {
		name   string
		source GeminiTokenSource
		want   string
	}{
		{
			name:   "a source that names a quota project sends the header",
			source: &quotaTokenSource{fakeTokenSource: fakeTokenSource{token: "t"}, project: "my-project"},
			want:   "my-project",
		},
		{
			name:   "a source with an empty quota project sends nothing",
			source: &quotaTokenSource{fakeTokenSource: fakeTokenSource{token: "t"}},
			want:   "",
		},
		{
			name:   "a source that does not know about quota projects sends nothing",
			source: &fakeTokenSource{token: "t"},
			want:   "",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var got string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				got = r.Header.Get("x-goog-user-project")
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(`{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`))
			}))
			defer srv.Close()

			client := &GeminiClient{
				BaseURL:     srv.URL,
				Vertex:      &VertexTarget{Project: "my-project", Location: "us-central1"},
				TokenSource: tc.source,
			}
			if _, err := client.Chat(context.Background(), gemUserRequest()); err != nil {
				t.Fatalf("Chat: %v", err)
			}
			if got != tc.want {
				t.Errorf("x-goog-user-project = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestVertexFailureAdvice: a 401, 403 or 404 from the Vertex endpoint says
// nothing an operator can act on ("Request is missing required authentication
// credential" - research §2.10), so amele appends the vocabulary the fix
// actually lives in: the role, the API, the project and the location.
//
// Every row is mode-dependent, which is why none of them is an entry in
// geminiErrorSignatures: the same status on the AI Studio backend is about an
// api_key or a model id, and that table is consulted for 400s on both backends
// alike.
func TestVertexFailureAdvice(t *testing.T) {
	tests := []struct {
		name string
		code int
		want []string
	}{
		{
			name: "401 names the credential sources and the role",
			code: http.StatusUnauthorized,
			want: []string{"roles/aiplatform.user", "my-project", "provider.vertex.credentials"},
		},
		{
			name: "403 names the role, the location, the API and the quota header",
			code: http.StatusForbidden,
			want: []string{
				"roles/aiplatform.user", "my-project", "europe-west4",
				"serviceusage.services.use", "aiplatform.googleapis.com",
			},
		},
		{
			// A 404 here is almost never "no such project": region support is
			// per-model and much narrower than the 47-region host list
			// (research §1.5 - gemini-3.5-flash is not served in us-central1,
			// and two Gemini 3 models are global-only), while the location is
			// never rerouted for the operator. Both halves of that answer have
			// to be in the message, because the API's own body says only that
			// the publisher model was not found.
			name: "404 names the model id and the location together",
			code: http.StatusNotFound,
			want: []string{"europe-west4", "model"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.code)
				_, _ = w.Write([]byte(`{"error":{"code":401,"status":"UNAUTHENTICATED"}}`))
			}))
			defer srv.Close()

			client := &GeminiClient{
				BaseURL:     srv.URL,
				Vertex:      &VertexTarget{Project: "my-project", Location: "europe-west4"},
				TokenSource: &fakeTokenSource{token: "t"},
			}
			_, err := client.Chat(context.Background(), gemUserRequest())
			if err == nil {
				t.Fatal("expected an error")
			}
			for _, want := range tc.want {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("error %q does not mention %q", err, want)
				}
			}
		})
	}
}

// TestAIStudioAuthFailureKeepsItsOwnMessage is the other half of the pin above:
// the vertex IAM advice must not attach to an AI Studio 401, where the answer
// is an api_key and no Google Cloud project exists to name.
func TestAIStudioAuthFailureKeepsItsOwnMessage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	client := &GeminiClient{BaseURL: srv.URL, APIKey: "k"}
	_, err := client.Chat(context.Background(), gemUserRequest())
	if err == nil {
		t.Fatal("expected an error")
	}
	if strings.Contains(err.Error(), "roles/aiplatform.user") {
		t.Errorf("the vertex IAM advice reached an AI Studio failure: %v", err)
	}
	if strings.Contains(err.Error(), "api keys are not supported") {
		t.Errorf("the express-mode advice claimed a plain AI Studio 401: %v", err)
	}
}

// expressKeyRejection is the live body the Vertex endpoint answers an API key
// with - reproduced twice against the real service, for both :generateContent
// and :streamGenerateContent (research §3.3). It is NOT an "invalid key" error:
// it is the generic ESF response for a method that has no API-key auth
// configured at all, which is why documented express-mode keys do not work
// here and why the advice sends the operator to a vertex block rather than to
// a different key.
const expressKeyRejection = `{"error":{"code":401,"message":"API keys are not supported by this API. ` +
	`Expected OAuth2 access token or other authentication credentials that assert a principal. ` +
	`See https://cloud.google.com/docs/authentication","status":"UNAUTHENTICATED"}}`

// TestExpressModeKeyRejectionAdvice: the one way an operator can reach this
// 401 through amele is the keyed backend pointed at a Vertex host - config
// refuses api_key next to a vertex block (exit 2), and the client never sends
// x-goog-api-key in vertex mode. So the advice must fire on the AI STUDIO half,
// where the config change is "stop using a key here".
func TestExpressModeKeyRejectionAdvice(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(expressKeyRejection))
	}))
	defer srv.Close()

	client := &GeminiClient{BaseURL: srv.URL, APIKey: "k"}
	_, err := client.Chat(context.Background(), gemUserRequest())
	if err == nil {
		t.Fatal("expected an error")
	}
	for _, want := range []string{
		"vertex requires OAuth credentials; api keys are not supported",
		"provider.vertex",
	} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

// TestExpressModeAdviceYieldsToTheCredentialAdvice: the same body reaching a
// client that IS in vertex mode (a proxy in front of the endpoint, say) must
// keep the credential advice. There, the api-key sentence would be a dead end -
// amele sent no key, and the operator's question is why the OAuth token was
// refused.
func TestExpressModeAdviceYieldsToTheCredentialAdvice(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(expressKeyRejection))
	}))
	defer srv.Close()

	client := &GeminiClient{
		BaseURL:     srv.URL,
		Vertex:      &VertexTarget{Project: "my-project", Location: "europe-west4"},
		TokenSource: &fakeTokenSource{token: "t"},
	}
	_, err := client.Chat(context.Background(), gemUserRequest())
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "roles/aiplatform.user") {
		t.Errorf("the vertex credential advice was lost: %v", err)
	}
	if strings.Contains(err.Error(), "api keys are not supported") {
		t.Errorf("the express-mode advice shadowed the credential advice: %v", err)
	}
}

// TestAIStudioNotFoundKeepsItsOwnMessage: the location half of the 404 advice
// is meaningless on a wire that has no locations, so it must not attach there.
func TestAIStudioNotFoundKeepsItsOwnMessage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	client := &GeminiClient{BaseURL: srv.URL, APIKey: "k"}
	_, err := client.Chat(context.Background(), gemUserRequest())
	if err == nil {
		t.Fatal("expected an error")
	}
	if strings.Contains(err.Error(), "location") {
		t.Errorf("the vertex 404 advice reached an AI Studio failure: %v", err)
	}
}
