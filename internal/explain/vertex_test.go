package explain

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/lasthumanintheloop/amele/internal/config"
)

// vertexCfg is a gemini/vertex config that produces zero warnings, so each
// golden below differs from the other in exactly one dimension: the credential.
func vertexCfg() *config.Config {
	cfg := baseCfg()
	cfg.Model = "gemini-3.5-flash"
	cfg.Provider.Type = config.ProviderTypeGemini
	cfg.Provider.APIKey = "" // exit 2 next to a vertex block; explain reports either way
	cfg.Provider.BaseURL = ""
	cfg.Provider.Vertex = &config.VertexConfig{Project: "my-project", Location: "europe-west4"}
	return cfg
}

// TestVertexReportGolden pins the two rows a vertex run cannot be read without:
// the address the request will actually reach (host AND path, so the project
// and the location are visible where they are spent), and WHICH credential
// amele will look for.
//
// Goldens rather than substrings, per the docs/engineering.md §6 rule for a UI
// surface, and two of them because the credential row is the whole decision:
// a config that names a key file and one that falls through to the ADC chain
// are the two ways an operator gets this wrong.
//
// SECURITY: the auth row prints a PATH and a mode, never a token and never a
// byte of the file. `explain` is what people paste into issues.
func TestVertexReportGolden(t *testing.T) {
	tests := []struct {
		name   string
		golden string
		mutate func(*config.Config)
	}{
		{
			name:   "service account key file",
			golden: "explain-vertex-sa.txt",
			mutate: func(c *config.Config) { c.Provider.Vertex.Credentials = "/etc/amele/sa-key.json" },
		},
		{
			name:   "application default credentials",
			golden: "explain-vertex-adc.txt",
			mutate: func(*config.Config) {},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := vertexCfg()
			tc.mutate(cfg)
			got := Render(cfg, registryWith(t, fsBuiltins...), nil, nil, alwaysFound, nil)

			goldenPath := filepath.Join("testdata", "golden", tc.golden)
			if *update {
				if err := os.WriteFile(goldenPath, []byte(got), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			want, err := os.ReadFile(goldenPath) //nolint:gosec // G304: fixed testdata path.
			if err != nil {
				t.Fatalf("reading golden (run with -update to create): %v", err)
			}
			if got != string(want) {
				t.Errorf("report differs from golden.\ngot:\n%s\nwant:\n%s", got, want)
			}
		})
	}
}

// TestVertexEndpointRowIsTheClientsOwn pins that the reported address is built
// by the same function the client sends with, not by a second copy of the
// mapping that could drift: base_url replaces the HOST and nothing else, and
// the location still decides both the host shape and the path segment.
func TestVertexEndpointRowIsTheClientsOwn(t *testing.T) {
	tests := []struct {
		name     string
		location string
		baseURL  string
		want     string
	}{
		{
			name:     "a region gets the regional host",
			location: "europe-west4",
			want: "  vertex endpoint: https://europe-west4-aiplatform.googleapis.com/v1/projects/my-project" +
				"/locations/europe-west4/publishers/google/models/gemini-3.5-flash:generateContent\n",
		},
		{
			name:     "global loses the host prefix but keeps the path segment",
			location: "global",
			want: "  vertex endpoint: https://aiplatform.googleapis.com/v1/projects/my-project" +
				"/locations/global/publishers/google/models/gemini-3.5-flash:generateContent\n",
		},
		{
			name:     "the eu multi-region gets its own .rep. host",
			location: "eu",
			want: "  vertex endpoint: https://aiplatform.eu.rep.googleapis.com/v1/projects/my-project" +
				"/locations/eu/publishers/google/models/gemini-3.5-flash:generateContent\n",
		},
		{
			name:     "base_url replaces the host only",
			location: "europe-west4",
			baseURL:  "https://vpc-sc.internal:8443",
			want: "  vertex endpoint: https://vpc-sc.internal:8443/v1/projects/my-project" +
				"/locations/europe-west4/publishers/google/models/gemini-3.5-flash:generateContent\n",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := vertexCfg()
			cfg.Provider.Vertex.Location = tc.location
			cfg.Provider.BaseURL = tc.baseURL
			got := Render(cfg, nil, nil, nil, alwaysFound, nil)
			if !strings.Contains(got, tc.want) {
				t.Errorf("report is missing\n%q\ngot:\n%s", tc.want, got)
			}
		})
	}
}

// TestVertexEndpointRowSurvivesABrokenConfig: explain reports on configs that
// FAIL validation (that is the contract on Render), and a project or location
// that cannot become a URL is one of them. The row must then say so rather than
// print half an address - and it must not be able to forge a row out of the
// value that broke it.
func TestVertexEndpointRowSurvivesABrokenConfig(t *testing.T) {
	cfg := vertexCfg()
	cfg.Provider.Vertex.Location = "evil\nvertex auth:     forged"
	got := Render(cfg, nil, nil, nil, alwaysFound, nil)

	if !strings.Contains(got, "vertex endpoint: (unresolved:") {
		t.Errorf("a location that cannot become a URL did not report as unresolved:\n%s", got)
	}
	if strings.Contains(got, "\nvertex auth:     forged") {
		t.Errorf("a newline in the location forged a row:\n%s", got)
	}
	// The same value also reaches the base_url placeholder, which is bare
	// prose rather than a quoted field - it had to be escaped there too.
	if strings.Contains(got, "evil\n") {
		t.Errorf("an unescaped newline from the location survived into the report:\n%s", got)
	}
	// The auth row is independent of the endpoint and must still be there: the
	// operator's next question after "that address is wrong" is "with which
	// credential was I going to reach it".
	if !strings.Contains(got, "vertex auth:") {
		t.Errorf("the auth row disappeared with the endpoint:\n%s", got)
	}
}

// TestVertexEndpointRowNamesAnUnsetModel: the model is the last path segment,
// so an unset one would print "models/:generateContent" - a rendering bug to
// anyone reading it. The placeholder keeps the shape of the address honest
// about the segment PROBLEMS is already complaining about.
func TestVertexEndpointRowNamesAnUnsetModel(t *testing.T) {
	cfg := vertexCfg()
	cfg.Model = ""
	got := Render(cfg, nil, nil, nil, alwaysFound, nil)

	want := "publishers/google/models/{model}:generateContent\n"
	if !strings.Contains(got, want) {
		t.Errorf("report is missing %q:\n%s", want, got)
	}

	// The stand-in is a real string on its way through the URL builder, so a
	// config that happens to name it must still report its OWN address rather
	// than inherit the placeholder.
	cfg.Model = "-model-unset-"
	got = Render(cfg, nil, nil, nil, alwaysFound, nil)
	if strings.Contains(got, "{model}") {
		t.Errorf("a model named like the placeholder was reported as unset:\n%s", got)
	}
	if !strings.Contains(got, "publishers/google/models/-model-unset-:generateContent\n") {
		t.Errorf("report does not carry the configured model:\n%s", got)
	}
}

// TestVertexRowsStayOffTheAIStudioHalf: the same wire, the other backend. A
// config with no vertex block has no project, no location and no Google
// credential, so naming any of them would describe a request it will not send.
func TestVertexRowsStayOffTheAIStudioHalf(t *testing.T) {
	cfg := baseCfg()
	cfg.Provider.Type = config.ProviderTypeGemini
	cfg.Provider.BaseURL = ""
	got := Render(cfg, nil, nil, nil, alwaysFound, nil)

	for _, unwanted := range []string{"vertex endpoint:", "vertex auth:"} {
		if strings.Contains(got, unwanted) {
			t.Errorf("AI Studio report carries %q:\n%s", unwanted, got)
		}
	}
}
