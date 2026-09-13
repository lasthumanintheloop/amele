package doctor

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/lasthumanintheloop/amele/internal/config"
)

// server answers the models listing with the given status and records the
// request's path and credential header.
func server(t *testing.T, status int, header string) (*httptest.Server, *http.Request) {
	t.Helper()
	var seen http.Request
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = *r.Clone(context.Background())
		w.WriteHeader(status)
	}))
	t.Cleanup(srv.Close)
	_ = header
	return srv, &seen
}

func baseConfig(t *testing.T, provider config.ProviderConfig) *config.Config {
	t.Helper()
	return &config.Config{Model: "m", Provider: provider, Workspace: t.TempDir()}
}

// expect fails the test unless check c has the wanted verdict and its detail
// carries substr.
func expect(t *testing.T, c Check, want Status, substr string) {
	t.Helper()
	if c.Status != want || !strings.Contains(c.Detail, substr) {
		t.Fatalf("%s = %+v, want %s containing %q", c.Name, c, want, substr)
	}
}

func find(t *testing.T, r Report, name string) Check {
	t.Helper()
	for _, c := range r.Checks {
		if c.Name == name {
			return c
		}
	}
	t.Fatalf("no check named %q in %+v", name, r.Checks)
	return Check{}
}

// TestProviderProbe pins the verdict per answer, and the request each wire
// sends: the listing path and the credential header spelled as the run
// spells it.
func TestProviderProbe(t *testing.T) {
	cases := []struct {
		name       string
		status     int
		provider   func(base string) config.ProviderConfig
		wantStatus Status
		wantPath   string
		wantHeader string
		wantDetail string
	}{
		{"openai wire, key accepted", 200, func(b string) config.ProviderConfig {
			return config.ProviderConfig{BaseURL: b + "/v1", APIKey: "sk-1"}
		}, Pass, "/v1/models", "Authorization", "key accepted"},
		{"openai wire, key rejected", 401, func(b string) config.ProviderConfig {
			return config.ProviderConfig{BaseURL: b + "/v1", APIKey: "sk-1"}
		}, Fail, "/v1/models", "Authorization", "key rejected (HTTP 401)"},
		{"a gateway without a listing", 404, func(b string) config.ProviderConfig {
			return config.ProviderConfig{BaseURL: b + "/v1", APIKey: "sk-1", Dialect: "openrouter"}
		}, Warn, "/v1/models", "Authorization", "no models listing"},
		{"a server error", 503, func(b string) config.ProviderConfig {
			return config.ProviderConfig{BaseURL: b + "/v1", APIKey: "sk-1"}
		}, Fail, "/v1/models", "Authorization", "HTTP 503"},
		{"anthropic wire", 200, func(b string) config.ProviderConfig {
			return config.ProviderConfig{Type: config.ProviderTypeAnthropic, BaseURL: b, APIKey: "sk-ant"}
		}, Pass, "/v1/models", "X-Api-Key", "anthropic: "},
		{"gemini wire", 200, func(b string) config.ProviderConfig {
			return config.ProviderConfig{Type: config.ProviderTypeGemini, BaseURL: b, APIKey: "AIza"}
		}, Pass, "/v1beta/models", "X-Goog-Api-Key", "gemini: "},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, seen := server(t, tc.status, "")
			cfg := baseConfig(t, tc.provider(srv.URL))
			got := find(t, Run(context.Background(), cfg, Options{}), "provider")
			if got.Status != tc.wantStatus || !strings.Contains(got.Detail, tc.wantDetail) {
				t.Fatalf("provider = %+v, want %s containing %q", got, tc.wantStatus, tc.wantDetail)
			}
			if seen.URL == nil || seen.URL.Path != tc.wantPath {
				t.Errorf("probed path = %v, want %s", seen.URL, tc.wantPath)
			}
			if seen.Header.Get(tc.wantHeader) == "" {
				t.Errorf("probe sent no %s header: %v", tc.wantHeader, seen.Header)
			}
			if strings.Contains(got.Detail, "sk-") || strings.Contains(got.Detail, "AIza") {
				t.Errorf("detail leaks the credential: %q", got.Detail)
			}
		})
	}

}

// TestProviderProbeEdges covers the answers that never reach the listing: a
// dead endpoint, an empty key, a Vertex target, and a chain.
func TestProviderProbeEdges(t *testing.T) {
	t.Run("unreachable", func(t *testing.T) {
		srv, _ := server(t, 200, "")
		srv.Close()
		cfg := baseConfig(t, config.ProviderConfig{BaseURL: srv.URL + "/v1", APIKey: "sk-1"})
		got := find(t, Run(context.Background(), cfg, Options{}), "provider")
		expect(t, got, Fail, "unreachable")
		if strings.Contains(got.Detail, "/v1/models") {
			t.Fatalf("the detail names the path: %q", got.Detail)
		}
	})
	t.Run("a redirect is not followed", func(t *testing.T) {
		// SECURITY: the elsewhere server must never see the key.
		var leaked bool
		elsewhere := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			leaked = leaked || r.Header.Get("X-Api-Key") != "" || r.Header.Get("Authorization") != ""
			w.WriteHeader(http.StatusOK)
		}))
		t.Cleanup(elsewhere.Close)
		redirecting := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Redirect(w, &http.Request{URL: &url.URL{}}, elsewhere.URL+"/v1/models", http.StatusMovedPermanently)
		}))
		t.Cleanup(redirecting.Close)
		cfg := baseConfig(t, config.ProviderConfig{Type: config.ProviderTypeAnthropic, BaseURL: redirecting.URL, APIKey: "sk-ant"})
		got := find(t, Run(context.Background(), cfg, Options{}), "provider")
		expect(t, got, Fail, "redirects the API (HTTP 301")
		if leaked {
			t.Fatal("the probe followed the redirect with the credential")
		}
	})
	t.Run("an empty key fails before any request", func(t *testing.T) {
		cfg := baseConfig(t, config.ProviderConfig{BaseURL: "http://127.0.0.1:1/v1"})
		expect(t, find(t, Run(context.Background(), cfg, Options{}), "provider"), Fail, "api_key is empty")
	})
	t.Run("vertex is not probed", func(t *testing.T) {
		cfg := baseConfig(t, config.ProviderConfig{Type: config.ProviderTypeGemini, Vertex: &config.VertexConfig{Project: "p", Location: "us-central1"}})
		expect(t, find(t, Run(context.Background(), cfg, Options{}), "provider"), Warn, "gemini/vertex: not probed")
	})
	t.Run("every fallback target is probed", func(t *testing.T) {
		ok, _ := server(t, 200, "")
		bad, _ := server(t, 401, "")
		cfg := baseConfig(t, config.ProviderConfig{BaseURL: ok.URL + "/v1", APIKey: "sk-1", Fallback: []config.FallbackTarget{
			{Model: "b", ProviderConfig: config.ProviderConfig{BaseURL: bad.URL + "/v1", APIKey: "sk-2"}},
		}})
		r := Run(context.Background(), cfg, Options{})
		expect(t, find(t, r, "provider"), Pass, "key accepted")
		expect(t, find(t, r, "provider.fallback[0]"), Fail, "key rejected")
		if !r.Failed() {
			t.Fatal("a failing fallback probe must fail the report")
		}
	})
}

// healthyProvider is an endpoint that accepts every probe.
func healthyProvider(t *testing.T) config.ProviderConfig {
	t.Helper()
	srv, _ := server(t, 200, "")
	return config.ProviderConfig{BaseURL: srv.URL + "/v1", APIKey: "sk-1"}
}

func TestHealthyConfigPasses(t *testing.T) {
	cfg := baseConfig(t, healthyProvider(t))
	cfg.SessionDir = filepath.Join(t.TempDir(), "sessions")
	r := Run(context.Background(), cfg, Options{})
	if r.Failed() {
		t.Fatalf("report failed:\n%s", r.Render())
	}
	expect(t, find(t, r, "session_dir"), Pass, "writable")
	if _, err := os.Stat(cfg.SessionDir); err != nil {
		t.Errorf("session_dir was not created: %v", err)
	}
	if out := r.Render(); !strings.HasSuffix(out, "failed\n") || !strings.Contains(out, "[PASS] config: loads and validates") {
		t.Errorf("render:\n%s", out)
	}
}

func TestProblemsFailTheConfigCheck(t *testing.T) {
	r := Run(context.Background(), baseConfig(t, healthyProvider(t)), Options{Problems: []string{"model is required", "x"}})
	expect(t, find(t, r, "config"), Fail, "model is required; x")
}

func TestWorkspaceChecks(t *testing.T) {
	t.Run("a missing workspace fails", func(t *testing.T) {
		cfg := baseConfig(t, healthyProvider(t))
		cfg.Workspace = filepath.Join(t.TempDir(), "gone")
		expect(t, find(t, Run(context.Background(), cfg, Options{}), "workspace"), Fail, "gone")
	})
	t.Run("an unwritable workspace warns without fs tools and fails with them", func(t *testing.T) {
		if os.Getuid() == 0 {
			t.Skip("root writes everywhere")
		}
		cfg := baseConfig(t, healthyProvider(t))
		if err := os.Chmod(cfg.Workspace, 0o500); err != nil { //nolint:gosec // G302: a read-only directory is the fixture.
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = os.Chmod(cfg.Workspace, 0o700) }) //nolint:gosec // G302: restoring the temp dir for cleanup.
		expect(t, find(t, Run(context.Background(), cfg, Options{}), "workspace"), Warn, "not writable")
		cfg.Tools.FS = true
		expect(t, find(t, Run(context.Background(), cfg, Options{}), "workspace"), Fail, "not writable")
	})
}

func TestExecutableChecks(t *testing.T) {
	cfg := baseConfig(t, healthyProvider(t))
	cfg.Tools.Subprocess = []config.SubprocessTool{{Name: "grep_logs", Command: []string{"rg", "-n"}}}
	cfg.MCP.Servers = []config.MCPServer{
		{Name: "files", Transport: config.MCPTransport{Type: config.MCPTransportStdio, Command: []string{"mcp-files"}}},
		{Name: "remote", Transport: config.MCPTransport{Type: config.MCPTransportHTTP, URL: "https://mcp.example.com/sse"}},
	}
	probe := func(name string) error {
		if name == "rg" {
			return nil
		}
		return errors.New("not on PATH")
	}
	r := Run(context.Background(), cfg, Options{Probe: probe})
	expect(t, find(t, r, "tool grep_logs"), Pass, "rg: found")
	expect(t, find(t, r, "mcp files"), Fail, "mcp-files: not found")
	expect(t, find(t, r, "mcp remote"), Pass, "mcp.example.com")
}

func TestTTYAndLockChecks(t *testing.T) {
	t.Run("ask policies against the terminal", func(t *testing.T) {
		cfg := baseConfig(t, healthyProvider(t))
		expect(t, find(t, Run(context.Background(), cfg, Options{}), "tty"), Pass, "no ask policy")
		cfg.Permissions.Tools = map[string]config.Policy{"shell": config.PolicyAsk}
		expect(t, find(t, Run(context.Background(), cfg, Options{}), "tty"), Warn, "auto-deny")
		expect(t, find(t, Run(context.Background(), cfg, Options{IsTTY: func() bool { return true }}), "tty"), Pass, "terminal is attached")
	})
	t.Run("the lock directory", func(t *testing.T) {
		cfg := baseConfig(t, healthyProvider(t))
		expect(t, find(t, Run(context.Background(), cfg, Options{}), "lock"), Pass, "not set")
		cfg.Lock = true
		lock := filepath.Join(t.TempDir(), "agent.yaml.lock")
		expect(t, find(t, Run(context.Background(), cfg, Options{LockPath: lock}), "lock"), Pass, "directory writable")
		missing := filepath.Join(t.TempDir(), "gone", "x.lock")
		expect(t, find(t, Run(context.Background(), cfg, Options{LockPath: missing}), "lock"), Fail, "cannot create the lock file")
	})
}
