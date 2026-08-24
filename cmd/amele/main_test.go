package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	goyaml "gopkg.in/yaml.v3"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/llm"
	"github.com/lasthumanintheloop/amele/internal/runlock"
	"github.com/lasthumanintheloop/amele/internal/schema"
	"github.com/lasthumanintheloop/amele/internal/session"
)

// capturedMessage is one conversation entry as it appeared on the wire.
type capturedMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// capturedRequest is the subset of an OpenAI-compatible request body the e2e
// tests assert on.
type capturedRequest struct {
	Model          string            `json:"model"`
	Messages       []capturedMessage `json:"messages"`
	ResponseFormat *struct {
		Type       string `json:"type"`
		JSONSchema struct {
			Name   string          `json:"name"`
			Schema json.RawMessage `json:"schema"`
		} `json:"json_schema"`
	} `json:"response_format"`
}

// capturingServer serves canned OpenAI-compatible responses in order and
// records every decoded request, so a test can assert on what the loop
// actually sent (the retry tests need to see the validator feedback on the
// wire). It is the e2e stand-in for a real provider; unit tests never leave
// localhost. A call beyond the scripted bodies fails the test, which is how a
// test pins the exact number of provider round-trips.
//
// The returned slice is only safe to read after the run has returned.
func capturingServer(t *testing.T, bodies ...string) (*httptest.Server, *[]capturedRequest) {
	t.Helper()
	var (
		call int
		reqs []capturedRequest
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var got capturedRequest
		_ = json.NewDecoder(r.Body).Decode(&got)
		reqs = append(reqs, got)
		if call >= len(bodies) {
			t.Errorf("unexpected extra provider call #%d", call+1)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte(bodies[call]))
		call++
	}))
	t.Cleanup(srv.Close)
	return srv, &reqs
}

// scriptedServer is capturingServer for the tests that only care about the
// responses, not about what was sent.
func scriptedServer(t *testing.T, bodies ...string) *httptest.Server {
	t.Helper()
	srv, _ := capturingServer(t, bodies...)
	return srv
}

// lastContent returns the content of the last message carrying this role, or
// "" when the request had none.
func lastContent(msgs []capturedMessage, role string) string {
	var content string
	for _, m := range msgs {
		if m.Role == role {
			content = m.Content
		}
	}
	return content
}

func textBody(content string) string {
	b, _ := json.Marshal(content)
	return fmt.Sprintf(`{
		"choices": [{"message": {"role": "assistant", "content": %s}, "finish_reason": "stop"}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5}
	}`, b)
}

func toolCallBody(name, args string) string {
	a, _ := json.Marshal(args)
	return fmt.Sprintf(`{
		"choices": [{"message": {"role": "assistant", "content": "",
			"tool_calls": [{"id": "c1", "type": "function", "function": {"name": %q, "arguments": %s}}]},
			"finish_reason": "tool_calls"}],
		"usage": {"prompt_tokens": 10, "completion_tokens": 5}
	}`, name, a)
}

// writeTestConfig renders a config bound to the test server and a temp
// workspace, returning the config path and the workspace dir.
func writeTestConfig(t *testing.T, baseURL, extra string) (string, string) {
	t.Helper()
	dir := t.TempDir()
	yaml := fmt.Sprintf(`
model: test-model
provider:
  base_url: %s/v1
  api_key: ${TEST_KEY}
system_prompt: "You are a test agent."
tools:
  fs: true
%s`, baseURL, extra)
	path := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	return path, dir
}

// testInterpolatedValue is an interpolated environment value that is NOT the
// API key - the case the session redactor exists for (a DB password in a
// prompt), and the one a redaction test must use to prove the redaction
// covers more than the key. Named around "interpolated" rather than around
// what it stands for so gosec's G101 name heuristic does not read a test
// fixture as a checked-in credential.
const testInterpolatedValue = "pw-hunter2-db-value"

// env returns the standard test environment.
func env(t *testing.T) func(string) (string, bool) {
	t.Helper()
	return func(key string) (string, bool) {
		switch key {
		case "TEST_KEY":
			return "sk-test-secret-key", true
		case "DB_PASSWORD":
			return testInterpolatedValue, true
		}
		return "", false
	}
}

// execCLI runs the CLI entry point with captured output.
func execCLI(t *testing.T, args []string, stdin string) (code int, stdout, stderr string) {
	t.Helper()
	var out, errBuf bytes.Buffer
	code = run(context.Background(), args, strings.NewReader(stdin), &out, &errBuf, env(t))
	return code, out.String(), errBuf.String()
}

func TestE2ESimpleAnswer(t *testing.T) {
	srv := scriptedServer(t, textBody("final answer"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "what", "is", "up"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	// CONTRACT: stdout carries only the final answer.
	if stdout != "final answer\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if !strings.Contains(stderr, "1 turn,") {
		t.Errorf("summary missing: %q", stderr)
	}
}

func TestE2EToolUseAndSession(t *testing.T) {
	srv := scriptedServer(t,
		toolCallBody("fs_read", `{"path":"data.txt"}`),
		textBody("read it"),
	)
	cfgPath, dir := writeTestConfig(t, srv.URL, "session_dir: sessions\n")
	if err := os.WriteFile(filepath.Join(dir, "data.txt"), []byte("workspace content"), 0o600); err != nil {
		t.Fatal(err)
	}

	code, stdout, _ := execCLI(t, []string{"run", cfgPath, "read data.txt"}, "")
	if code != ExitOK || stdout != "read it\n" {
		t.Fatalf("code=%d stdout=%q", code, stdout)
	}

	// The session JSONL must exist and contain the full event sequence.
	files, err := filepath.Glob(filepath.Join(dir, "sessions", "*.jsonl"))
	if err != nil || len(files) != 1 {
		t.Fatalf("session files: %v, %v", files, err)
	}
	data, _ := os.ReadFile(files[0])
	for _, evt := range []string{"run_start", "llm_response", "tool_call", "tool_result", "run_end"} {
		if !strings.Contains(string(data), evt) {
			t.Errorf("session log missing %q", evt)
		}
	}
	if strings.Contains(string(data), "sk-test-secret-key") {
		t.Error("api key leaked into session log")
	}
}

// TestE2EContentFreePromptExitsTwo is the CLI half of live-test finding B-A03:
// `prompt: "{{input}}"` with nothing on stdin used to send the model an empty
// user message and bill the operator for the round trip. The refusal is a
// config-level one (exit 2) and must happen before the provider is contacted -
// zero scripted bodies make an unexpected call fail the test.
func TestE2EContentFreePromptExitsTwo(t *testing.T) {
	srv := scriptedServer(t)
	cfgPath, dir := writeTestConfig(t, srv.URL, "prompt: \"{{input}}\"\nsession_dir: sessions\n")

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath}, "")
	if code != ExitConfigError {
		t.Fatalf("exit %d, want %d; stderr: %s", code, ExitConfigError, stderr)
	}
	if stdout != "" {
		t.Errorf("a refused run must write nothing to stdout: %q", stdout)
	}
	if !strings.Contains(stderr, "empty prompt") {
		t.Errorf("stderr must name the problem: %q", stderr)
	}
	// Nothing started, so nothing is logged: the refusal is as cheap as the
	// lock refusal above.
	if files, _ := filepath.Glob(filepath.Join(dir, "sessions", "*.jsonl")); len(files) != 0 {
		t.Errorf("a refused run must not open a session log: %v", files)
	}
}

// TestEmptyPromptMessageClaimsOnlyWhatItKnows: the refusal used to assert
// "both {{args}} and {{input}} were empty", which is a guess - a template with
// no placeholders at all (prompt: "   ") renders to whitespace no matter what
// the operator passed, and the sentence then blamed input that was never read.
func TestEmptyPromptMessageClaimsOnlyWhatItKnows(t *testing.T) {
	srv := scriptedServer(t) // any provider call fails the test
	cfgPath, _ := writeTestConfig(t, srv.URL, "prompt: \"   \"\n")

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "a real task"}, "piped text")
	if code != ExitConfigError {
		t.Fatalf("exit %d, want %d; stderr: %s", code, ExitConfigError, stderr)
	}
	if !strings.Contains(stderr, "empty prompt") {
		t.Errorf("stderr must name the problem: %q", stderr)
	}
	if strings.Contains(stderr, "were empty") {
		t.Errorf("stderr blames placeholders the template never had: %q", stderr)
	}
}

// TestWhitespaceOnlyArgsFallBackToStdin: whitespace-only task text is no task
// at all, so it must not suppress the stdin read. It used to: needStdin tested
// taskArgs untrimmed, so `amele run cfg " " < file` skipped the pipe and then
// refused with "pass task text as arguments or pipe input on stdin" - advice
// the operator had already followed twice over.
func TestWhitespaceOnlyArgsFallBackToStdin(t *testing.T) {
	srv, reqs := capturingServer(t, textBody("done"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "  "}, "the real task\n")
	if code != ExitOK {
		t.Fatalf("exit %d, want %d; stderr: %s", code, ExitOK, stderr)
	}
	if stdout != "done\n" {
		t.Errorf("stdout = %q", stdout)
	}
	if got := lastContent((*reqs)[0].Messages, "user"); !strings.Contains(got, "the real task") {
		t.Errorf("user message = %q, want the piped text", got)
	}
}

// TestE2EPromptTemplateWithRealStdinStillRuns is the other half: the guard must
// not touch the case the template exists for.
func TestE2EPromptTemplateWithRealStdinStillRuns(t *testing.T) {
	srv, reqs := capturingServer(t, textBody("summarized"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "prompt: \"Summarize: {{input}}\"\n")

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath}, "  log line  ")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "summarized\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if got := lastContent((*reqs)[0].Messages, "user"); got != "Summarize:   log line  " {
		t.Errorf("user message = %q, want the rendered template verbatim", got)
	}
}

// The run lock tests below share one technique: the concurrent "other run" is
// a live runlock.Acquire inside this test process. flock's owner is the open
// file description, so a second acquisition is refused exactly as a second
// process would be - no subprocess needed to reproduce the race.

// TestE2ERunLockHeldExitsSeven pins the contract a cron wrapper branches on: a
// run blocked by a concurrent one exits 7 and spends nothing.
func TestE2ERunLockHeldExitsSeven(t *testing.T) {
	// Zero scripted bodies: a provider call would fail the test, which is how
	// "the blocked run spends nothing" is pinned.
	srv := scriptedServer(t)
	cfgPath, _ := writeTestConfig(t, srv.URL, "lock: true\n")

	release, err := runlock.Acquire(cfgPath + ".lock")
	if err != nil {
		t.Fatalf("holding the lock: %v", err)
	}
	defer release()

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "task"}, "")
	if code != ExitLockHeld {
		t.Fatalf("exit %d, want %d; stderr: %s", code, ExitLockHeld, stderr)
	}
	if stdout != "" {
		t.Errorf("a blocked run must write nothing to stdout: %q", stdout)
	}
	if !strings.Contains(stderr, "another run holds the lock for this config") ||
		!strings.Contains(stderr, cfgPath+".lock") {
		t.Errorf("stderr must name the problem and the lock file: %q", stderr)
	}
}

// TestE2ERunLockReleasesAfterRun pins that the lock lives exactly as long as
// the run: the lock file stays behind (unlinking it would race a concurrent
// holder), the lock itself does not.
func TestE2ERunLockReleasesAfterRun(t *testing.T) {
	srv := scriptedServer(t, textBody("first"), textBody("second"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "lock: true\n")

	if code, _, stderr := execCLI(t, []string{"run", cfgPath, "task"}, ""); code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if _, err := os.Stat(cfgPath + ".lock"); err != nil {
		t.Errorf("lock file should persist after the run: %v", err)
	}
	if code, _, stderr := execCLI(t, []string{"run", cfgPath, "task"}, ""); code != ExitOK {
		t.Fatalf("second sequential run blocked, so the lock was not released: exit %d, stderr: %s", code, stderr)
	}
}

// TestE2ERunLockIsOptIn pins the default: without lock: true nothing locks and
// no lock file appears next to the config.
func TestE2ERunLockIsOptIn(t *testing.T) {
	srv := scriptedServer(t, textBody("ok"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	if code, _, stderr := execCLI(t, []string{"run", cfgPath, "task"}, ""); code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if _, err := os.Stat(cfgPath + ".lock"); !os.IsNotExist(err) {
		t.Errorf("a config without lock: true must not create a lock file (stat err = %v)", err)
	}
}

// TestE2ERunLockOnlyAppliesToRun pins that the inspection commands never lock:
// checking a config while a run is in progress is exactly when an operator
// reaches for them.
func TestE2ERunLockOnlyAppliesToRun(t *testing.T) {
	srv := scriptedServer(t)
	cfgPath, _ := writeTestConfig(t, srv.URL, "lock: true\n")

	release, err := runlock.Acquire(cfgPath + ".lock")
	if err != nil {
		t.Fatalf("holding the lock: %v", err)
	}
	defer release()

	for _, cmd := range []string{"validate", "explain"} {
		if code, _, stderr := execCLI(t, []string{cmd, cfgPath}, ""); code != ExitOK {
			t.Errorf("%s exited %d while the lock was held; stderr: %s", cmd, code, stderr)
		}
	}
}

// TestE2ERunLockUnusablePathIsConfigError pins the error taxonomy: a lock file
// that cannot be opened is a broken setup (exit 2), not contention (exit 7).
func TestE2ERunLockUnusablePathIsConfigError(t *testing.T) {
	srv := scriptedServer(t)
	cfgPath, _ := writeTestConfig(t, srv.URL, "lock: true\n")
	if err := os.Mkdir(cfgPath+".lock", 0o750); err != nil {
		t.Fatal(err)
	}

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "task"}, "")
	if code != ExitConfigError {
		t.Fatalf("exit %d, want %d; stderr: %s", code, ExitConfigError, stderr)
	}
	if strings.Contains(stderr, "another run holds the lock") {
		t.Errorf("a broken lock path must not be reported as contention: %q", stderr)
	}
}

// TestLockFilePath pins the lock path rule: the config's ABSOLUTE path plus
// ".lock". Absolute because cron and CI invoke amele from arbitrary working
// directories, and "./agent.yaml" and "/etc/amele/agent.yaml" must resolve to
// the same lock.
func TestLockFilePath(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)

	relative, err := lockFilePath("agent.yaml")
	if err != nil {
		t.Fatalf("lockFilePath: %v", err)
	}
	absolute, err := lockFilePath(filepath.Join(dir, "agent.yaml"))
	if err != nil {
		t.Fatalf("lockFilePath: %v", err)
	}
	if relative != absolute {
		t.Errorf("relative %q and absolute %q must name the same lock file", relative, absolute)
	}
	if want := filepath.Join(dir, "agent.yaml.lock"); absolute != want {
		t.Errorf("lockFilePath = %q, want %q", absolute, want)
	}
}

func TestE2EStdinTemplate(t *testing.T) {
	// Capture what the provider actually receives: the template must be
	// rendered with both {{args}} and {{input}} end to end.
	srv, reqs := capturingServer(t, textBody("ok"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "prompt: \"Task: {{args}} Input: {{input}}\"\n")

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "triage"}, "log line 1")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if got := lastContent((*reqs)[0].Messages, "user"); got != "Task: triage Input: log line 1" {
		t.Errorf("rendered task: %q", got)
	}
}

// TestE2EStdinReadHonorsTimeout: an open pipe that never yields data must not
// hang the run past limits.timeout - the timeout is armed BEFORE stdin is
// read, and the overrun maps to the budget exit code (review finding P2-7,
// the remaining half of live-test finding B3).
func TestE2EStdinReadHonorsTimeout(t *testing.T) {
	srv := scriptedServer(t) // provider must never be reached
	cfgPath, _ := writeTestConfig(t, srv.URL, "prompt: \"I:{{input}}\"\nlimits:\n  timeout: 100ms\n")

	done := make(chan struct{})
	t.Cleanup(func() { close(done) })

	var out, errBuf bytes.Buffer
	code := run(context.Background(), []string{"run", cfgPath, "task"}, hangingReader{done}, &out, &errBuf, env(t))
	if code != ExitBudgetExceeded {
		t.Errorf("exit %d, stderr: %s", code, errBuf.String())
	}
}

// hangingReader stands in for an open pipe that never delivers data or EOF
// until the test finishes.
type hangingReader struct{ done <-chan struct{} }

func (h hangingReader) Read([]byte) (int, error) {
	<-h.done
	return 0, io.EOF
}

// TestReadPipedInputTruncated: the 10MB stdin cap must be visible to the
// model - silently dropping the tail of a piped log would make the agent
// reason over incomplete data without knowing it (review finding P2-6).
func TestReadPipedInputTruncated(t *testing.T) {
	big := strings.Repeat("x", maxStdinBytes+100)
	got, err := readPipedInput(context.Background(), strings.NewReader(big))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) > maxStdinBytes+128 {
		t.Errorf("input not capped: %d bytes", len(got))
	}
	if !strings.Contains(got[len(got)-100:], "truncated") {
		t.Error("truncation must leave a visible marker")
	}

	small, err := readPipedInput(context.Background(), strings.NewReader("tiny"))
	if err != nil || small != "tiny" {
		t.Errorf("small input must pass through untouched: %q, %v", small, err)
	}
}

// schemaBlock is the output block used by the structured-output e2e tests: a
// tiny object schema that a wrong-typed field violates unambiguously.
const schemaBlock = `output:
  schema:
    type: object
    additionalProperties: false
    required: ["score"]
    properties:
      score:
        type: integer
`

// TestE2ESchemaFirstTryValid: a schema-conforming first answer exits 0 and
// stdout carries exactly the canonical JSON - nothing else, so `amele run … |
// jq` works. The request must also carry the provider-native response_format.
func TestE2ESchemaFirstTryValid(t *testing.T) {
	srv, reqs := capturingServer(t, textBody(`{"score": 7}`))
	cfgPath, _ := writeTestConfig(t, srv.URL, schemaBlock)

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "score it"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "{\"score\": 7}\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if len(*reqs) != 1 {
		t.Fatalf("provider calls: %d", len(*reqs))
	}
	rf := (*reqs)[0].ResponseFormat
	if rf == nil || rf.Type != "json_schema" || rf.JSONSchema.Name != "amele_output" {
		t.Fatalf("response_format not sent: %+v", rf)
	}
	if !strings.Contains(string(rf.JSONSchema.Schema), `"score"`) {
		t.Errorf("response_format schema: %s", rf.JSONSchema.Schema)
	}
	// The provider accepted response_format, so no downgrade happened and the
	// warning must NOT print - it would train operators to ignore it.
	if strings.Contains(stderr, "did not enforce output.schema") {
		t.Errorf("downgrade warning printed on the native-accepted path: %q", stderr)
	}
}

// downgradeWarning is the exact stderr line the run command prints when the
// provider did not enforce output.schema natively.
const downgradeWarning = "warning: provider did not enforce output.schema natively; the validate+retry layer was the only enforcement"

// TestE2ESchemaDowngradeWarningOpenAIFallback: when an OpenAI-compatible
// provider 400s on response_format and the client falls back to a plain
// request, the run still succeeds via validate+retry - but the operator must
// be told, once, on stderr, that native enforcement was silently unavailable.
func TestE2ESchemaDowngradeWarningOpenAIFallback(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		if calls == 1 {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error": {"message": "response_format is not supported by this model"}}`))
			return
		}
		_, _ = w.Write([]byte(textBody(`{"score": 7}`)))
	}))
	t.Cleanup(srv.Close)
	cfgPath, _ := writeTestConfig(t, srv.URL, schemaBlock)

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "score it"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "{\"score\": 7}\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if got := strings.Count(stderr, downgradeWarning); got != 1 {
		t.Errorf("downgrade warning printed %d times (want 1), stderr: %q", got, stderr)
	}
}

// TestE2ENoSchemaDowngradeWarningAnthropicNative: the Messages API enforces
// json_schema natively (output_config.format), so a schema-carrying run that
// the provider accepts must NOT print the downgrade warning. This inverts the
// former TestE2ESchemaDowngradeWarningAnthropic, which pinned the behavior of
// the client back when it sent no schema at all.
func TestE2ENoSchemaDowngradeWarningAnthropicNative(t *testing.T) {
	srv, _ := anthropicServer(t, anthropicTextBody(`{"score": 7}`))
	cfgPath, _ := writeAnthropicConfig(t, srv.URL, schemaBlock)

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "score it"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "{\"score\": 7}\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if strings.Contains(stderr, downgradeWarning) {
		t.Errorf("natively enforced schema must not warn, stderr: %q", stderr)
	}
}

// TestE2ESchemaDowngradeWarningAnthropicFallback: an endpoint that rejects
// output_config (an Anthropic-compatible gateway that never implemented it)
// makes the client repeat the call without the field. The run still succeeds
// via validate+retry - but the operator must be told, once, on stderr, that
// native enforcement was unavailable.
func TestE2ESchemaDowngradeWarningAnthropicFallback(t *testing.T) {
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		if calls == 1 {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"type":"error","error":{"message":"output_config: Extra inputs are not permitted"}}`))
			return
		}
		_, _ = w.Write([]byte(anthropicTextBody(`{"score": 7}`)))
	}))
	t.Cleanup(srv.Close)
	cfgPath, _ := writeAnthropicConfig(t, srv.URL, schemaBlock)

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "score it"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "{\"score\": 7}\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if got := strings.Count(stderr, downgradeWarning); got != 1 {
		t.Errorf("downgrade warning printed %d times (want 1), stderr: %q", got, stderr)
	}
}

// TestE2ENoDowngradeWarningWithoutSchema: without output.schema there is no
// contract to warn about - a plain anthropic run must not print the warning.
func TestE2ENoDowngradeWarningWithoutSchema(t *testing.T) {
	srv, _ := anthropicServer(t, anthropicTextBody("plain answer"))
	cfgPath, _ := writeAnthropicConfig(t, srv.URL, "")

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "say something"}, "")
	if code != ExitOK || stdout != "plain answer\n" {
		t.Fatalf("code=%d stdout=%q stderr=%s", code, stdout, stderr)
	}
	if strings.Contains(stderr, "did not enforce output.schema") {
		t.Errorf("warning printed without a schema: %q", stderr)
	}
}

// TestE2ESchemaFencedAnswerIsUnfenced pins the canonical-via-closure decision:
// Result.FinalText is the raw model text, so a fenced reply would put ``` on
// stdout. Schema mode must print the extracted JSON instead.
func TestE2ESchemaFencedAnswerIsUnfenced(t *testing.T) {
	srv := scriptedServer(t, textBody("Here you go:\n```json\n{\"score\": 3}\n```\n"))
	cfgPath, _ := writeTestConfig(t, srv.URL, schemaBlock)

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "score it"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "{\"score\": 3}\n" {
		t.Errorf("stdout must be the unfenced canonical JSON, got %q", stdout)
	}
}

// TestE2ESchemaRetryThenValid: a schema violation buys a feedback retry, and
// the feedback must reach the provider as a user message so the model can
// actually repair its answer.
func TestE2ESchemaRetryThenValid(t *testing.T) {
	srv, reqs := capturingServer(t,
		textBody(`{"score": "high"}`),
		textBody(`{"score": 9}`),
	)
	cfgPath, _ := writeTestConfig(t, srv.URL, schemaBlock)

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "score it"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "{\"score\": 9}\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if len(*reqs) != 2 {
		t.Fatalf("provider calls: %d, want 2", len(*reqs))
	}
	second := (*reqs)[1].Messages
	last := second[len(second)-1]
	if last.Role != "user" || !strings.Contains(last.Content, "rejected") {
		t.Errorf("retry must carry validator feedback as a user message, got %+v", last)
	}
	// The rejected answer stays in the history so the model can repair it.
	if len(second) < 2 || !strings.Contains(second[len(second)-2].Content, `"high"`) {
		t.Errorf("rejected answer missing from retry history: %+v", second)
	}
}

// TestE2ESchemaRetriesExhausted: when the model never satisfies the schema the
// run fails with the frozen exit 6 and writes NOTHING to stdout - a pipeline
// must never see partial or invalid JSON.
func TestE2ESchemaRetriesExhausted(t *testing.T) {
	// max_schema_retries: 1 → one feedback retry, then the second rejection
	// is fatal: exactly two provider calls.
	srv := scriptedServer(t,
		textBody(`{"score": "high"}`),
		textBody(`{"score": "still high"}`),
	)
	cfgPath, _ := writeTestConfig(t, srv.URL, schemaBlock+"  max_schema_retries: 1\n")

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "score it"}, "")
	if code != ExitSchemaUnmet {
		t.Fatalf("exit %d (want %d), stderr: %s", code, ExitSchemaUnmet, stderr)
	}
	if stdout != "" {
		t.Errorf("schema failure must not write stdout: %q", stdout)
	}
}

// TestE2ESchemaDefaultRetries pins the default retry budget applied by cmd
// (config leaves 0 = unset): 2 retries, so three provider calls happen before
// exit 6. scriptedServer fails the test on a fourth call.
func TestE2ESchemaDefaultRetries(t *testing.T) {
	srv := scriptedServer(t,
		textBody(`{"score": "a"}`),
		textBody(`{"score": "b"}`),
		textBody(`{"score": "c"}`),
	)
	cfgPath, _ := writeTestConfig(t, srv.URL, schemaBlock)

	code, stdout, _ := execCLI(t, []string{"run", cfgPath, "score it"}, "")
	if code != ExitSchemaUnmet {
		t.Fatalf("exit %d, want %d", code, ExitSchemaUnmet)
	}
	if stdout != "" {
		t.Errorf("stdout: %q", stdout)
	}
}

// TestE2EInvalidSchemaIsConfigError: a schema that cannot compile is the
// user's config being wrong, so it must fail before any provider call with
// exit 2 - not mid-run.
func TestE2EInvalidSchemaIsConfigError(t *testing.T) {
	srv := scriptedServer(t) // any provider call fails the test
	cfgPath, _ := writeTestConfig(t, srv.URL, "output:\n  schema:\n    type: 3.14\n")

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "score it"}, "")
	if code != ExitConfigError {
		t.Fatalf("exit %d (want %d), stderr: %s", code, ExitConfigError, stderr)
	}
	if stdout != "" {
		t.Errorf("stdout: %q", stdout)
	}
	if !strings.Contains(stderr, "schema") {
		t.Errorf("stderr must name the schema problem: %q", stderr)
	}
}

// TestValidateRejectsUncompilableSchema: `validate` is the "will this config
// work?" command, so it must catch a broken output.schema too - otherwise the
// first report of the problem is a failed run in cron.
func TestValidateRejectsUncompilableSchema(t *testing.T) {
	srv := scriptedServer(t)
	cfgPath, _ := writeTestConfig(t, srv.URL, "output:\n  schema:\n    type: 3.14\n")

	code, _, stderr := execCLI(t, []string{"validate", cfgPath}, "")
	if code != ExitConfigError || !strings.Contains(stderr, "schema") {
		t.Errorf("code=%d stderr=%q", code, stderr)
	}

	// A valid schema still validates clean.
	okPath, _ := writeTestConfig(t, srv.URL, schemaBlock)
	code, stdout, stderr := execCLI(t, []string{"validate", okPath}, "")
	if code != ExitOK || !strings.Contains(stdout, "OK") {
		t.Errorf("code=%d stdout=%q stderr=%q", code, stdout, stderr)
	}
}

func TestE2EExitCodes(t *testing.T) {
	t.Run("config error", func(t *testing.T) {
		code, _, stderr := execCLI(t, []string{"run", "/nonexistent.yaml", "task"}, "")
		if code != ExitConfigError {
			t.Errorf("exit %d, stderr: %s", code, stderr)
		}
	})

	t.Run("validation error", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "bad.yaml")
		if err := os.WriteFile(path, []byte("provider: {base_url: 'https://x/v1'}\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		code, _, stderr := execCLI(t, []string{"run", path, "task"}, "")
		if code != ExitConfigError || !strings.Contains(stderr, "model is required") {
			t.Errorf("exit %d, stderr: %s", code, stderr)
		}
	})

	t.Run("budget exceeded", func(t *testing.T) {
		srv := scriptedServer(t,
			toolCallBody("fs_list", `{}`),
			toolCallBody("fs_list", `{}`),
		)
		cfgPath, _ := writeTestConfig(t, srv.URL, "limits:\n  max_turns: 2\n")
		code, stdout, _ := execCLI(t, []string{"run", cfgPath, "loop forever"}, "")
		if code != ExitBudgetExceeded {
			t.Errorf("exit %d", code)
		}
		if stdout != "" {
			t.Errorf("failed run must not write stdout: %q", stdout)
		}
	})

	t.Run("provider error", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusUnauthorized)
		}))
		defer srv.Close()
		cfgPath, _ := writeTestConfig(t, srv.URL, "")
		code, _, _ := execCLI(t, []string{"run", cfgPath, "task"}, "")
		if code != ExitProviderError {
			t.Errorf("exit %d", code)
		}
	})

	t.Run("no task", func(t *testing.T) {
		srv := scriptedServer(t)
		cfgPath, _ := writeTestConfig(t, srv.URL, "")
		code, _, stderr := execCLI(t, []string{"run", cfgPath}, "")
		if code != ExitConfigError || !strings.Contains(stderr, "no task") {
			t.Errorf("exit %d, stderr: %s", code, stderr)
		}
	})
}

func TestModelOverride(t *testing.T) {
	srv, reqs := capturingServer(t, textBody("ok"))

	// The YAML has no model at all: --model alone must satisfy validation.
	dir := t.TempDir()
	path := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(path, []byte(fmt.Sprintf("provider:\n  base_url: %s/v1\n", srv.URL)), 0o600); err != nil {
		t.Fatal(err)
	}

	code, _, stderr := execCLI(t, []string{"run", path, "--model", "override-model", "task"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if got := (*reqs)[0].Model; got != "override-model" {
		t.Errorf("model sent: %q", got)
	}
}

func TestValidateCommand(t *testing.T) {
	srv := scriptedServer(t)
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	code, stdout, _ := execCLI(t, []string{"validate", cfgPath}, "")
	if code != ExitOK || !strings.Contains(stdout, "OK") {
		t.Errorf("code=%d stdout=%q", code, stdout)
	}

	code, _, stderr := execCLI(t, []string{"validate", "/nope.yaml"}, "")
	if code != ExitConfigError || stderr == "" {
		t.Errorf("code=%d", code)
	}
}

// TestValidateDirectoryArg: `amele validate <dir>` is shorthand for
// <dir>/agent.yaml - the pack UX (`git clone … && amele run .`).
func TestValidateDirectoryArg(t *testing.T) {
	srv := scriptedServer(t)
	_, dir := writeTestConfig(t, srv.URL, "")

	code, stdout, stderr := execCLI(t, []string{"validate", dir}, "")
	if code != ExitOK {
		t.Fatalf("exit = %d, stderr: %s", code, stderr)
	}
	if !strings.Contains(stdout, "OK") {
		t.Errorf("stdout = %q, want it to contain OK", stdout)
	}
}

// TestDirectoryArgWithoutAgentYAML: a directory without agent.yaml is a
// config error (exit 2) with a message naming both the file and the dir.
func TestDirectoryArgWithoutAgentYAML(t *testing.T) {
	dir := t.TempDir()
	code, _, stderr := execCLI(t, []string{"validate", dir}, "")
	if code != ExitConfigError {
		t.Fatalf("exit = %d, want %d", code, ExitConfigError)
	}
	if !strings.Contains(stderr, "no agent.yaml in") {
		t.Errorf("stderr %q missing 'no agent.yaml in'", stderr)
	}
}

// TestDirectoryArgSharesLockWithFileArg: `run pack/` and `run pack/agent.yaml`
// must contend on the SAME lock file, or `lock: true` single-flight is
// defeated by a spelling difference.
func TestDirectoryArgSharesLockWithFileArg(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte("model: m\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	pd, ok := parseAgentArgs("run", usageRun, []string{dir}, io.Discard)
	if !ok {
		t.Fatal("parseAgentArgs(dir) failed")
	}
	pf, ok := parseAgentArgs("run", usageRun, []string{filepath.Join(dir, "agent.yaml")}, io.Discard)
	if !ok {
		t.Fatal("parseAgentArgs(file) failed")
	}
	ld, err := lockFilePath(pd.configPath)
	if err != nil {
		t.Fatal(err)
	}
	lf, err := lockFilePath(pf.configPath)
	if err != nil {
		t.Fatal(err)
	}
	if ld != lf {
		t.Errorf("lock paths differ: dir arg %q vs file arg %q", ld, lf)
	}
}

// update refreshes golden files: go test ./cmd/amele -update
var update = flag.Bool("update", false, "rewrite golden files")

// TestValidateGolden pins the full human-readable validate output - the
// docs/engineering.md §6 golden requirement for validate. Substring checks cannot catch
// accidental wording/format drift in what is effectively a UI surface.
func TestValidateGolden(t *testing.T) {
	// Every violation here produces a deterministic message (no absolute
	// paths), so the output is byte-stable across machines. The two prompt
	// fields are both set on purpose: that rule used to abort the load, so the
	// golden could never show it next to anything else - this fixture is what
	// keeps it inside the accumulated report.
	yaml := `provider:
  base_url: "not a url"
system_prompt: inline
system_prompt_file: absent.md
limits:
  max_turns: -1
  max_tokens: -5
tools:
  subprocess:
    - name: "bad name!"
      description: ""
      command: []
    - name: fs_read
      description: d
      command: ["true"]
`
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(cfgPath, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}

	code, _, stderr := execCLI(t, []string{"validate", cfgPath}, "")
	if code != ExitConfigError {
		t.Fatalf("exit %d", code)
	}

	goldenPath := filepath.Join("testdata", "golden", "validate-errors.txt")
	if *update {
		if err := os.MkdirAll(filepath.Dir(goldenPath), 0o750); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(goldenPath, []byte(stderr), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(goldenPath) //nolint:gosec // G304: fixed testdata path.
	if err != nil {
		t.Fatalf("reading golden (run with -update to create): %v", err)
	}
	if stderr != string(want) {
		t.Errorf("validate output differs from golden.\ngot:\n%s\nwant:\n%s", stderr, want)
	}
}

func TestUsageAndUnknownCommand(t *testing.T) {
	code, _, stderr := execCLI(t, nil, "")
	if code != ExitConfigError || !strings.Contains(stderr, "Usage") {
		t.Errorf("no-args: code=%d", code)
	}
	code, _, stderr = execCLI(t, []string{"frobnicate"}, "")
	if code != ExitConfigError || !strings.Contains(stderr, "unknown command") {
		t.Errorf("unknown: code=%d stderr=%q", code, stderr)
	}
	code, stdout, _ := execCLI(t, []string{"help"}, "")
	if code != ExitOK || !strings.Contains(stdout, "Usage") {
		t.Errorf("help: code=%d", code)
	}
}

// TestSchemaCommand pins the `amele schema` contract: the config JSON Schema
// on stdout - parseable, newline-terminated, no other output - and a usage
// error (exit 2) for any argument, because the command takes none.
func TestSchemaCommand(t *testing.T) {
	code, stdout, stderr := execCLI(t, []string{"schema"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if !strings.HasSuffix(stdout, "\n") {
		t.Error("schema output is not newline-terminated")
	}
	var doc map[string]any
	if err := json.Unmarshal([]byte(stdout), &doc); err != nil {
		t.Fatalf("stdout is not valid JSON: %v", err)
	}
	if !strings.Contains(stdout, "$schema") {
		t.Error("schema output does not declare $schema")
	}
	if stderr != "" {
		t.Errorf("unexpected stderr: %q", stderr)
	}

	code, stdout, stderr = execCLI(t, []string{"schema", "extra"}, "")
	if code != ExitConfigError || !strings.Contains(stderr, "usage") {
		t.Errorf("extra args: code=%d stderr=%q", code, stderr)
	}
	if stdout != "" {
		t.Errorf("extra args: unexpected stdout %q", stdout)
	}

	if !strings.Contains(usageText, "amele schema") {
		t.Error("usageText does not mention the schema command")
	}
}

// versionLineRe pins the `amele version` output shape (Task 1 spec): one
// line, "amele <version> (commit <commit>, built <date>, <go version>,
// <GOOS>/<GOARCH>)". Default build (no -ldflags) reports "dev"/"unknown".
var versionLineRe = regexp.MustCompile(`^amele dev \(commit unknown, built unknown, go\S.*, \w+/\w+\)\n$`)

// TestVersionCommand pins `amele version`, `amele --version` and `amele -V`
// as three spellings of the same one-line, exit-0, stdout-only output.
func TestVersionCommand(t *testing.T) {
	for _, args := range [][]string{{"version"}, {"--version"}, {"-V"}} {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			code, stdout, stderr := execCLI(t, args, "")
			if code != ExitOK {
				t.Fatalf("exit %d, stderr: %s", code, stderr)
			}
			if !versionLineRe.MatchString(stdout) {
				t.Errorf("stdout = %q, want match of %s", stdout, versionLineRe)
			}
			if n := strings.Count(stdout, "\n"); n != 1 {
				t.Errorf("stdout has %d newlines, want exactly 1: %q", n, stdout)
			}
			if stderr != "" {
				t.Errorf("unexpected stderr: %q", stderr)
			}
		})
	}
}

// TestVersionCommandExtraArgs pins `amele version <anything>` as a usage
// error: the command takes no arguments, so a typo must not be silently
// ignored.
func TestVersionCommandExtraArgs(t *testing.T) {
	code, stdout, stderr := execCLI(t, []string{"version", "extra-arg"}, "")
	if code != ExitConfigError {
		t.Errorf("exit %d, want %d", code, ExitConfigError)
	}
	if stdout != "" {
		t.Errorf("unexpected stdout: %q", stdout)
	}
	if !strings.Contains(stderr, "usage") {
		t.Errorf("stderr = %q, want a usage message", stderr)
	}
}

// helpCommands is every command name that must own a detailed help page. It
// mirrors the dispatch switch in run(), so a command added without a page
// fails here instead of shipping undocumented.
var helpCommands = []string{"run", "chat", "validate", "explain", "schema", "init", "version", "completion", "mcp", "help"}

// helpSections is the man-page skeleton every detailed page promises. Tests
// assert on the section headers rather than on whole-page golden text: the
// prose is meant to be edited, the structure is the contract with the reader.
var helpSections = []string{"SYNOPSIS", "DESCRIPTION", "FLAGS", "STDIN", "STDOUT", "STDERR", "EXIT CODES", "EXAMPLES"}

// TestHelpPageStructure pins the delivery of every detailed page: stdout only,
// exit 0, all sections present, the command's own name in the synopsis, and
// the -h/--help flag documented on the page that flag reaches.
func TestHelpPageStructure(t *testing.T) {
	for _, cmd := range helpCommands {
		t.Run(cmd, func(t *testing.T) {
			code, stdout, stderr := execCLI(t, []string{"help", cmd}, "")
			if code != ExitOK {
				t.Fatalf("exit %d, stderr: %s", code, stderr)
			}
			if stderr != "" {
				t.Errorf("unexpected stderr: %q", stderr)
			}
			for _, section := range helpSections {
				if !strings.Contains(stdout, section) {
					t.Errorf("page for %q has no %s section", cmd, section)
				}
			}
			if !strings.Contains(stdout, "amele "+cmd) {
				t.Errorf("page for %q never spells out `amele %s`", cmd, cmd)
			}
			if !strings.Contains(stdout, "-h, --help") {
				t.Errorf("page for %q does not document -h, --help", cmd)
			}
			if !strings.HasSuffix(stdout, "\n") {
				t.Errorf("page for %q is not newline-terminated", cmd)
			}
		})
	}
}

// TestHelpPageContent pins the facts each page must carry - every flag it
// accepts, the stream semantics that differ per command, and real runnable
// examples. The strings are the ones docs/contracts/cli.md states, so a
// contract edit that never reaches the help output fails here.
func TestHelpPageContent(t *testing.T) {
	tests := []struct {
		cmd  string
		want []string
	}{
		{"run", []string{
			"--model MODEL", "-h, --help", "-q, --quiet", "-v, --verbose",
			"--set KEY=VALUE", "-w, --workspace DIR", "last entry for a key wins",
			"amele: turn 3: model requested fs_read",
			"10 MB", "[input truncated at 10MB by amele]",
			"output.schema", "lock: true", "exits 7",
			"amele run agent.yaml < app.log",
			"| jq .score",
		}},
		{"chat", []string{
			"--model MODEL", "-h, --help", "-q, --quiet", "-v, --verbose",
			"--set KEY=VALUE", "-w, --workspace DIR",
			"1 MB", "Ctrl-D", "output.schema is not enforced in chat",
			"amele chat agent.yaml", "amele chat agent.yaml < script.txt",
		}},
		{"validate", []string{
			"<config.yaml>: OK", "No network, no tokens, no session file",
			"amele validate agent.yaml", "--set KEY=VALUE", "-w, --workspace DIR",
		}},
		{"explain", []string{
			"tool registry", "WARNINGS", "amele explain agent.yaml",
			"--set KEY=VALUE", "(overridden via --set)",
		}},
		{"schema", []string{
			"config.schema.json", "amele schema > config.schema.json",
		}},
		{"init", []string{
			"agent.yaml", "never overwritten", "AMELE_API_KEY",
		}},
		{"version", []string{
			"commit", "built", "amele --version", "amele -V",
		}},
		{"completion", []string{
			"bash|zsh|fish", "bash", "zsh", "fish",
			"amele completion bash > /etc/bash_completion.d/amele",
		}},
		{"help", []string{
			"amele help run", "amele run --help",
		}},
	}
	for _, tt := range tests {
		t.Run(tt.cmd, func(t *testing.T) {
			_, stdout, _ := execCLI(t, []string{"help", tt.cmd}, "")
			for _, want := range tt.want {
				if !strings.Contains(stdout, want) {
					t.Errorf("page for %q does not mention %q", tt.cmd, want)
				}
			}
		})
	}
}

// TestHelpFlagPerCommand pins `amele <cmd> -h` / `amele <cmd> --help` as the
// very same page `amele help <cmd>` prints. Additive: -h was a usage error
// before this, never a valid config path.
func TestHelpFlagPerCommand(t *testing.T) {
	for _, cmd := range helpCommands {
		for _, flagName := range []string{"-h", "--help"} {
			t.Run(cmd+" "+flagName, func(t *testing.T) {
				_, want, _ := execCLI(t, []string{"help", cmd}, "")
				code, stdout, stderr := execCLI(t, []string{cmd, flagName}, "")
				if code != ExitOK {
					t.Fatalf("exit %d, stderr: %s", code, stderr)
				}
				if stdout != want {
					t.Errorf("`amele %s %s` printed a different page than `amele help %s`", cmd, flagName, cmd)
				}
				if stderr != "" {
					t.Errorf("unexpected stderr: %q", stderr)
				}
			})
		}
	}
}

// TestRunHelpPageStdinRule pins the run page against the FROZEN stdin rule
// (docs/contracts/cli.md): stdin is read only when the prompt template asks
// for {{input}} or there is no task text at all. The page once described (and
// gave an example of) task text and a redirect being merged, which the binary
// never did - the log file was silently dropped. Every fact below is a fact
// buildTask actually implements.
func TestRunHelpPageStdinRule(t *testing.T) {
	_, stdout, _ := execCLI(t, []string{"help", "run"}, "")

	for _, want := range []string{
		"stdin is never appended to task text",
		"amele run agent.yaml < app.log",
	} {
		if !strings.Contains(stdout, want) {
			t.Errorf("run page does not state %q", want)
		}
	}
	// The exact wording of the removed claim: a paraphrase would be just as
	// wrong, but pinning the original sentence keeps the regression named.
	if strings.Contains(stdout, "the piped input follows it") {
		t.Error("run page still claims piped input is appended to task text")
	}
}

// TestHelpFlagArity pins the help flag on the fixed-arity commands as
// something honored ONLY when it is the command's sole argument.
//
// CONTRACT: docs/contracts/cli.md - "every usage error (wrong argument count,
// bad flag) is exit 2". A -h scanned out of any position let
// `amele validate a.yaml b.yaml -h` exit 0, turning a wrong argument count
// into success and hiding the mistake from any script checking $?.
func TestHelpFlagArity(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want int
	}{
		{"validate two paths plus -h", []string{"validate", "a.yaml", "b.yaml", "-h"}, ExitConfigError},
		// The help flag in the config-path slot is the same rule seen from the
		// other side: sole argument or usage error, never a page and an exit 0
		// for an invocation that also named a file.
		{"validate -h plus a path", []string{"validate", "-h", "extra.yaml"}, ExitConfigError},
		{"validate --help plus a path", []string{"validate", "--help", "extra.yaml"}, ExitConfigError},
		{"explain -h plus a path", []string{"explain", "-h", "extra.yaml"}, ExitConfigError},
		{"validate path plus --help", []string{"validate", "a.yaml", "--help"}, ExitConfigError},
		{"explain two paths plus -h", []string{"explain", "a.yaml", "b.yaml", "-h"}, ExitConfigError},
		{"version arg plus -h", []string{"version", "foo", "-h"}, ExitConfigError},
		{"schema arg plus -h", []string{"schema", "foo", "-h"}, ExitConfigError},
		{"init path plus -h", []string{"init", "existing.yaml", "-h"}, ExitConfigError},
		{"completion shell plus -h", []string{"completion", "bash", "-h"}, ExitConfigError},
		{"validate -h alone", []string{"validate", "-h"}, ExitOK},
		{"explain --help alone", []string{"explain", "--help"}, ExitOK},
		{"schema -h alone", []string{"schema", "-h"}, ExitOK},
		{"version -h alone", []string{"version", "-h"}, ExitOK},
		{"init -h alone", []string{"init", "-h"}, ExitOK},
		{"completion -h alone", []string{"completion", "-h"}, ExitOK},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, stdout, stderr := execCLI(t, tt.args, "")
			if code != tt.want {
				t.Errorf("exit %d, want %d (stderr: %s)", code, tt.want, stderr)
			}
			switch tt.want {
			case ExitOK:
				if !strings.Contains(stdout, "SYNOPSIS") {
					t.Errorf("stdout is not a help page: %q", stdout)
				}
			default:
				if stdout != "" {
					t.Errorf("usage error wrote to stdout: %q", stdout)
				}
				if !strings.Contains(stderr, "usage") {
					t.Errorf("stderr = %q, want a usage message", stderr)
				}
			}
		})
	}
}

// synopsisLine returns the single SYNOPSIS line of a command's help page.
func synopsisLine(t *testing.T, cmd string) string {
	t.Helper()
	code, page, stderr := execCLI(t, []string{"help", cmd}, "")
	if code != ExitOK {
		t.Fatalf("help %s: exit %d, stderr: %s", cmd, code, stderr)
	}
	lines := strings.Split(page, "\n")
	for i, line := range lines {
		if strings.TrimSpace(line) == "SYNOPSIS" && i+1 < len(lines) {
			return strings.TrimSpace(lines[i+1])
		}
	}
	t.Fatalf("help %s has no SYNOPSIS section", cmd)
	return ""
}

// TestAgentUsageMatchesSynopsis: the one-line usage printed on a wrong
// argument count is the only spelling of the invocation an operator sees at
// that moment, and it drifted once already (it kept advertising just
// [--model MODEL] after -q/-v shipped). Pinning it to the help page's own
// SYNOPSIS line makes the two impossible to change apart.
func TestAgentUsageMatchesSynopsis(t *testing.T) {
	for _, cmd := range []string{"run", "chat"} {
		t.Run(cmd, func(t *testing.T) {
			want := "usage: " + synopsisLine(t, cmd)
			code, stdout, stderr := execCLI(t, []string{cmd}, "")
			if code != ExitConfigError {
				t.Errorf("exit %d, want %d", code, ExitConfigError)
			}
			if stdout != "" {
				t.Errorf("usage error wrote to stdout: %q", stdout)
			}
			if !strings.Contains(stderr, want) {
				t.Errorf("stderr = %q, want it to contain %q", stderr, want)
			}
		})
	}
}

// TestHelpFlagAfterConfigPath pins that the help flag is still recognized in
// the flag position of run/chat - after the config path, where --model lives.
// The config is never loaded, so a nonexistent path still yields help, not
// exit 2.
func TestHelpFlagAfterConfigPath(t *testing.T) {
	for _, cmd := range []string{"run", "chat"} {
		t.Run(cmd, func(t *testing.T) {
			_, want, _ := execCLI(t, []string{"help", cmd}, "")
			code, stdout, stderr := execCLI(t, []string{cmd, "no-such-config.yaml", "--help"}, "")
			if code != ExitOK {
				t.Fatalf("exit %d, stderr: %s", code, stderr)
			}
			if stdout != want {
				t.Errorf("`amele %s <cfg> --help` printed a different page than `amele help %s`", cmd, cmd)
			}
		})
	}
}

// TestHelpFlagAfterTaskTextIsTaskText pins the FROZEN flag-stop rule: run
// stops parsing flags at the first non-flag argument, so a -h that follows the
// task text is task text. The help system must not reach past that boundary,
// or `amele run cfg.yaml "explain the -h flag"` would silently print help
// instead of doing the job.
func TestHelpFlagAfterTaskTextIsTaskText(t *testing.T) {
	srv, reqs := capturingServer(t, textBody("ok"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "explain", "-h"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "ok\n" {
		t.Errorf("stdout = %q, want the model answer", stdout)
	}
	if got := lastContent((*reqs)[0].Messages, "user"); !strings.Contains(got, "-h") {
		t.Errorf("user message = %q, want the -h to survive as task text", got)
	}
}

// TestHelpForAliases pins that the alternate spellings of a command resolve to
// that command's page, so `amele help --version` is not an unknown command.
func TestHelpForAliases(t *testing.T) {
	for alias, canonical := range map[string]string{"--version": "version", "-V": "version", "-h": "help", "--help": "help"} {
		t.Run(alias, func(t *testing.T) {
			_, want, _ := execCLI(t, []string{"help", canonical}, "")
			code, stdout, _ := execCLI(t, []string{"help", alias}, "")
			if code != ExitOK || stdout != want {
				t.Errorf("`amele help %s`: code=%d, page differs from %q", alias, code, canonical)
			}
		})
	}
}

// TestHelpUnknownCommand pins `amele help bogus` as a usage error carrying the
// short usage, so a mistyped command still teaches the reader the real ones.
func TestHelpUnknownCommand(t *testing.T) {
	code, stdout, stderr := execCLI(t, []string{"help", "bogus"}, "")
	if code != ExitConfigError {
		t.Errorf("exit %d, want %d", code, ExitConfigError)
	}
	if stdout != "" {
		t.Errorf("unexpected stdout: %q", stdout)
	}
	if !strings.Contains(stderr, "unknown command") || !strings.Contains(stderr, "Usage") {
		t.Errorf("stderr = %q, want an unknown-command error plus the usage", stderr)
	}
}

// TestHelpTooManyArguments pins that help names exactly one command: a second
// argument is a usage error rather than being silently ignored (same stance as
// `amele schema anything`).
func TestHelpTooManyArguments(t *testing.T) {
	code, stdout, stderr := execCLI(t, []string{"help", "run", "extra"}, "")
	if code != ExitConfigError {
		t.Errorf("exit %d, want %d", code, ExitConfigError)
	}
	if stdout != "" {
		t.Errorf("unexpected stdout: %q", stdout)
	}
	if !strings.Contains(stderr, "usage: amele help") {
		t.Errorf("stderr = %q, want the help usage line", stderr)
	}
}

// TestShortUsageText pins the short usage as the map of the CLI: every command
// with a one-line description, the pointer at the detailed pages, the full
// exit-code table, and the project one-liner. `amele help` keeps its frozen
// stdout/exit-0 delivery.
func TestShortUsageText(t *testing.T) {
	code, stdout, stderr := execCLI(t, []string{"help"}, "")
	if code != ExitOK || stderr != "" {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	want := []string{
		"Usage:",
		"Commands:",
		"Run 'amele help <command>' for details",
		// The override and verbosity flags belong in the map of the CLI, not
		// only in the detailed pages: they are the ones an operator reaches
		// for first.
		"amele run <config.yaml|dir> [--model MODEL] [--set key=value] [-w DIR] [-q|-v] [task...]",
		"amele chat <config.yaml|dir> [--model MODEL] [--set key=value] [-w DIR] [-q|-v]",
		"amele validate <config.yaml|dir> [--set key=value] [-w DIR]",
		"amele explain <config.yaml|dir> [--set key=value] [-w DIR]",
		"one static Go binary plus one YAML file is a working AI agent.",
		"0 success", "1 task failed", "2 config error", "3 budget exceeded",
		"4 permission denied", "5 provider error", "6 output schema unmet",
		"7 run lock held",
	}
	for _, cmd := range helpCommands {
		want = append(want, "amele "+cmd)
	}
	for _, w := range want {
		if !strings.Contains(stdout, w) {
			t.Errorf("short usage does not mention %q", w)
		}
	}
}

func TestBuildTask(t *testing.T) {
	tests := []struct {
		name   string
		prompt string
		args   string
		stdin  string
		want   string
	}{
		{"args only", "", "do it", "", "do it"},
		{"stdin only when no args", "", "", "piped", "piped"},
		// CONTRACT: with task args and no {{input}} template, stdin is
		// ignored (and never read) so an open pipe cannot hang the run.
		{"args win over stdin", "", "do it", "piped", "do it"},
		{"template", "T:{{args}}|I:{{input}}", "a", "b", "T:a|I:b"},
		{"template empty input", "T:{{args}}|I:{{input}}", "a", "", "T:a|I:"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.Config{Prompt: tt.prompt}
			got, err := buildTask(context.Background(), cfg, tt.args, strings.NewReader(tt.stdin))
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Errorf("got %q want %q", got, tt.want)
			}
		})
	}

	t.Run("empty everything errors", func(t *testing.T) {
		if _, err := buildTask(context.Background(), &config.Config{}, "", strings.NewReader("")); err == nil {
			t.Error("expected error")
		}
	})

	// TestBuildTask/"content-free prompt errors" is the regression for
	// live-test finding B-A03: a template that renders to nothing (or to
	// nothing but whitespace) used to buy a provider call that asked the model
	// nothing. Every such rendering must be refused BEFORE the provider is
	// contacted, whatever produced it.
	t.Run("content-free prompt errors", func(t *testing.T) {
		contentFree := []struct {
			name   string
			prompt string
			args   string
			stdin  string
		}{
			{"input-only template with empty stdin", "{{input}}", "", ""},
			{"args-only template with no task text", "{{args}}", "", ""},
			{"template renders to whitespace", "  {{args}}\n{{input}}\t", "", ""},
			{"whitespace task text", "", "   \t", ""},
			{"whitespace stdin", "", "", "  \n "},
		}
		for _, tt := range contentFree {
			t.Run(tt.name, func(t *testing.T) {
				cfg := &config.Config{Prompt: tt.prompt}
				got, err := buildTask(context.Background(), cfg, tt.args, strings.NewReader(tt.stdin))
				if err == nil {
					t.Fatalf("expected a refusal, got task %q", got)
				}
			})
		}
	})

	// The mirror image: whitespace AROUND real content is content. A template
	// whose fixed text carries the instruction is the whole point of prompt
	// templates, and it must survive an empty {{args}}/{{input}}.
	t.Run("whitespace-bearing prompts still run", func(t *testing.T) {
		bearing := []struct {
			name   string
			prompt string
			args   string
			stdin  string
			want   string
		}{
			{"fixed template text is content", "\nSummarize:\n{{input}}\n", "", "", "\nSummarize:\n\n"},
			{"padded task text", "", "  do it  ", "", "  do it  "},
			{"padded stdin", "", "", " piped\n", " piped\n"},
		}
		for _, tt := range bearing {
			t.Run(tt.name, func(t *testing.T) {
				cfg := &config.Config{Prompt: tt.prompt}
				got, err := buildTask(context.Background(), cfg, tt.args, strings.NewReader(tt.stdin))
				if err != nil {
					t.Fatalf("refused a prompt that carries content: %v", err)
				}
				if got != tt.want {
					t.Errorf("got %q want %q", got, tt.want)
				}
			})
		}
	})

	// TestBuildTaskDoesNotReadStdinWhenUnneeded is the regression for the
	// live-test hang (B3): with task text and no {{input}} template, stdin
	// must not be read at all - otherwise an open, dataless pipe hangs the
	// process before any timeout is armed. blockingReader fails the test
	// loudly (instead of hanging) if stdin is touched.
	t.Run("task arg does not read stdin", func(t *testing.T) {
		got, err := buildTask(context.Background(), &config.Config{}, "do it", blockingReader{t})
		if err != nil {
			t.Fatal(err)
		}
		if got != "do it" {
			t.Errorf("got %q", got)
		}
	})

	t.Run("template with input still reads stdin", func(t *testing.T) {
		cfg := &config.Config{Prompt: "T:{{args}} I:{{input}}"}
		got, err := buildTask(context.Background(), cfg, "a", strings.NewReader("piped"))
		if err != nil || got != "T:a I:piped" {
			t.Errorf("got %q, %v", got, err)
		}
	})
}

// blockingReader stands in for an open pipe that never yields data or EOF.
// Any Read call means buildTask touched stdin when it should not have.
type blockingReader struct{ t *testing.T }

func (b blockingReader) Read([]byte) (int, error) {
	b.t.Error("stdin was read when it should not have been")
	return 0, io.EOF
}

// TestE2EPermissionDenyContinues: a tool denied by policy must come back to
// the model as a "permission denied" tool result - the run keeps going with
// the tools it still has and ends with a normal answer.
func TestE2EPermissionDenyContinues(t *testing.T) {
	srv, reqs := capturingServer(t,
		toolCallBody("fs_read", `{"path":"data.txt"}`),
		textBody("could not read it"),
	)
	cfgPath, dir := writeTestConfig(t, srv.URL,
		"session_dir: sessions\npermissions:\n  tools:\n    fs_read: deny\n")
	if err := os.WriteFile(filepath.Join(dir, "data.txt"), []byte("secret"), 0o600); err != nil {
		t.Fatal(err)
	}

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "read data.txt"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "could not read it\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if len(*reqs) != 2 {
		t.Fatalf("provider calls: %d", len(*reqs))
	}
	toolResult := lastContent((*reqs)[1].Messages, "tool")
	if !strings.Contains(toolResult, "permission denied") {
		t.Errorf("tool result should report the denial, got %q", toolResult)
	}
	// The denied file must never have been read.
	if strings.Contains(toolResult, "secret") {
		t.Error("denied tool still produced output")
	}
	files, err := filepath.Glob(filepath.Join(dir, "sessions", "*.jsonl"))
	if err != nil || len(files) != 1 {
		t.Fatalf("session files: %v, %v", files, err)
	}
	data, _ := os.ReadFile(files[0])
	if !strings.Contains(string(data), "permission denied") {
		t.Error("session log should record the denial")
	}
}

// TestE2EAskWithoutTTYAutoDenies is the headless fail-safe (docs/engineering.md §5.5):
// stdin here is not a terminal, so an `ask` policy degrades to deny, the note
// explaining why lands on stderr, and the run continues.
func TestE2EAskWithoutTTYAutoDenies(t *testing.T) {
	srv, reqs := capturingServer(t,
		toolCallBody("fs_write", `{"path":"out.txt","content":"x"}`),
		textBody("did not write"),
	)
	cfgPath, dir := writeTestConfig(t, srv.URL, "permissions:\n  tools:\n    fs_write: ask\n")

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "write out.txt"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "did not write\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if !strings.Contains(stderr, "fs_write") || !strings.Contains(stderr, "TTY") {
		t.Errorf("stderr must explain the auto-deny: %q", stderr)
	}
	if _, err := os.Stat(filepath.Join(dir, "out.txt")); !os.IsNotExist(err) {
		t.Error("the denied write must not have happened")
	}
	if toolResult := lastContent((*reqs)[1].Messages, "tool"); !strings.Contains(toolResult, "permission denied") {
		t.Errorf("tool result: %q", toolResult)
	}
}

// TestE2EInvalidPermissionPolicyIsConfigError: a typo'd policy must fail at
// validation (exit 2), before a single token is spent.
func TestE2EInvalidPermissionPolicyIsConfigError(t *testing.T) {
	srv := scriptedServer(t) // provider must never be reached
	cfgPath, _ := writeTestConfig(t, srv.URL, "permissions:\n  default: auto_approve\n")

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "task"}, "")
	if code != ExitConfigError || !strings.Contains(stderr, "permissions.default") {
		t.Errorf("exit %d, stderr: %s", code, stderr)
	}
}

// TestIsTerminal: only a character device counts as a human at the keyboard.
// Everything else - a pipe, a file, a plain io.Reader - is headless, which is
// the direction that fails safe (ask → deny).
func TestIsTerminal(t *testing.T) {
	if isTerminal(strings.NewReader("x")) {
		t.Error("a plain io.Reader is not a terminal")
	}

	f, err := os.CreateTemp(t.TempDir(), "stdin")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	if isTerminal(f) {
		t.Error("a regular file is not a terminal")
	}

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	defer func() { _ = w.Close() }()
	if isTerminal(r) {
		t.Error("a pipe is not a terminal")
	}

	// /dev/null is a character device like a TTY, but nobody is sitting behind
	// it: `cron` runs its jobs with stdin on /dev/null, so counting it as a
	// terminal made an `ask` policy prompt into the void and then log the EOF
	// as if a human had refused - instead of the no-TTY auto-deny the contract
	// promises (docs/engineering.md §5.5).
	devNull, err := os.Open(os.DevNull)
	if err != nil {
		t.Skipf("cannot open %s: %v", os.DevNull, err)
	}
	defer func() { _ = devNull.Close() }()
	if isTerminal(devNull) {
		t.Errorf("%s is not a terminal: no human can answer an approval on it", os.DevNull)
	}
}

// TestE2EAskWithDevNullStdinAutoDenies is the cron case of the headless
// fail-safe: stdin is /dev/null - a character device, but not a terminal - so
// an `ask` policy must auto-deny with the no-TTY reason rather than prompt and
// read the immediate EOF back as a human's refusal.
func TestE2EAskWithDevNullStdinAutoDenies(t *testing.T) {
	srv := scriptedServer(t,
		toolCallBody("fs_write", `{"path":"out.txt","content":"x"}`),
		textBody("did not write"),
	)
	cfgPath, dir := writeTestConfig(t, srv.URL, "permissions:\n  tools:\n    fs_write: ask\n")

	devNull, err := os.Open(os.DevNull)
	if err != nil {
		t.Skipf("cannot open %s: %v", os.DevNull, err)
	}
	defer func() { _ = devNull.Close() }()

	var stdout, stderr bytes.Buffer
	code := run(context.Background(), []string{"run", cfgPath, "write out.txt"}, devNull, &stdout, &stderr, env(t))
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "TTY") {
		t.Errorf("stderr must explain the auto-deny with the no-TTY reason: %q", stderr.String())
	}
	// The prompt is the proof of the bug: it must never have been asked.
	if strings.Contains(stderr.String(), "[y/N]") {
		t.Errorf("an approval was prompted with no terminal attached: %q", stderr.String())
	}
	if _, err := os.Stat(filepath.Join(dir, "out.txt")); !os.IsNotExist(err) {
		t.Error("the denied write must not have happened")
	}
}

// TestNewPrompter: only an explicit yes approves; everything else - including
// EOF (Ctrl-D) and an empty line - is a refusal.
func TestNewPrompter(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  bool
	}{
		{"y", "y\n", true},
		{"yes", "yes\n", true},
		{"uppercase Y", "Y\n", true},
		{"mixed case YeS", "YeS\n", true},
		{"padded", "  y  \n", true},
		{"n", "n\n", false},
		{"empty line", "\n", false},
		{"other word", "sure\n", false},
		{"eof", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var errBuf bytes.Buffer
			prompt := newPrompter(newLineReader(strings.NewReader(tt.input)), &errBuf)
			got, err := prompt("fs_write", `{"path":"a.txt"}`, "")
			if err != nil {
				t.Fatalf("prompt: %v", err)
			}
			if got != tt.want {
				t.Errorf("got %v want %v", got, tt.want)
			}
			// The question goes to stderr so stdout stays the answer channel.
			if !strings.Contains(errBuf.String(), "fs_write") {
				t.Errorf("question missing the tool name: %q", errBuf.String())
			}
			if !strings.Contains(errBuf.String(), "a.txt") {
				t.Errorf("question missing the arguments: %q", errBuf.String())
			}
		})
	}

	t.Run("consecutive prompts share the reader", func(t *testing.T) {
		var errBuf bytes.Buffer
		prompt := newPrompter(newLineReader(strings.NewReader("y\nn\n")), &errBuf)
		first, _ := prompt("a", "{}", "")
		second, _ := prompt("b", "{}", "")
		if !first || second {
			t.Errorf("answers = %v, %v; want true, false", first, second)
		}
	})

	t.Run("long arguments are clipped", func(t *testing.T) {
		var errBuf bytes.Buffer
		prompt := newPrompter(newLineReader(strings.NewReader("n\n")), &errBuf)
		if _, err := prompt("fs_write", strings.Repeat("z", 5000), ""); err != nil {
			t.Fatal(err)
		}
		if errBuf.Len() > 512 {
			t.Errorf("prompt not clipped: %d bytes", errBuf.Len())
		}
	})
}

// evilToolName is the terminal-spoofing attack from review: a prompt-injected
// model names its tool so that the escape sequence erases the real question
// and redraws a benign-looking one, tricking the operator into approving an
// fs_write while they believe they are allowing an fs_read.
const evilToolName = "fs_write\x1b[2K\ramele: allow tool fs_read with {\"path\":\"README.md\"}? [y/N] "

// hasControlBytes reports whether s contains any C0 control byte or DEL -
// exactly what a terminal interprets as a command rather than as text.
func hasControlBytes(s string) bool {
	for _, r := range s {
		if r < 0x20 || r == 0x7f {
			return true
		}
	}
	return false
}

// TestSafeForTerminal is the regression for the spoofing vector: everything
// derived from provider JSON must be stripped of control bytes and clipped
// before it reaches the operator's terminal.
func TestSafeForTerminal(t *testing.T) {
	got := safeForTerminal(evilToolName, maxToolName)
	if strings.ContainsRune(got, 0x1b) || strings.ContainsRune(got, '\r') {
		t.Errorf("escape sequence survived sanitization: %q", got)
	}
	if hasControlBytes(got) {
		t.Errorf("control bytes survived sanitization: %q", got)
	}
	if len(got) > maxToolName+len(clipMarker) {
		t.Errorf("name not clipped: %q", got)
	}

	if got := safeForTerminal("fs_read", maxToolName); got != "fs_read" {
		t.Errorf("an honest name must pass through untouched: %q", got)
	}
	if got := safeForTerminal("a\tb\nc", maxToolName); got != "abc" {
		t.Errorf("tabs and newlines must be dropped too: %q", got)
	}
	if got := safeForTerminal("a\x7fä", maxToolName); got != "aä" {
		t.Errorf("DEL must be dropped and non-ASCII kept: %q", got)
	}
	long := strings.Repeat("z", 5000)
	if got := safeForTerminal(long, maxPromptArgs); len(got) > maxPromptArgs+len(clipMarker) {
		t.Errorf("args not clipped: %d bytes", len(got))
	}
}

// TestSafeForTerminalStripsC1AndBidi covers the two escapes the C0-only strip
// let through. U+009B is the single-character CSI: a terminal in an 8-bit
// mode reads it exactly like "\x1b[", so it re-opens the question-spoofing
// vector without ever using an ESC byte. The bidi overrides and isolates
// reorder the rendered line, so a tool name can be made to READ as a
// different one than the model actually asked for - the same lie, told by
// layout instead of by escape sequence.
func TestSafeForTerminalStripsC1AndBidi(t *testing.T) {
	for _, tt := range []struct {
		name string
		in   string
		want string
	}{
		{"C1 CSI (the 8-bit ESC[)", "fs_write\u009b2K\rok", "fs_write2Kok"},
		{"C1 range ends", "a\u0080b\u009fc", "abc"},
		{"bidi override", "fs_read\u202etxt.exe", "fs_readtxt.exe"},
		{"bidi embedding", "a\u202ab\u202cc", "abc"},
		{"bidi isolate", "a\u2066b\u2069c", "abc"},
		{"honest non-ASCII text survives", "fs_read ölçüm", "fs_read ölçüm"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := safeForTerminal(tt.in, maxPromptArgs); got != tt.want {
				t.Errorf("safeForTerminal(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

// TestPrompterSanitizesUntrustedText: the rendered question must never carry
// provider-controlled control bytes, whether they arrive via the tool name or
// via the arguments.
func TestPrompterSanitizesUntrustedText(t *testing.T) {
	var errBuf bytes.Buffer
	prompt := newPrompter(newLineReader(strings.NewReader("n\n")), &errBuf)
	if _, err := prompt(evilToolName, "{\"path\":\"a.txt\x1b[2K\r\"}", ""); err != nil {
		t.Fatal(err)
	}
	if hasControlBytes(errBuf.String()) {
		t.Errorf("prompt line contains control bytes: %q", errBuf.String())
	}
	// A spoofed second question would show up as a second "[y/N]".
	if strings.Count(errBuf.String(), "[y/N]") != 1 {
		t.Errorf("prompt must render exactly one question: %q", errBuf.String())
	}
}

// TestE2EDenyNoteSanitizesToolName covers the second render path: perm runs
// BEFORE Registry.Get, so an unregistered (attacker-chosen) tool name reaches
// the stderr audit note. It must be sanitized there too.
func TestE2EDenyNoteSanitizesToolName(t *testing.T) {
	name, _ := json.Marshal(evilToolName)
	body := fmt.Sprintf(`{
		"choices": [{"message": {"role": "assistant", "content": "",
			"tool_calls": [{"id": "c1", "type": "function", "function": {"name": %s, "arguments": "{}"}}]},
			"finish_reason": "tool_calls"}],
		"usage": {"prompt_tokens": 1, "completion_tokens": 1}
	}`, name)
	srv := scriptedServer(t, body, textBody("done"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "permissions:\n  default: deny\n")

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "task"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	for _, line := range strings.Split(stderr, "\n") {
		if strings.Contains(line, "amele: tool ") && hasControlBytes(line) {
			t.Errorf("audit note contains control bytes: %q", line)
		}
	}
	if !strings.Contains(stderr, "denied by policy") {
		t.Errorf("stderr should carry the denial note: %q", stderr)
	}
}

// --- chat ---------------------------------------------------------------

// TestE2EChatCarriesHistory is the core chat contract: the conversation is
// cumulative. The second request must contain the first line's user message
// AND the assistant reply, stdout carries one final answer per line, and EOF
// (Ctrl-D) is a clean exit with a cumulative summary on stderr.
func TestE2EChatCarriesHistory(t *testing.T) {
	srv, reqs := capturingServer(t, textBody("first answer"), textBody("second answer"))
	cfgPath, dir := writeTestConfig(t, srv.URL, "session_dir: sessions\n")

	code, stdout, stderr := execCLI(t, []string{"chat", cfgPath}, "hello\nagain\n")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	// CONTRACT: stdout stays the answer channel - the answers, each followed
	// by a newline, and nothing else. (These scripted answers happen to be
	// single-line; a real one may span several, which is why the contract is a
	// stream rather than one line per answer.)
	if stdout != "first answer\nsecond answer\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if len(*reqs) != 2 {
		t.Fatalf("provider calls: %d, want 2", len(*reqs))
	}

	second := (*reqs)[1].Messages
	if !hasMessage(second, "user", "hello") || !hasMessage(second, "assistant", "first answer") {
		t.Errorf("second request lost the first turn: %+v", second)
	}
	if last := second[len(second)-1]; last.Role != "user" || last.Content != "again" {
		t.Errorf("second request must end with the new user line, got %+v", last)
	}

	// The prompt is on stderr so it cannot pollute the answer channel.
	if !strings.Contains(stderr, "> ") {
		t.Errorf("stderr missing the prompt: %q", stderr)
	}
	// One cumulative summary for the whole session (2 turns, not 1+1).
	if !strings.Contains(stderr, "2 turns") {
		t.Errorf("stderr missing the cumulative summary: %q", stderr)
	}

	assertChatSessionLog(t, dir)
}

// hasMessage reports whether the captured request carries a message with
// exactly this role and content.
func hasMessage(msgs []capturedMessage, role, content string) bool {
	for _, m := range msgs {
		if m.Role == role && m.Content == content {
			return true
		}
	}
	return false
}

// assertChatSessionLog pins the chat session record: ONE file for the whole
// conversation, opened with a chat run_start and closed with a single run_end
// carrying the session totals - not one pair per line.
//
// It also pins the turn numbering. CONTRACT: `turn` is monotonic inside one
// run_start/run_end pair and run_end.turns equals the highest turn logged.
// Without a turn base every chat line would restart at 1, so a session log
// with three lines would read 1,1,1 while run_end claimed 3 turns - the JSONL
// event schema is frozen public API and a replay consumer cannot repair that.
func assertChatSessionLog(t *testing.T, dir string) {
	t.Helper()
	events := readSessionEvents(t, dir)

	var (
		counts    = map[string]int{}
		turns     []int
		runEnd    session.Event
		sawRunEnd bool
	)
	for _, e := range events {
		counts[e.Type]++
		switch e.Type {
		case "llm_response":
			turns = append(turns, e.Turn)
		case "run_end":
			runEnd, sawRunEnd = e, true
		case "run_start":
			if e.Task != chatTaskLabel {
				t.Errorf("run_start task = %q, want %q", e.Task, chatTaskLabel)
			}
		}
	}

	for _, want := range []struct {
		event string
		count int
	}{
		{"run_start", 1},
		{"run_end", 1},
		{"llm_response", 2},
	} {
		if counts[want.event] != want.count {
			t.Errorf("%s count = %d, want %d", want.event, counts[want.event], want.count)
		}
	}

	// Two lines, one turn each: the second line's turn must be 2, not 1 again.
	if len(turns) != 2 || turns[0] != 1 || turns[1] != 2 {
		t.Errorf("llm_response turns = %v, want [1 2]", turns)
	}
	if !sawRunEnd || runEnd.Turns != 2 {
		t.Errorf("run_end.turns = %d, want 2 (must match the highest logged turn)", runEnd.Turns)
	}
}

// readSessionEvents decodes the single session file written under dir.
func readSessionEvents(t *testing.T, dir string) []session.Event {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(dir, "sessions", "*.jsonl"))
	if err != nil || len(files) != 1 {
		t.Fatalf("session files: %v, %v", files, err)
	}
	data, err := os.ReadFile(files[0])
	if err != nil {
		t.Fatal(err)
	}
	var events []session.Event
	for _, line := range strings.Split(strings.TrimSpace(string(data)), "\n") {
		var e session.Event
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Fatalf("decoding session line %q: %v", line, err)
		}
		events = append(events, e)
	}
	return events
}

// TestE2ERunTurnNumberingUnaffected guards the other side of the turn-base
// change: a one-shot run has no offset, so its turns still start at 1.
func TestE2ERunTurnNumberingUnaffected(t *testing.T) {
	srv := scriptedServer(t,
		toolCallBody("fs_list", `{}`),
		textBody("done"),
	)
	cfgPath, dir := writeTestConfig(t, srv.URL, "session_dir: sessions\n")

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "list"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}

	var turns []int
	for _, e := range readSessionEvents(t, dir) {
		if e.Type == "llm_response" {
			turns = append(turns, e.Turn)
		}
	}
	if len(turns) != 2 || turns[0] != 1 || turns[1] != 2 {
		t.Errorf("run turns = %v, want [1 2]", turns)
	}
}

// TestE2EChatMultiLineAnswer pins what stdout actually promises: the answer
// verbatim plus one newline. A multi-line answer is NOT escaped or delimited,
// which is exactly why the contract is a stream and not one line per answer.
func TestE2EChatMultiLineAnswer(t *testing.T) {
	srv := scriptedServer(t, textBody("line one\nline two"), textBody("second"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	code, stdout, stderr := execCLI(t, []string{"chat", cfgPath}, "a\nb\n")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "line one\nline two\nsecond\n" {
		t.Errorf("stdout: %q", stdout)
	}
}

// TestE2EChatSharesOneBudgetPool: max_turns bounds the whole chat session, not
// each line. With max_turns 2 and one turn per line, the third line must be
// refused before any provider call (scriptedServer fails the test on a third
// request) with the frozen budget exit code.
func TestE2EChatSharesOneBudgetPool(t *testing.T) {
	srv := scriptedServer(t, textBody("a"), textBody("b"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "limits:\n  max_turns: 2\n")

	code, stdout, stderr := execCLI(t, []string{"chat", cfgPath}, "one\ntwo\nthree\n")
	if code != ExitBudgetExceeded {
		t.Fatalf("exit %d (want %d), stderr: %s", code, ExitBudgetExceeded, stderr)
	}
	if stdout != "a\nb\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if !strings.Contains(stderr, "max_turns") {
		t.Errorf("stderr must name the exhausted budget: %q", stderr)
	}
}

// TestE2EChatTokenPoolIsCumulative: max_tokens is a session pool too. Each
// scripted reply reports 15 tokens, so the second line overruns a 20-token
// budget mid-call.
func TestE2EChatTokenPoolIsCumulative(t *testing.T) {
	srv := scriptedServer(t, textBody("a"), textBody("b"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "limits:\n  max_tokens: 20\n")

	code, stdout, stderr := execCLI(t, []string{"chat", cfgPath}, "one\ntwo\nthree\n")
	if code != ExitBudgetExceeded {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "a\n" {
		t.Errorf("stdout: %q", stdout)
	}
}

// TestE2EChatSkipsEmptyLines: a bare Enter must not cost a turn or a token -
// scriptedServer fails the test on any call beyond the one scripted reply.
func TestE2EChatSkipsEmptyLines(t *testing.T) {
	srv := scriptedServer(t, textBody("ok"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	code, stdout, stderr := execCLI(t, []string{"chat", cfgPath}, "\n   \n\t\nhi\n\n")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "ok\n" {
		t.Errorf("stdout: %q", stdout)
	}
}

// TestE2EChatPreservesLineWhitespace: only a whitespace-ONLY line is free
// (docs/contracts/cli.md, chat section) - the emptiness test must not also
// rewrite the message. Pasting indented code into a chat used to arrive at the
// model with its indentation stripped, because one TrimSpace both tested the
// line and became the line that was sent.
func TestE2EChatPreservesLineWhitespace(t *testing.T) {
	srv, reqs := capturingServer(t, textBody("ok"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	const line = "    indented code\t"
	code, _, stderr := execCLI(t, []string{"chat", cfgPath}, line+"\n")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if len(*reqs) != 1 {
		t.Fatalf("provider calls: %d, want 1", len(*reqs))
	}
	if got := lastContent((*reqs)[0].Messages, "user"); got != line {
		t.Errorf("user message = %q, want %q (whitespace must survive verbatim)", got, line)
	}
}

// TestE2EChatEmptySession: closing stdin immediately is a clean exit with an
// honest zero summary - and no provider call at all.
func TestE2EChatEmptySession(t *testing.T) {
	srv := scriptedServer(t)
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	code, stdout, stderr := execCLI(t, []string{"chat", cfgPath}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "" {
		t.Errorf("stdout: %q", stdout)
	}
	if !strings.Contains(stderr, "0 turns") {
		t.Errorf("stderr: %q", stderr)
	}
}

// TestE2EChatConfigErrors: chat validates exactly like run - a bad config is
// exit 2 before anything interactive starts.
func TestE2EChatConfigErrors(t *testing.T) {
	t.Run("missing file", func(t *testing.T) {
		code, _, stderr := execCLI(t, []string{"chat", "/nonexistent.yaml"}, "hi\n")
		if code != ExitConfigError || stderr == "" {
			t.Errorf("exit %d, stderr: %q", code, stderr)
		}
	})

	t.Run("no config argument", func(t *testing.T) {
		code, _, stderr := execCLI(t, []string{"chat"}, "")
		if code != ExitConfigError || !strings.Contains(stderr, "usage") {
			t.Errorf("exit %d, stderr: %q", code, stderr)
		}
	})

	t.Run("task arguments are rejected", func(t *testing.T) {
		srv := scriptedServer(t) // a stray task must not reach the provider
		cfgPath, _ := writeTestConfig(t, srv.URL, "")
		code, _, stderr := execCLI(t, []string{"chat", cfgPath, "do", "the", "thing"}, "")
		if code != ExitConfigError || !strings.Contains(stderr, "amele run") {
			t.Errorf("exit %d, stderr: %q", code, stderr)
		}
	})

	t.Run("invalid config", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "bad.yaml")
		if err := os.WriteFile(path, []byte("provider: {base_url: 'not a url'}\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		code, _, stderr := execCLI(t, []string{"chat", path}, "hi\n")
		if code != ExitConfigError || !strings.Contains(stderr, "model is required") {
			t.Errorf("exit %d, stderr: %q", code, stderr)
		}
	})
}

// TestE2EChatProviderError maps a failing endpoint onto the frozen exit code
// and ends the session rather than looping on a dead provider.
func TestE2EChatProviderError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	code, stdout, _ := execCLI(t, []string{"chat", cfgPath}, "hi\nstill there?\n")
	if code != ExitProviderError {
		t.Errorf("exit %d, want %d", code, ExitProviderError)
	}
	if stdout != "" {
		t.Errorf("stdout: %q", stdout)
	}
}

// TestE2EChatIgnoresOutputSchema pins the documented Phase 2 decision:
// output.schema is a one-shot contract, so chat neither enforces it nor
// swallows a non-JSON answer - but it does say so on stderr.
func TestE2EChatIgnoresOutputSchema(t *testing.T) {
	srv, reqs := capturingServer(t, textBody("plain prose, not JSON"))
	cfgPath, _ := writeTestConfig(t, srv.URL, schemaBlock)

	code, stdout, stderr := execCLI(t, []string{"chat", cfgPath}, "hi\n")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "plain prose, not JSON\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if !strings.Contains(stderr, "output.schema") {
		t.Errorf("stderr must warn that the schema is ignored: %q", stderr)
	}
	if len(*reqs) != 1 || (*reqs)[0].ResponseFormat != nil {
		t.Errorf("chat must not send response_format: %+v", *reqs)
	}
}

// TestE2EChatBadSchemaIsConfigError: an uncompilable output.schema is still a
// broken config, even though chat ignores the schema itself.
func TestE2EChatBadSchemaIsConfigError(t *testing.T) {
	srv := scriptedServer(t)
	cfgPath, _ := writeTestConfig(t, srv.URL, "output:\n  schema:\n    type: 3.14\n")

	code, _, stderr := execCLI(t, []string{"chat", cfgPath}, "hi\n")
	if code != ExitConfigError || !strings.Contains(stderr, "schema") {
		t.Errorf("exit %d, stderr: %q", code, stderr)
	}
}

// TestE2EChatModelOverride: --model behaves exactly as in run.
func TestE2EChatModelOverride(t *testing.T) {
	srv, reqs := capturingServer(t, textBody("ok"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	// stdin has no trailing newline: the last unterminated line must still be
	// answered before EOF ends the session.
	code, stdout, stderr := execCLI(t, []string{"chat", cfgPath, "--model", "override-model"}, "hi")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "ok\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if got := (*reqs)[0].Model; got != "override-model" {
		t.Errorf("model sent: %q", got)
	}
}

// TestE2EChatToolCallsWork proves the loop's tool dispatch is live in chat too
// and that the tool round-trip does not break history: the follow-up line
// still carries the previous answer.
func TestE2EChatToolCallsWork(t *testing.T) {
	srv, reqs := capturingServer(t,
		toolCallBody("fs_read", `{"path":"data.txt"}`),
		textBody("it says hi"),
		textBody("you are welcome"),
	)
	cfgPath, dir := writeTestConfig(t, srv.URL, "")
	if err := os.WriteFile(filepath.Join(dir, "data.txt"), []byte("hi"), 0o600); err != nil {
		t.Fatal(err)
	}

	code, stdout, stderr := execCLI(t, []string{"chat", cfgPath}, "read data.txt\nthanks\n")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "it says hi\nyou are welcome\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if !strings.Contains(stderr, "1 tool call,") {
		t.Errorf("summary must count the tool call: %q", stderr)
	}
	if third := (*reqs)[2].Messages; !hasMessage(third, "assistant", "it says hi") {
		t.Errorf("third request lost the previous answer: %+v", third)
	}
}

// TestE2EChatCancelledContext: Ctrl-C at the prompt ends the session with the
// interrupted-task code (1), never the budget code - an operator interrupt
// must not look like a budget overrun in monitoring.
func TestE2EChatCancelledContext(t *testing.T) {
	srv := scriptedServer(t) // no provider call may happen
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	// stdin never delivers a line, so the only way out is the cancelled ctx.
	done := make(chan struct{})
	t.Cleanup(func() { close(done) })

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	var out, errBuf bytes.Buffer
	code := run(ctx, []string{"chat", cfgPath}, hangingReader{done}, &out, &errBuf, env(t))
	if code != ExitTaskFailed {
		t.Errorf("exit %d (want %d), stderr: %s", code, ExitTaskFailed, errBuf.String())
	}
}

// TestLineReaderSharedWithPrompter is the regression for the two-consumers bug:
// the REPL and the permission prompter read the SAME stdin, so they must share
// one buffered reader. Two independent bufio readers would let the prompter's
// buffer swallow the next chat line.
func TestLineReaderSharedWithPrompter(t *testing.T) {
	var errBuf bytes.Buffer
	lines := newLineReader(strings.NewReader("y\nnext chat line\n"))
	prompt := newPrompter(lines, &errBuf)

	ok, err := prompt("fs_write", "{}", "")
	if err != nil || !ok {
		t.Fatalf("approval = %v, %v", ok, err)
	}
	got, err := lines.ReadLine()
	if err != nil {
		t.Fatalf("ReadLine: %v", err)
	}
	if got != "next chat line" {
		t.Errorf("the prompter swallowed the next line: %q", got)
	}
}

// TestLineReader covers the line-splitting contract: no trailing newline, CRLF
// tolerated, EOF surfaced with any final unterminated line, and a pathological
// line bounded so a runaway paste cannot exhaust memory.
func TestLineReader(t *testing.T) {
	lines := newLineReader(strings.NewReader("a\r\nb"))
	if got, err := lines.ReadLine(); got != "a" || err != nil {
		t.Errorf("first line: %q, %v", got, err)
	}
	got, err := lines.ReadLine()
	if got != "b" || !errors.Is(err, io.EOF) {
		t.Errorf("last unterminated line: %q, %v", got, err)
	}
	if _, err := lines.ReadLine(); !errors.Is(err, io.EOF) {
		t.Errorf("exhausted reader: %v", err)
	}

	t.Run("long line is capped", func(t *testing.T) {
		long := strings.Repeat("x", maxChatLineBytes+5000)
		lines := newLineReader(strings.NewReader(long + "\nnext\n"))
		got, err := lines.ReadLine()
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != maxChatLineBytes {
			t.Errorf("line not capped: %d bytes", len(got))
		}
		// The rest of the over-long line is discarded, not re-served as if it
		// were the next line the user typed.
		if next, _ := lines.ReadLine(); next != "next" {
			t.Errorf("next line: %q", next)
		}
	})
}

// TestBuildRegistryShell pins the default-off contract: the shell tool exists
// only when tools.shell.enabled is explicitly true.
func TestBuildRegistryShell(t *testing.T) {
	tests := []struct {
		name  string
		tools config.ToolsConfig
		want  bool
	}{
		{"absent block", config.ToolsConfig{}, false},
		{"explicitly disabled", config.ToolsConfig{Shell: config.ShellConfig{Enabled: false, Allow: []string{"git *"}}}, false},
		{"enabled", config.ToolsConfig{Shell: config.ShellConfig{Enabled: true}}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.Config{Workspace: t.TempDir(), Tools: tt.tools}
			registry, err := buildRegistry(cfg)
			if err != nil {
				t.Fatalf("buildRegistry: %v", err)
			}
			if got := slices.Contains(registry.Names(), "shell"); got != tt.want {
				t.Errorf("shell registered = %v want %v (names: %v)", got, tt.want, registry.Names())
			}
		})
	}
}

// TestE2EShellToolCall walks a shell call through the loop, proving the tool is
// reachable by the model and that its output comes back as a tool result.
func TestE2EShellToolCall(t *testing.T) {
	srv := scriptedServer(t,
		toolCallBody("shell", `{"command": "echo shell-ran"}`),
		textBody("done"),
	)
	cfgPath, _ := writeTestConfig(t, srv.URL, `  shell:
    enabled: true
    allow: ["echo *"]
`)
	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "go"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "done\n" {
		t.Errorf("stdout: %q", stdout)
	}
}

// anthropicCapturedRequest is the subset of an Anthropic Messages request body
// the e2e tests assert on.
type anthropicCapturedRequest struct {
	Model     string `json:"model"`
	MaxTokens int    `json:"max_tokens"`
	Messages  []struct {
		Role    string `json:"role"`
		Content []struct {
			Type      string `json:"type"`
			Text      string `json:"text"`
			ToolUseID string `json:"tool_use_id"`
			Content   string `json:"content"`
		} `json:"content"`
	} `json:"messages"`
}

// anthropicServer is capturingServer's sibling for the native Anthropic wire
// format: it serves canned /v1/messages responses in order and asserts the
// protocol facts every request must carry (path, x-api-key, anthropic-version).
func anthropicServer(t *testing.T, bodies ...string) (*httptest.Server, *[]anthropicCapturedRequest) {
	t.Helper()
	var (
		call int
		reqs []anthropicCapturedRequest
	)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The client owns the /v1/messages suffix - the config's base_url must
		// never end up doubled or dropped.
		if r.URL.Path != "/v1/messages" {
			t.Errorf("request path: got %q want /v1/messages", r.URL.Path)
		}
		if got := r.Header.Get("x-api-key"); got != "sk-test-secret-key" {
			t.Errorf("x-api-key header: got %q", got)
		}
		if r.Header.Get("anthropic-version") == "" {
			t.Error("anthropic-version header missing")
		}
		var got anthropicCapturedRequest
		_ = json.NewDecoder(r.Body).Decode(&got)
		reqs = append(reqs, got)
		if call >= len(bodies) {
			t.Errorf("unexpected extra provider call #%d", call+1)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		_, _ = w.Write([]byte(bodies[call]))
		call++
	}))
	t.Cleanup(srv.Close)
	return srv, &reqs
}

// anthropicTextBody is textBody's sibling for the Messages wire format: a
// final-answer response carrying text and nothing else.
func anthropicTextBody(text string) string {
	b, _ := json.Marshal(text)
	return fmt.Sprintf(`{"content":[{"type":"text","text":%s}],
		"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5}}`, b)
}

// writeAnthropicConfig is writeTestConfig's sibling for the native Anthropic
// path: it renders a provider.type anthropic config bound to the fake
// endpoint, appending extra verbatim, and returns the config path and the
// workspace dir.
func writeAnthropicConfig(t *testing.T, baseURL, extra string) (string, string) {
	t.Helper()
	dir := t.TempDir()
	yaml := fmt.Sprintf(`
model: claude-test
provider:
  type: anthropic
  base_url: %s
  api_key: ${TEST_KEY}
  max_output_tokens: 512
system_prompt: "You are a test agent."
tools:
  fs: true
%s`, baseURL, extra)
	path := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(path, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	return path, dir
}

// TestE2EAnthropicToolRoundTrip drives `amele run` end to end against a fake
// Anthropic endpoint: provider.type selects the native client, a tool_use
// round-trip reaches the sandboxed fs_read, the tool result travels back on
// the wire, and only the final text lands on stdout (exit 0).
func TestE2EAnthropicToolRoundTrip(t *testing.T) {
	srv, reqs := anthropicServer(t,
		`{"content":[{"type":"tool_use","id":"tu1","name":"fs_read","input":{"path":"data.txt"}}],
		  "stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5}}`,
		anthropicTextBody("anthropic answer"),
	)
	cfgPath, dir := writeAnthropicConfig(t, srv.URL, "")
	if err := os.WriteFile(filepath.Join(dir, "data.txt"), []byte("workspace content"), 0o600); err != nil {
		t.Fatal(err)
	}

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "read data.txt"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	// CONTRACT: stdout carries only the final answer.
	if stdout != "anthropic answer\n" {
		t.Errorf("stdout: %q", stdout)
	}

	got := *reqs
	if len(got) != 2 {
		t.Fatalf("provider calls: got %d want 2", len(got))
	}
	// provider.max_output_tokens must reach the wire as max_tokens.
	if got[0].MaxTokens != 512 {
		t.Errorf("max_tokens: got %d want 512", got[0].MaxTokens)
	}
	// The second request must carry the fs_read result as a tool_result block
	// tied to the tool_use id.
	var foundResult bool
	for _, msg := range got[1].Messages {
		for _, block := range msg.Content {
			if block.Type == "tool_result" && block.ToolUseID == "tu1" &&
				strings.Contains(block.Content, "workspace content") {
				foundResult = true
			}
		}
	}
	if !foundResult {
		t.Errorf("second request missing the tool_result for tu1: %+v", got[1].Messages)
	}
}

// TestBuildProviderSelectsByType pins the type-to-client mapping and the field
// wiring for both clients - the request timeout must reach whichever client is
// chosen, and max_output_tokens must reach (only) the Anthropic one.
func TestBuildProviderSelectsByType(t *testing.T) {
	pc := config.ProviderConfig{
		BaseURL:         "https://x.example.com",
		APIKey:          "k",
		RequestTimeout:  config.Duration(5 * time.Second),
		MaxOutputTokens: 256,
	}

	for _, typ := range []string{"", config.ProviderTypeOpenAI} {
		cfg := &config.Config{Provider: pc}
		cfg.Provider.Type = typ
		provider, err := buildProvider(cfg)
		if err != nil {
			t.Fatalf("type %q: buildProvider: %v", typ, err)
		}
		client, ok := provider.(*llm.OpenAIClient)
		if !ok {
			t.Fatalf("type %q: want *llm.OpenAIClient", typ)
		}
		if client.BaseURL != pc.BaseURL || client.APIKey != pc.APIKey || client.RequestTimeout != 5*time.Second {
			t.Errorf("type %q: fields not wired: %+v", typ, client)
		}
	}

	cfg := &config.Config{Provider: pc}
	cfg.Provider.Type = config.ProviderTypeAnthropic
	// A dialect is inert on this wire: it must not fail the construction.
	cfg.Provider.Dialect = "deepseek"
	provider, err := buildProvider(cfg)
	if err != nil {
		t.Fatalf("type anthropic: buildProvider: %v", err)
	}
	client, ok := provider.(*llm.AnthropicClient)
	if !ok {
		t.Fatal("type anthropic: want *llm.AnthropicClient")
	}
	if client.BaseURL != pc.BaseURL || client.APIKey != pc.APIKey ||
		client.RequestTimeout != 5*time.Second || client.MaxOutputTokens != 256 {
		t.Errorf("anthropic fields not wired: %+v", client)
	}
}

// TestExplainCommand covers the explain wiring: a valid config prints the
// report to stdout and exits 0 without any provider traffic (the zero-body
// capturing server fails the test on any request), and every config problem is
// exit 2 on stderr.
func TestExplainCommand(t *testing.T) {
	t.Run("valid config", func(t *testing.T) {
		srv := scriptedServer(t) // zero scripted bodies: any provider call fails the test
		// The subprocess argv carries the interpolated ${TEST_KEY} so the e2e
		// exercises the real leak path (regression for the review finding:
		// struct-literal fixtures have an empty interpolated list and cannot).
		cfgPath, _ := writeTestConfig(t, srv.URL, `  subprocess:
    - name: upload
      description: uploads a report
      command: ["curl", "-H", "Authorization: Bearer ${TEST_KEY}"]
`)

		code, stdout, stderr := execCLI(t, []string{"explain", cfgPath}, "")
		if code != ExitOK {
			t.Fatalf("exit %d, stderr: %s", code, stderr)
		}
		for _, section := range []string{"MODEL & PROVIDER", "TOOLS", "PERMISSIONS", "BUDGETS", "CONCURRENCY", "OUTPUT", "SESSION", "WARNINGS"} {
			if !strings.Contains(stdout, section) {
				t.Errorf("report missing section %q:\n%s", section, stdout)
			}
		}
		// SECURITY: the interpolated API key must never reach the report -
		// not via the (omitted) api_key field and not via the argv vector.
		if strings.Contains(stdout, "sk-test-secret-key") {
			t.Errorf("interpolated secret leaked into explain output:\n%s", stdout)
		}
		if !strings.Contains(stdout, "Authorization: Bearer [REDACTED]") {
			t.Errorf("argv secret not redacted:\n%s", stdout)
		}
	})

	t.Run("missing file", func(t *testing.T) {
		code, _, stderr := execCLI(t, []string{"explain", "/nope.yaml"}, "")
		if code != ExitConfigError || stderr == "" {
			t.Errorf("code=%d stderr=%q", code, stderr)
		}
	})

	t.Run("usage errors", func(t *testing.T) {
		for _, args := range [][]string{{"explain"}, {"explain", "a.yaml", "extra"}} {
			code, _, stderr := execCLI(t, args, "")
			if code != ExitConfigError || !strings.Contains(stderr, "usage: amele explain") {
				t.Errorf("args %v: code=%d stderr=%q", args, code, stderr)
			}
		}
	})
}

// TestExplainGolden pins the full report for a config exercising every section
// - the docs/engineering.md §6 golden requirement for a UI surface.
//
// Determinism: config.Load resolves workspace/session_dir against the config's
// temp directory, which changes every run, so the output is normalized by
// replacing the temp dir with "<TMP>" before comparing (and before -update
// writes the file). Everything else in the report is deterministic by
// construction (sorted maps, fixed defaults).
func TestExplainGolden(t *testing.T) {
	yaml := `model: golden-model
provider:
  base_url: https://api.example.com/v1
  api_key: ${TEST_KEY}
workspace: ws
session_dir: sessions
lock: true
tools:
  fs: true
  shell:
    enabled: true
    env: ["TZ"]
  subprocess:
    - name: grep_logs
      description: search the logs
      command: ["sh", "-c", "grep -i error /var/log/app.log"]
      timeout: 30s
      env: ["TZ", "LOG_DIR"]
limits:
  max_turns: 10
  timeout: 5m
permissions:
  default: allow
  tools:
    fs_write: ask
    ghost_tool: deny
output:
  schema:
    type: object
`
	// extraArgs is built from the temp dir, so each case names the same
	// directory the normalization below rewrites to <TMP>.
	cases := []struct {
		name   string
		golden string
		args   func(dir string) []string
	}{
		{"plain", "explain.txt", func(string) []string { return nil }},
		{
			// The overridden report is a golden of its own: the OVERRIDES block
			// and the per-line markers are the surface an operator reads to
			// tell a command-line value from one the YAML actually contains.
			name:   "with overrides",
			golden: "explain-overrides.txt",
			args: func(dir string) []string {
				return []string{
					"--set", "model=cli-model",
					"-w", filepath.Join(dir, "other-ws"),
					"--set", "limits.max_turns=3",
					"--set", "limits.max_tokens=500",
					"--set", "session_dir=",
					"--set", "output.max_schema_retries=4",
					"--set", "prompt=summarize {{input}}",
				}
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			for _, sub := range []string{"ws", "other-ws"} {
				if err := os.Mkdir(filepath.Join(dir, sub), 0o750); err != nil {
					t.Fatal(err)
				}
			}
			cfgPath := filepath.Join(dir, "agent.yaml")
			if err := os.WriteFile(cfgPath, []byte(yaml), 0o600); err != nil {
				t.Fatal(err)
			}

			args := append([]string{"explain", cfgPath}, tc.args(dir)...)
			code, stdout, stderr := execCLI(t, args, "")
			if code != ExitOK {
				t.Fatalf("exit %d, stderr: %s", code, stderr)
			}
			got := strings.ReplaceAll(stdout, dir, "<TMP>")

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
				t.Errorf("explain output differs from golden.\ngot:\n%s\nwant:\n%s", got, want)
			}
		})
	}
}

// TestExplainProblemsGolden pins the report of a config that cannot run: the
// PROBLEMS block, the MISSING marks and the fact that every other section is
// still there. It is the surface the "explain reports, run gates" decision
// created, so it gets a golden of its own.
//
// The fixture avoids OS-dependent error text (no stat failures): an absent
// `model` and an unset ${VAR} are reported in words amele owns.
func TestExplainProblemsGolden(t *testing.T) {
	yaml := `provider:
  base_url: https://api.example.com/v1
  api_key: ${PACK_KEY}
tools:
  subprocess:
    - name: notify
      description: sends a notification
      command: ["/opt/nonexistent/notify"]
      env: ["MAILTO"]
limits:
  max_turns: 4
  max_tokens: 1000
`
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "agent.yaml")
	if err := os.WriteFile(cfgPath, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}

	var out, errBuf bytes.Buffer
	code := run(context.Background(), []string{"explain", cfgPath}, strings.NewReader(""), &out, &errBuf, emptyEnv)
	if code != ExitOK {
		t.Fatalf("exit = %d, want %d; stderr: %s", code, ExitOK, errBuf.String())
	}
	if errBuf.String() != "" {
		t.Errorf("stderr should stay empty, got %q", errBuf.String())
	}
	got := strings.ReplaceAll(out.String(), dir, "<TMP>")

	goldenPath := filepath.Join("testdata", "golden", "explain-problems.txt")
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
		t.Errorf("explain output differs from golden.\ngot:\n%s\nwant:\n%s", got, want)
	}
}

// TestExplainExitTwoCases pins the exit-2 set docs/contracts/cli.md
// enumerates. Since explain became a report, exit 2 means "the loader gave me
// no config to describe" - which is more than the file being unreadable or
// unparseable: the literal-api_key ban runs on the raw bytes, and strict
// decoding rejects unknown keys, both on files that read and parse fine. The
// doc claimed otherwise until a review caught it; this table is what keeps
// the two honest. Every case must also leave stdout empty: a partial report
// on a config that never loaded would be a lie.
func TestExplainExitTwoCases(t *testing.T) {
	cases := []struct {
		name    string
		yaml    string
		wantSub string
	}{
		{
			name:    "unparseable yaml",
			yaml:    "model: [\n",
			wantSub: "parsing",
		},
		{
			name:    "empty file",
			yaml:    "",
			wantSub: "file is empty",
		},
		{
			name:    "multiple documents",
			yaml:    "model: m\nprovider:\n  base_url: https://x/v1\n---\nmodel: other\n",
			wantSub: "multiple YAML documents",
		},
		{
			name:    "unknown key",
			yaml:    "model: m\nmax_token: 5\nprovider:\n  base_url: https://x/v1\n",
			wantSub: "max_token",
		},
		{
			name:    "literal api_key in a readable, parseable file",
			yaml:    "model: m\nprovider:\n  base_url: https://x/v1\n  api_key: sk-live-literal\n",
			wantSub: "literal secrets in YAML are forbidden",
		},
		{
			name:    "unreadable system_prompt_file",
			yaml:    "model: m\nsystem_prompt_file: nope.md\nprovider:\n  base_url: https://x/v1\n",
			wantSub: "system_prompt_file",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "agent.yaml")
			if err := os.WriteFile(path, []byte(tc.yaml), 0o600); err != nil {
				t.Fatal(err)
			}
			var out, errBuf bytes.Buffer
			code := run(context.Background(), []string{"explain", path}, strings.NewReader(""), &out, &errBuf, emptyEnv)
			if code != ExitConfigError {
				t.Fatalf("exit = %d, want %d; stdout: %s stderr: %s", code, ExitConfigError, out.String(), errBuf.String())
			}
			if out.String() != "" {
				t.Errorf("stdout must stay empty when no config loaded, got %q", out.String())
			}
			if !strings.Contains(errBuf.String(), tc.wantSub) {
				t.Errorf("stderr %q does not mention %q", errBuf.String(), tc.wantSub)
			}
		})
	}
}

// TestPromptConflictIsReportedNotShortCircuited pins where the two prompt
// fields being set at once now lands: it is a validation violation, so
// `validate` refuses the config (exit 2) while `explain` describes it (exit 0,
// named in PROBLEMS) - the same split every other violation follows. It used
// to abort the load, which put explain in the exit-2 set (see
// TestExplainExitTwoCases) and hid the file's other violations behind it.
func TestPromptConflictIsReportedNotShortCircuited(t *testing.T) {
	dir := t.TempDir()
	yaml := "model: m\nsystem_prompt: a\nsystem_prompt_file: b.md\nprovider:\n  base_url: https://x/v1\n"
	if err := os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}

	code, out, errBuf := execCLI(t, []string{"explain", dir}, "")
	if code != ExitOK {
		t.Fatalf("explain exit = %d, want %d; stderr: %s", code, ExitOK, errBuf)
	}
	if !strings.Contains(out, "PROBLEMS") || !strings.Contains(out, "mutually exclusive") {
		t.Errorf("explain report does not name the conflict in PROBLEMS:\n%s", out)
	}

	// The gating half: every command that would actually run the agent still
	// refuses. `validate` is the one usually tested; run and chat share the
	// load path but not the code around it, and the tolerant/strict split is
	// exactly the kind of change that could let one of them through.
	for _, argv := range [][]string{
		{"validate", dir},
		{"run", dir, "do the thing"},
		{"chat", dir},
	} {
		code, _, errBuf := execCLI(t, argv, "hi\n")
		if code != ExitConfigError {
			t.Errorf("%v exit = %d, want %d; stderr: %s", argv, code, ExitConfigError, errBuf)
		}
		if !strings.Contains(errBuf, "mutually exclusive") {
			t.Errorf("%v stderr does not name the conflict:\n%s", argv, errBuf)
		}
	}
}

// TestMissingEnvGatesRunAndValidate is the other side of
// TestExplainReportsMissingEnv, and the reason the tolerant/strict split is
// safe: explain DESCRIBES a config with an undefined ${VAR} (exit 0), while
// every command that would spend money on it refuses with exit 2. Only
// internal/config pinned this; a cmd-level wiring slip (explain's tolerant
// loader reused by run) would not have been caught there.
func TestMissingEnvGatesRunAndValidate(t *testing.T) {
	dir := t.TempDir()
	yaml := "model: m\nprovider:\n  base_url: https://x/v1\n  api_key: ${PACK_KEY}\n"
	if err := os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}

	for _, argv := range [][]string{
		{"validate", dir},
		{"run", dir, "do the thing"},
	} {
		var out, errBuf bytes.Buffer
		code := run(context.Background(), argv, strings.NewReader(""), &out, &errBuf, emptyEnv)
		if code != ExitConfigError {
			t.Errorf("%v exit = %d, want %d; stderr: %s", argv, code, ExitConfigError, errBuf.String())
		}
		if !strings.Contains(errBuf.String(), "PACK_KEY") {
			t.Errorf("%v stderr does not name the undefined variable:\n%s", argv, errBuf.String())
		}
	}

	var out, errBuf bytes.Buffer
	if code := run(context.Background(), []string{"explain", dir}, strings.NewReader(""), &out, &errBuf, emptyEnv); code != ExitOK {
		t.Errorf("explain exit = %d, want %d; stderr: %s", code, ExitOK, errBuf.String())
	}
}

// emptyEnv defines no environment variables at all, for tests exercising the
// undefined-${VAR} path (explain's requirements report on a missing key).
func emptyEnv(string) (string, bool) { return "", false }

// TestExplainReportsMissingEnv: explain reports, run gates. A new user
// pointing explain at a pack with unset env vars gets the WHOLE report - the
// point is to learn what the pack does before setting anything up - with the
// variables marked MISSING and named in PROBLEMS, and exit 0. Exit 2 is left
// to configs the loader rejected outright (TestExplainExitTwoCases).
func TestExplainReportsMissingEnv(t *testing.T) {
	dir := t.TempDir()
	yaml := "model: m\nprovider:\n  base_url: https://x/v1\n  api_key: ${PACK_KEY}\n"
	if err := os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errBuf bytes.Buffer
	code := run(context.Background(), []string{"explain", dir}, strings.NewReader(""), &out, &errBuf, emptyEnv)
	if code != ExitOK {
		t.Fatalf("exit = %d, want %d; stderr: %s", code, ExitOK, errBuf.String())
	}
	for _, want := range []string{
		"PROBLEMS", "PACK_KEY", "✗ MISSING", "MODEL & PROVIDER", "WARNINGS",
	} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("stdout missing %q:\n%s", want, out.String())
		}
	}
	// A successful command says nothing on stderr: the problems are IN the
	// report, and a cron wrapper that mails stderr must not be woken by a
	// deliberate inspection.
	if errBuf.String() != "" {
		t.Errorf("stderr should stay empty, got %q", errBuf.String())
	}
}

// TestExplainReportsValidationProblems: a config that parses but does not
// validate is exactly the one an operator needs described - refusing to
// describe it (the old exit-2 gate) hid the requirements list behind the very
// error the reader was trying to understand.
func TestExplainReportsValidationProblems(t *testing.T) {
	dir := t.TempDir()
	yaml := `model: m
provider:
  base_url: https://x/v1
  api_key: ${PACK_KEY}
workspace: /nonexistent-workspace-for-explain
tools:
  subprocess:
    - name: notify
      description: d
      command: ["/opt/nonexistent/notify"]
`
	if err := os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	env := func(name string) (string, bool) {
		if name == "PACK_KEY" {
			return "sk-test", true
		}
		return "", false
	}
	var out, errBuf bytes.Buffer
	code := run(context.Background(), []string{"explain", dir}, strings.NewReader(""), &out, &errBuf, env)
	if code != ExitOK {
		t.Fatalf("exit = %d, want %d; stderr: %s", code, ExitOK, errBuf.String())
	}
	for _, want := range []string{
		"PROBLEMS", "is not accessible", // the validation violation is stated
		"REQUIREMENTS", "PACK_KEY", // ... and the requirements still render
		"executables:", "/opt/nonexistent/notify",
	} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("stdout missing %q:\n%s", want, out.String())
		}
	}
}

// TestExplainShowsNonSecretEnvValues: the pack idiom is to parameterise a
// config through the environment, so a pre-flight that masked every
// interpolated value could not answer "which model will this cron line buy?".
// Credentials stay masked - by variable name, and unconditionally for
// whatever feeds provider.api_key.
func TestExplainShowsNonSecretEnvValues(t *testing.T) {
	dir := t.TempDir()
	yaml := `model: ${MODEL}
provider:
  base_url: https://x/v1
  api_key: ${PACK_KEY}
tools:
  shell:
    enabled: true
    deny: ["curl * ${GH_TOKEN} *"]
`
	if err := os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	env := func(name string) (string, bool) {
		v, ok := map[string]string{
			"MODEL":    "gpt-4o-mini",
			"PACK_KEY": "sk-live-pack",
			"GH_TOKEN": "ghp-abc",
		}[name]
		return v, ok
	}
	var out, errBuf bytes.Buffer
	code := run(context.Background(), []string{"explain", dir}, strings.NewReader(""), &out, &errBuf, env)
	if code != ExitOK {
		t.Fatalf("exit = %d, want %d; stderr: %s", code, ExitOK, errBuf.String())
	}
	if !strings.Contains(out.String(), `model:           "gpt-4o-mini"`) {
		t.Errorf("stdout does not show the interpolated model:\n%s", out.String())
	}
	for _, leak := range []string{"sk-live-pack", "ghp-abc"} {
		if strings.Contains(out.String(), leak) {
			t.Fatalf("secret %q leaked into explain output:\n%s", leak, out.String())
		}
	}
	if !strings.Contains(out.String(), "curl * [REDACTED] *") {
		t.Errorf("shell pattern secret not redacted:\n%s", out.String())
	}
}

// TestExplainFailsOnUnreadablePromptFileWithoutMissingEnv pins the exit-2
// regression at the cmd layer: a pack with no ${VAR} references at all but a
// system_prompt_file that does not exist is a real config error, so
// `explain` must reject it (exit 2, empty stdout) exactly as `run`/`validate`
// would - not report clean because LoadTolerant's env-missing skip used to
// fire unconditionally.
func TestExplainFailsOnUnreadablePromptFileWithoutMissingEnv(t *testing.T) {
	dir := t.TempDir()
	yaml := "model: m\nprovider:\n  base_url: https://x/v1\n  api_key: ${API_KEY}\nsystem_prompt_file: missing.md\n"
	if err := os.WriteFile(filepath.Join(dir, "agent.yaml"), []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}
	// API_KEY is defined (so EnvMissing is empty); system_prompt_file names a
	// plain path with no ${VAR} of its own, so its absence is a real error.
	definedAPIKey := func(name string) (string, bool) {
		if name == "API_KEY" {
			return "sk-test", true
		}
		return "", false
	}
	var out, errBuf bytes.Buffer
	code := run(context.Background(), []string{"explain", dir}, strings.NewReader(""), &out, &errBuf, definedAPIKey)
	if code != ExitConfigError {
		t.Fatalf("exit = %d, want %d; stdout: %s stderr: %s", code, ExitConfigError, out.String(), errBuf.String())
	}
	if out.String() != "" {
		t.Errorf("stdout should stay empty on a real config error, got %q", out.String())
	}
	if !strings.Contains(errBuf.String(), "system_prompt_file") {
		t.Errorf("stderr %q does not name system_prompt_file", errBuf.String())
	}
}

// initEnv is the environment for init tests: it defines AMELE_API_KEY, the one
// variable the starter template references, so the written file must load and
// validate under it with no other setup.
func initEnv(key string) (string, bool) {
	if key == "AMELE_API_KEY" {
		return "sk-starter-test-key", true
	}
	return "", false
}

// runInitCLI is execCLI with the init environment instead of the standard
// test one.
func runInitCLI(t *testing.T, args []string) (code int, stdout, stderr string) {
	t.Helper()
	var out, errBuf bytes.Buffer
	code = run(context.Background(), args, strings.NewReader(""), &out, &errBuf, initEnv)
	return code, out.String(), errBuf.String()
}

// TestInitWritesValidatingConfig pins the core `amele init` contract: the
// scaffold passes `amele validate` exactly as written (the workspace defaults
// to the config's directory, so a temp dir suffices), stdout stays empty
// (pipe discipline), and the stderr note names the path and the next step.
func TestInitWritesValidatingConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "agent.yaml")
	code, stdout, stderr := runInitCLI(t, []string{"init", path})
	if code != ExitOK {
		t.Fatalf("init exit %d, stderr: %s", code, stderr)
	}
	if stdout != "" {
		t.Errorf("init must not write stdout, got %q", stdout)
	}
	if !strings.Contains(stderr, "wrote "+path) || !strings.Contains(stderr, "amele validate") {
		t.Errorf("success note missing path or next step: %q", stderr)
	}
	code, _, stderr = runInitCLI(t, []string{"validate", path})
	if code != ExitOK {
		t.Fatalf("generated config does not validate: exit %d, stderr: %s", code, stderr)
	}
}

// TestInitDefaultPath pins that a bare `amele init` writes agent.yaml in the
// working directory.
func TestInitDefaultPath(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	code, stdout, stderr := runInitCLI(t, []string{"init"})
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "" {
		t.Errorf("init must not write stdout, got %q", stdout)
	}
	if _, err := os.Stat(filepath.Join(dir, "agent.yaml")); err != nil {
		t.Errorf("agent.yaml not written to the working directory: %v", err)
	}
}

// TestInitRefusesOverwrite pins the no-overwrite guarantee: an existing file
// survives byte-for-byte, the error names the path, and the exit code is 2.
func TestInitRefusesOverwrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "agent.yaml")
	original := []byte("model: precious-user-edits\n")
	if err := os.WriteFile(path, original, 0o600); err != nil {
		t.Fatal(err)
	}
	code, stdout, stderr := runInitCLI(t, []string{"init", path})
	if code != ExitConfigError {
		t.Fatalf("exit %d, want %d", code, ExitConfigError)
	}
	if !strings.Contains(stderr, path) {
		t.Errorf("error does not name the path: %q", stderr)
	}
	if stdout != "" {
		t.Errorf("init must not write stdout, got %q", stdout)
	}
	got, err := os.ReadFile(path) //nolint:gosec // G304: test-owned temp path.
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, original) {
		t.Errorf("existing file was modified: %q", got)
	}
}

// failingWriteCloser is a created-but-unwritable file: the state a full disk,
// a quota or an I/O error leaves `amele init` in after the exclusive create
// already succeeded.
type failingWriteCloser struct{}

func (failingWriteCloser) Write([]byte) (int, error) { return 0, errors.New("no space left on device") }
func (failingWriteCloser) Close() error              { return nil }

// TestWriteStarterConfigRemovesPartialFile pins the retry story: a write that
// fails after the O_EXCL create must not leave the (empty or half-written)
// file behind, because the next `amele init` would then refuse with "already
// exists" and hand the user a broken config to clean up by hand.
//
// The failure is injected through the file creator rather than faked: the
// creator really creates the file, exactly as the production one does, and
// only the write fails - so the cleanup path under test is the real one.
func TestWriteStarterConfigRemovesPartialFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "agent.yaml")
	create := func(p string) (io.WriteCloser, error) {
		f, err := os.OpenFile(p, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600) //nolint:gosec // G304: test-owned temp path.
		if err != nil {
			return nil, err
		}
		if err := f.Close(); err != nil {
			return nil, err
		}
		return failingWriteCloser{}, nil
	}

	err := writeStarterConfig(path, create)
	if err == nil {
		t.Fatal("a failed write must be reported")
	}
	if !strings.Contains(err.Error(), path) {
		t.Errorf("the error must name the path: %v", err)
	}
	if _, statErr := os.Stat(path); !os.IsNotExist(statErr) {
		t.Errorf("the unwritable file survived (%v); a retry would hit 'already exists'", statErr)
	}
}

// TestInitUsage pins the argument contract (at most one path) and that the
// command is advertised in the usage text.
func TestInitUsage(t *testing.T) {
	code, stdout, stderr := runInitCLI(t, []string{"init", "a.yaml", "b.yaml"})
	if code != ExitConfigError || !strings.Contains(stderr, "usage") {
		t.Errorf("exit %d, stderr: %q", code, stderr)
	}
	if stdout != "" {
		t.Errorf("init must not write stdout, got %q", stdout)
	}
	if !strings.Contains(usageText, "amele init") {
		t.Error("usageText does not mention the init command")
	}
}

// TestInitTemplateMatchesSchema validates the starter template against the
// published config JSON Schema - the same check the repo applies to every
// example YAML, so the scaffold cannot drift from the contract either.
func TestInitTemplateMatchesSchema(t *testing.T) {
	validator, err := schema.Compile(config.SchemaJSONBytes())
	if err != nil {
		t.Fatalf("compiling config schema: %v", err)
	}
	var doc any
	if err := goyaml.Unmarshal([]byte(starterConfig), &doc); err != nil {
		t.Fatalf("starter template is not valid YAML: %v", err)
	}
	jsonDoc, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("converting template to JSON: %v", err)
	}
	if _, feedback, ok := validator.Validate(string(jsonDoc)); !ok {
		t.Errorf("starter template does not validate against the config schema:\n%s", feedback)
	}
}

// The -q / -v tests below. Both flags are additive on `run` and `chat`, and
// both live entirely on stderr: stdout is the product channel and neither flag
// may touch it (docs/contracts/cli.md).

// TestQuietSuppressesSummary pins the headline effect: a successful quiet run
// says nothing at all, so a cron job's MAILTO only ever fires on trouble.
func TestQuietSuppressesSummary(t *testing.T) {
	for _, flagName := range []string{"-q", "--quiet"} {
		t.Run(flagName, func(t *testing.T) {
			srv := scriptedServer(t, textBody("final answer"))
			cfgPath, _ := writeTestConfig(t, srv.URL, "")

			code, stdout, stderr := execCLI(t, []string{"run", cfgPath, flagName, "task"}, "")
			if code != ExitOK {
				t.Fatalf("exit %d, stderr: %s", code, stderr)
			}
			if stdout != "final answer\n" {
				t.Errorf("quiet must not touch stdout: %q", stdout)
			}
			if stderr != "" {
				t.Errorf("quiet run wrote to stderr: %q", stderr)
			}
		})
	}
}

// TestQuietKeepsErrors: quiet silences notes, never failures. A run that
// failed must still say why, and its exit code must be unchanged.
func TestQuietKeepsErrors(t *testing.T) {
	resp := `{"choices": [{"message": {"role": "assistant", "content": "half"}, "finish_reason": "length"}],
		"usage": {"prompt_tokens": 1, "completion_tokens": 1}}`
	srv := scriptedServer(t, resp)
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "-q", "task"}, "")
	if code != ExitTaskFailed {
		t.Fatalf("exit %d, want %d; stderr: %s", code, ExitTaskFailed, stderr)
	}
	if stdout != "" {
		t.Errorf("a failed run must write nothing to stdout: %q", stdout)
	}
	if !strings.Contains(stderr, "truncated") {
		t.Errorf("quiet must keep the error: %q", stderr)
	}
	if strings.Contains(stderr, "turns,") {
		t.Errorf("quiet must still suppress the summary: %q", stderr)
	}
}

// TestQuietSuppressesDowngradeWarning: the native-enforcement warning is a
// note, and quiet is what an operator who has read it once reaches for.
func TestQuietSuppressesDowngradeWarning(t *testing.T) {
	srv, _ := anthropicServer(t, anthropicTextBody(`{"score": 7}`))
	cfgPath, _ := writeAnthropicConfig(t, srv.URL, schemaBlock)

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "-q", "score it"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "{\"score\": 7}\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if stderr != "" {
		t.Errorf("quiet must suppress the downgrade warning: %q", stderr)
	}
}

// TestQuietKeepsPermissionDecisions: a denial is an audit fact, not a note.
// Hiding it would let a quiet cron run silently lose a tool call.
func TestQuietKeepsPermissionDecisions(t *testing.T) {
	srv := scriptedServer(t,
		toolCallBody("fs_write", `{"path":"out.txt","content":"x"}`),
		textBody("did not write"),
	)
	cfgPath, _ := writeTestConfig(t, srv.URL, "permissions:\n  tools:\n    fs_write: ask\n")

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "-q", "write out.txt"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if !strings.Contains(stderr, "fs_write") {
		t.Errorf("quiet must keep the permission decision: %q", stderr)
	}
}

// TestChatQuiet: the same two suppressions in chat - the schema note and the
// closing summary - while the prompt stays, because a prompt is interaction.
func TestChatQuiet(t *testing.T) {
	srv := scriptedServer(t, textBody(`{"score": 7}`))
	cfgPath, _ := writeTestConfig(t, srv.URL, schemaBlock)

	code, stdout, stderr := execCLI(t, []string{"chat", cfgPath, "--quiet"}, "hello\n")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "{\"score\": 7}\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if strings.Contains(stderr, "output.schema is ignored in chat") {
		t.Errorf("quiet must suppress the chat schema note: %q", stderr)
	}
	if strings.Contains(stderr, "turns,") {
		t.Errorf("quiet must suppress the session summary: %q", stderr)
	}
	if !strings.Contains(stderr, chatPrompt) {
		t.Errorf("quiet must keep the prompt - it is interaction, not noise: %q", stderr)
	}
}

// verboseToolLineRe pins the tool-result line, whose duration is real elapsed
// time and therefore cannot be compared literally.
var verboseToolLineRe = regexp.MustCompile(`amele: turn 1: fs_read ok \(\d+\.\ds\)\n`)

// TestVerboseProgressLines pins the exact per-event format on stderr: one line
// per tool call, one per result, one for the accepted final answer.
func TestVerboseProgressLines(t *testing.T) {
	for _, flagName := range []string{"-v", "--verbose"} {
		t.Run(flagName, func(t *testing.T) {
			srv := scriptedServer(t,
				toolCallBody("fs_read", `{"path":"data.txt"}`),
				textBody("read it"),
			)
			cfgPath, dir := writeTestConfig(t, srv.URL, "")
			if err := os.WriteFile(filepath.Join(dir, "data.txt"), []byte("content"), 0o600); err != nil {
				t.Fatal(err)
			}

			code, stdout, stderr := execCLI(t, []string{"run", cfgPath, flagName, "read it"}, "")
			if code != ExitOK {
				t.Fatalf("exit %d, stderr: %s", code, stderr)
			}
			if stdout != "read it\n" {
				t.Errorf("verbose must not touch stdout: %q", stdout)
			}
			if !strings.Contains(stderr, `amele: turn 1: model requested fs_read {"path":"data.txt"}`+"\n") {
				t.Errorf("missing the tool call line: %q", stderr)
			}
			if !verboseToolLineRe.MatchString(stderr) {
				t.Errorf("missing the tool result line: %q", stderr)
			}
			// textBody reports 5 completion tokens.
			if !strings.Contains(stderr, "amele: turn 2: final answer (5 tokens)\n") {
				t.Errorf("missing the final answer line: %q", stderr)
			}
			// Verbose adds lines, it does not replace the summary.
			if !strings.Contains(stderr, "2 turns") {
				t.Errorf("verbose lost the summary: %q", stderr)
			}
		})
	}
}

// TestVerboseToolFailureLine pins the error shape of the result line.
func TestVerboseToolFailureLine(t *testing.T) {
	srv := scriptedServer(t,
		toolCallBody("ghost_tool", `{}`),
		textBody("recovered"),
	)
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "-v", "task"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if !strings.Contains(stderr, "amele: turn 1: ghost_tool error: unknown tool\n") {
		t.Errorf("missing the tool error line: %q", stderr)
	}
}

// TestVerboseSanitizesModelText: SECURITY - a progress line carries the tool
// name and arguments the MODEL chose, straight to the operator's terminal. It
// must go through the same sanitizer as the approval question, or a prompt-
// injected model can erase and forge terminal output with escape sequences.
func TestVerboseSanitizesModelText(t *testing.T) {
	srv := scriptedServer(t,
		toolCallBody("fs_read", "{\"path\":\"\x1b[2K\ramele: allow tool fs_read? [y/N] y\"}"),
		textBody("done"),
	)
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "-v", "task"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if strings.ContainsAny(stderr, "\x1b\r") {
		t.Errorf("control bytes reached the terminal: %q", stderr)
	}
}

// TestVerboseRedactsSecrets: SECURITY - a progress line carries the tool
// arguments the model chose, and a model that read a secret (from a prompt, a
// file, an earlier tool result) can put it straight back into the next call's
// arguments. The session log redacts every registered secret by value before
// writing; -v opens a SECOND persisted channel for the same text (a cron job's
// stderr lands in journald or a mail spool), so it must redact identically.
//
// Both sources are exercised: an interpolated ${DB_PASSWORD} that is not a
// credential of the provider's, and the interpolated API key itself.
func TestVerboseRedactsSecrets(t *testing.T) {
	const apiKey = "sk-test-secret-key"
	srv := scriptedServer(t,
		toolCallBody("fs_read", fmt.Sprintf(`{"path":"dump-%s-%s.txt"}`, testInterpolatedValue, apiKey)),
		textBody("done"),
	)
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "agent.yaml")
	// system_prompt carries the non-API-key secret, which is what puts it in
	// Config.InterpolatedSecrets - exactly how a real leak would be seeded.
	yaml := fmt.Sprintf(`
model: test-model
provider:
  base_url: %s/v1
  api_key: ${TEST_KEY}
system_prompt: "connect to the db with ${DB_PASSWORD}"
tools:
  fs: true
`, srv.URL)
	if err := os.WriteFile(cfgPath, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "-v", "task"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	for _, secret := range []string{testInterpolatedValue, apiKey} {
		if strings.Contains(stderr, secret) {
			t.Errorf("secret %q leaked to the -v progress feed: %q", secret, stderr)
		}
		if strings.Contains(stdout, secret) {
			t.Errorf("secret %q leaked to stdout: %q", secret, stdout)
		}
	}
	// The line must still be there and still be readable - redaction replaces
	// the value, it does not drop the event.
	if !strings.Contains(stderr, "[REDACTED]") {
		t.Errorf("stderr shows no redaction marker: %q", stderr)
	}
	if !strings.Contains(stderr, "amele: turn 1: model requested fs_read") {
		t.Errorf("dispatch event missing: %q", stderr)
	}
}

// TestVerboseClipIsNotAFixedRunePromise guards the contract against the drift
// that already happened once: cli.md promised `-v` arguments were "clipped to
// 120 runes" while the loop had deliberately been raised to 512 (so by-value
// secret redaction can still match a full-length secret - see
// TestVerboseRedactsLongSecrets). A number in the contract is a promise; this
// bound is a readability knob, so the document must describe the behavior
// without freezing a figure the code is free to retune.
func TestVerboseClipIsNotAFixedRunePromise(t *testing.T) {
	raw, err := os.ReadFile("../../docs/contracts/cli.md")
	if err != nil {
		t.Fatal(err)
	}
	doc := string(raw)
	if m := regexp.MustCompile(`(?s)clipped[^.]{0,80}?(\d+) runes`).FindStringSubmatch(doc); m != nil {
		t.Errorf("cli.md promises a fixed clip of %s runes; the bound is an implementation detail: %q", m[1], m[0])
	}
	// The behavior itself must still be documented - dropping the sentence
	// would "fix" the mismatch by hiding it.
	for _, want := range []string{"clipped", "[REDACTED]"} {
		if !strings.Contains(doc, want) {
			t.Errorf("cli.md no longer describes %q for -v output", want)
		}
	}
}

// TestVerboseRedactsLongSecrets: a secret longer than the loop's old 120-rune
// argument clip must not leak its prefix - the loop's clip bound has to exceed
// any plausible secret length so by-value redaction in cmd still matches
// (review finding: a clipped secret cannot be matched by value).
func TestVerboseRedactsLongSecrets(t *testing.T) {
	longSecret := "jwt-" + strings.Repeat("a1b2c3d4", 25) // 204 runes > 120
	srv := scriptedServer(t,
		toolCallBody("fs_read", fmt.Sprintf(`{"path":"dump-%s.txt"}`, longSecret)),
		textBody("done"),
	)
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "agent.yaml")
	yaml := fmt.Sprintf(`
model: test-model
provider:
  base_url: %s/v1
  api_key: ${TEST_KEY}
system_prompt: "token is ${LONG_SECRET}"
tools:
  fs: true
`, srv.URL)
	if err := os.WriteFile(cfgPath, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}

	envFn := func(key string) (string, bool) {
		switch key {
		case "TEST_KEY":
			return "sk-test-secret-key", true
		case "LONG_SECRET":
			return longSecret, true
		}
		return "", false
	}
	var out, errBuf bytes.Buffer
	code := run(context.Background(), []string{"run", cfgPath, "-v", "task"},
		strings.NewReader(""), &out, &errBuf, envFn)
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, errBuf.String())
	}
	// The dangerous leak is the prefix: if the clip cut the secret before
	// redaction saw it, the head of the token would print verbatim.
	if strings.Contains(errBuf.String(), longSecret[:40]) {
		t.Errorf("long secret prefix leaked to the -v progress feed: %q", errBuf.String())
	}
	if !strings.Contains(errBuf.String(), "[REDACTED]") {
		t.Errorf("stderr shows no redaction marker: %q", errBuf.String())
	}
}

// TestVerboseChat: the flag is wired on chat too, and the turn numbers there
// keep counting across lines (the session numbering, not a per-line restart).
func TestVerboseChat(t *testing.T) {
	srv := scriptedServer(t, textBody("first"), textBody("second"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	code, _, stderr := execCLI(t, []string{"chat", cfgPath, "-v"}, "hello\nagain\n")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	for _, want := range []string{
		"amele: turn 1: final answer (5 tokens)\n",
		"amele: turn 2: final answer (5 tokens)\n",
	} {
		if !strings.Contains(stderr, want) {
			t.Errorf("stderr missing %q: %q", want, stderr)
		}
	}
}

// TestQuietVerboseConflict: the two flags ask for opposite things, and
// silently letting one win would hide the mistake. CONTRACT: usage error,
// exit 2, before anything is loaded.
func TestQuietVerboseConflict(t *testing.T) {
	tests := [][]string{
		{"run", "no-such-config.yaml", "-q", "-v", "task"},
		{"run", "no-such-config.yaml", "--verbose", "--quiet", "task"},
		{"chat", "no-such-config.yaml", "-v", "-q"},
	}
	for _, args := range tests {
		t.Run(strings.Join(args, " "), func(t *testing.T) {
			code, stdout, stderr := execCLI(t, args, "")
			if code != ExitConfigError {
				t.Errorf("exit %d, want %d", code, ExitConfigError)
			}
			if stdout != "" {
				t.Errorf("usage error wrote to stdout: %q", stdout)
			}
			if !strings.Contains(stderr, "--quiet") || !strings.Contains(stderr, "--verbose") {
				t.Errorf("stderr must name both flags: %q", stderr)
			}
		})
	}
}

// TestQuietVerboseAfterTaskTextIsTaskText: the flag-stop rule is frozen, and
// the new flags obey it exactly like --model.
func TestQuietVerboseAfterTaskTextIsTaskText(t *testing.T) {
	srv, reqs := capturingServer(t, textBody("ok"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "explain", "-q"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "ok\n" {
		t.Errorf("stdout: %q", stdout)
	}
	if got := lastContent((*reqs)[0].Messages, "user"); got != "explain -q" {
		t.Errorf("user message = %q, want the -q to survive as task text", got)
	}
	// It was task text, so the run was never quiet.
	if !strings.Contains(stderr, "1 turn,") {
		t.Errorf("summary missing, so -q was parsed as a flag: %q", stderr)
	}
}

// TestHelpWinsOverQuietVerboseConflict: someone who asked for the manual gets
// the manual - the page is where the two flags are explained, so answering
// with a usage error instead would be unhelpful.
func TestHelpWinsOverQuietVerboseConflict(t *testing.T) {
	_, want, _ := execCLI(t, []string{"help", "run"}, "")
	code, stdout, stderr := execCLI(t, []string{"run", "no-such-config.yaml", "-q", "-v", "-h"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != want {
		t.Errorf("stdout is not the run page: %q", stdout)
	}
}

// --- Task 4: --set overrides and the -w shortcut ---------------------------

// TestSetOverridesReachTheRun pins the basic mechanism end to end: an override
// given on the command line is what the provider sees, and it participates in
// validation like a value read from the YAML.
func TestSetOverridesReachTheRun(t *testing.T) {
	srv, reqs := capturingServer(t, textBody("ok"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "--set", "model=set-model", "--set", "limits.max_turns=3", "hi"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if stdout != "ok\n" {
		t.Errorf("stdout = %q", stdout)
	}
	if got := (*reqs)[0].Model; got != "set-model" {
		t.Errorf("model on the wire = %q, want the overridden one", got)
	}
}

// TestSetSupplesMissingRequiredField: overrides run BEFORE Validate, so a
// config with no model plus --set model=X is valid - the same rule --model has
// carried since Phase 1.
func TestSetSuppliesMissingRequiredField(t *testing.T) {
	srv, reqs := capturingServer(t, textBody("ok"))
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "agent.yaml")
	yaml := fmt.Sprintf("provider:\n  base_url: %s/v1\n  api_key: ${TEST_KEY}\n", srv.URL)
	if err := os.WriteFile(cfgPath, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}

	if code, _, _ := execCLI(t, []string{"run", cfgPath, "hi"}, ""); code != ExitConfigError {
		t.Fatalf("a config with no model must be exit %d, got %d", ExitConfigError, code)
	}
	code, _, stderr := execCLI(t, []string{"run", cfgPath, "--set", "model=from-cli", "hi"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if got := (*reqs)[0].Model; got != "from-cli" {
		t.Errorf("model on the wire = %q", got)
	}
}

// TestOverrideMergeOrder pins the documented merge rule for the sugar flags:
// --model, -w and --set all append to ONE ordered list in the order they were
// written, and the last occurrence wins. Anything else (a fixed precedence
// between the spellings) would make the effective value depend on a rule the
// command line cannot show.
func TestOverrideMergeOrder(t *testing.T) {
	for _, tt := range []struct {
		name string
		args []string
		want string
	}{
		{"set after model wins", []string{"--model", "from-flag", "--set", "model=from-set"}, "from-set"},
		{"model after set wins", []string{"--set", "model=from-set", "--model", "from-flag"}, "from-flag"},
		{"last set wins", []string{"--set", "model=a", "--set", "model=b"}, "b"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			srv, reqs := capturingServer(t, textBody("ok"))
			cfgPath, _ := writeTestConfig(t, srv.URL, "")
			args := append([]string{"run", cfgPath}, tt.args...)
			code, _, stderr := execCLI(t, append(args, "hi"), "")
			if code != ExitOK {
				t.Fatalf("exit %d, stderr: %s", code, stderr)
			}
			if got := (*reqs)[0].Model; got != tt.want {
				t.Errorf("model on the wire = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestEmptyModelFlagIsIgnored pins FROZEN Phase 1 behavior through the
// rewrite: `--model ""` (an unset shell variable in a wrapper script) means
// "no override", not "no model". --set model= is how one asks for the empty
// value on purpose.
func TestEmptyModelFlagIsIgnored(t *testing.T) {
	srv, reqs := capturingServer(t, textBody("ok"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "--model", "", "hi"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if got := (*reqs)[0].Model; got != "test-model" {
		t.Errorf("model on the wire = %q, want the config's own", got)
	}
}

// TestWorkspaceShortcutChangesTheSandbox: -w must move the fs sandbox for
// real, not just the report - the same config run against a different
// directory is the whole point of the shortcut.
func TestWorkspaceShortcutChangesTheSandbox(t *testing.T) {
	for _, flagName := range []string{"-w", "--workspace"} {
		t.Run(flagName, func(t *testing.T) {
			srv, reqs := capturingServer(t,
				toolCallBody("fs_read", `{"path":"only-here.txt"}`),
				textBody("read it"))
			cfgPath, _ := writeTestConfig(t, srv.URL, "")

			other := t.TempDir()
			if err := os.WriteFile(filepath.Join(other, "only-here.txt"), []byte("elsewhere-content"), 0o600); err != nil {
				t.Fatal(err)
			}

			code, _, stderr := execCLI(t, []string{"run", cfgPath, flagName, other, "read it"}, "")
			if code != ExitOK {
				t.Fatalf("exit %d, stderr: %s", code, stderr)
			}
			if got := lastContent((*reqs)[1].Messages, "tool"); !strings.Contains(got, "elsewhere-content") {
				t.Errorf("tool result = %q, want the file from the -w workspace", got)
			}
		})
	}
}

// TestOverridePathsResolveAgainstCwd pins the path rule: a path typed on the
// command line means what it means in the shell that typed it, NOT what it
// would mean relative to the config file (config.Load's rule for YAML values).
func TestOverridePathsResolveAgainstCwd(t *testing.T) {
	srv, reqs := capturingServer(t,
		toolCallBody("fs_read", `{"path":"cwd-marker.txt"}`),
		textBody("read it"))
	// The config lives in its own directory, so a workspace resolved against
	// the config dir could not possibly find the file below.
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	cwd := t.TempDir()
	if err := os.Mkdir(filepath.Join(cwd, "ws"), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cwd, "ws", "cwd-marker.txt"), []byte("cwd-content"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(cwd)

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "--set", "workspace=ws", "read it"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if got := lastContent((*reqs)[1].Messages, "tool"); !strings.Contains(got, "cwd-content") {
		t.Errorf("tool result = %q, want the file under the cwd-relative workspace", got)
	}
}

// TestSetSystemPromptFileReReadsTheFile: the override must replace the system
// prompt the config already loaded, not just the path - the model is told what
// the operator asked for.
func TestSetSystemPromptFileReReadsTheFile(t *testing.T) {
	srv, reqs := capturingServer(t, textBody("ok"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "")
	promptPath := filepath.Join(t.TempDir(), "prompt.txt")
	if err := os.WriteFile(promptPath, []byte("You are the overridden agent."), 0o600); err != nil {
		t.Fatal(err)
	}

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "--set", "system_prompt_file=" + promptPath, "hi"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if got := lastContent((*reqs)[0].Messages, "system"); got != "You are the overridden agent." {
		t.Errorf("system message = %q, want the overridden prompt file's content", got)
	}
}

// TestSetSessionDirDisablesLogging: an empty value is a real setting for
// session_dir - the way to run one config both with and without an audit
// trail.
func TestSetSessionDirDisablesLogging(t *testing.T) {
	srv := scriptedServer(t, textBody("ok"))
	cfgPath, dir := writeTestConfig(t, srv.URL, "session_dir: sessions\n")

	code, _, stderr := execCLI(t, []string{"run", cfgPath, "--set", "session_dir=", "hi"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if entries, err := os.ReadDir(filepath.Join(dir, "sessions")); err == nil && len(entries) > 0 {
		t.Errorf("session files written despite --set session_dir=: %v", entries)
	}
}

// TestSetExcludedKeyIsConfigError is the SECURITY test at the CLI boundary:
// the capability-granting fields are not settable, and the refusal names the
// keys that are (docs/threat-model.md §2 - the YAML stays the audited grant of
// authority).
func TestSetExcludedKeyIsConfigError(t *testing.T) {
	srv := scriptedServer(t) // zero bodies: any provider call fails the test
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	// lock joined this list on 2026-08-12: it was the only settable key that
	// could weaken a run (--set lock=false disarms the single-flight guard an
	// audited `lock: true` armed), so it left the allowlist. The YAML field is
	// untouched - only the CLI override is gone.
	for _, key := range []string{"tools.fs", "tools.shell.enabled", "permissions.default", "provider.api_key", "lock"} {
		t.Run(key, func(t *testing.T) {
			code, stdout, stderr := execCLI(t, []string{"run", cfgPath, "--set", key + "=true", "hi"}, "")
			if code != ExitConfigError {
				t.Errorf("exit %d, want %d", code, ExitConfigError)
			}
			if stdout != "" {
				t.Errorf("config error wrote to stdout: %q", stdout)
			}
			if !strings.Contains(stderr, "cannot override \""+key+"\"") || !strings.Contains(stderr, "settable keys:") {
				t.Errorf("stderr = %q, want the excluded-key refusal with the settable list", stderr)
			}
		})
	}
}

// TestSetMalformedPairIsConfigError: a missing "=" is a usage error before
// anything is spent.
func TestSetMalformedPairIsConfigError(t *testing.T) {
	srv := scriptedServer(t)
	cfgPath, _ := writeTestConfig(t, srv.URL, "")
	code, _, stderr := execCLI(t, []string{"run", cfgPath, "--set", "model", "hi"}, "")
	if code != ExitConfigError || !strings.Contains(stderr, "key=value") {
		t.Errorf("code=%d stderr=%q", code, stderr)
	}
}

// TestChatHonorsSet: chat and run load a config the same way, so the override
// layer must be in the shared path rather than bolted onto run.
func TestChatHonorsSet(t *testing.T) {
	srv, reqs := capturingServer(t, textBody("hello"))
	cfgPath, _ := writeTestConfig(t, srv.URL, "")

	code, _, stderr := execCLI(t, []string{"chat", cfgPath, "--set", "model=chat-model"}, "hi\n")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if got := (*reqs)[0].Model; got != "chat-model" {
		t.Errorf("model on the wire = %q", got)
	}
}

// TestValidateHonorsSet: validate exists to answer "will this run work?", so
// it must answer it for the parametrized invocation too - a config that is
// only valid WITH an override must validate with that override and fail
// without it.
func TestValidateHonorsSet(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "agent.yaml")
	yaml := "provider:\n  base_url: https://api.example.com/v1\n  api_key: ${TEST_KEY}\n"
	if err := os.WriteFile(cfgPath, []byte(yaml), 0o600); err != nil {
		t.Fatal(err)
	}

	if code, _, _ := execCLI(t, []string{"validate", cfgPath}, ""); code != ExitConfigError {
		t.Fatalf("a config with no model must not validate")
	}
	code, stdout, stderr := execCLI(t, []string{"validate", cfgPath, "--set", "model=from-cli"}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	if !strings.Contains(stdout, ": OK") {
		t.Errorf("stdout = %q", stdout)
	}
}

// TestExplainHonorsSet: the dry run must describe the parametrized run, and
// say which values did not come from the file.
func TestExplainHonorsSet(t *testing.T) {
	srv := scriptedServer(t)
	cfgPath, _ := writeTestConfig(t, srv.URL, "")
	other := t.TempDir()

	code, stdout, stderr := execCLI(t, []string{"explain", cfgPath, "--set", "model=explained", "-w", other}, "")
	if code != ExitOK {
		t.Fatalf("exit %d, stderr: %s", code, stderr)
	}
	for _, want := range []string{
		"OVERRIDES",
		"--set model=\"explained\"",
		"model:           \"explained\" (overridden via --set)",
		"workspace: " + strconv.Quote(other) + " (overridden via --set)",
	} {
		if !strings.Contains(stdout, want) {
			t.Errorf("report missing %q:\n%s", want, stdout)
		}
	}
}

// TestInspectFlagsFollowTheConfigPath pins the argument shape validate and
// explain share with run and chat: the config path comes FIRST, then flags. A
// flag written in the path slot is a usage error rather than a confusing
// "cannot read --set" - the CLI has one argument order, not two.
func TestInspectFlagsFollowTheConfigPath(t *testing.T) {
	for _, cmd := range []string{"validate", "explain"} {
		t.Run(cmd, func(t *testing.T) {
			code, stdout, stderr := execCLI(t, []string{cmd, "--set", "model=x", "agent.yaml"}, "")
			if code != ExitConfigError {
				t.Errorf("exit %d, want %d", code, ExitConfigError)
			}
			if stdout != "" {
				t.Errorf("usage error wrote to stdout: %q", stdout)
			}
			if !strings.Contains(stderr, "usage: amele "+cmd) {
				t.Errorf("stderr = %q, want the usage line", stderr)
			}
		})
	}
}

// TestInspectUnknownFlagIsUsageError: an unrecognized flag must not be
// mistaken for a second positional argument, and the message must still carry
// the usage line (the CLI contract's "every usage error is exit 2").
func TestInspectUnknownFlagIsUsageError(t *testing.T) {
	for _, cmd := range []string{"validate", "explain"} {
		t.Run(cmd, func(t *testing.T) {
			code, _, stderr := execCLI(t, []string{cmd, "agent.yaml", "--bogus"}, "")
			if code != ExitConfigError {
				t.Errorf("exit %d, want %d", code, ExitConfigError)
			}
			if !strings.Contains(stderr, "usage: amele "+cmd) {
				t.Errorf("stderr = %q, want the usage line", stderr)
			}
		})
	}
}

// TestFlagInConfigPathSlotIsDiagnosed covers all four config-taking commands:
// a flag written where the config path belongs must be diagnosed as the
// argument-order mistake it is.
//
// Regression (live test A-7): `amele run --set model=x agent.yaml` read
// "--set" as the config path and reported "open --set: no such file or
// directory" - a filesystem error for a command line that never named a
// missing file. GNU-style tools take flags first, so this is the ordering a
// new user tries; the answer has to teach the real order instead of sending
// them looking for a file. validate/explain rejected it already, but with the
// bare usage line and no diagnosis.
func TestFlagInConfigPathSlotIsDiagnosed(t *testing.T) {
	for _, cmd := range []string{"run", "chat", "validate", "explain"} {
		t.Run(cmd, func(t *testing.T) {
			code, stdout, stderr := execCLI(t, []string{cmd, "--set", "model=x", "agent.yaml"}, "")
			if code != ExitConfigError {
				t.Errorf("exit %d, want %d", code, ExitConfigError)
			}
			if stdout != "" {
				t.Errorf("usage error wrote to stdout: %q", stdout)
			}
			if !strings.Contains(stderr, "amele "+cmd+": \"--set\" is a flag, but the first argument is the config path") {
				t.Errorf("stderr = %q, want the argument-order diagnosis", stderr)
			}
			if !strings.Contains(stderr, "usage: amele "+cmd) {
				t.Errorf("stderr = %q, want the usage line", stderr)
			}
			if strings.Contains(stderr, "no such file") {
				t.Errorf("stderr = %q, still misdiagnoses the flag as a missing file", stderr)
			}
		})
	}
}

// TestAgentUnknownFlagIsUsageError: run and chat must answer an unknown flag
// with the same curated one-liner validate and explain give.
//
// Regression (live test A-4): they let the flag package print its own error
// AND its defaults block, so the operator was shown single-dash spellings
// (-model, -set, -w) that no documentation mentions and no example uses -
// inviting a second wrong invocation. The FlagSet's output is discarded and
// the message is reprinted in the house format instead.
func TestAgentUnknownFlagIsUsageError(t *testing.T) {
	for _, cmd := range []string{"run", "chat"} {
		t.Run(cmd, func(t *testing.T) {
			code, stdout, stderr := execCLI(t, []string{cmd, "agent.yaml", "--bogus", "hi"}, "")
			if code != ExitConfigError {
				t.Errorf("exit %d, want %d", code, ExitConfigError)
			}
			if stdout != "" {
				t.Errorf("usage error wrote to stdout: %q", stdout)
			}
			if !strings.Contains(stderr, "amele "+cmd+": flag provided but not defined: -bogus") {
				t.Errorf("stderr = %q, want the curated flag error", stderr)
			}
			if !strings.Contains(stderr, "usage: amele "+cmd) {
				t.Errorf("stderr = %q, want the usage line", stderr)
			}
			// The stdlib dump: a "Usage of <command>:" header followed by the
			// defaults block, which advertises single-dash flag spellings.
			for _, leak := range []string{"Usage of " + cmd + ":", "-model value", "-set value"} {
				if strings.Contains(stderr, leak) {
					t.Errorf("stderr leaks the flag package's usage dump (%q):\n%s", leak, stderr)
				}
			}
		})
	}
}

// TestHelpPagesListEverySettableKey guards the drift the closed allowlist
// invites: a key added to config.SettableKeys() but not to the pages leaves
// operators guessing, and a key removed from the allowlist but left in a page
// sends them to a flag that now exits 2.
func TestHelpPagesListEverySettableKey(t *testing.T) {
	_, page, _ := execCLI(t, []string{"help", "run"}, "")
	for _, key := range config.SettableKeys() {
		if !strings.Contains(page, key) {
			t.Errorf("the run help page does not list settable key %q", key)
		}
	}
}

// TestUsageListsExitCode8 pins exit code 8 into the general usage text.
// CONTRACT: docs/contracts/exit-codes.md v1.2 - an operator scripting around a
// missing MCP dependency must find the code without leaving the CLI.
func TestUsageListsExitCode8(t *testing.T) {
	code, stdout, _ := execCLI(t, []string{"help"}, "")
	if code != ExitOK {
		t.Fatalf("help exit %d", code)
	}
	if !strings.Contains(stdout, "8") || !strings.Contains(stdout, "required MCP server unavailable") {
		t.Fatalf("usage does not document exit code 8:\n%s", stdout)
	}
}

// TestPrompterHint: an MCP annotation ("this tool is destructive") is the only
// extra fact available when the human is asked, so it must appear in the
// question - and must not leave an empty pair of parentheses when absent.
func TestPrompterHint(t *testing.T) {
	t.Run("hint rendered", func(t *testing.T) {
		var errBuf bytes.Buffer
		prompt := newPrompter(newLineReader(strings.NewReader("n\n")), &errBuf)
		if _, err := prompt("github__delete_repo", "{}", "server marks this destructive"); err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(errBuf.String(), "(server marks this destructive)") {
			t.Errorf("question missing the hint: %q", errBuf.String())
		}
	})

	t.Run("no hint, no parentheses", func(t *testing.T) {
		var errBuf bytes.Buffer
		prompt := newPrompter(newLineReader(strings.NewReader("n\n")), &errBuf)
		if _, err := prompt("fs_write", "{}", ""); err != nil {
			t.Fatal(err)
		}
		if strings.Contains(errBuf.String(), "(") {
			t.Errorf("empty hint must not render parentheses: %q", errBuf.String())
		}
	})

	t.Run("hint is sanitized", func(t *testing.T) {
		var errBuf bytes.Buffer
		prompt := newPrompter(newLineReader(strings.NewReader("n\n")), &errBuf)
		// SECURITY: the hint comes from a remote MCP server's tool
		// description, so it is untrusted text on the same terminal line.
		if _, err := prompt("x", "{}", "drop\x1b[2K\rall? [y/N] "); err != nil {
			t.Fatal(err)
		}
		if hasControlBytes(errBuf.String()) {
			t.Errorf("hint leaked control bytes: %q", errBuf.String())
		}
		// Without control bytes a hint can only ever add visible text, so the
		// real question still owns the end of the line the human answers.
		if !strings.HasSuffix(errBuf.String(), "[y/N] ") {
			t.Errorf("hint displaced the real question: %q", errBuf.String())
		}
	})
}

// TestProgressLoggerRedactsLateSecret: -v is a persisted sink wired once, at
// startup. It holds the run's live registry, so a credential registered later
// is scrubbed from the very next progress line.
func TestProgressLoggerRedactsLateSecret(t *testing.T) {
	var buf bytes.Buffer
	secrets := session.NewSecretSet([]string{"sk-initial"})
	log := progressLogger(&buf, secrets)

	log("tool fs_read {\"path\":\"sk-initial\"}")
	secrets.Add("oauth-access-token")
	log("tool mcp__gh__issues {\"auth\":\"oauth-access-token\"}")

	out := buf.String()
	for _, leak := range []string{"sk-initial", "oauth-access-token"} {
		if strings.Contains(out, leak) {
			t.Errorf("-v leaked %q: %q", leak, out)
		}
	}
	if want := 2; strings.Count(out, "[REDACTED]") != want {
		t.Errorf("[REDACTED] count = %d, want %d: %q", strings.Count(out, "[REDACTED]"), want, out)
	}
}
