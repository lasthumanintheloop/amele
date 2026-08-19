package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/lasthumanintheloop/amele/internal/config"
	"github.com/lasthumanintheloop/amele/internal/llm"
	"github.com/lasthumanintheloop/amele/internal/schema"
	"github.com/lasthumanintheloop/amele/internal/tools"
	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Result wordings. They are constants because the model reads them as task
// information and an operator greps for them in a session log; the
// indeterminate one in particular is load-bearing (see below).
const (
	// badArgsText prefixes arguments the model did not manage to encode as
	// JSON. The call never leaves the harness, so the model can just retry.
	badArgsText = errorPrefix + "invalid JSON arguments: "
	// abortedText is shown when the RUN ended under the call.
	abortedText = errorPrefix + "run aborted"
	// lostText is what the model is told when a request left amele but the
	// answer did not come back.
	//
	// CONTRACT: the wording tells the model NOT to assume failure. A retried
	// "send the email" is worse than a missing one, so amele never retries the
	// call itself and never reports it as a plain error.
	lostText = errorPrefix + "response lost; the action may or may not have happened. " +
		"Do not assume it failed; verify before retrying."
	// hintDestructive and hintReadOnly are the operator-facing phrasings of the
	// server's annotations. SECURITY: they are shown in an ask prompt, never
	// used to decide anything - a hostile server would simply claim read-only.
	hintDestructive = "server marks this tool destructive"
	hintReadOnly    = "server marks this tool read-only"
)

// Tool is one MCP tool exposed to the model.
//
// It implements tools.Tool and the loop's optional outcomeInvoker, because
// almost everything an MCP call can do - the server reporting its own failure,
// a call timing out, a response getting lost - is information the model reads
// as ordinary result text while the operator needs it classified.
//
// A Tool is immutable after discovery and safe for concurrent use; the mutable
// state (the session) lives in its Server.
type Tool struct {
	srv          *Server
	def          llm.ToolDef
	original     string
	outputSchema *schema.Validator
	annotations  Annotations
}

// Def returns the definition advertised to the provider. The returned value is
// a copy, so a caller cannot rename a tool behind the server's back.
func (t *Tool) Def() llm.ToolDef {
	def := t.def
	def.Parameters = append(json.RawMessage(nil), t.def.Parameters...)
	return def
}

// Annotations returns the server's hints for this tool. Fields are nil when the
// server said nothing.
func (t *Tool) Annotations() Annotations { return t.annotations }

// Hint returns the short phrase an ask prompt shows beside this tool, or "" if
// the server published nothing worth saying. Destructive wins over read-only:
// when a server contradicts itself, the operator sees the alarming half.
func (t *Tool) Hint() string {
	if t.annotations.Destructive != nil && *t.annotations.Destructive {
		return hintDestructive
	}
	if t.annotations.ReadOnly != nil && *t.annotations.ReadOnly {
		return hintReadOnly
	}
	return ""
}

// Invoke implements tools.Tool. It is InvokeOutcome without the classification,
// for callers that only need the text.
func (t *Tool) Invoke(ctx context.Context, rawArgs string) (string, error) {
	text, _, err := t.InvokeOutcome(ctx, rawArgs)
	return text, err
}

// InvokeOutcome calls the tool on its server and reports both what the model
// should read and how the call ended.
//
// A Go error is returned ONLY when the harness could not dispatch the call at
// all (no session, and the reconnect failed). Everything else - bad arguments,
// a server-side error, a timeout, an aborted run, a lost response - comes back
// as text plus an outcome, because it is task information the model must be
// able to act on.
//
// CONTRACT: a lost response yields OutcomeIndeterminate and is never retried.
func (t *Tool) InvokeOutcome(ctx context.Context, rawArgs string) (string, tools.Outcome, error) {
	args, err := decodeArgs(rawArgs)
	if err != nil {
		// The request never left amele: nothing happened, and the model can
		// fix its own JSON on the next turn.
		return badArgsText + err.Error(), tools.Outcome{Kind: tools.OutcomeToolError}, nil
	}

	timeout := t.srv.cfg.CallTimeout.Std()
	if timeout <= 0 {
		timeout = config.DefaultMCPCallTimeout
	}
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	sess, err := t.srv.session(cctx)
	if err != nil {
		// Not counted here: the failed reconnect already counted once. Counting
		// per caller would multiply one server failure by the number of tool
		// calls in the turn.
		return "", tools.Outcome{}, fmt.Errorf("mcp server %q: %w", t.srv.Name(), err)
	}

	res, err := sess.CallTool(cctx, &sdk.CallToolParams{Name: t.original, Arguments: args})
	switch {
	case err == nil:
		text, out := RenderResult(res, t.outputSchema)
		return text, out, nil
	case ctx.Err() != nil:
		// The RUN ended (timeout or signal), not this call's own budget.
		return abortedText, tools.Outcome{Kind: tools.OutcomeAborted}, nil
	case errors.Is(cctx.Err(), context.DeadlineExceeded):
		return fmt.Sprintf("%stool call timed out after %s", errorPrefix, timeout), tools.Outcome{Kind: tools.OutcomeTimedOut}, nil
	case isConnectionLoss(err):
		// The request may have reached the server. Mark the session dead so the
		// NEXT call reconnects; do not reconnect-and-retry here, because that
		// would be exactly the double side effect the indeterminate outcome
		// exists to prevent.
		t.srv.markDead(sess)
		t.srv.countError()
		return lostText, tools.Outcome{Kind: tools.OutcomeIndeterminate}, nil
	default:
		// A JSON-RPC error: unknown tool, invalid params, server exception.
		// The server is healthy; this one call failed.
		return errorPrefix + err.Error(), tools.Outcome{Kind: tools.OutcomeToolError}, nil
	}
}

// decodeArgs turns the model's raw argument string into the object MCP expects.
// An empty string means "no arguments", which is not an error: a tool with no
// parameters is legitimate.
func decodeArgs(rawArgs string) (map[string]any, error) {
	if strings.TrimSpace(rawArgs) == "" {
		return nil, nil
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(rawArgs), &args); err != nil {
		return nil, err
	}
	return args, nil
}
