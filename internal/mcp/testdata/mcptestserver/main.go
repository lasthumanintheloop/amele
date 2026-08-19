// Command mcptestserver is a throwaway MCP stdio server used by the tests of
// internal/mcp: it exposes echo/env/sleep/big/fail/structured tools and, with
// -spawn-child, forks a grandchild that must die with the process group.
//
// It lives under testdata/ so that it is never linked into the amele binary;
// the tests build it with `go build` into a temporary directory.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// echoIn is the argument object of the echo tool.
type echoIn struct {
	Text string `json:"text"`
}

// envIn is the argument object of the env tool.
type envIn struct {
	Name string `json:"name"`
}

// sleepIn is the argument object of the sleep tool.
type sleepIn struct {
	Ms int `json:"ms"`
}

// bigIn is the argument object of the big tool.
type bigIn struct {
	N int `json:"n"`
}

// structuredOut is the typed result of the structured tool; its Go type is
// what makes the SDK publish an outputSchema for that tool.
type structuredOut struct {
	A int `json:"a"`
}

// text builds the one-content-block result the simple tools return.
func text(s string) *sdk.CallToolResult {
	return &sdk.CallToolResult{Content: []sdk.Content{&sdk.TextContent{Text: s}}}
}

func main() {
	spawnChild := flag.Bool("spawn-child", false, "spawn a long-lived grandchild in the same process group")
	exitOnStart := flag.Bool("exit-on-start", false, "exit(3) immediately instead of serving")
	stderrBytes := flag.Int("stderr-bytes", 0, "write this many bytes of banner noise to stderr before serving")
	flag.Parse()

	if *exitOnStart {
		os.Exit(3)
	}
	if *stderrBytes > 0 {
		// One long line: the reader side must relay it whole, however chatty.
		fmt.Fprintln(os.Stderr, strings.Repeat("e", *stderrBytes))
	}

	if *spawnChild {
		// The grandchild inherits this process's group, which is what the
		// process-group kill test asserts on.
		child := exec.Command("sleep", "300")
		if err := child.Start(); err != nil {
			fmt.Fprintf(os.Stderr, "spawn child: %v\n", err)
			os.Exit(4)
		}
		fmt.Fprintf(os.Stderr, "child=%d\n", child.Process.Pid)
	}

	server := sdk.NewServer(&sdk.Implementation{Name: "mcptestserver", Version: "0"}, nil)

	sdk.AddTool(server, &sdk.Tool{Name: "echo", Description: "echo the text back"},
		func(_ context.Context, _ *sdk.CallToolRequest, in echoIn) (*sdk.CallToolResult, any, error) {
			return text(in.Text), nil, nil
		})

	sdk.AddTool(server, &sdk.Tool{Name: "env", Description: "report one environment variable"},
		func(_ context.Context, _ *sdk.CallToolRequest, in envIn) (*sdk.CallToolResult, any, error) {
			v, ok := os.LookupEnv(in.Name)
			if !ok {
				v = "<unset>"
			}
			return text(v), nil, nil
		})

	sdk.AddTool(server, &sdk.Tool{Name: "sleep", Description: "sleep for n milliseconds"},
		func(ctx context.Context, _ *sdk.CallToolRequest, in sleepIn) (*sdk.CallToolResult, any, error) {
			select {
			case <-time.After(time.Duration(in.Ms) * time.Millisecond):
			case <-ctx.Done():
				return nil, nil, ctx.Err()
			}
			return text("slept"), nil, nil
		})

	sdk.AddTool(server, &sdk.Tool{Name: "big", Description: "return n bytes of payload"},
		func(_ context.Context, _ *sdk.CallToolRequest, in bigIn) (*sdk.CallToolResult, any, error) {
			return text(strings.Repeat("x", in.N)), nil, nil
		})

	sdk.AddTool(server, &sdk.Tool{Name: "fail", Description: "always fail"},
		func(_ context.Context, _ *sdk.CallToolRequest, _ struct{}) (*sdk.CallToolResult, any, error) {
			r := text("boom")
			r.IsError = true
			return r, nil, nil
		})

	sdk.AddTool(server, &sdk.Tool{Name: "structured", Description: "return a structured result"},
		func(_ context.Context, _ *sdk.CallToolRequest, _ struct{}) (*sdk.CallToolResult, structuredOut, error) {
			return nil, structuredOut{A: 1}, nil
		})

	if err := server.Run(context.Background(), &sdk.StdioTransport{}); err != nil {
		fmt.Fprintf(os.Stderr, "serve: %v\n", err)
		os.Exit(5)
	}
}
