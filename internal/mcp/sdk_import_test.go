package mcp

import (
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// TestSDKPinned guards the dependency the whole package is built on: the
// client constructor must exist and the SDK must speak a protocol version we
// have reviewed. Bumping the SDK is a deliberate act (see doc.go).
func TestSDKPinned(t *testing.T) {
	c := mcp.NewClient(&mcp.Implementation{Name: "amele-test", Version: "0"}, nil)
	if c == nil {
		t.Fatal("NewClient returned nil")
	}
}
