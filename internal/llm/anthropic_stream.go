package llm

// This file assembles one Messages API response from its streamed events. It
// is the Anthropic wire's half of ChatStream: the transport (readSSE) and the
// retry loop (chat) live elsewhere, and everything here is about turning
// content_block events back into the content array the non-streaming path
// decodes whole.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

// anEvent is the union of the streaming event shapes. One struct decodes them
// all; the Type says which members are meaningful.
type anEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	// Message is message_start's message: it carries the input-side usage.
	Message *struct {
		Usage *anUsage `json:"usage"`
	} `json:"message"`
	// ContentBlock is content_block_start's block, which arrives with its
	// type and identity (a tool_use's id and name, a redacted_thinking's
	// data) and empty content.
	ContentBlock *struct {
		Type      string `json:"type"`
		Text      string `json:"text"`
		ID        string `json:"id"`
		Name      string `json:"name"`
		Data      string `json:"data"`
		Thinking  string `json:"thinking"`
		Signature string `json:"signature"`
	} `json:"content_block"`
	// Delta is content_block_delta's fragment, or message_delta's stop
	// reason; both spell the member they carry after their own type.
	Delta *struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		PartialJSON string `json:"partial_json"`
		Thinking    string `json:"thinking"`
		Signature   string `json:"signature"`
		StopReason  string `json:"stop_reason"`
	} `json:"delta"`
	// Usage is message_delta's cumulative output-side usage.
	Usage *anUsage `json:"usage"`
	// Error is the mid-stream error event's payload.
	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// anStreamBlock is one content block under assembly.
type anStreamBlock struct {
	kind      string
	text      bytes.Buffer // text and thinking blocks
	input     bytes.Buffer // tool_use: the partial_json fragments
	id, name  string
	data      string // redacted_thinking
	signature string
}

// anStreamState accumulates the events of one message.
type anStreamState struct {
	blocks  []*anStreamBlock
	byIndex map[int]*anStreamBlock
	resp    anResponse
	usage   anUsage
	sawMsg  bool
	stopped bool
}

// anReadStream reads the SSE body and returns the response the events add up
// to, delivering every text_delta to sink on the way. A stream that ends
// without message_stop is a cut connection, reported as a provider error
// rather than assembled into a shorter answer.
func anReadStream(body io.Reader, sink func(string)) (anResponse, error) {
	st := &anStreamState{byIndex: map[int]*anStreamBlock{}}
	err := readSSE(body, func(ev sseEvent) error {
		var e anEvent
		if err := json.Unmarshal([]byte(ev.data), &e); err != nil {
			return fmt.Errorf("%w: decoding stream event: %v", ErrProvider, err)
		}
		return st.apply(e, sink)
	})
	if err != nil {
		return anResponse{}, err
	}
	if !st.sawMsg {
		return anResponse{}, fmt.Errorf("%w: stream carried no message", ErrProvider)
	}
	if !st.stopped {
		return anResponse{}, fmt.Errorf("%w: stream ended before message_stop", ErrProvider)
	}
	st.resp.Content = anEncodeBlocks(st.blocks)
	usage := st.usage
	st.resp.Usage = &usage
	return st.resp, nil
}

// apply folds one event in. ping, content_block_stop and unknown types carry
// nothing to keep - additive on a versioned wire.
func (st *anStreamState) apply(e anEvent, sink func(string)) error {
	switch e.Type {
	case "message_start":
		st.sawMsg = true
		if e.Message != nil && e.Message.Usage != nil {
			st.usage = *e.Message.Usage
		}
	case "content_block_start":
		return st.start(e)
	case "content_block_delta":
		return st.delta(e, sink)
	case "message_delta":
		st.messageDelta(e)
	case "message_stop":
		st.stopped = true
	case "error":
		msg := "unknown error"
		if e.Error != nil {
			msg = e.Error.Type + ": " + e.Error.Message
		}
		return fmt.Errorf("%w: stream error: %s", ErrProvider, msg)
	}
	return nil
}

// start opens a block at its index with the identity the event carries.
func (st *anStreamState) start(e anEvent) error {
	if e.ContentBlock == nil {
		return fmt.Errorf("%w: content_block_start without a block", ErrProvider)
	}
	b := &anStreamBlock{kind: e.ContentBlock.Type, id: e.ContentBlock.ID, name: e.ContentBlock.Name,
		data: e.ContentBlock.Data, signature: e.ContentBlock.Signature}
	b.text.WriteString(e.ContentBlock.Text)
	b.text.WriteString(e.ContentBlock.Thinking)
	st.blocks = append(st.blocks, b)
	st.byIndex[e.Index] = b
	return nil
}

// delta appends one fragment to its block; only text_delta reaches the sink.
func (st *anStreamState) delta(e anEvent, sink func(string)) error {
	b, ok := st.byIndex[e.Index]
	if !ok || e.Delta == nil {
		return fmt.Errorf("%w: content_block_delta for an unknown block %d", ErrProvider, e.Index)
	}
	switch e.Delta.Type {
	case "text_delta":
		b.text.WriteString(e.Delta.Text)
		if e.Delta.Text != "" {
			sink(e.Delta.Text)
		}
	case "thinking_delta":
		b.text.WriteString(e.Delta.Thinking)
	case "signature_delta":
		b.signature += e.Delta.Signature
	case "input_json_delta":
		b.input.WriteString(e.Delta.PartialJSON)
	}
	// Any other delta type is ignored: additive on a versioned wire.
	return nil
}

// messageDelta records the stop reason and the cumulative output usage. The
// output count of the final message_delta is the turn's; the input side stays
// what message_start said unless this event repeats it.
func (st *anStreamState) messageDelta(e anEvent) {
	if e.Delta != nil && e.Delta.StopReason != "" {
		st.resp.StopReason = e.Delta.StopReason
	}
	if e.Usage == nil {
		return
	}
	st.usage.OutputTokens = e.Usage.OutputTokens
	if e.Usage.InputTokens != 0 {
		st.usage.InputTokens = e.Usage.InputTokens
	}
	if e.Usage.CacheReadInputTokens != 0 || e.Usage.CacheCreationInputTokens != 0 {
		st.usage.CacheReadInputTokens = e.Usage.CacheReadInputTokens
		st.usage.CacheCreationInputTokens = e.Usage.CacheCreationInputTokens
	}
}

// anEncodeBlocks renders the assembled blocks as the content array the API
// would have sent whole, in order: text, tool_use (its input parsed from the
// partial_json fragments, "{}" when they add up to nothing), thinking (text
// plus signature) and redacted_thinking (data). A block of a type this client
// does not assemble is rendered with its type alone, so a new block type
// neither breaks the array nor is silently dropped.
func anEncodeBlocks(blocks []*anStreamBlock) json.RawMessage {
	var b bytes.Buffer
	b.WriteByte('[')
	for i, blk := range blocks {
		if i > 0 {
			b.WriteByte(',')
		}
		switch blk.kind {
		case blockText:
			fmt.Fprintf(&b, `{"type":"text","text":%s}`, mustJSON(blk.text.String()))
		case blockToolUse:
			input := json.RawMessage(blk.input.Bytes())
			if len(bytes.TrimSpace(input)) == 0 || !json.Valid(input) {
				// The tool layer is where a malformed argument string fails,
				// exactly as on the OpenAI wire; an empty one is the API's own
				// spelling of "no arguments".
				if len(bytes.TrimSpace(input)) == 0 {
					input = json.RawMessage("{}")
				} else {
					input = mustJSON(blk.input.String())
				}
			}
			fmt.Fprintf(&b, `{"type":"tool_use","id":%s,"name":%s,"input":%s}`, mustJSON(blk.id), mustJSON(blk.name), compactJSONObject(input))
		case blockThinking:
			fmt.Fprintf(&b, `{"type":"thinking","thinking":%s,"signature":%s}`, mustJSON(blk.text.String()), mustJSON(blk.signature))
		case blockRedactedThinking:
			fmt.Fprintf(&b, `{"type":"redacted_thinking","data":%s}`, mustJSON(blk.data))
		default:
			fmt.Fprintf(&b, `{"type":%s}`, mustJSON(blk.kind))
		}
	}
	b.WriteByte(']')
	return b.Bytes()
}
