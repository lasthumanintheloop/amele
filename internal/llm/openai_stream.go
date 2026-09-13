package llm

// This file assembles one chat completion from its streamed chunks. It is the
// OpenAI wire's half of ChatStream: the transport (readSSE) and the retry loop
// (chat) live elsewhere, and everything here is about turning deltas back into
// the one message the non-streaming path decodes whole.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
)

// oaChunk is one streamed chat.completion.chunk. Only the delta fields amele
// assembles are named; the rest of the object is ignored, as on the
// non-streaming path.
type oaChunk struct {
	Choices []struct {
		Delta struct {
			Role    string `json:"role"`
			Content string `json:"content"`
			// ToolCalls arrive as fragments addressed by index: the first
			// fragment of a call carries its id and name, every fragment
			// carries a slice of the arguments string.
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
			// The three reasoning spellings, as deltas: reasoning_content and
			// the bare reasoning are string fragments, reasoning_details a
			// list of item fragments (see reasoningDetails).
			ReasoningContent string            `json:"reasoning_content"`
			ReasoningDetails []json.RawMessage `json:"reasoning_details"`
			Reasoning        string            `json:"reasoning"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	// Usage rides on the final chunk when stream_options asked for it.
	Usage *oaUsage `json:"usage"`
	// Error is the mid-stream failure shape OpenRouter documents: a 200 that
	// carries an error object in a chunk instead of a choice.
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// sseDone is the OpenAI wire's stream terminator.
const sseDone = "[DONE]"

// maxStreamedToolCalls bounds the tool_calls index a chunk may address. No
// turn asks for anywhere near it; it exists so a hostile index cannot size an
// allocation.
const maxStreamedToolCalls = 256

// oaStreamState accumulates the chunks of one completion.
type oaStreamState struct {
	content          bytes.Buffer
	reasoningContent bytes.Buffer
	reasoning        bytes.Buffer
	details          reasoningDetails
	calls            []oaToolCall
	finish           string
	usage            *oaUsage
	sawChoice        bool
	// done records the [DONE] terminator. A stream that neither reached it
	// nor carried a finish reason was cut, and a cut answer must not pass
	// as a finished one.
	done bool
}

// readStream reads the SSE body and returns the assembled message, the last
// finish reason and the usage, delivering every content delta to sink on the
// way. A stream that ends with no choice at all is the streaming shape of
// "response has no choices".
func (c *OpenAIClient) readStream(body interface{ Read([]byte) (int, error) }, sink func(string)) (oaMessage, string, *oaUsage, error) {
	var st oaStreamState
	err := readSSE(body, func(ev sseEvent) error {
		if ev.data == sseDone {
			st.done = true
			return nil
		}
		var chunk oaChunk
		if err := json.Unmarshal([]byte(ev.data), &chunk); err != nil {
			return fmt.Errorf("%w: decoding stream chunk: %v", ErrProvider, err)
		}
		if chunk.Error != nil {
			return fmt.Errorf("%w: stream error: %s", ErrProvider, chunk.Error.Message)
		}
		return st.apply(chunk, sink)
	})
	if err != nil {
		return oaMessage{}, "", nil, err
	}
	if !st.sawChoice {
		return oaMessage{}, "", nil, fmt.Errorf("%w: response has no choices", ErrProvider)
	}
	if !st.done && st.finish == "" {
		// EOF between two well-formed chunks: the connection was cut. The
		// text so far is not an answer, and a tool turn assembled from it
		// could dispatch a call the model never finished.
		return oaMessage{}, "", nil, fmt.Errorf("%w: stream ended before the completion finished", ErrProvider)
	}
	msg := oaMessage{Role: RoleAssistant, Content: st.content.String(), ToolCalls: st.calls}
	// The carriers are re-encoded values: each is the same JSON the provider
	// would have sent whole. A carrier is only set when its text arrived at
	// all, so a stream with no reasoning yields the same nil fields as a
	// response with none.
	if st.reasoningContent.Len() > 0 {
		msg.ReasoningContent = mustJSON(st.reasoningContent.String())
	}
	if st.reasoning.Len() > 0 {
		msg.Reasoning = mustJSON(st.reasoning.String())
	}
	if len(st.details.items) > 0 {
		msg.ReasoningDetails = st.details.encode()
	}
	return msg, st.finish, st.usage, nil
}

// apply folds one chunk into the state.
func (st *oaStreamState) apply(chunk oaChunk, sink func(string)) error {
	if chunk.Usage != nil {
		st.usage = chunk.Usage
	}
	if len(chunk.Choices) == 0 {
		return nil
	}
	st.sawChoice = true
	choice := chunk.Choices[0]
	if choice.FinishReason != "" {
		st.finish = choice.FinishReason
	}
	d := choice.Delta
	if d.Content != "" {
		st.content.WriteString(d.Content)
		sink(d.Content)
	}
	st.reasoningContent.WriteString(d.ReasoningContent)
	st.reasoning.WriteString(d.Reasoning)
	for _, item := range d.ReasoningDetails {
		if err := st.details.add(item); err != nil {
			return err
		}
	}
	for _, tc := range d.ToolCalls {
		// The index is provider-controlled input: a negative one would panic
		// and an absurd one would allocate past every bound the body has.
		if tc.Index < 0 || tc.Index >= maxStreamedToolCalls {
			return fmt.Errorf("%w: tool call index %d out of range", ErrProvider, tc.Index)
		}
		for len(st.calls) <= tc.Index {
			st.calls = append(st.calls, oaToolCall{Type: "function"})
		}
		call := &st.calls[tc.Index]
		if tc.ID != "" {
			call.ID = tc.ID
		}
		if tc.Function.Name != "" {
			call.Function.Name = tc.Function.Name
		}
		call.Function.Arguments += tc.Function.Arguments
	}
	return nil
}

// mustJSON encodes a string as a JSON string. Encoding a Go string cannot
// fail, so the error is dropped by construction rather than by neglect.
func mustJSON(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

// reasoningDetails accumulates OpenRouter's streamed reasoning_details items.
//
// Each chunk carries a list of item fragments; a fragment names its item by
// `index`, and an item's text arrives across several fragments. Items are
// therefore merged by index - a fragment for an index already seen appends
// its string-valued fields (text, summary, data, signature) onto the item's
// and fills in any key the item lacked - and kept in first-seen order, with
// the key order of their first fragment, so the assembled array is
// deterministic. A fragment without an index is an item of its own.
//
// Live-unverified (#17): the fragment shape comes from OpenRouter's
// documentation of the field, not from a stream amele has recorded.
type reasoningDetails struct {
	items []*orderedObject
	byIdx map[int]*orderedObject
}

// orderedObject is a JSON object as an ordered list of members.
type orderedObject struct {
	keys   []string
	values map[string]json.RawMessage
}

func (d *reasoningDetails) add(raw json.RawMessage) error {
	obj, err := decodeOrdered(raw)
	if err != nil {
		return fmt.Errorf("%w: decoding reasoning_details: %v", ErrProvider, err)
	}
	idx, hasIdx := obj.index()
	if hasIdx {
		if existing, ok := d.byIdx[idx]; ok {
			existing.merge(obj)
			return nil
		}
		if d.byIdx == nil {
			d.byIdx = map[int]*orderedObject{}
		}
		d.byIdx[idx] = obj
	}
	d.items = append(d.items, obj)
	return nil
}

// encode renders the items as one JSON array.
func (d *reasoningDetails) encode() json.RawMessage {
	var b bytes.Buffer
	b.WriteByte('[')
	for i, item := range d.items {
		if i > 0 {
			b.WriteByte(',')
		}
		item.encode(&b)
	}
	b.WriteByte(']')
	return b.Bytes()
}

// decodeOrdered decodes one JSON object keeping its key order. A token stream
// is used because encoding/json's map decoding loses the order.
func decodeOrdered(raw json.RawMessage) (*orderedObject, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '{' {
		return nil, fmt.Errorf("item is not an object")
	}
	obj := &orderedObject{values: map[string]json.RawMessage{}}
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return nil, err
		}
		key, ok := keyTok.(string)
		if !ok {
			return nil, fmt.Errorf("object key is not a string")
		}
		var value json.RawMessage
		if err := dec.Decode(&value); err != nil {
			return nil, err
		}
		obj.keys = append(obj.keys, key)
		obj.values[key] = value
	}
	if _, err := dec.Token(); err != nil { // the closing brace
		return nil, err
	}
	return obj, nil
}

// index returns the item's index member when it carries an integer one.
func (o *orderedObject) index() (int, bool) {
	raw, ok := o.values["index"]
	if !ok {
		return 0, false
	}
	n, err := strconv.Atoi(string(bytes.TrimSpace(raw)))
	if err != nil {
		return 0, false
	}
	return n, true
}

// merge folds a later fragment of the same item in: the content members
// (text, summary, data) are appended, every other member - type, id, format,
// signature - is set when absent and otherwise kept as first seen, so a
// repeated type label does not double.
func (o *orderedObject) merge(frag *orderedObject) {
	for _, key := range frag.keys {
		value := frag.values[key]
		existing, present := o.values[key]
		if !present {
			o.keys = append(o.keys, key)
			o.values[key] = value
			continue
		}
		if !streamedContentKey(key) {
			// A placeholder (null or "") in an early fragment yields to the
			// real value of a later one: OpenRouter sends the signature
			// null first and filled in last.
			if isJSONPlaceholder(existing) && !isJSONPlaceholder(value) {
				o.values[key] = value
			}
			continue
		}
		var a, b string
		if json.Unmarshal(existing, &a) == nil && json.Unmarshal(value, &b) == nil {
			o.values[key] = mustJSON(a + b)
		}
	}
}

// isJSONPlaceholder reports whether a member value is null or the empty
// string - the two ways a fragment says "not yet".
func isJSONPlaceholder(raw json.RawMessage) bool {
	t := bytes.TrimSpace(raw)
	return len(t) == 0 || bytes.Equal(t, []byte("null")) || bytes.Equal(t, []byte(`""`))
}

// streamedContentKey names the reasoning_details members that arrive in
// fragments: the reasoning text, its summary, and the encrypted data blob.
func streamedContentKey(key string) bool {
	return key == "text" || key == "summary" || key == "data"
}

// encode writes the object with its members in order.
func (o *orderedObject) encode(b *bytes.Buffer) {
	b.WriteByte('{')
	for i, key := range o.keys {
		if i > 0 {
			b.WriteByte(',')
		}
		b.Write(mustJSON(key))
		b.WriteByte(':')
		var compact bytes.Buffer
		if err := json.Compact(&compact, o.values[key]); err == nil {
			b.Write(compact.Bytes())
		} else {
			b.Write(o.values[key])
		}
	}
	b.WriteByte('}')
}
