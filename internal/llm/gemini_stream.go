package llm

// This file assembles one generateContent response from its streamed events.
// It is the Gemini wire's half of ChatStream: the transport (readSSE) and the
// retry loop (chat) live elsewhere, and everything here is about turning the
// per-event parts back into the array the non-streaming path decodes whole.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

// gemStreamPart is one part under assembly. Text parts of one kind (answer
// or thought) that arrive consecutively merge into one; a functionCall part
// is kept as the raw object the event sent, so a signed call goes back as it
// came.
type gemStreamPart struct {
	text      bytes.Buffer
	thought   bool
	signature string
	raw       json.RawMessage // a functionCall (or any non-text) part, verbatim
}

// gemReadStream reads the SSE body and returns the response the events add up
// to, delivering every answer-text part to sink on the way. usageMetadata is
// cumulative on this wire, so the last one seen is the turn's; the finish
// reason likewise comes on the last candidate that carries one.
func gemReadStream(body io.Reader, sink func(string)) (gemResponse, error) {
	var (
		parts  []*gemStreamPart
		resp   gemResponse
		events int
	)
	err := readSSE(body, func(ev sseEvent) error {
		var chunk gemResponse
		if err := json.Unmarshal([]byte(ev.data), &chunk); err != nil {
			return fmt.Errorf("%w: decoding stream event: %v", ErrProvider, err)
		}
		events++
		if chunk.UsageMetadata != nil {
			resp.UsageMetadata = chunk.UsageMetadata
		}
		if chunk.PromptFeedback != nil {
			resp.PromptFeedback = chunk.PromptFeedback
		}
		if len(chunk.Candidates) == 0 {
			return nil
		}
		candidate := chunk.Candidates[0]
		if candidate.FinishReason != "" {
			resp.Candidates = []gemCandidate{{FinishReason: candidate.FinishReason}}
		} else if len(resp.Candidates) == 0 {
			resp.Candidates = []gemCandidate{{}}
		}
		var err error
		parts, err = gemAppendParts(parts, candidate.Content.Parts, sink)
		return err
	})
	if err != nil {
		return gemResponse{}, err
	}
	if events == 0 {
		return gemResponse{}, fmt.Errorf("%w: stream carried no event", ErrProvider)
	}
	if len(resp.Candidates) > 0 {
		resp.Candidates[0].Content.Parts = gemEncodeParts(parts)
	}
	return resp, nil
}

// gemAppendParts folds one event's parts into the assembly: a call part is
// kept verbatim, a text part joins the previous text part of the same kind or
// opens a new one, and answer text (not thought text) goes to the sink.
func gemAppendParts(parts []*gemStreamPart, raw json.RawMessage, sink func(string)) ([]*gemStreamPart, error) {
	raws, err := gemRawParts(raw)
	if err != nil {
		return parts, fmt.Errorf("%w: decoding stream event: %v", ErrProvider, err)
	}
	for _, raw := range raws {
		var part gemPart
		if err := json.Unmarshal(raw, &part); err != nil {
			return parts, fmt.Errorf("%w: decoding stream event: %v", ErrProvider, err)
		}
		if part.FunctionCall != nil || part.FunctionResponse != nil {
			parts = append(parts, &gemStreamPart{raw: raw})
			continue
		}
		last := lastTextPart(parts)
		if last == nil || last.thought != part.Thought {
			last = &gemStreamPart{thought: part.Thought}
			parts = append(parts, last)
		}
		last.text.WriteString(part.Text)
		if part.ThoughtSignature != "" {
			last.signature = part.ThoughtSignature
		}
		if !part.Thought && part.Text != "" {
			sink(part.Text)
		}
	}
	return parts, nil
}

// lastTextPart returns the most recent part when it is a text part, so a
// text fragment can join it; nil when the previous part was a call.
func lastTextPart(parts []*gemStreamPart) *gemStreamPart {
	if len(parts) == 0 {
		return nil
	}
	last := parts[len(parts)-1]
	if last.raw != nil {
		return nil
	}
	return last
}

// gemRawParts splits a parts array into its raw elements without decoding
// them, so a functionCall part can be kept byte for byte.
func gemRawParts(raw json.RawMessage) ([]json.RawMessage, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, nil
	}
	var parts []json.RawMessage
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, fmt.Errorf("parts: %w", err)
	}
	return parts, nil
}

// gemEncodeParts renders the assembled parts as the array the API would have
// sent whole: text parts with their thought flag and signature, call parts
// verbatim. Nil when nothing arrived, which the decoder reads as an empty
// turn.
func gemEncodeParts(parts []*gemStreamPart) json.RawMessage {
	if len(parts) == 0 {
		return nil
	}
	var b bytes.Buffer
	b.WriteByte('[')
	for i, p := range parts {
		if i > 0 {
			b.WriteByte(',')
		}
		if p.raw != nil {
			var compact bytes.Buffer
			if err := json.Compact(&compact, p.raw); err == nil {
				b.Write(compact.Bytes())
			} else {
				b.Write(p.raw)
			}
			continue
		}
		encoded, _ := json.Marshal(gemPart{Text: p.text.String(), Thought: p.thought, ThoughtSignature: p.signature})
		b.Write(encoded)
	}
	b.WriteByte(']')
	return b.Bytes()
}
