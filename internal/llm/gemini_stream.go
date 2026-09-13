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

// gemReadStream reads the SSE body and returns the response the events add up
// to, delivering every answer-text part to sink on the way. usageMetadata is
// cumulative on this wire, so the last one seen is the turn's; the finish
// reason comes on the last candidate that carries one, and a stream that
// ends without one was cut.
//
// CONTRACT: every part is kept exactly as its event sent it - no merging, no
// re-encoding. Google requires signed parts back unmodified and forbids
// merging or splitting them, and a part's signature can ride on a text part
// as well as on a call; the only faithful assembly is the concatenation of
// the streamed parts, which the decoder reads like any multi-part answer
// (text parts concatenate). Live-unverified (#17).
func gemReadStream(body io.Reader, sink func(string)) (gemResponse, error) {
	var (
		parts  []json.RawMessage
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
		if resp.Candidates[0].FinishReason == "" {
			return gemResponse{}, fmt.Errorf("%w: stream ended before the candidate finished", ErrProvider)
		}
		resp.Candidates[0].Content.Parts = gemEncodeParts(parts)
	}
	return resp, nil
}

// gemAppendParts adds one event's parts to the assembly verbatim and sends
// the answer text (not thought text) to the sink.
func gemAppendParts(parts []json.RawMessage, raw json.RawMessage, sink func(string)) ([]json.RawMessage, error) {
	raws, err := gemRawParts(raw)
	if err != nil {
		return parts, fmt.Errorf("%w: decoding stream event: %v", ErrProvider, err)
	}
	for _, raw := range raws {
		var part gemPart
		if err := json.Unmarshal(raw, &part); err != nil {
			return parts, fmt.Errorf("%w: decoding stream event: %v", ErrProvider, err)
		}
		parts = append(parts, raw)
		if !part.Thought && part.FunctionCall == nil && part.Text != "" {
			sink(part.Text)
		}
	}
	return parts, nil
}

// gemRawParts splits a parts array into its raw elements without decoding
// them, so every part can be kept byte for byte.
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

// gemEncodeParts renders the streamed parts as one array, each compacted and
// otherwise untouched. Nil when nothing arrived, which the decoder reads as
// an empty turn.
func gemEncodeParts(parts []json.RawMessage) json.RawMessage {
	if len(parts) == 0 {
		return nil
	}
	var b bytes.Buffer
	b.WriteByte('[')
	for i, p := range parts {
		if i > 0 {
			b.WriteByte(',')
		}
		var compact bytes.Buffer
		if err := json.Compact(&compact, p); err == nil {
			b.Write(compact.Bytes())
		} else {
			b.Write(p)
		}
	}
	b.WriteByte(']')
	return b.Bytes()
}
