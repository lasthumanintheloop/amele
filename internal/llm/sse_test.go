package llm

import (
	"errors"
	"strings"
	"testing"
)

func TestReadSSE(t *testing.T) {
	body := "event: message_start\ndata: {\"a\":1}\n\n" +
		": keep-alive\n" +
		"data: first\ndata: second\n\n" +
		"id: 7\nretry: 100\ndata:{\"b\":2}\r\n\r\n" +
		"data: tail without a blank line"
	var got []sseEvent
	err := readSSE(strings.NewReader(body), func(ev sseEvent) error {
		got = append(got, ev)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []sseEvent{
		{event: "message_start", data: `{"a":1}`},
		{data: "first\nsecond"},
		{data: `{"b":2}`},
		{data: "tail without a blank line"},
	}
	if len(got) != len(want) {
		t.Fatalf("events = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("event %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestReadSSEStopsWhenTheCallbackSaysSo(t *testing.T) {
	stop := errors.New("stop")
	calls := 0
	err := readSSE(strings.NewReader("data: 1\n\ndata: 2\n\n"), func(sseEvent) error {
		calls++
		return stop
	})
	if !errors.Is(err, stop) || calls != 1 {
		t.Fatalf("err = %v, calls = %d", err, calls)
	}
}

func TestReadSSEIsBounded(t *testing.T) {
	huge := strings.Repeat("data: x\n\n", maxResponseBody/8+1)
	err := readSSE(strings.NewReader(huge), func(sseEvent) error { return nil })
	if !errors.Is(err, ErrProvider) || !strings.Contains(err.Error(), "exceeded") {
		t.Fatalf("err = %v, want the bound to be reported", err)
	}
}
