package sidecaradapter

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"
)

// fakeServer simulates the Rust sidecar's protocol.rs behavior: read one
// ndjson request line, write back whatever handle returns, loop. It lets
// tests exercise Adapter's request/response orchestration without the
// real compiled binary or model weights.
type fakeServer struct {
	requests []wireRequest
	handle   func(wireRequest) string // returns the raw response line (no trailing newline)
}

// startFakeServer returns both the Adapter under test and the
// fakeServer backing it, so tests can inspect srv.requests — reading it
// after an Embed() call returns is safe without extra synchronization,
// since the goroutine below always appends a request before writing its
// response, and Embed() doesn't return until it has read that response.
func startFakeServer(t *testing.T, handle func(wireRequest) string) (*Adapter, *fakeServer) {
	t.Helper()

	reqR, reqW := io.Pipe()
	respR, respW := io.Pipe()

	srv := &fakeServer{handle: handle}
	done := make(chan struct{})
	go func() {
		defer close(done)
		scanner := bufio.NewScanner(reqR)
		for scanner.Scan() {
			var req wireRequest
			if err := json.Unmarshal(scanner.Bytes(), &req); err != nil {
				return
			}
			srv.requests = append(srv.requests, req)
			line := srv.handle(req)
			if _, err := respW.Write([]byte(line + "\n")); err != nil {
				return
			}
		}
	}()

	closeFn := func() error {
		err := reqW.Close()
		<-done
		return err
	}

	return newFromIO(reqW, respR, closeFn), srv
}

func TestEmbedSendsRequestAndParsesVectors(t *testing.T) {
	adapter, _ := startFakeServer(t, func(req wireRequest) string {
		out := make([][]float32, len(req.Chunks))
		for i := range req.Chunks {
			out[i] = []float32{float32(i), 0.5}
		}
		resp, _ := json.Marshal(wireResponse{Vectors: out})
		return string(resp)
	})
	defer adapter.Close()

	got, err := adapter.Embed(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	want := [][]float32{{0, 0.5}, {1, 0.5}}
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if len(got[i]) != len(want[i]) || got[i][0] != want[i][0] || got[i][1] != want[i][1] {
			t.Errorf("got[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestEmbedPropagatesSidecarError(t *testing.T) {
	adapter, _ := startFakeServer(t, func(req wireRequest) string {
		resp, _ := json.Marshal(wireResponse{Error: "tokenizing: boom"})
		return string(resp)
	})
	defer adapter.Close()

	_, err := adapter.Embed(context.Background(), []string{"a"})
	if err == nil {
		t.Fatal("Embed() error = nil, want non-nil when the sidecar reports an error")
	}
	if got := err.Error(); !strings.Contains(got, "tokenizing: boom") {
		t.Errorf("Embed() error = %q, want it to contain the sidecar's reported message", got)
	}
}

func TestEmbedOneRequestPerCall(t *testing.T) {
	adapter, srv := startFakeServer(t, func(req wireRequest) string {
		out := make([][]float32, len(req.Chunks))
		for i := range req.Chunks {
			out[i] = []float32{0}
		}
		resp, _ := json.Marshal(wireResponse{Vectors: out})
		return string(resp)
	})
	defer adapter.Close()

	if _, err := adapter.Embed(context.Background(), []string{"first"}); err != nil {
		t.Fatalf("Embed() #1 error = %v", err)
	}
	if _, err := adapter.Embed(context.Background(), []string{"second", "third"}); err != nil {
		t.Fatalf("Embed() #2 error = %v", err)
	}

	if len(srv.requests) != 2 {
		t.Fatalf("server saw %d requests, want 2 (one per Embed call)", len(srv.requests))
	}
	if len(srv.requests[0].Chunks) != 1 || srv.requests[0].Chunks[0] != "first" {
		t.Errorf("request 1 = %v, want [\"first\"]", srv.requests[0].Chunks)
	}
	if len(srv.requests[1].Chunks) != 2 || srv.requests[1].Chunks[0] != "second" || srv.requests[1].Chunks[1] != "third" {
		t.Errorf("request 2 = %v, want [\"second\", \"third\"]", srv.requests[1].Chunks)
	}
}

func TestEmbedHandlesMalformedResponse(t *testing.T) {
	adapter, _ := startFakeServer(t, func(req wireRequest) string {
		return "not json"
	})
	defer adapter.Close()

	if _, err := adapter.Embed(context.Background(), []string{"a"}); err == nil {
		t.Error("Embed() error = nil, want non-nil for a malformed response line")
	}
}

func TestCloseInvokesCloseFn(t *testing.T) {
	called := false
	adapter := newFromIO(io.Discard, discardReader{}, func() error {
		called = true
		return nil
	})

	if err := adapter.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if !called {
		t.Error("Close() did not invoke the provided close function")
	}
}

// discardReader is an io.Reader that always reports EOF, used where a
// test needs a reader but never actually reads from it.
type discardReader struct{}

func (discardReader) Read(p []byte) (int, error) { return 0, io.EOF }
