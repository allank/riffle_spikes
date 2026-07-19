// Package sidecaradapter wraps Spike 3's Rust sidecar binary
// (spike3_rust_sidecar) — spawned as a long-lived child process and
// communicated with over its ndjson stdio protocol — to satisfy
// goldeneval.Embedder.
package sidecaradapter

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"

	goldeneval "github.com/allank/riffle_spikes/golden_eval"
)

// wireRequest and wireResponse mirror the Rust sidecar's protocol.rs
// types exactly: one JSON line in, one JSON line out.
type wireRequest struct {
	Chunks []string `json:"chunks"`
}

type wireResponse struct {
	Vectors [][]float32 `json:"vectors,omitempty"`
	Error   string      `json:"error,omitempty"`
}

// Adapter communicates with a long-lived sidecar process over its
// stdin/stdout: one ndjson request line written per Embed call, one
// ndjson response line read back.
type Adapter struct {
	writer  io.Writer
	reader  *bufio.Reader
	closeFn func() error
}

var _ goldeneval.Embedder = (*Adapter)(nil)

// New spawns the compiled sidecar binary at binaryPath (built via
// `cargo build --release`, not `cargo run`, to avoid compile overhead
// polluting cold-start measurements) with --model/--tokenizer args,
// wiring its stdin/stdout pipes. The child's stderr is passed through
// to this process's stderr, so sidecar-side errors are visible directly
// rather than silently swallowed.
func New(binaryPath, modelPath, tokenizerPath string) (*Adapter, error) {
	cmd := exec.Command(binaryPath, "--model", modelPath, "--tokenizer", tokenizerPath)
	cmd.Stderr = os.Stderr

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("sidecaradapter: opening stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("sidecaradapter: opening stdout pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("sidecaradapter: starting sidecar process: %w", err)
	}

	closeFn := func() error {
		// cmd.Wait() always runs, even if closing stdin errors, so the
		// process is never left unreaped on that error path.
		closeErr := stdin.Close()
		waitErr := cmd.Wait()
		if closeErr != nil {
			return fmt.Errorf("sidecaradapter: closing stdin: %w", closeErr)
		}
		if waitErr != nil {
			return fmt.Errorf("sidecaradapter: waiting for process: %w", waitErr)
		}
		return nil
	}

	return newFromIO(stdin, stdout, closeFn), nil
}

// newFromIO builds an Adapter directly from a writer/reader pair
// without spawning a process. New wires these to a real subprocess's
// pipes; tests wire them to an in-memory fake server instead.
func newFromIO(w io.Writer, r io.Reader, closeFn func() error) *Adapter {
	return &Adapter{
		writer:  w,
		reader:  bufio.NewReader(r),
		closeFn: closeFn,
	}
}

// Embed writes chunks as one ndjson request line and reads back one
// ndjson response line, returning its vectors or propagating the
// sidecar's reported error. bufio.Reader (not bufio.Scanner) is used
// deliberately: Scanner's default token-size limit is far smaller than
// a full-corpus batch response can be.
func (a *Adapter) Embed(_ context.Context, chunks []string) ([][]float32, error) {
	reqBytes, err := json.Marshal(wireRequest{Chunks: chunks})
	if err != nil {
		return nil, fmt.Errorf("sidecaradapter: encoding request: %w", err)
	}
	reqBytes = append(reqBytes, '\n')

	if _, err := a.writer.Write(reqBytes); err != nil {
		return nil, fmt.Errorf("sidecaradapter: writing request: %w", err)
	}

	// If line is non-empty, any err here (typically io.EOF on a final
	// unterminated line) is deliberately deferred to the JSON decode
	// below rather than surfaced directly — a non-EOF read error on a
	// partial line is rare and will still produce a clear decode error.
	line, err := a.reader.ReadString('\n')
	if line == "" {
		if err != nil {
			return nil, fmt.Errorf("sidecaradapter: reading response: %w", err)
		}
		return nil, fmt.Errorf("sidecaradapter: sidecar closed its output without responding")
	}

	var resp wireResponse
	if err := json.Unmarshal([]byte(line), &resp); err != nil {
		return nil, fmt.Errorf("sidecaradapter: decoding response %q: %w", line, err)
	}
	if resp.Error != "" {
		return nil, fmt.Errorf("sidecaradapter: sidecar reported an error: %s", resp.Error)
	}
	return resp.Vectors, nil
}

// Close closes the sidecar's stdin (so it exits cleanly on EOF, per
// its own read loop) and waits for the process — same teardown
// contract as onnxadapter.Close().
func (a *Adapter) Close() error {
	return a.closeFn()
}
