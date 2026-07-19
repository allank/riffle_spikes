// Package onnxadapter wraps onnx_test's ONNX Runtime BGE-small path —
// the reference implementation every other spike's embedding output
// gets compared against — to satisfy goldeneval.Embedder.
package onnxadapter

import (
	"context"
	"fmt"

	"github.com/allank/onnx_test/bge_bench/onnxpath"

	goldeneval "github.com/allank/riffle_spikes/golden_eval"
)

// Adapter wraps a single ONNX Runtime session, loaded once at
// construction. Callers must call Close when done to release it.
type Adapter struct {
	embedder *onnxpath.Embedder
}

var _ goldeneval.Embedder = (*Adapter)(nil)

// New loads modelPath (a model.onnx file), tokenizerPath (a
// tokenizer.json file), and initializes ONNX Runtime from the shared
// library at libPath. All three are onnx_test-local assets not
// committed to either repo, so callers must supply real paths on disk.
func New(modelPath, tokenizerPath, libPath string) (*Adapter, error) {
	embedder, err := onnxpath.New(modelPath, tokenizerPath, libPath)
	if err != nil {
		return nil, fmt.Errorf("onnxadapter: loading model: %w", err)
	}
	return &Adapter{embedder: embedder}, nil
}

// Embed runs each chunk through the ONNX Runtime session. onnxpath's
// Embedder tokenizes internally and already returns L2-normalized
// float32 vectors, so each result passes through unchanged.
func (a *Adapter) Embed(_ context.Context, chunks []string) ([][]float32, error) {
	out := make([][]float32, len(chunks))
	for i, chunk := range chunks {
		vec, err := a.embedder.Embed(chunk)
		if err != nil {
			return nil, fmt.Errorf("onnxadapter: embedding chunk %d: %w", i, err)
		}
		out[i] = vec
	}
	return out, nil
}

// Close releases the underlying ONNX Runtime session.
func (a *Adapter) Close() error {
	return a.embedder.Close()
}
