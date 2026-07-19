// Package puregoadapter wraps onnx_test's pure-Go BGE-small path
// (puregopath.Model + tokenizer.Tokenizer) — the approach ADR-0002
// adopted — to satisfy goldeneval.Embedder.
package puregoadapter

import (
	"context"
	"fmt"

	"github.com/allank/onnx_test/bge_bench/puregopath"
	"github.com/allank/onnx_test/bge_bench/tokenizer"

	goldeneval "github.com/allank/riffle_spikes/golden_eval"
)

// maxSeqLen matches onnx_test's own bge_bench usage (main.go: tok.Encode(s, 512)).
const maxSeqLen = 512

// Adapter loads the model weights and tokenizer once at construction, so
// repeated Embed calls reuse them rather than reloading per call.
type Adapter struct {
	model *puregopath.Model
	tok   *tokenizer.Tokenizer
}

var _ goldeneval.Embedder = (*Adapter)(nil)

// New loads modelPath (a model.safetensors file) and tokenizerPath (a
// tokenizer.json file). Both are onnx_test-local assets not committed to
// either repo, so callers must supply real paths on disk.
func New(modelPath, tokenizerPath string) (*Adapter, error) {
	model, err := puregopath.Load(modelPath)
	if err != nil {
		return nil, fmt.Errorf("puregoadapter: loading model: %w", err)
	}

	tok, err := tokenizer.LoadFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("puregoadapter: loading tokenizer: %w", err)
	}

	return &Adapter{model: model, tok: tok}, nil
}

// Embed tokenizes each chunk with the BGE-small WordPiece tokenizer and
// runs the pure-Go forward pass, converting its float64 output to the
// Embedder interface's float32 vectors.
func (a *Adapter) Embed(_ context.Context, chunks []string) ([][]float32, error) {
	out := make([][]float32, len(chunks))
	for i, chunk := range chunks {
		enc := a.tok.Encode(chunk, maxSeqLen)
		vec64 := a.model.Embed(enc.InputIDs)

		vec32 := make([]float32, len(vec64))
		for j, v := range vec64 {
			vec32[j] = float32(v)
		}
		out[i] = vec32
	}
	return out, nil
}
