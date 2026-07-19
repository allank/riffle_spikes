// Package stubembedder is a deterministic, model-free placeholder
// Embedder shared by this repo's CLI entrypoints. It exists purely to
// smoke-test a pipeline end-to-end before a real onnx_test-backed
// adapter is wired in, and has no relation to any real embedding model.
package stubembedder

import (
	"context"
	"hash/fnv"
	"strings"
)

const vectorDim = 32

// Embedder is a bag-of-words hash embedding: deterministic, but not a
// real model.
type Embedder struct{}

func (Embedder) Embed(_ context.Context, chunks []string) ([][]float32, error) {
	out := make([][]float32, len(chunks))
	for i, c := range chunks {
		out[i] = hashEmbed(c)
	}
	return out, nil
}

func hashEmbed(text string) []float32 {
	vec := make([]float32, vectorDim)
	for _, word := range strings.Fields(strings.ToLower(text)) {
		h := fnv.New32a()
		_, _ = h.Write([]byte(word))
		vec[h.Sum32()%vectorDim]++
	}
	return vec
}
