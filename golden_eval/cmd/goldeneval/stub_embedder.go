package main

import (
	"context"
	"hash/fnv"
	"strings"
)

const stubVectorDim = 32

// stubEmbedder is a placeholder for this ticket only: a deterministic
// bag-of-words hash embedding with no relation to any real model. It
// exists purely to prove the golden eval pipeline runs end-to-end
// (load -> embed -> rank -> score -> print) before real onnx_test-backed
// adapters exist (riffle_spikes#1's later tickets: puregoadapter,
// onnxadapter).
type stubEmbedder struct{}

func (stubEmbedder) Embed(_ context.Context, chunks []string) ([][]float32, error) {
	out := make([][]float32, len(chunks))
	for i, c := range chunks {
		out[i] = hashEmbed(c)
	}
	return out, nil
}

func hashEmbed(text string) []float32 {
	vec := make([]float32, stubVectorDim)
	for _, word := range strings.Fields(strings.ToLower(text)) {
		h := fnv.New32a()
		_, _ = h.Write([]byte(word))
		vec[h.Sum32()%stubVectorDim]++
	}
	return vec
}
