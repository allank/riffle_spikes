package bgeembed

import "math"

const hiddenSize = 384

// clsAndNormalize takes the first token's ([CLS]) hidden state and
// L2-normalizes it. BGE's model card is explicit that this is the
// correct pooling method, not the BERT pooler head. Pure math, no ONNX
// Runtime dependency, so unlike Embedder it doesn't need a per-backend
// copy.
func clsAndNormalize(hidden []float32) []float32 {
	cls := make([]float32, hiddenSize)
	copy(cls, hidden[:hiddenSize])
	var norm float64
	for _, v := range cls {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range cls {
			cls[i] = float32(float64(cls[i]) / norm)
		}
	}
	return cls
}
