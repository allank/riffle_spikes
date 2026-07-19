// Package goldeneval scores embedding approaches against a fixed,
// hand-authored corpus so every spike is measured against the same
// retrieval-quality bar (PRD Section 6).
package goldeneval

import "context"

// Embedder is the contract every spike's embedding approach implements.
// It must tolerate an out-of-process implementation (Spike 3's Rust
// sidecar embeds over stdio, not in-process), so the interface is
// batch-in, vectors-out with no assumption about where the work happens.
type Embedder interface {
	Embed(ctx context.Context, chunks []string) ([][]float32, error)
}
