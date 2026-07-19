package spike2measure

import (
	"context"
	"fmt"
	"time"

	goldeneval "github.com/allank/riffle_spikes/golden_eval"
)

// warmSampleCount is how many single-chunk Embed calls after the first
// (cold) one are averaged into Stats.LatencyWarm.
const warmSampleCount = 10

// Report is a full benchmark run's output.
type Report struct {
	Stats Stats

	// ColdStart is wall-clock time from immediately before the
	// embedder's construction to immediately after its first
	// successful Embed call returns — the only measurement that
	// includes model/tokenizer loading.
	ColdStart time.Duration
}

// Run measures throughput, cold/warm latency, and cold start for
// embedder against corpus.
//
// constructionDuration is however long the caller's own call to
// construct embedder took (e.g. puregoadapter.New). Run has no way to
// measure that itself, since it receives an already-constructed
// Embedder — the caller must time construction and pass it in for
// ColdStart to be accurate. Pass 0 if embedder had no real construction
// cost (e.g. a stub).
func Run(ctx context.Context, embedder goldeneval.Embedder, corpus []string, constructionDuration time.Duration) (Report, error) {
	if len(corpus) == 0 {
		return Report{}, fmt.Errorf("spike2measure: corpus must be non-empty")
	}

	// Single-chunk calls run before the batched throughput call, so the
	// very first call ever made to embedder is the one measured as cold
	// latency, matching ColdStart's "time to first successful embedding"
	// definition. If the batched call ran first instead, a real
	// embedder's lazy-loading cost would fire there and be invisible to
	// ColdStart.
	singleChunkDurations := make([]time.Duration, 0, warmSampleCount+1)
	for i := range warmSampleCount + 1 {
		chunk := corpus[i%len(corpus)]
		start := time.Now()
		if _, err := embedder.Embed(ctx, []string{chunk}); err != nil {
			return Report{}, fmt.Errorf("single-chunk embed %d: %w", i, err)
		}
		singleChunkDurations = append(singleChunkDurations, time.Since(start))
	}

	throughputStart := time.Now()
	if _, err := embedder.Embed(ctx, corpus); err != nil {
		return Report{}, fmt.Errorf("throughput embed: %w", err)
	}
	throughputDuration := time.Since(throughputStart)

	stats, err := ComputeStats(len(corpus), throughputDuration, singleChunkDurations)
	if err != nil {
		return Report{}, err
	}

	return Report{
		Stats:     stats,
		ColdStart: constructionDuration + stats.LatencyCold,
	}, nil
}
