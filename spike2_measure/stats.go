// Package spike2measure benchmarks any goldeneval.Embedder against the
// PRD's Section 7 timing criteria: throughput, latency (cold/warm), and
// cold start. It's embedder-agnostic by design, so the same measurement
// code works unchanged once Spike 3's sidecar adapter exists.
package spike2measure

import (
	"fmt"
	"time"
)

// Stats is the throughput/latency timing criteria computed from a
// benchmark run's raw measurements.
type Stats struct {
	ThroughputChunksPerSec float64
	LatencyCold            time.Duration
	LatencyWarm            time.Duration
}

// ComputeStats aggregates raw measurements into Stats.
//
// throughputDuration is the wall-clock time of one batched Embed call
// across corpusSize chunks; throughput is corpusSize divided by that
// duration. singleChunkDurations are the durations of sequential
// single-chunk Embed calls: the first element is the cold (first-call)
// latency, and the mean of any remaining elements becomes LatencyWarm.
// singleChunkDurations must have at least one element (the cold
// sample).
func ComputeStats(corpusSize int, throughputDuration time.Duration, singleChunkDurations []time.Duration) (Stats, error) {
	if len(singleChunkDurations) == 0 {
		return Stats{}, fmt.Errorf("spike2measure: ComputeStats requires at least one single-chunk duration (the cold-latency sample)")
	}

	stats := Stats{LatencyCold: singleChunkDurations[0]}

	if corpusSize > 0 && throughputDuration > 0 {
		stats.ThroughputChunksPerSec = float64(corpusSize) / throughputDuration.Seconds()
	}

	warmSamples := singleChunkDurations[1:]
	if len(warmSamples) > 0 {
		var sum time.Duration
		for _, d := range warmSamples {
			sum += d
		}
		stats.LatencyWarm = sum / time.Duration(len(warmSamples))
	}

	return stats, nil
}
