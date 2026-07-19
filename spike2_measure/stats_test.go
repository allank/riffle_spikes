package spike2measure

import (
	"testing"
	"time"
)

func TestComputeStats(t *testing.T) {
	// cold = 100ms, warm samples = 10ms, 20ms, 30ms -> mean 20ms.
	durations := []time.Duration{
		100 * time.Millisecond,
		10 * time.Millisecond,
		20 * time.Millisecond,
		30 * time.Millisecond,
	}

	stats, err := ComputeStats(1000, 2*time.Second, durations)
	if err != nil {
		t.Fatalf("ComputeStats() error = %v", err)
	}

	wantThroughput := 500.0 // 1000 chunks / 2 seconds
	if stats.ThroughputChunksPerSec != wantThroughput {
		t.Errorf("ThroughputChunksPerSec = %v, want %v", stats.ThroughputChunksPerSec, wantThroughput)
	}
	if stats.LatencyCold != 100*time.Millisecond {
		t.Errorf("LatencyCold = %v, want 100ms", stats.LatencyCold)
	}
	wantWarm := 20 * time.Millisecond
	if stats.LatencyWarm != wantWarm {
		t.Errorf("LatencyWarm = %v, want %v", stats.LatencyWarm, wantWarm)
	}
}

func TestComputeStatsRequiresAtLeastOneSample(t *testing.T) {
	if _, err := ComputeStats(1000, time.Second, nil); err == nil {
		t.Error("ComputeStats() error = nil, want non-nil for an empty duration slice")
	}
}

func TestComputeStatsSingleSampleHasNoWarmLatency(t *testing.T) {
	stats, err := ComputeStats(1000, time.Second, []time.Duration{50 * time.Millisecond})
	if err != nil {
		t.Fatalf("ComputeStats() error = %v", err)
	}
	if stats.LatencyCold != 50*time.Millisecond {
		t.Errorf("LatencyCold = %v, want 50ms", stats.LatencyCold)
	}
	if stats.LatencyWarm != 0 {
		t.Errorf("LatencyWarm = %v, want 0 (no warm samples)", stats.LatencyWarm)
	}
}

func TestComputeStatsZeroThroughputDurationYieldsZeroThroughput(t *testing.T) {
	stats, err := ComputeStats(1000, 0, []time.Duration{time.Millisecond})
	if err != nil {
		t.Fatalf("ComputeStats() error = %v", err)
	}
	if stats.ThroughputChunksPerSec != 0 {
		t.Errorf("ThroughputChunksPerSec = %v, want 0", stats.ThroughputChunksPerSec)
	}
}

func TestComputeStatsZeroCorpusSizeYieldsZeroThroughput(t *testing.T) {
	stats, err := ComputeStats(0, time.Second, []time.Duration{time.Millisecond})
	if err != nil {
		t.Fatalf("ComputeStats() error = %v", err)
	}
	if stats.ThroughputChunksPerSec != 0 {
		t.Errorf("ThroughputChunksPerSec = %v, want 0", stats.ThroughputChunksPerSec)
	}
}
