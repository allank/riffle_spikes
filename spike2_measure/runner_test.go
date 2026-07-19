package spike2measure

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	goldeneval "github.com/allank/riffle_spikes/golden_eval"
)

// fakeEmbedder returns a fixed-size zero vector for every chunk. Timing
// correctness isn't assertable against a fake (real durations vary run
// to run), so these tests check Run's orchestration — call counts,
// error propagation, and Report's structural shape — not real numbers.
type fakeEmbedder struct {
	err error
}

func (f fakeEmbedder) Embed(_ context.Context, chunks []string) ([][]float32, error) {
	if f.err != nil {
		return nil, f.err
	}
	out := make([][]float32, len(chunks))
	for i := range chunks {
		out[i] = []float32{0, 0}
	}
	return out, nil
}

var _ goldeneval.Embedder = fakeEmbedder{}

func TestRunProducesAReportWithFakeEmbedder(t *testing.T) {
	corpus := []string{"a", "b", "c"}
	constructionDuration := 5 * time.Millisecond

	report, err := Run(context.Background(), fakeEmbedder{}, corpus, constructionDuration)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if report.ColdStart < constructionDuration {
		t.Errorf("ColdStart = %v, want >= constructionDuration (%v)", report.ColdStart, constructionDuration)
	}
	if report.Stats.LatencyCold < 0 {
		t.Errorf("LatencyCold = %v, want >= 0", report.Stats.LatencyCold)
	}
	if report.Stats.ThroughputChunksPerSec <= 0 {
		t.Errorf("ThroughputChunksPerSec = %v, want > 0", report.Stats.ThroughputChunksPerSec)
	}
}

func TestRunRequiresNonEmptyCorpus(t *testing.T) {
	if _, err := Run(context.Background(), fakeEmbedder{}, nil, 0); err == nil {
		t.Error("Run() error = nil, want non-nil for an empty corpus")
	}
}

func TestRunPropagatesEmbedderError(t *testing.T) {
	wantErr := errors.New("boom")
	corpus := []string{"a", "b", "c"}

	if _, err := Run(context.Background(), fakeEmbedder{err: wantErr}, corpus, 0); err == nil {
		t.Error("Run() error = nil, want non-nil when the embedder fails")
	}
}

// spyEmbedder records the length of each chunks slice it's called with,
// in call order, so tests can assert on Run's call sequence rather than
// just its final output.
type spyEmbedder struct {
	callSizes *[]int
}

func (s spyEmbedder) Embed(_ context.Context, chunks []string) ([][]float32, error) {
	*s.callSizes = append(*s.callSizes, len(chunks))
	out := make([][]float32, len(chunks))
	for i := range chunks {
		out[i] = []float32{0}
	}
	return out, nil
}

var _ goldeneval.Embedder = spyEmbedder{}

func TestRunMeasuresColdLatencyBeforeTheBatchedThroughputCall(t *testing.T) {
	corpus := []string{"a", "b", "c"}
	var callSizes []int

	if _, err := Run(context.Background(), spyEmbedder{callSizes: &callSizes}, corpus, 0); err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	if len(callSizes) == 0 {
		t.Fatal("embedder was never called")
	}
	// The first call must be single-chunk (the cold-latency sample), not
	// the full-corpus batch — otherwise, against a real embedder, any
	// lazy-loading cost would fire during the batch call and be
	// invisible to ColdStart. See the ordering comment in Run.
	if callSizes[0] != 1 {
		t.Errorf("first Embed call size = %d, want 1 (single-chunk cold-latency sample); call sizes were %v", callSizes[0], callSizes)
	}
	// The full-corpus batched call must still happen somewhere in the
	// sequence, for throughput.
	if !slices.Contains(callSizes, len(corpus)) {
		t.Errorf("no call of size %d (full-corpus batch) found among call sizes %v", len(corpus), callSizes)
	}
}
