package goldeneval

import (
	"context"
	"math"
	"testing"
)

// fakeEmbedder returns a fixed, deterministic vector per input chunk,
// looked up by exact text match. It lets scorer tests control exactly
// which note ends up "closest" to a query without any real model.
type fakeEmbedder struct {
	vectors map[string][]float32
}

func (f fakeEmbedder) Embed(_ context.Context, chunks []string) ([][]float32, error) {
	out := make([][]float32, len(chunks))
	for i, c := range chunks {
		v, ok := f.vectors[c]
		if !ok {
			return nil, errUnknownChunk(c)
		}
		out[i] = v
	}
	return out, nil
}

type errUnknownChunk string

func (e errUnknownChunk) Error() string { return "fakeEmbedder: no vector for chunk " + string(e) }

func threeNoteCorpus() Corpus {
	return Corpus{
		Notes: []Note{
			{ID: "A", Text: "note A"},
			{ID: "B", Text: "note B"},
			{ID: "C", Text: "note C"},
		},
		Queries: []string{"Q"},
		Expected: map[string][]string{
			"Q": {"A", "B"}, // A = best match (rel 2), B = distractor (rel 1), C unlisted (rel 0)
		},
	}
}

func TestRunPerfectRankingScoresMaxNDCGAndMRR(t *testing.T) {
	corpus := threeNoteCorpus()
	// Query vector closest to A, then B, then C — matches Expected exactly.
	embedder := fakeEmbedder{vectors: map[string][]float32{
		"note A": {1, 0, 0},
		"note B": {0.7, 0.7, 0},
		"note C": {0, 0, 1},
		"Q":      {1, 0, 0},
	}}

	report, err := Run(context.Background(), corpus, embedder)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if len(report.Queries) != 1 {
		t.Fatalf("len(report.Queries) = %d, want 1", len(report.Queries))
	}

	got := report.Queries[0]
	if !approxEqual(got.NDCG, 1.0) {
		t.Errorf("NDCG = %v, want 1.0", got.NDCG)
	}
	if !approxEqual(got.MRR, 1.0) {
		t.Errorf("MRR = %v, want 1.0", got.MRR)
	}
	if !approxEqual(report.AggregateNDCG, 1.0) {
		t.Errorf("AggregateNDCG = %v, want 1.0", report.AggregateNDCG)
	}
	if !approxEqual(report.AggregateMRR, 1.0) {
		t.Errorf("AggregateMRR = %v, want 1.0", report.AggregateMRR)
	}
	if report.CosineSimilarity != nil {
		t.Errorf("CosineSimilarity = %v, want nil (not populated by Run)", report.CosineSimilarity)
	}
}

func TestRunInvertedRankingScoresPenalized(t *testing.T) {
	corpus := threeNoteCorpus()
	// Query vector closest to C, then B, then A — exact inverse of Expected.
	embedder := fakeEmbedder{vectors: map[string][]float32{
		"note A": {0, 0, 1},
		"note B": {0.5, 0.5, 0},
		"note C": {1, 0, 0},
		"Q":      {1, 0, 0},
	}}

	report, err := Run(context.Background(), corpus, embedder)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	got := report.Queries[0]
	// Independently-derived reference values for ranking [C, B, A] against
	// relevance {A: 2, B: 1, C: 0}: DCG = 0/log2(2) + 1/log2(3) + 2/log2(4),
	// IDCG (ideal ranking [A, B]) = 2/log2(2) + 1/log2(3).
	wantDCG := 0/math.Log2(2) + 1/math.Log2(3) + 2/math.Log2(4)
	wantIDCG := 2/math.Log2(2) + 1/math.Log2(3)
	wantNDCG := wantDCG / wantIDCG
	if !approxEqual(got.NDCG, wantNDCG) {
		t.Errorf("NDCG = %v, want %v", got.NDCG, wantNDCG)
	}
	// First relevant note (B, rel 1) appears at rank 2.
	wantMRR := 1.0 / 2.0
	if !approxEqual(got.MRR, wantMRR) {
		t.Errorf("MRR = %v, want %v", got.MRR, wantMRR)
	}
}

func TestRunQueryWithNoExpectedRankingScoresZero(t *testing.T) {
	corpus := Corpus{
		Notes: []Note{
			{ID: "A", Text: "note A"},
		},
		Queries:  []string{"unranked query"},
		Expected: map[string][]string{}, // no entry for this query
	}
	embedder := fakeEmbedder{vectors: map[string][]float32{
		"note A":         {1, 0},
		"unranked query": {1, 0},
	}}

	report, err := Run(context.Background(), corpus, embedder)
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}

	got := report.Queries[0]
	if got.NDCG != 0 {
		t.Errorf("NDCG = %v, want 0", got.NDCG)
	}
	if got.MRR != 0 {
		t.Errorf("MRR = %v, want 0", got.MRR)
	}
}

func TestRunPropagatesEmbedderError(t *testing.T) {
	corpus := threeNoteCorpus()
	embedder := fakeEmbedder{vectors: map[string][]float32{}} // every lookup fails

	if _, err := Run(context.Background(), corpus, embedder); err == nil {
		t.Error("Run() error = nil, want non-nil when embedder fails")
	}
}

func approxEqual(a, b float64) bool {
	const eps = 1e-9
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff < eps
}
