package spike2measure

import (
	"strings"
	"testing"
)

func TestGenerateCorpusIsDeterministic(t *testing.T) {
	a := GenerateCorpus(42)
	b := GenerateCorpus(42)
	if len(a) != len(b) {
		t.Fatalf("len(a) = %d, len(b) = %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("chunk %d differs between two runs with the same seed", i)
		}
	}
}

func TestGenerateCorpusDifferentSeedsDiffer(t *testing.T) {
	a := GenerateCorpus(1)
	b := GenerateCorpus(2)
	if a[0] == b[0] {
		t.Error("expected different seeds to produce different first chunks")
	}
}

func TestGenerateCorpusSizeAndWordCounts(t *testing.T) {
	chunks := GenerateCorpus(42)
	if len(chunks) != corpusSize {
		t.Fatalf("len(chunks) = %d, want %d", len(chunks), corpusSize)
	}
	for i, c := range chunks {
		wc := len(strings.Fields(c))
		if wc < minWords || wc > maxWords {
			t.Errorf("chunk %d has %d words, want between %d and %d", i, wc, minWords, maxWords)
		}
	}
}

func TestGenerateQueryCorpusIsDeterministic(t *testing.T) {
	a := GenerateQueryCorpus(42)
	b := GenerateQueryCorpus(42)
	if len(a) != len(b) {
		t.Fatalf("len(a) = %d, len(b) = %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("query %d differs between two runs with the same seed", i)
		}
	}
}

func TestGenerateQueryCorpusDifferentSeedsDiffer(t *testing.T) {
	a := GenerateQueryCorpus(1)
	b := GenerateQueryCorpus(2)
	if a[0] == b[0] {
		t.Error("expected different seeds to produce different first queries")
	}
}

func TestGenerateQueryCorpusSizeAndWordCounts(t *testing.T) {
	queries := GenerateQueryCorpus(42)
	if len(queries) != queryCorpusSize {
		t.Fatalf("len(queries) = %d, want %d", len(queries), queryCorpusSize)
	}
	for i, q := range queries {
		wc := len(strings.Fields(q))
		if wc < minQueryWords || wc > maxQueryWords {
			t.Errorf("query %d has %d words, want between %d and %d", i, wc, minQueryWords, maxQueryWords)
		}
	}
}

func TestGenerateQueryCorpusProducesShorterStringsThanGenerateCorpus(t *testing.T) {
	// Checks actual generated output, not just the constants: a
	// query-latency measurement is meaningless if this generator
	// accidentally produces index-length text.
	queries := GenerateQueryCorpus(42)
	chunks := GenerateCorpus(42)

	maxQueryWordCount := 0
	for _, q := range queries {
		if wc := len(strings.Fields(q)); wc > maxQueryWordCount {
			maxQueryWordCount = wc
		}
	}

	minChunkWordCount := len(strings.Fields(chunks[0]))
	for _, c := range chunks {
		if wc := len(strings.Fields(c)); wc < minChunkWordCount {
			minChunkWordCount = wc
		}
	}

	if maxQueryWordCount >= minChunkWordCount {
		t.Errorf("longest generated query has %d words, want fewer than the shortest generated chunk's %d words", maxQueryWordCount, minChunkWordCount)
	}
}
