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
