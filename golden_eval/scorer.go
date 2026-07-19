package goldeneval

import (
	"context"
	"fmt"
	"math"
	"sort"
)

// QueryScore is one query's retrieval-quality result.
type QueryScore struct {
	Query string
	NDCG  float64
	MRR   float64
}

// Report is the golden eval's output for a full run: per-query scores
// and their aggregate.
type Report struct {
	Queries       []QueryScore
	AggregateNDCG float64
	AggregateMRR  float64

	// CosineSimilarity is per-note cosine similarity between embedder's
	// output and a reference vector set, keyed by note ID. Nil unless
	// Run was called with a non-nil reference.
	CosineSimilarity map[string]float64
}

// Run embeds every corpus note and query with embedder, ranks notes per
// query by embedding similarity, and scores each ranking against
// corpus.Expected using nDCG and MRR (PRD Section 6).
//
// If reference is non-nil, Run also computes per-note cosine similarity
// between embedder's note vectors and reference's vectors for the same
// note ID, populating Report.CosineSimilarity — the harness's
// numerical-drift check, since ranking metrics alone can mask or
// amplify small vector-level drift. Build reference with
// EmbedNotesByID against a reference-quality embedder (e.g. the ONNX
// adapter). A note with no corresponding entry in reference is omitted
// from CosineSimilarity rather than scored as 0.
func Run(ctx context.Context, corpus Corpus, embedder Embedder, reference map[string][]float32) (Report, error) {
	noteVectorsByID, err := EmbedNotesByID(ctx, corpus, embedder)
	if err != nil {
		return Report{}, err
	}

	queryVectors, err := embedder.Embed(ctx, corpus.Queries)
	if err != nil {
		return Report{}, fmt.Errorf("embedding queries: %w", err)
	}
	if len(queryVectors) != len(corpus.Queries) {
		return Report{}, fmt.Errorf("embedder returned %d vectors for %d queries", len(queryVectors), len(corpus.Queries))
	}

	report := Report{Queries: make([]QueryScore, len(corpus.Queries))}
	var sumNDCG, sumMRR float64

	for i, query := range corpus.Queries {
		ranked := rankNotes(corpus.Notes, queryVectors[i], noteVectorsByID)
		relevance := relevanceGrades(corpus.Expected[query])

		queryNDCG := ndcgScore(ranked, relevance)
		queryMRR := mrrScore(ranked, relevance)

		report.Queries[i] = QueryScore{Query: query, NDCG: queryNDCG, MRR: queryMRR}
		sumNDCG += queryNDCG
		sumMRR += queryMRR
	}

	if n := len(corpus.Queries); n > 0 {
		report.AggregateNDCG = sumNDCG / float64(n)
		report.AggregateMRR = sumMRR / float64(n)
	}

	if reference != nil {
		report.CosineSimilarity = cosineSimilarityToReference(noteVectorsByID, reference)
	}

	return report, nil
}

// EmbedNotesByID embeds every corpus note with embedder, returning the
// resulting vectors keyed by note ID. Used both by Run and by callers
// building a reference vector set (e.g. from the ONNX adapter) to pass
// into Run's comparison mode.
func EmbedNotesByID(ctx context.Context, corpus Corpus, embedder Embedder) (map[string][]float32, error) {
	noteTexts := make([]string, len(corpus.Notes))
	for i, n := range corpus.Notes {
		noteTexts[i] = n.Text
	}

	vectors, err := embedder.Embed(ctx, noteTexts)
	if err != nil {
		return nil, fmt.Errorf("embedding notes: %w", err)
	}
	if len(vectors) != len(corpus.Notes) {
		return nil, fmt.Errorf("embedder returned %d vectors for %d notes", len(vectors), len(corpus.Notes))
	}

	byID := make(map[string][]float32, len(corpus.Notes))
	for i, n := range corpus.Notes {
		byID[n.ID] = vectors[i]
	}
	return byID, nil
}

// cosineSimilarityToReference computes, for each note ID present in
// both subject and reference, the cosine similarity between their
// vectors. A note present in only one of the two maps is omitted.
func cosineSimilarityToReference(subject, reference map[string][]float32) map[string]float64 {
	out := make(map[string]float64, len(subject))
	for id, vec := range subject {
		refVec, ok := reference[id]
		if !ok {
			continue
		}
		out[id] = cosineSimilarity(vec, refVec)
	}
	return out
}

// rankNotes orders note IDs by descending cosine similarity to
// queryVector, breaking ties on note ID for determinism.
func rankNotes(notes []Note, queryVector []float32, noteVectorsByID map[string][]float32) []string {
	type scored struct {
		id    string
		score float64
	}

	scoredNotes := make([]scored, len(notes))
	for i, n := range notes {
		scoredNotes[i] = scored{id: n.ID, score: cosineSimilarity(queryVector, noteVectorsByID[n.ID])}
	}
	sort.SliceStable(scoredNotes, func(i, j int) bool {
		if scoredNotes[i].score != scoredNotes[j].score {
			return scoredNotes[i].score > scoredNotes[j].score
		}
		return scoredNotes[i].id < scoredNotes[j].id
	})

	ranked := make([]string, len(scoredNotes))
	for i, s := range scoredNotes {
		ranked[i] = s.id
	}
	return ranked
}

func cosineSimilarity(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// relevanceGrades assigns graded relevance from an expected ranking
// (best match first): the first note gets the highest grade, counting
// down to 1 for the last. Notes absent from expected are implicitly 0.
func relevanceGrades(expected []string) map[string]float64 {
	grades := make(map[string]float64, len(expected))
	for i, id := range expected {
		grades[id] = float64(len(expected) - i)
	}
	return grades
}

// ndcgScore is normalized discounted cumulative gain: the ranking's DCG
// divided by the DCG of the ideal ranking (relevant notes sorted by
// grade, most relevant first).
func ndcgScore(ranked []string, relevance map[string]float64) float64 {
	dcg := dcgScore(ranked, relevance)

	ideal := make([]string, 0, len(relevance))
	for id := range relevance {
		ideal = append(ideal, id)
	}
	sort.Slice(ideal, func(i, j int) bool { return relevance[ideal[i]] > relevance[ideal[j]] })
	idcg := dcgScore(ideal, relevance)

	if idcg == 0 {
		return 0
	}
	return dcg / idcg
}

func dcgScore(ranked []string, relevance map[string]float64) float64 {
	var sum float64
	for i, id := range ranked {
		sum += relevance[id] / math.Log2(float64(i+2))
	}
	return sum
}

// mrrScore is the reciprocal rank of the first note with positive
// relevance, or 0 if none appears in ranked.
func mrrScore(ranked []string, relevance map[string]float64) float64 {
	for i, id := range ranked {
		if relevance[id] > 0 {
			return 1.0 / float64(i+1)
		}
	}
	return 0
}
