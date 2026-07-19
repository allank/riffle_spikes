package spike2measure

import (
	"math/rand"
	"strings"
)

const (
	corpusSize = 1000
	minWords   = 50
	maxWords   = 400
)

// vocabulary is a small, fixed word list drawn from to build chunks.
// Content is meaningless for a throughput benchmark — only volume and
// length distribution matter, unlike the golden eval's hand-authored
// corpus, which needs known-correct relevance.
var vocabulary = []string{
	"the", "and", "of", "to", "in", "a", "is", "that", "for", "on",
	"with", "as", "was", "at", "by", "an", "be", "this", "which", "or",
	"from", "had", "not", "but", "what", "all", "were", "when", "we", "there",
	"can", "your", "one", "if", "each", "how", "up", "out", "them", "then",
	"embedding", "vector", "index", "chunk", "note", "vault", "retrieval",
	"query", "graph", "search", "hybrid", "semantic", "hash", "merkle",
	"riffle", "onnx", "runtime", "model", "tokenizer", "inference", "latency",
	"throughput", "benchmark", "spike", "golden", "reference", "adapter",
	"pure", "go", "rust", "sidecar", "candle", "tract", "weights", "layer",
	"attention", "transformer", "score", "rank", "cosine", "similarity",
}

// GenerateCorpus deterministically generates 1,000 text chunks with a
// length distribution representative of Obsidian note chunks (50-400
// words), for throughput benchmarking. The same seed always produces
// the same corpus.
func GenerateCorpus(seed int64) []string {
	rng := rand.New(rand.NewSource(seed))

	chunks := make([]string, corpusSize)
	for i := range chunks {
		wordCount := minWords + rng.Intn(maxWords-minWords+1)
		words := make([]string, wordCount)
		for j := range words {
			words[j] = vocabulary[rng.Intn(len(vocabulary))]
		}
		chunks[i] = strings.Join(words, " ")
	}
	return chunks
}
