// Command goldeneval runs the golden eval (PRD Section 6) against an
// Embedder and prints a results table.
//
// With no flags, it runs against a placeholder stub embedder (a
// deterministic bag-of-words hash) purely to smoke-test the pipeline.
// Passing -model and -tokenizer switches it to onnx_test's pure-Go
// BGE-small path (puregoadapter) for a real baseline. See
// riffle_spikes#1's remaining ticket for the ONNX reference adapter this
// CLI will grow a further flag to select.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"text/tabwriter"

	"github.com/allank/riffle_spikes/golden_eval/adapters/puregoadapter"

	goldeneval "github.com/allank/riffle_spikes/golden_eval"
)

func main() {
	corpusDir := flag.String("corpus", "golden_eval/corpus", "path to the golden eval corpus directory")
	modelPath := flag.String("model", os.Getenv("GOLDENEVAL_MODEL_PATH"),
		"path to onnx_test's model.safetensors; set together with -tokenizer to use the pure-Go adapter instead of the stub embedder (env: GOLDENEVAL_MODEL_PATH)")
	tokenizerPath := flag.String("tokenizer", os.Getenv("GOLDENEVAL_TOKENIZER_PATH"),
		"path to onnx_test's tokenizer.json (env: GOLDENEVAL_TOKENIZER_PATH)")
	flag.Parse()

	corpus, err := goldeneval.LoadCorpus(*corpusDir)
	if err != nil {
		log.Fatalf("loading corpus: %v", err)
	}

	embedder, err := selectEmbedder(*modelPath, *tokenizerPath)
	if err != nil {
		log.Fatalf("selecting embedder: %v", err)
	}

	report, err := goldeneval.Run(context.Background(), corpus, embedder)
	if err != nil {
		log.Fatalf("running golden eval: %v", err)
	}

	printReport(report)
}

func selectEmbedder(modelPath, tokenizerPath string) (goldeneval.Embedder, error) {
	switch {
	case modelPath != "" && tokenizerPath != "":
		return puregoadapter.New(modelPath, tokenizerPath)
	case modelPath != "" || tokenizerPath != "":
		return nil, fmt.Errorf("-model and -tokenizer must both be set to use the pure-Go adapter")
	default:
		return stubEmbedder{}, nil
	}
}

func printReport(report goldeneval.Report) {
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	defer w.Flush()

	fmt.Fprintln(w, "QUERY\tNDCG\tMRR")
	for _, q := range report.Queries {
		fmt.Fprintf(w, "%s\t%.4f\t%.4f\n", q.Query, q.NDCG, q.MRR)
	}
	fmt.Fprintf(w, "AGGREGATE\t%.4f\t%.4f\n", report.AggregateNDCG, report.AggregateMRR)
}
