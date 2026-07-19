// Command goldeneval runs the golden eval (PRD Section 6) against an
// Embedder and prints a results table.
//
// This ticket wires the pipeline up with a placeholder stub embedder
// only — see riffle_spikes#1's later tickets for the real onnx_test-backed
// adapters (puregoadapter, onnxadapter) this CLI will grow flags to select
// between.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"text/tabwriter"

	goldeneval "github.com/allank/riffle_spikes/golden_eval"
)

func main() {
	corpusDir := "golden_eval/corpus"
	if len(os.Args) > 1 {
		corpusDir = os.Args[1]
	}

	corpus, err := goldeneval.LoadCorpus(corpusDir)
	if err != nil {
		log.Fatalf("loading corpus: %v", err)
	}

	report, err := goldeneval.Run(context.Background(), corpus, stubEmbedder{})
	if err != nil {
		log.Fatalf("running golden eval: %v", err)
	}

	printReport(report)
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
