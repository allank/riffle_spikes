// Command goldeneval runs the golden eval (PRD Section 6) against an
// Embedder and prints a results table.
//
// With no flags, it runs against a placeholder stub embedder (a
// deterministic bag-of-words hash) purely to smoke-test the pipeline.
// Passing -model (with -tokenizer) switches it to onnx_test's pure-Go
// BGE-small path (puregoadapter). Passing -onnx-model and -onnx-lib
// (with -tokenizer) switches it to onnx_test's ONNX Runtime path
// (onnxadapter), the reference every other spike is compared against.
// The two real adapters are mutually exclusive in a single run — see
// riffle_spikes#1's remaining ticket for a comparison mode that runs
// both together.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"text/tabwriter"

	goldeneval "github.com/allank/riffle_spikes/golden_eval"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

// run does the actual work, as opposed to main, so that a deferred
// closeEmbedder() (which releases the ONNX adapter's session) always
// executes before the process exits — main's log.Fatal calls os.Exit,
// which skips deferred functions, so cleanup must happen inside a
// function that returns normally first.
func run() error {
	corpusDir := flag.String("corpus", "golden_eval/corpus", "path to the golden eval corpus directory")
	tokenizerPath := flag.String("tokenizer", os.Getenv("GOLDENEVAL_TOKENIZER_PATH"),
		"path to onnx_test's tokenizer.json, shared by both real adapters below (env: GOLDENEVAL_TOKENIZER_PATH)")
	modelPath := flag.String("model", os.Getenv("GOLDENEVAL_MODEL_PATH"),
		"path to onnx_test's model.safetensors; set together with -tokenizer to use the pure-Go adapter instead of the stub embedder (env: GOLDENEVAL_MODEL_PATH)")
	onnxModelPath := flag.String("onnx-model", os.Getenv("GOLDENEVAL_ONNX_MODEL_PATH"),
		"path to onnx_test's model.onnx; set together with -onnx-lib and -tokenizer to use the ONNX reference adapter (env: GOLDENEVAL_ONNX_MODEL_PATH)")
	onnxLibPath := flag.String("onnx-lib", os.Getenv("GOLDENEVAL_ONNX_LIB_PATH"),
		"path to the ONNX Runtime shared library, e.g. libonnxruntime.dylib (env: GOLDENEVAL_ONNX_LIB_PATH)")
	flag.Parse()

	corpus, err := goldeneval.LoadCorpus(*corpusDir)
	if err != nil {
		return fmt.Errorf("loading corpus: %w", err)
	}

	embedder, closeEmbedder, err := selectEmbedder(embedderFlags{
		tokenizerPath: *tokenizerPath,
		modelPath:     *modelPath,
		onnxModelPath: *onnxModelPath,
		onnxLibPath:   *onnxLibPath,
	})
	if err != nil {
		return fmt.Errorf("selecting embedder: %w", err)
	}
	defer closeEmbedder()

	report, err := goldeneval.Run(context.Background(), corpus, embedder)
	if err != nil {
		return fmt.Errorf("running golden eval: %w", err)
	}

	printReport(report)
	return nil
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
