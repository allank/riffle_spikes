// Command measure runs the Spike 2 benchmark (PRD Section 7: throughput,
// latency, cold start) against an Embedder and prints a results table.
//
// -mode selects the corpus shape: "index" (default) benchmarks
// note-chunk-length text (50-400 words, GenerateCorpus); "query"
// benchmarks short search-query-length text (2-15 words,
// GenerateQueryCorpus) — the shape the PRD's <100ms query-time latency
// goal is actually about, distinct from indexing throughput.
//
// With no flags, it runs against a placeholder stub embedder (the same
// deterministic bag-of-words hash the golden eval CLI uses) purely to
// smoke-test the pipeline. Passing -model (with -tokenizer) switches it
// to onnx_test's pure-Go BGE-small path (puregoadapter). Passing
// -onnx-model and -onnx-lib (with -tokenizer) switches it to onnx_test's
// ONNX Runtime path (onnxadapter), the reference every other spike's
// timing is compared against. The two real adapters are mutually
// exclusive in a single run — this CLI benchmarks one embedder at a
// time, unlike the golden eval CLI's comparison mode.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"text/tabwriter"

	spike2measure "github.com/allank/riffle_spikes/spike2_measure"
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
	seed := flag.Int64("seed", 42, "seed for the deterministic benchmark corpus generator")
	mode := flag.String("mode", "index",
		"'index' benchmarks note-chunk-length text (50-400 words); 'query' benchmarks short search-query-length text (2-15 words), to measure the PRD's query-time latency goal specifically")
	tokenizerPath := flag.String("tokenizer", os.Getenv("GOLDENEVAL_TOKENIZER_PATH"),
		"path to onnx_test's tokenizer.json, shared by both real adapters below (env: GOLDENEVAL_TOKENIZER_PATH)")
	modelPath := flag.String("model", os.Getenv("GOLDENEVAL_MODEL_PATH"),
		"path to onnx_test's model.safetensors; set together with -tokenizer to use the pure-Go adapter instead of the stub embedder (env: GOLDENEVAL_MODEL_PATH)")
	onnxModelPath := flag.String("onnx-model", os.Getenv("GOLDENEVAL_ONNX_MODEL_PATH"),
		"path to onnx_test's model.onnx; set together with -onnx-lib and -tokenizer to use the ONNX reference adapter (env: GOLDENEVAL_ONNX_MODEL_PATH)")
	onnxLibPath := flag.String("onnx-lib", os.Getenv("GOLDENEVAL_ONNX_LIB_PATH"),
		"path to the ONNX Runtime shared library, e.g. libonnxruntime.dylib (env: GOLDENEVAL_ONNX_LIB_PATH)")
	flag.Parse()

	corpus, err := generateCorpus(*mode, *seed)
	if err != nil {
		return err
	}

	embedder, constructionDuration, closeEmbedder, err := buildEmbedder(embedderFlags{
		tokenizerPath: *tokenizerPath,
		modelPath:     *modelPath,
		onnxModelPath: *onnxModelPath,
		onnxLibPath:   *onnxLibPath,
	})
	if err != nil {
		return fmt.Errorf("selecting embedder: %w", err)
	}
	defer closeEmbedder()

	report, err := spike2measure.Run(context.Background(), embedder, corpus, constructionDuration)
	if err != nil {
		return fmt.Errorf("running benchmark: %w", err)
	}

	printReport(report)
	return nil
}

func generateCorpus(mode string, seed int64) ([]string, error) {
	switch mode {
	case "index":
		return spike2measure.GenerateCorpus(seed), nil
	case "query":
		return spike2measure.GenerateQueryCorpus(seed), nil
	default:
		return nil, fmt.Errorf("-mode must be 'index' or 'query', got %q", mode)
	}
}

func printReport(report spike2measure.Report) {
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	defer w.Flush()

	fmt.Fprintln(w, "METRIC\tVALUE")
	fmt.Fprintf(w, "Throughput\t%.1f chunks/sec\n", report.Stats.ThroughputChunksPerSec)
	fmt.Fprintf(w, "Latency (cold)\t%s\n", report.Stats.LatencyCold)
	fmt.Fprintf(w, "Latency (warm)\t%s\n", report.Stats.LatencyWarm)
	fmt.Fprintf(w, "Cold start\t%s\n", report.ColdStart)
}
