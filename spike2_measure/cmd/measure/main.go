// Command measure runs the Spike 2 benchmark (PRD Section 7: throughput,
// latency, cold start) against an Embedder and prints a results table.
//
// With no flags, it runs against a placeholder stub embedder (the same
// deterministic bag-of-words hash the golden eval CLI uses) purely to
// smoke-test the pipeline. See riffle_spikes#6's remaining tickets for
// real onnx_test-backed adapters this CLI will grow flags to select.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"text/tabwriter"
	"time"

	spike2measure "github.com/allank/riffle_spikes/spike2_measure"

	"github.com/allank/riffle_spikes/internal/stubembedder"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	seed := flag.Int64("seed", 42, "seed for the deterministic benchmark corpus generator")
	flag.Parse()

	corpus := spike2measure.GenerateCorpus(*seed)

	constructStart := time.Now()
	embedder := stubembedder.Embedder{}
	constructionDuration := time.Since(constructStart)

	report, err := spike2measure.Run(context.Background(), embedder, corpus, constructionDuration)
	if err != nil {
		return fmt.Errorf("running benchmark: %w", err)
	}

	printReport(report)
	return nil
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
