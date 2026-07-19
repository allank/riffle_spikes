package main

import (
	"fmt"
	"log"
	"time"

	"github.com/allank/riffle_spikes/golden_eval/adapters/onnxadapter"
	"github.com/allank/riffle_spikes/golden_eval/adapters/puregoadapter"
	"github.com/allank/riffle_spikes/internal/stubembedder"

	goldeneval "github.com/allank/riffle_spikes/golden_eval"
)

// embedderFlags is the set of asset-path flags buildEmbedder chooses
// between. Shares the goldeneval CLI's flag names and env-var
// convention (GOLDENEVAL_MODEL_PATH etc.) so the same environment
// works for both CLIs.
type embedderFlags struct {
	tokenizerPath string
	modelPath     string
	onnxModelPath string
	onnxLibPath   string
}

// buildEmbedder constructs whichever embedder f's flags select — the
// stub embedder (no flags), the pure-Go adapter, or the ONNX adapter —
// timing how long construction took so the caller can pass it into
// spike2measure.Run for an accurate ColdStart. The returned close func
// releases any resource the chosen embedder owns (the ONNX adapter's
// session); it is a no-op for embedders that own nothing. Callers must
// defer closeFn() and ensure it runs even on a later error (see run()
// in main.go).
func buildEmbedder(f embedderFlags) (embedder goldeneval.Embedder, constructionDuration time.Duration, closeFn func(), err error) {
	noop := func() {}

	pureGoRequested := f.modelPath != ""
	onnxRequested := f.onnxModelPath != "" || f.onnxLibPath != ""

	switch {
	case pureGoRequested && onnxRequested:
		return nil, 0, noop, fmt.Errorf("-model and -onnx-model/-onnx-lib are mutually exclusive; run one adapter per CLI invocation")

	case pureGoRequested:
		if f.tokenizerPath == "" {
			return nil, 0, noop, fmt.Errorf("-tokenizer must be set alongside -model to use the pure-Go adapter")
		}
		start := time.Now()
		adapter, err := puregoadapter.New(f.modelPath, f.tokenizerPath)
		if err != nil {
			return nil, 0, noop, err
		}
		return adapter, time.Since(start), noop, nil

	case onnxRequested:
		if f.onnxModelPath == "" || f.onnxLibPath == "" || f.tokenizerPath == "" {
			return nil, 0, noop, fmt.Errorf("-onnx-model, -onnx-lib, and -tokenizer must all be set to use the ONNX adapter")
		}
		start := time.Now()
		adapter, err := onnxadapter.New(f.onnxModelPath, f.tokenizerPath, f.onnxLibPath)
		if err != nil {
			return nil, 0, noop, err
		}
		constructionDuration := time.Since(start)
		return adapter, constructionDuration, func() {
			if cerr := adapter.Close(); cerr != nil {
				log.Printf("closing ONNX adapter: %v", cerr)
			}
		}, nil

	default:
		return stubembedder.Embedder{}, 0, noop, nil
	}
}
