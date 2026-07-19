package main

import (
	"fmt"
	"log"
	"time"

	"github.com/allank/riffle_spikes/golden_eval/adapters/onnxadapter"
	"github.com/allank/riffle_spikes/golden_eval/adapters/puregoadapter"
	"github.com/allank/riffle_spikes/golden_eval/adapters/sidecaradapter"
	"github.com/allank/riffle_spikes/internal/stubembedder"

	goldeneval "github.com/allank/riffle_spikes/golden_eval"
)

// embedderFlags is the set of asset-path flags buildEmbedder chooses
// between. Shares the goldeneval CLI's flag names and env-var
// convention (GOLDENEVAL_MODEL_PATH etc.) so the same environment
// works for both CLIs.
type embedderFlags struct {
	tokenizerPath     string
	modelPath         string
	onnxModelPath     string
	onnxLibPath       string
	sidecarBinaryPath string
	sidecarModelPath  string
}

// buildEmbedder constructs whichever embedder f's flags select — the
// stub embedder (no flags), the pure-Go adapter, the ONNX adapter, or
// the sidecar adapter — timing how long construction took so the
// caller can pass it into spike2measure.Run for an accurate ColdStart.
// The returned close func releases any resource the chosen embedder
// owns (the ONNX and sidecar adapters' subprocess/session); it is a
// no-op for embedders that own nothing. Callers must defer closeFn()
// and ensure it runs even on a later error (see run() in main.go).
//
// Unlike the golden eval CLI, this one has no comparison mode —
// spike2measure.Run benchmarks exactly one Embedder — so the three real
// adapters are mutually exclusive: only one may be requested per run.
func buildEmbedder(f embedderFlags) (embedder goldeneval.Embedder, constructionDuration time.Duration, closeFn func(), err error) {
	noop := func() {}

	pureGoRequested := f.modelPath != ""
	onnxRequested := f.onnxModelPath != "" || f.onnxLibPath != ""
	sidecarRequested := f.sidecarBinaryPath != "" || f.sidecarModelPath != ""

	requested := 0
	for _, r := range []bool{pureGoRequested, onnxRequested, sidecarRequested} {
		if r {
			requested++
		}
	}
	if requested > 1 {
		return nil, 0, noop, fmt.Errorf("-model, -onnx-model/-onnx-lib, and -sidecar-binary/-sidecar-model are mutually exclusive; run one adapter per CLI invocation")
	}

	switch {
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

	case sidecarRequested:
		if f.sidecarBinaryPath == "" || f.sidecarModelPath == "" || f.tokenizerPath == "" {
			return nil, 0, noop, fmt.Errorf("-sidecar-binary, -sidecar-model, and -tokenizer must all be set to use the sidecar adapter")
		}
		start := time.Now()
		adapter, err := sidecaradapter.New(f.sidecarBinaryPath, f.sidecarModelPath, f.tokenizerPath)
		if err != nil {
			return nil, 0, noop, err
		}
		constructionDuration := time.Since(start)
		return adapter, constructionDuration, func() {
			if cerr := adapter.Close(); cerr != nil {
				log.Printf("closing sidecar adapter: %v", cerr)
			}
		}, nil

	default:
		return stubembedder.Embedder{}, 0, noop, nil
	}
}
