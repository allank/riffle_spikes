package main

import (
	"fmt"
	"log"

	"github.com/allank/riffle_spikes/golden_eval/adapters/onnxadapter"
	"github.com/allank/riffle_spikes/golden_eval/adapters/puregoadapter"

	goldeneval "github.com/allank/riffle_spikes/golden_eval"
)

// embedderFlags is the set of asset-path flags selectEmbedder chooses
// between. Grouped into one type since -model/-tokenizer and
// -onnx-model/-onnx-lib/-tokenizer are decided together, not one flag
// at a time.
type embedderFlags struct {
	tokenizerPath string
	modelPath     string
	onnxModelPath string
	onnxLibPath   string
}

// selectEmbedder chooses which Embedder to run against based on which
// flags were set. The returned close func releases any resource the
// chosen embedder owns (the ONNX adapter's session); it is a no-op for
// embedders that own nothing. Callers must defer closeFn() and ensure it
// runs even on a later error (see run() in main.go).
func selectEmbedder(f embedderFlags) (embedder goldeneval.Embedder, closeFn func(), err error) {
	noop := func() {}

	pureGoRequested := f.modelPath != ""
	onnxRequested := f.onnxModelPath != "" || f.onnxLibPath != ""

	switch {
	case pureGoRequested && onnxRequested:
		return nil, noop, fmt.Errorf("-model and -onnx-model/-onnx-lib are mutually exclusive; run one adapter per CLI invocation")

	case pureGoRequested:
		if f.tokenizerPath == "" {
			return nil, noop, fmt.Errorf("-tokenizer must be set alongside -model to use the pure-Go adapter")
		}
		adapter, err := puregoadapter.New(f.modelPath, f.tokenizerPath)
		return adapter, noop, err

	case onnxRequested:
		if f.onnxModelPath == "" || f.onnxLibPath == "" || f.tokenizerPath == "" {
			return nil, noop, fmt.Errorf("-onnx-model, -onnx-lib, and -tokenizer must all be set to use the ONNX adapter")
		}
		adapter, err := onnxadapter.New(f.onnxModelPath, f.tokenizerPath, f.onnxLibPath)
		if err != nil {
			return nil, noop, err
		}
		return adapter, func() {
			if cerr := adapter.Close(); cerr != nil {
				log.Printf("closing ONNX adapter: %v", cerr)
			}
		}, nil

	default:
		return stubEmbedder{}, noop, nil
	}
}
