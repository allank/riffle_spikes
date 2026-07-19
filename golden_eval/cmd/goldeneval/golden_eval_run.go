package main

import (
	"context"
	"fmt"
	"log"

	"github.com/allank/riffle_spikes/golden_eval/adapters/onnxadapter"
	"github.com/allank/riffle_spikes/golden_eval/adapters/puregoadapter"

	goldeneval "github.com/allank/riffle_spikes/golden_eval"
)

// embedderFlags is the set of asset-path flags buildReport chooses
// between. Grouped into one type since -model/-tokenizer and
// -onnx-model/-onnx-lib/-tokenizer are decided together, not one flag
// at a time.
type embedderFlags struct {
	tokenizerPath string
	modelPath     string
	onnxModelPath string
	onnxLibPath   string
}

// buildReport runs the golden eval in whichever mode f's flags select:
// the stub embedder (no flags), the pure-Go adapter alone, the ONNX
// adapter alone, or — when both -model and -onnx-model/-onnx-lib are
// set — a comparison run that scores the pure-Go adapter's rankings
// while also computing its per-note cosine similarity against the ONNX
// adapter's output as reference (PRD Section 6's numerical-drift
// check). Any adapter resource this constructs is released before
// buildReport returns.
func buildReport(ctx context.Context, corpus goldeneval.Corpus, f embedderFlags) (goldeneval.Report, error) {
	pureGoRequested := f.modelPath != ""
	onnxRequested := f.onnxModelPath != "" || f.onnxLibPath != ""

	switch {
	case pureGoRequested && onnxRequested:
		return runComparison(ctx, corpus, f)

	case pureGoRequested:
		if f.tokenizerPath == "" {
			return goldeneval.Report{}, fmt.Errorf("-tokenizer must be set alongside -model to use the pure-Go adapter")
		}
		adapter, err := puregoadapter.New(f.modelPath, f.tokenizerPath)
		if err != nil {
			return goldeneval.Report{}, err
		}
		return goldeneval.Run(ctx, corpus, adapter, nil)

	case onnxRequested:
		if f.onnxModelPath == "" || f.onnxLibPath == "" || f.tokenizerPath == "" {
			return goldeneval.Report{}, fmt.Errorf("-onnx-model, -onnx-lib, and -tokenizer must all be set to use the ONNX adapter")
		}
		adapter, err := onnxadapter.New(f.onnxModelPath, f.tokenizerPath, f.onnxLibPath)
		if err != nil {
			return goldeneval.Report{}, err
		}
		defer closeOnnxAdapter(adapter)
		return goldeneval.Run(ctx, corpus, adapter, nil)

	default:
		return goldeneval.Run(ctx, corpus, stubEmbedder{}, nil)
	}
}

// runComparison scores the pure-Go adapter's rankings and, using the
// ONNX adapter's output as the reference vector set, populates per-note
// cosine similarity alongside them.
func runComparison(ctx context.Context, corpus goldeneval.Corpus, f embedderFlags) (goldeneval.Report, error) {
	if f.tokenizerPath == "" {
		return goldeneval.Report{}, fmt.Errorf("-tokenizer must be set to run a comparison")
	}
	if f.onnxModelPath == "" || f.onnxLibPath == "" {
		return goldeneval.Report{}, fmt.Errorf("-onnx-model and -onnx-lib must both be set to run a comparison")
	}

	pureGo, err := puregoadapter.New(f.modelPath, f.tokenizerPath)
	if err != nil {
		return goldeneval.Report{}, fmt.Errorf("loading pure-Go adapter: %w", err)
	}

	onnx, err := onnxadapter.New(f.onnxModelPath, f.tokenizerPath, f.onnxLibPath)
	if err != nil {
		return goldeneval.Report{}, fmt.Errorf("loading ONNX adapter: %w", err)
	}
	defer closeOnnxAdapter(onnx)

	reference, err := goldeneval.EmbedNotesByID(ctx, corpus, onnx)
	if err != nil {
		return goldeneval.Report{}, fmt.Errorf("embedding reference notes: %w", err)
	}

	return goldeneval.Run(ctx, corpus, pureGo, reference)
}

func closeOnnxAdapter(a *onnxadapter.Adapter) {
	if err := a.Close(); err != nil {
		log.Printf("closing ONNX adapter: %v", err)
	}
}
