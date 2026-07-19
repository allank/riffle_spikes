package main

import (
	"context"
	"fmt"
	"log"

	"github.com/allank/riffle_spikes/golden_eval/adapters/onnxadapter"
	"github.com/allank/riffle_spikes/golden_eval/adapters/puregoadapter"
	"github.com/allank/riffle_spikes/golden_eval/adapters/sidecaradapter"
	"github.com/allank/riffle_spikes/internal/stubembedder"

	goldeneval "github.com/allank/riffle_spikes/golden_eval"
)

// embedderFlags is the set of asset-path flags buildReport chooses
// between. Grouped into one type since -model/-tokenizer,
// -onnx-model/-onnx-lib/-tokenizer, and -sidecar-binary/-sidecar-model/
// -tokenizer are each decided together, not one flag at a time.
type embedderFlags struct {
	tokenizerPath     string
	modelPath         string
	onnxModelPath     string
	onnxLibPath       string
	sidecarBinaryPath string
	sidecarModelPath  string
}

// buildReport runs the golden eval in whichever mode f's flags select:
// the stub embedder (no flags), the pure-Go adapter alone, the sidecar
// adapter alone, the ONNX adapter alone, or — when either pure-Go or
// the sidecar is set together with -onnx-model/-onnx-lib — a comparison
// run that scores the subject adapter's rankings while also computing
// its per-note cosine similarity against the ONNX adapter's output as
// reference (PRD Section 6's numerical-drift check). Pure-Go and the
// sidecar are mutually exclusive: only one subject adapter per run. Any
// adapter resource this constructs is released before buildReport
// returns.
func buildReport(ctx context.Context, corpus goldeneval.Corpus, f embedderFlags) (goldeneval.Report, error) {
	pureGoRequested := f.modelPath != ""
	sidecarRequested := f.sidecarBinaryPath != "" || f.sidecarModelPath != ""
	onnxRequested := f.onnxModelPath != "" || f.onnxLibPath != ""

	if pureGoRequested && sidecarRequested {
		return goldeneval.Report{}, fmt.Errorf("-model and -sidecar-binary/-sidecar-model are mutually exclusive; only one subject embedder per run")
	}

	switch {
	case (pureGoRequested || sidecarRequested) && onnxRequested:
		return runComparison(ctx, corpus, f)

	case pureGoRequested || sidecarRequested:
		if f.tokenizerPath == "" {
			return goldeneval.Report{}, fmt.Errorf("-tokenizer must be set alongside -model or -sidecar-binary/-sidecar-model")
		}
		subject, closeSubject, err := buildSubject(f)
		if err != nil {
			return goldeneval.Report{}, err
		}
		defer closeSubject()
		return goldeneval.Run(ctx, corpus, subject, nil)

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
		return goldeneval.Run(ctx, corpus, stubembedder.Embedder{}, nil)
	}
}

// runComparison scores the subject adapter's (pure-Go or sidecar,
// whichever f selects) rankings and, using the ONNX adapter's output as
// the reference vector set, populates per-note cosine similarity
// alongside them.
func runComparison(ctx context.Context, corpus goldeneval.Corpus, f embedderFlags) (goldeneval.Report, error) {
	if f.tokenizerPath == "" {
		return goldeneval.Report{}, fmt.Errorf("-tokenizer must be set to run a comparison")
	}
	if f.onnxModelPath == "" || f.onnxLibPath == "" {
		return goldeneval.Report{}, fmt.Errorf("-onnx-model and -onnx-lib must both be set to run a comparison")
	}

	subject, closeSubject, err := buildSubject(f)
	if err != nil {
		return goldeneval.Report{}, fmt.Errorf("loading subject adapter: %w", err)
	}
	defer closeSubject()

	onnx, err := onnxadapter.New(f.onnxModelPath, f.tokenizerPath, f.onnxLibPath)
	if err != nil {
		return goldeneval.Report{}, fmt.Errorf("loading ONNX adapter: %w", err)
	}
	defer closeOnnxAdapter(onnx)

	reference, err := goldeneval.EmbedNotesByID(ctx, corpus, onnx)
	if err != nil {
		return goldeneval.Report{}, fmt.Errorf("embedding reference notes: %w", err)
	}

	return goldeneval.Run(ctx, corpus, subject, reference)
}

// buildSubject constructs whichever of the pure-Go or sidecar adapters
// f selects as the "subject" embedder — the one whose rankings (and,
// in comparison mode, cosine similarity) get scored. Shared by
// buildReport's standalone-subject case and runComparison, so adding a
// third subject adapter only means adding one case here.
func buildSubject(f embedderFlags) (goldeneval.Embedder, func(), error) {
	noop := func() {}

	switch {
	case f.modelPath != "":
		adapter, err := puregoadapter.New(f.modelPath, f.tokenizerPath)
		if err != nil {
			return nil, noop, err
		}
		return adapter, noop, nil

	case f.sidecarBinaryPath != "" || f.sidecarModelPath != "":
		if f.sidecarBinaryPath == "" || f.sidecarModelPath == "" {
			return nil, noop, fmt.Errorf("-sidecar-binary and -sidecar-model must both be set to use the sidecar adapter")
		}
		adapter, err := sidecaradapter.New(f.sidecarBinaryPath, f.sidecarModelPath, f.tokenizerPath)
		if err != nil {
			return nil, noop, err
		}
		return adapter, func() {
			if cerr := adapter.Close(); cerr != nil {
				log.Printf("closing sidecar adapter: %v", cerr)
			}
		}, nil

	default:
		return nil, noop, fmt.Errorf("no subject embedder requested")
	}
}

func closeOnnxAdapter(a *onnxadapter.Adapter) {
	if err := a.Close(); err != nil {
		log.Printf("closing ONNX adapter: %v", err)
	}
}
