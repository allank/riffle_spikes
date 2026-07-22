//go:build darwin && amd64

package bgeembed

import (
	"fmt"
	"sync"

	ort "github.com/allank/riffle_spikes/internal/ortlegacy"
)

var initOnce sync.Once
var initErr error

// Embedder wraps an ONNX Runtime session for BGE-small.
type Embedder struct {
	session *ort.DynamicAdvancedSession
	tok     *Tokenizer
}

// New loads the ONNX model and tokenizer, initializing the ONNX Runtime
// shared library at libPath. On darwin/amd64 this is Intel Mac's only
// working embedded path: ONNX Runtime's last official osx-x86_64 build
// (v1.23.0, API version 23) needs this vendored, API-23-compatible
// binding — the live one requests API 25, which no osx-x86_64 build
// supports. See
// docs/specs/2026-07-21-self-contained-onnx-embedder-design.md.
func New(modelPath, tokenizerPath, libPath string) (*Embedder, error) {
	initOnce.Do(func() {
		ort.SetSharedLibraryPath(libPath)
		initErr = ort.InitializeEnvironment()
	})
	if initErr != nil {
		return nil, fmt.Errorf("bgeembed: onnxruntime init: %w", initErr)
	}
	tok, err := LoadFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("bgeembed: loading tokenizer: %w", err)
	}
	inputNames := []string{"input_ids", "attention_mask", "token_type_ids"}
	outputNames := []string{"last_hidden_state"}
	session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, nil)
	if err != nil {
		return nil, fmt.Errorf("bgeembed: creating onnx session: %w", err)
	}
	return &Embedder{session: session, tok: tok}, nil
}

// Embed tokenizes text and returns the L2-normalized [CLS] sentence
// embedding.
func (e *Embedder) Embed(text string) ([]float32, error) {
	enc := e.tok.Encode(text, 512)
	seqLen := int64(len(enc.InputIDs))

	inputIDs, err := ort.NewTensor(ort.NewShape(1, seqLen), enc.InputIDs)
	if err != nil {
		return nil, fmt.Errorf("bgeembed: building input_ids tensor: %w", err)
	}
	defer inputIDs.Destroy()

	attnMask, err := ort.NewTensor(ort.NewShape(1, seqLen), enc.AttentionMask)
	if err != nil {
		return nil, fmt.Errorf("bgeembed: building attention_mask tensor: %w", err)
	}
	defer attnMask.Destroy()

	tokenTypes, err := ort.NewTensor(ort.NewShape(1, seqLen), enc.TokenTypeIDs)
	if err != nil {
		return nil, fmt.Errorf("bgeembed: building token_type_ids tensor: %w", err)
	}
	defer tokenTypes.Destroy()

	outputData := make([]float32, seqLen*hiddenSize)
	output, err := ort.NewTensor(ort.NewShape(1, seqLen, hiddenSize), outputData)
	if err != nil {
		return nil, fmt.Errorf("bgeembed: building output tensor: %w", err)
	}
	defer output.Destroy()

	if err := e.session.Run([]ort.Value{inputIDs, attnMask, tokenTypes}, []ort.Value{output}); err != nil {
		return nil, fmt.Errorf("bgeembed: running onnx session: %w", err)
	}

	hidden := output.GetData()
	return clsAndNormalize(hidden), nil
}

// Close releases the underlying ONNX Runtime session.
func (e *Embedder) Close() error {
	return e.session.Destroy()
}
