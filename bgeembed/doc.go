// Package bgeembed runs BGE-small-en-v1.5 inference via ONNX Runtime:
// WordPiece tokenization plus CLS-token pooling, BGE's documented
// sentence-embedding method (not the BERT pooler head). Ported from
// onnx_test's bge_bench/tokenizer and onnxpath packages, but owned here
// going forward — this package imports nothing from onnx_test, since
// both onnx_test and riffle_spikes are ephemeral investigation repos
// not meant to be maintained long-term, and this code is meant to
// migrate cleanly into riffle later. See
// docs/specs/2026-07-21-self-contained-onnx-embedder-design.md.
//
// The Embedder type (New/Embed/Close) is implemented twice, in
// embedder_modern.go and embedder_legacy_darwin_amd64.go, gated by
// build tags — one per ONNX Runtime binding this repo supports (see
// docs/specs/2026-07-21-self-contained-onnx-embedder-design.md's
// "Implementation outcome" for why: Go resolves imports per file, not
// per package, so a single shared file can't vary which binding it
// imports by platform without a type-alias shim; duplicating the ~90
// lines that actually touch ONNX Runtime was chosen over that
// indirection). Tokenizer (tokenizer.go) and clsAndNormalize
// (pooling.go) have no such split — neither touches ONNX Runtime
// types, so there's nothing to duplicate.
package bgeembed
