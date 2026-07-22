// Package ortlegacy is a vendored copy of github.com/yalue/onnxruntime_go
// v1.25.0 (MIT license, see LICENSE in this directory), used only on
// darwin/amd64 — the one platform where ONNX Runtime's last official
// osx-x86_64 release (v1.23.0, API version 23) is incompatible with the
// live binding riffle_spikes otherwise depends on (v1.29.0, which
// requests API version 25). See
// docs/specs/2026-07-21-self-contained-onnx-embedder-design.md for the
// full rationale.
//
// This is a frozen copy, not a tracked dependency: it has no go.mod of
// its own and receives no upstream updates. If Microsoft ever resumes
// publishing osx-x86_64 builds at a current API version, this package
// (and bgeembed's darwin/amd64 build tag split) can be deleted entirely
// in favor of the live binding for every platform.
//
// Source: https://github.com/yalue/onnxruntime_go/tree/v1.25.0 — the
// test suite (onnxruntime_test.go) and its test_data/ fixtures were
// intentionally not vendored, since this package is exercised through
// riffle_spikes's own golden-eval verification instead, not its own
// unit tests.
package ortlegacy
