# riffle_spikes

Investigation repo testing local embedding-inference approaches for
[Riffle](https://github.com/allank/riffle), driven by the
embedding-inference PRD. Riffle needs to generate BGE-small embeddings
from text chunks locally; this repo answers how fast, how correct, and
how easy to distribute each candidate approach actually is — with real
measurements, not estimates.

## What's here

Four embedding-inference paths, all wrapped behind one shared
`Embedder` interface (`golden_eval/embedder.go`) so the same
measurement code runs against every one of them unchanged:

| Path | Where | Status |
|---|---|---|
| ONNX Runtime (reference) | `golden_eval/adapters/onnxadapter` | done |
| Pure Go | `golden_eval/adapters/puregoadapter` | done |
| Rust sidecar — tract | `spike3_rust_sidecar/` + `golden_eval/adapters/sidecaradapter` | done |
| Rust sidecar — candle (CPU) | `spike3_candle_sidecar/` + `sidecaradapter` | done |
| Rust sidecar — candle (Metal) | same binary, `CANDLE_DEVICE=metal` | code done; numbers blocked on Apple Silicon hardware — see [#18](https://github.com/allank/riffle_spikes/issues/18) |

Two harnesses measure all of them:

- **`golden_eval/`** — correctness: nDCG/MRR retrieval quality against
  a hand-authored corpus, plus per-note cosine similarity vs. the ONNX
  reference. CLI: `golden_eval/cmd/goldeneval`.
- **`spike2_measure/`** — performance: throughput, latency (cold/warm),
  cold start, on both index-length (50–400 word) and query-length
  (2–15 word) text. CLI: `spike2_measure/cmd/measure`.

Real, measured results live in
[`docs/decision-criteria.md`](docs/decision-criteria.md) — that file is
the source of truth for numbers, not this README.

## Repo layout

```
golden_eval/            correctness harness + CLI + adapters
spike2_measure/         performance harness + CLI
spike3_rust_sidecar/    tract-backed Rust sidecar (ndjson over stdio)
spike3_candle_sidecar/  candle-backed Rust sidecar (same protocol)
internal/stubembedder/  deterministic placeholder Embedder, shared by both CLIs
docs/                   decision-criteria.md (results), agents/ (agent-skill config)
CONTEXT.md              domain glossary
Makefile                run everything — see `make help`
```

`onnx_test` (a sibling checkout, `../onnx_test`) and `riffle` (the
project this all feeds back into) are separate repos this one depends
on but doesn't own.

## Machine setup

- **Go** 1.22+ (this repo's `go.mod` pins 1.22.3; developed against
  1.26).
- **Rust** — `rustc`/`cargo`, a reasonably recent stable toolchain
  (both sidecars use edition 2024, which needs Rust 1.85+; developed
  against 1.96).
- **ONNX Runtime** — needed for the ONNX reference adapter and any
  golden-eval comparison mode. `brew install onnxruntime` gets you the
  shared library. Default paths this repo's Makefile/CLIs assume:
  `/opt/homebrew/lib/libonnxruntime.dylib` (Apple Silicon Homebrew) or
  `/usr/local/lib/libonnxruntime.dylib` (Intel Homebrew) — override via
  `-onnx-lib` (CLIs) or `ONNX_LIB` (Makefile) if yours differs. On
  Apple Silicon, `-onnx-embedded` (or `make golden-onnx-embedded`) skips
  `brew install` entirely — it downloads and caches ONNX Runtime on
  first use instead (see `embeddedonnx`'s package doc and
  `docs/specs/2026-07-21-embed-onnxruntime-design.md`). Not yet
  supported on Intel Mac — see that spec for why.
- **`onnx_test`**, checked out as a sibling directory (`../onnx_test`
  relative to this repo) — `golden_eval`'s Go module has a `replace`
  directive pointing there. Its model assets (`model.onnx`,
  `model.safetensors`, `tokenizer.json`, `config.json`, under
  `inference/bge_bench/data/`) are gitignored in that repo and not
  distributed via git — they're the standard `BAAI/bge-small-en-v1.5`
  HuggingFace model files. `tokenizer.json`/`config.json` come with the
  `onnx_test` checkout; `model.onnx`/`model.safetensors` are fetched
  automatically (via `curl`, straight from HuggingFace, no Python/torch
  needed) the first time any Makefile target needs them, or standalone
  via `make fetch-bge-model`. If you already have these files from
  elsewhere (e.g. a private fine-tune), just copy them into place — the
  fetch only runs when a file is missing, so it won't overwrite them.
- **Zig** — not currently required by anything in this repo. The PRD
  mentions `cargo zigbuild` for future cross-platform distribution
  builds (Spike 1), but nothing here does that yet — both Rust
  sidecars are only ever built natively for the machine running them.

## Running tests

```
make test
```

Or individually: `go test ./...` (Go); `cd spike3_rust_sidecar &&
cargo test`; `cd spike3_candle_sidecar && cargo test` (Rust — these
cover wire-protocol logic only; model-touching code needs real local
assets and is verified manually against them, not by `cargo test`).

## Running benchmarks

```
make help
```

lists every target. Highlights:

- `make golden-eval-all` — correctness (cosine similarity vs. ONNX) for
  every real adapter.
- `make bench-all-query` — fast performance sanity check, all five
  engines, seconds each.
- `make bench-all` / `make all` — everything, including index mode,
  which is slow (pure-Go alone takes ~26 minutes for the 1,000-chunk
  corpus).
- `make metal-verify` — candle on Metal specifically, the one piece
  still pending real numbers; see
  [#18](https://github.com/allank/riffle_spikes/issues/18).

All targets accept `ONNX_TEST_DIR`/`ONNX_LIB` overrides if your local
paths differ from the defaults above — see the `Makefile`'s header
comment or `make help`.

## Background

- Domain glossary and design decisions: [`CONTEXT.md`](CONTEXT.md).
- Results: [`docs/decision-criteria.md`](docs/decision-criteria.md).
- Issue tracker: GitHub Issues on this repo — see
  `docs/agents/issue-tracker.md` for the convention, or just browse
  [open](https://github.com/allank/riffle_spikes/issues) and
  [closed](https://github.com/allank/riffle_spikes/issues?q=is%3Aissue+is%3Aclosed)
  issues for the full history of specs and tickets behind this work.
