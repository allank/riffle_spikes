# Embedding-inference PRD spikes

**Status:** All done except candle-on-Metal — blocked on Apple Silicon hardware access (issue #18, deliberately left open).
**Updated:** 2026-07-20

## Goal
Test local embedding-inference options for Riffle (pure Go, ONNX Runtime, Rust sidecars via tract/candle) with real measured evidence, per the embedding-inference PRD, to decide what Riffle ships with.

## Current state
- Golden eval harness (correctness: nDCG/MRR + cosine similarity vs ONNX) — done, closed (#1–#5).
- Spike 2 benchmark harness (throughput/latency/cold-start, index + query modes) — done, closed (#6–#10).
- Spike 3 tract sidecar — done, closed (#11–#15). 1.5 chunks/sec index throughput, cosine sim 1.000000 vs ONNX — correct but only ~5x slower than ONNX's 7.6, not the "same order of magnitude" the PRD hoped for.
- Spike 3 candle sidecar (CPU) — done, closed (#17, under parent #16). 3.2 chunks/sec, cosine sim 0.9999999999998 — materially better than tract (~2.4x slower than ONNX, not ~5x), meaningfully changes the PRD's kill-criteria calculus in candle's favor.
- Spike 3 candle sidecar (Metal) — code done (`CANDLE_DEVICE=metal`/`cpu`/unset all implemented and tested), but real numbers blocked: dev machine is Intel (2020 13" MBP, Iris Plus Graphics) — `Device::new_metal(0)` constructs fine, first inference fails ("Failed to create pipeline"), a hardware/driver limit, not a code bug. Issue #18 intentionally left open, not closed.
- Added `Makefile` (`make metal-verify` built specifically for the Apple Silicon follow-up) and `README.md` for setup/onboarding.
- `CANDLE_DEVICE=auto` (canary-tested Metal-with-CPU-fallback) was designed in conversation but explicitly deferred — user wants clean explicit CPU-vs-Metal runs first.

## Decisions
- Each spike went through grill/spec → tickets → implement → two-axis code review, with real hardware verification (no committed test touches real model weights — established repo convention).
- `riffle_spikes` never gets its own ADRs — consequential findings graduate to `riffle`'s ADRs instead.
- Golden eval corpus is hand-authored/synthetic; Spike 2 corpus is generated (1,000 chunks for index mode, 50 short queries for query mode).
- `CANDLE_DEVICE` semantics (unset/`cpu`→CPU, `metal`→force+error) kept exactly as shipped — no default-behavior change, to protect the Makefile's `*-cpu-*` target meanings.

## Open threads
- Issue #18 is the only open ticket — needs real Apple Silicon hardware.
- PRD's Spike 3 kill criteria not yet formally evaluated now that candle CPU looks strong — worth revisiting once Metal numbers land.
- Full-vault indexing time (~10k chunks, PRD Goals section) never measured — out of scope so far.

## Next steps
1. On Apple Silicon: clone `riffle_spikes` + `onnx_test` as siblings, obtain `onnx_test`'s gitignored model assets, run `make metal-verify`.
2. Fill real numbers into `docs/decision-criteria.md` footnote ⁶ and the two "blocked on hardware" rows; close #18.
3. Revisit the PRD's Spike 3 kill/keep decision with all five numbers in hand (ONNX/pure-Go/tract/candle-CPU/candle-Metal) — likely candle becomes the Horizon 2 feasibility gate per the PRD.
4. If still wanted: implement `CANDLE_DEVICE=auto` (design already discussed this session).

## Artifacts
- Results: `docs/decision-criteria.md`
- Glossary: `CONTEXT.md`
- Setup/run: `README.md`, `Makefile` (`make help`)
- Issues: https://github.com/allank/riffle_spikes/issues (all closed except #18)
- PRD source doc: on Allan's machine, not in this repo

## Suggested promotions
None identified this session.
