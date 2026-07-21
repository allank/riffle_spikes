# Embedding-inference PRD spikes

**Status:** All spikes done, all tickets closed (including parent #16). Candle-Metal measured for real on Apple Silicon (#18 closed) and `make` on a fresh machine is now fully automated (#19 closed). Only remaining open thread is the PRD's Spike 3 kill/keep decision itself — a judgment call, not a ticket, deliberately left open.
**Updated:** 2026-07-20

## Goal
Test local embedding-inference options for Riffle (pure Go, ONNX Runtime, Rust sidecars via tract/candle) with real measured evidence, per the embedding-inference PRD, to decide what Riffle ships with.

## Current state
- Golden eval harness (correctness: nDCG/MRR + cosine similarity vs ONNX) — done, closed (#1–#5).
- Spike 2 benchmark harness (throughput/latency/cold-start, index + query modes) — done, closed (#6–#10).
- Spike 3 tract sidecar — done, closed (#11–#15). 1.5 chunks/sec index throughput, cosine sim 1.000000 vs ONNX — correct but only ~5x slower than ONNX's 7.6, not the "same order of magnitude" the PRD hoped for.
- Spike 3 candle sidecar (CPU) — done, closed (#17, under parent #16). 3.2 chunks/sec, cosine sim 0.9999999999998 — materially better than tract (~2.4x slower than ONNX, not ~5x), meaningfully changes the PRD's kill-criteria calculus in candle's favor.
- Spike 3 candle sidecar (Metal) — **done, closed (#18)**. Measured on a second machine — MacBook Pro (MacBookPro17,1), Apple M1, 8-core, 8GB RAM — since the original dev machine was Intel (2020 13" MBP, Iris Plus Graphics) and couldn't run Metal at all. Correct (cosine sim 1.000000 vs ONNX) and the fastest sidecar option measured so far: 9.1 chunks/sec index throughput, only ~1.7x slower than ONNX (15.4) — closer than tract's ~5x or candle-CPU's ~2.4x, both measured on Intel. Also ~1.9x faster than candle-CPU on the same Apple Silicon machine. Full numbers in `docs/decision-criteria.md`'s "Apple Silicon (Apple M1) results" section, additive to (not replacing) the original Intel numbers.
- While re-running benchmarks on the Apple Silicon machine, hit a real onboarding gap: `onnx_test`'s model assets (`model.onnx`/`model.safetensors`) had no automated way to obtain on a fresh machine. Spec'd, implemented, reviewed, and closed as **#19**: `make fetch-bge-model` downloads both directly from HuggingFace's `BAAI/bge-small-en-v1.5` repo via `curl` (no Python/optimum-cli needed — both files are hosted there as plain static downloads), wired as a transparent prerequisite of every target that needs them, cached via Make's own file-existence tracking.
- Also fixed, as a side effect of getting this machine working: `~/.cargo/bin` had stale symlinks from an old `rustup-init` install pointing at a long-gone Cellar path, shadowing the working Homebrew-managed `rustup` toolchain. Not a repo issue — a local machine fix, done directly (not ticketed).
- Added `Makefile` (`make metal-verify` built specifically for the Apple Silicon follow-up, now includes `fetch-bge-model`) and `README.md` for setup/onboarding, both up to date.
- `CANDLE_DEVICE=auto` (canary-tested Metal-with-CPU-fallback) was designed in conversation but explicitly deferred — user wants clean explicit CPU-vs-Metal runs first. Still deferred; not revisited this session.

## Decisions
- Each spike went through grill/spec → tickets → implement → two-axis code review, with real hardware verification (no committed test touches real model weights — established repo convention). The same pattern (brainstorm → spec → issue → implement → two-axis review) was reused for #19 even though it's tooling, not a spike.
- `riffle_spikes` never gets its own ADRs — consequential findings graduate to `riffle`'s ADRs instead.
- Golden eval corpus is hand-authored/synthetic; Spike 2 corpus is generated (1,000 chunks for index mode, 50 short queries for query mode).
- `CANDLE_DEVICE` semantics (unset/`cpu`→CPU, `metal`→force+error) kept exactly as shipped — no default-behavior change, to protect the Makefile's `*-cpu-*` target meanings.
- Per-machine results are additive, never overwritten: `docs/decision-criteria.md` keeps the original Intel table fully intact and adds a separate Apple Silicon section, rather than merging/replacing numbers — the two machines' results are genuinely different data points (different hardware), not corrections of each other.

## Open threads
- **PRD's Spike 3 kill/keep decision** — not yet formally made. All five numbers now exist (ONNX/pure-Go/tract/candle-CPU/candle-Metal, on both Intel and Apple Silicon); the evidence is fully in hand, candle looks materially stronger than tract on both counts (CPU and Metal), but the actual go/no-go call and any resulting `riffle` ADR is a decision for Allan to make, not something resolved by this repo's tickets.
- Full-vault indexing time (~10k chunks, PRD Goals section) never measured — out of scope so far.
- `CANDLE_DEVICE=auto` design (Metal-with-CPU-fallback) — deferred, not built.

## Next steps
1. Make the PRD's Spike 3 kill/keep call with all five numbers in hand — likely candle becomes the Horizon 2 feasibility gate per the PRD, but this is Allan's decision to make, not an agent's.
2. If that decision is consequential, graduate it into a `riffle` ADR per this repo's established convention (`riffle_spikes` doesn't keep its own ADRs).
3. If still wanted: implement `CANDLE_DEVICE=auto` (design already discussed in an earlier session).
4. If ever revisited: full-vault indexing time at ~10k chunks (PRD Goals section), currently unmeasured.

## Artifacts
- Results: `docs/decision-criteria.md` (Intel table + Apple Silicon section)
- Glossary: `CONTEXT.md`
- Setup/run: `README.md`, `Makefile` (`make help`, now includes `fetch-bge-model`)
- Specs: `docs/specs/2026-07-20-fetch-bge-model-design.md`
- Issues: https://github.com/allank/riffle_spikes/issues — all closed
- PRD source doc: on Allan's machine, not in this repo

## Suggested promotions
- The Spike 3 kill/keep decision (candle vs. tract vs. ONNX, CPU + Metal, both machines) is exactly the kind of consequential finding this repo's convention says should graduate into a `riffle` ADR once made — flagging it here so it isn't lost, not writing it prematurely on Allan's behalf.
