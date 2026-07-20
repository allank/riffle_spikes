# Handoffs

## Active topics
| Topic | Status | Updated |
|---|---|---|
| [embedding-inference-prd](embedding-inference-prd.md) | All spikes done, all tickets closed except parent #16 (pending confirmation). Remaining thread is the PRD's Spike 3 kill/keep decision itself. | 2026-07-20 |

## Session log
- 2026-07-20 — [embedding-inference-prd](embedding-inference-prd.md) — built golden eval + Spike 2 benchmark harnesses, tract sidecar, candle sidecar (CPU done, Metal blocked on hardware). Only #18 (candle Metal) remained open.
- 2026-07-20 (cont.) — measured candle-Metal for real on Apple Silicon (M1), closing #18: correct, and the fastest sidecar yet (~1.7x slower than ONNX). Added `make fetch-bge-model` for automated model-asset setup on fresh machines (#19, closed). Results in `docs/decision-criteria.md` as a new Apple Silicon section, additive to the existing Intel numbers. Only the PRD's kill/keep decision (and parent issue #16) remain open.
