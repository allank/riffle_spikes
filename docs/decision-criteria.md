# Spike Decision Criteria

One row per spike, filled in as each completes, so the final choice is a table read (PRD Section 7). Backing data for each row lives under `docs/golden-eval-results/<spike>/`.

| Spike | Throughput (chunks/sec) | Latency (cold / warm) | Cold start | Binary/asset size | Install steps | Cross-compile complexity | Numerical fidelity (golden eval) | Maintenance surface |
|---|---|---|---|---|---|---|---|---|
| Golden eval baseline (ONNX reference) | 7.6 | cold 98.8ms / warm 120.8ms | 821.3ms | | | | nDCG 1.0000, MRR 1.0000 (aggregate, golden eval corpus, `onnxadapter`) | |
| Spike 2 — pure Go | 0.6 | cold 794.1ms / warm 1471.9ms¹ | 1186.6ms | | | | nDCG 1.0000, MRR 1.0000 (aggregate, golden eval corpus, `puregoadapter`); cosine similarity 1.000000 vs ONNX reference on all 5 corpus notes | |
| Spike 3 — Rust sidecar (tract/candle) | | | | | | | | |

Throughput/latency/cold-start numbers: `spike2_measure/cmd/measure` against the real 1,000-chunk generated corpus (`spike2_measure.GenerateCorpus`, seed 42; chunks 50–400 words, averaging ~225), using onnx_test's local `model.onnx`/`model.safetensors` + `tokenizer.json` (+ `libonnxruntime.dylib` for ONNX). Throughput ratio (ONNX 7.6 / pure-Go 0.6 ≈ 12.7x) is consistent with ADR-0002's addendum (~10-13x after the raw-slice optimization), now measured against realistic chunk lengths rather than short test sentences.

¹ Pure Go's cold/warm gap isn't a clean warmup signal: `Run`'s warm samples cycle through different corpus chunks (`corpus[i % len(corpus)]`) rather than repeating one, and chunk length varies 50–400 words. Pure Go's cost scales steeply with sequence length (attention is roughly quadratic), so chunk-length variance likely dominates the gap more than genuine warmup — ONNX's much smaller cold/warm gap (98.8ms vs 120.8ms) supports this reading.
