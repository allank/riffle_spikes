# Spike Decision Criteria

One row per spike, filled in as each completes, so the final choice is a table read (PRD Section 7). Backing data for each row lives under `docs/golden-eval-results/<spike>/`.

| Spike | Throughput (chunks/sec) | Latency (cold / warm) | Cold start | Binary/asset size | Install steps | Cross-compile complexity | Numerical fidelity (golden eval) | Maintenance surface |
|---|---|---|---|---|---|---|---|---|
| Golden eval baseline (ONNX reference) | 7.6 | cold 98.8ms / warm 120.8ms | 821.3ms | | | | nDCG 1.0000, MRR 1.0000 (aggregate, golden eval corpus, `onnxadapter`) | |
| Spike 2 — pure Go | 0.6 | cold 794.1ms / warm 1471.9ms¹ | 1186.6ms | | | | nDCG 1.0000, MRR 1.0000 (aggregate, golden eval corpus, `puregoadapter`); cosine similarity 1.000000 vs ONNX reference on all 5 corpus notes | |
| Spike 3 — Rust sidecar (tract/candle) | | | | | | | | |

Throughput/latency/cold-start numbers: `spike2_measure/cmd/measure` against the real 1,000-chunk generated corpus (`spike2_measure.GenerateCorpus`, seed 42; chunks 50–400 words, averaging ~225), using onnx_test's local `model.onnx`/`model.safetensors` + `tokenizer.json` (+ `libonnxruntime.dylib` for ONNX). Throughput ratio (ONNX 7.6 / pure-Go 0.6 ≈ 12.7x) is consistent with ADR-0002's addendum (~10-13x after the raw-slice optimization), now measured against realistic chunk lengths rather than short test sentences.

¹ Pure Go's cold/warm gap isn't a clean warmup signal: `Run`'s warm samples cycle through different corpus chunks (`corpus[i % len(corpus)]`) rather than repeating one, and chunk length varies 50–400 words. Pure Go's cost scales steeply with sequence length (attention is roughly quadratic), so chunk-length variance likely dominates the gap more than genuine warmup — ONNX's much smaller cold/warm gap (98.8ms vs 120.8ms) supports this reading.

## Query-time latency (PRD Section 3 goal: <100ms including model load amortisation)

The table above measures index-time chunk latency (50–400 words). The PRD's own goal, and Spike 2's stated hypothesis ("query-time embedding is likely tens of milliseconds — measure to confirm"), is about short search-query-length text specifically. Measured separately via `spike2_measure/cmd/measure -mode query` (`GenerateQueryCorpus`, 50 queries, 2–15 words):

| Engine | Latency (cold) | Latency (warm) | Cold start | Meets <100ms goal? |
|---|---|---|---|---|
| ONNX reference | 7.9ms | 4.1ms | 599.3ms | Yes — both cold and warm |
| Pure Go | 42.1ms | 39.6ms | 476.7ms | Yes — both cold and warm |

**Confirmed**: Spike 2's hypothesis holds — pure Go is sufficient at query time (39.6–42.1ms, well under the 100ms goal), even though it's ~12.7x slower than ONNX at indexing throughput. Per the PRD's own framing (Section 8, Spike 2): *"if query latency is fine and indexing a full vault in pure Go is a one-time cost, everything below becomes optimisation rather than requirement."* This means Spike 3 (and beyond) is about improving **indexing throughput**, not query latency — pure Go's query-time performance is not a blocker.
