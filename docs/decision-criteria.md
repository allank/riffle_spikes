# Spike Decision Criteria

One row per spike, filled in as each completes, so the final choice is a table read (PRD Section 7). Backing data for each row lives under `docs/golden-eval-results/<spike>/`.

| Spike | Throughput (chunks/sec) | Latency (cold / warm) | Cold start | Binary/asset size | Install steps | Cross-compile complexity | Numerical fidelity (golden eval) | Maintenance surface |
|---|---|---|---|---|---|---|---|---|
| Golden eval baseline (ONNX reference) | 7.6 | cold 98.8ms / warm 120.8ms | 821.3ms | | | | nDCG 1.0000, MRR 1.0000 (aggregate, golden eval corpus, `onnxadapter`) | |
| Spike 2 — pure Go | 0.6 | cold 794.1ms / warm 1471.9ms¹ | 1186.6ms | | | | nDCG 1.0000, MRR 1.0000 (aggregate, golden eval corpus, `puregoadapter`); cosine similarity 1.000000 vs ONNX reference on all 5 corpus notes | |
| Spike 3 — Rust sidecar (tract/candle) | 1.5 | cold 1047.6ms / warm 681.7ms² | 1050.2ms | | | | nDCG 1.0000, MRR 1.0000 (aggregate, golden eval corpus, `sidecaradapter`/`tract`); cosine similarity 1.000000 vs ONNX reference on all 5 corpus notes | |

Throughput/latency/cold-start numbers: `spike2_measure/cmd/measure` against the real 1,000-chunk generated corpus (`spike2_measure.GenerateCorpus`, seed 42; chunks 50–400 words, averaging ~225), using onnx_test's local `model.onnx`/`model.safetensors` + `tokenizer.json` (+ `libonnxruntime.dylib` for ONNX). Throughput ratio (ONNX 7.6 / pure-Go 0.6 ≈ 12.7x) is consistent with ADR-0002's addendum (~10-13x after the raw-slice optimization), now measured against realistic chunk lengths rather than short test sentences. tract lands between the two: 1.5 chunks/sec is ~5x slower than ONNX but ~2.5x faster than pure Go — a real, if middling, result, not the "same order of magnitude as ONNX" the PRD's Spike 3 section hoped for.

¹ Pure Go's cold/warm gap isn't a clean warmup signal: `Run`'s warm samples cycle through different corpus chunks (`corpus[i % len(corpus)]`) rather than repeating one, and chunk length varies 50–400 words. Pure Go's cost scales steeply with sequence length (attention is roughly quadratic), so chunk-length variance likely dominates the gap more than genuine warmup — ONNX's much smaller cold/warm gap (98.8ms vs 120.8ms) supports this reading.

² tract's cold/warm gap is a genuine warmup cost, not chunk-length variance (unlike pure Go's¹): `ColdStart` (1050.2ms) ≈ `LatencyCold` (1047.6ms) alone — construction itself took only ~2.6ms, meaning almost the entire "cold" measurement is the *first inference call*, which is consistently ~700ms–1s slower than every subsequent call regardless of chunk length (confirmed in query mode too, see below: cold ~750ms vs warm ~18ms on 2–15 word text). This looks like `tract`'s runnable plan doing significant one-time work (buffer allocation, kernel dispatch setup, or paging in memory-mapped weights) on its first `.run()`, not on model load. Directly answers one of the PRD's own Spike 3 questions ("sidecar lifecycle: per-invocation spawn cost vs. keeping a warm process") — a per-invocation-spawn design would pay this ~1s penalty on every single request; the long-lived process this repo already built avoids it entirely after the first call.

## Query-time latency (PRD Section 3 goal: <100ms including model load amortisation)

The table above measures index-time chunk latency (50–400 words). The PRD's own goal, and Spike 2's stated hypothesis ("query-time embedding is likely tens of milliseconds — measure to confirm"), is about short search-query-length text specifically. Measured separately via `spike2_measure/cmd/measure -mode query` (`GenerateQueryCorpus`, 50 queries, 2–15 words):

| Engine | Latency (cold) | Latency (warm) | Cold start | Meets <100ms goal? |
|---|---|---|---|---|
| ONNX reference | 7.9ms | 4.1ms | 599.3ms | Yes — both cold and warm |
| Pure Go | 42.1ms | 39.6ms | 476.7ms | Yes — both cold and warm |
| Rust sidecar (tract) | ~750ms³ | 18.4ms | ~750ms³ | Warm: yes. Cold: **no** |

**Confirmed**: Spike 2's hypothesis holds — pure Go is sufficient at query time (39.6–42.1ms, well under the 100ms goal), even though it's ~12.7x slower than ONNX at indexing throughput. Per the PRD's own framing (Section 8, Spike 2): *"if query latency is fine and indexing a full vault in pure Go is a one-time cost, everything below becomes optimisation rather than requirement."* This means Spike 3 (and beyond) is about improving **indexing throughput**, not query latency — pure Go's query-time performance is not a blocker.

The sidecar's *warm* query latency (18.4ms) beats pure Go and comfortably meets the goal — but its very first query after the process starts pays the same ~700ms–1s warmup cost documented in the index-time table above (footnote ²), which the <100ms goal's own "including model load amortisation" wording doesn't obviously cover for a cost this large or one that's separate from model load itself. Not a blocker for a long-lived indexing process, but a real consideration for the sidecar's query-time UX if it were ever spawned fresh per query rather than kept warm.

³ Two of three query-mode runs landed at ~750ms cold / ~18ms warm; one outlier run hit 1.25s cold. See the repeat-invocation section below for why: that first run was genuinely disk-cold in a way the other two weren't, but disk-cache warming only explains part of the gap — see below for the full picture.

## Repeat-invocation caching: ONNX benefits, tract mostly doesn't

All numbers above come from single fresh CLI invocations. A real question (raised after reviewing the results): does "cold start" mean something different across *separate* process invocations, once the OS has already read a given model file once? Checked by running the ONNX query-mode benchmark three times as fresh, separate `go run` invocations back to back:

| Run | Latency (cold) | Cold start | Implied construction time (cold start − latency cold) |
|---|---|---|---|
| 1 (first ever) | 8.6ms | 641.8ms | ~633ms |
| 2 | 5.0ms | 372.4ms | ~367ms |
| 3 | 5.2ms | 367.3ms | ~362ms |

**ONNX's construction time (session creation, which reads the 127MB `model.onnx` file) drops by ~40% after the first invocation and then plateaus** — consistent with the OS page cache keeping the file's blocks in RAM across separate processes. The embed call itself (`LatencyCold`) was already small and stable throughout; the entire improvement is in construction.

Contrast with tract's own three query-mode runs (footnote ³ above): 1.25s → 750ms → 748ms. Unlike ONNX, tract's `ColdStart ≈ LatencyCold` in every run — construction was already near-zero from the start (footnote ²), so there's no construction-time component for page-caching to speed up. The drop from run 1 to run 2 (1.25s → 750ms) is plausibly a smaller, secondary page-cache effect on top of tract's own first-`.run()` setup cost, but that setup cost itself doesn't go away — it plateaus at ~750ms rather than continuing toward something ONNX-like.

**Net effect: ONNX's real advantage over tract on cold start is larger than the headline numbers already suggested.** A CLI invoked repeatedly through a work session would mostly see ONNX's cached ~370ms construction cost, not its first-ever 642ms — while tract's ~700–750ms first-call cost doesn't have an equivalent discount. This reinforces, rather than changes, the sidecar-lifecycle conclusion already drawn: a long-lived process (paying tract's setup cost once, ever) remains the right architecture regardless of this caching effect, since it makes the per-invocation question moot in production use.
