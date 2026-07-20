# Spike Decision Criteria

One row per spike, filled in as each completes, so the final choice is a table read (PRD Section 7). The table below reflects the Intel dev machine used throughout this repo's earlier work; see the "Apple Silicon (Apple M1) results" section further down for the same measurements on Apple Silicon.

| Spike | Throughput (chunks/sec) | Latency (cold / warm) | Cold start | Binary/asset size | Install steps | Cross-compile complexity | Numerical fidelity (golden eval) | Maintenance surface |
|---|---|---|---|---|---|---|---|---|
| Golden eval baseline (ONNX reference) | 7.6 | cold 98.8ms / warm 120.8ms | 821.3ms | | | | nDCG 1.0000, MRR 1.0000 (aggregate, golden eval corpus, `onnxadapter`) | |
| Spike 2 — pure Go | 0.6 | cold 794.1ms / warm 1471.9ms¹ | 1186.6ms | | | | nDCG 1.0000, MRR 1.0000 (aggregate, golden eval corpus, `puregoadapter`); cosine similarity 1.000000 vs ONNX reference on all 5 corpus notes | |
| Spike 3 — Rust sidecar (tract) | 1.5 | cold 1047.6ms / warm 681.7ms² | 1050.2ms | | | | nDCG 1.0000, MRR 1.0000 (aggregate, golden eval corpus, `sidecaradapter`/`tract`); cosine similarity 1.000000 vs ONNX reference on all 5 corpus notes | |
| Spike 3 — Rust sidecar (candle, CPU) | 3.2 | cold 362.6ms / warm 306.1ms⁴ | 367.2ms | | | | nDCG 1.0000, MRR 1.0000 (aggregate, golden eval corpus, `sidecaradapter`/candle CPU); cosine similarity 0.9999999999998 vs ONNX reference on all 5 corpus notes | |
| Spike 3 — Rust sidecar (candle, Metal) | **blocked on hardware⁶** | — | — | | | | — | |

Throughput/latency/cold-start numbers: `spike2_measure/cmd/measure` against the real 1,000-chunk generated corpus (`spike2_measure.GenerateCorpus`, seed 42; chunks 50–400 words, averaging ~225), using onnx_test's local `model.onnx`/`model.safetensors` + `tokenizer.json` (+ `libonnxruntime.dylib` for ONNX). Throughput ratio (ONNX 7.6 / pure-Go 0.6 ≈ 12.7x) is consistent with ADR-0002's addendum (~10-13x after the raw-slice optimization), now measured against realistic chunk lengths rather than short test sentences. tract lands between the two: 1.5 chunks/sec is ~5x slower than ONNX but ~2.5x faster than pure Go — a real, if middling, result, not the "same order of magnitude as ONNX" the PRD's Spike 3 section hoped for. **candle's CPU path does materially better**: 3.2 chunks/sec is only ~2.4x slower than ONNX and more than 2x faster than tract — the PRD's kill criteria ("tract lands materially slower than ONNX Runtime *and* candle's CPU path does too") looks much harder to satisfy on this evidence, since candle's CPU gap to ONNX is meaningfully smaller than tract's.

¹ Pure Go's cold/warm gap isn't a clean warmup signal: `Run`'s warm samples cycle through different corpus chunks (`corpus[i % len(corpus)]`) rather than repeating one, and chunk length varies 50–400 words. Pure Go's cost scales steeply with sequence length (attention is roughly quadratic), so chunk-length variance likely dominates the gap more than genuine warmup — ONNX's much smaller cold/warm gap (98.8ms vs 120.8ms) supports this reading.

² tract's cold/warm gap is a genuine warmup cost, not chunk-length variance (unlike pure Go's¹): `ColdStart` (1050.2ms) ≈ `LatencyCold` (1047.6ms) alone — construction itself took only ~2.6ms, meaning almost the entire "cold" measurement is the *first inference call*, which is consistently ~700ms–1s slower than every subsequent call regardless of chunk length (confirmed in query mode too, see below: cold ~750ms vs warm ~18ms on 2–15 word text). This looks like `tract`'s runnable plan doing significant one-time work (buffer allocation, kernel dispatch setup, or paging in memory-mapped weights) on its first `.run()`, not on model load. Directly answers one of the PRD's own Spike 3 questions ("sidecar lifecycle: per-invocation spawn cost vs. keeping a warm process") — a per-invocation-spawn design would pay this ~1s penalty on every single request; the long-lived process this repo already built avoids it entirely after the first call.

⁴ candle shows the same structural pattern as tract (`ColdStart` ≈ `LatencyCold`, so the cost is in the first inference call, not construction) but the cost itself is much smaller: ~360ms vs tract's ~1050ms in index mode, ~190ms vs ~750ms in query mode (see below). `candle-transformers`' `BertModel` apparently does less first-call setup work than `tract`'s runnable plan — plausibly because it doesn't do the same kind of ahead-of-time graph optimization/kernel-dispatch preparation `tract`'s `into_optimized()`/`into_runnable()` step does, trading a smaller first-call cost for (per the throughput numbers) faster steady-state execution too. Both are real, reproducible numbers (query mode: 3 separate runs each), not single-sample noise.

## Query-time latency (PRD Section 3 goal: <100ms including model load amortisation)

The table above measures index-time chunk latency (50–400 words). The PRD's own goal, and Spike 2's stated hypothesis ("query-time embedding is likely tens of milliseconds — measure to confirm"), is about short search-query-length text specifically. Measured separately via `spike2_measure/cmd/measure -mode query` (`GenerateQueryCorpus`, 50 queries, 2–15 words):

| Engine | Latency (cold) | Latency (warm) | Cold start | Meets <100ms goal? |
|---|---|---|---|---|
| ONNX reference | 7.9ms | 4.1ms | 599.3ms | Yes — both cold and warm |
| Pure Go | 42.1ms | 39.6ms | 476.7ms | Yes — both cold and warm |
| Rust sidecar (tract) | ~750ms³ | 18.4ms | ~750ms³ | Warm: yes. Cold: **no** |
| Rust sidecar (candle, CPU) | ~195ms⁵ | ~18.9ms | ~198ms⁵ | Warm: yes. Cold: **no** |
| Rust sidecar (candle, Metal) | blocked on hardware⁶ | — | — | — |

**Confirmed**: Spike 2's hypothesis holds — pure Go is sufficient at query time (39.6–42.1ms, well under the 100ms goal), even though it's ~12.7x slower than ONNX at indexing throughput. Per the PRD's own framing (Section 8, Spike 2): *"if query latency is fine and indexing a full vault in pure Go is a one-time cost, everything below becomes optimisation rather than requirement."* This means Spike 3 (and beyond) is about improving **indexing throughput**, not query latency — pure Go's query-time performance is not a blocker.

Both sidecars' *warm* query latency (tract 18.4ms, candle ~18.9ms — essentially tied) beats pure Go and comfortably meets the goal — but each pays a first-call warmup cost that the <100ms goal's own "including model load amortisation" wording doesn't obviously cover, since it's separate from model load itself (footnotes ², ⁴): tract's is severe (~700ms–1s), candle's is real but much smaller (~195ms). Neither is a blocker for a long-lived indexing process, but it's a real consideration for either sidecar's query-time UX if ever spawned fresh per query rather than kept warm — and candle is the clearly better-behaved of the two on this specific criterion.

³ Two of three query-mode runs landed at ~750ms cold / ~18ms warm; one outlier run hit 1.25s cold. See the repeat-invocation section below for why: that first run was genuinely disk-cold in a way the other two weren't, but disk-cache warming only explains part of the gap — see below for the full picture.

⁵ Three query-mode runs: cold 197.0ms / 206.6ms / 181.7ms (avg ~195ms), warm 19.2ms / 19.7ms / 17.7ms (avg ~18.9ms), cold start 200.6ms / 209.1ms / 184.2ms (avg ~198ms) — consistent across runs, no outlier the way tract had one. candle's warm query latency (~18.9ms) is essentially identical to tract's (18.4ms) and both comfortably beat pure Go (39.6ms), but candle's cold cost is roughly a quarter of tract's — see footnote ⁴.

⁶ Metal device *construction* (`Device::new_metal(0)`) succeeds, but the first inference call fails: `"running inference: Metal error Failed to create pipeline"` — a Metal compute-pipeline-state creation failure, not a bug in this repo's code. Hardware: a 2020 13" Intel MacBook Pro with integrated Intel Iris Plus Graphics (`system_profiler` reports "Metal Support: Metal 3"), macOS 26.3.1. `candle`'s Metal backend (`candle-metal-kernels`) is developed and tested primarily against Apple Silicon GPUs; a compiled kernel that validates fine on an Apple GPU family can still fail pipeline creation on a different GPU architecture (Intel's integrated graphics) even when the OS reports Metal 3 support at the device level — that flag doesn't guarantee every kernel a given Metal *client library* ships compiles for that specific GPU. No quick fix found (a short search turned up no known issue matching this exact error on Intel integrated graphics). The `CANDLE_DEVICE` env var and `Device::new_metal(0)` construction code itself is implemented and working correctly per the ticket's spec; only the downstream Metal compute path was blocked **on this Intel machine**. **Resolved 2026-07-20**: measured successfully on real Apple Silicon hardware — see the "Apple Silicon (Apple M1)" section below for full correctness and performance numbers. riffle_spikes#18 is closed.

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

## Apple Silicon (Apple M1) results

A second, complete data set — **all five engines**, correctness and both performance modes — measured on real Apple Silicon hardware: MacBook Pro (MacBookPro17,1), Apple M1, 8-core (4 performance + 4 efficiency), 8GB RAM, macOS 26.5.2. This is additive to the Intel numbers above, not a replacement — the two tables above reflect the Intel dev machine used throughout this repo's earlier work; this section is the from-scratch run on different hardware, primarily to unblock candle-Metal (riffle_spikes#18) but captured for every engine so the two machines are directly comparable.

| Spike | Throughput (chunks/sec) | Latency (cold / warm) | Cold start | Numerical fidelity (golden eval) |
|---|---|---|---|---|
| Golden eval baseline (ONNX reference) | 15.4 | cold 35.0ms / warm 64.9ms | 1201.0ms | nDCG 1.0000, MRR 1.0000 (aggregate, golden eval corpus, `onnxadapter`) |
| Spike 2 — pure Go | 0.6 | cold 925.8ms / warm 1621.6ms⁷ | 1074.2ms | nDCG 1.0000, MRR 1.0000; cosine similarity 1.000000 vs ONNX reference on all 5 corpus notes |
| Spike 3 — Rust sidecar (tract) | 6.5 | cold 420.1ms / warm 143.6ms | 422.1ms | nDCG 1.0000, MRR 1.0000; cosine similarity 1.000000 vs ONNX reference on all 5 corpus notes |
| Spike 3 — Rust sidecar (candle, CPU) | 4.9 | cold 570.7ms / warm 178.2ms | 572.4ms | nDCG 1.0000, MRR 1.0000; cosine similarity 1.000000 vs ONNX reference on all 5 corpus notes⁹ |
| Spike 3 — Rust sidecar (candle, Metal) | 9.1 | cold 222.6ms / warm 104.4ms⁸ | 223.7ms | nDCG 1.0000, MRR 1.0000; cosine similarity 1.000000 vs ONNX reference on all 5 corpus notes |

**candle-Metal is correct and, on this hardware, the fastest sidecar option measured so far**: 9.1 chunks/sec is only ~1.7x slower than ONNX (15.4) — closer to ONNX than any sidecar has gotten on either machine (tract on Intel: ~5x slower; candle-CPU on Intel: ~2.4x slower). It's also ~1.9x faster than candle-CPU on this same machine (9.1 vs 4.9), confirming the GPU path is a real, worthwhile win on Apple Silicon specifically — unlike the Intel integrated GPU, which couldn't run it at all (footnote ⁶).

### Query-time latency — Apple Silicon (Apple M1)

Mirroring the Intel-machine query-mode table above (`spike2_measure/cmd/measure -mode query`, same 50-query, 2–15 word corpus):

| Engine | Latency (cold) | Latency (warm) | Cold start | Meets <100ms goal? |
|---|---|---|---|---|
| ONNX reference | 4.3ms | 4.1ms | 312.0ms | Yes — both cold and warm |
| Pure Go | 60.2ms | 64.2ms | 244.2ms | Yes — both cold and warm |
| Rust sidecar (tract) | 654.7ms | 7.7ms | 655.9ms | Warm: yes. Cold: **no** |
| Rust sidecar (candle, CPU) | 401.5ms | 17.6ms | 402.7ms | Warm: yes. Cold: **no** |
| Rust sidecar (candle, Metal) | 607.9ms | 7.5ms⁸ | 609.6ms | Warm: yes. Cold: **no** |

Same shape as the Intel results: every engine comfortably meets the <100ms goal once warm; the two sidecars pay a first-call setup cost that doesn't amortize until the second call. candle-Metal's warm query latency (7.5ms) is the best of any engine measured on either machine, including ONNX's own warm number (4.1ms is still lower in absolute terms, but candle-Metal is now closer to ONNX warm-for-warm than any sidecar has been).

⁷ Pure Go's warm latency (1621.6ms) being *higher* than its cold latency (925.8ms) in index mode is the opposite of the usual warmup pattern, and differs from the Intel machine's pure-Go result (footnote ¹, where warm was also higher than cold but for a documented chunk-length-variance reason). The most likely explanation here is this machine's 8GB of RAM: pure Go's index-mode run processes 1,000 generated chunks back-to-back with no external process boundary, and sustained CPU/memory pressure from that workload on an 8GB M1 plausibly triggers memory pressure or thermal throttling partway through the run that a single early "cold" sample wouldn't show. Not independently confirmed (no memory-pressure telemetry was captured during the run) — flagged as the most plausible reading, not a certainty.

⁸ candle-Metal shows the same `ColdStart ≈ LatencyCold` structural pattern already documented for tract (footnote ²) and candle-CPU (footnote ⁴): 223.7ms cold start vs 222.6ms cold latency in index mode, 609.6ms vs 607.9ms in query mode. This is consistent with Metal compute-pipeline-state creation happening lazily on the first `.run()` call rather than at device/model construction — the same first-call tax GPU compute APIs typically impose, distinct from CPU-only tract/candle-CPU's own (smaller) version of the same phenomenon. Only one run was captured for each mode here (unlike tract's/candle-CPU's repeated-run checks, footnotes ³/⁵) — worth a repeat-run check later if this number matters for a production decision.

⁹ candle-CPU's cosine similarity here is a clean `1.000000` across all 5 notes, versus `0.9999999999998` recorded for the same engine on the Intel machine. This is a real, expected difference, not a regression: `candle`'s CPU backend dispatches to different SIMD code paths per architecture (ARM NEON on Apple Silicon vs x86 AVX/SSE on Intel), and floating-point addition/multiplication is not associative — different instruction orderings produce bit-identical-to-ONNX results on one architecture and a ~1e-13 divergence on another. Both are correct to any practical tolerance; the difference is a floating-point-numerics curiosity, not a correctness concern for either machine.
