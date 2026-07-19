# Spike Decision Criteria

One row per spike, filled in as each completes, so the final choice is a table read (PRD Section 7). Backing data for each row lives under `docs/golden-eval-results/<spike>/`.

| Spike | Throughput (chunks/sec) | Latency (cold / warm) | Cold start | Binary/asset size | Install steps | Cross-compile complexity | Numerical fidelity (golden eval) | Maintenance surface |
|---|---|---|---|---|---|---|---|---|
| Golden eval baseline (ONNX reference) | | | | | | | nDCG 1.0000, MRR 1.0000 (aggregate, golden eval corpus, `onnxadapter`) | |
| Spike 2 — pure Go | | | | | | | nDCG 1.0000, MRR 1.0000 (aggregate, golden eval corpus, `puregoadapter`) | |
| Spike 3 — Rust sidecar (tract/candle) | | | | | | | | |
