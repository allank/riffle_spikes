# Context: riffle_spikes

Glossary for this repo. No implementation details — see `docs/adr/` for decisions and their rationale.

## Terms

**riffle_spikes** (this repo)
Net-new territory for the spikes described in the embedding-inference PRD (`Downloads/embedding-inference-prd.md`). Distinct from `onnx_test`: `onnx_test` is the frozen, closed-out pure-Go-vs-ONNX benchmark harness that produced the numbers behind `riffle` ADR-0002. `riffle_spikes` may import `onnx_test`'s Go packages (e.g. `bge_bench/tokenizer`) where useful, but does not rewrite or supersede them.

**Golden eval**
The Section 6 prerequisite harness: a frozen corpus, a fixed query set, and expected rankings per query, scored via nDCG/MRR plus per-stage cosine similarity against the reference ONNX embedding output. Every spike's retrieval-quality claim is measured against this, not against vector-level closeness alone (RRF fusion can mask or amplify small drift in the underlying vectors).

**Golden eval corpus**
Synthetic, not sourced from a real vault or bootstrapped from `riffle`'s existing output. Notes and queries are hand-authored so that expected relevance is true by construction.

**Distractor (note)**
A synthetic note deliberately authored to be topically adjacent to a query but the wrong facet of it — included in the corpus specifically to test whether an embedding approach can discriminate close-but-wrong matches, not just obviously-unrelated ones.

**Spike**
One numbered investigation from the PRD (Spikes 1–6), each with its own assumed effort, questions to answer, and kill criteria. Sequenced per the PRD: golden eval (Section 6) → Spike 2 (pure-Go measurement) → Spike 3 (Rust sidecar) → others as triggered.

**Embedder (interface)**
The single contract every spike's embedding approach implements: batch of text chunks in, vectors out, tolerant of an out-of-process implementation (needed once Spike 3's sidecar exists). Defined in `golden_eval` from the start — pulled forward from Spike 2, which named it as a requirement — rather than deferred until Spike 2 begins. `onnx_test`'s pure-Go and ONNX paths get thin adapters onto this interface rather than being called directly.

**Decision criteria table**
The Section 7 comparison table (throughput, latency, cold start, binary size, install steps, cross-compile complexity, numerical fidelity, maintenance surface) recording one row per spike as it completes. Lives in `riffle_spikes` (not `onnx_test`, despite the PRD's original wording), since that's where the spikes producing the numbers now run.

**Sidecar (Spike 3)**
A standalone Rust binary (`spike3_rust_sidecar/`), spawned as a child process and communicated with over stdio, rather than linked in-process. First implementation uses `tract` (loads the existing `model.onnx` file as-is, no conversion — direct comparability with the ONNX baseline) over an ndjson protocol: one JSON request line per `Embed` call (a batch of chunks), one JSON response line back (the batch of vectors). Tokenises internally with the HF `tokenizers` crate, so it receives raw text over the wire, not pre-tokenized IDs — matching `onnxadapter`'s shape, not `puregoadapter`'s. The sidecar process is long-lived (spawned once at adapter construction, killed on `Close`), not respawned per call — the existing `Embedder`-consuming harnesses (`golden_eval`, `spike2_measure`) already assume a constructed-once, called-many-times embedder. `candle` (Metal-on-macOS option) is deferred to a follow-up spec once this sidecar's plumbing exists.
