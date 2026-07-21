# Design: `fetch-bge-model` Makefile target

## Problem

Running this repo's benchmarks requires `model.onnx` and `model.safetensors`
in `../onnx_test/inference/bge_bench/data/` alongside the already-present
`tokenizer.json`/`config.json`. These two files are gitignored in `onnx_test`
and not distributed via git. Until now, obtaining them was a manual step the
README only vaguely described (copy from an existing checkout, or export via
`sentence-transformers`/`optimum-cli`). On a fresh machine (e.g. the Apple
Silicon hardware issue #18 needs), this is a hard blocker with no automated
path.

## Source of truth

The HuggingFace repo `BAAI/bge-small-en-v1.5` ships both files as static,
directly downloadable objects (no LFS pointer gate, no conversion needed):

- `https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/model.safetensors`
  (~133MB) — standard HF BERT tensor names (`embeddings.word_embeddings.weight`,
  `encoder.layer.N...`, etc.), exactly what `puregopath/model.go` expects.
- `https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx`
  (~133MB) — standard `input_ids`/`attention_mask`/`token_type_ids` →
  `last_hidden_state` I/O shape, exactly what `onnxpath/model.go` expects.

This means a plain `curl` is sufficient — no Python, torch, `optimum-cli`, or
`sentence-transformers` dependency required.

## Mechanism

Add `$(MODEL_ONNX)` and `$(MODEL_SAFETENSORS)` as real Make file-targets
(not phony), each downloading to a `.tmp` path, sanity-checking the size
(reject anything under 100MB — catches truncated downloads or an HTML error
page saved in place of the binary), and only then `mv`-ing into place:

```make
HF_REPO := BAAI/bge-small-en-v1.5
HF_BASE := https://huggingface.co/$(HF_REPO)/resolve/main

$(MODEL_ONNX):
	@mkdir -p $(dir $@)
	curl -fL -o $@.tmp $(HF_BASE)/onnx/model.onnx
	@size=$$(stat -f%z $@.tmp 2>/dev/null || stat -c%s $@.tmp); \
	  if [ "$$size" -lt 100000000 ]; then echo "fetch-bge-model: model.onnx too small ($$size bytes)" >&2; rm -f $@.tmp; exit 1; fi
	mv $@.tmp $@

$(MODEL_SAFETENSORS):
	@mkdir -p $(dir $@)
	curl -fL -o $@.tmp $(HF_BASE)/model.safetensors
	@size=$$(stat -f%z $@.tmp 2>/dev/null || stat -c%s $@.tmp); \
	  if [ "$$size" -lt 100000000 ]; then echo "fetch-bge-model: model.safetensors too small ($$size bytes)" >&2; rm -f $@.tmp; exit 1; fi
	mv $@.tmp $@

.PHONY: fetch-bge-model
fetch-bge-model: $(MODEL_ONNX) $(MODEL_SAFETENSORS)
```

Using real file-targets means Make's own dependency tracking *is* the cache:
present + right-sized → skipped on every subsequent run; missing → fetched
once. The `.tmp`-then-`mv` sequencing means a failed/truncated download is
never left at the real path, so a corrupt partial file can't be mistaken for
a cached one on the next run.

## Wiring into existing targets

Every existing target whose recipe references `$(MODEL_ONNX)` and/or
`$(MODEL_SAFETENSORS)` gets that file added as a prerequisite, the same way
`golden-tract: build-tract` already declares its Rust-binary dependency:

| Target | New prerequisite(s) |
|---|---|
| `golden-onnx` | `$(MODEL_ONNX)` |
| `golden-tract` | `$(MODEL_ONNX)` (in addition to existing `build-tract`) |
| `golden-puregopath` | `$(MODEL_SAFETENSORS)` |
| `golden-candle-cpu` | `$(MODEL_SAFETENSORS)` (in addition to existing `build-candle`) |
| `golden-candle-metal` | `$(MODEL_SAFETENSORS)` (in addition to existing `build-candle`) |
| `golden-puregopath-vs-onnx` | `$(MODEL_SAFETENSORS) $(MODEL_ONNX)` |
| `golden-tract-vs-onnx` | `$(MODEL_ONNX)` (in addition to existing `build-tract`) |
| `golden-candle-cpu-vs-onnx` | `$(MODEL_SAFETENSORS) $(MODEL_ONNX)` (in addition to existing `build-candle`) |
| `golden-candle-metal-vs-onnx` | `$(MODEL_SAFETENSORS) $(MODEL_ONNX)` (in addition to existing `build-candle`) |
| `bench-onnx-index`, `bench-onnx-query` | `$(MODEL_ONNX)` |
| `bench-tract-index`, `bench-tract-query` | `$(MODEL_ONNX)` (in addition to existing `build-tract`) |
| `bench-puregopath-index`, `bench-puregopath-query` | `$(MODEL_SAFETENSORS)` |
| `bench-candle-cpu-index`, `bench-candle-cpu-query` | `$(MODEL_SAFETENSORS)` (in addition to existing `build-candle`) |
| `bench-candle-metal-index`, `bench-candle-metal-query` | `$(MODEL_SAFETENSORS)` (in addition to existing `build-candle`) |

`golden-eval-all`, `bench-all`, `bench-all-query`, `metal-verify`, and `all`
need no changes — they inherit the fetch transitively through the targets
above. `golden-stub` is untouched; it doesn't use real model weights.

## Docs

Update the README's "Machine setup" section: replace the "obtain them
separately" guidance (which named `sentence-transformers`/`optimum-cli
export onnx` as the only path) with a note that any Makefile target fetches
them automatically on first run (or `make fetch-bge-model` to do it
standalone), keeping a short fallback line for anyone who already has
weights from elsewhere (e.g. a private fine-tune) to just copy them into
place instead.

Also add `fetch-bge-model` to the `make help` output, under a new "Setup:"
section above "Build:".

## Out of scope

- Checksum pinning (SHA-256) — explicitly declined in favor of a simpler
  size-sanity check; revisit only if HuggingFace-side corruption/tampering
  becomes a real concern.
- Any change to `onnx_test` itself — the fetch writes into its already-
  gitignored `data/` directory but doesn't modify that repo's tracked
  contents or tooling.
- Retry/backoff logic on the `curl` call — not warranted for a one-time,
  interactively-run fetch.
