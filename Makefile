# Runs this repo's golden eval (correctness) and Spike 2 (performance)
# benchmarks against every real adapter: pure-Go, ONNX, the tract
# sidecar, and the candle sidecar (CPU and Metal).
#
# `make help` lists targets. `make metal-verify` is the one most likely
# useful right after cloning onto Apple Silicon — it runs exactly the
# steps riffle_spikes#18 is waiting on.
#
# Paths below assume the layout this repo's own work has used
# throughout: onnx_test checked out as a sibling directory, and
# ONNX Runtime installed via Homebrew. Override any of these on the
# command line if your setup differs, e.g.:
#   make golden-eval-all ONNX_TEST_DIR=/path/to/onnx_test/inference/bge_bench/data
#   make bench-all ONNX_LIB=/usr/local/lib/libonnxruntime.dylib

ONNX_TEST_DIR ?= ../onnx_test/inference/bge_bench/data
MODEL_ONNX := $(ONNX_TEST_DIR)/model.onnx
MODEL_SAFETENSORS := $(ONNX_TEST_DIR)/model.safetensors
TOKENIZER := $(ONNX_TEST_DIR)/tokenizer.json

# Apple Silicon Homebrew installs to /opt/homebrew; Intel Homebrew (and
# this repo's own prior runs) used /usr/local — override if yours
# differs.
ONNX_LIB ?= /opt/homebrew/lib/libonnxruntime.dylib

# BAAI/bge-small-en-v1.5's HuggingFace repo hosts model.safetensors (root)
# and model.onnx (onnx/ subfolder) as plain static downloads — no
# conversion/export tooling needed, just curl.
HF_REPO := BAAI/bge-small-en-v1.5
HF_BASE := https://huggingface.co/$(HF_REPO)/resolve/main
MIN_MODEL_BYTES := 100000000

TRACT_BINARY := spike3_rust_sidecar/target/release/spike3_rust_sidecar
CANDLE_BINARY := spike3_candle_sidecar/target/release/spike3_candle_sidecar

GOLDENEVAL := go run ./golden_eval/cmd/goldeneval
MEASURE := go run ./spike2_measure/cmd/measure

.PHONY: help
help:
	@echo "Setup:"
	@echo "  fetch-bge-model          download+cache BGE model assets from HuggingFace"
	@echo "                           (also runs automatically as needed by any target below)"
	@echo ""
	@echo "Build:"
	@echo "  build-sidecars           build both Rust sidecars in release mode"
	@echo ""
	@echo "Golden eval (correctness) — comparison mode vs the ONNX reference:"
	@echo "  golden-puregopath-vs-onnx"
	@echo "  golden-tract-vs-onnx"
	@echo "  golden-candle-cpu-vs-onnx"
	@echo "  golden-candle-metal-vs-onnx"
	@echo "  golden-eval-all          all four of the above"
	@echo ""
	@echo "Spike 2 benchmark (performance) — index and query modes:"
	@echo "  bench-<engine>-index / bench-<engine>-query"
	@echo "    where <engine> is puregopath, onnx, tract, candle-cpu, candle-metal"
	@echo "  bench-all                all engines, both modes (index mode is SLOW —"
	@echo "                           pure-Go alone took ~26min in this repo's own runs)"
	@echo "  bench-all-query          all engines, query mode only (fast, ~seconds each)"
	@echo ""
	@echo "Convenience:"
	@echo "  metal-verify             exactly what riffle_spikes#18 needs: candle Metal"
	@echo "                           correctness + both performance modes"
	@echo "  all                      golden-eval-all + bench-all (slow, run overnight-ish)"

.PHONY: build-sidecars build-tract build-candle
build-sidecars: build-tract build-candle

build-tract:
	cd spike3_rust_sidecar && cargo build --release

build-candle:
	cd spike3_candle_sidecar && cargo build --release

# --- Fetch BGE model assets from HuggingFace (real file-targets, so Make's
# own dependency tracking is the cache: a run only fetches if the file is
# entirely absent; size is validated at download time, not re-checked on
# later runs against a file already in place) ---

# Canned recipe, parameterized on the source URL ($(1)); mkdir -> curl to a
# .tmp path -> size-sanity-check-or-die -> mv, so a failed/truncated
# download is never left at the real path for a later run to mistake for
# an already-cached file.
define fetch-and-verify
	@mkdir -p $(dir $@)
	curl -fL -o $@.tmp $(1)
	@size=$$(stat -f%z $@.tmp 2>/dev/null || stat -c%s $@.tmp); \
	  if [ "$$size" -lt $(MIN_MODEL_BYTES) ]; then echo "fetch-bge-model: $(notdir $@) too small ($$size bytes)" >&2; rm -f $@.tmp; exit 1; fi
	mv $@.tmp $@
endef

.PHONY: fetch-bge-model
fetch-bge-model: $(MODEL_ONNX) $(MODEL_SAFETENSORS)

$(MODEL_ONNX):
	$(call fetch-and-verify,$(HF_BASE)/onnx/model.onnx)

$(MODEL_SAFETENSORS):
	$(call fetch-and-verify,$(HF_BASE)/model.safetensors)

# --- Golden eval: single-adapter smoke checks (no ONNX reference needed) ---

.PHONY: golden-stub golden-puregopath golden-onnx golden-tract golden-candle-cpu golden-candle-metal
golden-stub:
	$(GOLDENEVAL)

golden-puregopath: $(MODEL_SAFETENSORS)
	$(GOLDENEVAL) -model $(MODEL_SAFETENSORS) -tokenizer $(TOKENIZER)

golden-onnx: $(MODEL_ONNX)
	$(GOLDENEVAL) -onnx-model $(MODEL_ONNX) -tokenizer $(TOKENIZER) -onnx-lib $(ONNX_LIB)

golden-tract: build-tract $(MODEL_ONNX)
	$(GOLDENEVAL) -sidecar-binary $(TRACT_BINARY) -sidecar-model $(MODEL_ONNX) -tokenizer $(TOKENIZER)

golden-candle-cpu: build-candle $(MODEL_SAFETENSORS)
	$(GOLDENEVAL) -sidecar-binary $(CANDLE_BINARY) -sidecar-model $(MODEL_SAFETENSORS) -tokenizer $(TOKENIZER)

golden-candle-metal: build-candle $(MODEL_SAFETENSORS)
	CANDLE_DEVICE=metal $(GOLDENEVAL) -sidecar-binary $(CANDLE_BINARY) -sidecar-model $(MODEL_SAFETENSORS) -tokenizer $(TOKENIZER)

# --- Golden eval: comparison mode vs the ONNX reference (the numbers that go in docs/decision-criteria.md) ---

.PHONY: golden-puregopath-vs-onnx golden-tract-vs-onnx golden-candle-cpu-vs-onnx golden-candle-metal-vs-onnx golden-eval-all
golden-puregopath-vs-onnx: $(MODEL_SAFETENSORS) $(MODEL_ONNX)
	$(GOLDENEVAL) -model $(MODEL_SAFETENSORS) -tokenizer $(TOKENIZER) -onnx-model $(MODEL_ONNX) -onnx-lib $(ONNX_LIB)

golden-tract-vs-onnx: build-tract $(MODEL_ONNX)
	$(GOLDENEVAL) -sidecar-binary $(TRACT_BINARY) -sidecar-model $(MODEL_ONNX) -tokenizer $(TOKENIZER) -onnx-model $(MODEL_ONNX) -onnx-lib $(ONNX_LIB)

golden-candle-cpu-vs-onnx: build-candle $(MODEL_SAFETENSORS) $(MODEL_ONNX)
	$(GOLDENEVAL) -sidecar-binary $(CANDLE_BINARY) -sidecar-model $(MODEL_SAFETENSORS) -tokenizer $(TOKENIZER) -onnx-model $(MODEL_ONNX) -onnx-lib $(ONNX_LIB)

golden-candle-metal-vs-onnx: build-candle $(MODEL_SAFETENSORS) $(MODEL_ONNX)
	CANDLE_DEVICE=metal $(GOLDENEVAL) -sidecar-binary $(CANDLE_BINARY) -sidecar-model $(MODEL_SAFETENSORS) -tokenizer $(TOKENIZER) -onnx-model $(MODEL_ONNX) -onnx-lib $(ONNX_LIB)

golden-eval-all: golden-puregopath-vs-onnx golden-tract-vs-onnx golden-candle-cpu-vs-onnx golden-candle-metal-vs-onnx

# --- Spike 2 benchmark: index mode (1,000-chunk corpus — slow) and query mode (50 short queries — fast) ---

.PHONY: bench-puregopath-index bench-puregopath-query
.PHONY: bench-onnx-index bench-onnx-query
.PHONY: bench-tract-index bench-tract-query
.PHONY: bench-candle-cpu-index bench-candle-cpu-query
.PHONY: bench-candle-metal-index bench-candle-metal-query
.PHONY: bench-all bench-all-query

bench-puregopath-index: $(MODEL_SAFETENSORS)
	$(MEASURE) -mode index -model $(MODEL_SAFETENSORS) -tokenizer $(TOKENIZER)

bench-puregopath-query: $(MODEL_SAFETENSORS)
	$(MEASURE) -mode query -model $(MODEL_SAFETENSORS) -tokenizer $(TOKENIZER)

bench-onnx-index: $(MODEL_ONNX)
	$(MEASURE) -mode index -onnx-model $(MODEL_ONNX) -tokenizer $(TOKENIZER) -onnx-lib $(ONNX_LIB)

bench-onnx-query: $(MODEL_ONNX)
	$(MEASURE) -mode query -onnx-model $(MODEL_ONNX) -tokenizer $(TOKENIZER) -onnx-lib $(ONNX_LIB)

bench-tract-index: build-tract $(MODEL_ONNX)
	$(MEASURE) -mode index -sidecar-binary $(TRACT_BINARY) -sidecar-model $(MODEL_ONNX) -tokenizer $(TOKENIZER)

bench-tract-query: build-tract $(MODEL_ONNX)
	$(MEASURE) -mode query -sidecar-binary $(TRACT_BINARY) -sidecar-model $(MODEL_ONNX) -tokenizer $(TOKENIZER)

bench-candle-cpu-index: build-candle $(MODEL_SAFETENSORS)
	$(MEASURE) -mode index -sidecar-binary $(CANDLE_BINARY) -sidecar-model $(MODEL_SAFETENSORS) -tokenizer $(TOKENIZER)

bench-candle-cpu-query: build-candle $(MODEL_SAFETENSORS)
	$(MEASURE) -mode query -sidecar-binary $(CANDLE_BINARY) -sidecar-model $(MODEL_SAFETENSORS) -tokenizer $(TOKENIZER)

bench-candle-metal-index: build-candle $(MODEL_SAFETENSORS)
	CANDLE_DEVICE=metal $(MEASURE) -mode index -sidecar-binary $(CANDLE_BINARY) -sidecar-model $(MODEL_SAFETENSORS) -tokenizer $(TOKENIZER)

bench-candle-metal-query: build-candle $(MODEL_SAFETENSORS)
	CANDLE_DEVICE=metal $(MEASURE) -mode query -sidecar-binary $(CANDLE_BINARY) -sidecar-model $(MODEL_SAFETENSORS) -tokenizer $(TOKENIZER)

bench-all: bench-puregopath-index bench-puregopath-query bench-onnx-index bench-onnx-query bench-tract-index bench-tract-query bench-candle-cpu-index bench-candle-cpu-query bench-candle-metal-index bench-candle-metal-query

bench-all-query: bench-puregopath-query bench-onnx-query bench-tract-query bench-candle-cpu-query bench-candle-metal-query

# --- Convenience ---

.PHONY: metal-verify all test
metal-verify: golden-candle-metal-vs-onnx bench-candle-metal-index bench-candle-metal-query

all: golden-eval-all bench-all

test:
	go test ./...
	cd spike3_rust_sidecar && cargo test
	cd spike3_candle_sidecar && cargo test
