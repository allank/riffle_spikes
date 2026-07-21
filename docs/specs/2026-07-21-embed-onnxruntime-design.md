# Design: embedding ONNX Runtime into the CLI (no `brew install`)

## Problem

ONNX Runtime is the fastest engine measured in `docs/decision-criteria.md`
by a wide margin (7.6–15.4 chunks/sec vs. candle's 3.2–9.1), but this
repo's own README currently tells a user to run `brew install
onnxruntime` before the ONNX adapter works at all. That's a macOS-only,
Homebrew-only instruction with no Linux or Windows equivalent documented
or tested. If ONNX is the engine Riffle ships with, a CLI user shouldn't
need a package manager, a matching install path, or any manual step
before `riffle index` works — the binary itself needs to be enough.

This spec scopes a feasibility investigation, not a finished mechanism:
unlike `fetch-bge-model` (a solved problem), there are real open
questions here — particularly a platform gap discovered while writing
this spec (see below) — that need verifying empirically before this
becomes a ticket.

## What "embedding" actually means here

This repo's ONNX adapter goes through `github.com/yalue/onnxruntime_go`
(`golden_eval/adapters/onnxadapter/adapter.go` → `onnx_test`'s
`onnxpath` package), which loads ONNX Runtime at **runtime**, not link
time: `ort.SetSharedLibraryPath(libPath)` followed by
`ort.InitializeEnvironment()`. It's a `dlopen`-based binding (via
`purego`), not `cgo`-linked against the library at build time. That
means "statically embed ONNX" doesn't mean a linker-level static link —
it means: ship the shared library's bytes inside the Go binary via
`go:embed`, extract them to a cache path on first run, and point
`SetSharedLibraryPath` at that path instead of a Homebrew-installed one.
No cgo toolchain requirement changes as a result; the binding already
doesn't need one.

## Source of truth: what Microsoft actually publishes

`microsoft/onnxruntime`'s GitHub Releases ship prebuilt, MIT-licensed
shared-library archives, one per platform/arch, as plain downloadable
assets (checked directly via `gh release view`, not assumed):

| Platform | Asset | Size (v1.27.1) |
|---|---|---|
| Linux x64 | `onnxruntime-linux-x64-<ver>.tgz` | ~8MB |
| Linux arm64 | `onnxruntime-linux-aarch64-<ver>.tgz` | ~7MB |
| macOS arm64 | `onnxruntime-osx-arm64-<ver>.tgz` | ~30MB |
| Windows x64 | `onnxruntime-win-x64-<ver>.zip` | ~73MB |
| Windows arm64 | `onnxruntime-win-arm64-<ver>.zip` | ~74MB |

MIT license (confirmed via the repo's `LICENSE` file) — redistribution
inside a bundled CLI is unrestricted, no attribution mechanism beyond
keeping the license text needed.

### The gap: no macOS x64 (Intel) prebuilt since v1.25.0

Microsoft stopped publishing `onnxruntime-osx-x86_64-*.tgz` and
`onnxruntime-osx-universal2-*.tgz` starting with v1.25.0 (confirmed:
present through v1.23.0, absent from v1.25.0 onward — v1.24.0's release
page returned no assets at all via the API, so the exact cutover point
between 1.23.0 and 1.25.0 wasn't pinned further). Only `osx-arm64` ships
now.

This isn't a hypothetical edge case — it directly affects this repo's
own dev machine (Intel MacBook Pro). `brew list onnxruntime --versions`
on this machine reports **1.25.1**, i.e. Homebrew's formula is building
onnxruntime from source for Intel Mac rather than using Microsoft's
prebuilt binary (which doesn't exist for that combination at that
version). That's *why* `brew install onnxruntime` still works here
today — Homebrew is doing work this embedding approach can't shortcut
by just downloading an asset.

Options if Intel Mac support matters to Riffle's user base: pin to
Microsoft's last osx-x64 release (v1.23.0) as the embedded version, or
accept Apple Silicon-only support for the embedded-ONNX path and let
Intel Mac users fall back to the existing `brew install` + `-onnx-lib`
override. Not resolved in this spec — flagging for the decision, not
deciding it here.

## Proposed mechanism (to verify empirically, not yet built)

1. Per target platform/arch, download the matching Microsoft release
   asset at build time (a new Makefile step, same shape as
   `fetch-bge-model`'s size-sanity-checked `curl`), unpack the shared
   library into a location `go:embed` can reach.
2. `go:embed` the single shared-library file appropriate to the build's
   `GOOS`/`GOARCH` (build-tag-gated per platform, so a given compiled
   binary only embeds its own platform's library, not all five).
3. At CLI startup, check a cache path (e.g.
   `$XDG_CACHE_HOME/riffle/onnxruntime-<version>-<os>-<arch>/`) for an
   already-extracted library of the right size; if absent, write the
   embedded bytes there once.
4. Call `ort.SetSharedLibraryPath` with that path before
   `InitializeEnvironment` — everything downstream of that is unchanged
   from the current adapter.

## Open questions this spike needs to answer

- Does the extract-once-then-`dlopen` pattern actually work unmodified
  on Linux and Windows, or does either platform impose something the
  current macOS-only testing hasn't hit (e.g. Windows DLL search-path
  rules, SELinux/AppArmor confinement on Linux)?
- macOS Gatekeeper/quarantine: does a dylib extracted to a cache
  directory at runtime (rather than installed via a signed installer or
  Homebrower) get quarantine-flagged or require ad-hoc codesigning
  before `dlopen` will load it?
- Binary size impact: embedding adds ~7–74MB depending on target
  platform (Windows roughly 10x Linux's size) — is that acceptable for
  a CLI distributed via `go install`/direct download, or does it argue
  for a separate first-run download instead of `go:embed` for the
  larger platforms?
- Resolve the Intel-Mac gap above (pin to v1.23.0, or drop that
  platform from the embedded path).

## Verification / acceptance criteria (if this becomes a ticket)

- A CLI in this repo (following the existing spike pattern) embeds the
  platform-appropriate ONNX Runtime library, extracts and loads it with
  zero manual install step, and produces correct golden-eval results
  (nDCG/MRR, cosine similarity vs. the existing `brew`-installed
  reference) — verified on at least this machine.
- Second-run startup doesn't re-extract (cache hit, sanity-checked by
  size like `fetch-bge-model` does for model assets).
- The Intel-Mac decision above is made explicitly, not left implicit.

## Out of scope

- GPU/CUDA builds — irrelevant to Riffle's CPU-only inference use case;
  Microsoft's gpu-variant assets are ignored entirely.
- Deciding the ONNX-vs-candle kill/keep call itself — that's the
  separate, already-flagged-open Spike 3 decision in
  `handoffs/embedding-inference-prd.md`. This spec only removes one
  input to that decision (ONNX's distribution story), it doesn't make
  the call.
- Building the sidecar-elimination path via cgo-embedded candle
  (discussed and explicitly declined this session — warm latency is
  already ~19ms, and the measured sidecar construction cost is ~2.6ms,
  making that overhead immaterial next to steady-state inference cost).
