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

## Implementation outcome (riffle_spikes#20)

Two things changed materially between this spec and what shipped, both
discovered during implementation rather than anticipated up front.

### The mechanism moved from `go:embed` to download-and-cache

`go:embed`-ing the shared library into the binary was the original plan
(see "Proposed mechanism" above), but it means every compiled binary
carries the library's full size forever, and needing the asset present
on the *build* machine for every target platform. Since
`yalue/onnxruntime_go` already loads ONNX Runtime via runtime `dlopen`
(not a link-time dependency — see "What 'embedding' actually means
here" above), there's no reason the bytes need to be compiled in at all.
`embeddedonnx` instead downloads the platform's release tarball on first
use, extracts the one shared-library member it needs (via Go's stdlib
`archive/tar`/`compress/gzip` — no shell-out to `tar`), and caches it at
`$XDG_CACHE_HOME/riffle/onnxruntime-<version>-<os>-<arch>/`. A cache hit
costs zero network calls — the check happens before any download is
attempted. This is the same download-once-cache-forever shape this
repo's own `fetch-bge-model` Makefile target already uses for the BGE
model weights, just applied to ONNX Runtime and moved from build time to
CLI runtime.

This resolves the "binary size impact" open question directly: binary
size no longer depends on target platform at all, so there's no
Windows-is-10x-bigger tradeoff to make.

### Apple Silicon only, for a sharper reason than expected — and why

The Intel-Mac gap (see above) turned out to be more than "Microsoft
stopped shipping an osx-x64 binary." `yalue/onnxruntime_go` v1.29.0
(already pinned in this repo's `go.mod`) hardcodes a request for ONNX
Runtime **API version 25** — the C header shipped with each Go binding
version fixes this at compile time (`ort_api =
api_base->GetApi(ORT_API_VERSION)` in `onnxruntime_wrapper.c`). Checking
each ONNX Runtime release's own bundled C header directly (not assumed)
against Microsoft's source tree:

| ONNX Runtime version | Max API version supported |
|---|---|
| v1.23.0 (last with an official osx-x64 build) | 23 |
| v1.24.1 | 24 |
| **v1.25.0** (first version without an osx-x64 build) | **25** |

The x64-binary gap and the API-version gap land on the exact same
release boundary. There is no ONNX Runtime version where "Microsoft
publishes an official osx-x64 binary" and "supports API 25" are both
true — confirmed empirically, not just by reading headers: `make
golden-onnx ONNX_LIB=/usr/local/lib/libonnxruntime.dylib` against this
machine's Homebrew-installed 1.25.1 works fine (nDCG 1.0000), proving
1.25.1 does support API 25 — but Homebrew's Intel build is compiled from
source, not downloaded from Microsoft, which is exactly the workaround
gap already flagged above. Pinning to v1.23.0 for Intel Mac (the
original plan) was tried and fails at the first inference call: `The
requested API version [25] is not available, only API versions [1, 23]
are supported in this build.`

**Decision**: `embeddedonnx` supports darwin/arm64 only for now, pinned
to v1.27.1 (current latest; Microsoft still publishes an official arm64
build at every release). darwin/amd64 is a registered-but-absent entry
in `embeddedonnx`'s platform manifest with a clear error, not a silent
gap. Intel Mac users are unaffected — `-onnx-lib` + `brew install
onnxruntime` continues to work exactly as before.

**Path to Intel Mac support, if wanted later**: not a version-pin
change — Go modules resolve one version of a given import path for the
whole build, so there's no way to make `yalue/onnxruntime_go` request
API 22 for one `GOARCH` and API 25 for another. The real fix is a
second, build-tag-selected backend behind the same `goldeneval.Embedder`
interface every adapter already implements: a minimal, hand-written cgo
bridge against ONNX Runtime's C API hardcoded to an older
`ORT_API_VERSION`, used only when building for darwin/amd64, alongside
(not replacing) today's `onnxadapter`-backed path for every other
platform. This is a genuinely separate, sizeable unit of work — it means
maintaining a second ONNX Runtime C binding, not a config change — so
it's scoped as its own follow-up ticket rather than folded into #20.

### Verified, and what's still hardware-blocked

- **Download+extract functions, verified directly**: `resolveAsset`,
  `downloadBytes`, and `extractTarMember` — the pieces that don't depend
  on `runtime.GOOS`/`GOARCH` — were called directly (bypassing `Path()`,
  which does depend on them) against the real `microsoft/onnxruntime`
  v1.27.1 osx-arm64 release asset from this Intel machine: tarball
  downloaded at exactly the expected 31,959,937 bytes, extracted
  `libonnxruntime.1.27.1.dylib` member matched the expected 38,502,216
  bytes exactly. `Path()` itself has **not** been run end-to-end on any
  machine yet, arm64 included — it refuses to run at all on this
  darwin/amd64 machine by design (`resolveAsset` has no darwin/amd64
  entry), so the actual code path a user hits is still unverified as a
  whole, only piece-by-piece.
- **Gatekeeper/quarantine: partially checked, not resolved**. Files
  written via `os.WriteFile`+`os.Rename` (the same primitive
  `extractTo` uses) don't receive the `com.apple.quarantine` extended
  attribute on this machine — confirmed by `xattr -l` on a file written
  through that exact code path; only the non-blocking
  `com.apple.provenance` attribute is present. That answers "does the
  write itself get quarantined" (no), but the AC's real question —
  whether `dlopen` hits any Gatekeeper friction at load time — is still
  unverified, since no downloaded library has actually been `dlopen`'d
  in this diff, on any machine.
- **Still blocked on hardware**: the full pipeline — `Path()` end to
  end, `dlopen`, and a real golden-eval run against the extracted
  library — needs actual Apple Silicon. Same situation issue #18
  (candle-Metal) was in before real M1 hardware became available. `make
  golden-onnx-embedded` on Apple Silicon is the next step, and the one
  that actually closes out the remaining ACs.

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
