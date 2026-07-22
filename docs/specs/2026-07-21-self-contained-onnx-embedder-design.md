# Design: a self-contained ONNX embedder (fills the Intel Mac gap, drops the `onnx_test` dependency)

## Problem

Two independent problems turned out to share a fix location, but **not
a cause**: removing the `onnx_test` dependency, on its own, does
nothing for Intel. If `bgeembed`'s glue code just imported the live
`github.com/yalue/onnxruntime_go` directly — fully self-contained, zero
`onnx_test` — it would still fail on Intel Mac with the exact API
version [25] is not available, only [1, 23] error #20 already hit. The
API-version mismatch is about which *compiled binding version* gets
linked in; it has nothing to do with which repo the surrounding
tokenizer/session code lives in. They're addressed together here only
because the same glue file needs rewriting either way, not because one
resolves the other.

**1. Intel Mac has no embedded-ONNX path.** `riffle_spikes`#20 shipped
`embeddedonnx` for darwin/arm64 only. The reason (see
`docs/specs/2026-07-21-embed-onnxruntime-design.md`'s "Implementation
outcome" section): the live `yalue/onnxruntime_go` binding (v1.29.0,
pinned in `go.mod`) requests ONNX Runtime API version 25, and no
official osx-x86_64 release supports that — the last one (v1.23.0) tops
out at API 23. Fixing this needs a second binding pinned to an older API
version, used only for darwin/amd64.

**2. The existing ONNX path depends on `onnx_test`, which shouldn't be
depended on.** `golden_eval/adapters/onnxadapter` wraps `onnx_test`'s
`onnxpath` package (via a `replace` directive in `go.mod`), which in
turn imports `onnx_test`'s own `tokenizer` package. Both `onnx_test` and
`riffle_spikes` are explicitly ephemeral — investigation repos, not
meant to be maintained going forward (`CLAUDE.md`: "`docs/adr/`
intentionally stays empty... consequential findings graduate into ADRs
in `riffle` instead"). Code meant to eventually migrate into `riffle`
can't carry a dependency on a repo that isn't coming with it. This
applies to every platform's ONNX path today, not just the Intel gap —
darwin/arm64 already ships (#20) with this same dependency, since
`embeddedonnx.Path()` only supplies a library path; the adapter that
consumes it is still `onnxadapter` → `onnx_test`.

## What's being reused vs. rebuilt

Read both of `onnx_test`'s relevant files directly rather than assuming
their size or complexity:

- **`onnx_test/inference/bge_bench/tokenizer/tokenizer.go`** — 154
  lines, stdlib-only (`encoding/json`, `os`, `strings`, `unicode`). A
  complete WordPiece tokenizer: loads `tokenizer.json`'s vocab, does
  BERT-style pre-tokenization (whitespace/punctuation splitting) plus
  greedy longest-match WordPiece, adds `[CLS]`/`[SEP]`, truncates.
  Already proven correct — every nDCG 1.0000 / cosine 1.000000 number in
  `docs/decision-criteria.md` was produced against its output. No
  license header (Allan's own code, not third-party) — moving it is
  just relocating code within repos Allan owns.
- **`onnx_test/inference/bge_bench/onnxpath/model.go`** — 111 lines:
  session creation, three input tensors (`input_ids`, `attention_mask`,
  `token_type_ids`), `session.Run`, CLS-token pooling + L2 normalize
  (BGE's documented pooling method, not the BERT pooler head).
- **`yalue/onnxruntime_go`'s public API is stable across the versions
  that matter.** Diffed `onnxruntime_go.go` between v1.25.0 (requests
  API 23 — matches ONNX Runtime v1.23.0, the last osx-x86_64 release)
  and v1.29.0 (requests API 25, currently pinned) directly from GitHub:
  every function this repo's glue code calls —
  `SetSharedLibraryPath`, `InitializeEnvironment`, `NewShape`,
  `NewTensor`, `NewDynamicAdvancedSession`, plus `Run`/`Destroy`/
  `GetData` on the resulting values — is identical between the two
  versions. v1.29.0 only *adds* symbols (`NewArenaCfg`,
  `NewArenaCfgV2`); nothing used here changed shape. That means the
  session/tensor/pooling glue can be **written once** and compiled
  against either binding, rather than duplicated per platform.
- **`onnxruntime_go` is MIT-licensed** (checked directly) — vendoring an
  old release is permitted, with the license file carried along.
  `riffle_spikes` itself now has an explicit root `LICENSE` (MIT, added
  alongside this spec) — same license as what's being vendored, so
  there's no compatibility question to resolve, just the normal MIT
  obligation to carry `ortlegacy`'s own license notice with it.

## Architecture

```
bgeembed/                        new, self-contained, no onnx_test import
  tokenizer.go                   WordPiece tokenizer, ported from onnx_test's
  tokenizer_test.go              (own it going forward; not a copy-and-forget)
  embedder.go                    session/tensor/pooling glue, written once
                                  against a package-local `ort` alias
  ort_modern.go                  //go:build !(darwin && amd64)
                                  import ort "github.com/yalue/onnxruntime_go"
                                  (the live, already-pinned v1.29.0, API 25)
  ort_legacy_darwin_amd64.go     //go:build darwin && amd64
                                  import ort "github.com/allank/riffle_spikes/internal/ortlegacy"

internal/ortlegacy/              vendored copy of yalue/onnxruntime_go v1.25.0
  LICENSE                        (API 23 — matches ONNX Runtime v1.23.0's max),
  onnxruntime_go.go               MIT license file carried along, frozen —
  onnxruntime_wrapper.c/.h         no upstream updates ever land here, by
  onnxruntime_c_api.h              design. Used only when GOOS=darwin,
  ...                              GOARCH=amd64.
```

- `embeddedonnx`'s manifest (`manifest.go`) gains a `darwin/amd64` entry:
  downloads ONNX Runtime v1.23.0's `onnxruntime-osx-x86_64-1.23.0.tgz` —
  the exact release that failed against the *live* binding during #20
  (`API version [25] is not available, only... [1, 23]`). It'll succeed
  this time, because darwin/amd64 builds link against the vendored
  API-23 `ortlegacy` package instead of the live one.
- `golden_eval/adapters/onnxadapter` (and anything else currently
  wrapping `onnx_test`'s `onnxpath`) gets rewired onto `bgeembed`
  instead — this removes the `onnx_test` dependency from the ONNX path
  on **every** platform, not just the new Intel one, which is the actual
  point: the goal isn't "Intel Mac also works," it's "this code doesn't
  depend on an ephemeral repo," and Intel Mac support falls out of that
  as a side effect once both backends exist behind one glue
  implementation.

### Non-Intel platforms are unaffected, by construction

The build tags are what make this safe: `ort_modern.go`'s
`!(darwin && amd64)` constraint means every platform *except* Intel Mac
— Apple Silicon, Linux, Windows — keeps resolving to whatever
`onnxruntime_go` version `go.mod` pins, exactly as today. Each compiled
binary only includes one of the two `ort_*.go` files, chosen per target
platform at build time (cross-compiling for different `GOOS`/`GOARCH`
combinations is already separate `go build` invocations). `go.mod`
itself still lists exactly one real dependency on `onnxruntime_go` — the
vendored copy has no `go.mod` entry at all, since it's copied source,
not a module dependency, so it never factors into version resolution
for any other platform. If `onnxruntime_go` bumps to v1.30 or beyond
later, arm64/Linux/Windows pick it up the moment `go.mod` is bumped;
only the frozen `ortlegacy` copy stays where it is until someone
deliberately re-cuts it.

### The runtime download already resolves per-platform correctly

`embeddedonnx.Path()` calls `resolveAsset(runtime.GOOS, runtime.GOARCH)`
— and `runtime.GOOS`/`GOARCH` are fixed per compiled binary (an arm64
build always reports `"arm64"`, an amd64 build always reports
`"amd64"`, regardless of what machine it's later run on). So each build
only ever looks up and downloads its own platform's manifest entry;
there's no cross-platform ambiguity to introduce by adding the
`darwin/amd64` row. This is existing, already-shipped behavior from
#20 — the new manifest entry is the only change needed on this side.

**One thing this does *not* enforce automatically**: the downloaded
runtime version (`manifest.go`'s `darwin/amd64` entry) and the
compiled-in binding version (`ortlegacy`'s vendored release) have to
agree, and nothing currently checks that pairing at compile time or
runtime — they're two independent pieces of source that need to stay in
lockstep by convention. If `ortlegacy` is ever re-cut from a newer
release without updating the manifest's version/URL/size (or vice
versa), nothing catches the mismatch until a real run fails with an
API-version or checksum-shaped error. See the acceptance criteria
below for how this ticket should guard against that, rather than
leaving it as tribal knowledge.

## Verification

- **darwin/amd64 (this machine) is fully testable today** — unlike #20's
  arm64 gap, Intel hardware, ONNX Runtime v1.23.0, and the vendored
  API-23 binding are all available right now. Real verification (golden
  eval nDCG/MRR + cosine similarity vs. the existing brew-installed
  reference) should happen as part of this ticket, not deferred.
- **darwin/arm64 still needs real Apple Silicon** for `dlopen` and a
  golden-eval run — same blocker #20 already documented, unaffected by
  this change (the modern backend's binding doesn't change, only where
  its glue code lives).
- Once both platforms are verified, `docs/decision-criteria.md`'s ONNX
  row(s) should reflect the self-contained path, not the `onnx_test`
  one.

## Acceptance criteria

- `bgeembed` exists, imports neither `onnx_test` nor anything under its
  module path, and `onnxadapter` (or its replacement) is rewired onto
  it — verified by grepping the diff for `onnx_test` import paths
  outside `puregoadapter`.
- `embeddedonnx`'s manifest gains a `darwin/amd64` entry; `Path()` on
  this Intel machine downloads and caches ONNX Runtime v1.23.0
  successfully.
- A real golden-eval run on this machine, using `bgeembed`'s
  darwin/amd64 path end to end (`ortlegacy` + the downloaded v1.23.0
  library), produces nDCG 1.0000 / MRR 1.0000 and cosine similarity
  matching the existing brew-installed reference.
- darwin/arm64 continues to build and pass its existing tests
  unchanged (proving the build-tag isolation actually holds, not just
  in theory) — real `dlopen`/golden-eval verification on that platform
  remains hardware-blocked, same as #20.
- A test (or equivalent explicit check) asserts `ortlegacy`'s compiled
  `ORT_API_VERSION` is compatible with the ONNX Runtime version named in
  `embeddedonnx`'s `darwin/amd64` manifest entry, so a future edit to
  either side that breaks the pairing fails loudly (build or test
  failure) instead of silently at runtime on someone's machine.

## Decisions to make explicit

- **Vendoring `onnxruntime_go` v1.25.0 is a permanent maintenance
  commitment, not a one-time copy.** `internal/ortlegacy` will never
  receive upstream security fixes or bug fixes — if `onnxruntime_go`
  ships a correctness fix affecting the C API bridge itself (not just
  new features), it has to be manually back-ported or the vendored copy
  re-cut from a newer-but-still-API-23-compatible release. Worth
  revisiting if Microsoft ever resumes shipping osx-x64 builds at a
  current API version, at which point `ortlegacy` can be deleted
  entirely in favor of the live binding for all platforms.
- **`puregoadapter` and `sidecaradapter`'s `onnx_test` dependencies are
  out of scope here.** `sidecaradapter` already has zero `onnx_test`
  dependency (confirmed via grep — the Rust sidecars tokenize
  internally via the HF `tokenizers` crate). `puregoadapter` still wraps
  `onnx_test`'s `puregopath` (a full pure-Go BERT forward pass, not a
  111-line glue file) — bringing that in-house is a much larger
  undertaking than this spec, and not what was asked for. Flagging so
  it isn't mistaken for an oversight, not proposing to do it here.

## Out of scope

- Linux/Windows embedded-ONNX support — tracked separately as
  riffle_spikes#21/#22, unaffected by this change (both already use the
  live binding, no API-version conflict exists there).
- Migrating `puregoadapter` off `onnx_test` (see above).
- `CANDLE_DEVICE=auto` (Metal-with-CPU-fallback) — already discussed and
  deferred earlier in this project; unrelated to ONNX.
