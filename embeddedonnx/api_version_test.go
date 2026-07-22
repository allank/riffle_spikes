package embeddedonnx

import (
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"testing"
)

// TestDarwinAmd64ManifestMatchesVendoredLegacyAPIVersion guards against
// the darwin/amd64 manifest entry (which pins an ONNX Runtime release)
// drifting out of sync with internal/ortlegacy's vendored binding
// (which compiles in a fixed ORT_API_VERSION) — see manifest.go's doc
// comment. A mismatch here is exactly the failure mode riffle_spikes#20
// hit at runtime ("requested API version [25] is not available, only
// ... [1, 23]"); this test catches it at test time instead, by reading
// ortlegacy's actual header rather than assuming it's unchanged.
func TestDarwinAmd64ManifestMatchesVendoredLegacyAPIVersion(t *testing.T) {
	a, err := resolveAsset("darwin", "amd64")
	if err != nil {
		t.Fatalf("resolveAsset(darwin, amd64): %v", err)
	}

	// ONNX Runtime v1.23.0's own C API header (checked directly against
	// Microsoft's source at that tag, not assumed) declares
	// ORT_API_VERSION 23 — the max API version that release's runtime
	// supports. If the manifest is ever repinned to a different
	// release, this assumption needs re-verifying against that
	// release's own header, not just bumped blindly.
	const pinnedRuntimeVersion = "1.23.0"
	const pinnedRuntimeMaxAPIVersion = 23
	if a.version != pinnedRuntimeVersion {
		t.Fatalf("darwin/amd64 manifest version is %s, but this test's known max-API-version fact (%d) is only verified for %s — verify the new version's own onnxruntime_c_api.h and update this test before trusting it",
			a.version, pinnedRuntimeMaxAPIVersion, pinnedRuntimeVersion)
	}

	compiledAPIVersion := ortlegacyCompiledAPIVersion(t)
	if compiledAPIVersion > pinnedRuntimeMaxAPIVersion {
		t.Fatalf("internal/ortlegacy compiles in ORT_API_VERSION %d, but embeddedonnx's darwin/amd64 manifest pins ONNX Runtime %s (max API version %d) — these must stay compatible or darwin/amd64 will fail at runtime exactly like riffle_spikes#20 did",
			compiledAPIVersion, a.version, pinnedRuntimeMaxAPIVersion)
	}
}

// ortlegacyCompiledAPIVersion reads internal/ortlegacy's bundled
// onnxruntime_c_api.h directly, rather than importing the ortlegacy
// package itself — that package is darwin/amd64-only (build-tag
// gated), but this check should run on every platform's test suite,
// since the manifest entry it's guarding is data, not
// platform-specific code.
func ortlegacyCompiledAPIVersion(t *testing.T) int {
	t.Helper()

	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("could not determine this test file's own path")
	}
	headerPath := filepath.Join(filepath.Dir(thisFile), "..", "internal", "ortlegacy", "onnxruntime_c_api.h")

	header, err := os.ReadFile(headerPath)
	if err != nil {
		t.Fatalf("reading %s: %v", headerPath, err)
	}

	m := regexp.MustCompile(`#define ORT_API_VERSION (\d+)`).FindSubmatch(header)
	if m == nil {
		t.Fatalf("could not find \"#define ORT_API_VERSION\" in %s", headerPath)
	}

	version, err := strconv.Atoi(string(m[1]))
	if err != nil {
		t.Fatalf("parsing ORT_API_VERSION from %s: %v", headerPath, err)
	}
	return version
}
