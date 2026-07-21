package embeddedonnx

import "fmt"

// asset describes one platform's ONNX Runtime release: where to
// download its tarball from, the tarball's expected size (a
// truncated/corrupt download's simplest tell), the shared library's
// path inside that tarball, that library's own expected size (used to
// both validate the extraction and to detect an already-cached copy
// without touching the network), and the filename it's cached under.
type asset struct {
	version       string
	tarballURL    string
	tarballSize   int64
	memberPath    string
	dylibSize     int64
	cacheFilename string
}

// supportedAssets lists every (GOOS, GOARCH) this package knows how to
// fetch ONNX Runtime for. Only darwin/arm64 is populated today; sizes
// and paths confirmed against the real v1.27.1 release asset, not
// assumed.
//
// darwin/amd64 (Intel Mac) is deliberately absent, not merely unbuilt:
// this repo's binding (github.com/yalue/onnxruntime_go v1.29.0) requests
// ONNX Runtime API version 25 at compile time, and Microsoft's last
// official osx-x86_64 release (v1.23.0) tops out at API version 23 — the
// two gaps (no recent osx-x64 build, and no old-enough-API osx-x64
// build) land on the exact same release boundary (v1.25.0), so no
// version satisfies both. Supporting Intel Mac needs a second,
// build-tag-selected backend pinned to an older API version — tracked
// as a separate ticket, not implemented here. See
// docs/specs/2026-07-21-embed-onnxruntime-design.md.
var supportedAssets = map[string]asset{
	"darwin/arm64": {
		version:       "1.27.1",
		tarballURL:    "https://github.com/microsoft/onnxruntime/releases/download/v1.27.1/onnxruntime-osx-arm64-1.27.1.tgz",
		tarballSize:   31_959_937,
		memberPath:    "onnxruntime-osx-arm64-1.27.1/lib/libonnxruntime.1.27.1.dylib",
		dylibSize:     38_502_216,
		cacheFilename: "libonnxruntime.1.27.1.dylib",
	},
}

// resolveAsset returns the asset for goos/goarch, or a clear error naming
// the unsupported platform if none is registered.
func resolveAsset(goos, goarch string) (asset, error) {
	a, ok := supportedAssets[goos+"/"+goarch]
	if !ok {
		return asset{}, fmt.Errorf("embeddedonnx: no embedded ONNX Runtime available for %s/%s", goos, goarch)
	}
	return a, nil
}
