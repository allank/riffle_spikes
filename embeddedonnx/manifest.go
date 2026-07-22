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
// fetch ONNX Runtime for; sizes and paths confirmed against the real
// release assets, not assumed.
//
// darwin/amd64 is pinned to v1.23.0 — Microsoft's last official
// osx-x86_64 release — rather than v1.27.1 like darwin/arm64, because
// bgeembed links darwin/amd64 against a vendored ONNX Runtime binding
// (internal/ortlegacy) frozen at API version 23, the max v1.23.0's own
// C API supports. The live binding used everywhere else requests API
// 25, which no osx-x86_64 release satisfies — see
// docs/specs/2026-07-21-self-contained-onnx-embedder-design.md for the
// full rationale. This pairing (manifest version <-> ortlegacy's
// compiled API version) isn't checked by the compiler; see
// api_version_test.go for the guard against them drifting apart.
var supportedAssets = map[string]asset{
	"darwin/arm64": {
		version:       "1.27.1",
		tarballURL:    "https://github.com/microsoft/onnxruntime/releases/download/v1.27.1/onnxruntime-osx-arm64-1.27.1.tgz",
		tarballSize:   31_959_937,
		memberPath:    "onnxruntime-osx-arm64-1.27.1/lib/libonnxruntime.1.27.1.dylib",
		dylibSize:     38_502_216,
		cacheFilename: "libonnxruntime.1.27.1.dylib",
	},
	"darwin/amd64": {
		version:       "1.23.0",
		tarballURL:    "https://github.com/microsoft/onnxruntime/releases/download/v1.23.0/onnxruntime-osx-x86_64-1.23.0.tgz",
		tarballSize:   11_621_905,
		memberPath:    "onnxruntime-osx-x86_64-1.23.0/lib/libonnxruntime.1.23.0.dylib",
		dylibSize:     39_582_416,
		cacheFilename: "libonnxruntime.1.23.0.dylib",
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
