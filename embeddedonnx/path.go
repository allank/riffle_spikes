package embeddedonnx

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

// Path returns the local path to this platform's ONNX Runtime shared
// library, downloading and caching it first if it isn't already cached.
// A cache hit costs zero network calls: the check happens before any
// download is attempted, not just before any disk write.
func Path() (string, error) {
	a, err := resolveAsset(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		return "", err
	}

	dest, err := cachePath(a)
	if err != nil {
		return "", err
	}

	if info, err := os.Stat(dest); err == nil && info.Size() == a.dylibSize {
		return dest, nil
	}

	archive, err := downloadBytes(a.tarballURL, a.tarballSize)
	if err != nil {
		return "", fmt.Errorf("embeddedonnx: downloading ONNX Runtime %s: %w", a.version, err)
	}

	dylib, err := extractTarMember(archive, a.memberPath)
	if err != nil {
		return "", fmt.Errorf("embeddedonnx: extracting %s: %w", a.memberPath, err)
	}
	if int64(len(dylib)) != a.dylibSize {
		return "", fmt.Errorf("embeddedonnx: extracted %s is %d bytes, expected %d (truncated or corrupt archive)", a.memberPath, len(dylib), a.dylibSize)
	}

	return extractTo(dylib, dest)
}

func cachePath(a asset) (string, error) {
	base, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("embeddedonnx: resolving user cache dir: %w", err)
	}
	return filepath.Join(base, "riffle",
		fmt.Sprintf("onnxruntime-%s-%s-%s", a.version, runtime.GOOS, runtime.GOARCH),
		a.cacheFilename), nil
}
