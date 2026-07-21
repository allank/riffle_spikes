// Package embeddedonnx downloads ONNX Runtime's shared library on first
// use and caches it locally, so callers can pass the cached path to
// github.com/yalue/onnxruntime_go's SetSharedLibraryPath without
// requiring a package-manager install (e.g. brew install onnxruntime)
// first.
//
// Only platforms listed in supportedAssets (manifest.go) are available;
// Path returns a clear error naming the platform otherwise. Notably,
// darwin/amd64 (Intel Mac) isn't supported yet — see manifest.go's doc
// comment and docs/specs/2026-07-21-embed-onnxruntime-design.md for why
// that's a real gap, not an oversight.
package embeddedonnx

import (
	"fmt"
	"os"
	"path/filepath"
)

// extractTo writes libBytes to dest unless a file of the same size is
// already there (the cache-hit case, which is left untouched rather than
// rewritten). Writing goes through a temp-file-then-rename so a failed or
// interrupted write can never be mistaken for a valid cache entry on a
// later run — the same pattern this repo's fetch-bge-model Makefile
// target uses for downloaded model assets.
func extractTo(libBytes []byte, dest string) (string, error) {
	if info, err := os.Stat(dest); err == nil && info.Size() == int64(len(libBytes)) {
		return dest, nil
	}

	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return "", fmt.Errorf("embeddedonnx: creating cache dir: %w", err)
	}

	tmp := dest + ".tmp"
	if err := os.WriteFile(tmp, libBytes, 0o644); err != nil {
		return "", fmt.Errorf("embeddedonnx: writing library to %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, dest); err != nil {
		return "", fmt.Errorf("embeddedonnx: finalizing library at %s: %w", dest, err)
	}
	return dest, nil
}
