package embeddedonnx

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"strings"
)

// extractTarMember reads a single member's content out of an in-memory
// .tar.gz archive, without ever writing the archive itself to disk —
// only the one file this package actually needs (the shared library)
// gets persisted, by the caller. memberPath is matched against entry
// names in the clean, no-prefix form; a leading "./" on the archive's
// own entry names is stripped before comparing, since not every ONNX
// Runtime release tarball is packaged the same way — Microsoft's real
// v1.23.0 osx-x86_64 release names its entries "./onnxruntime-.../...",
// unlike v1.27.1's osx-arm64 release, which doesn't.
func extractTarMember(archive []byte, memberPath string) ([]byte, error) {
	gz, err := gzip.NewReader(bytes.NewReader(archive))
	if err != nil {
		return nil, fmt.Errorf("embeddedonnx: opening gzip stream: %w", err)
	}
	defer gz.Close()

	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil, fmt.Errorf("embeddedonnx: %s not found in archive", memberPath)
		}
		if err != nil {
			return nil, fmt.Errorf("embeddedonnx: reading tar entry: %w", err)
		}
		if strings.TrimPrefix(hdr.Name, "./") != memberPath {
			continue
		}
		content, err := io.ReadAll(tr)
		if err != nil {
			return nil, fmt.Errorf("embeddedonnx: reading %s from archive: %w", memberPath, err)
		}
		return content, nil
	}
}
