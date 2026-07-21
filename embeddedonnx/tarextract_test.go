package embeddedonnx

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"testing"
)

// buildTarGz constructs an in-memory .tar.gz containing the given
// path -> content entries, for tests to exercise extractTarMember
// without a real network download.
func buildTarGz(t *testing.T, files map[string][]byte) []byte {
	t.Helper()

	var buf bytes.Buffer
	gw := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gw)

	for path, content := range files {
		hdr := &tar.Header{
			Name: path,
			Mode: 0o644,
			Size: int64(len(content)),
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatalf("writing tar header for %s: %v", path, err)
		}
		if _, err := tw.Write(content); err != nil {
			t.Fatalf("writing tar content for %s: %v", path, err)
		}
	}

	if err := tw.Close(); err != nil {
		t.Fatalf("closing tar writer: %v", err)
	}
	if err := gw.Close(); err != nil {
		t.Fatalf("closing gzip writer: %v", err)
	}
	return buf.Bytes()
}

func TestExtractTarMemberReturnsMatchingMemberContent(t *testing.T) {
	want := []byte("the shared library bytes")
	archive := buildTarGz(t, map[string][]byte{
		"onnxruntime-osx-arm64-1.27.1/README.md":                       []byte("not this one"),
		"onnxruntime-osx-arm64-1.27.1/lib/libonnxruntime.1.27.1.dylib": want,
	})

	got, err := extractTarMember(archive, "onnxruntime-osx-arm64-1.27.1/lib/libonnxruntime.1.27.1.dylib")
	if err != nil {
		t.Fatalf("extractTarMember: %v", err)
	}
	if string(got) != string(want) {
		t.Fatalf("extractTarMember = %q, want %q", got, want)
	}
}

func TestExtractTarMemberErrorsWhenMemberAbsent(t *testing.T) {
	archive := buildTarGz(t, map[string][]byte{
		"onnxruntime-osx-arm64-1.27.1/README.md": []byte("only this"),
	})

	_, err := extractTarMember(archive, "onnxruntime-osx-arm64-1.27.1/lib/libonnxruntime.1.27.1.dylib")
	if err == nil {
		t.Fatal("extractTarMember: expected an error for a missing member, got nil")
	}
}

func TestExtractTarMemberErrorsOnInvalidArchive(t *testing.T) {
	_, err := extractTarMember([]byte("not a tarball at all"), "anything")
	if err == nil {
		t.Fatal("extractTarMember: expected an error for a corrupt archive, got nil")
	}
}
