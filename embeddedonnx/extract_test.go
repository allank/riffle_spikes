package embeddedonnx

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestExtractToWritesFileWhenAbsent(t *testing.T) {
	dest := filepath.Join(t.TempDir(), "nested", "libonnxruntime.dylib")
	want := []byte("fake onnx runtime bytes")

	got, err := extractTo(want, dest)
	if err != nil {
		t.Fatalf("extractTo: %v", err)
	}
	if got != dest {
		t.Fatalf("extractTo returned %q, want %q", got, dest)
	}

	content, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("reading extracted file: %v", err)
	}
	if string(content) != string(want) {
		t.Fatalf("extracted content = %q, want %q", content, want)
	}
}

func TestExtractToSkipsWriteWhenCorrectlySizedFileAlreadyExists(t *testing.T) {
	dest := filepath.Join(t.TempDir(), "libonnxruntime.dylib")
	original := bytes.Repeat([]byte("a"), 64)
	if err := os.WriteFile(dest, original, 0o644); err != nil {
		t.Fatalf("seeding dest: %v", err)
	}

	replacement := bytes.Repeat([]byte("b"), 64)

	if _, err := extractTo(replacement, dest); err != nil {
		t.Fatalf("extractTo: %v", err)
	}

	content, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("reading dest: %v", err)
	}
	if string(content) != string(original) {
		t.Fatalf("extractTo overwrote a correctly-sized cache hit: got %q, want unchanged %q", content, original)
	}
}

func TestExtractToOverwritesWhenExistingFileIsWrongSize(t *testing.T) {
	dest := filepath.Join(t.TempDir(), "libonnxruntime.dylib")
	if err := os.WriteFile(dest, []byte("truncated"), 0o644); err != nil {
		t.Fatalf("seeding dest: %v", err)
	}

	want := []byte("the real, correctly-sized library bytes")
	if _, err := extractTo(want, dest); err != nil {
		t.Fatalf("extractTo: %v", err)
	}

	content, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("reading dest: %v", err)
	}
	if string(content) != string(want) {
		t.Fatalf("extractTo did not overwrite a wrong-sized file: got %q, want %q", content, want)
	}
}

func TestExtractToLeavesNoTempFileOnSuccess(t *testing.T) {
	dir := t.TempDir()
	dest := filepath.Join(dir, "libonnxruntime.dylib")

	if _, err := extractTo([]byte("bytes"), dest); err != nil {
		t.Fatalf("extractTo: %v", err)
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("reading dir: %v", err)
	}
	if len(entries) != 1 || entries[0].Name() != "libonnxruntime.dylib" {
		t.Fatalf("dir contains unexpected entries: %v", entries)
	}
}
