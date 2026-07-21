package embeddedonnx

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestDownloadBytesReturnsBodyWhenSizeMatches(t *testing.T) {
	want := []byte("fake tarball bytes")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(want)
	}))
	defer srv.Close()

	got, err := downloadBytes(srv.URL, int64(len(want)))
	if err != nil {
		t.Fatalf("downloadBytes: %v", err)
	}
	if string(got) != string(want) {
		t.Fatalf("downloadBytes = %q, want %q", got, want)
	}
}

func TestDownloadBytesRejectsSizeMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("short"))
	}))
	defer srv.Close()

	_, err := downloadBytes(srv.URL, 99999)
	if err == nil {
		t.Fatal("downloadBytes: expected an error for a truncated download, got nil")
	}
}

func TestDownloadBytesRejectsNonOKStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer srv.Close()

	_, err := downloadBytes(srv.URL, 9)
	if err == nil {
		t.Fatal("downloadBytes: expected an error for a 404 response, got nil")
	}
}
