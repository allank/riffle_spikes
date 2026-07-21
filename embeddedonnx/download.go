package embeddedonnx

import (
	"fmt"
	"io"
	"net/http"
)

// downloadBytes fetches url and returns its body, rejecting anything
// other than a 200 response or a body whose length doesn't match
// expectedSize — the same truncated/corrupt-download guard this repo's
// fetch-bge-model Makefile target uses, applied at runtime instead of
// build time.
func downloadBytes(url string, expectedSize int64) ([]byte, error) {
	resp, err := http.Get(url) //nolint:gosec // url comes from this package's own asset manifest, not user input
	if err != nil {
		return nil, fmt.Errorf("embeddedonnx: fetching %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embeddedonnx: fetching %s: unexpected status %s", url, resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("embeddedonnx: reading response body from %s: %w", url, err)
	}
	if int64(len(body)) != expectedSize {
		return nil, fmt.Errorf("embeddedonnx: %s: downloaded %d bytes, expected %d (truncated or corrupt download)", url, len(body), expectedSize)
	}
	return body, nil
}
