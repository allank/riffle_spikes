package embeddedonnx

import (
	"strings"
	"testing"
)

func TestResolveAssetReturnsKnownPlatform(t *testing.T) {
	for _, platform := range []struct{ goos, goarch string }{
		{"darwin", "arm64"},
		{"darwin", "amd64"},
	} {
		a, err := resolveAsset(platform.goos, platform.goarch)
		if err != nil {
			t.Fatalf("resolveAsset(%s, %s): %v", platform.goos, platform.goarch, err)
		}
		if a.version == "" || a.tarballURL == "" || a.tarballSize <= 0 || a.memberPath == "" || a.dylibSize <= 0 || a.cacheFilename == "" {
			t.Fatalf("resolveAsset(%s, %s) returned an incomplete asset: %+v", platform.goos, platform.goarch, a)
		}
	}
}

func TestResolveAssetErrorsOnUnsupportedPlatform(t *testing.T) {
	_, err := resolveAsset("linux", "amd64")
	if err == nil {
		t.Fatal("resolveAsset(linux, amd64): expected an error, got nil")
	}
	if !strings.Contains(err.Error(), "linux/amd64") {
		t.Fatalf("resolveAsset(linux, amd64) error = %q, want it to name the unsupported platform", err.Error())
	}
}

func TestResolveAssetErrorsOnUnknownPlatform(t *testing.T) {
	_, err := resolveAsset("plan9", "386")
	if err == nil {
		t.Fatal("resolveAsset(plan9, 386): expected an error, got nil")
	}
}
