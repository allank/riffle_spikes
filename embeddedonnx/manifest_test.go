package embeddedonnx

import (
	"strings"
	"testing"
)

func TestResolveAssetReturnsKnownPlatform(t *testing.T) {
	a, err := resolveAsset("darwin", "arm64")
	if err != nil {
		t.Fatalf("resolveAsset(darwin, arm64): %v", err)
	}
	if a.version == "" || a.tarballURL == "" || a.tarballSize <= 0 || a.memberPath == "" || a.dylibSize <= 0 || a.cacheFilename == "" {
		t.Fatalf("resolveAsset(darwin, arm64) returned an incomplete asset: %+v", a)
	}
}

func TestResolveAssetErrorsOnUnsupportedPlatform(t *testing.T) {
	_, err := resolveAsset("darwin", "amd64")
	if err == nil {
		t.Fatal("resolveAsset(darwin, amd64): expected an error, got nil")
	}
	if !strings.Contains(err.Error(), "darwin/amd64") {
		t.Fatalf("resolveAsset(darwin, amd64) error = %q, want it to name the unsupported platform", err.Error())
	}
}

func TestResolveAssetErrorsOnUnknownPlatform(t *testing.T) {
	_, err := resolveAsset("plan9", "386")
	if err == nil {
		t.Fatal("resolveAsset(plan9, 386): expected an error, got nil")
	}
}
