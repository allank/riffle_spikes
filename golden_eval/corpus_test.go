package goldeneval

import (
	"sort"
	"strings"
	"testing"
)

func TestLoadCorpusReadsCommittedFixtures(t *testing.T) {
	corpus, err := LoadCorpus("corpus")
	if err != nil {
		t.Fatalf("LoadCorpus() error = %v", err)
	}

	wantNoteIDs := []string{
		"billing-invoice-export.md",
		"oauth-scopes-and-consent.md",
		"oauth-token-refresh.md",
		"reindex-merkle-hashing.md",
		"vector-index-overview.md",
	}
	gotNoteIDs := make([]string, len(corpus.Notes))
	for i, n := range corpus.Notes {
		gotNoteIDs[i] = n.ID
	}
	sort.Strings(gotNoteIDs)
	if !equalStrings(gotNoteIDs, wantNoteIDs) {
		t.Errorf("note IDs = %v, want %v", gotNoteIDs, wantNoteIDs)
	}

	for _, n := range corpus.Notes {
		if n.ID == "oauth-token-refresh.md" && !strings.Contains(n.Text, "OAuth 2.0 Token Refresh") {
			t.Errorf("oauth-token-refresh.md text missing expected heading, got: %q", n.Text)
		}
	}

	wantQueries := []string{
		"OAuth token refresh",
		"how does riffle decide what to re-embed on reindex",
	}
	if !equalStrings(corpus.Queries, wantQueries) {
		t.Errorf("Queries = %v, want %v", corpus.Queries, wantQueries)
	}

	wantExpected := map[string][]string{
		"OAuth token refresh": {
			"oauth-token-refresh.md",
			"oauth-scopes-and-consent.md",
		},
		"how does riffle decide what to re-embed on reindex": {
			"reindex-merkle-hashing.md",
			"vector-index-overview.md",
		},
	}
	if len(corpus.Expected) != len(wantExpected) {
		t.Fatalf("Expected has %d queries, want %d", len(corpus.Expected), len(wantExpected))
	}
	for query, want := range wantExpected {
		got, ok := corpus.Expected[query]
		if !ok {
			t.Errorf("Expected missing query %q", query)
			continue
		}
		if !equalStrings(got, want) {
			t.Errorf("Expected[%q] = %v, want %v", query, got, want)
		}
	}
}

func TestLoadCorpusMissingDirectory(t *testing.T) {
	if _, err := LoadCorpus("corpus/does-not-exist"); err == nil {
		t.Error("LoadCorpus() with missing directory: expected error, got nil")
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
