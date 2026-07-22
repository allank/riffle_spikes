package bgeembed

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// testVocab builds a minimal tokenizer.json-shaped fixture, small enough
// to reason about by hand, rather than depending on the real ~700KB BGE
// vocab. Includes a continuing-subword-prefix case ("##ing") to exercise
// WordPiece's actual subword-splitting behavior, not just whole-word
// lookups.
func testVocabJSON(t *testing.T) []byte {
	t.Helper()
	doc := map[string]any{
		"model": map[string]any{
			"vocab": map[string]int64{
				"[UNK]": 0,
				"[CLS]": 1,
				"[SEP]": 2,
				"hello": 3,
				"world": 4,
				"##ing": 5,
				"test":  6,
				"!":     7,
			},
			"unk_token":                 "[UNK]",
			"continuing_subword_prefix": "##",
		},
	}
	data, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("marshaling test vocab: %v", err)
	}
	return data
}

func TestEncodeAddsClsAndSep(t *testing.T) {
	tok, err := Load(testVocabJSON(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	out := tok.Encode("hello world!", 10)

	want := []int64{1, 3, 4, 7, 2} // [CLS] hello world ! [SEP]
	if !int64SliceEqual(out.InputIDs, want) {
		t.Fatalf("InputIDs = %v, want %v", out.InputIDs, want)
	}
	for i, m := range out.AttentionMask {
		if m != 1 {
			t.Fatalf("AttentionMask[%d] = %d, want 1 (no padding in this Encode call)", i, m)
		}
	}
	for i, tt := range out.TokenTypeIDs {
		if tt != 0 {
			t.Fatalf("TokenTypeIDs[%d] = %d, want 0 (single-segment input)", i, tt)
		}
	}
}

func TestEncodeTruncatesToMaxLen(t *testing.T) {
	tok, err := Load(testVocabJSON(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	out := tok.Encode("hello world!", 4)

	want := []int64{1, 3, 4, 2} // [CLS] hello world [SEP] -- "!" dropped to fit maxLen-2
	if !int64SliceEqual(out.InputIDs, want) {
		t.Fatalf("InputIDs = %v, want %v (truncated to maxLen=4)", out.InputIDs, want)
	}
}

func TestEncodeSplitsContinuingSubwords(t *testing.T) {
	tok, err := Load(testVocabJSON(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	out := tok.Encode("testing", 10)

	want := []int64{1, 6, 5, 2} // [CLS] test ##ing [SEP]
	if !int64SliceEqual(out.InputIDs, want) {
		t.Fatalf("InputIDs = %v, want %v (WordPiece split into test + ##ing)", out.InputIDs, want)
	}
}

func TestEncodeUsesUnkForUnknownWords(t *testing.T) {
	tok, err := Load(testVocabJSON(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	out := tok.Encode("xyz", 10)

	want := []int64{1, 0, 2} // [CLS] [UNK] [SEP]
	if !int64SliceEqual(out.InputIDs, want) {
		t.Fatalf("InputIDs = %v, want %v (unrecognized word maps to [UNK])", out.InputIDs, want)
	}
}

func TestLoadFileReadsFromDisk(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := os.WriteFile(path, testVocabJSON(t), 0o644); err != nil {
		t.Fatalf("writing fixture: %v", err)
	}

	tok, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile: %v", err)
	}
	if tok == nil {
		t.Fatal("LoadFile returned a nil tokenizer")
	}
}

func int64SliceEqual(a, b []int64) bool {
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
