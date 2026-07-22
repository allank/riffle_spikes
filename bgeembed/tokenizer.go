package bgeembed

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode"
)

// Tokenizer is a WordPiece tokenizer for BERT-family models (BGE-small
// included), loaded from a HuggingFace tokenizer.json file. Ported from
// onnx_test's bge_bench/tokenizer package — already proven correct by
// every nDCG/cosine-similarity number in this repo's decision-criteria
// history — and owned here going forward rather than imported from that
// ephemeral sibling repo.
type Tokenizer struct {
	vocab      map[string]int64
	unkID      int64
	clsID      int64
	sepID      int64
	contPrefix string // "##"
}

// Output is the tensor-ready result of tokenizing one piece of text.
type Output struct {
	InputIDs      []int64
	AttentionMask []int64
	TokenTypeIDs  []int64
}

type tokenizerJSON struct {
	Model struct {
		Vocab                   map[string]int64 `json:"vocab"`
		UnkToken                string           `json:"unk_token"`
		ContinuingSubwordPrefix string           `json:"continuing_subword_prefix"`
	} `json:"model"`
}

// LoadFile reads and parses a tokenizer.json file from disk.
func LoadFile(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("bgeembed: reading %s: %w", path, err)
	}
	return Load(data)
}

// Load parses tokenizer.json bytes already in memory.
func Load(data []byte) (*Tokenizer, error) {
	var tf tokenizerJSON
	if err := json.Unmarshal(data, &tf); err != nil {
		return nil, fmt.Errorf("bgeembed: parsing tokenizer.json: %w", err)
	}
	t := &Tokenizer{
		vocab:      tf.Model.Vocab,
		contPrefix: tf.Model.ContinuingSubwordPrefix,
	}
	t.unkID = t.vocab[tf.Model.UnkToken]
	t.clsID = t.vocab["[CLS]"]
	t.sepID = t.vocab["[SEP]"]
	if t.contPrefix == "" {
		t.contPrefix = "##"
	}
	return t, nil
}

// Encode tokenizes text using WordPiece, adds [CLS]+[SEP], and
// truncates to maxLen.
func (t *Tokenizer) Encode(text string, maxLen int) Output {
	words := bertPreTokenize(strings.ToLower(text))
	var ids []int64
	for _, w := range words {
		ids = append(ids, t.wordpiece(w)...)
	}
	// Reserve 2 slots for [CLS] and [SEP].
	if len(ids) > maxLen-2 {
		ids = ids[:maxLen-2]
	}
	result := make([]int64, 0, len(ids)+2)
	result = append(result, t.clsID)
	result = append(result, ids...)
	result = append(result, t.sepID)

	n := len(result)
	mask := make([]int64, n)
	types := make([]int64, n)
	for i := range mask {
		mask[i] = 1
	}
	return Output{InputIDs: result, AttentionMask: mask, TokenTypeIDs: types}
}

// wordpiece applies the WordPiece algorithm to a single pre-tokenized
// word: greedy longest-match-first against the vocab, prefixing
// continuation pieces (after the first) with contPrefix.
func (t *Tokenizer) wordpiece(word string) []int64 {
	if len(word) > 100 {
		return []int64{t.unkID}
	}
	runes := []rune(word)
	var result []int64
	start := 0
	for start < len(runes) {
		end := len(runes)
		found := false
		for end > start {
			substr := string(runes[start:end])
			if start > 0 {
				substr = t.contPrefix + substr
			}
			if id, ok := t.vocab[substr]; ok {
				result = append(result, id)
				start = end
				found = true
				break
			}
			end--
		}
		if !found {
			return []int64{t.unkID}
		}
	}
	return result
}

// bertPreTokenize splits on whitespace and punctuation, matching
// BertPreTokenizer's behavior.
func bertPreTokenize(text string) []string {
	var words []string
	var cur strings.Builder
	for _, r := range text {
		if unicode.IsSpace(r) {
			if cur.Len() > 0 {
				words = append(words, cur.String())
				cur.Reset()
			}
		} else if isPunct(r) {
			if cur.Len() > 0 {
				words = append(words, cur.String())
				cur.Reset()
			}
			words = append(words, string(r))
		} else {
			cur.WriteRune(r)
		}
	}
	if cur.Len() > 0 {
		words = append(words, cur.String())
	}
	return words
}

func isPunct(r rune) bool {
	if r >= 33 && r <= 47 {
		return true
	}
	if r >= 58 && r <= 64 {
		return true
	}
	if r >= 91 && r <= 96 {
		return true
	}
	if r >= 123 && r <= 126 {
		return true
	}
	return unicode.Is(unicode.P, r) || unicode.Is(unicode.S, r)
}
