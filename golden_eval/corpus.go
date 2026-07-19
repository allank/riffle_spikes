package goldeneval

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// Note is a single hand-authored corpus document.
type Note struct {
	// ID is the note's filename (e.g. "oauth-token-refresh.md"), used
	// throughout the golden eval as the note's identity.
	ID   string
	Text string
}

// Corpus is the golden eval's fixed, hand-authored fixture set: notes,
// the queries run against them, and the expected relevant-note ranking
// per query (PRD Section 6). Relevance is true by construction — each
// note was authored to be a top match, a distractor, or unrelated for a
// given query — so Expected only lists a query's relevant notes, ranked
// best first; any note absent from that list is treated as irrelevant.
type Corpus struct {
	Notes    []Note
	Queries  []string
	Expected map[string][]string
}

type queriesFile struct {
	Queries []string `yaml:"queries"`
}

type expectedFile struct {
	Rankings map[string][]string `yaml:"rankings"`
}

// LoadCorpus reads a corpus directory in the shape produced by
// golden_eval/corpus: a notes/*.md directory, a queries.yaml, and an
// expected.yaml.
func LoadCorpus(dir string) (Corpus, error) {
	notes, err := loadNotes(filepath.Join(dir, "notes"))
	if err != nil {
		return Corpus{}, fmt.Errorf("loading notes: %w", err)
	}

	queries, err := loadQueries(filepath.Join(dir, "queries.yaml"))
	if err != nil {
		return Corpus{}, fmt.Errorf("loading queries: %w", err)
	}

	expected, err := loadExpected(filepath.Join(dir, "expected.yaml"))
	if err != nil {
		return Corpus{}, fmt.Errorf("loading expected rankings: %w", err)
	}

	return Corpus{Notes: notes, Queries: queries, Expected: expected}, nil
}

func loadNotes(dir string) ([]Note, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var notes []Note
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".md" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			return nil, err
		}
		notes = append(notes, Note{ID: entry.Name(), Text: string(data)})
	}

	return notes, nil
}

func loadQueries(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f queriesFile
	if err := yaml.Unmarshal(data, &f); err != nil {
		return nil, err
	}
	return f.Queries, nil
}

func loadExpected(path string) (map[string][]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f expectedFile
	if err := yaml.Unmarshal(data, &f); err != nil {
		return nil, err
	}
	return f.Rankings, nil
}
