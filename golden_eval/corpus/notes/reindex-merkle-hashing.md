riffle/indexing
merkle-hashing.md

# Change Detection via Merkle Hashing

Describes how Riffle detects which parts of a vault changed since the last index. On re-index, Riffle compares the stored root hash to the recomputed root hash. If equal, nothing has changed. If different, it descends only into directories whose hash has changed, re-embedding only those — avoiding a full re-embed of the vault on every run.

Related: [[Vector Index Overview]], [[Reindex Scheduling]]
