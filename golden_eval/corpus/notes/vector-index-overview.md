riffle/indexing
vector-index-overview.md

# Vector Index Overview

Describes the HNSW vector store used for semantic search over embedded chunks: index construction parameters, approximate nearest-neighbour search, and how it combines with the Merkle-hash structural layer. Distinct from change detection: this note covers how similarity search works once vectors exist, not how Riffle decides what to re-embed.

Related: [[Change Detection via Merkle Hashing]], [[Hybrid Retrieval]]
