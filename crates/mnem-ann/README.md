# mnem-ann

Approximate-nearest-neighbour (HNSW) vector index for mnem. Feature-gated alternative to the built-in brute-force index in `mnem-core`.

Implements the `VectorIndex` trait from `mnem-core` with an HNSW backend.
The built-in `BruteForceVectorIndex` is 100%-recall and O(n x dim) per
query; `HnswVectorIndex` trades ~1% recall for O(log n x dim) queries and
is the right choice once a repo grows past roughly 10k indexed vectors or
for long-lived servers where query latency dominates. Both impls return
`Vec<VectorHit>` sorted by descending score with `NodeId`-ASC tiebreak, so
replay stays byte-stable either way. Kept in a separate crate because most
HNSW implementations carry SIMD intrinsics or architecture-specific unsafe
blocks that would fail `mnem-core`'s `#![forbid(unsafe_code)]` and WASM
targets.

```rust
use mnem_ann::HnswVectorIndex;
use mnem_core::index::vector::VectorIndex;

let idx = HnswVectorIndex::build_from_repo(&repo, "openai:text-embedding-3-small")?;
let hits = idx.search(&query_vec, 10)?;
```

Workspace top: [`../../README.md`](../../README.md).

Licensed under Apache-2.0.
