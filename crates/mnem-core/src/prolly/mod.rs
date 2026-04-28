//! Prolly tree - content-defined Merkle-tree index .
//!
//! A Prolly tree is the scalable, history-independent index mnem uses for
//! the node, edge, and schema trees of every [`Commit`][crate::objects::Commit].
//! Its key property:
//! given the same logical `(key, value)` set, every conforming
//! implementation produces the same root CID, regardless of insertion
//! order. This is what makes merge and diff cheap and what makes
//! content-addressed indexes possible at 10M+ key scale.
//!
//! ## M5 staging
//!
//! This module was built in phases:
//!
//! - **M5.1** - [`Chunker`] + [`chunk_boundaries`]: the rolling-hash
//!   boundary primitive. Pure, stateful-per-chunk, testable in isolation.
//! - **M5.2** - Leaf / Internal chunk types (implementing SPEC §4.3),
//!   streaming tree builder, root-CID output.
//! - **M5.3** - Lookup, ordered cursor, structural diff.
//! - **M5.4 (complete, retired)** - HAMT comparison implementation and
//!   benchmark that grounded . Retired from the tree post-benchmark;
//!   empirical record preserved in `docs/benchmarks/prolly-vs-hamt.md`.
//!
//! ## Determinism of the boundary rule
//!
//! SPEC §5.1 v0.1.0 specifies a logistic CDF for the boundary probability.
//! This implementation uses a simpler **flat-probability rule bounded by
//! hard min/max** which achieves the same geometric-mean chunk size with
//! purely integer arithmetic (no IEEE-754 portability hazards across
//! `x86_64` / `aarch64` / `wasm32`). SPEC §5.1 will be amended to match
//! before `mnem/1.0`; this is documented in the review triggers.

pub mod chunker;
pub mod constants;
pub mod cursor;
pub mod diff;
pub mod lookup;
pub mod tree;

pub use chunker::{Chunker, chunk_boundaries};
pub use constants::{
    MAX_ENTRIES_PER_CHUNK, MIN_ENTRIES_PER_CHUNK, PROLLY_KEY_BYTES, ProllyKey, ROLLING_KEY,
    ROLLING_WINDOW_BYTES, TARGET_AVG_ENTRIES_PER_CHUNK, THRESHOLD,
};
pub use cursor::Cursor;
pub use diff::{DiffEntry, diff};
pub use lookup::lookup;
pub use tree::{Internal, Leaf, TreeChunk, build_tree, load_chunk, load_tree_chunk};
