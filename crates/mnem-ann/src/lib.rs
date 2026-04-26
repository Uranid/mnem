//! # mnem-ann
//!
//! Approximate-nearest-neighbour vector indexes for mnem. Alternative
//! backend to [`mnem_core::index::vector::BruteForceVectorIndex`]; the
//! trait surface is shared so a retriever built against `VectorIndex`
//! works with either impl.
//!
//! | Index | Recall | Query latency | Build latency | When to use |
//! |---|---|---|---|---|
//! | [`mnem_core::index::vector::BruteForceVectorIndex`] | 100% | O(n * dim) | O(1) | N ≤ ~10k; cold repos; CI tests |
//! | [`HnswVectorIndex`] (this crate, `hnsw` feature) | ~99% | O(log n * dim) | O(n * log n * dim) | N > 10k; warm long-lived servers |
//!
//! Both return `Vec<VectorHit>` sorted by descending score with
//! `NodeId`-ASC tiebreak for byte-stable replay.
//!
//! ## Why a separate crate
//!
//! `mnem-core` is `#![forbid(unsafe_code)]` and WASM-clean.
//! Most high-performance ANN implementations carry SIMD
//! intrinsics or architecture-specific unsafe blocks that don't
//! compile to wasm32. Keeping HNSW out of core preserves both
//! properties; users on WASM targets simply don't depend on this
//! crate.
//!
//! ## Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use mnem_ann::HnswVectorIndex;
//! use mnem_core::index::vector::VectorIndex;
//! use mnem_core::repo::ReadonlyRepo;
//!
//! # fn demo(repo: &ReadonlyRepo) -> Result<(), Box<dyn std::error::Error>> {
//! let idx = HnswVectorIndex::build_from_repo(repo, "openai:text-embedding-3-small")?;
//! let query = vec![0.1_f32; idx.dim() as usize];
//! let hits  = idx.search(&query, 10)?;
//! for h in hits {
//!     println!("{}  {:.3}", h.node_id.to_uuid_string(), h.score);
//! }
//! # Ok(()) }
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[cfg(feature = "hnsw")]
mod hnsw;

pub mod knn_edges;

#[cfg(feature = "hnsw")]
pub use hnsw::{HnswConfig, HnswVectorIndex};

pub use knn_edges::{DistanceMetric, KnnEdge, KnnEdgeIndex, derive_knn_edges_from_vectors};

#[cfg(feature = "hnsw")]
pub use knn_edges::derive_knn_edges;
