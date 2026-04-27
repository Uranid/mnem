//! LLM-free `GraphRAG` primitives over mnem's [`AdjacencyIndex`].
//!
//! This crate fuses two experiments onto one substrate:
//!
//! - **E1 (Leiden)** - [`community`] provides modularity-optimising
//!   community detection producing a deterministic, content-addressable
//!   [`CommunityAssignment`] from any `AdjacencyIndex` (authored, KNN,
//!   or hybrid).
//! - **E4 (Summarize)** - [`summarize`] provides an extractive
//!   Centroid+MMR summarizer over community members, reusing
//!   [`mnem_embed_providers::Embedder`].
//! - **Gap 16 (Calibration)** - [`calibration`] emits scale-free
//!   per-query score quantiles and a categorical distribution-shape
//!   label so agents can interpret dense-retrieval scores without a
//!   trained cross-embedder scaler.
//!
//! # Non-goals
//!
//! - No LLM: summarization is *extractive*, returning existing sentences.
//! - No BM25 .
//! - No network, no heavy deps beyond mnem-core / mnem-embed-providers.
//!
//! # Determinism
//!
//! Every public entry point in this crate is seeded. Given the same
//! input and seed, two independent runs produce byte-identical output.
//!
//! [`AdjacencyIndex`]: mnem_core::index::AdjacencyIndex

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod calibration;
pub mod community;
pub mod confidence;
pub mod summarize;

pub use calibration::{
    K_MIN, ScoreDistribution, ShapeLabel, WILSON_WIDTH_TARGET, WILSON_Z, derive_k_min,
    distribution_shape, node_score_quantiles, score_quantiles,
};
pub use community::{CommunityAssignment, CommunityId, compute_communities};
pub use confidence::{
    K_MIN_SHAPE_GATE, RankAgreement, median_topk_margin_pct, normalized_entropy, rank_agreement,
};
pub use summarize::{Summary, SummaryItem, summarize_community};
