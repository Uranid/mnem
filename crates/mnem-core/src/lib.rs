//! # mnem-core
//!
//! The core library for [mnem] - a content-addressed, versioned substrate for
//! AI agent memory.
//!
//! This crate contains the format types, canonical encoding, content hashing,
//! Prolly-tree algorithms, operation-log machinery, and repository API.
//!
//! ## Scope
//!
//! `mnem-core` is deliberately factored to be embeddable in every runtime.
//! It has:
//!
//! - No terminal I/O (`println!`, `eprintln!` are forbidden inside this crate)
//! - No config-file loading (callers supply config; we consume it)
//! - No direct filesystem access - storage is behind the [`Blockstore`][store::Blockstore] trait
//!   (implemented in `mnem-backend-redb` or callers' own backends)
//! - No `tokio` runtime binding (sync API; async wrappers live in callers)
//!
//! These constraints are what make the same source compile to native binaries,
//! WASM, and FFI-consumed libraries in Python / Node / Go.
//!
//! ## Modules
//!
//! - [`id`] - identity primitives: `Multihash`, [`Cid`][id::Cid], stable
//!   [`NodeId`][id::NodeId] / [`EdgeId`][id::EdgeId] /
//!   [`ChangeId`][id::ChangeId] / [`OperationId`][id::OperationId], and
//!   phantom-typed [`Link<T>`][id::Link].
//! - [`codec`] - canonical DAG-CBOR encode/decode and DAG-JSON debug export.
//! - [`objects`] - [`Node`], [`Edge`], [`Commit`], [`Operation`], [`View`],
//!   [`IndexSet`] types. Prolly tree chunks live under [`prolly::TreeChunk`].
//! - [`prolly`] - Prolly tree algorithms (chunker, builder, lookup, cursor,
//!   diff, merge).
//! - [`store`] - [`Blockstore`][store::Blockstore] and
//!   [`OpHeadsStore`][store::OpHeadsStore] traits, plus in-memory
//!   reference implementations.
//! - [`repo`] - [`ReadonlyRepo`], [`Transaction`] facade.
//! - [`index`] - secondary indexes ([`Query`],
//!   [`BruteForceVectorIndex`]).
//! - [`retrieve`] - agent-facing [`Retriever`] that composes filters,
//!   vector and sparse ranking, and token-budget packing.
//! - [`sign`] - Ed25519 signing and revocation-list verification.
//!
//! ## Crate-level invariants
//!
//! - `#![forbid(unsafe_code)]` - no `unsafe` in this crate.
//! - Every object type preserves the byte-exact canonical-encoding round-trip
//!   property (`decode(encode(x)) == x` and `encode(decode(b)) == b`).
//! - Every `put` to a [`Blockstore`][store::Blockstore] verifies `cid == cid_of(bytes)`.
//! - No panic on user input. All fallible paths return [`Error`].
//!
//! ## Status
//!
//! Core library, CLI, MCP, Python bindings, and retrieval surface are shipped.
//! Remote protocol is next. See `docs/ROADMAP.md` for the current phase state
//! and scope.
//!
//! [mnem]: https://github.com/Uranid/mnem

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod anchor;
pub mod codec;
pub mod error;
pub mod guard;
pub mod id;
pub mod index;
pub mod llm;
pub mod objects;
pub mod ppr;
pub mod prolly;
pub mod repo;
pub mod rerank;
pub mod retrieve;
pub mod sign;
pub mod sparse;
pub mod store;

pub use error::{Error, RepoError, Result};

// Agent-facing retrieval shortcuts (re-exports so callers don't need
// to reach into `mnem_core::index::*` paths for the common types).
pub use index::{BruteForceVectorIndex, PropPredicate, Query, QueryHit, VectorHit, VectorIndex};
pub use objects::{
    Commit, Dtype, Edge, Embedding, EmbeddingBucket, EmbeddingEntry, IndexSet, Node, Operation,
    RefTarget, View,
};
pub use repo::{ReadonlyRepo, Transaction};
pub use retrieve::{
    HeuristicEstimator, RetrievalResult, RetrievedItem, Retriever, TokenEstimator, render_node,
};

/// Library version (tracks the workspace package version).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// mnem format version this crate implements (see `docs/SPEC.md` §13).
///
/// Bumps when the on-wire object schema changes in a non-backward-compatible
/// way. Pre-0.2 nodes are still decodable because new fields (e.g.
/// `Node.summary` added in 0.2) are encoded with `skip_serializing_if`.
pub const FORMAT_VERSION: &str = "mnem/0.2";

/// Canonical prefix for branch refs (e.g. `refs/heads/main`).
///
/// Every branch in mnem lives under this namespace. Using this constant
/// instead of the raw string literal ensures that a single rename is
/// sufficient if the convention ever changes (BUG-13).
pub const HEADS_PREFIX: &str = "refs/heads/";

/// Canonical prefix for tag refs (e.g. `refs/tags/v1.0`).
///
/// Every tag in mnem lives under this namespace. Using this constant
/// instead of the raw string literal ensures that a single rename is
/// sufficient if the convention ever changes (BUG-41).
pub const TAGS_PREFIX: &str = "refs/tags/";
