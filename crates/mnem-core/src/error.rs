//! Top-level error type for `mnem-core`.
//!
//! Each module defines its own `thiserror`-based error enum that `From`-converts
//! into [`enum@Error`]. Public-facing APIs return `Result<T, Error>`.
//!
//! `mnem-core` never panics on user input. Invariant violations that are
//! logically impossible (e.g. a `NodeId` of the wrong length after validated
//! construction) use `debug_assert!` in debug builds and return `Error` in
//! release builds.

use thiserror::Error;

/// Top-level error type returned by `mnem-core` public APIs.
///
/// Variants are intentionally coarse-grained. Each module's native error type
/// carries the detail; this top-level enum exists so callers can match on
/// category without depending on every sub-module's error shape.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// An identity primitive (stable ID, multihash, CID, link) was malformed.
    #[error("id: {0}")]
    Id(#[from] IdError),
    /// A canonical-encoding round-trip failed.
    #[error("codec: {0}")]
    Codec(#[from] CodecError),
    /// A blockstore operation (has/get/put/delete) failed.
    #[error("store: {0}")]
    Store(#[from] StoreError),
    /// An object (Node/Edge/Tree/Commit/...) was malformed or invalid.
    #[error("object: {0}")]
    Object(#[from] ObjectError),
    /// A repository-level operation failed (init, open, commit, ...).
    #[error("repo: {0}")]
    Repo(#[from] RepoError),
    /// A signing or verification operation failed.
    #[error("sign: {0}")]
    Sign(#[from] SignError),
    // Remote, etc. variants land as those modules arrive.
}

/// Convenient result alias.
pub type Result<T> = core::result::Result<T, Error>;

impl Error {
    /// `true` iff this error means "the op-heads store is empty; call
    /// [`crate::repo::ReadonlyRepo::init`] first". Callers typically
    /// use it to decide whether to auto-initialise vs. propagate.
    ///
    /// Prefer this over stringly-typed `format!("{e}").contains(...)`
    /// matches: the latter silently breaks on any wording change.
    #[must_use]
    pub const fn is_uninitialized(&self) -> bool {
        matches!(self, Self::Repo(RepoError::Uninitialized))
    }
}

/// Errors from [`crate::id`] - stable IDs, multihash, CID, link.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum IdError {
    /// A stable-ID byte string has the wrong length (expected 16).
    #[error("stable id: expected 16 bytes, got {got}")]
    StableIdLength {
        /// Length of the input in bytes.
        got: usize,
    },
    /// A stable-ID string was not a valid UUID.
    #[error("stable id: not a valid uuid: {source}")]
    StableIdParse {
        /// Underlying `uuid` crate error.
        #[source]
        source: uuid::Error,
    },
    /// A multihash could not be constructed or decoded.
    #[error("multihash: {source}")]
    Multihash {
        /// Underlying `multihash` crate error.
        #[source]
        source: multihash::Error,
    },
    /// A CID could not be constructed or decoded.
    #[error("cid: {source}")]
    Cid {
        /// Underlying `cid` crate error.
        #[source]
        source: cid::Error,
    },
    /// A link was annotated with a codec that doesn't match the underlying CID.
    #[error("link: expected codec 0x{expected:x}, got 0x{got:x}")]
    LinkWrongCodec {
        /// Codec the caller expected the link to point at.
        expected: u64,
        /// Codec the CID actually carries.
        got: u64,
    },
}

/// Errors from [`crate::codec`] - canonical encode/decode.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CodecError {
    /// Encoding a value to canonical DAG-CBOR (or DAG-JSON) failed.
    #[error("encode: {0}")]
    Encode(String),
    /// Decoding canonical bytes back into a value failed.
    #[error("decode: {0}")]
    Decode(String),
    /// The value contained a form forbidden by the mnem canonical rules
    /// (NaN/Inf float, indefinite-length marker, non-string map key, etc.).
    #[error("non-canonical form: {0}")]
    NonCanonical(String),
}

/// Errors from [`crate::store`] - blockstore operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum StoreError {
    /// The CID attached to a `put` does not match a hash of the bytes.
    /// Returned only by backends that choose to verify - the default
    /// contract trusts the caller (ARCHITECTURE §3.1; see also the
    /// Phase 1 risk review: re-hashing on every put is a hot-path tax).
    #[error("cid mismatch: claimed {claimed}, computed {computed}")]
    CidMismatch {
        /// CID the caller asserted.
        claimed: crate::id::Cid,
        /// CID the backend computed from the bytes.
        computed: crate::id::Cid,
    },
    /// A CID referenced by a tree walk (lookup, cursor, diff, GC, …) is
    /// not present in the blockstore. Indicates a broken or partial tree.
    #[error("not found: {cid}")]
    NotFound {
        /// The CID that was looked up and missing.
        cid: crate::id::Cid,
    },
    /// Backend-specific I/O failure, translated into a string.
    #[error("io: {0}")]
    Io(String),
    /// On-disk content does not hash to the CID it is stored under.
    /// Indicates silent disk corruption or a store-level bug (e.g. a
    /// `put_trusted` caller that violated the safety contract).
    /// The caller MUST treat this as an unrecoverable integrity failure
    /// for the affected block.
    #[error("corruption: block stored under {cid} hashes to a different CID: {detail}")]
    Corruption {
        /// The CID that was requested (and used as the storage key).
        cid: String,
        /// Human-readable description of the mismatch.
        detail: String,
    },
}

/// Errors from [`crate::sign`] - commit / operation signing + verification.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SignError {
    /// Object carries no `signature` field; verification requires one.
    #[error("no signature attached")]
    NoSignature,
    /// Signature field uses an algorithm this implementation doesn't support.
    #[error("unsupported signature algorithm: {got}")]
    WrongAlgorithm {
        /// Algorithm tag found on the object.
        got: String,
    },
    /// Public key is not the expected 32 bytes or fails Ed25519 decode.
    #[error("malformed public key")]
    MalformedKey,
    /// Signature bytes are not the expected 64 bytes.
    #[error("malformed signature")]
    MalformedSignature,
    /// Ed25519 verify rejected the signature.
    #[error("signature verification failed")]
    InvalidSignature,
    /// Re-canonicalising the object for verification failed.
    #[error("encoding: {0}")]
    Encoding(String),
    /// The signing key is present in the revocation list and the object's
    /// timestamp is strictly after the revocation (SPEC §9.2).
    #[error("key revoked at {revoked_at} µs (object time: {time} µs)")]
    RevokedKey {
        /// Microseconds-since-epoch moment the key was revoked.
        revoked_at: u64,
        /// The signed object's `time`.
        time: u64,
    },
}

/// Errors from [`crate::repo`] - repository lifecycle operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RepoError {
    /// `open` was called on an op-heads store with no heads. Call
    /// [`crate::repo::ReadonlyRepo::init`] first.
    #[error("repository not initialized (op-heads store is empty)")]
    Uninitialized,
    /// A CAS or linearized operation observed state that no longer
    /// matches the caller's expectations (SPEC §6.4 / §6.5). Retry
    /// against a fresh `ReadonlyRepo`.
    #[error("stale: observed state is no longer current")]
    Stale,
    /// Op-DAG is malformed - heads do not share any common ancestor.
    /// Cannot happen in a well-formed repository (all heads descend
    /// from the root op); indicates corruption or partial import.
    #[error("op-DAG has no common ancestor across the current heads")]
    NoCommonAncestor,
    /// `Query::one` (or a similar precondition API) found zero matches.
    #[error("query found zero matches")]
    NotFound,
    /// `Query::one` (or a similar precondition API) found multiple
    /// matches where exactly one was required.
    #[error("query found multiple matches where exactly one was required")]
    AmbiguousMatch,
    /// A secondary index pointed at a block that is missing, malformed,
    /// or whose contents contradict the index (wrong label, wrong prop
    /// value). Indicates corruption or a partial import; does not
    /// trigger on a simple "no such key" miss.
    #[error("index corruption: {context} (cid = {cid})")]
    IndexCorrupt {
        /// Short description of which index + which key was involved.
        context: String,
        /// The CID the index pointed at.
        cid: crate::id::Cid,
    },
    /// A vector-search query's dimension did not match the index
    /// dimension. Each vector index binds to one model + dim at build
    /// time; agents must pass a query vector of the exact same shape.
    #[error("vector dim mismatch: index dim is {index_dim}, query dim is {query_dim}")]
    VectorDimMismatch {
        /// Dimension the index was built at.
        index_dim: u32,
        /// Dimension of the query vector the caller passed.
        query_dim: usize,
    },
    /// A [`crate::retrieve::Retriever`] was executed without any
    /// filters or rankers configured. Retrieval needs at least one
    /// label / prop / text / vector input to produce a useful result.
    ///
    /// audit-2026-04-25 P2-1 / P1-3: the error now spells out the
    /// common remediation path (text query needs an embedder) so CLI
    /// and MCP callers do not have to guess.
    #[error(
        "retrieve: no filters or rankers configured. \
         A text query requires an embedder: \
         `mnem config set embed.provider ollama && \
         mnem config set embed.model nomic-embed-text` \
         (or pass `--where K=V` / `--label L` for a pure filter query)."
    )]
    RetrievalEmpty,
    /// C8: `add_edge` was called with an endpoint that does not exist
    /// in the current view (neither in the base commit's node tree nor
    /// in nodes staged in this transaction). Committing such an edge
    /// would produce a dangling reference that retrieval and graph-
    /// expand cannot resolve.
    #[error("dangling edge: node {id} ({role}) does not exist in the current view")]
    DanglingEdge {
        /// The `NodeId` that was not found.
        id: crate::id::NodeId,
        /// `"src"` or `"dst"` - which endpoint is missing.
        role: &'static str,
    },
}

/// Errors from [`crate::objects`] - Node/Edge/Tree/... validation and decode.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ObjectError {
    /// The `_kind` discriminator on the wire doesn't match the expected
    /// Rust type. For example, decoding an Edge as a Node.
    #[error("wrong kind: expected '{expected}', got '{got}'")]
    WrongKind {
        /// The `_kind` value the Rust type expects.
        expected: &'static str,
        /// The `_kind` value found in the encoded bytes.
        got: String,
    },
    /// An [`crate::objects::Embedding`]'s `vector` length does not match
    /// `dim × bytes_per_dtype(dtype)` (SPEC §4.1).
    #[error("embedding size mismatch: expected {expected} bytes, got {got}")]
    EmbeddingSizeMismatch {
        /// Required vector length: `dim × bytes_per_dtype(dtype)`.
        expected: usize,
        /// Actual vector length.
        got: usize,
    },
}
