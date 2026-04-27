//! Transport-layer error type.
//!
//! Kept deliberately coarse-grained: callers distinguish between "the
//! CAR bytes were malformed" (`Car`), "the blockstore refused a block"
//! (`Store`), and "the underlying I/O stream failed" (`Io`). Richer
//! context rides in the variant strings.

use thiserror::Error;

use mnem_core::error::{CodecError, StoreError};
use mnem_core::id::Cid;

/// Errors raised by CAR export / import.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TransportError {
    /// The CAR bytes were malformed (bad header, truncated block,
    /// invalid varint, wrong CAR version).
    #[error("car: {0}")]
    Car(String),

    /// A block inside an imported CAR did not hash to the CID it
    /// claimed. The caller's CAR is untrusted input; this is rejected
    /// instead of silently accepting corrupt blocks.
    #[error("cid mismatch: claimed {claimed}, computed {computed}")]
    CidMismatch {
        /// CID the CAR asserted for the block.
        claimed: Cid,
        /// CID the importer computed from the payload bytes.
        computed: Cid,
    },

    /// A block's CID used a hash algorithm this build does not know how
    /// to recompute. Currently only SHA-256 and BLAKE3-256 are verified
    /// on import; anything else is rejected.
    #[error("unsupported hash algorithm: {0:#x}")]
    UnsupportedHash(u64),

    /// The CAR header advertised a root CID that was not present in any
    /// block actually delivered in the body. Rejected so a downstream
    /// caller can't be deceived into walking a non-existent root.
    #[error("declared root not present in body: {root}")]
    MissingRoot {
        /// Root CID that appeared in the header but never arrived.
        root: Cid,
    },

    /// Total block-payload bytes exceeded the import-time cap. Rejected
    /// mid-stream before the excess data reaches the blockstore.
    #[error("import exceeded size limit: {observed} > {limit} bytes")]
    SizeLimit {
        /// Cap requested by the caller (bytes).
        limit: u64,
        /// Observed total at the moment of rejection (bytes).
        observed: u64,
    },

    /// Wrapped [`CodecError`] from the underlying DAG-CBOR codec
    /// (header decode, link extraction).
    #[error("codec: {0}")]
    Codec(#[from] CodecError),

    /// Wrapped [`StoreError`] from the target blockstore.
    #[error("store: {0}")]
    Store(#[from] StoreError),

    /// Wrapped [`std::io::Error`] from the underlying byte stream.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors raised by the async HTTP `RemoteClient` (only meaningful
/// with the `client` feature). Kept separate from [`TransportError`]
/// so the pure-sync CAR surface stays free of async / reqwest types.
///
/// Variants split the failure modes a caller cares about:
///
/// - `Network` - the HTTP request never reached a response (DNS, TLS,
///   connect, body-read mid-flight). Retryable at the caller's
///   discretion.
/// - `Framing` - the response arrived but the CAR body was malformed.
///   Not retryable on the same URL without a server fix; wraps the
///   pure-sync [`TransportError`].
/// - `CasMismatch` - the server rejected `advance-head` because the
///   stored `old` CID no longer matches the one the client sent.
///   Caller must refresh refs and retry.
/// - `Auth` - 401 / 403 on a push endpoint. The bearer token is
///   missing, expired, or lacks the required capability.
/// - `Protocol` - the server responded with an unexpected status
///   code, missing header, or a `mnem-protocol` version mismatch.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ClientError {
    /// Underlying HTTP request failed before a complete response was
    /// received (connect, TLS handshake, mid-body socket close, ...).
    #[error("network: {0}")]
    Network(String),

    /// CAR body received but failed framing / CID verification.
    #[error("framing: {0}")]
    Framing(#[from] TransportError),

    /// `advance-head` rejected: the stored `old` value drifted between
    /// the client's read and its write. Caller refreshes refs and
    /// retries.
    #[error(
        "cas mismatch on ref {ref_name:?}: expected old={expected}, server reports actual={actual}"
    )]
    CasMismatch {
        /// Ref name the client tried to advance (e.g. `main`).
        ref_name: String,
        /// `old` CID the client sent.
        expected: Cid,
        /// `old` CID the server actually holds.
        actual: Cid,
    },

    /// 401 / 403 on a push endpoint. The bearer token is missing,
    /// expired, or lacks the required capability.
    #[error("auth failed: {0}")]
    Auth(String),

    /// Server responded with an unexpected status code, missing
    /// header, or `mnem-protocol` version mismatch.
    #[error("protocol: {0}")]
    Protocol(String),

    /// JSON serialisation or deserialisation failed. Only raised by
    /// the `client` feature.
    #[cfg(feature = "client")]
    #[error("serde: {0}")]
    Serde(String),
}

#[cfg(feature = "client")]
impl From<reqwest::Error> for ClientError {
    fn from(e: reqwest::Error) -> Self {
        // `reqwest::Error` carries enough context in its Display impl
        // to diagnose connect / TLS / body-read failures.
        Self::Network(e.to_string())
    }
}

#[cfg(feature = "client")]
impl From<serde_json::Error> for ClientError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serde(e.to_string())
    }
}
