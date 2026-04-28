//! Canonical encoding and decoding of mnem objects.
//!
//! Two codecs are exposed:
//!
//! - [`dagcbor`] - the canonical storage format. Byte-exact deterministic
//!   per SPEC §3. Every content hash in mnem is computed over DAG-CBOR
//!   output.
//! - [`dagjson`] - a debug / inspection format. Never hashed, never written
//!   to the object store as canonical content. Useful for
//!   `mnem cat-file --json` and error messages.
//!
//! The most common operation is [`dagcbor::hash_to_cid`] which encodes a
//! value to canonical CBOR and computes its content-addressed [CID].
//!
//! [CID]: crate::id::Cid

pub mod dagcbor;
pub mod dagjson;
pub mod json;

pub use dagcbor::{extract_links, from_canonical_bytes, hash_to_cid, to_canonical_bytes};
pub use dagjson::{from_json_bytes, to_json_bytes};
pub use json::{IPLD_MAX_DEPTH, JsonIpldError, json_to_ipld};
