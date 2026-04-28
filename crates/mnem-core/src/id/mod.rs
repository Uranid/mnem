//! Identity primitives for mnem.
//!
//! mnem distinguishes two orthogonal identities for every persistent object
//! (SPEC §2):
//!
//! - **Content hash** - a [Multihash] wrapped in a [CID]; identifies the
//!   byte-exact canonical encoding of the object. Changes whenever the
//!   content changes.
//! - **Stable identifier** - a 128-bit `UUIDv7` identifier ([`NodeId`],
//!   [`EdgeId`], [`ChangeId`], [`OperationId`]) that survives edits. Edges
//!   point at [`NodeId`] values, never at content hashes, so that a node
//!   property edit does not invalidate every edge.
//!
//! - Multihash + CID] and [- Dual identity].
//!
//! [Multihash]: https://github.com/multiformats/multihash
//! [CID]: https://github.com/multiformats/cid
//! [- Multihash + CID]: https://github.com/Uranid/mnem/blob/main/
//! [- Dual identity]: https://github.com/Uranid/mnem/blob/main/

pub mod cid;
pub mod link;
pub mod multihash;
pub mod stable;

pub use cid::{CODEC_DAG_CBOR, CODEC_RAW, Cid};
pub use link::Link;
pub use multihash::{HASH_BLAKE3_256, HASH_SHA2_256, Multihash};
pub use stable::{ChangeId, EdgeId, NodeId, OperationId, StableId};
