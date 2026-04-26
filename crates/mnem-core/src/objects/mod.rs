//! Canonical object types: Node, Edge, Tree, Commit, Operation, View.
//!
//! Each type serializes and deserializes as canonical DAG-CBOR (SPEC §3, §4).
//! Every Rust struct in this module has the `_kind` discriminator baked in
//! via a custom (de)serialize pair, so a Node encoded on the wire cannot
//! round-trip into an Edge by accident.
//!
//! Forward-compat per SPEC §3.2: every object type carries an `extra:
//! BTreeMap<String, Ipld>` extension map that catches fields this version
//! doesn't know about. The extras are preserved on re-encode so that
//! signed objects remain verifiable across version upgrades.

pub mod commit;
pub mod edge;
pub mod embedding_set;
pub mod index_set;
pub mod node;
pub mod operation;
pub mod tombstone;
pub mod view;

pub use commit::{Commit, Signature};
pub use edge::Edge;
pub use embedding_set::{EmbeddingBucket, EmbeddingEntry};
pub use index_set::{
    AdjacencyBucket, AdjacencyEntry, IncomingAdjacencyBucket, IncomingAdjacencyEntry, IndexSet,
};
pub use node::{Dtype, Embedding, Node};
pub use operation::Operation;
pub use tombstone::Tombstone;
pub use view::{RefTarget, View};
