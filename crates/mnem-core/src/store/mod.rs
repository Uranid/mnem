//! Storage-layer traits and reference implementations.
//!
//! The bottom-most storage abstraction is [`Blockstore`]: a content-addressed
//! byte store with `has`/`get`/`put`/`delete` methods. Everything above - the
//! typed object store, the op store, the op-heads store, the repository
//! facade - composes over this trait.
//!
//! - storage backend strategy] and
//! [ARCHITECTURE §3.1].
//!
//! `mnem-core` ships one reference implementation, [`MemoryBlockstore`],
//! that keeps all content in a `HashMap<Cid, Bytes>`. It is used by:
//!
//! - Every unit and integration test in `mnem-core` that doesn't need
//!   persistent state.
//! - WASM builds in browsers where persistent storage is OPFS-driven and
//!   wrapped separately.
//! - `mnem-cli` during dry-run / ephemeral workflows.
//!
//! The persistent backend (`mnem-backend-redb`) lives in its own crate and
//! depends on `mnem-core` for the trait.
//!
//! 

pub mod blockstore;
pub mod op_heads;

pub use blockstore::{Blockstore, MemoryBlockstore};
pub use op_heads::{MemoryOpHeadsStore, OpHeadsStore};
