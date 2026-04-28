//! Repository facade - the user-facing entry point to mnem.
//!
//! `mnem-core`'s repository API composes the pieces built in M1–M7 into
//! something that looks like a VCS:
//!
//! - [`ReadonlyRepo::init`] bootstraps a fresh repository (SPEC §7.5
//!   state: root View with empty heads, root Operation, one entry in
//!   the op-heads store).
//! - [`ReadonlyRepo::open`] loads an existing repository pinned to its
//!   current op-head.
//! - [`ReadonlyRepo::start_transaction`] returns a [`Transaction`] that
//!   accumulates add/remove mutations.
//! - [`Transaction::commit`] atomically rebuilds the node / edge /
//!   schema Prolly trees from the base commit + mutations, writes a
//!   new Commit / View / Operation, and advances the op-head. Returns
//!   a fresh [`ReadonlyRepo`] pinned to the new op.
//!
//! When the op-heads store has >1 current head, [`open`] transparently
//! runs the M8.5 3-way view merge : it finds the
//! op-DAG common ancestor, merges each head's view against it (emitting
//! `RefTarget::Conflicted` for divergent refs), writes a synthetic merge
//! Operation, advances the op-heads store back to a single head, and
//! returns a `ReadonlyRepo` pinned to the merge op. The merge is
//! deterministic so concurrent readers converge on byte-identical ops.
//!
//! [`open`]: ReadonlyRepo::open

pub mod conflict;
pub mod lca;
pub mod merge;
pub mod readonly;
pub mod transaction;

pub use conflict::{
    Conflict, ConflictCategory, ConflictPolicy, EdgeKey, MERGE_CONFLICTS_SCHEMA, MergeConflicts,
    PropTiebreak, detect_conflicts, detect_conflicts_with_policy, detect_conflicts_with_views,
};
pub use lca::{LcaCache, find_lca, find_lca_many};
pub use merge::{
    ConflictSide, MergeOutcome, MergeStrategy, conflict_category_counts, merge_three_way,
    picks_from_strategy,
};
pub use readonly::ReadonlyRepo;
pub use transaction::{CommitOptions, Transaction};
