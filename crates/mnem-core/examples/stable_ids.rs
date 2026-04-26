//! Exercise `mnem-core::id::stable` from a standalone binary.
//!
//! Intended as a capability smoke-test during Phase 1 M2. Run with:
//!
//! ```console
//! cargo run --example stable_ids
//! ```
//!
//! The binary generates one of each stable-ID role, prints them, and
//! exercises the byte / UUID round trips. Any panic indicates a regression
//! in the stable-ID primitives. The output is written to stdout; during
//! development the operator redirects it to `/tmp/mnem-test/stable_ids.out`
//! for inspection.

use mnem_core::id::{ChangeId, EdgeId, NodeId, OperationId};

fn main() {
    println!("# mnem stable IDs - capability smoke test");
    println!("# mnem-core version: {}", mnem_core::VERSION);
    println!("# mnem format version: {}", mnem_core::FORMAT_VERSION);
    println!();

    let n = NodeId::new_v7();
    let e = EdgeId::new_v7();
    let c = ChangeId::new_v7();
    let o = OperationId::new_v7();

    println!("NodeId       = {n:?}");
    println!("EdgeId       = {e:?}");
    println!("ChangeId     = {c:?}");
    println!("OperationId  = {o:?}");
    println!();

    // Byte round-trip
    let n_bytes = *n.as_bytes();
    let n2 = NodeId::from_bytes(&n_bytes).expect("16 bytes should parse");
    assert_eq!(n, n2, "NodeId byte round-trip mismatch");
    println!("byte round-trip: ok");

    // UUID-string round-trip
    let n_str = n.to_uuid_string();
    let n3 = NodeId::parse_uuid(&n_str).expect("valid uuid");
    assert_eq!(n, n3, "NodeId UUID-string round-trip mismatch");
    println!("uuid-string round-trip: ok  ({n_str})");

    // Bulk generation - asserts no panics under tight loop
    let ids: Vec<NodeId> = (0..10_000).map(|_| NodeId::new_v7()).collect();
    assert_eq!(ids.len(), 10_000);
    println!("bulk-gen 10k NodeIds: ok");

    // Uniqueness - with UUIDv7's 74 random bits, 10k in-process are expected
    // to be unique with overwhelming probability. If this ever fires, the
    // RNG source is broken.
    let mut seen = std::collections::HashSet::with_capacity(10_000);
    for id in &ids {
        assert!(seen.insert(*id.as_bytes()), "duplicate NodeId");
    }
    println!("uniqueness over 10k: ok");

    println!();
    println!("# all stable-id smoke tests passed.");
}
