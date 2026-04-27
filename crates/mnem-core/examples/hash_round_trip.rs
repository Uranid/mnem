//! Exercise `mnem-core::codec` + `::id` end-to-end.
//!
//! Builds a small typed record, encodes it to canonical DAG-CBOR, computes
//! its content-addressed CID, decodes it back, re-encodes for byte-identity
//! verification, and mutates a field to confirm CIDs diverge under content
//! change. Also exports the DAG-JSON debug form. Any panic indicates a
//! regression in the codec / id primitives.
//!
//! Run with:
//!
//! ```console
//! cargo run --example hash_round_trip
//! ```
//!
//! Development operator redirects output to
//! `/tmp/mnem-test/hash_round_trip.out` for inspection.

use mnem_core::codec::{from_canonical_bytes, hash_to_cid, to_canonical_bytes, to_json_bytes};
use mnem_core::id::NodeId;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
struct Ping {
    id: NodeId,
    label: String,
    number: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("# mnem codec - hash + round-trip smoke test");
    println!("# mnem-core version: {}", mnem_core::VERSION);
    println!("# mnem format:       {}", mnem_core::FORMAT_VERSION);
    println!();

    let original = Ping {
        id: NodeId::new_v7(),
        label: "hello".into(),
        number: 42,
    };
    println!("original = {original:?}");

    let (bytes, cid) = hash_to_cid(&original)?;
    println!("encoded:  {} bytes", bytes.len());
    println!("cid:      {cid}");
    println!();

    // --- Round-trip: decode back, compare ---
    let decoded: Ping = from_canonical_bytes(&bytes)?;
    assert_eq!(original, decoded, "round-trip decode mismatch");
    println!("round-trip decode:       ok");

    // --- Byte-identity: re-encode the decoded value, compare bytes & CID ---
    let bytes2 = to_canonical_bytes(&decoded)?;
    assert_eq!(
        bytes, bytes2,
        "canonical re-encode must produce byte-identical output"
    );
    let (_, cid2) = hash_to_cid(&decoded)?;
    assert_eq!(cid, cid2, "re-encoded CID must match");
    println!("canonical byte identity: ok");

    // --- Content sensitivity: flip a field, confirm CID changes ---
    let mut mutated = original.clone();
    mutated.number = 43;
    let (_, cid_mut) = hash_to_cid(&mutated)?;
    assert_ne!(cid, cid_mut, "mutation must change CID");
    println!("content sensitivity:     ok  ({cid} != {cid_mut})");

    // --- DAG-JSON debug export ---
    let json = to_json_bytes(&original)?;
    let json_str = std::str::from_utf8(&json)?;
    println!();
    println!("dag-json:");
    println!("{json_str}");

    println!();
    println!("# all codec / id smoke tests passed.");
    Ok(())
}
