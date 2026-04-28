//! Canonical DAG-CBOR encode / decode.
//!
//! All of mnem's on-disk and on-wire object bytes are produced by this
//! module. The canonical-form guarantees (definite-length encoding, sorted
//! map keys, no NaN/Inf, CID tag 42) come from [`serde_ipld_dagcbor`];
//! this module is a thin wrapper that:
//!
//! 1. Translates encode / decode errors into mnem's own [`CodecError`],
//!    hiding the underlying crate's type names from public signatures.
//! 2. Provides the combined "encode + hash-to-CID" operation that is the
//!    unit of most callers' work.

use bytes::Bytes;
use ipld_core::ipld::Ipld;
use serde::{Serialize, de::DeserializeOwned};

use crate::error::CodecError;
use crate::id::cid::{CODEC_DAG_CBOR, Cid};
use crate::id::multihash::Multihash;

/// Encode a value to canonical DAG-CBOR bytes.
///
/// # Errors
///
/// Returns [`CodecError::Encode`] if the value cannot be serialized under
/// the DAG-CBOR canonical rules (e.g. contains a NaN/Inf float, an
/// indefinite-length marker, or a non-string map key).
pub fn to_canonical_bytes<T: Serialize>(value: &T) -> Result<Bytes, CodecError> {
    serde_ipld_dagcbor::to_vec(value)
        .map(Bytes::from)
        .map_err(|e| CodecError::Encode(e.to_string()))
}

/// Decode canonical DAG-CBOR bytes into a value.
///
/// # Errors
///
/// Returns [`CodecError::Decode`] if the bytes are malformed or do not
/// match the target type `T`.
pub fn from_canonical_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, CodecError> {
    serde_ipld_dagcbor::from_slice(bytes).map_err(|e| CodecError::Decode(e.to_string()))
}

/// Encode a value to canonical DAG-CBOR and compute the resulting CID.
///
/// This is the principal operation on content-addressed objects. The
/// returned bytes are suitable for `Blockstore::put`; the returned CID is
/// the object's address.
///
/// # Errors
///
/// Returns [`CodecError::Encode`] if encoding fails.
pub fn hash_to_cid<T: Serialize>(value: &T) -> Result<(Bytes, Cid), CodecError> {
    let bytes = to_canonical_bytes(value)?;
    let hash = Multihash::sha2_256(&bytes);
    let cid = Cid::new(CODEC_DAG_CBOR, hash);
    Ok((bytes, cid))
}

/// Maximum nesting depth [`extract_links`] will traverse.
///
/// Matches the `json_to_ipld` cap in the mnem-http / mnem-cli /
/// mnem-mcp input paths so a payload that round-trips through either
/// layer behaves consistently. Nothing in mnem's typed object set
/// nests past ~8 (commit -> view -> keys -> entry -> props -> list),
/// so 64 is ~8x the deepest legitimate shape while keeping attacker
/// stack-overflow well out of reach.
pub const WALK_IPLD_MAX_DEPTH: usize = 64;

/// Extract every CID link embedded in a DAG-CBOR block.
///
/// The block is decoded into a generic [`Ipld`] value and walked for
/// [`Ipld::Link`] leaves, which are then converted back into mnem's
/// [`Cid`] type via the binary wire form (`to_bytes` / `from_bytes`
/// round-trip). Raw blocks (codec 0x55) have no outgoing links by
/// definition; callers typically sniff the codec and skip this step
/// rather than paying the decode cost.
///
/// Order is deterministic but unspecified beyond that: it is a
/// depth-first traversal of the decoded Ipld tree. Callers that need
/// a reproducible serialisation order should sort the result.
///
/// Duplicates are NOT removed - the same CID referenced twice in a
/// block yields two entries. Higher layers that care about
/// deduplication (e.g. blockstore walk, CAR export) handle it there so
/// this helper stays a pure tree walk.
///
/// # Errors
///
/// Returns [`CodecError::Decode`] if `bytes` is not valid DAG-CBOR,
/// or if the decoded tree nests past [`WALK_IPLD_MAX_DEPTH`].
/// Unknown-CID variants inside the Ipld tree propagate as
/// [`CodecError::Decode`] with context - in practice this cannot happen
/// on bytes produced by `to_canonical_bytes`.
pub fn extract_links(bytes: &[u8]) -> Result<Vec<Cid>, CodecError> {
    let root: Ipld = from_canonical_bytes(bytes)?;
    let mut out = Vec::new();
    walk_ipld(&root, &mut out, 0)?;
    Ok(out)
}

fn walk_ipld(node: &Ipld, out: &mut Vec<Cid>, depth: usize) -> Result<(), CodecError> {
    if depth >= WALK_IPLD_MAX_DEPTH {
        return Err(CodecError::Decode(format!(
            "walk_ipld: nesting exceeds depth cap of {WALK_IPLD_MAX_DEPTH}"
        )));
    }
    match node {
        Ipld::Null
        | Ipld::Bool(_)
        | Ipld::Integer(_)
        | Ipld::Float(_)
        | Ipld::String(_)
        | Ipld::Bytes(_) => Ok(()),
        Ipld::List(xs) => {
            for x in xs {
                walk_ipld(x, out, depth + 1)?;
            }
            Ok(())
        }
        Ipld::Map(m) => {
            for v in m.values() {
                walk_ipld(v, out, depth + 1)?;
            }
            Ok(())
        }
        Ipld::Link(c) => {
            // Round-trip via binary form to translate from
            // `ipld_core::cid::Cid` (what Ipld::Link holds) into
            // mnem-core's own `Cid` without leaking the dependency.
            let bytes = c.to_bytes();
            let cid =
                Cid::from_bytes(&bytes).map_err(|e| CodecError::Decode(format!("link: {e}")))?;
            out.push(cid);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::NodeId;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
    struct Fixture {
        id: NodeId,
        label: String,
        number: u64,
    }

    fn sample() -> Fixture {
        Fixture {
            id: NodeId::from_bytes_raw([7u8; 16]),
            label: "hello".into(),
            number: 42,
        }
    }

    #[test]
    fn round_trip_byte_identity() {
        let original = sample();
        let bytes1 = to_canonical_bytes(&original).expect("encode");
        let decoded: Fixture = from_canonical_bytes(&bytes1).expect("decode");
        assert_eq!(original, decoded, "decode mismatch");
        let bytes2 = to_canonical_bytes(&decoded).expect("re-encode");
        assert_eq!(bytes1, bytes2, "canonical re-encode must be byte-identical");
    }

    #[test]
    fn hash_to_cid_is_deterministic() {
        let sample = sample();
        let (bytes1, cid1) = hash_to_cid(&sample).expect("encode+hash");
        let (bytes2, cid2) = hash_to_cid(&sample).expect("encode+hash again");
        assert_eq!(bytes1, bytes2);
        assert_eq!(cid1, cid2);
        assert_eq!(cid1.codec(), CODEC_DAG_CBOR);
    }

    #[test]
    fn different_content_different_cid() {
        let mut a = sample();
        let mut b = sample();
        b.number = 43;
        let (_, cid_a) = hash_to_cid(&a).expect("encode a");
        let (_, cid_b) = hash_to_cid(&b).expect("encode b");
        assert_ne!(cid_a, cid_b);
        // Restoring b to match a yields the same CID.
        b.number = a.number;
        a.label.clear();
        a.label.push_str("hello");
        let (_, cid_b2) = hash_to_cid(&b).expect("encode b restored");
        assert_eq!(cid_a, cid_b2);
    }

    #[test]
    fn stable_id_encodes_as_byte_string() {
        // NodeId::from_bytes_raw([7; 16]) serialized on its own in DAG-CBOR
        // must be exactly 17 bytes: 0x50 (major type 2, length 16) + 16x 0x07.
        let id = NodeId::from_bytes_raw([7u8; 16]);
        let bytes = to_canonical_bytes(&id).expect("encode");
        assert_eq!(bytes.len(), 17);
        assert_eq!(bytes[0], 0x50);
        for &b in &bytes[1..] {
            assert_eq!(b, 0x07);
        }
    }

    #[test]
    fn extract_links_on_leaf_block_is_empty() {
        // A block with no embedded CIDs yields an empty link list.
        let bytes = to_canonical_bytes(&sample()).expect("encode");
        let links = extract_links(&bytes).expect("extract");
        assert!(links.is_empty(), "leaf block has no links, got {links:?}");
    }

    #[test]
    fn extract_links_finds_cid_tags() {
        // Construct a block that embeds two Link values, one at top level
        // and one nested under a list + map. Both must be returned.
        use crate::id::{CODEC_RAW, Multihash};
        use ipld_core::ipld::Ipld;
        use std::collections::BTreeMap;

        let make_inner = |seed: u8| -> ipld_core::cid::Cid {
            let ours = Cid::new(CODEC_RAW, Multihash::sha2_256(&[seed]));
            ipld_core::cid::Cid::try_from(ours.to_bytes().as_slice()).unwrap()
        };
        let a_inner = make_inner(1);
        let b_inner = make_inner(2);

        let mut top = BTreeMap::new();
        top.insert("direct".to_string(), Ipld::Link(a_inner.clone()));
        top.insert(
            "nested".to_string(),
            Ipld::List(vec![Ipld::Map(
                [("x".to_string(), Ipld::Link(b_inner.clone()))]
                    .into_iter()
                    .collect(),
            )]),
        );
        let value = Ipld::Map(top);
        let bytes = to_canonical_bytes(&value).expect("encode");

        let links = extract_links(&bytes).expect("extract");
        assert_eq!(links.len(), 2);
        let a_ours = Cid::from_bytes(&a_inner.to_bytes()).unwrap();
        let b_ours = Cid::from_bytes(&b_inner.to_bytes()).unwrap();
        // DAG-CBOR sorts map keys, so "direct" comes before "nested".
        assert_eq!(links[0], a_ours);
        assert_eq!(links[1], b_ours);
    }

    #[test]
    fn extract_links_rejects_malformed_bytes() {
        let err = extract_links(b"\xff\xff garbage").expect_err("must fail");
        assert!(matches!(err, CodecError::Decode(_)));
    }

    #[test]
    fn walk_ipld_rejects_deeply_nested_structure() {
        // Build a list nested `WALK_IPLD_MAX_DEPTH + 4` deep. A
        // well-formed DAG-CBOR encoder has no depth cap of its own,
        // so an attacker can ship arbitrary depth. `extract_links`
        // must bail instead of recursing into stack overflow.
        use ipld_core::ipld::Ipld;
        let mut v = Ipld::Null;
        for _ in 0..(WALK_IPLD_MAX_DEPTH + 4) {
            v = Ipld::List(vec![v]);
        }
        let bytes = to_canonical_bytes(&v).expect("encode");
        let err = extract_links(&bytes).expect_err("depth cap must trip");
        match err {
            CodecError::Decode(msg) => assert!(msg.contains("depth cap"), "got {msg}"),
            other => panic!("wrong variant: {other:?}"),
        }
    }
}
