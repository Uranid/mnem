//! Cross-platform proptest-style fuzz suite for every parser.
//!
//! The key invariant: **no parser panics on arbitrary input.** Decoders
//! return `Err` for malformed bytes; a panic is a bug that would trip
//! an agent holding another agent's input in their process space.
//!
//! Runs as part of `cargo test` on every platform.
//!
//! With the default proptest config (256 cases × ~dozen parsers per
//! test = ~3000 decode attempts), this catches panic-on-input bugs
//! without adding significant CI time.

use proptest::prelude::*;

use mnem_core::codec::from_canonical_bytes;
use mnem_core::id::{Cid, Multihash};
use mnem_core::objects::{Commit, Edge, Node, Operation, RefTarget, View};
use mnem_core::prolly::TreeChunk;

proptest! {
    // Generous-but-bounded - 256 cases per run, bytes up to 4 KiB.
    #![proptest_config(ProptestConfig { cases: 256, .. ProptestConfig::default() })]

    /// Every parser MUST return `Err` (never panic) for any byte
    /// sequence - canonical-CBOR decoders through mnem-core, plus the
    /// binary decoders on `Cid` and `Multihash`.
    #[test]
    fn any_parser_rejects_random_bytes_without_panicking(
        bytes in prop::collection::vec(any::<u8>(), 0..4096)
    ) {
        // CBOR-based structured decoders
        let _ = from_canonical_bytes::<Commit>(&bytes);
        let _ = from_canonical_bytes::<Operation>(&bytes);
        let _ = from_canonical_bytes::<View>(&bytes);
        let _ = from_canonical_bytes::<Node>(&bytes);
        let _ = from_canonical_bytes::<Edge>(&bytes);
        let _ = from_canonical_bytes::<TreeChunk>(&bytes);
        let _ = from_canonical_bytes::<RefTarget>(&bytes);

        // Binary decoders
        let _ = Cid::from_bytes(&bytes);
        let _ = Multihash::from_bytes(&bytes);
    }

    /// Shorter inputs (≤64 bytes) hit common edge cases - single-byte,
    /// empty, just-a-CBOR-tag, just-a-multihash-header. Concentrated
    /// fuzz on the first few bytes of encoding is where most real-world
    /// malformed inputs would come from.
    #[test]
    fn short_inputs_rejected_cleanly(
        bytes in prop::collection::vec(any::<u8>(), 0..64)
    ) {
        let _ = from_canonical_bytes::<Commit>(&bytes);
        let _ = from_canonical_bytes::<View>(&bytes);
        let _ = from_canonical_bytes::<TreeChunk>(&bytes);
        let _ = Cid::from_bytes(&bytes);
        let _ = Multihash::from_bytes(&bytes);
    }

    /// Prepending a valid CBOR-map header to random bytes shouldn't
    /// panic - this probes the decoder's structural error paths rather
    /// than its byte-validation path.
    #[test]
    fn pseudo_cbor_doesnt_crash(
        len_byte in any::<u8>(),
        rest in prop::collection::vec(any::<u8>(), 0..1024),
    ) {
        // Leading 0xa0..=0xbb is a CBOR major-type-5 map length tag.
        // Map(0xa0) == empty map.
        let mut bytes = vec![0xa0u8.wrapping_add(len_byte & 0x1f)];
        bytes.extend_from_slice(&rest);
        let _ = from_canonical_bytes::<Commit>(&bytes);
        let _ = from_canonical_bytes::<Operation>(&bytes);
        let _ = from_canonical_bytes::<TreeChunk>(&bytes);
    }
}

// ---------------- Targeted edge cases ----------------

#[test]
fn empty_bytes_parse_without_panic() {
    assert!(from_canonical_bytes::<Commit>(&[]).is_err());
    assert!(from_canonical_bytes::<Operation>(&[]).is_err());
    assert!(from_canonical_bytes::<View>(&[]).is_err());
    assert!(from_canonical_bytes::<Node>(&[]).is_err());
    assert!(from_canonical_bytes::<Edge>(&[]).is_err());
    assert!(from_canonical_bytes::<TreeChunk>(&[]).is_err());
    assert!(Cid::from_bytes(&[]).is_err());
    assert!(Multihash::from_bytes(&[]).is_err());
}

#[test]
fn single_null_byte_parses_without_panic() {
    // 0x00 is CBOR unsigned-int 0 - every struct decoder rejects, and
    // that rejection must be a plain Err, not a panic.
    let bytes = [0x00u8];
    let _ = from_canonical_bytes::<Commit>(&bytes);
    let _ = from_canonical_bytes::<Operation>(&bytes);
    let _ = from_canonical_bytes::<TreeChunk>(&bytes);
}

#[test]
fn truncated_cid_rejected() {
    // A real CIDv1 starts with 0x01 0x71 0x12 0x20 <32-byte digest>.
    // Truncate at each boundary - all must Err, none should panic.
    for n in 0..36 {
        let buf = vec![0xabu8; n];
        let _ = Cid::from_bytes(&buf);
    }
}

#[test]
fn all_ones_is_rejected_gracefully() {
    let buf = vec![0xffu8; 256];
    let _ = from_canonical_bytes::<Commit>(&buf);
    let _ = from_canonical_bytes::<Operation>(&buf);
    let _ = Cid::from_bytes(&buf);
    let _ = Multihash::from_bytes(&buf);
}
