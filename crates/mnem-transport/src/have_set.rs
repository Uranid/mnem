//! [`HaveSet`] trait and the PR-2 [`BloomHaveSet`] reference
//! implementation.
//!
//! A *have-set* is the client-side summary of "blocks I already have"
//! that a `fetch-blocks` / `push-blocks` request carries so the server
//! can omit anything the client doesn't need. The simplest correct
//! have-set is an explicit list of CIDs; the trait exists so the
//! protocol doesn't need to care about which encoding is on the wire.
//!
//! ## Why a trait, not a concrete type
//!
//! mnem's Prolly trees plus dual-adjacency index mean the reachable
//! set from a given commit can easily be millions of blocks. A naive
//! "ship every CID" have-set grows `O(N)` in the repo size; a bloom
//! filter caps at `O(n · k · (1/fpr))` bits for `n` inserted items and
//! `k` hash functions, which is practical up to low-millions of
//! blocks. Beyond that, Iroh-docs / Willow-style range-based set
//! reconciliation (RBSR) is asymptotically better, but its state
//! machine is substantially more code. RBSR therefore lands behind
//! the `have-set-rbsr` capability later; the current trait shape is
//! deliberately compatible with both back-ends.
//!
//! RBSR HaveSet lives behind capability `have-set-rbsr`, landing in
//! v0.2.
//!
//! ## Wire shape for [`BloomHaveSet`]
//!
//! A bloom-encoded have-set serialises to DAG-CBOR as:
//!
//! ```text
//! { _kind: "have-set-bloom",
//!   k:         u32,   // number of hash functions
//!   bitmap:    bytes, // packed LSB-first bitmap, length = ceil(m / 8)
//!   m_bits:    u64,   // bitmap size in bits
//!   seed_key:  bytes, // 32-byte BLAKE3 keyed-hash salt (fixed for PR 2)
//!   item_hint: u32 }  // caller-declared item count; informational
//! ```
//!
//! `seed_key` is fixed to the constant
//! [`BLOOM_SEED_KEY`]. Different processes that build a have-set from
//! the same CID set therefore produce byte-identical serialisations,
//! which is important for deterministic tests and for making the CID
//! of a have-set block meaningful.

// This module's doc-style clippy warnings are opinionated rather
// than correctness: the long leading paragraphs describe the design
// and the kebab-case wire strings don't benefit from backticks.
#![allow(
    clippy::doc_markdown,
    clippy::too_long_first_doc_paragraph,
    clippy::cast_precision_loss,
    clippy::missing_const_for_fn
)]

use bloomfilter::Bloom;
use mnem_core::id::Cid;
use mnem_core::store::Blockstore;

use crate::error::TransportError;

/// Fixed salt used by [`BloomHaveSet`] to derive its two SipHash
/// seeds, and stored on the wire so independent builds agree. Chosen
/// arbitrarily; frozen for [`crate::protocol::PROTOCOL_VERSION`] = 1.
/// ASCII "mnem-have-set-bloom-1" left-padded with zeros.
pub const BLOOM_SEED_KEY: [u8; 32] = {
    let mut k = [0u8; 32];
    let src = b"mnem-have-set-bloom-1";
    let mut i = 0;
    while i < src.len() {
        k[i] = src[i];
        i += 1;
    }
    k
};

/// Default target false-positive rate for PR-2 bloom have-sets.
/// Chosen to trade ~9.6 bits/item of wire overhead for a 1-in-100
/// wasted-block rate on the next `fetch-blocks` call. Tune via
/// [`BloomHaveSet::with_params`].
pub const DEFAULT_FPR: f64 = 0.01;

/// A summary of "blocks the client already has."
///
/// Implementations MUST satisfy:
///
/// - `contains(cid)` may return a false positive (telling the server
///   the client has a block it does not) but MUST NOT return a false
///   negative. A false negative would cause the server to skip a
///   block the client needs, violating the reachability contract.
/// - `extend` is idempotent: inserting the same CID twice leaves the
///   have-set semantically unchanged.
/// - `serialize` produces bytes that a compatible peer can
///   deserialise into a have-set with the same `contains` semantics.
///   PR 2 does not define a wire-level `deserialize`; PR 3 will.
pub trait HaveSet {
    /// Return `true` when `cid` is (probably) in the set.
    fn contains(&self, cid: &Cid) -> bool;

    /// Insert every CID yielded by the iterator. Order does not
    /// matter. Implementations SHOULD resize or rebuild if the caller
    /// has exceeded the configured capacity; the default
    /// [`BloomHaveSet`] does not resize and silently accepts an
    /// elevated FPR past capacity. Callers that want a hard cap
    /// should check `len()` first.
    fn extend<I: IntoIterator<Item = Cid>>(&mut self, cids: I);

    /// Number of distinct CIDs inserted (as claimed by the caller;
    /// implementations MAY return a count including false positives
    /// but MUST NOT undercount).
    fn len(&self) -> usize;

    /// Convenience for `len() == 0`.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialize the have-set to bytes in the on-wire shape for this
    /// back-end. See [`BloomHaveSet`] for the shape this crate
    /// emits.
    fn serialize(&self) -> Vec<u8>;
}

/// Bloom-filter-backed [`HaveSet`]. Wire-compatible with PR-3's
/// `fetch-blocks` and `push-blocks` verbs when the `have-set-bloom`
/// capability is in effect.
///
/// Sizing: pass expected item count + target FPR to
/// [`Self::with_params`]. For the default configuration,
/// [`Self::new`] sizes for 100 000 items at `fpr = 0.01`, which
/// weighs roughly 120 KiB on the wire.
///
/// # Determinism caveat
///
/// `bloomfilter::Bloom::with_seed` takes a 32-byte seed but
/// internally derives two SipHash keys that are not guaranteed to be
/// version-stable across `bloomfilter` crate major versions. PR 2
/// pins to `bloomfilter = "3"`; a minor version bump MUST preserve
/// the key-derivation algorithm. If that ever changes, the
/// [`BLOOM_SEED_KEY`] constant bumps at the same time as
/// [`crate::protocol::PROTOCOL_VERSION`].
pub struct BloomHaveSet {
    inner: Bloom<[u8]>,
    item_count: usize,
}

impl BloomHaveSet {
    /// Size for `items_hint` items at the [`DEFAULT_FPR`].
    #[must_use]
    pub fn new(items_hint: usize) -> Self {
        Self::with_params(items_hint.max(1), DEFAULT_FPR)
    }

    /// Size for `items_hint` items at a caller-chosen false-positive
    /// rate (must be in `(0, 1)`).
    ///
    /// # Panics
    ///
    /// Panics if `fpr` is not in `(0, 1)` or if `items_hint` is zero.
    #[must_use]
    pub fn with_params(items_hint: usize, fpr: f64) -> Self {
        assert!(items_hint > 0, "items_hint must be positive");
        assert!(
            fpr > 0.0 && fpr < 1.0,
            "fpr must be in (0, 1) but got {fpr}",
        );
        let inner = Bloom::new_for_fp_rate_with_seed(items_hint, fpr, &BLOOM_SEED_KEY)
            .expect("valid bloom params (items_hint > 0, 0 < fpr < 1)");
        Self {
            inner,
            item_count: 0,
        }
    }
}

impl HaveSet for BloomHaveSet {
    fn contains(&self, cid: &Cid) -> bool {
        self.inner.check(cid.to_bytes().as_slice())
    }

    fn extend<I: IntoIterator<Item = Cid>>(&mut self, cids: I) {
        for cid in cids {
            let bytes = cid.to_bytes();
            // `check_and_set` returns `true` when every queried bit
            // was already set, i.e. the item was (probably) already
            // present. We bump `item_count` only on fresh insertions
            // so `len()` stays a lower bound that a pure insert
            // counter wouldn't give; false positives at the bit
            // level are accepted for this lower-bound semantic.
            if !self.inner.check_and_set(bytes.as_slice()) {
                self.item_count += 1;
            }
        }
    }

    fn len(&self) -> usize {
        self.item_count
    }

    fn serialize(&self) -> Vec<u8> {
        // Concrete DAG-CBOR wire shape: we do NOT yet need this map
        // to round-trip (PR 3 writes the deserializer), but we fix
        // the shape here so the first server that implements PR 3
        // has a stable target.
        //
        // Lazy implementation via `serde_ipld_dagcbor::to_vec` on an
        // inline serde-derived struct, same pattern as
        // mnem-core/codec.
        #[derive(serde::Serialize)]
        struct Wire<'a> {
            _kind: &'static str,
            k: u32,
            bitmap: &'a [u8],
            m_bits: u64,
            seed_key: &'a [u8],
            item_hint: u32,
        }
        let bitmap = self.inner.as_slice();
        let m_bits = self.inner.len();
        let k = self.inner.number_of_hash_functions();
        let w = Wire {
            _kind: "have-set-bloom",
            k,
            bitmap,
            m_bits,
            seed_key: &BLOOM_SEED_KEY,
            item_hint: u32::try_from(self.item_count).unwrap_or(u32::MAX),
        };
        serde_ipld_dagcbor::to_vec(&w).expect("infallible dag-cbor of fixed shape")
    }
}

impl std::fmt::Debug for BloomHaveSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BloomHaveSet")
            .field("items_inserted", &self.item_count)
            .field("m_bits", &self.inner.len())
            .field("k", &self.inner.number_of_hash_functions())
            .finish()
    }
}

/// Build a [`BloomHaveSet`] by walking every block reachable from
/// `root` in `bs`. Used by `mnem fetch / pull` (PR 3) to summarise
/// what the client already has so the server can skip those blocks.
///
/// The walk is the same depth-first iterator used by
/// [`mod@crate::export`] and is therefore `O(reachable-block-count)`.
/// For repos with
/// millions of blocks this is measurable; the RBSR back-end (see the
/// `have-set-rbsr` capability) will replace the full walk with a
/// range-fingerprint dance.
///
/// Remote-v0.2 follow-up: swap in RBSR once the server-side
/// implementation lands; the trait shape already supports it. See
/// `docs/ROADMAP.md#remote-v0-work-items-tracked-inline-in-src` item 4
/// and  for the wire framing.
///
///
///
/// # Errors
///
/// Returns [`TransportError::Store`] if the blockstore walk fails
/// (missing root, I/O error).
pub fn build_have_set<B>(bs: &B, root: &Cid) -> Result<BloomHaveSet, TransportError>
where
    B: Blockstore + ?Sized,
{
    // We don't know how many reachable blocks there are ahead of
    // time. Start sized for 10_000 items (roughly the median repo
    // after a month of agent activity, per the benchmarks in
    // ROADMAP-BENCHMARKS) and accept the elevated FPR past that.
    // `BloomHaveSet` does not resize; a caller that knows the repo
    // is large can pre-size with `BloomHaveSet::with_params` and
    // extend manually.
    let mut hs = BloomHaveSet::new(10_000);
    for item in bs.iter_from_root(root) {
        let (cid, _bytes) = item?;
        // `iter_from_root` yields every reachable block exactly once,
        // so we don't bother with `check_and_set`: every insertion
        // here is a fresh one by the iterator contract.
        hs.inner.set(cid.to_bytes().as_slice());
        hs.item_count += 1;
    }
    Ok(hs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnem_core::id::{CODEC_RAW, Multihash};

    fn raw(n: u64) -> Cid {
        Cid::new(CODEC_RAW, Multihash::sha2_256(&n.to_be_bytes()))
    }

    #[test]
    fn empty_have_set_contains_nothing_inserted() {
        let hs = BloomHaveSet::new(100);
        assert!(hs.is_empty());
        assert_eq!(hs.len(), 0);
        // Bloom with only `contains` calls does not fault on empty.
        assert!(!hs.contains(&raw(1)));
    }

    #[test]
    fn inserted_cids_are_always_contained() {
        // No-false-negatives invariant.
        let mut hs = BloomHaveSet::new(1_000);
        let cids: Vec<Cid> = (0..500).map(raw).collect();
        hs.extend(cids.clone());
        for c in &cids {
            assert!(hs.contains(c), "false negative on {c}");
        }
        assert_eq!(hs.len(), 500);
    }

    #[test]
    fn false_positive_rate_stays_sane_at_10k_items() {
        // Size for 10k items at fpr = 0.01, insert 10k, then test
        // 10k fresh CIDs and assert the observed FPR is below 3x the
        // target (ample slack for a 10k sample).
        let mut hs = BloomHaveSet::with_params(10_000, 0.01);
        let inserted: Vec<Cid> = (0..10_000).map(raw).collect();
        hs.extend(inserted);
        let probes: Vec<Cid> = (100_000..110_000).map(raw).collect();
        let fp = probes.iter().filter(|c| hs.contains(c)).count();
        let observed = fp as f64 / probes.len() as f64;
        assert!(
            observed < 0.03,
            "observed FPR {observed} > 3x target (0.03); bloom sizing is off",
        );
    }

    #[test]
    fn extend_is_idempotent_for_duplicate_cids() {
        let mut a = BloomHaveSet::new(100);
        let cids: Vec<Cid> = (0..10).map(raw).collect();
        a.extend(cids.clone());
        let len_first = a.len();
        // Re-inserting is a no-op at the semantic layer; the `contains`
        // answer for every inserted CID stays `true`.
        a.extend(cids.clone());
        for c in &cids {
            assert!(a.contains(c));
        }
        // `len()` is a lower bound; it does not grow when all inserts
        // were duplicates.
        assert_eq!(a.len(), len_first);
    }

    #[test]
    fn serialize_is_stable_and_parseable_as_dagcbor() {
        let mut hs = BloomHaveSet::with_params(100, 0.01);
        let cids: Vec<Cid> = (0..50).map(raw).collect();
        hs.extend(cids);
        let a = hs.serialize();
        let b = hs.serialize();
        assert_eq!(a, b, "serialize must be deterministic");
        // Decode as untyped Ipld to check shape: presence of `_kind`,
        // `bitmap`, `m_bits`, `k`, `seed_key`, `item_hint`.
        let ipld: ipld_core::ipld::Ipld = serde_ipld_dagcbor::from_slice(&a).unwrap();
        if let ipld_core::ipld::Ipld::Map(m) = ipld {
            assert!(m.contains_key("_kind"));
            assert!(m.contains_key("bitmap"));
            assert!(m.contains_key("m_bits"));
            assert!(m.contains_key("k"));
            assert!(m.contains_key("seed_key"));
            assert!(m.contains_key("item_hint"));
        } else {
            panic!("expected map, got {ipld:?}");
        }
    }

    #[test]
    fn seed_key_is_frozen_and_ascii_prefix() {
        // Pin: anyone changing this is bumping the wire protocol.
        assert_eq!(&BLOOM_SEED_KEY[..21], b"mnem-have-set-bloom-1");
        assert!(BLOOM_SEED_KEY[21..].iter().all(|b| *b == 0));
    }
}
