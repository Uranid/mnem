//! On-wire Prolly tree chunks (leaf and internal) + streaming builder.
//!
//! Per SPEC §4.3, a Prolly tree is a Merkle DAG of [`TreeChunk`] objects.
//! A leaf carries a sorted `(ProllyKey, Cid)` list; an internal node
//! carries separator keys and child-chunk CIDs. Both share the `_kind =
//! "tree"` discriminator; the `leaf: bool` field picks the variant.
//!
//! The builder in this module consumes a **sorted** iterator of
//! `(key, value)` pairs, writes every chunk it emits to a
//! [`Blockstore`], and returns the root chunk's CID. M5.2 covers
//! construction; lookup, cursor, and diff land in M5.3.

use std::collections::BTreeMap;

use ipld_core::ipld::Ipld;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::codec::{from_canonical_bytes, hash_to_cid};
use crate::error::Error;
use crate::id::Cid;
use crate::prolly::chunker::Chunker;
use crate::prolly::constants::ProllyKey;
use crate::store::Blockstore;

// ---------------- Shapes ----------------

/// On-wire content of a Prolly leaf chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Leaf {
    /// Sorted (key, value) entries.
    pub entries: Vec<(ProllyKey, Cid)>,
    /// Forward-compat extension map per SPEC §3.2.
    pub extra: BTreeMap<String, Ipld>,
}

impl Leaf {
    /// Construct a leaf with no extension fields.
    #[must_use]
    pub const fn new(entries: Vec<(ProllyKey, Cid)>) -> Self {
        Self {
            entries,
            extra: BTreeMap::new(),
        }
    }
}

/// On-wire content of a Prolly internal (non-leaf) chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Internal {
    /// `children.len() - 1` separator keys, strictly ascending.
    /// `boundaries[i]` is the inclusive lower bound of child `i + 1`.
    pub boundaries: Vec<ProllyKey>,
    /// `boundaries.len() + 1` child-chunk CIDs in order.
    pub children: Vec<Cid>,
    /// Forward-compat extension map per SPEC §3.2.
    pub extra: BTreeMap<String, Ipld>,
}

impl Internal {
    /// Construct an internal chunk with no extension fields.
    ///
    /// # Panics
    ///
    /// In debug builds: `boundaries.len() + 1 != children.len()`.
    #[must_use]
    pub fn new(boundaries: Vec<ProllyKey>, children: Vec<Cid>) -> Self {
        debug_assert_eq!(boundaries.len() + 1, children.len());
        Self {
            boundaries,
            children,
            extra: BTreeMap::new(),
        }
    }
}

/// A Prolly tree chunk on the wire - either a leaf or an internal node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TreeChunk {
    /// A leaf chunk carrying `(key, value)` entries.
    Leaf(Leaf),
    /// An internal chunk carrying separator keys and child pointers.
    Internal(Internal),
}

impl TreeChunk {
    /// The `_kind` discriminator on the wire. `"tree"` for both variants.
    pub const KIND: &'static str = "tree";

    /// `true` iff this is a leaf chunk.
    #[must_use]
    pub const fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf(_))
    }

    /// Number of entries (leaf) or children (internal).
    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::Leaf(l) => l.entries.len(),
            Self::Internal(i) => i.children.len(),
        }
    }

    /// Whether the chunk carries zero entries/children.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------- Serde ----------------

#[derive(Serialize, Deserialize)]
struct TreeWire {
    #[serde(rename = "_kind")]
    kind: String,
    leaf: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    entries: Option<Vec<(ProllyKey, Cid)>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    boundaries: Option<Vec<ProllyKey>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    children: Option<Vec<Cid>>,
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    extra: BTreeMap<String, Ipld>,
}

impl Serialize for TreeChunk {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let wire = match self {
            Self::Leaf(l) => TreeWire {
                kind: Self::KIND.into(),
                leaf: true,
                entries: Some(l.entries.clone()),
                boundaries: None,
                children: None,
                extra: l.extra.clone(),
            },
            Self::Internal(i) => TreeWire {
                kind: Self::KIND.into(),
                leaf: false,
                entries: None,
                boundaries: Some(i.boundaries.clone()),
                children: Some(i.children.clone()),
                extra: i.extra.clone(),
            },
        };
        wire.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TreeChunk {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;
        let w = TreeWire::deserialize(deserializer)?;
        if w.kind != Self::KIND {
            return Err(D::Error::custom(format!(
                "expected _kind='tree', got '{}'",
                w.kind
            )));
        }
        if w.leaf {
            let entries = w
                .entries
                .ok_or_else(|| D::Error::custom("leaf chunk missing 'entries'"))?;
            if w.boundaries.is_some() || w.children.is_some() {
                return Err(D::Error::custom(
                    "leaf chunk must not carry 'boundaries' or 'children'",
                ));
            }
            Ok(Self::Leaf(Leaf {
                entries,
                extra: w.extra,
            }))
        } else {
            let boundaries = w
                .boundaries
                .ok_or_else(|| D::Error::custom("internal chunk missing 'boundaries'"))?;
            let children = w
                .children
                .ok_or_else(|| D::Error::custom("internal chunk missing 'children'"))?;
            if w.entries.is_some() {
                return Err(D::Error::custom("internal chunk must not carry 'entries'"));
            }
            if boundaries.len() + 1 != children.len() {
                return Err(D::Error::custom(format!(
                    "internal chunk: boundaries.len()={} must equal children.len()-1 ({}-1)",
                    boundaries.len(),
                    children.len()
                )));
            }
            Ok(Self::Internal(Internal {
                boundaries,
                children,
                extra: w.extra,
            }))
        }
    }
}

// ---------------- Read helper ----------------

/// Decode a [`TreeChunk`] from canonical DAG-CBOR bytes.
///
/// # Errors
///
/// Returns a codec error if the bytes are malformed or the decoded
/// chunk violates SPEC §4.3 invariants (`_kind != "tree"`, mismatched
/// `boundaries` / `children` lengths, etc.).
pub fn load_chunk(bytes: &[u8]) -> Result<TreeChunk, Error> {
    Ok(from_canonical_bytes(bytes)?)
}

/// Fetch a chunk from a blockstore by CID and decode it.
///
/// Combines [`Blockstore::get`] and [`load_chunk`] with explicit
/// `NotFound` error mapping - the standard "load a tree chunk" pattern.
///
/// # Errors
///
/// Returns [`crate::error::StoreError::NotFound`] if the CID is absent
/// from the store, plus any underlying store or codec error.
pub fn load_tree_chunk<B: Blockstore + ?Sized>(store: &B, cid: &Cid) -> Result<TreeChunk, Error> {
    let bytes = store
        .get(cid)?
        .ok_or_else(|| crate::error::StoreError::NotFound { cid: cid.clone() })?;
    load_chunk(&bytes)
}

// ---------------- Builder ----------------

/// Build a Prolly tree from a sorted iterator of `(key, value)` pairs.
///
/// Writes every chunk to `blockstore` and returns the root chunk's CID.
/// Input MUST be sorted ascending by key; the builder does not sort.
///
/// Empty input produces a single empty leaf chunk whose CID is returned.
///
/// # Errors
///
/// Propagates blockstore and codec errors; otherwise infallible given
/// sorted input.
pub fn build_tree<B, I>(blockstore: &B, entries: I) -> Result<Cid, Error>
where
    B: Blockstore + ?Sized,
    I: IntoIterator<Item = (ProllyKey, Cid)>,
{
    let level_zero = build_leaf_level(blockstore, entries)?;
    let mut current = level_zero;
    while current.len() > 1 {
        current = build_internal_level(blockstore, current)?;
    }
    // `build_leaf_level` always produces at least one chunk (empty-leaf
    // sentinel for empty input), so `current` has exactly 1 element here.
    Ok(current
        .into_iter()
        .next()
        .expect("build_leaf_level emits >= 1 chunk")
        .1)
}

fn build_leaf_level<B, I>(blockstore: &B, entries: I) -> Result<Vec<(ProllyKey, Cid)>, Error>
where
    B: Blockstore + ?Sized,
    I: IntoIterator<Item = (ProllyKey, Cid)>,
{
    let mut chunker = Chunker::new();
    let mut cur: Vec<(ProllyKey, Cid)> = Vec::with_capacity(128);
    let mut out: Vec<(ProllyKey, Cid)> = Vec::new();

    for (k, v) in entries {
        chunker.push_key(&k);
        cur.push((k, v));
        if chunker.should_split_at(cur.len()) {
            emit_leaf(blockstore, &mut cur, &mut out)?;
            chunker.reset();
        }
    }

    if !cur.is_empty() {
        emit_leaf(blockstore, &mut cur, &mut out)?;
    }

    if out.is_empty() {
        // Empty-input sentinel: one empty leaf chunk.
        let chunk = TreeChunk::Leaf(Leaf::new(Vec::new()));
        let (bytes, cid) = hash_to_cid(&chunk)?;
        // safety: cid computed above via hash_to_cid
        blockstore.put_trusted(cid.clone(), bytes)?;
        out.push((ProllyKey::default(), cid));
    }

    Ok(out)
}

fn emit_leaf<B: Blockstore + ?Sized>(
    blockstore: &B,
    cur: &mut Vec<(ProllyKey, Cid)>,
    out: &mut Vec<(ProllyKey, Cid)>,
) -> Result<(), Error> {
    let first = cur[0].0;
    let chunk = TreeChunk::Leaf(Leaf::new(std::mem::take(cur)));
    let (bytes, cid) = hash_to_cid(&chunk)?;
    // safety: cid computed above via hash_to_cid
    blockstore.put_trusted(cid.clone(), bytes)?;
    out.push((first, cid));
    Ok(())
}

fn build_internal_level<B: Blockstore + ?Sized>(
    blockstore: &B,
    children: Vec<(ProllyKey, Cid)>,
) -> Result<Vec<(ProllyKey, Cid)>, Error> {
    let mut chunker = Chunker::new();
    let mut out: Vec<(ProllyKey, Cid)> = Vec::new();
    let mut cur_children: Vec<Cid> = Vec::new();
    let mut cur_boundaries: Vec<ProllyKey> = Vec::new();
    let mut cur_first: Option<ProllyKey> = None;

    for (first_key, child_cid) in children {
        chunker.push_key(&first_key);
        if cur_first.is_none() {
            cur_first = Some(first_key);
        } else {
            // Separator between the previous child and this one.
            cur_boundaries.push(first_key);
        }
        cur_children.push(child_cid);

        if chunker.should_split_at(cur_children.len()) {
            emit_internal(
                blockstore,
                &mut cur_boundaries,
                &mut cur_children,
                &mut cur_first,
                &mut out,
            )?;
            chunker.reset();
        }
    }

    if !cur_children.is_empty() {
        emit_internal(
            blockstore,
            &mut cur_boundaries,
            &mut cur_children,
            &mut cur_first,
            &mut out,
        )?;
    }

    Ok(out)
}

fn emit_internal<B: Blockstore + ?Sized>(
    blockstore: &B,
    cur_boundaries: &mut Vec<ProllyKey>,
    cur_children: &mut Vec<Cid>,
    cur_first: &mut Option<ProllyKey>,
    out: &mut Vec<(ProllyKey, Cid)>,
) -> Result<(), Error> {
    debug_assert_eq!(cur_boundaries.len() + 1, cur_children.len());
    let chunk = TreeChunk::Internal(Internal::new(
        std::mem::take(cur_boundaries),
        std::mem::take(cur_children),
    ));
    let (bytes, cid) = hash_to_cid(&chunk)?;
    // safety: cid computed above via hash_to_cid
    blockstore.put_trusted(cid.clone(), bytes)?;
    out.push((cur_first.take().expect("first key set above"), cid));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{from_canonical_bytes, to_canonical_bytes};
    use crate::id::{CODEC_RAW, Multihash};
    use crate::store::MemoryBlockstore;

    fn dummy_cid(n: u32) -> Cid {
        Cid::new(CODEC_RAW, Multihash::sha2_256(&n.to_be_bytes()))
    }

    fn keyed(i: u32) -> ProllyKey {
        let mut k = [0u8; 16];
        k[12..16].copy_from_slice(&i.to_be_bytes());
        ProllyKey(k)
    }

    #[test]
    fn leaf_round_trip() {
        let leaf = TreeChunk::Leaf(Leaf::new(vec![
            (keyed(1), dummy_cid(1)),
            (keyed(2), dummy_cid(2)),
        ]));
        let bytes = to_canonical_bytes(&leaf).unwrap();
        let decoded: TreeChunk = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(leaf, decoded);
        let bytes2 = to_canonical_bytes(&decoded).unwrap();
        assert_eq!(bytes, bytes2);
    }

    #[test]
    fn internal_round_trip() {
        let internal = TreeChunk::Internal(Internal::new(
            vec![keyed(10), keyed(20)],
            vec![dummy_cid(100), dummy_cid(200), dummy_cid(300)],
        ));
        let bytes = to_canonical_bytes(&internal).unwrap();
        let decoded: TreeChunk = from_canonical_bytes(&bytes).unwrap();
        assert_eq!(internal, decoded);
    }

    #[test]
    fn wrong_kind_rejected() {
        // Craft a tree-shaped wire with wrong _kind.
        let w = TreeWire {
            kind: "node".into(),
            leaf: true,
            entries: Some(vec![]),
            boundaries: None,
            children: None,
            extra: BTreeMap::new(),
        };
        let bytes = serde_ipld_dagcbor::to_vec(&w).unwrap();
        let err = serde_ipld_dagcbor::from_slice::<TreeChunk>(&bytes).unwrap_err();
        assert!(err.to_string().contains("_kind"));
    }

    #[test]
    fn internal_rejects_inconsistent_lengths() {
        let w = TreeWire {
            kind: "tree".into(),
            leaf: false,
            entries: None,
            boundaries: Some(vec![keyed(10), keyed(20)]),
            children: Some(vec![dummy_cid(1)]),
            extra: BTreeMap::new(),
        };
        let bytes = serde_ipld_dagcbor::to_vec(&w).unwrap();
        let err = serde_ipld_dagcbor::from_slice::<TreeChunk>(&bytes).unwrap_err();
        assert!(err.to_string().contains("boundaries.len()"));
    }

    #[test]
    fn build_empty_produces_single_empty_leaf() {
        let store = MemoryBlockstore::new();
        let root = build_tree(&store, std::iter::empty()).unwrap();
        assert_eq!(store.len(), 1);
        let bytes = store.get(&root).unwrap().unwrap();
        let chunk: TreeChunk = from_canonical_bytes(&bytes).unwrap();
        match chunk {
            TreeChunk::Leaf(l) => assert!(l.entries.is_empty()),
            TreeChunk::Internal(_) => panic!("empty tree should be a leaf"),
        }
    }

    #[test]
    fn build_single_entry_is_leaf() {
        let store = MemoryBlockstore::new();
        let root = build_tree(&store, [(keyed(1), dummy_cid(1))]).unwrap();
        let bytes = store.get(&root).unwrap().unwrap();
        let chunk: TreeChunk = from_canonical_bytes(&bytes).unwrap();
        assert!(chunk.is_leaf());
        assert_eq!(chunk.len(), 1);
    }

    #[test]
    fn build_many_entries_produces_internal_root() {
        let store = MemoryBlockstore::new();
        let entries: Vec<_> = (0..1000u32).map(|i| (keyed(i), dummy_cid(i))).collect();
        let root = build_tree(&store, entries).unwrap();
        let bytes = store.get(&root).unwrap().unwrap();
        let chunk: TreeChunk = from_canonical_bytes(&bytes).unwrap();
        assert!(
            !chunk.is_leaf(),
            "1000 entries should require an internal root"
        );
    }

    #[test]
    fn build_is_deterministic_across_builds() {
        let entries: Vec<_> = (0..1_000u32).map(|i| (keyed(i), dummy_cid(i))).collect();
        let s1 = MemoryBlockstore::new();
        let r1 = build_tree(&s1, entries.clone()).unwrap();
        let s2 = MemoryBlockstore::new();
        let r2 = build_tree(&s2, entries).unwrap();
        assert_eq!(r1, r2);
        assert_eq!(s1.len(), s2.len(), "chunk count must match");
    }

    #[test]
    fn internal_children_are_reachable_from_root() {
        let store = MemoryBlockstore::new();
        let entries: Vec<_> = (0..500u32).map(|i| (keyed(i), dummy_cid(i))).collect();
        let root = build_tree(&store, entries).unwrap();

        // Walk: root must exist; every child must exist.
        fn walk(cid: &Cid, store: &MemoryBlockstore, leaves: &mut usize, internals: &mut usize) {
            let bytes = store.get(cid).unwrap().unwrap();
            let chunk: TreeChunk = from_canonical_bytes(&bytes).unwrap();
            match chunk {
                TreeChunk::Leaf(_) => *leaves += 1,
                TreeChunk::Internal(i) => {
                    *internals += 1;
                    for c in &i.children {
                        walk(c, store, leaves, internals);
                    }
                }
            }
        }

        let mut leaves = 0;
        let mut internals = 0;
        walk(&root, &store, &mut leaves, &mut internals);
        assert!(leaves >= 1);
        assert!(
            internals >= 1,
            "500 entries should yield at least one internal chunk"
        );
        assert_eq!(leaves + internals, store.len(), "no unreachable chunks");
    }
}
