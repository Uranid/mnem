//! End-to-end Prolly tree builder smoke test.
//!
//! Builds trees of varying sizes into a [`MemoryBlockstore`], walks the
//! resulting DAG, and checks that content-addressing gives a stable root
//! across builds of the same input.
//!
//! Run:
//!
//! ```console
//! cargo run --example prolly_tree
//! ```
//!
//! Operator redirects output to `/tmp/mnem-test/prolly_tree.out`.

use mnem_core::codec::from_canonical_bytes;
use mnem_core::id::{CODEC_RAW, Cid, Multihash};
use mnem_core::prolly::{ProllyKey, TreeChunk, build_tree};
use mnem_core::store::{Blockstore, MemoryBlockstore};

fn keyed(i: u32) -> ProllyKey {
    let mut k = [0u8; 16];
    k[12..16].copy_from_slice(&i.to_be_bytes());
    ProllyKey(k)
}

fn dummy_value(i: u32) -> Cid {
    Cid::new(CODEC_RAW, Multihash::sha2_256(&i.to_be_bytes()))
}

fn walk(cid: &Cid, store: &MemoryBlockstore) -> (usize, usize, usize) {
    // Returns (leaves, internals, depth)
    let bytes = store.get(cid).unwrap().unwrap();
    let chunk: TreeChunk = from_canonical_bytes(&bytes).unwrap();
    match chunk {
        TreeChunk::Leaf(_) => (1, 0, 1),
        TreeChunk::Internal(i) => {
            let mut leaves = 0;
            let mut internals = 1;
            let mut depth = 0;
            for child in &i.children {
                let (l, it, d) = walk(child, store);
                leaves += l;
                internals += it;
                depth = depth.max(d);
            }
            (leaves, internals, depth + 1)
        }
    }
}

fn main() {
    println!("# mnem Prolly tree builder smoke test");
    println!("# mnem-core: {}", mnem_core::VERSION);
    println!();

    for &n in &[1u32, 17, 100, 1_000, 10_000] {
        let store = MemoryBlockstore::new();
        let entries: Vec<_> = (0..n).map(|i| (keyed(i), dummy_value(i))).collect();
        let root = build_tree(&store, entries).unwrap();
        let (leaves, internals, depth) = walk(&root, &store);
        println!(
            "n_entries={:>6}  chunks_total={:>4}  leaves={:>4}  internals={:>3}  depth={}  root={}",
            n,
            store.len(),
            leaves,
            internals,
            depth,
            root
        );
    }

    // Determinism: two independent builds of the same input produce the same root CID.
    let entries: Vec<_> = (0..10_000u32).map(|i| (keyed(i), dummy_value(i))).collect();
    let s1 = MemoryBlockstore::new();
    let r1 = build_tree(&s1, entries.clone()).unwrap();
    let s2 = MemoryBlockstore::new();
    let r2 = build_tree(&s2, entries).unwrap();
    assert_eq!(r1, r2, "root CID not deterministic");
    assert_eq!(s1.len(), s2.len(), "chunk count not deterministic");
    println!();
    println!(
        "determinism: ok  (two independent builds of 10k entries → same root {r1}, same chunk count {})",
        s1.len()
    );

    println!();
    println!("# all prolly-tree smoke tests passed.");
}
