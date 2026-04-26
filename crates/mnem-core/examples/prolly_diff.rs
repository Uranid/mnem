//! End-to-end Prolly read-path smoke test: lookup, cursor, diff.
//!
//! Demonstrates the Prolly payoff on a 10,000-entry tree: a diff between
//! two trees that share all but 10 keys touches only the O(d · log n)
//! chunks on the edit paths - proven by counting `Blockstore::get` calls.
//!
//! Run:
//!
//! ```console
//! cargo run --example prolly_diff
//! ```
//!
//! Operator redirects output to `/tmp/mnem-test/prolly_diff.out`.

use std::sync::atomic::{AtomicUsize, Ordering as AO};

use bytes::Bytes;
use mnem_core::error::StoreError;
use mnem_core::id::{CODEC_RAW, Cid, Multihash};
use mnem_core::prolly::{Cursor, DiffEntry, ProllyKey, build_tree, diff, lookup};
use mnem_core::store::{Blockstore, MemoryBlockstore};

/// Tiny wrapper around [`MemoryBlockstore`] that counts reads - useful for
/// showing the Prolly pruning effect in the demo output.
#[derive(Debug, Default)]
struct CountingStore {
    inner: MemoryBlockstore,
    reads: AtomicUsize,
}

impl CountingStore {
    fn new() -> Self {
        Self::default()
    }
    fn read_count(&self) -> usize {
        self.reads.load(AO::SeqCst)
    }
    fn reset_reads(&self) {
        self.reads.store(0, AO::SeqCst);
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl Blockstore for CountingStore {
    fn has(&self, cid: &Cid) -> Result<bool, StoreError> {
        self.inner.has(cid)
    }
    fn get(&self, cid: &Cid) -> Result<Option<Bytes>, StoreError> {
        self.reads.fetch_add(1, AO::SeqCst);
        self.inner.get(cid)
    }
    fn put(&self, cid: Cid, data: Bytes) -> Result<(), StoreError> {
        self.inner.put(cid, data)
    }
    fn delete(&self, cid: &Cid) -> Result<(), StoreError> {
        self.inner.delete(cid)
    }
}

fn keyed(i: u32) -> ProllyKey {
    let mut k = [0u8; 16];
    k[12..16].copy_from_slice(&i.to_be_bytes());
    ProllyKey(k)
}

fn val(i: u32) -> Cid {
    Cid::new(CODEC_RAW, Multihash::sha2_256(&i.to_be_bytes()))
}

fn main() {
    println!("# mnem Prolly read-path smoke test (lookup + cursor + diff)");
    println!("# mnem-core: {}", mnem_core::VERSION);
    println!();

    let store = CountingStore::new();
    let n: u32 = 10_000;

    // --- Build tree A ---
    let entries_a: Vec<_> = (0..n).map(|i| (keyed(i), val(i))).collect();
    let root_a = build_tree(&store, entries_a.clone()).unwrap();
    let chunks_a = store.len();
    println!("built tree A: {n} entries, {chunks_a} chunks, root {root_a}");

    // --- Build tree B: same keys, 10 values changed (every 1000th key) ---
    let mut entries_b = entries_a;
    for i in 0..10 {
        let idx = (i * 1_000) as usize;
        entries_b[idx].1 = val(100_000 + i);
    }
    let root_b = build_tree(&store, entries_b).unwrap();
    let chunks_b_delta = store.len() - chunks_a;
    println!(
        "built tree B: same keys, 10 value edits, +{chunks_b_delta} new chunks, root {root_b}"
    );

    // --- Lookup: hit and miss ---
    store.reset_reads();
    let hit = lookup(&store, &root_a, &keyed(4_242)).unwrap();
    let lookup_reads_hit = store.read_count();
    println!(
        "lookup key=4242 in A: {} reads, result = {}",
        lookup_reads_hit,
        match hit {
            Some(ref c) => format!("Some({c})"),
            None => "None".to_string(),
        }
    );

    store.reset_reads();
    let miss = lookup(&store, &root_a, &keyed(123_456)).unwrap();
    println!(
        "lookup key=123456 in A (absent): {} reads, result = {:?}",
        store.read_count(),
        miss
    );

    // --- Cursor: full iteration ---
    store.reset_reads();
    let cursor = Cursor::new(&store, &root_a).unwrap();
    let all: Vec<_> = cursor.map(|r| r.unwrap()).collect();
    assert_eq!(all.len(), n as usize);
    for w in all.windows(2) {
        assert!(w[0].0 < w[1].0);
    }
    println!(
        "cursor traversed {} entries in sorted order with {} chunk reads",
        all.len(),
        store.read_count()
    );

    // --- Diff payoff: 10 edits in a 10k-entry tree ---
    store.reset_reads();
    let changes = diff(&store, &root_a, &root_b).unwrap();
    let diff_reads = store.read_count();
    assert_eq!(changes.len(), 10);
    for c in &changes {
        assert!(matches!(c, DiffEntry::Changed { .. }));
    }
    println!();
    println!("=== diff payoff ===");
    println!("  tree A has {chunks_a} chunks total.",);
    println!(
        "  diff(A, B) reported {} entries using only {} chunk reads.",
        changes.len(),
        diff_reads
    );
    let naive_reads = chunks_a * 2;
    println!(
        "  (a naive entry-by-entry compare would need ~{} reads = {}x more)",
        naive_reads,
        naive_reads / diff_reads.max(1)
    );

    println!();
    println!("# all prolly read-path smoke tests passed.");
}
