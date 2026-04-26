//! Smoke-test the Prolly chunker on a synthetic sorted key sequence.
//!
//! Run:
//!
//! ```console
//! cargo run --example prolly_chunker
//! ```
//!
//! Operator redirects output to `/tmp/mnem-test/prolly_chunker.out`.

use mnem_core::prolly::{
    MAX_ENTRIES_PER_CHUNK, MIN_ENTRIES_PER_CHUNK, ProllyKey, chunk_boundaries,
};

fn synthesize(n: u32) -> Vec<ProllyKey> {
    (0..n)
        .map(|i| {
            let mut k = [0u8; 16];
            k[12..16].copy_from_slice(&i.to_be_bytes());
            ProllyKey(k)
        })
        .collect()
}

fn main() {
    println!("# mnem prolly chunker smoke test");
    println!("# mnem-core: {}", mnem_core::VERSION);
    println!("# mnem format: {}", mnem_core::FORMAT_VERSION);
    println!();

    for &n in &[100u32, 1_000, 10_000, 100_000] {
        let keys = synthesize(n);
        let boundaries = chunk_boundaries(&keys);
        let n_chunks = boundaries.len() + 1;

        let mut ends = boundaries.clone();
        ends.push(keys.len());
        let mut prev = 0usize;
        let mut sum_size = 0usize;
        let mut max_size = 0usize;
        let mut min_size = usize::MAX;
        for &e in &ends {
            let sz = e - prev;
            sum_size += sz;
            max_size = max_size.max(sz);
            min_size = min_size.min(sz);
            prev = e;
        }
        let avg = sum_size as f64 / ends.len() as f64;

        println!(
            "keys={n:>6}  chunks={n_chunks:>4}  min_size={min_size:>3}  avg_size={avg:>5.1}  max_size={max_size:>3}",
        );
    }

    let keys = synthesize(10_000);
    let b1 = chunk_boundaries(&keys);
    let b2 = chunk_boundaries(&keys);
    assert_eq!(b1, b2, "chunker not deterministic across runs");
    println!();
    println!("determinism across two runs of 10k keys: ok");
    println!("bounds: min={MIN_ENTRIES_PER_CHUNK}, max={MAX_ENTRIES_PER_CHUNK}");

    println!();
    println!("# chunker smoke test passed.");
}
