//! Rolling-hash chunker for Prolly trees.
//!
//! See [`crate::prolly`] for the module overview and SPEC §5.2 for the
//! normative algorithm.
//!
//! Two entry points:
//!
//! - [`Chunker`] - the low-level state machine. Push [`ProllyKey`] values
//!   one at a time via [`Chunker::push_key`]; read the boundary decision
//!   via [`Chunker::should_split_at`]; [`Chunker::reset`] between chunks.
//! - [`chunk_boundaries`] - convenience wrapper over a `&[ProllyKey]` that
//!   returns boundary indices directly.
//!
//! Both are purely computational - no I/O, no allocations beyond a small
//! internal buffer.

use blake3;

use crate::prolly::constants::{
    MAX_ENTRIES_PER_CHUNK, MIN_ENTRIES_PER_CHUNK, PROLLY_KEY_BYTES, ProllyKey, ROLLING_KEY,
    ROLLING_WINDOW_BYTES, THRESHOLD,
};

/// Rolling-hash state for a single chunk.
///
/// One instance spans exactly one chunk: push keys, check the boundary,
/// and [`reset`](Self::reset) before starting the next chunk.
#[derive(Clone, Copy, Debug)]
pub struct Chunker {
    /// Last 64 bytes of the concatenation of keys in order (SPEC §5.2).
    ///
    /// Initialized to zeros; each new key (16 bytes) is appended at the
    /// right (bytes `48..64`) and the previous contents shift left by
    /// 16 bytes. The left side remains zero-padded for the first three
    /// keys.
    window: [u8; ROLLING_WINDOW_BYTES],
    /// Count of keys pushed since the last [`reset`](Self::reset). Also
    /// the number of entries in the current chunk.
    n_keys: usize,
}

impl Default for Chunker {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunker {
    /// Fresh chunker, window all zeros, counter at 0.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            window: [0u8; ROLLING_WINDOW_BYTES],
            n_keys: 0,
        }
    }

    /// Number of keys pushed since the last reset.
    #[must_use]
    pub const fn n_keys(&self) -> usize {
        self.n_keys
    }

    /// Reset the state. Call this after emitting a chunk boundary.
    pub const fn reset(&mut self) {
        self.window = [0u8; ROLLING_WINDOW_BYTES];
        self.n_keys = 0;
    }

    /// Push a [`ProllyKey`] into the rolling window.
    ///
    /// Increments [`Self::n_keys`] by one. The window shifts left by
    /// `PROLLY_KEY_BYTES` bytes and the new key lands at the right.
    pub fn push_key(&mut self, key: &ProllyKey) {
        // Shift left by 16 bytes: [a, b, c, d] -> [b, c, d, _].
        self.window
            .copy_within(PROLLY_KEY_BYTES..ROLLING_WINDOW_BYTES, 0);
        // Place the new key at the right.
        self.window[ROLLING_WINDOW_BYTES - PROLLY_KEY_BYTES..].copy_from_slice(key.as_bytes());
        self.n_keys += 1;
    }

    /// Compute the 64-bit rolling hash of the current window.
    ///
    /// Computed as `u64::from_le_bytes(BLAKE3::keyed_hash(ROLLING_KEY, window)[0..8])`
    /// per SPEC §5.2.
    #[must_use]
    pub fn rolling_hash(&self) -> u64 {
        let out = blake3::keyed_hash(&ROLLING_KEY, &self.window);
        let mut u64_bytes = [0u8; 8];
        u64_bytes.copy_from_slice(&out.as_bytes()[0..8]);
        u64::from_le_bytes(u64_bytes)
    }

    /// Decide whether a boundary should be emitted at the current
    /// position, given the number of entries accumulated in the chunk
    /// so far (including the just-pushed key).
    #[must_use]
    pub fn should_split_at(&self, entries_in_chunk: usize) -> bool {
        if entries_in_chunk < MIN_ENTRIES_PER_CHUNK {
            return false;
        }
        if entries_in_chunk >= MAX_ENTRIES_PER_CHUNK {
            return true;
        }
        self.rolling_hash() < THRESHOLD
    }
}

/// Compute chunk-boundary positions for a sorted sequence of keys.
///
/// Returns a vector of 1-based split indices into `keys`. Chunks are
/// `keys[0..i₀]`, `keys[i₀..i₁]`, …, `keys[iₙ..]`. The final (implicit)
/// chunk's terminal index is `keys.len()`, not appended to the result.
#[must_use]
pub fn chunk_boundaries(keys: &[ProllyKey]) -> Vec<usize> {
    let mut chunker = Chunker::new();
    let mut boundaries = Vec::new();
    let mut entries_in_chunk: usize = 0;

    for (i, key) in keys.iter().enumerate() {
        chunker.push_key(key);
        entries_in_chunk += 1;

        if chunker.should_split_at(entries_in_chunk) {
            boundaries.push(i + 1);
            chunker.reset();
            entries_in_chunk = 0;
        }
    }

    boundaries
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_keys(n: u32) -> Vec<ProllyKey> {
        (0..n)
            .map(|i| {
                let mut k = [0u8; PROLLY_KEY_BYTES];
                k[PROLLY_KEY_BYTES - 4..].copy_from_slice(&i.to_be_bytes());
                ProllyKey(k)
            })
            .collect()
    }

    #[test]
    fn push_shifts_window_and_counts_keys() {
        let mut c = Chunker::new();
        assert_eq!(c.n_keys(), 0);
        assert_eq!(c.window, [0u8; 64]);

        let k1 = ProllyKey([1u8; PROLLY_KEY_BYTES]);
        c.push_key(&k1);
        assert_eq!(c.n_keys(), 1);
        assert_eq!(&c.window[0..48], &[0u8; 48]);
        assert_eq!(&c.window[48..64], k1.as_bytes());

        let k2 = ProllyKey([2u8; PROLLY_KEY_BYTES]);
        c.push_key(&k2);
        assert_eq!(c.n_keys(), 2);
        assert_eq!(&c.window[0..32], &[0u8; 32]);
        assert_eq!(&c.window[32..48], k1.as_bytes());
        assert_eq!(&c.window[48..64], k2.as_bytes());

        let k3 = ProllyKey([3u8; PROLLY_KEY_BYTES]);
        let k4 = ProllyKey([4u8; PROLLY_KEY_BYTES]);
        let k5 = ProllyKey([5u8; PROLLY_KEY_BYTES]);
        c.push_key(&k3);
        c.push_key(&k4);
        c.push_key(&k5);
        assert_eq!(&c.window[0..16], k2.as_bytes());
        assert_eq!(&c.window[16..32], k3.as_bytes());
        assert_eq!(&c.window[32..48], k4.as_bytes());
        assert_eq!(&c.window[48..64], k5.as_bytes());
    }

    #[test]
    fn reset_clears_state() {
        let mut c = Chunker::new();
        c.push_key(&ProllyKey([7u8; PROLLY_KEY_BYTES]));
        assert_ne!(c.n_keys(), 0);
        c.reset();
        assert_eq!(c.n_keys(), 0);
        assert_eq!(c.window, [0u8; 64]);
    }

    #[test]
    fn rolling_hash_is_deterministic() {
        let mut c1 = Chunker::new();
        let mut c2 = Chunker::new();
        for i in 1..=3u8 {
            let k = ProllyKey([i; PROLLY_KEY_BYTES]);
            c1.push_key(&k);
            c2.push_key(&k);
        }
        assert_eq!(c1.rolling_hash(), c2.rolling_hash());
    }

    #[test]
    fn should_split_respects_hard_min() {
        let mut c = Chunker::new();
        for i in 0..MIN_ENTRIES_PER_CHUNK {
            c.push_key(&ProllyKey([(i as u8) % 255; PROLLY_KEY_BYTES]));
            assert!(
                !c.should_split_at(c.n_keys()),
                "must not split below min at n={i}"
            );
        }
    }

    #[test]
    fn should_split_respects_hard_max() {
        let mut c = Chunker::new();
        for _ in 0..MAX_ENTRIES_PER_CHUNK {
            c.push_key(&ProllyKey([0u8; PROLLY_KEY_BYTES]));
        }
        assert!(
            c.should_split_at(MAX_ENTRIES_PER_CHUNK),
            "must split at hard max"
        );
    }

    #[test]
    fn chunk_boundaries_are_deterministic_across_runs() {
        let keys = sorted_keys(4_096);
        let b1 = chunk_boundaries(&keys);
        let b2 = chunk_boundaries(&keys);
        assert_eq!(b1, b2);
    }

    #[test]
    fn chunk_boundaries_respect_min_and_max() {
        let keys = sorted_keys(4_096);
        let boundaries = chunk_boundaries(&keys);
        let mut start = 0usize;
        for &end in &boundaries {
            let size = end - start;
            assert!(
                size >= MIN_ENTRIES_PER_CHUNK,
                "chunk size {size} < MIN {MIN_ENTRIES_PER_CHUNK}"
            );
            assert!(
                size <= MAX_ENTRIES_PER_CHUNK,
                "chunk size {size} > MAX {MAX_ENTRIES_PER_CHUNK}"
            );
            start = end;
        }
    }

    #[test]
    fn chunk_boundaries_average_near_target() {
        let keys = sorted_keys(32_768);
        let mut boundaries = chunk_boundaries(&keys);
        boundaries.push(keys.len());
        let mut total: usize = 0;
        let mut prev: usize = 0;
        let mut count: usize = 0;
        for &b in &boundaries {
            let size = b - prev;
            if size >= MIN_ENTRIES_PER_CHUNK {
                total += size;
                count += 1;
            }
            prev = b;
        }
        assert!(count > 0);
        let mean = total as f64 / count as f64;
        let target = super::super::constants::TARGET_AVG_ENTRIES_PER_CHUNK as f64;
        let ratio = mean / target;
        assert!(
            (0.7..=1.3).contains(&ratio),
            "mean chunk size {mean} outside ±30% of target {target} (ratio {ratio})"
        );
    }
}
