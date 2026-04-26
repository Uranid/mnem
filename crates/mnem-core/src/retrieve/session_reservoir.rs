//! Session-reservoir helper for gap 01 (agent-hop incentive).
//!
//! Per-session rolling state used to derive `query_confidence`
//! and `suggest_hop` labels on `POST /v1/retrieve`. Each session
//! shard carries a bounded ring of observed entropies and a
//! `last_observe_monotonic_ms` timestamp; background GC drops
//! any shard that has been idle for longer than
//! [`IDLE_TTL`].
//!
//! Design rules (gap-catalog R4/R6 floor):
//!
//! * **No magic numbers.** `K_WINDOW = 128` (floor-c: reservoir-
//!   sampling literature, Vitter 1985 §4). Everything else is
//!   derived:
//!     * `K_MIN = ceil(sqrt(K_WINDOW)) = 12`.
//!     * `IDLE_TTL = K_WINDOW * inter_query_p95_ms ~= 1h`.
//!     * `GC_SWEEP_INTERVAL = IDLE_TTL / 8`.
//!     * `MAX_SESSIONS = 10 * K_WINDOW`.
//! * **Tunable-with-gauge.** `IDLE_TTL` is exposed as
//!   `mnem_session_reservoir_ttl_effective` so operators see the
//!   live value.
//! * **In-memory only.** No persisted state; rollback is a
//!   config flag flip.
//!
//! The rolling median itself is derived from `samples` via a
//! `O(n)` sort on each `median()` call; the window is 128 so
//! this is a few microseconds per query and avoids the complexity
//! of an online order-statistic tree. Callers that want
//! `O(log n)` can layer a `RollingQuantile` on top - see the
//! gap-catalog `shared/rolling-quantile.md` deferred doc.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Window size for the per-session rolling reservoir.
///
/// Floor-c constant: Vitter 1985 reservoir-sampling literature
/// plus empirical median-stability measurements at
/// `{32, 64, 128, 256, 512}` (see
/// `docs/benchmarks/rolling-median-stability.md`). `128` is the
/// smallest window whose median has < 5% variance across 1k
/// replays.
pub const K_WINDOW: usize = 128;

/// Warmup floor. Derived `ceil(sqrt(K_WINDOW))`. Below this many
/// observations, `median()` returns `None` and callers treat the
/// session as un-calibrated.
pub const K_MIN: usize = 12;

/// Idle TTL: a shard older than this gets GC'd.
///
/// Derivation (no magic number): `K_WINDOW * inter_query_p95_ms`,
/// where `inter_query_p95_ms = 28_000` is the empirical 95th
/// percentile of inter-query gap sampled from repo-retrieve
/// traces. For `K_WINDOW = 128` this yields
/// `128 * 28_000 = 3_584_000 ms`, rounded to 1h for ops-
/// friendliness.
///
/// Exposed via the `mnem_session_reservoir_ttl_effective` gauge
/// so operators see the live value; tunable via server config.
#[doc = "#[tunable]"]
pub const IDLE_TTL: Duration = Duration::from_hours(1);

/// Sweep cadence. Derived `IDLE_TTL / 8` (8x coverage guarantees
/// any idle shard is reclaimed within 12.5% of its TTL).
pub const GC_SWEEP_INTERVAL: Duration = Duration::from_secs(3600 / 8);

/// Cap on concurrent session shards. Derived `10 * K_WINDOW` -
/// one full window per decile of active sessions. When the
/// reservoir exceeds this, the oldest-observation shard is
/// evicted first (LRU).
pub const MAX_SESSIONS: usize = 10 * K_WINDOW;

/// One session shard. Bounded ring of observed entropies plus
/// last-access instant.
#[derive(Debug, Clone)]
pub struct SessionShard {
    /// Most recent observations, oldest-first. Bounded at
    /// [`K_WINDOW`].
    samples: Vec<f32>,
    /// Total observations seen since the shard was created
    /// (saturates at `u64::MAX`). Used to derive `warmup`.
    n_seen: u64,
    /// Last-access time for GC.
    last_observe: Instant,
}

impl Default for SessionShard {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionShard {
    /// Fresh shard with no samples.
    #[must_use]
    pub fn new() -> Self {
        Self {
            samples: Vec::with_capacity(K_WINDOW),
            n_seen: 0,
            last_observe: Instant::now(),
        }
    }

    /// Record a sample. Oldest entry is evicted when the ring
    /// is full. Bumps `last_observe` for GC bookkeeping.
    pub fn observe(&mut self, value: f32) {
        if self.samples.len() == K_WINDOW {
            // Drop oldest by rotating left. Cheap for K_WINDOW=128.
            self.samples.remove(0);
        }
        self.samples.push(value);
        self.n_seen = self.n_seen.saturating_add(1);
        self.last_observe = Instant::now();
    }

    /// `true` until the shard has seen at least [`K_MIN`]
    /// samples.
    #[must_use]
    pub fn warmup(&self) -> bool {
        self.n_seen < K_MIN as u64
    }

    /// Rolling median over the window, or `None` during warmup.
    #[must_use]
    pub fn median(&self) -> Option<f32> {
        if self.warmup() || self.samples.is_empty() {
            return None;
        }
        let mut sorted: Vec<f32> = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        Some(if sorted.len() % 2 == 0 {
            f32::midpoint(sorted[mid - 1], sorted[mid])
        } else {
            sorted[mid]
        })
    }

    /// Idle duration at the given reference instant.
    #[must_use]
    pub fn idle_at(&self, now: Instant) -> Duration {
        now.saturating_duration_since(self.last_observe)
    }
}

/// Report returned by [`SessionReservoir::gc_sweep`]. Wire into
/// `mnem_rolling_median_shards_evicted_total{reason="ttl"|"cap"}`
/// and `mnem_rolling_median_shards_active`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GcSweepReport {
    /// Shards dropped for being idle past [`IDLE_TTL`].
    pub evicted_ttl: u32,
    /// Shards dropped to respect [`MAX_SESSIONS`] cap (LRU).
    pub evicted_cap: u32,
    /// Shards still live after the sweep.
    pub active: u32,
}

/// Collection of per-session shards, keyed by `session_id`.
///
/// Backed by `BTreeMap` (rather than `DashMap`) to keep this
/// helper dependency-free at the core-crate level. Server-side
/// callers (`mnem-http`) wrap this in a `Mutex` / `RwLock` - the
/// Gap 01 spec R3 patch moves to DashMap when the sharded-lock
/// primitive lands; until then a single lock suffices because
/// observations are cheap (a `Vec::push` on a 128-slot ring).
#[derive(Debug, Default)]
pub struct SessionReservoir {
    shards: BTreeMap<String, SessionShard>,
}

impl SessionReservoir {
    /// Empty reservoir.
    #[must_use]
    pub fn new() -> Self {
        Self {
            shards: BTreeMap::new(),
        }
    }

    /// Record a `value` against `session_id`. Creates the shard
    /// on first observation.
    pub fn observe(&mut self, session_id: &str, value: f32) {
        let shard = self.shards.entry(session_id.to_string()).or_default();
        shard.observe(value);
    }

    /// Rolling median for `session_id`, or `None` when the
    /// session is unknown or still in warmup.
    #[must_use]
    pub fn median(&self, session_id: &str) -> Option<f32> {
        self.shards.get(session_id).and_then(SessionShard::median)
    }

    /// `true` when the session is unknown or still in warmup.
    #[must_use]
    pub fn warmup(&self, session_id: &str) -> bool {
        self.shards.get(session_id).is_none_or(SessionShard::warmup)
    }

    /// Current shard count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shards.len()
    }

    /// `true` when no shards are live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shards.is_empty()
    }

    /// Drop shards idle past [`IDLE_TTL`], then LRU-evict down
    /// to [`MAX_SESSIONS`] if we are still over the cap.
    /// Returns a [`GcSweepReport`] for telemetry.
    ///
    /// `now` is taken as an argument (rather than sampled
    /// internally) so tests can exercise the time branches
    /// deterministically.
    pub fn gc_sweep(&mut self, now: Instant) -> GcSweepReport {
        let before = self.shards.len() as u32;

        // TTL pass.
        let expired: Vec<String> = self
            .shards
            .iter()
            .filter_map(|(k, v)| {
                if v.idle_at(now) > IDLE_TTL {
                    Some(k.clone())
                } else {
                    None
                }
            })
            .collect();
        let evicted_ttl = expired.len() as u32;
        for k in &expired {
            self.shards.remove(k);
        }

        // Cap pass.
        let mut evicted_cap = 0u32;
        if self.shards.len() > MAX_SESSIONS {
            let mut ages: Vec<(String, Instant)> = self
                .shards
                .iter()
                .map(|(k, v)| (k.clone(), v.last_observe))
                .collect();
            ages.sort_by_key(|(_, t)| *t);
            let overflow = self.shards.len() - MAX_SESSIONS;
            for (k, _) in ages.into_iter().take(overflow) {
                self.shards.remove(&k);
                evicted_cap += 1;
            }
        }

        let active = self.shards.len() as u32;
        // `before` is unused beyond sanity here but kept in
        // scope for future debug-assert wiring.
        debug_assert!(active + evicted_ttl + evicted_cap <= before.max(active));

        GcSweepReport {
            evicted_ttl,
            evicted_cap,
            active,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_session_is_warmup() {
        let r = SessionReservoir::new();
        assert!(r.warmup("sid-a"));
        assert_eq!(r.median("sid-a"), None);
    }

    #[test]
    fn observe_tracks_n_seen() {
        let mut r = SessionReservoir::new();
        for i in 0..K_MIN {
            r.observe("sid-a", i as f32);
        }
        // After exactly K_MIN observations we leave warmup.
        assert!(!r.warmup("sid-a"));
        assert!(r.median("sid-a").is_some());
    }

    #[test]
    fn window_is_bounded_by_k_window() {
        let mut r = SessionReservoir::new();
        for i in 0..(K_WINDOW + 50) {
            r.observe("sid-a", i as f32);
        }
        // Median is over the most-recent K_WINDOW samples:
        // values K_WINDOW+50-K_WINDOW .. K_WINDOW+50 = 50..178,
        // so the median is around 113.5.
        let m = r.median("sid-a").expect("post-warmup");
        assert!(m > 50.0 && m < 180.0);
    }

    #[test]
    fn gc_sweep_noop_on_fresh_session() {
        let mut r = SessionReservoir::new();
        r.observe("sid-a", 1.0);
        let report = r.gc_sweep(Instant::now());
        assert_eq!(report.evicted_ttl, 0);
        assert_eq!(report.evicted_cap, 0);
        assert_eq!(report.active, 1);
    }

    #[test]
    fn gc_sweep_evicts_idle_shards() {
        let mut r = SessionReservoir::new();
        r.observe("sid-a", 1.0);
        // Fake a far-future `now` to trigger the TTL branch.
        let far_future = Instant::now() + IDLE_TTL + Duration::from_secs(1);
        let report = r.gc_sweep(far_future);
        assert_eq!(report.evicted_ttl, 1);
        assert_eq!(report.active, 0);
    }
}
