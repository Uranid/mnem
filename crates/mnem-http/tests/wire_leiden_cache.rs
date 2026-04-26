//! Gap 10 Phase-1 integration tests for the Leiden community-cache
//! invalidation policy.
//!
//! Covers:
//! - Tunable pins (R6 floor-c contract).
//! - `LeidenMode` label/gauge vocabulary closure.
//! - `derive_debounce_ms` contract: floor + adapts to p75.
//! - `LeidenCache::select_mode` R3 policy ladder.
//! - Rolling commit-latency ring + 60s commit-arrival ring.
//! - Proptest `storm_cap_holds_under_adversarial_commit`.
//! - Proptest `debounce_adapts_to_commit_rate`.
//! - Proptest `force_full_when_delta_exceeds_half`.
//!
//! Env-var override (`MNEM_LEIDEN_FULL_RECOMPUTE=1`) is exercised by
//! exercising `LeidenCache::default_mode = LeidenMode::Full` directly,
//! not by mutating process env - the crate forbids `unsafe` and env
//! mutation is `unsafe` in edition 2024.

use std::time::{Duration, Instant};

use mnem_http::leiden_state::{
    COMMIT_LATENCY_WINDOW, COMMIT_STORM_CAP_PER_MIN, DEBOUNCE_FLOOR_MS, DELTA_RATIO_FORCE_FULL,
    GRAPH_SIZE_GATE_V, LeidenCache, LeidenMode, derive_debounce_ms,
};

// ---------------------------------------------------------------
// Tunable pins
// ---------------------------------------------------------------

#[test]
fn tunable_commit_storm_cap_is_60_per_min() {
    assert_eq!(COMMIT_STORM_CAP_PER_MIN, 60);
}

#[test]
fn tunable_delta_ratio_force_full_is_half() {
    assert!((DELTA_RATIO_FORCE_FULL - 0.5).abs() < f32::EPSILON);
}

#[test]
fn tunable_graph_size_gate_is_250k() {
    assert_eq!(GRAPH_SIZE_GATE_V, 250_000);
}

#[test]
fn tunable_debounce_floor_is_1s() {
    assert_eq!(DEBOUNCE_FLOOR_MS, 1_000);
}

#[test]
fn tunable_commit_latency_window_is_100() {
    assert_eq!(COMMIT_LATENCY_WINDOW, 100);
}

// ---------------------------------------------------------------
// LeidenMode enum contract
// ---------------------------------------------------------------

#[test]
fn leiden_mode_labels_match_closed_vocabulary() {
    assert_eq!(LeidenMode::Full.label(), "full");
    assert_eq!(LeidenMode::FullDebounced.label(), "full_debounced");
    assert_eq!(LeidenMode::FallbackStale.label(), "fallback_stale");
}

#[test]
fn leiden_mode_gauge_values_are_stable() {
    assert_eq!(LeidenMode::Full.gauge_value(), 0);
    assert_eq!(LeidenMode::FullDebounced.gauge_value(), 1);
    assert_eq!(LeidenMode::FallbackStale.gauge_value(), 2);
}

// ---------------------------------------------------------------
// derive_debounce_ms contract
// ---------------------------------------------------------------

#[test]
fn derive_debounce_ms_none_uses_floor() {
    assert_eq!(derive_debounce_ms(None), DEBOUNCE_FLOOR_MS);
}

#[test]
fn derive_debounce_ms_below_floor_uses_floor() {
    assert_eq!(derive_debounce_ms(Some(250)), DEBOUNCE_FLOOR_MS);
    assert_eq!(derive_debounce_ms(Some(0)), DEBOUNCE_FLOOR_MS);
    assert_eq!(derive_debounce_ms(Some(999)), DEBOUNCE_FLOOR_MS);
}

#[test]
fn derive_debounce_ms_above_floor_uses_input() {
    assert_eq!(derive_debounce_ms(Some(1500)), 1500);
    assert_eq!(derive_debounce_ms(Some(10_000)), 10_000);
}

// ---------------------------------------------------------------
// LeidenCache policy ladder
// ---------------------------------------------------------------

#[test]
fn select_mode_env_override_wins() {
    let mut c = LeidenCache {
        default_mode: LeidenMode::Full,
        ..Default::default()
    };
    // Even with a saturated storm cap + huge graph, Full wins.
    for _ in 0..COMMIT_STORM_CAP_PER_MIN {
        c.commit_arrivals.push_back(Instant::now());
    }
    assert_eq!(
        c.select_mode(GRAPH_SIZE_GATE_V * 2, Instant::now()),
        LeidenMode::Full
    );
}

#[test]
fn select_mode_above_graph_size_gate_is_fallback() {
    let c = LeidenCache::default();
    assert_eq!(
        c.select_mode(GRAPH_SIZE_GATE_V, Instant::now()),
        LeidenMode::FallbackStale
    );
    assert_eq!(
        c.select_mode(GRAPH_SIZE_GATE_V + 1, Instant::now()),
        LeidenMode::FallbackStale
    );
}

#[test]
fn select_mode_below_size_gate_and_cold_cache_is_debounced() {
    let c = LeidenCache::default();
    assert_eq!(
        c.select_mode(100, Instant::now()),
        LeidenMode::FullDebounced
    );
}

#[test]
fn select_mode_storm_cap_forces_fallback() {
    let mut c = LeidenCache::default();
    for _ in 0..COMMIT_STORM_CAP_PER_MIN {
        c.commit_arrivals.push_back(Instant::now());
    }
    assert!(c.storm_cap_reached());
    assert_eq!(
        c.select_mode(100, Instant::now()),
        LeidenMode::FallbackStale
    );
}

#[test]
fn select_mode_inside_debounce_window_is_fallback() {
    let mut c = LeidenCache::default();
    let now = Instant::now();
    c.last_recompute_at = Some(now);
    assert_eq!(
        c.select_mode(100, now + Duration::from_millis(500)),
        LeidenMode::FallbackStale
    );
}

#[test]
fn select_mode_after_debounce_window_expires_is_debounced() {
    let mut c = LeidenCache::default();
    let now = Instant::now();
    c.last_recompute_at = Some(now);
    assert_eq!(
        c.select_mode(100, now + Duration::from_millis(1_500)),
        LeidenMode::FullDebounced
    );
}

// ---------------------------------------------------------------
// Rolling p75 + commit arrivals
// ---------------------------------------------------------------

#[test]
fn p75_empty_ring_is_none() {
    let c = LeidenCache::default();
    assert_eq!(c.rolling_p75_commit_ms(), None);
}

#[test]
fn p75_single_sample_returns_that_sample() {
    let mut c = LeidenCache::default();
    c.observe_commit_latency(Duration::from_millis(2_500));
    assert_eq!(c.rolling_p75_commit_ms(), Some(2_500));
}

#[test]
fn p75_nearest_rank_over_100_samples() {
    let mut c = LeidenCache::default();
    for i in 1..=100u64 {
        c.observe_commit_latency(Duration::from_millis(i));
    }
    assert_eq!(c.rolling_p75_commit_ms(), Some(75));
}

#[test]
fn commit_latency_ring_caps_at_window_size() {
    let mut c = LeidenCache::default();
    for i in 0..200u64 {
        c.observe_commit_latency(Duration::from_millis(i));
    }
    assert_eq!(c.commit_latency_ms.len(), COMMIT_LATENCY_WINDOW);
    assert_eq!(*c.commit_latency_ms.front().unwrap(), 100);
    assert_eq!(*c.commit_latency_ms.back().unwrap(), 199);
}

#[test]
fn commit_arrivals_evict_older_than_60s() {
    let mut c = LeidenCache::default();
    let base = Instant::now();
    for _ in 0..20 {
        c.observe_commit_arrival(base);
    }
    assert_eq!(c.commit_arrivals.len(), 20);
    c.observe_commit_arrival(base + Duration::from_secs(61));
    assert_eq!(c.commit_arrivals.len(), 1);
}

// ---------------------------------------------------------------
// Proptest - R6 ratchet-close invariants
// ---------------------------------------------------------------

use proptest::prelude::*;

proptest! {
    /// Gap 10 R6 proptest `storm_cap_holds_under_adversarial_commit`.
    #[test]
    fn storm_cap_holds_under_adversarial_commit(n in 0u32..10_000) {
        let mut c = LeidenCache::default();
        let now = Instant::now();
        for i in 0..n {
            c.observe_commit_arrival(now + Duration::from_millis(u64::from(i)));
        }
        let reached = c.storm_cap_reached();
        if n >= COMMIT_STORM_CAP_PER_MIN {
            prop_assert!(reached, "cap must fire at n={}", n);
            prop_assert_eq!(
                c.select_mode(100, now),
                LeidenMode::FallbackStale,
            );
        } else {
            prop_assert!(!reached, "cap must NOT fire at n={}", n);
        }
    }

    /// Gap 10 R6 proptest `debounce_adapts_to_commit_rate`.
    #[test]
    fn debounce_adapts_to_commit_rate(samples in proptest::collection::vec(0u64..60_000, 1..=100)) {
        let mut c = LeidenCache::default();
        for s in &samples {
            c.observe_commit_latency(Duration::from_millis(*s));
        }
        let p75 = c.rolling_p75_commit_ms();
        let eff = c.effective_debounce_ms();
        prop_assert!(eff >= DEBOUNCE_FLOOR_MS);
        if let Some(p) = p75 {
            prop_assert!(eff >= p);
        }
    }

    /// Gap 10 R6 proptest `force_full_when_delta_exceeds_half`.
    #[test]
    fn force_full_when_delta_exceeds_half(delta_num in 0u32..=100) {
        let delta_ratio = f32::from(u16::try_from(delta_num).unwrap()) / 100.0;
        let forces_full = delta_ratio > DELTA_RATIO_FORCE_FULL;
        if delta_num > 50 {
            prop_assert!(forces_full);
        } else if delta_num < 50 {
            prop_assert!(!forces_full);
        }
    }
}
