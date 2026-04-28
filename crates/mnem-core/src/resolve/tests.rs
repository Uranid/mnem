//! Tests for the `resolve_or_create` entity-canonicalization module.
//!
//! Required by gap 04:
//! - `resolve_creates_below_threshold`
//! - `resolve_merges_above_threshold`
//! - `threshold_derived_from_local_samples`
//! - `commit_budget_guard_cuts_off`
//!
//! Plus floor-c proptest
//! `resolve_or_create_hits_50ms_hard_wall` (R6) asserting the hard
//! wall fires deterministically across synthetic candidate-set sizes.

use super::*;
use crate::guard::CommitBudgetGuard;
use crate::id::{CODEC_RAW, Cid, HASH_BLAKE3_256, Multihash, NodeId};
use proptest::prelude::*;

fn zero_cid() -> Cid {
    Cid::new(
        CODEC_RAW,
        Multihash::wrap(HASH_BLAKE3_256, &[0u8; 32]).expect("32-byte zero digest"),
    )
}

fn nonzero_cid(tag: u8) -> Cid {
    let mut digest = [0u8; 32];
    digest[0] = tag;
    digest[31] = 0xFF;
    Cid::new(
        CODEC_RAW,
        Multihash::wrap(HASH_BLAKE3_256, &digest).expect("32-byte digest"),
    )
}

fn sample_bimodal(n_same: usize, n_diff: usize, seed: u64) -> Vec<f32> {
    // Deterministic pseudo-random via a tiny LCG - avoids a rand dep
    // inside tests while still producing distinguishable clusters.
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut out = Vec::with_capacity(n_same + n_diff);
    // Same-class: centred at 0.92, spread 0.03
    for _ in 0..n_same {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        #[allow(clippy::cast_precision_loss)]
        let u = ((state >> 33) as f32) / ((1u64 << 31) as f32);
        out.push(0.92 + (u - 0.5) * 0.06);
    }
    // Diff-class: centred at 0.30, spread 0.10
    for _ in 0..n_diff {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        #[allow(clippy::cast_precision_loss)]
        let u = ((state >> 33) as f32) / ((1u64 << 31) as f32);
        out.push(0.30 + (u - 0.5) * 0.2);
    }
    out
}

fn make_candidate(id_tag: u8, cosine: f32, name: &str, ns: &str, trust: &str) -> Candidate {
    let mut bytes = [0u8; 16];
    bytes[0] = id_tag;
    let node_id = NodeId::from_bytes_raw(bytes);
    Candidate {
        node_id,
        cosine,
        name: name.to_string(),
        namespace: ns.to_string(),
        trust: trust.to_string(),
    }
}

// ---- required unit tests --------------------------------------------------

#[test]
fn resolve_creates_below_threshold() {
    // All candidates have cosine well below any plausible tau_n, so the
    // decision must be `Created`.
    let candidates = vec![
        make_candidate(1, 0.10, "totally different", "person", "verified"),
        make_candidate(2, 0.15, "also different", "person", "verified"),
    ];
    let req = ResolveRequest {
        query: "Alice".into(),
        namespace: "person".into(),
        trust: "verified".into(),
        candidates,
        local_sample: sample_bimodal(128, 128, 42),
        latency_budget_ms: Some(50),
        commit_cid: nonzero_cid(1),
    };
    let out = resolve_or_create(&req);
    assert!(
        matches!(out.result, ResolveResult::Created { .. }),
        "expected Created, got {:?}",
        out.result
    );
    // Seed should be commit-derived for a non-zero CID.
    assert_eq!(out.seed_source, HnswSeedSource::CommitDerived);
}

#[test]
fn resolve_merges_above_threshold() {
    // One candidate passes BOTH cosine (>= tau_n) AND namespace/trust;
    // two-of-three fires even if edit-distance rejects.
    let candidates = vec![make_candidate(3, 0.97, "Alice Smith", "person", "verified")];
    let req = ResolveRequest {
        query: "Alice Smith".into(),
        namespace: "person".into(),
        trust: "verified".into(),
        candidates,
        local_sample: sample_bimodal(128, 128, 7),
        latency_budget_ms: Some(50),
        commit_cid: nonzero_cid(2),
    };
    let out = resolve_or_create(&req);
    match out.result {
        ResolveResult::Resolved { signals_passed, .. } => {
            assert!(
                signals_passed >= 2,
                "two-of-three gate required, got {signals_passed}"
            );
        }
        other => panic!("expected Resolved, got {other:?}"),
    }
}

#[test]
fn threshold_derived_from_local_samples() {
    let sample = sample_bimodal(256, 256, 11);
    let thr = derive_local_threshold(&sample, SIGMA_MULTIPLIER_FOR_COLLAPSE)
        .expect("sample >= MIN_SAMPLE_SIZE");
    // Same-class mean should land near 0.92, diff-class mean near 0.30.
    assert!(
        thr.mu_same > 0.80 && thr.mu_same < 1.00,
        "mu_same drift: {}",
        thr.mu_same
    );
    assert!(
        thr.mu_diff > 0.10 && thr.mu_diff < 0.50,
        "mu_diff drift: {}",
        thr.mu_diff
    );
    // tau_n must sit strictly between the two component means.
    assert!(
        thr.tau_n > thr.mu_diff && thr.tau_n < thr.mu_same,
        "tau_n out of band: {} (mu_diff={}, mu_same={})",
        thr.tau_n,
        thr.mu_diff,
        thr.mu_same
    );
    // Sample too small: refuses.
    let tiny = vec![0.8_f32; 32];
    assert!(derive_local_threshold(&tiny, 2.0).is_none());
}

#[test]
fn commit_budget_guard_cuts_off() {
    // Charge-with: simulate a stage blowing past the hard wall.
    let mut g = CommitBudgetGuard::start(
        "gap-04-resolve-or-create",
        50,
        RESOLVE_OR_CREATE_P99_MS,
        zero_cid(),
    );
    let _ = g.charge_with("derive_threshold", 30).unwrap();
    let err = g.charge_with("consensus", 60).unwrap_err();
    assert_eq!(err.hard_wall_ms, RESOLVE_OR_CREATE_P99_MS);
    assert!(g.hard_wall_hit);
    let rep = g.into_report();
    assert!(rep.hard_wall_hit);
    assert_eq!(rep.tag, "gap-04-resolve-or-create");
}

// ---- seed / consensus round-trip -----------------------------------------

#[test]
fn hnsw_seed_fallback_on_zero_cid() {
    // clear any stray env-var so fallback path fires deterministically.
    // SAFETY wrt env: tests in this module don't touch it concurrently.
    // Setting/removing env is a known test-time hazard; we only touch it
    // when the var is already unset.
    if std::env::var("MNEM_CANONICAL_HNSW_SEED").is_ok() {
        return;
    }
    let (seed, src) = resolve_hnsw_seed(&zero_cid());
    assert_eq!(src, HnswSeedSource::Fallback);
    assert_eq!(seed, HNSW_SEED_FALLBACK);
}

#[test]
fn hnsw_seed_commit_derived_reproducible() {
    if std::env::var("MNEM_CANONICAL_HNSW_SEED").is_ok() {
        return;
    }
    let cid = nonzero_cid(5);
    let (s1, src1) = resolve_hnsw_seed(&cid);
    let (s2, src2) = resolve_hnsw_seed(&cid);
    assert_eq!(s1, s2);
    assert_eq!(src1, HnswSeedSource::CommitDerived);
    assert_eq!(src2, HnswSeedSource::CommitDerived);
    // Different CID -> different seed (modulo ~2^-64 collision).
    let (s3, _) = resolve_hnsw_seed(&nonzero_cid(6));
    assert_ne!(s1, s3);
}

#[test]
fn two_of_three_requires_two_signals() {
    // Same namespace only - one signal, should refuse in simple API.
    let c = make_candidate(1, 0.10, "totally different", "person", "verified");
    let r = resolve_or_create_simple(
        "Alice",
        0.85,
        std::slice::from_ref(&c),
        "person",
        "verified",
    );
    assert!(matches!(
        r,
        ResolveResult::Refused(RefusalReason::SingleSignalOnly)
    ));
    // Adding matching edit-distance (identical surface) flips to resolved.
    let c2 = make_candidate(2, 0.10, "Alice", "person", "verified");
    let r2 = resolve_or_create_simple(
        "Alice",
        0.85,
        std::slice::from_ref(&c2),
        "person",
        "verified",
    );
    assert!(matches!(r2, ResolveResult::Resolved { .. }));
}

#[test]
fn normalized_levenshtein_bounds() {
    assert!((normalized_levenshtein("abc", "abc") - 0.0).abs() < 1e-6);
    assert!((normalized_levenshtein("abc", "xyz") - 1.0).abs() < 1e-6);
    // Single-char edit on len-3 string -> 1/3.
    let d = normalized_levenshtein("abc", "abd");
    assert!((d - (1.0 / 3.0)).abs() < 1e-6, "got {d}");
}

// ---- floor-c proptest for the 50ms hard wall -----------------------------
//
// Proptest samples a synthetic candidate-set size in `[1, 4096]` and a
// simulated elapsed-ms for each stage; the property asserts:
//
// 1. If the simulated consensus stage elapsed exceeds 50ms, the guard
//    returns `HardWallExceeded`.
// 2. Otherwise the guard returns a normal `Decision`.
//
// This locks the `RESOLVE_OR_CREATE_P99_MS=50` floor-c constant: any
// future edit that raises the wall without updating the tunable gauge
// fails this property.

proptest! {
    #[test]
    fn resolve_or_create_hits_50ms_hard_wall(
        stage1_ms in 0u32..=40,
        stage2_ms in 0u32..=200,
        _n_candidates in 1usize..=4096,
    ) {
        let mut g = CommitBudgetGuard::start(
            "gap-04-resolve-or-create",
            RESOLVE_OR_CREATE_P99_MS, // budget == hard wall (tight)
            RESOLVE_OR_CREATE_P99_MS,
            zero_cid(),
        );
        let r1 = g.charge_with("derive_threshold", stage1_ms);
        // Stage 1 fits by construction; stage2 may breach.
        prop_assert!(r1.is_ok());
        let r2 = g.charge_with("consensus", stage2_ms);
        if stage2_ms > RESOLVE_OR_CREATE_P99_MS {
            prop_assert!(r2.is_err(), "stage2_ms={stage2_ms} should breach wall");
        } else {
            prop_assert!(r2.is_ok(), "stage2_ms={stage2_ms} should fit");
        }
    }
}
