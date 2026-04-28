//! Entity canonicalization - `resolve_or_create` (gap-catalog gap 04).
//!
//! This module implements the collapse-or-create decision for a string
//! `query` against an HNSW-indexed population of already-known nodes.
//! The design follows `research/gap-catalog/04-entity-canonicalization/`
//! R1-R6, in particular:
//!
//! - **Distribution-derived collapse threshold** `tau_n` computed from
//!   a k=2 Gaussian Mixture over the HNSW-local cosine sample. No global
//!   magic cosine constant; the threshold tracks the corpus geometry.
//!   `tau_n = max(mu_same - 2*sigma_same, mu_diff + sigma_diff)`.
//! - **Two-of-three consensus collapse gate**: at least two of
//!   (cosine, normalized_levenshtein, namespace/trust) must agree for
//!   two nodes to be merged. Single-signal collapses are refused.
//! - **Commit-id-derived HNSW seed**: the HNSW walk seed is
//!   `BLAKE3(commit_cid || domain_sep)[..8]` - two runs against the
//!   same commit get the same seed; different commits get independent
//!   seeds. Bootstrap fallback `0xCANO_N_0001_u64` when the commit CID
//!   is the zero CID.
//! - **`CommitBudgetGuard` wiring**: caller passes
//!   `latency_budget_ms: Option<u32>` and the module opens a guard at
//!   [`RESOLVE_OR_CREATE_P99_MS`] hard wall; exhaustion returns
//!   [`ResolveResult::BudgetExhausted`] carrying the best-effort
//!   candidate.
//!
//! # p99 floor-c apparatus (R6)
//!
//! [`RESOLVE_OR_CREATE_P99_MS`] is a tunable floor-c constant:
//!
//! - Reference standard: `p95_hnsw_walk_ms + consensus_overhead_ms
//!   + p99_headroom = 50` on the reference repo (`|V|=1M`,
//!   `avg_degree=12`).
//! - Gauge: `mnem_resolve_or_create_p99_breach_total`.
//! - Proptest: [`tests::resolve_or_create_hits_50ms_hard_wall`].
//! - Unit test:
//!   [`tests::resolve_creates_below_threshold`],
//!   [`tests::resolve_merges_above_threshold`],
//!   [`tests::threshold_derived_from_local_samples`],
//!   [`tests::commit_budget_guard_cuts_off`].
//!
//! # Rollback template (see `scripts/rollback-gap-04.sql`)
//!
//! Rolling canonicalization back uses the following idempotent SQL
//! template, kept here as a comment so readers don't have to chase the
//! script file:
//!
//! ```sql
//! -- scripts/rollback-gap-04.sql
//! -- Rollback entity canonicalization emitted after <ROLLBACK_CID>.
//! -- Invocation: mnem admin rollback --feature=canonicalization --after=<CID>
//! -- Idempotent: re-running is safe (second run is a no-op).
//!
//! BEGIN TRANSACTION;
//!
//! -- 1. Drop canonical_cid props from nodes committed after the point.
//! UPDATE nodes
//!    SET props = json_remove(props, '$.canonical_cid')
//!  WHERE commit_cid > :ROLLBACK_CID
//!    AND json_extract(props, '$.canonical_cid') IS NOT NULL;
//!
//! -- 2. Drop the canonical cluster manifest rows.
//! DROP TABLE IF EXISTS canonical_manifest_staging;
//! DELETE FROM canonical_manifest
//!  WHERE commit_cid > :ROLLBACK_CID;
//!
//! -- 3. Cache-flush NOTIFY handled post-SQL by mnem admin rollback:
//! --    posts INTERNAL ResetCanonicalCache event to runtime, which
//! --    drains AppState::canonical_cache + rebuilds lazily.
//! NOTIFY canonical_cache_flush, :ROLLBACK_CID;
//!
//! -- 4. Reset rolling-telemetry derived counters so SLO alerting
//! --    does not attribute post-rollback baselines to rolled commits.
//! UPDATE rolling_stats
//!    SET p50_canonicalize_ms = NULL,
//!        p99_canonicalize_ms = NULL
//!  WHERE last_updated_commit_cid > :ROLLBACK_CID;
//!
//! COMMIT;
//! ```

use crate::guard::CommitBudgetGuard;
use crate::id::{Cid, NodeId};

/// R5 numeric p99 SLO for `mnem_resolve_or_create`.
///
/// Derivation (floor-c, R6): `p95_hnsw_walk_ms (~35ms) +
/// consensus_overhead_ms (~5ms) + p99_headroom (~10ms) = 50`.
/// Labelled tunable. Exposed via `mnem_resolve_or_create_p99_breach_total`.
#[doc = "#[tunable]"]
pub const RESOLVE_OR_CREATE_P99_MS: u32 = 50;

/// R4 pinned ef_search for canonicalization HNSW handle. Separate
/// from retrieve ef_search to avoid cross-path drift. Reference
/// standard: Malkov-Yashunin 2016 §4 recall-vs-latency envelope
/// (ef=128 yields recall >= 0.95 at p95 latency < 20ms for 768-dim).
pub const EF_SEARCH_CANONICAL: u32 = 128;

/// R5 bootstrap-only HNSW seed fallback for when `commit_cid` is the
/// zero CID (e.g. the first commit in an empty repo).
pub const HNSW_SEED_FALLBACK: u64 = 0xCA_00_00_00_01_00_00_00_u64;

/// R3 same-class sigma multiplier for collapse threshold. Derivation:
/// DBSCAN-/HDBSCAN-style inlier boundary `mean - 2*sigma`. Clamped to
/// `[1.5, 3.0]` at manifest-load time.
pub const SIGMA_MULTIPLIER_FOR_COLLAPSE: f32 = 2.0;

/// R3 same-class edit-distance tau (embedder-calibrated).
/// Max 25% normalized Levenshtein distance qualifies as an edit-dist
/// collapse signal.
pub const EDIT_DISTANCE_TAU: f32 = 0.25;

/// R4 minimum HNSW neighbourhood size below which threshold
/// derivation refuses to emit `canonical_cid`.
pub const MIN_SAMPLE_SIZE: usize = 128;

/// Origin of the HNSW build seed used for this run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HnswSeedSource {
    /// Seed was derived via BLAKE3(commit_cid || domain_sep).
    CommitDerived,
    /// Seed came from `MNEM_CANONICAL_HNSW_SEED` env var.
    EnvOverride,
    /// `commit_cid.is_zero()` path: bootstrap constant.
    Fallback,
}

/// Reasons a resolve call was refused (not merged, not created).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalReason {
    /// HNSW-local sample below [`MIN_SAMPLE_SIZE`]; threshold cannot
    /// be derived with statistical significance.
    SampleTooSmall,
    /// Best candidate passed only one of the three consensus signals.
    SingleSignalOnly,
}

/// Outcome of [`resolve_or_create`].
#[derive(Debug, Clone, PartialEq)]
pub enum ResolveResult {
    /// Query collapsed onto an existing node.
    Resolved {
        /// The existing canonical node.
        node_id: NodeId,
        /// Number of consensus signals that agreed (2 or 3).
        signals_passed: u8,
    },
    /// Query did not match any existing node; caller should create.
    Created {
        /// Threshold used to decide the above-threshold mass was empty.
        tau_n: f32,
    },
    /// Guard ran the wall-clock budget out. `best_effort` is the
    /// top HNSW candidate if any; caller may retry with a larger
    /// budget.
    BudgetExhausted {
        /// Best candidate observed before the budget ran out.
        best_effort: Option<NodeId>,
    },
    /// Refused to emit a decision (see [`RefusalReason`]).
    Refused(RefusalReason),
}

/// A sampled (candidate_id, cosine_to_query, name_for_edit_dist,
/// namespace, trust) tuple. Lifetime-free for testability: a real
/// caller pulls these from the HNSW walk.
#[derive(Debug, Clone)]
pub struct Candidate {
    /// Stable id of the candidate node.
    pub node_id: NodeId,
    /// Cosine similarity of candidate's embedding to the query.
    pub cosine: f32,
    /// Surface-form name of the candidate (for edit-distance signal).
    pub name: String,
    /// Candidate's namespace (e.g. "person", "company").
    pub namespace: String,
    /// Candidate's trust label.
    pub trust: String,
}

/// Per-node distribution-derived threshold and its component stats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalThreshold {
    /// The derived `tau_n = max(mu_same - k*sigma_same, mu_diff + sigma_diff)`.
    pub tau_n: f32,
    /// Mean of the same-class (higher-mean) GMM component.
    pub mu_same: f32,
    /// Std-dev of the same-class GMM component.
    pub sigma_same: f32,
    /// Mean of the different-class (lower-mean) GMM component.
    pub mu_diff: f32,
    /// Std-dev of the different-class GMM component.
    pub sigma_diff: f32,
    /// Size of the HNSW-local sample used.
    pub sample_size: usize,
}

/// Run k=2 Gaussian Mixture on a pre-computed HNSW-local cosine
/// sample and return the distribution-derived collapse threshold.
///
/// The sample must contain at least [`MIN_SAMPLE_SIZE`] observations;
/// smaller samples return `None` (caller must then refuse to emit a
/// `canonical_cid` and log
/// `mnem_canonical_threshold_sample_insufficient_total`).
///
/// The implementation is a 1-D two-component GMM via a compact EM
/// loop - no external crate. The loop is bounded at 32 iterations
/// with a tolerance of `1e-4` on component-mean movement, both
/// sufficient for 1-D bi-modal separation of embedding cosines.
#[must_use]
pub fn derive_local_threshold(cosines: &[f32], sigma_multiplier: f32) -> Option<LocalThreshold> {
    if cosines.len() < MIN_SAMPLE_SIZE {
        return None;
    }
    // k=2 EM with deterministic init from sample min/max.
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &c in cosines {
        if c < lo {
            lo = c;
        }
        if c > hi {
            hi = c;
        }
    }
    if hi <= lo {
        // degenerate constant sample - no bimodality; return a
        // single-gaussian surrogate so the caller still gets a tau.
        let mu = f32::midpoint(hi, lo);
        return Some(LocalThreshold {
            tau_n: mu,
            mu_same: mu,
            sigma_same: 0.0,
            mu_diff: mu,
            sigma_diff: 0.0,
            sample_size: cosines.len(),
        });
    }
    // Deterministic init: low-quartile mean vs high-quartile mean.
    let mut mu0 = lo + (hi - lo) * 0.25;
    let mut mu1 = lo + (hi - lo) * 0.75;
    let mut s0 = (hi - lo) / 4.0;
    let mut s1 = (hi - lo) / 4.0;
    let mut w0 = 0.5_f32;
    let mut w1 = 0.5_f32;
    for _ in 0..32 {
        // E-step: soft responsibilities.
        let mut n0 = 0.0f32;
        let mut n1 = 0.0f32;
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sq0 = 0.0f32;
        let mut sq1 = 0.0f32;
        for &x in cosines {
            let p0 = w0 * gaussian_pdf(x, mu0, s0.max(1e-6));
            let p1 = w1 * gaussian_pdf(x, mu1, s1.max(1e-6));
            let z = p0 + p1;
            let (r0, r1) = if z > 0.0 {
                (p0 / z, p1 / z)
            } else {
                (0.5, 0.5)
            };
            n0 += r0;
            n1 += r1;
            sum0 += r0 * x;
            sum1 += r1 * x;
            sq0 += r0 * x * x;
            sq1 += r1 * x * x;
        }
        // M-step.
        let new_mu0 = if n0 > 0.0 { sum0 / n0 } else { mu0 };
        let new_mu1 = if n1 > 0.0 { sum1 / n1 } else { mu1 };
        // clippy::suspicious_operation_groupings flags `a*a` adjacent to
        // `b/c - d*d`; the expression is the standard
        // variance = E[X^2] - (E[X])^2 form so we silence the lint.
        #[allow(clippy::suspicious_operation_groupings)]
        let new_s0 = if n0 > 0.0 {
            ((sq0 / n0) - new_mu0 * new_mu0).max(1e-8).sqrt()
        } else {
            s0
        };
        #[allow(clippy::suspicious_operation_groupings)]
        let new_s1 = if n1 > 0.0 {
            ((sq1 / n1) - new_mu1 * new_mu1).max(1e-8).sqrt()
        } else {
            s1
        };
        let n_total = n0 + n1;
        let new_w0 = if n_total > 0.0 { n0 / n_total } else { 0.5 };
        let new_w1 = 1.0 - new_w0;
        let moved = (new_mu0 - mu0).abs() + (new_mu1 - mu1).abs();
        mu0 = new_mu0;
        mu1 = new_mu1;
        s0 = new_s0;
        s1 = new_s1;
        w0 = new_w0;
        w1 = new_w1;
        if moved < 1e-4 {
            break;
        }
    }
    // Same-class = higher-mean component.
    let (mu_same, sigma_same, mu_diff, sigma_diff) = if mu1 >= mu0 {
        (mu1, s1, mu0, s0)
    } else {
        (mu0, s0, mu1, s1)
    };
    let low = mu_same - sigma_multiplier * sigma_same;
    let high = mu_diff + sigma_diff;
    let tau_n = if low >= high { low } else { high };
    Some(LocalThreshold {
        tau_n,
        mu_same,
        sigma_same,
        mu_diff,
        sigma_diff,
        sample_size: cosines.len(),
    })
}

/// 1-D gaussian PDF. Pulled inline - no external `statrs` dep.
#[inline]
fn gaussian_pdf(x: f32, mu: f32, sigma: f32) -> f32 {
    let inv = 1.0 / (sigma * (2.0 * core::f32::consts::PI).sqrt());
    let z = (x - mu) / sigma;
    inv * (-0.5 * z * z).exp()
}

/// Normalized Levenshtein distance in `[0, 1]`. `0` = identical,
/// `1` = maximally different. Used by the edit-distance consensus
/// signal. Implementation is the classic O(m*n) DP matrix, pure-Rust,
/// no extra crate. Short names dominate here so memory is a non-issue.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn normalized_levenshtein(a: &str, b: &str) -> f32 {
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    if av.is_empty() && bv.is_empty() {
        return 0.0;
    }
    let m = av.len();
    let n = bv.len();
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut cur: Vec<usize> = vec![0; n + 1];
    for i in 1..=m {
        cur[0] = i;
        for j in 1..=n {
            let cost = usize::from(av[i - 1] != bv[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let max_len = m.max(n);
    #[allow(clippy::cast_precision_loss)]
    let d = prev[n] as f32 / max_len as f32;
    d.clamp(0.0, 1.0)
}

/// Two-of-three consensus: returns `(signals_passed, per_signal)` where
/// `per_signal = [cosine_ok, edit_ok, namespace_ok]`.
///
/// - `cosine_ok`: `cand.cosine >= tau_n` AND `cand.cosine >= tau_query`
///   (symmetric collapse). We pass the same `tau_n` twice for the
///   standalone resolve path (query has no pre-existing neighbourhood).
/// - `edit_ok`: `normalized_levenshtein(query, cand.name) <= EDIT_DISTANCE_TAU`.
/// - `namespace_ok`: `cand.namespace == query_namespace
///   AND cand.trust == query_trust`.
#[must_use]
pub fn two_of_three_consensus(
    query: &str,
    query_namespace: &str,
    query_trust: &str,
    cand: &Candidate,
    tau_n: f32,
) -> (u8, [bool; 3]) {
    let cosine_ok = cand.cosine >= tau_n;
    let edit_ok = normalized_levenshtein(query, &cand.name) <= EDIT_DISTANCE_TAU;
    let namespace_ok = cand.namespace == query_namespace && cand.trust == query_trust;
    let passed = u8::from(cosine_ok) + u8::from(edit_ok) + u8::from(namespace_ok);
    (passed, [cosine_ok, edit_ok, namespace_ok])
}

/// Resolve the commit-derived HNSW build seed.
///
/// - `MNEM_CANONICAL_HNSW_SEED` env var wins (decimal or `0x...` hex).
/// - Else `BLAKE3(commit_cid.to_bytes() || domain_sep)[..8]` little-endian.
/// - Else (commit_cid is zero): [`HNSW_SEED_FALLBACK`].
///
/// Note: `commit_cid.is_zero()` is approximated by comparing the CID's
/// binary form to the zero-digest for the configured codec / hash.
/// For testing we check whether every byte of the multihash digest is
/// zero.
#[must_use]
pub fn resolve_hnsw_seed(commit_cid: &Cid) -> (u64, HnswSeedSource) {
    if let Ok(s) = std::env::var("MNEM_CANONICAL_HNSW_SEED") {
        if let Some(val) = parse_u64_dec_or_hex(&s) {
            return (val, HnswSeedSource::EnvOverride);
        }
    }
    if cid_has_zero_digest(commit_cid) {
        return (HNSW_SEED_FALLBACK, HnswSeedSource::Fallback);
    }
    let mut h = blake3::Hasher::new();
    let bytes = commit_cid.to_bytes();
    h.update(&bytes);
    h.update(b"mnem-gap-04-canonical-hnsw-v1");
    let digest = h.finalize();
    let d = digest.as_bytes();
    let seed = u64::from_le_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]);
    (seed, HnswSeedSource::CommitDerived)
}

fn parse_u64_dec_or_hex(s: &str) -> Option<u64> {
    let t = s.trim();
    if let Some(rest) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        u64::from_str_radix(rest, 16).ok()
    } else {
        t.parse::<u64>().ok()
    }
}

fn cid_has_zero_digest(cid: &Cid) -> bool {
    let bytes = cid.to_bytes();
    // A CID in wire form is (version || codec || multihash). The
    // multihash is (hash_code || len || digest). The trailing
    // `len` bytes of `bytes` are the digest. We approximate
    // "zero CID" as "all digest bytes are zero" which matches the
    // `zero_cid()` helper used in tests.
    bytes.iter().rev().take(32).all(|&b| b == 0)
}

/// Request payload for [`resolve_or_create`].
#[derive(Debug, Clone)]
pub struct ResolveRequest {
    /// The surface-form string to resolve.
    pub query: String,
    /// Namespace (e.g. "person", "company").
    pub namespace: String,
    /// Trust label (e.g. "verified").
    pub trust: String,
    /// Candidates pre-sampled from the HNSW walk. In a real pipeline
    /// this is populated by the caller from the canonicalization
    /// HNSW handle keyed by the commit-derived seed.
    pub candidates: Vec<Candidate>,
    /// HNSW-local cosine sample for threshold derivation. Must be at
    /// least [`MIN_SAMPLE_SIZE`] long; shorter samples return
    /// [`ResolveResult::Refused`].
    pub local_sample: Vec<f32>,
    /// Caller-supplied latency budget override. `None` means use
    /// [`RESOLVE_OR_CREATE_P99_MS`].
    pub latency_budget_ms: Option<u32>,
    /// The commit the resolve is happening under (for HNSW seed +
    /// guard envelope).
    pub commit_cid: Cid,
}

/// Full outcome of a resolve call, including the guard's report for
/// embedding in the commit envelope and the (seed, source) pair used
/// for the HNSW walk.
#[derive(Debug)]
pub struct ResolveOutcome {
    /// Primary resolution decision.
    pub result: ResolveResult,
    /// Budget-guard report - host embeds in the commit envelope and
    /// feeds to the metric sink.
    pub report: crate::guard::CommitBudgetReport,
    /// Seed used for the HNSW walk (for audit / replay determinism).
    pub seed: u64,
    /// Source of the HNSW seed for this run.
    pub seed_source: HnswSeedSource,
    /// The per-node distribution-derived threshold. `None` when the
    /// local sample was too small.
    pub threshold: Option<LocalThreshold>,
}

/// Resolve a query string onto an existing canonical node, or
/// decide that a new node should be created.
///
/// The function is pure except for (a) wall-clock reads via the
/// `CommitBudgetGuard` and (b) the optional
/// `MNEM_CANONICAL_HNSW_SEED` env-var. Both are documented.
///
/// For a concrete single-shot API `(query, threshold) -> ResolveResult`
/// see [`resolve_or_create_simple`]; this function is the full
/// production shape carrying candidates + local sample.
///
/// # Panics
///
/// Does not panic. All fallible paths map to [`ResolveResult`] variants.
pub fn resolve_or_create(req: &ResolveRequest) -> ResolveOutcome {
    let budget_ms = req.latency_budget_ms.unwrap_or(RESOLVE_OR_CREATE_P99_MS);
    let mut guard = CommitBudgetGuard::start(
        "gap-04-resolve-or-create",
        budget_ms,
        RESOLVE_OR_CREATE_P99_MS,
        req.commit_cid.clone(),
    );
    let (seed, seed_source) = resolve_hnsw_seed(&req.commit_cid);

    // Stage 1: derive distribution threshold.
    let threshold = derive_local_threshold(&req.local_sample, SIGMA_MULTIPLIER_FOR_COLLAPSE);
    let charge1 = guard.charge("derive_threshold");
    if charge1.is_err() {
        let report = guard.into_report();
        return ResolveOutcome {
            result: ResolveResult::BudgetExhausted { best_effort: None },
            report,
            seed,
            seed_source,
            threshold,
        };
    }
    let Some(thr) = threshold else {
        let report = guard.into_report();
        return ResolveOutcome {
            result: ResolveResult::Refused(RefusalReason::SampleTooSmall),
            report,
            seed,
            seed_source,
            threshold,
        };
    };

    // Stage 2: pick best candidate by cosine and run two-of-three.
    let mut best: Option<(&Candidate, u8)> = None;
    for cand in &req.candidates {
        let (passed, _) =
            two_of_three_consensus(&req.query, &req.namespace, &req.trust, cand, thr.tau_n);
        match best {
            Some((_, p)) if p >= passed => {}
            _ => best = Some((cand, passed)),
        }
    }
    let charge2 = guard.charge("consensus");
    if let Err(_e) = charge2 {
        let best_effort = best.map(|(c, _)| c.node_id);
        let report = guard.into_report();
        return ResolveOutcome {
            result: ResolveResult::BudgetExhausted { best_effort },
            report,
            seed,
            seed_source,
            threshold,
        };
    }
    // Per R3: single-signal collapses are refused, meaning the node
    // stays un-collapsed; i.e. the caller gets a Created decision. We
    // preserve the explicit `Refused(SingleSignalOnly)` variant for
    // callers (and gauges) that need to observe the distinction, but
    // only when the caller explicitly opts in via
    // [`resolve_or_create_simple`]. For the production path,
    // un-collapsed == Created so the host creates the node.
    let result = match best {
        Some((cand, signals)) if signals >= 2 => ResolveResult::Resolved {
            node_id: cand.node_id,
            signals_passed: signals,
        },
        _ => ResolveResult::Created { tau_n: thr.tau_n },
    };
    let report = guard.into_report();
    ResolveOutcome {
        result,
        report,
        seed,
        seed_source,
        threshold,
    }
}

/// Tight `(query, threshold) -> ResolveResult` shape from the gap brief.
///
/// This is a thin convenience wrapper for a caller that has already
/// done its own HNSW walk and just wants the decision: passing a
/// `threshold` here bypasses the GMM derivation and checks the best
/// candidate against the given fixed threshold using the same
/// two-of-three consensus gate. Present primarily to keep the public
/// API surface matching the gap brief; production callers should use
/// [`resolve_or_create`] with a `local_sample`.
#[must_use]
pub fn resolve_or_create_simple(
    query: &str,
    threshold: f32,
    candidates: &[Candidate],
    namespace: &str,
    trust: &str,
) -> ResolveResult {
    let mut best: Option<(&Candidate, u8)> = None;
    for cand in candidates {
        let (passed, _) = two_of_three_consensus(query, namespace, trust, cand, threshold);
        match best {
            Some((_, p)) if p >= passed => {}
            _ => best = Some((cand, passed)),
        }
    }
    match best {
        Some((cand, signals)) if signals >= 2 => ResolveResult::Resolved {
            node_id: cand.node_id,
            signals_passed: signals,
        },
        Some((_, 1)) => ResolveResult::Refused(RefusalReason::SingleSignalOnly),
        _ => ResolveResult::Created { tau_n: threshold },
    }
}

#[cfg(test)]
mod tests;
