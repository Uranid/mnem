//! Trust-boundary gate for opt-in typed-relation inference.
//!
//! This module implements the adversarial trust model described in
//! `research/gap-catalog/03-typed-relation-inference/solution.md`
//! (Round 3). It is the mandatory admission check between a
//! candidate inferred edge and any downstream consumer (PPR, multihop
//! traversal, retrieve) that might amplify the edge's weight in a
//! ranking signal.
//!
//! # Design
//!
//! The gate is intentionally tiny: a single `admit` function over a
//! [`TrustBoundary`] policy and a [`Candidate`] descriptor. No policy
//! defaults are magic: every floor must be constructed explicitly
//! from `InferenceBudget`-derived or spec-pinned values (floor-c). The
//! caller is expected to plumb the same floor through gauges so the
//! runtime view matches the code.
//!
//! # Determinism
//!
//! `admit` is a pure function of its inputs. No clocks, no randomness,
//! no global state - safe to call inside property tests and inside
//! the deterministic ingest pipeline.
//!
//! # Rate-limit fingerprint
//!
//! For the per-author token-bucket rate limiter in
//! [`AuthorFingerprint`], the author identifier is hashed with
//! SHA-256 and truncated to the first 8 bytes. This is documented in
//! solution.md R3 §4: *no PII, forensics-ready*. The truncation
//! boundary is 8 bytes (64 bits) so that collisions are vanishingly
//! unlikely for real-world author-population sizes but still too
//! short to recover the original id.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Default PPR-amplification confidence floor - solution.md R3 §3.
///
/// Reference: Tong 2008 §4 PPR-skip-risk envelope. Floor-c tunable
/// backed by gauge `mnem_ppr_amplification_floor` and proptest
/// [`proptests::admit_rejects_below_confidence_floor`].
pub const PPR_AMPLIFICATION_FLOOR: f32 = 0.75;

/// Truncation length for the SHA-256 author fingerprint, in bytes.
///
/// 8 bytes = 64 bits. See module docs for rationale.
pub const AUTHOR_FINGERPRINT_BYTES: usize = 8;

/// Width of the rate-limit rolling window, in seconds.
///
/// Fixed at 60s per solution.md R3 §4 ("rolling 1-min window").
pub const AUTHOR_RATE_LIMIT_WINDOW_SECS: u64 = 60;

/// Opt-in policy for admitting an inferred typed relation into a
/// downstream ranking signal.
///
/// All fields are explicit: there is no `Default` because every caller
/// MUST set a floor that matches its own gauge and proptest. The
/// library-provided constant [`PPR_AMPLIFICATION_FLOOR`] is the
/// solution.md-pinned value when the caller is PPR.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrustBoundary {
    /// Minimum confidence for admission. Candidates strictly below
    /// this value are rejected.
    pub confidence_floor: f32,

    /// Whether the downstream caller has explicitly opted into
    /// inferred relations. When `false`, every candidate is rejected
    /// regardless of confidence. Mirrors the `opt_in` flag on the
    /// `ProvenanceTag::InferredRelation` variant (solution.md R3 §2).
    pub consumer_opt_in: bool,
}

impl TrustBoundary {
    /// Construct a new trust boundary.
    ///
    /// # Errors
    ///
    /// Returns `None` when `confidence_floor` is NaN or outside
    /// `[0.0, 1.0]`. Callers that want the spec floor can use
    /// [`TrustBoundary::ppr_default`] instead.
    #[must_use]
    pub fn new(confidence_floor: f32, consumer_opt_in: bool) -> Option<Self> {
        if !confidence_floor.is_finite() || !(0.0..=1.0).contains(&confidence_floor) {
            return None;
        }
        Some(Self {
            confidence_floor,
            consumer_opt_in,
        })
    }

    /// Construct the spec-pinned PPR trust boundary
    /// ([`PPR_AMPLIFICATION_FLOOR`]) with the caller's opt-in flag.
    #[must_use]
    pub fn ppr_default(consumer_opt_in: bool) -> Self {
        Self {
            confidence_floor: PPR_AMPLIFICATION_FLOOR,
            consumer_opt_in,
        }
    }

    /// Decide whether `candidate` is admissible under this policy.
    ///
    /// The function is a total, side-effect-free predicate over
    /// `(self, candidate)`. It rejects on any of:
    ///
    /// 1. `consumer_opt_in == false`.
    /// 2. `candidate.confidence` is NaN / non-finite.
    /// 3. `candidate.confidence < self.confidence_floor`.
    /// 4. `candidate.opt_in == false`.
    #[must_use]
    pub fn admit(&self, candidate: &Candidate) -> bool {
        if !self.consumer_opt_in {
            return false;
        }
        if !candidate.opt_in {
            return false;
        }
        if !candidate.confidence.is_finite() {
            return false;
        }
        candidate.confidence >= self.confidence_floor
    }
}

/// A candidate typed relation awaiting admission.
///
/// Contains only the fields the trust gate needs; the full
/// `TypedRelation` payload is kept in the inference module so that
/// this file stays free of clustering-shape types.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    /// Clustering-assigned confidence in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Whether the producer marked this edge opt-in (solution.md R3
    /// §2 `ProvenanceTag::InferredRelation { opt_in: true }`).
    pub opt_in: bool,
}

/// Truncated SHA-256 fingerprint of an author identifier.
///
/// Produced by [`AuthorFingerprint::from_author_id`]. The raw bytes
/// are exposed as a hex string via [`AuthorFingerprint::as_hex`] for
/// the metric `mnem_infer_author_ratelimit_hits_total{author_fingerprint}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AuthorFingerprint([u8; AUTHOR_FINGERPRINT_BYTES]);

impl AuthorFingerprint {
    /// Hash `author_id` with SHA-256 and truncate to
    /// [`AUTHOR_FINGERPRINT_BYTES`] bytes.
    #[must_use]
    pub fn from_author_id(author_id: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(author_id.as_bytes());
        let digest = hasher.finalize();
        let mut out = [0u8; AUTHOR_FINGERPRINT_BYTES];
        out.copy_from_slice(&digest[..AUTHOR_FINGERPRINT_BYTES]);
        Self(out)
    }

    /// Raw fingerprint bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; AUTHOR_FINGERPRINT_BYTES] {
        &self.0
    }

    /// Lowercase hex encoding, suitable for metric labels.
    #[must_use]
    pub fn as_hex(&self) -> String {
        let mut s = String::with_capacity(AUTHOR_FINGERPRINT_BYTES * 2);
        for b in &self.0 {
            use std::fmt::Write as _;
            let _ = write!(s, "{b:02x}");
        }
        s
    }
}

/// Rolling per-author token-bucket rate limiter over a 1-minute
/// window (solution.md R3 §4).
///
/// The bucket tracks `(fingerprint, window_start_secs, count)`
/// tuples. `window_start_secs` is a monotonic clock value supplied by
/// the caller - the limiter has no global clock access, keeping it
/// deterministic under test. When the caller advances the window
/// ([`AuthorRateLimiter::tick`]) beyond
/// [`AUTHOR_RATE_LIMIT_WINDOW_SECS`], counts reset to zero.
#[derive(Debug, Clone)]
pub struct AuthorRateLimiter {
    per_commit_cap: u32,
    window_start_secs: u64,
    buckets: std::collections::HashMap<AuthorFingerprint, u32>,
}

impl AuthorRateLimiter {
    /// Create a new limiter with `per_commit_cap` phrases per
    /// `(author, 1-minute window)` pair.
    ///
    /// `per_commit_cap` comes from
    /// `InferenceBudget::author_rate_limit_per_commit` (corpus-
    /// derived, not magic).
    #[must_use]
    pub fn new(per_commit_cap: u32, now_secs: u64) -> Self {
        Self {
            per_commit_cap,
            window_start_secs: now_secs,
            buckets: std::collections::HashMap::new(),
        }
    }

    /// Advance the monotonic window clock. When `now_secs` is more
    /// than [`AUTHOR_RATE_LIMIT_WINDOW_SECS`] past the current window
    /// start, all buckets reset and the window rolls forward.
    pub fn tick(&mut self, now_secs: u64) {
        if now_secs >= self.window_start_secs
            && now_secs - self.window_start_secs >= AUTHOR_RATE_LIMIT_WINDOW_SECS
        {
            self.buckets.clear();
            self.window_start_secs = now_secs;
        }
    }

    /// Try to admit one phrase for `author`.
    ///
    /// Returns `true` when the bucket is under the cap (and increments
    /// the counter), `false` when the cap is reached. Callers should
    /// drop the phrase from Leiden input on `false` and emit the
    /// `mnem_infer_author_ratelimit_hits_total` counter.
    pub fn admit(&mut self, author: &AuthorFingerprint) -> bool {
        let counter = self.buckets.entry(*author).or_insert(0);
        if *counter >= self.per_commit_cap {
            return false;
        }
        *counter += 1;
        true
    }

    /// Read-only count for `author` in the current window.
    #[must_use]
    pub fn count(&self, author: &AuthorFingerprint) -> u32 {
        self.buckets.get(author).copied().unwrap_or(0)
    }

    /// Current window start (monotonic seconds, caller-supplied).
    #[must_use]
    pub fn window_start_secs(&self) -> u64 {
        self.window_start_secs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admit_rejects_when_consumer_not_opted_in() {
        let tb = TrustBoundary::new(0.5, false).unwrap();
        let c = Candidate {
            confidence: 0.99,
            opt_in: true,
        };
        assert!(!tb.admit(&c));
    }

    #[test]
    fn admit_rejects_when_producer_did_not_opt_in() {
        let tb = TrustBoundary::new(0.5, true).unwrap();
        let c = Candidate {
            confidence: 0.99,
            opt_in: false,
        };
        assert!(!tb.admit(&c));
    }

    #[test]
    fn admit_rejects_below_confidence_floor() {
        let tb = TrustBoundary::new(0.75, true).unwrap();
        let c = Candidate {
            confidence: 0.7499,
            opt_in: true,
        };
        assert!(!tb.admit(&c));
    }

    #[test]
    fn admit_accepts_at_and_above_floor() {
        let tb = TrustBoundary::new(0.75, true).unwrap();
        assert!(tb.admit(&Candidate {
            confidence: 0.75,
            opt_in: true
        }));
        assert!(tb.admit(&Candidate {
            confidence: 0.99,
            opt_in: true
        }));
    }

    #[test]
    fn admit_rejects_nan_confidence() {
        let tb = TrustBoundary::new(0.5, true).unwrap();
        assert!(!tb.admit(&Candidate {
            confidence: f32::NAN,
            opt_in: true
        }));
    }

    #[test]
    fn new_rejects_out_of_range_floor() {
        assert!(TrustBoundary::new(-0.1, true).is_none());
        assert!(TrustBoundary::new(1.1, true).is_none());
        assert!(TrustBoundary::new(f32::NAN, true).is_none());
    }

    #[test]
    fn ppr_default_uses_spec_pinned_floor() {
        let tb = TrustBoundary::ppr_default(true);
        assert!((tb.confidence_floor - PPR_AMPLIFICATION_FLOOR).abs() < f32::EPSILON);
    }

    #[test]
    fn fingerprint_is_deterministic_and_truncated() {
        let a = AuthorFingerprint::from_author_id("alice@example.com");
        let b = AuthorFingerprint::from_author_id("alice@example.com");
        assert_eq!(a, b);
        assert_eq!(a.as_bytes().len(), AUTHOR_FINGERPRINT_BYTES);
        assert_eq!(a.as_hex().len(), AUTHOR_FINGERPRINT_BYTES * 2);
    }

    #[test]
    fn fingerprint_distinguishes_distinct_authors() {
        let a = AuthorFingerprint::from_author_id("alice");
        let b = AuthorFingerprint::from_author_id("bob");
        assert_ne!(a, b);
    }

    #[test]
    fn rate_limiter_admits_up_to_cap_then_rejects() {
        let author = AuthorFingerprint::from_author_id("author-x");
        let mut rl = AuthorRateLimiter::new(3, 0);
        assert!(rl.admit(&author));
        assert!(rl.admit(&author));
        assert!(rl.admit(&author));
        assert!(!rl.admit(&author));
        assert_eq!(rl.count(&author), 3);
    }

    #[test]
    fn rate_limiter_resets_after_window_elapses() {
        let author = AuthorFingerprint::from_author_id("author-x");
        let mut rl = AuthorRateLimiter::new(2, 0);
        assert!(rl.admit(&author));
        assert!(rl.admit(&author));
        assert!(!rl.admit(&author));
        rl.tick(AUTHOR_RATE_LIMIT_WINDOW_SECS);
        assert!(rl.admit(&author));
        assert_eq!(rl.window_start_secs(), AUTHOR_RATE_LIMIT_WINDOW_SECS);
    }

    #[test]
    fn rate_limiter_does_not_reset_within_window() {
        let author = AuthorFingerprint::from_author_id("author-x");
        let mut rl = AuthorRateLimiter::new(2, 0);
        assert!(rl.admit(&author));
        assert!(rl.admit(&author));
        rl.tick(AUTHOR_RATE_LIMIT_WINDOW_SECS - 1);
        assert!(!rl.admit(&author));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Floor-c proptest: no candidate strictly below the floor is
        /// ever admitted. Backs gauge `mnem_ppr_amplification_floor`.
        #[test]
        fn admit_rejects_below_confidence_floor(
            floor in 0.0f32..=1.0,
            delta in 0.0001f32..0.5,
            opt_in in any::<bool>(),
        ) {
            let tb = TrustBoundary::new(floor, true).unwrap();
            let conf = (floor - delta).max(0.0);
            if conf < floor {
                let c = Candidate { confidence: conf, opt_in };
                prop_assert!(!tb.admit(&c));
            }
        }

        /// Floor-c proptest: every candidate at or above floor with
        /// both opt-ins set is admitted.
        #[test]
        fn admit_accepts_above_floor_with_opt_in(
            floor in 0.0f32..=1.0,
            above in 0.0f32..=0.5,
        ) {
            let tb = TrustBoundary::new(floor, true).unwrap();
            let conf = (floor + above).min(1.0);
            let c = Candidate { confidence: conf, opt_in: true };
            prop_assert!(tb.admit(&c));
        }

        /// SHA-256 fingerprint is collision-stable across repeated
        /// calls (determinism proptest).
        #[test]
        fn fingerprint_stable_across_calls(s in "[a-zA-Z0-9@._-]{1,64}") {
            let a = AuthorFingerprint::from_author_id(&s);
            let b = AuthorFingerprint::from_author_id(&s);
            prop_assert_eq!(a, b);
        }
    }
}
