//! Shared commit-budget guard primitive (gap-catalog shared/commit-budget-guard).
//!
//! `CommitBudgetGuard` wraps a commit's variable-cost stages
//! (canonicalization, Leiden recompute, kNN insert batch) with a single
//! wall-clock budget envelope. Stages are `charge()`-d as they complete;
//! once the running elapsed exceeds `budget_ms` the guard returns
//! [`Decision::ShouldDefer`] so callers can push tail work to the next
//! commit deterministically, and once it exceeds `hard_wall_ms` the
//! guard returns [`HardWallExceeded`] so callers abort.
//!
//! # Determinism
//!
//! - Construction captures `commit_cid` so replay is reproducible: a
//!   commit replay re-enters with the same CID and the envelope-stored
//!   `deferred_stages` tells the replayer which stages to skip.
//! - `Decision::ShouldDefer` vs `Decision::Proceed` is timing-dependent
//!   in live mode, but the commit envelope records `deferred_stages`;
//!   replay reads the envelope and skips those stages.
//! - `HardWallExceeded` is an *error*, not a deferral: caller aborts.
//!
//! # Emitted metrics (drop / `into_report`)
//!
//! - `mnem_commit_budget_elapsed_ms{tag}` histogram.
//! - `mnem_commit_budget_breached_total{tag}` counter (soft breach).
//! - `mnem_commit_hard_wall_hit_total{tag}` counter.
//! - `mnem_commit_deferred_stages_total{tag,stage}` counter.
//!
//! This module is deliberately no-I/O: metrics are exposed through the
//! returned [`CommitBudgetReport`] so the host runtime wires them into
//! its own counter sink (prometheus, OTel, etc.). `mnem-core` stays
//! terminal-free per `lib.rs` invariants.

use std::time::{Duration, Instant};

use crate::id::Cid;

/// Shared wall-clock budget envelope for a commit's variable-cost stages.
///
/// See the module docs for wiring semantics and determinism notes.
#[derive(Debug)]
pub struct CommitBudgetGuard {
    /// Caller tag (used as metric label). Pass the stable
    /// gap-shorthand, e.g. `"gap-04-resolve-or-create"`.
    pub tag: &'static str,
    /// Monotonic clock anchor set at construction.
    pub start: Instant,
    /// Soft budget in milliseconds. Exceeding it yields
    /// [`Decision::ShouldDefer`].
    pub budget_ms: u32,
    /// Hard wall in milliseconds. Exceeding it yields
    /// [`HardWallExceeded`] and aborts the commit.
    pub hard_wall_ms: u32,
    /// CID of the commit this guard is embedded in, stored in the
    /// envelope for replay determinism.
    pub commit_cid: Cid,
    /// Stages deferred to the next commit (pushed by [`Self::defer`]).
    pub deferred: Vec<&'static str>,
    /// Stages charged so far; order-preserving for the envelope.
    pub charged: Vec<(&'static str, u32)>,
    /// Sticky flag: true once any `charge()` returns `ShouldDefer`.
    /// The envelope carries this so replay can short-circuit.
    pub breached: bool,
    /// Sticky flag: true once any `charge()` returns `HardWallExceeded`.
    pub hard_wall_hit: bool,
}

/// Outcome of a successful `charge()` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Running elapsed is under the soft budget - continue.
    Proceed,
    /// Running elapsed exceeds the soft budget but is under the hard
    /// wall. Caller should `.defer()` the remaining tail stages.
    ShouldDefer,
}

/// Error returned by `charge()` when the hard wall is exceeded.
///
/// This is a structural abort signal: the commit MUST unwind. Callers
/// should log, increment the hard-wall counter (done automatically via
/// the report), and return the commit's abort outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HardWallExceeded {
    /// Elapsed time at the moment of breach, in milliseconds.
    pub elapsed_ms: u32,
    /// The hard wall that was exceeded, in milliseconds.
    pub hard_wall_ms: u32,
}

impl core::fmt::Display for HardWallExceeded {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "commit-budget hard wall exceeded: elapsed {}ms > wall {}ms",
            self.elapsed_ms, self.hard_wall_ms
        )
    }
}

impl std::error::Error for HardWallExceeded {}

/// Report snapshot produced by [`CommitBudgetGuard::into_report`].
///
/// Callers embed this in their commit envelope and also feed it to the
/// host's metric sink.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitBudgetReport {
    /// Caller tag, e.g. `"gap-04-resolve-or-create"`.
    pub tag: &'static str,
    /// Total wall-clock elapsed, in milliseconds.
    pub elapsed_ms: u32,
    /// Soft budget that was in effect.
    pub budget_ms: u32,
    /// Hard wall that was in effect.
    pub hard_wall_ms: u32,
    /// True iff the soft budget was breached (but hard wall wasn't hit).
    pub breached: bool,
    /// True iff the hard wall was hit (commit aborted).
    pub hard_wall_hit: bool,
    /// Stages pushed to the deferred queue for next commit.
    pub deferred_stages: Vec<&'static str>,
    /// Ordered list of (stage, elapsed_ms_at_charge).
    pub charged_stages: Vec<(&'static str, u32)>,
}

impl CommitBudgetGuard {
    /// Start a new guard. `hard_wall_ms >= budget_ms` is enforced by
    /// clamping hard_wall to at least budget; an explicit violation
    /// would be a caller-bug but we accept the floor silently.
    #[must_use]
    pub fn start(tag: &'static str, budget_ms: u32, hard_wall_ms: u32, commit_cid: Cid) -> Self {
        Self {
            tag,
            start: Instant::now(),
            budget_ms,
            hard_wall_ms: hard_wall_ms.max(budget_ms),
            commit_cid,
            deferred: Vec::new(),
            charged: Vec::new(),
            breached: false,
            hard_wall_hit: false,
        }
    }

    /// Elapsed milliseconds since construction, saturated to `u32::MAX`.
    #[must_use]
    pub fn elapsed_ms(&self) -> u32 {
        u32::try_from(self.start.elapsed().as_millis()).unwrap_or(u32::MAX)
    }

    /// Charge the running elapsed after a completed stage.
    ///
    /// Returns `Err(HardWallExceeded)` if the elapsed exceeds
    /// `hard_wall_ms` - caller aborts. Returns
    /// `Ok(Decision::ShouldDefer)` if it exceeds `budget_ms` but not
    /// the hard wall - caller should stop doing new work and `defer`
    /// the remaining stages. Returns `Ok(Decision::Proceed)` otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`HardWallExceeded`] when the running elapsed would
    /// exceed `hard_wall_ms`.
    pub fn charge(&mut self, stage: &'static str) -> Result<Decision, HardWallExceeded> {
        let elapsed = self.elapsed_ms();
        self.charged.push((stage, elapsed));
        if elapsed > self.hard_wall_ms {
            self.hard_wall_hit = true;
            return Err(HardWallExceeded {
                elapsed_ms: elapsed,
                hard_wall_ms: self.hard_wall_ms,
            });
        }
        if elapsed > self.budget_ms {
            self.breached = true;
            return Ok(Decision::ShouldDefer);
        }
        Ok(Decision::Proceed)
    }

    /// Charge using an externally-provided elapsed value (used by
    /// deterministic proptests that need a synthetic clock).
    ///
    /// # Errors
    ///
    /// Returns [`HardWallExceeded`] when `elapsed_ms > hard_wall_ms`.
    #[doc(hidden)]
    pub fn charge_with(
        &mut self,
        stage: &'static str,
        elapsed_ms: u32,
    ) -> Result<Decision, HardWallExceeded> {
        self.charged.push((stage, elapsed_ms));
        if elapsed_ms > self.hard_wall_ms {
            self.hard_wall_hit = true;
            return Err(HardWallExceeded {
                elapsed_ms,
                hard_wall_ms: self.hard_wall_ms,
            });
        }
        if elapsed_ms > self.budget_ms {
            self.breached = true;
            return Ok(Decision::ShouldDefer);
        }
        Ok(Decision::Proceed)
    }

    /// Push a stage onto the deferred queue for the next commit.
    pub fn defer(&mut self, stage: &'static str) {
        self.deferred.push(stage);
    }

    /// Freeze the guard into a report for the commit envelope.
    #[must_use]
    pub fn into_report(self) -> CommitBudgetReport {
        CommitBudgetReport {
            tag: self.tag,
            elapsed_ms: u32::try_from(self.start.elapsed().as_millis()).unwrap_or(u32::MAX),
            budget_ms: self.budget_ms,
            hard_wall_ms: self.hard_wall_ms,
            breached: self.breached,
            hard_wall_hit: self.hard_wall_hit,
            deferred_stages: self.deferred,
            charged_stages: self.charged,
        }
    }
}

/// Sleep helper used only by tests in this module / downstream callers
/// that need to synthesise wall-clock delays without importing
/// `std::thread::sleep` at every call site.
#[doc(hidden)]
#[must_use]
pub fn since(start: Instant) -> Duration {
    start.elapsed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::Multihash;

    fn zero_cid() -> Cid {
        Cid::new(
            crate::id::CODEC_RAW,
            Multihash::wrap(crate::id::HASH_BLAKE3_256, &[0u8; 32]).expect("32-byte digest"),
        )
    }

    #[test]
    fn charge_under_budget_proceeds() {
        let mut g = CommitBudgetGuard::start("test", 100, 200, zero_cid());
        let d = g.charge_with("stage_a", 50).unwrap();
        assert_eq!(d, Decision::Proceed);
        assert!(!g.breached);
    }

    #[test]
    fn charge_over_budget_defers() {
        let mut g = CommitBudgetGuard::start("test", 50, 200, zero_cid());
        let d = g.charge_with("stage_a", 75).unwrap();
        assert_eq!(d, Decision::ShouldDefer);
        assert!(g.breached);
    }

    #[test]
    fn charge_over_hard_wall_aborts() {
        let mut g = CommitBudgetGuard::start("test", 50, 100, zero_cid());
        let err = g.charge_with("stage_a", 150).unwrap_err();
        assert_eq!(err.elapsed_ms, 150);
        assert_eq!(err.hard_wall_ms, 100);
        assert!(g.hard_wall_hit);
    }

    #[test]
    fn hard_wall_clamped_to_at_least_budget() {
        // caller bug: hard_wall < budget. We clamp.
        let g = CommitBudgetGuard::start("test", 100, 50, zero_cid());
        assert_eq!(g.hard_wall_ms, 100);
    }

    #[test]
    fn report_records_charged_and_deferred() {
        let mut g = CommitBudgetGuard::start("test", 50, 200, zero_cid());
        let _ = g.charge_with("a", 10).unwrap();
        let _ = g.charge_with("b", 80).unwrap(); // defer
        g.defer("c");
        g.defer("d");
        let rep = g.into_report();
        assert_eq!(rep.tag, "test");
        assert!(rep.breached);
        assert!(!rep.hard_wall_hit);
        assert_eq!(rep.deferred_stages, vec!["c", "d"]);
        assert_eq!(rep.charged_stages.len(), 2);
        assert_eq!(rep.charged_stages[0].0, "a");
        assert_eq!(rep.charged_stages[1].0, "b");
    }
}
