//! Wire-level protocol constants and capability vocabulary for mnem's
//! remote transport.
//!
//! This module is the freeze-point for the on-wire handshake. Nothing
//! here knows about HTTP, TLS, tokens, or framing; those live in the
//! server crate. The types here are the common language two mnem peers
//! agree on before the first byte of a CAR body hits the socket.
//!
//! ## The five primitives frozen in PR 2
//!
//! | Name | Purpose |
//! |---|---|
//! | [`PROTOCOL_VERSION`] | Single `u32` that every request and response MUST advertise. Bumping this is a breaking change. |
//! | [`PROTOCOL_HEADER`] | The HTTP header name (`mnem-protocol`) carrying the version. |
//! | [`CAPABILITIES_HEADER`] | The HTTP header name (`mnem-capabilities`) echoing the agreed capability set. |
//! | [`Capability`] | Enum of every capability string this codebase knows. Unknown capabilities MUST be tolerated on read. |
//! | [`parse_capabilities`] / [`serialize_capabilities`] | Free functions for the comma-separated wire form. |
//!
//! Capabilities are the exact Git-v2 lesson: if you ship a protocol
//! without a capability ad, every feature flag becomes a version bump.
//! With an ad, v0 servers and v1 servers can share a wire as long as
//! they agree on the intersection of their advertised sets.

// Pedantic doc-length warnings on the module-level doc paragraphs
// are opinionated; the design prose is deliberately verbose.
#![allow(clippy::too_long_first_doc_paragraph, clippy::missing_const_for_fn)]

use std::collections::BTreeSet;
use std::fmt;
use std::str::FromStr;

/// Frozen wire-protocol version for the mnem remote transport.
///
/// Every request and every response MUST carry a [`PROTOCOL_HEADER`]
/// header whose value parses to this integer. A server that receives a
/// version it does not implement MUST reject with HTTP 400 and JSON
/// `{"error": "protocol-version"}`; a client that receives an
/// unexpected version in a response MUST treat the response as
/// malformed.
///
/// Bumping this constant is a breaking change.
pub const PROTOCOL_VERSION: u32 = 1;

/// Canonical HTTP header name carrying the protocol version on every
/// request and response of the remote transport.
pub const PROTOCOL_HEADER: &str = "mnem-protocol";

/// Canonical HTTP header name carrying the comma-separated list of
/// capabilities a response was produced under. Clients echo this on
/// requests that want to force a downgrade (e.g. ignore `delta-fetch`)
/// and servers echo it on responses so clients can detect a silent
/// downgrade mid-session.
pub const CAPABILITIES_HEADER: &str = "mnem-capabilities";

/// Capabilities the mnem remote transport knows about.
///
/// Capabilities are stable kebab-case strings on the wire
/// (`have-set-bloom`, `push-negotiate`, ...). New variants MAY be added
/// in minor versions; they MUST be additive. Unknown capability
/// strings MUST be tolerated on read - every parser here and
/// downstream returns `None` rather than failing on an unrecognised
/// value.
///
/// Roadmap capabilities (documented but not yet used by any code path)
/// are included so that clients can begin advertising them before the
/// server-side implementation lands.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum Capability {
    /// Client advertises a bloom-filter have-set. Gates the
    /// `have-set` field in `fetch-blocks` and `push-blocks` requests.
    /// PR 2 ships the [`crate::have_set::BloomHaveSet`] reference
    /// implementation of the serialised shape.
    HaveSetBloom,
    /// Client advertises a range-based set-reconciliation have-set.
    /// Reserved for v0.2; mnem-transport v0.1.0 does not emit these.
    /// §"Roadmap".
    HaveSetRbsr,
    /// Client and server support the push-side have-set negotiation
    /// (`POST /remote/v1/push-negotiate`). Without this capability,
    /// `push-blocks` ships every reachable block from the new head.
    /// Reserved for v0.2.
    PushNegotiate,
    /// Client and server support the partial-fetch filter language
    /// (`filter: { "embed": "omit" }` etc.) in `fetch-blocks` and
    /// `push-blocks`. Reserved for v0.2. §"Roadmap".
    FilterSpec,
    /// Client and server support batching `push-blocks` +
    /// `advance-head` in a single all-or-nothing request. Lands with
    /// PR 3.
    AtomicPush,
    /// Server requires Ed25519-signed commits on every push and will
    /// return `signature-invalid` / `signer-revoked` / `policy-require`
    /// rejection reasons. Lands with PR 4.
    SignedPush,
    /// Server publishes a self-certifying repo identifier
    /// (BLAKE3 of the root-op signing key) on `GET /remote/v1/refs`,
    /// so that deep links from signed commits do not need DNS trust.
    /// Reserved for v0.1.0.
    SelfCertifyingRepoId,
}

impl Capability {
    /// Stable kebab-case wire name for this capability.
    ///
    /// This is the single source of truth for what goes on the wire;
    /// [`Capability::from_str`] is the inverse.
    #[must_use]
    pub const fn as_wire_str(&self) -> &'static str {
        match self {
            Self::HaveSetBloom => "have-set-bloom",
            Self::HaveSetRbsr => "have-set-rbsr",
            Self::PushNegotiate => "push-negotiate",
            Self::FilterSpec => "filter-spec",
            Self::AtomicPush => "atomic-push",
            Self::SignedPush => "signed-push",
            Self::SelfCertifyingRepoId => "self-certifying-repo-id",
        }
    }

    /// Every capability known to this build, in a stable order
    /// (wire-string ascending). Useful for tests and for server-side
    /// capability ads when the operator has not opted out of any.
    #[must_use]
    pub fn all() -> &'static [Self] {
        // Keep this list in sync with `as_wire_str`. Ordered
        // wire-string ascending so comparisons against sorted wire
        // output are trivially stable.
        const ALL: &[Capability] = &[
            Capability::AtomicPush,
            Capability::FilterSpec,
            Capability::HaveSetBloom,
            Capability::HaveSetRbsr,
            Capability::PushNegotiate,
            Capability::SelfCertifyingRepoId,
            Capability::SignedPush,
        ];
        ALL
    }
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_wire_str())
    }
}

impl FromStr for Capability {
    type Err = UnknownCapability;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "have-set-bloom" => Ok(Self::HaveSetBloom),
            "have-set-rbsr" => Ok(Self::HaveSetRbsr),
            "push-negotiate" => Ok(Self::PushNegotiate),
            "filter-spec" => Ok(Self::FilterSpec),
            "atomic-push" => Ok(Self::AtomicPush),
            "signed-push" => Ok(Self::SignedPush),
            "self-certifying-repo-id" => Ok(Self::SelfCertifyingRepoId),
            _ => Err(UnknownCapability(s.to_owned())),
        }
    }
}

/// Serde support: capabilities serialise as their stable kebab-case
/// wire string; unknown strings produce a deserialisation error, so
/// untyped consumers that want forward-compat MUST carry the raw string
/// alongside (see [`parse_capabilities`] which tolerates unknowns).
impl serde::Serialize for Capability {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_wire_str())
    }
}

impl<'de> serde::Deserialize<'de> for Capability {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = <&str as serde::Deserialize>::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// Raised by [`Capability::from_str`] when the wire string is not in
/// this build's vocabulary. Callers that want forward-compat use
/// [`parse_capabilities`] instead, which silently drops unknowns.
#[derive(Debug, thiserror::Error)]
#[error("unknown capability: {0}")]
pub struct UnknownCapability(pub String);

/// Parse a comma-separated capability list off the wire, discarding
/// any unknown capabilities (forward-compat). Whitespace around
/// commas is tolerated. Empty entries are skipped.
///
/// Returns a sorted-unique [`BTreeSet`] so intersection and
/// set-difference against the server's capability ad are cheap.
#[must_use]
pub fn parse_capabilities(s: &str) -> BTreeSet<Capability> {
    s.split(',')
        .map(str::trim)
        .filter(|tok| !tok.is_empty())
        .filter_map(|tok| tok.parse::<Capability>().ok())
        .collect()
}

/// Serialize a capability set to the comma-separated wire form. The
/// output is deterministic: capabilities are emitted in the `all()`
/// order, i.e. wire-string ascending.
#[must_use]
pub fn serialize_capabilities<I>(caps: I) -> String
where
    I: IntoIterator<Item = Capability>,
{
    let set: BTreeSet<Capability> = caps.into_iter().collect();
    let mut out = String::new();
    let mut first = true;
    // Iterate in ascending wire-string order for determinism.
    let mut sorted: Vec<Capability> = set.into_iter().collect();
    sorted.sort_by_key(Capability::as_wire_str);
    for c in sorted {
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(c.as_wire_str());
    }
    out
}

/// A set of capabilities agreed between two peers.
///
/// Thin wrapper around `BTreeSet<Capability>` whose only job is to
/// name the [`CapabilitySet::intersect`] operation that both clients
/// and servers perform on handshake: each side advertises the
/// capabilities it supports, and the *intersection* is the set both
/// agree to use for the rest of the session.
///
/// This type stays pure-data (no network, no HTTP) so both
/// `mnem-http` and `mnem-transport::client` can consume it.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CapabilitySet(BTreeSet<Capability>);

impl CapabilitySet {
    /// Empty capability set.
    #[must_use]
    pub fn new() -> Self {
        Self(BTreeSet::new())
    }

    /// Build from any capability iterator. Duplicates are collapsed.
    #[must_use]
    pub fn with_caps<I: IntoIterator<Item = Capability>>(caps: I) -> Self {
        Self(caps.into_iter().collect())
    }

    /// Every capability this build knows about. Equivalent to
    /// `with_caps(Capability::all().iter().copied())` but allocates
    /// once rather than walking a slice.
    #[must_use]
    pub fn all_known() -> Self {
        Self(Capability::all().iter().copied().collect())
    }

    /// Parse from the comma-separated wire form; unknown entries are
    /// dropped (forward-compat, same rule as [`parse_capabilities`]).
    #[must_use]
    pub fn parse(s: &str) -> Self {
        Self(parse_capabilities(s))
    }

    /// Serialize to the comma-separated wire form, ascending order.
    #[must_use]
    pub fn serialize(&self) -> String {
        serialize_capabilities(self.0.iter().copied())
    }

    /// Capability intersection: the set of capabilities both peers
    /// advertised. This is the agreed-upon feature set for the
    /// remainder of the session.
    ///
    /// ```text
    /// intersect(A, B) = { c | c in A and c in B }
    /// ```
    #[must_use]
    pub fn intersect(&self, other: &Self) -> Self {
        Self(self.0.intersection(&other.0).copied().collect())
    }

    /// True if the given capability is in this set.
    #[must_use]
    pub fn contains(&self, cap: Capability) -> bool {
        self.0.contains(&cap)
    }

    /// Number of capabilities in the set.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// True iff the set has no capabilities.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Borrow the underlying sorted set.
    #[must_use]
    pub const fn as_set(&self) -> &BTreeSet<Capability> {
        &self.0
    }
}

impl From<BTreeSet<Capability>> for CapabilitySet {
    fn from(s: BTreeSet<Capability>) -> Self {
        Self(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_version_is_frozen() {
        assert_eq!(PROTOCOL_VERSION, 1);
        assert_eq!(PROTOCOL_HEADER, "mnem-protocol");
        assert_eq!(CAPABILITIES_HEADER, "mnem-capabilities");
    }

    #[test]
    fn capability_wire_strings_are_stable_kebab_case() {
        // This test pins the wire format. Changing any string here is a
        // breaking change and requires bumping PROTOCOL_VERSION.
        assert_eq!(Capability::HaveSetBloom.as_wire_str(), "have-set-bloom");
        assert_eq!(Capability::HaveSetRbsr.as_wire_str(), "have-set-rbsr");
        assert_eq!(Capability::PushNegotiate.as_wire_str(), "push-negotiate");
        assert_eq!(Capability::FilterSpec.as_wire_str(), "filter-spec");
        assert_eq!(Capability::AtomicPush.as_wire_str(), "atomic-push");
        assert_eq!(Capability::SignedPush.as_wire_str(), "signed-push");
        assert_eq!(
            Capability::SelfCertifyingRepoId.as_wire_str(),
            "self-certifying-repo-id",
        );
    }

    #[test]
    fn capability_round_trips_through_str() {
        for c in Capability::all() {
            let s = c.as_wire_str();
            let parsed: Capability = s.parse().unwrap();
            assert_eq!(parsed, *c, "round-trip failed for {s}");
        }
    }

    #[test]
    fn unknown_capability_parses_as_err_not_panic() {
        let res: Result<Capability, _> = "no-such-thing".parse();
        assert!(res.is_err());
    }

    #[test]
    fn parse_capabilities_tolerates_unknowns_and_whitespace() {
        let caps = parse_capabilities(" have-set-bloom , no-such-thing,atomic-push ");
        assert_eq!(caps.len(), 2);
        assert!(caps.contains(&Capability::HaveSetBloom));
        assert!(caps.contains(&Capability::AtomicPush));
    }

    #[test]
    fn parse_capabilities_skips_empty_entries() {
        let caps = parse_capabilities(",,have-set-bloom,,");
        assert_eq!(caps.len(), 1);
        assert!(caps.contains(&Capability::HaveSetBloom));
    }

    #[test]
    fn serialize_capabilities_is_deterministic() {
        let caps = [
            Capability::SignedPush,
            Capability::HaveSetBloom,
            Capability::AtomicPush,
        ];
        let a = serialize_capabilities(caps);
        let b = serialize_capabilities(caps.iter().copied().rev());
        assert_eq!(a, b, "output must be order-independent");
        // Ascending by wire string.
        assert_eq!(a, "atomic-push,have-set-bloom,signed-push");
    }

    #[test]
    fn serialize_then_parse_round_trips() {
        let original: BTreeSet<Capability> = [
            Capability::HaveSetBloom,
            Capability::PushNegotiate,
            Capability::FilterSpec,
        ]
        .into_iter()
        .collect();
        let wire = serialize_capabilities(original.iter().copied());
        let parsed = parse_capabilities(&wire);
        assert_eq!(parsed, original);
    }

    #[test]
    fn capability_serde_round_trips_through_json() {
        // Exercise the serde impls so downstream JSON bodies compile.
        let c = Capability::HaveSetBloom;
        let j = serde_json::to_string(&c).unwrap();
        assert_eq!(j, "\"have-set-bloom\"");
        let back: Capability = serde_json::from_str(&j).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn capability_set_intersect_empty() {
        // Empty intersection with anything is empty.
        let a = CapabilitySet::new();
        let b = CapabilitySet::all_known();
        assert!(a.intersect(&b).is_empty());
        assert!(b.intersect(&a).is_empty());
    }

    #[test]
    fn capability_set_intersect_identical() {
        // Intersection of a set with itself is the set.
        let a = CapabilitySet::all_known();
        let r = a.intersect(&a);
        assert_eq!(r, a);
        assert_eq!(r.len(), Capability::all().len());
    }

    #[test]
    fn capability_set_intersect_disjoint() {
        // Disjoint capability sets intersect to the empty set.
        let a = CapabilitySet::with_caps([Capability::HaveSetBloom, Capability::AtomicPush]);
        let b = CapabilitySet::with_caps([Capability::SignedPush, Capability::FilterSpec]);
        let r = a.intersect(&b);
        assert!(r.is_empty());
    }

    #[test]
    fn capability_set_intersect_partial() {
        // Partial overlap keeps only the shared capabilities.
        let a = CapabilitySet::with_caps([
            Capability::HaveSetBloom,
            Capability::AtomicPush,
            Capability::SignedPush,
        ]);
        let b = CapabilitySet::with_caps([Capability::AtomicPush, Capability::FilterSpec]);
        let r = a.intersect(&b);
        assert_eq!(r.len(), 1);
        assert!(r.contains(Capability::AtomicPush));
    }

    #[test]
    fn capability_set_wire_round_trip() {
        // serialize -> parse round-trips and is insensitive to input
        // order.
        let a = CapabilitySet::with_caps([Capability::HaveSetBloom, Capability::AtomicPush]);
        let wire = a.serialize();
        assert_eq!(wire, "atomic-push,have-set-bloom");
        let b = CapabilitySet::parse(&wire);
        assert_eq!(a, b);
    }
}
