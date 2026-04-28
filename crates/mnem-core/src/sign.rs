//! Ed25519 signing + revocation-list verification (SPEC §9).
//!
//! # Sign
//!
//! [`Signer`] wraps an `ed25519-dalek` signing key. Call
//! [`Signer::sign_commit`] or [`Signer::sign_operation`] on a mutable
//! [`Commit`] / [`Operation`]; the method canonicalises the object with
//! the `signature` field absent (SPEC §9.1), computes the Ed25519
//! signature over the canonical bytes, and re-attaches a [`Signature`]
//! map that carries `algo = "ed25519"`, the 32-byte public key, and
//! the 64-byte signature.
//!
//! # Verify
//!
//! [`Verifier`] checks a signed object in three stages:
//!
//! 1. Algorithm + byte-length gate (`algo == "ed25519"`, 32-byte key,
//!    64-byte signature).
//! 2. Ed25519 `verify_strict` over the canonical pre-image - identical
//!    to what [`Signer`] signed.
//! 3. Revocation check (SPEC §9.2): if the signing key appears in the
//!    revocation list passed to [`Verifier::with_revocations`] and the
//!    object's `time` is strictly greater than the revocation's
//!    `revoked_at`, reject with [`SignError::RevokedKey`].
//!
//! The time-of-use semantics mean signatures produced **at or before**
//! the revocation moment remain valid (SPEC §9.2: "commits signed by
//! a since-revoked key whose `time` is at or before `revoked_at`
//! remain valid").

use bytes::Bytes;
use ed25519_dalek::{Signature as EdSignature, Signer as _, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::codec::to_canonical_bytes;
use crate::error::{Error, SignError};
use crate::objects::{Commit, Operation, Signature};

/// The `algo` tag emitted for every mnem signature in this crate.
pub const ALGO_ED25519: &str = "ed25519";

// ---------------- Signer ----------------

/// An Ed25519 signer. Construct from a 32-byte seed via
/// [`Signer::from_seed_bytes`]. Real-world callers should generate the
/// seed from an OS-level CSPRNG (getrandom, `rand_core::OsRng`, …).
pub struct Signer {
    inner: SigningKey,
}

impl Signer {
    /// Build a signer from a 32-byte Ed25519 seed.
    ///
    /// The seed is NOT hashed or stretched; pass a uniformly random
    /// value from a secure RNG. Deterministic seeds are fine for tests
    /// and smoke demos; use real entropy in production.
    #[must_use]
    pub fn from_seed_bytes(seed: [u8; 32]) -> Self {
        Self {
            inner: SigningKey::from_bytes(&seed),
        }
    }

    /// The 32-byte Ed25519 public key associated with this signer.
    #[must_use]
    pub fn public_key_bytes(&self) -> [u8; 32] {
        *self.inner.verifying_key().as_bytes()
    }

    /// Sign a [`Commit`] in place.
    ///
    /// Clears any existing `signature`, canonicalises the Commit,
    /// signs the bytes, and attaches the resulting [`Signature`].
    ///
    /// # Errors
    ///
    /// Codec errors while canonicalising.
    pub fn sign_commit(&self, commit: &mut Commit) -> Result<(), Error> {
        let bytes = canonical_bytes_for_commit(commit)?;
        let sig = self.inner.sign(&bytes);
        commit.signature = Some(signature_from_parts(
            self.inner.verifying_key().as_bytes(),
            &sig.to_bytes(),
        ));
        Ok(())
    }

    /// Sign an [`Operation`] in place. Same protocol as `sign_commit`.
    ///
    /// # Errors
    ///
    /// Codec errors while canonicalising.
    pub fn sign_operation(&self, op: &mut Operation) -> Result<(), Error> {
        let bytes = canonical_bytes_for_operation(op)?;
        let sig = self.inner.sign(&bytes);
        op.signature = Some(signature_from_parts(
            self.inner.verifying_key().as_bytes(),
            &sig.to_bytes(),
        ));
        Ok(())
    }
}

// ---------------- Verifier + Revocation ----------------

/// One entry in the repository's revocation list (SPEC §9.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Revocation {
    /// 32-byte Ed25519 public key being revoked.
    pub public_key: Bytes,
    /// Microseconds since Unix epoch - the instant the key became
    /// distrusted. Signatures at `time <= revoked_at` remain valid;
    /// signatures at `time > revoked_at` are rejected.
    pub revoked_at: u64,
    /// Optional free-form rationale (e.g. `"compromised"`, `"rotated"`).
    pub reason: String,
}

/// Checks signatures against a (possibly empty) revocation list.
///
/// `Verifier::new()` is the trust-everything-algorithmically form.
/// Repositories with a security posture declare their revocation list
/// in `.mnem/config.cbor` and load it here via
/// [`Verifier::with_revocations`].
#[derive(Debug, Default)]
pub struct Verifier {
    revocations: Vec<Revocation>,
}

impl Verifier {
    /// Verifier with no revocations.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Verifier seeded with a revocation list.
    #[must_use]
    pub const fn with_revocations(revocations: Vec<Revocation>) -> Self {
        Self { revocations }
    }

    /// Verify a [`Commit`]'s signature.
    ///
    /// # Errors
    ///
    /// Returns the matching [`SignError`] variant on any failure in
    /// algorithm gate, signature verification, or revocation check.
    pub fn verify_commit(&self, commit: &Commit) -> Result<(), SignError> {
        let sig = commit.signature.as_ref().ok_or(SignError::NoSignature)?;
        let (vk, ed_sig) = extract_verifier_inputs(sig)?;
        let bytes =
            canonical_bytes_for_commit(commit).map_err(|e| SignError::Encoding(e.to_string()))?;
        vk.verify_strict(&bytes, &ed_sig)
            .map_err(|_| SignError::InvalidSignature)?;
        self.check_revocation(sig.public_key.as_ref(), commit.time)
    }

    /// Verify an [`Operation`]'s signature.
    ///
    /// # Errors
    ///
    /// See [`Verifier::verify_commit`].
    pub fn verify_operation(&self, op: &Operation) -> Result<(), SignError> {
        let sig = op.signature.as_ref().ok_or(SignError::NoSignature)?;
        let (vk, ed_sig) = extract_verifier_inputs(sig)?;
        let bytes =
            canonical_bytes_for_operation(op).map_err(|e| SignError::Encoding(e.to_string()))?;
        vk.verify_strict(&bytes, &ed_sig)
            .map_err(|_| SignError::InvalidSignature)?;
        self.check_revocation(sig.public_key.as_ref(), op.time)
    }

    fn check_revocation(&self, public_key: &[u8], time: u64) -> Result<(), SignError> {
        for rev in &self.revocations {
            if rev.public_key.as_ref() == public_key && time > rev.revoked_at {
                return Err(SignError::RevokedKey {
                    revoked_at: rev.revoked_at,
                    time,
                });
            }
        }
        Ok(())
    }
}

// ---------------- Helpers ----------------

fn signature_from_parts(public_key: &[u8; 32], sig: &[u8; 64]) -> Signature {
    Signature {
        algo: ALGO_ED25519.into(),
        public_key: Bytes::copy_from_slice(public_key),
        sig: Bytes::copy_from_slice(sig),
    }
}

fn extract_verifier_inputs(sig: &Signature) -> Result<(VerifyingKey, EdSignature), SignError> {
    if sig.algo != ALGO_ED25519 {
        return Err(SignError::WrongAlgorithm {
            got: sig.algo.clone(),
        });
    }
    let pk_arr: [u8; 32] = sig
        .public_key
        .as_ref()
        .try_into()
        .map_err(|_| SignError::MalformedKey)?;
    let vk = VerifyingKey::from_bytes(&pk_arr).map_err(|_| SignError::MalformedKey)?;
    let sig_arr: [u8; 64] = sig
        .sig
        .as_ref()
        .try_into()
        .map_err(|_| SignError::MalformedSignature)?;
    let ed_sig = EdSignature::from_bytes(&sig_arr);
    Ok((vk, ed_sig))
}

fn canonical_bytes_for_commit(commit: &Commit) -> Result<Vec<u8>, Error> {
    let mut c = commit.clone();
    c.signature = None;
    Ok(to_canonical_bytes(&c)?.to_vec())
}

fn canonical_bytes_for_operation(op: &Operation) -> Result<Vec<u8>, Error> {
    let mut o = op.clone();
    o.signature = None;
    Ok(to_canonical_bytes(&o)?.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::{CODEC_RAW, ChangeId, Cid, Multihash};

    fn raw(n: u32) -> Cid {
        Cid::new(CODEC_RAW, Multihash::sha2_256(&n.to_be_bytes()))
    }

    fn sample_commit(time: u64) -> Commit {
        Commit::new(
            ChangeId::from_bytes_raw([1u8; 16]),
            raw(1),
            raw(2),
            raw(3),
            "alice@example.org",
            time,
            "init",
        )
    }

    fn sample_operation(time: u64) -> Operation {
        Operation::new(raw(1), "alice@example.org", time, "commit: init")
    }

    #[test]
    fn sign_then_verify_commit() {
        let signer = Signer::from_seed_bytes([0x42u8; 32]);
        let mut c = sample_commit(1_000);
        signer.sign_commit(&mut c).unwrap();
        Verifier::new().verify_commit(&c).unwrap();
    }

    #[test]
    fn sign_then_verify_operation() {
        let signer = Signer::from_seed_bytes([0x21u8; 32]);
        let mut op = sample_operation(1_000);
        signer.sign_operation(&mut op).unwrap();
        Verifier::new().verify_operation(&op).unwrap();
    }

    #[test]
    fn verify_with_no_signature_errors() {
        let c = sample_commit(1_000);
        let err = Verifier::new().verify_commit(&c).unwrap_err();
        assert!(matches!(err, SignError::NoSignature));
    }

    #[test]
    fn tampered_commit_fails_verify() {
        let signer = Signer::from_seed_bytes([0x42u8; 32]);
        let mut c = sample_commit(1_000);
        signer.sign_commit(&mut c).unwrap();
        // Tamper: change message post-sign.
        c.message = "I am a thief".into();
        let err = Verifier::new().verify_commit(&c).unwrap_err();
        assert!(matches!(err, SignError::InvalidSignature));
    }

    #[test]
    fn wrong_algorithm_rejected() {
        let signer = Signer::from_seed_bytes([0x1u8; 32]);
        let mut c = sample_commit(1_000);
        signer.sign_commit(&mut c).unwrap();
        // Swap algo tag.
        let sig = c.signature.as_mut().unwrap();
        sig.algo = "rsa".into();
        let err = Verifier::new().verify_commit(&c).unwrap_err();
        assert!(matches!(err, SignError::WrongAlgorithm { .. }));
    }

    #[test]
    fn malformed_key_length_rejected() {
        let signer = Signer::from_seed_bytes([0x1u8; 32]);
        let mut c = sample_commit(1_000);
        signer.sign_commit(&mut c).unwrap();
        let sig = c.signature.as_mut().unwrap();
        sig.public_key = Bytes::from(vec![0u8; 16]); // too short
        let err = Verifier::new().verify_commit(&c).unwrap_err();
        assert!(matches!(err, SignError::MalformedKey));
    }

    #[test]
    fn revocation_after_commit_time_still_valid() {
        let signer = Signer::from_seed_bytes([0x42u8; 32]);
        let mut c = sample_commit(1_000);
        signer.sign_commit(&mut c).unwrap();
        let verifier = Verifier::with_revocations(vec![Revocation {
            public_key: Bytes::copy_from_slice(&signer.public_key_bytes()),
            revoked_at: 2_000, // strictly after c.time
            reason: "rotated".into(),
        }]);
        verifier.verify_commit(&c).unwrap();
    }

    #[test]
    fn revocation_before_commit_time_rejects() {
        let signer = Signer::from_seed_bytes([0x42u8; 32]);
        let mut c = sample_commit(1_000);
        signer.sign_commit(&mut c).unwrap();
        let verifier = Verifier::with_revocations(vec![Revocation {
            public_key: Bytes::copy_from_slice(&signer.public_key_bytes()),
            revoked_at: 500, // strictly before c.time
            reason: "compromised".into(),
        }]);
        let err = verifier.verify_commit(&c).unwrap_err();
        match err {
            SignError::RevokedKey { revoked_at, time } => {
                assert_eq!(revoked_at, 500);
                assert_eq!(time, 1_000);
            }
            e => panic!("wrong variant: {e:?}"),
        }
    }

    #[test]
    fn revocation_equals_commit_time_still_valid() {
        // SPEC §9.2: "signatures whose time <= revoked_at remain valid"
        let signer = Signer::from_seed_bytes([0x42u8; 32]);
        let mut c = sample_commit(1_000);
        signer.sign_commit(&mut c).unwrap();
        let verifier = Verifier::with_revocations(vec![Revocation {
            public_key: Bytes::copy_from_slice(&signer.public_key_bytes()),
            revoked_at: 1_000, // exactly equal
            reason: "rotated".into(),
        }]);
        verifier.verify_commit(&c).unwrap();
    }

    #[test]
    fn re_signing_is_idempotent() {
        // Signing the same commit twice with the same key produces a
        // valid signature both times.
        let signer = Signer::from_seed_bytes([0x42u8; 32]);
        let mut c1 = sample_commit(1_000);
        signer.sign_commit(&mut c1).unwrap();
        let mut c2 = sample_commit(1_000);
        signer.sign_commit(&mut c2).unwrap();
        Verifier::new().verify_commit(&c1).unwrap();
        Verifier::new().verify_commit(&c2).unwrap();
        // Ed25519 is deterministic (RFC 8032), so the signatures match.
        assert_eq!(c1.signature, c2.signature);
    }
}
