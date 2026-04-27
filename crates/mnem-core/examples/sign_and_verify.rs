//! M12 smoke test: ed25519 signing + revocation-list verification.
//!
//! Demonstrates the full SPEC §9 flow:
//!
//! 1. Build and sign a Commit.
//! 2. Verify with a clean Verifier.
//! 3. Tamper with the Commit - verify detects.
//! 4. Wrong-algorithm rejection.
//! 5. Revocation AFTER commit.time - still valid (backward-compat).
//! 6. Revocation BEFORE commit.time - rejected.
//!
//! Run:
//!
//! ```console
//! cargo run -p mnem-core --example sign_and_verify
//! ```
//!
//! Output captured to `/tmp/mnem-test/sign_and_verify.out`.

use bytes::Bytes;
use mnem_core::error::SignError;
use mnem_core::id::{CODEC_RAW, ChangeId, Cid, Multihash};
use mnem_core::objects::Commit;
use mnem_core::sign::{Revocation, Signer, Verifier};

fn raw(n: u32) -> Cid {
    Cid::new(CODEC_RAW, Multihash::sha2_256(&n.to_be_bytes()))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("# mnem M12 smoke test: ed25519 signing + revocation");
    println!("# mnem-core: {}", mnem_core::VERSION);
    println!();

    // Deterministic seed for demo; real usage draws from OsRng / getrandom.
    let signer = Signer::from_seed_bytes([0x42u8; 32]);
    let pk = signer.public_key_bytes();
    println!(
        "public key (first 4 bytes): {:02x}{:02x}{:02x}{:02x}...",
        pk[0], pk[1], pk[2], pk[3]
    );
    println!();

    // Build + sign a commit
    let commit_time = 1_700_000_000_000_000u64;
    let mut commit = Commit::new(
        ChangeId::from_bytes_raw([0xABu8; 16]),
        raw(1),
        raw(2),
        raw(3),
        "alice@example.org",
        commit_time,
        "initial commit",
    );
    signer.sign_commit(&mut commit)?;
    let sig = commit.signature.as_ref().unwrap();
    println!("signed commit:");
    println!("  algo = {:?}", sig.algo);
    println!(
        "  sig  = {:02x}{:02x}{:02x}{:02x}... ({}  bytes)",
        sig.sig[0],
        sig.sig[1],
        sig.sig[2],
        sig.sig[3],
        sig.sig.len()
    );
    println!();

    // 1. Clean verify
    Verifier::new().verify_commit(&commit)?;
    println!("verify (clean):                      ok");

    // 2. Tamper detection
    let mut tampered = commit.clone();
    tampered.message = "I am a thief".into();
    match Verifier::new().verify_commit(&tampered) {
        Err(SignError::InvalidSignature) => {
            println!("verify (tampered message):           rejected");
        }
        other => panic!("expected InvalidSignature, got {other:?}"),
    }

    // 3. Wrong algorithm
    let mut wrong_algo = commit.clone();
    wrong_algo.signature.as_mut().unwrap().algo = "rsa".into();
    match Verifier::new().verify_commit(&wrong_algo) {
        Err(SignError::WrongAlgorithm { got }) => {
            println!("verify (wrong algo='{got}'):             rejected");
        }
        other => panic!("expected WrongAlgorithm, got {other:?}"),
    }

    // 4. Revocation AFTER the commit's time - backward-compat
    let later_rev = Verifier::with_revocations(vec![Revocation {
        public_key: Bytes::copy_from_slice(&pk),
        revoked_at: commit_time + 1, // strictly after
        reason: "rotated".into(),
    }]);
    later_rev.verify_commit(&commit)?;
    println!("verify (revocation AFTER commit):    ok (backward-compat)");

    // 5. Revocation EQUAL to commit time - valid per SPEC §9.2
    let equal_rev = Verifier::with_revocations(vec![Revocation {
        public_key: Bytes::copy_from_slice(&pk),
        revoked_at: commit_time, // exactly equal
        reason: "rotated".into(),
    }]);
    equal_rev.verify_commit(&commit)?;
    println!("verify (revocation EQUAL commit):    ok (≤ revoked_at is valid)");

    // 6. Revocation BEFORE the commit's time - rejected
    let early_rev = Verifier::with_revocations(vec![Revocation {
        public_key: Bytes::copy_from_slice(&pk),
        revoked_at: commit_time - 1, // strictly before
        reason: "compromised".into(),
    }]);
    match early_rev.verify_commit(&commit) {
        Err(SignError::RevokedKey { revoked_at, time }) => {
            println!("verify (revocation BEFORE commit):   rejected ({revoked_at} < {time})");
        }
        other => panic!("expected RevokedKey, got {other:?}"),
    }

    println!();
    println!("# M12 sign + verify smoke test: ok");
    Ok(())
}
