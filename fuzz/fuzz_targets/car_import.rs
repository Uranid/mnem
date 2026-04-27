//! Fuzz target: CAR import.
//!
//! Feeds arbitrary bytes into [`mnem_transport::import`] via an
//! in-memory [`MemoryBlockstore`]. The expected contract is:
//!
//! 1. **No panics.** Any byte sequence (well-formed, malformed,
//!    tampered, truncated) must resolve to `Ok(ImportStats)` or a
//!    typed `Err(TransportError)`. A panic is a critical finding -
//!    CAR is the offline interop boundary and panic-on-bytes means
//!    we crash any tooling that points a CAR at us.
//! 2. **Bounded memory.** The default 4 GiB cap in
//!    [`mnem_transport::DEFAULT_MAX_IMPORT_BYTES`] guards against
//!    size-bomb streams. We drive the explicit-limit entry point
//!    with a much smaller cap (256 KiB) so the fuzzer can churn
//!    millions of inputs without host-OOM.
//! 3. **CID integrity.** Any tampered block (claimed CID differs
//!    from the computed CID) must be rejected with
//!    `TransportError::CidMismatch`. A tampered block passing
//!    through silently would corrupt the blockstore invariant.
//!
//! The target does not seed a corpus on this branch. Operators who
//! want coverage-guided fuzzing against real CAR samples can drop
//! files into `fuzz/corpus/car_import/` (already created as an
//! empty directory) - libFuzzer picks them up automatically.

#![no_main]

use libfuzzer_sys::fuzz_target;
use mnem_core::store::MemoryBlockstore;
use mnem_transport::import::import_with_limit;

/// Import cap used by the fuzzer. Smaller than the production
/// default (4 GiB) so a deliberate size-bomb input cannot exhaust
/// host memory during a long fuzzing run.
const FUZZ_IMPORT_CAP: u64 = 256 * 1024;

fuzz_target!(|data: &[u8]| {
    let bs = MemoryBlockstore::new();
    let mut cursor = data;
    // Intentionally discard the Result: the property under test is
    // "no panic on arbitrary bytes", not any specific outcome shape.
    // A valid CAR lands as Ok(ImportStats); anything else lands as
    // Err(TransportError). Both are acceptable.
    let _ = import_with_limit(&mut cursor, &bs, FUZZ_IMPORT_CAP);
});
