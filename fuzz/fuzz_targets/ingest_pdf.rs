//! Fuzz target: ingest PDF parser.
//!
//! Feeds arbitrary bytes into [`mnem_ingest::pdf::parse_pdf`]. The
//! expected contract is:
//!
//! 1. **No panics.** Any byte sequence (well-formed, malformed,
//!    truncated, adversarial) must resolve to `Ok(Vec<Section>)` or a
//!    typed `Err(mnem_ingest::Error::ParseFailed { what: "pdf", .. })`.
//!    Phase-B5b already wraps the underlying `pdf-extract` call in
//!    `catch_unwind`; this fuzz target regression-guards that
//!    contract against future refactors.
//!
//! 2. **Bounded output.** A valid PDF must split its pages on
//!    form-feed (`\x0C`); malformed input must either be rejected
//!    with `ParseFailed` or return a small `Vec<Section>` with
//!    byte-boundary-safe slices. We don't assert a specific shape -
//!    the property under test is panic-freedom, not output structure.
//!
//! The target is **stateless** and **dep-light**: it imports only
//! `mnem_ingest`, which in turn needs nothing more than the default
//! (non-feature-gated) pipeline. No sidecar, no LLM - this fuzzes the
//! pure-Rust `pdf-extract` path that every ingest run traverses.
//!
//! Seed corpus lives under `fuzz/corpus/ingest_pdf/`. Operators with
//! real-world PDFs can drop them there for coverage-guided runs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use mnem_ingest::pdf::parse_pdf;

fuzz_target!(|data: &[u8]| {
    // Intentionally discard the Result: the property under test is
    // "no panic on arbitrary bytes", not any specific outcome shape.
    // A valid PDF lands as Ok(Vec<Section>); anything else lands as
    // Err(ParseFailed). Both are acceptable.
    let _ = parse_pdf(data);
});
