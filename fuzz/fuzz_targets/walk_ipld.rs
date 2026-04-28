//! Fuzz target: DAG-CBOR decode + `walk_ipld`.
//!
//! Feeds arbitrary bytes through
//! [`mnem_core::codec::from_canonical_bytes`] into an
//! [`ipld_core::ipld::Ipld`] value, then exercises
//! [`mnem_core::codec::extract_links`] (the public entry point that
//! wraps the internal `walk_ipld`). The contract under test:
//!
//! 1. **No panics on malformed input.** DAG-CBOR is untrusted bytes
//!    the moment it arrives on the wire (CAR body, REST handler,
//!    MCP tool arg after round-trip). Any panic is a critical
//!    finding.
//! 2. **Bounded depth.** `extract_links` enforces
//!    [`mnem_core::codec::dagcbor::WALK_IPLD_MAX_DEPTH`] (64). A
//!    deeply-nested payload must resolve to `CodecError::Decode`,
//!    not stack-overflow.
//! 3. **Bounded time.** `extract_links` is a linear tree walk. A
//!    pathological input must not spiral into exponential work.
//!    libFuzzer's per-input timeout catches this; the coverage
//!    guide steers toward shapes that touch every match arm.
//!
//! The target fuzzes the *link extraction* pipeline rather than
//! just the decoder because the post-decode walk is where
//! `WALK_IPLD_MAX_DEPTH` is enforced. Fuzzing the decode alone
//! would miss depth-bomb regressions in `walk_ipld`.

#![no_main]

use ipld_core::ipld::Ipld;
use libfuzzer_sys::fuzz_target;
use mnem_core::codec::{extract_links, from_canonical_bytes};

fuzz_target!(|data: &[u8]| {
    // Pipeline 1: raw decode. Any panic here is a decoder bug.
    let decoded: Result<Ipld, _> = from_canonical_bytes(data);

    // Pipeline 2: link extraction via the public API. This runs
    // its own fresh decode internally, so a bug in the decode ->
    // walk handshake surfaces here even if the raw decode above
    // succeeded.
    let _ = extract_links(data);

    // If we successfully decoded to Ipld, re-encoding must also
    // not panic. (Round-trip byte-identity is NOT asserted:
    // `from_canonical_bytes` accepts a broader input set than
    // `to_canonical_bytes` produces - the spec strictness asymmetry
    // is intentional. We just require that the re-encode doesn't
    // crash.)
    if let Ok(v) = decoded {
        let _ = serde_ipld_dagcbor::to_vec(&v);
    }
});
