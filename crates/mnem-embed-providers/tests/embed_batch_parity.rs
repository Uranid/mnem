//! Byte-identity contract for [`Embedder::embed_batch`].
//!
//! C3 FIX-2: `mnem-graphrag::summarize::summarize_community` calls
//! `embed_batch` exactly once per retrieve with the canonical-ordered
//! candidate-sentence list, instead of N `embed()` calls. For the
//! graph-summarize path to be a pure-speed change (no recall drift,
//! no bench-gaming) the batched output MUST equal the fan-out output
//! byte-for-byte on every `Embedder` impl we ship.
//!
//! This test pins that contract for the portable providers
//! (MockEmbedder; always available). The native-batched ONNX path is
//! covered indirectly: its output is mean-pool + L2-norm of a row in
//! the batched last_hidden_state, which is the same function of
//! (ids, mask) as the single-item path modulo f32 rounding on
//! zero-padded positions that the attention mask zeroes out before
//! the denominator.

use mnem_embed_providers::{Embedder, MockEmbedder};

/// Fan-out default impl: batched output must equal per-item output.
#[test]
fn mock_embed_batch_equals_fanout() {
    let e = MockEmbedder::new("test:mock", 64);
    let texts = [
        "alpha",
        "beta",
        "gamma",
        "delta is a longer string to vary tokenization",
        "",
        "gamma", // duplicate - batched and per-item must agree
    ];
    let as_refs: Vec<&str> = texts.to_vec();
    let batched = e.embed_batch(&as_refs).unwrap();
    let mut fanout: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    for t in &as_refs {
        fanout.push(e.embed(t).unwrap());
    }
    assert_eq!(batched.len(), fanout.len());
    for (i, (b, f)) in batched.iter().zip(fanout.iter()).enumerate() {
        assert_eq!(b.len(), f.len(), "dim mismatch at row {i}");
        for (j, (bv, fv)) in b.iter().zip(f.iter()).enumerate() {
            assert_eq!(
                bv.to_bits(),
                fv.to_bits(),
                "byte drift at row {i} dim {j}: batch={bv} fanout={fv}"
            );
        }
    }
}

/// Empty input returns empty vec; no panic, no error.
#[test]
fn mock_embed_batch_empty_is_empty() {
    let e = MockEmbedder::new("test:mock", 16);
    let out = e.embed_batch(&[]).unwrap();
    assert!(out.is_empty());
}

/// Single-text batch equals single embed. Important because the ONNX
/// native impl shortcuts batch-of-one to preserve the single-path
/// byte identity exactly.
#[test]
fn mock_embed_batch_single_equals_embed() {
    let e = MockEmbedder::new("test:mock", 32);
    let text = "solo";
    let b = e.embed_batch(&[text]).unwrap();
    let s = e.embed(text).unwrap();
    assert_eq!(b.len(), 1);
    assert_eq!(b[0].len(), s.len());
    for (i, (bv, sv)) in b[0].iter().zip(s.iter()).enumerate() {
        assert_eq!(bv.to_bits(), sv.to_bits(), "mismatch at dim {i}");
    }
}
