//! Import a CAR v1 archive into a [`Blockstore`].

use std::collections::HashSet;
use std::io::Read;

use bytes::Bytes;
use mnem_core::id::Cid;
use mnem_core::store::{Blockstore, blockstore::recompute_cid};

use crate::car::{CarBlockReader, CarHeader, read_header, usize_to_u64};
use crate::error::TransportError;

/// Hard ceiling on a single import's total block-payload bytes.
/// 4 GiB comfortably covers a mid-size knowledge-graph export while
/// refusing to absorb an unbounded attacker-controlled stream that
/// might exhaust disk. Callers needing more can invoke
/// [`import_with_limit`] with an explicit bound.
pub const DEFAULT_MAX_IMPORT_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Summary statistics returned by [`import`].
#[derive(Debug, Clone, Default)]
pub struct ImportStats {
    /// Number of blocks newly stored (including idempotent writes of
    /// already-present CIDs).
    pub blocks: u64,
    /// Total payload bytes written to the blockstore (sum of block
    /// sizes; excludes CAR framing overhead).
    pub bytes: u64,
    /// The CAR's declared root CIDs, for callers that need to know
    /// what they just imported without re-parsing the header. Every
    /// entry here has been verified to be present in the block set
    /// that was actually imported (see [`import`] doc).
    pub roots: Vec<Cid>,
}

/// Read a CAR v1 archive from `r` and write every block into `bs`.
///
/// Every block's CID is verified against its payload bytes before
/// the `put` goes through: a CAR is untrusted input, and silently
/// storing a block whose CID does not match its content would
/// corrupt the blockstore's core invariant. Mismatches raise
/// [`TransportError::CidMismatch`] and the import aborts.
///
/// **Root verification.** Every CID listed in the CAR's header
/// `roots:` field must be present in the set of blocks actually
/// delivered in the body. A malicious CAR that declares a root it
/// never ships is rejected with [`TransportError::MissingRoot`].
/// Without this check, a caller treating `stats.roots[0]` as
/// authenticated would be deceived into walking into an invalid
/// CID.
///
/// **Size cap.** The total block-payload bytes are capped at
/// [`DEFAULT_MAX_IMPORT_BYTES`] (4 GiB). Exceeding the cap raises
/// [`TransportError::SizeLimit`] and the import aborts before the
/// excess data reaches the blockstore. Use [`import_with_limit`]
/// to override. Header and per-block caps live in
/// [`crate::car`].
///
/// # Errors
///
/// - [`TransportError::Car`] for malformed CAR bytes.
/// - [`TransportError::CidMismatch`] for tampered / corrupt blocks.
/// - [`TransportError::MissingRoot`] if a declared root was not
///   shipped in the body.
/// - [`TransportError::SizeLimit`] if total payload exceeds the
///   cap.
/// - [`TransportError::Store`] if the target blockstore refuses a
///   put.
/// - [`TransportError::Io`] on read failure.
pub fn import<R, B>(r: &mut R, bs: &B) -> Result<ImportStats, TransportError>
where
    R: Read + ?Sized,
    B: Blockstore + ?Sized,
{
    import_with_limit(r, bs, DEFAULT_MAX_IMPORT_BYTES)
}

/// Explicit-limit variant of [`import`]. Callers that have
/// out-of-band assurance of the stream's provenance (e.g. local
/// filesystem under their own control) can raise the cap; for
/// network-sourced CARs the default is strongly recommended.
///
/// # Errors
///
/// Same as [`import`].
#[tracing::instrument(
    name = "import_with_limit",
    level = "info",
    target = "mnem::transport::import",
    skip(r, bs),
    fields(
        max_total_bytes,
        block_count = tracing::field::Empty,
        bytes = tracing::field::Empty,
    )
)]
pub fn import_with_limit<R, B>(
    r: &mut R,
    bs: &B,
    max_total_bytes: u64,
) -> Result<ImportStats, TransportError>
where
    R: Read + ?Sized,
    B: Blockstore + ?Sized,
{
    let header: CarHeader = read_header(r)?;
    let roots = header.roots;

    let mut reader = CarBlockReader::new(r);
    let mut blocks: u64 = 0;
    let mut bytes: u64 = 0;
    let mut imported_cids: HashSet<Cid> = HashSet::new();
    while let Some((claimed_cid, data)) = reader.next_block()? {
        // Recompute the CID and compare. `recompute_cid` returns
        // `None` on hash algorithms we don't implement.
        let computed = recompute_cid(&claimed_cid, &data)
            .ok_or_else(|| TransportError::UnsupportedHash(claimed_cid.multihash().code()))?;
        if computed != claimed_cid {
            return Err(TransportError::CidMismatch {
                claimed: claimed_cid,
                computed,
            });
        }
        let payload_len = usize_to_u64(data.len());
        let next_total = bytes
            .checked_add(payload_len)
            .ok_or(TransportError::SizeLimit {
                limit: max_total_bytes,
                observed: u64::MAX,
            })?;
        if next_total > max_total_bytes {
            return Err(TransportError::SizeLimit {
                limit: max_total_bytes,
                observed: next_total,
            });
        }
        bytes = next_total;
        // safety: claimed_cid verified against `data` via
        // `recompute_cid` above; this is the transport-level integrity
        // check that makes `put_trusted` correct here (the CAR block
        // was hashed exactly once, not three times).
        bs.put_trusted(claimed_cid.clone(), Bytes::from(data))?;
        imported_cids.insert(claimed_cid);
        blocks += 1;
    }

    // Root verification: every declared root must actually be in the
    // block set we just imported. Without this, a malicious CAR that
    // lies about its root CID can trick a downstream caller into
    // walking a non-existent tree.
    for root in &roots {
        if !imported_cids.contains(root) {
            return Err(TransportError::MissingRoot { root: root.clone() });
        }
    }

    let span = tracing::Span::current();
    span.record("block_count", blocks);
    span.record("bytes", bytes);
    Ok(ImportStats {
        blocks,
        bytes,
        roots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnem_core::codec::hash_to_cid;
    use mnem_core::store::MemoryBlockstore;
    use serde::Serialize;

    use crate::car::{write_block, write_header};
    use crate::export::export;

    #[derive(Serialize)]
    struct Leaf {
        tag: &'static str,
        n: u32,
    }

    #[test]
    fn import_rejects_tampered_cid() {
        // Build a CAR containing one valid block, then flip a byte
        // inside the payload (which invalidates its claimed CID).
        let src = MemoryBlockstore::new();
        let (bytes, cid) = hash_to_cid(&Leaf { tag: "ok", n: 1 }).unwrap();
        src.put(cid.clone(), bytes).unwrap();

        let mut car = Vec::new();
        export(&src, &cid, &mut car).unwrap();

        // Flip the final payload byte. The header and block-length
        // prefixes live at the front; the payload occupies the tail.
        let last = car.len() - 1;
        car[last] ^= 0xff;

        let dst = MemoryBlockstore::new();
        let err = import(&mut &car[..], &dst).unwrap_err();
        match err {
            TransportError::CidMismatch { .. } => {}
            other => panic!("expected CidMismatch, got {other:?}"),
        }
    }

    #[test]
    fn import_rejects_header_root_not_in_body() {
        // Build a CAR whose declared root CID does NOT appear in any
        // body block. Import must reject with MissingRoot.
        let (_real_bytes, real_cid) = hash_to_cid(&Leaf { tag: "real", n: 1 }).unwrap();
        let (fake_bytes, fake_cid) = hash_to_cid(&Leaf { tag: "fake", n: 2 }).unwrap();

        let mut car = Vec::new();
        // Header advertises real_cid as root.
        let header = CarHeader {
            version: 1,
            roots: vec![real_cid.clone()],
        };
        write_header(&mut car, &header).unwrap();
        // Body ships ONLY fake_cid + its bytes; real_cid never arrives.
        write_block(&mut car, &fake_cid, &fake_bytes).unwrap();

        let dst = MemoryBlockstore::new();
        let err = import(&mut &car[..], &dst).unwrap_err();
        match err {
            TransportError::MissingRoot { root } => assert_eq!(root, real_cid),
            other => panic!("expected MissingRoot, got {other:?}"),
        }
    }

    #[test]
    fn import_enforces_total_bytes_cap() {
        // A legitimate CAR that exceeds the explicit cap must be
        // rejected mid-stream.
        let src = MemoryBlockstore::new();
        let (bytes, cid) = hash_to_cid(&Leaf { tag: "ok", n: 1 }).unwrap();
        let payload_len = bytes.len();
        src.put(cid.clone(), bytes).unwrap();

        let mut car = Vec::new();
        export(&src, &cid, &mut car).unwrap();

        let dst = MemoryBlockstore::new();
        // Cap below the payload size - first block trips the limit.
        let cap = u64::try_from(payload_len).unwrap() - 1;
        let err = import_with_limit(&mut &car[..], &dst, cap).unwrap_err();
        match err {
            TransportError::SizeLimit { limit, observed } => {
                assert_eq!(limit, cap);
                assert!(observed > limit);
            }
            other => panic!("expected SizeLimit, got {other:?}"),
        }
    }
}
