//! Export a reachable subtree of a [`Blockstore`] to CAR v1.

use std::io::Write;

use mnem_core::id::Cid;
use mnem_core::store::Blockstore;

use crate::car::{CarHeader, usize_to_u64, write_block, write_header};
use crate::error::TransportError;

/// Summary statistics returned by [`export`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ExportStats {
    /// Number of blocks written. Equals the reachable-set size from
    /// the root, deduplicated.
    pub blocks: u64,
    /// Total bytes written to the writer (header varint + header body
    /// + every `varint || cid || data` triple).
    pub bytes: u64,
}

/// Walk the DAG rooted at `root` in the blockstore and emit every
/// reachable block to `w` as a CAR v1 archive.
///
/// - Header declares `[root]` as the sole root CID.
/// - Block order matches
///   [`mnem_core::store::Blockstore::iter_from_root`]'s
///   deterministic depth-first walk.
/// - Each block is written verbatim - no re-encoding, no
///   canonicalisation. CIDs are whatever the blockstore returned.
///
/// # Errors
///
/// - [`TransportError::Io`] on write failure to `w`.
/// - [`TransportError::Store`] on blockstore read failure or
///   missing-block reference (partial DAG).
/// - [`TransportError::Codec`] on unreadable link structure inside a
///   block.
#[tracing::instrument(
    name = "export",
    level = "info",
    target = "mnem::transport::export",
    skip(bs, w),
    fields(
        root = %root,
        block_count = tracing::field::Empty,
        bytes = tracing::field::Empty,
    )
)]
pub fn export<W, B>(bs: &B, root: &Cid, w: &mut W) -> Result<ExportStats, TransportError>
where
    W: Write + ?Sized,
    B: Blockstore + ?Sized,
{
    // ---- header ----
    let header = CarHeader::single_root(root.clone());
    // Count bytes by writing through a counter first for the header,
    // then continuing with raw writes on the real stream. Counting
    // the header separately avoids wrapping `w` in a trait-object
    // adapter for every block write.
    let mut header_counter = CountingWriter::new(Vec::<u8>::new());
    write_header(&mut header_counter, &header)?;
    let header_buf = header_counter.into_inner();
    let header_len = usize_to_u64(header_buf.len());
    w.write_all(&header_buf)?;

    // ---- blocks ----
    let mut block_counter = CountingWriter::new(w);
    let mut blocks: u64 = 0;
    for item in bs.iter_from_root(root) {
        let (cid, data) = item?;
        write_block(&mut block_counter, &cid, &data)?;
        blocks += 1;
    }
    let block_bytes = block_counter.count;

    let total_bytes = header_len + block_bytes;
    let span = tracing::Span::current();
    span.record("block_count", blocks);
    span.record("bytes", total_bytes);
    Ok(ExportStats {
        blocks,
        bytes: total_bytes,
    })
}

/// Tiny `Write` adapter that counts bytes. Kept private so the public
/// `export` signature stays simple.
struct CountingWriter<W: Write + ?Sized> {
    count: u64,
    inner: W,
}

impl<W: Write> CountingWriter<W> {
    const fn new(inner: W) -> Self {
        Self { count: 0, inner }
    }

    fn into_inner(self) -> W {
        self.inner
    }
}

impl<W: Write + ?Sized> Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.count += usize_to_u64(n);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnem_core::codec::hash_to_cid;
    use mnem_core::store::MemoryBlockstore;
    use serde::Serialize;

    #[derive(Serialize)]
    struct Leaf {
        tag: &'static str,
        n: u32,
    }

    #[test]
    fn export_single_leaf_emits_header_plus_one_block() {
        let store = MemoryBlockstore::new();
        let (bytes, cid) = hash_to_cid(&Leaf { tag: "x", n: 1 }).unwrap();
        store.put(cid.clone(), bytes).unwrap();

        let mut sink = Vec::new();
        let stats = export(&store, &cid, &mut sink).unwrap();
        assert_eq!(stats.blocks, 1);
        assert_eq!(usize::try_from(stats.bytes).unwrap(), sink.len());
        assert!(!sink.is_empty());
    }
}
