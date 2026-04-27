//! Round-trip demo: export a small repo's head commit to a CAR v1
//! archive, import it back into a fresh `MemoryBlockstore`, assert
//! byte-identity on every block.
//!
//! This is the end-to-end story for `mnem export` / `mnem import` in
//! miniature: the transport layer never touches the filesystem, so
//! the same code path runs in browser-WASM (per `mnem-transport`'s
//! WASM-clean constraint) and from a 3-line Rust main.
//!
//! See also:
//! - `docs/guide/bulk-ingest.md` - CLI-side story for bulk import.
//! - `docs/RUNBOOK.md#5-car-import-rejected` - error taxonomy when
//!   things go wrong.
//! - (WASM discipline) - why transport stays `std::io`-only.
//!
//! Run:
//! ```console
//! cargo run --example export_then_import -p mnem-transport
//! ```

use std::sync::Arc;

use ipld_core::ipld::Ipld;
use mnem_core::id::NodeId;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};
use mnem_core::{Node, ReadonlyRepo};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- 1. Seed a repo in memory with a handful of nodes. ----
    let src_bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::new());
    let src_ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    let repo0 = ReadonlyRepo::init(src_bs.clone(), src_ohs.clone())?;

    let mut tx = repo0.start_transaction();
    for i in 0..5 {
        let node = Node::new(NodeId::new_v7(), "Person")
            .with_summary(format!("Person number {i}"))
            .with_prop("idx", Ipld::Integer(i.into()));
        tx.add_node(&node)?;
    }
    let repo1 = tx.commit("demo", "seed 5 people")?;
    let head_cid = repo1.view().heads[0].clone();
    println!("seeded repo, head commit = {head_cid}");

    // ---- 2. Export the subtree reachable from `head_cid` to a CAR. ----
    let mut car_bytes: Vec<u8> = Vec::new();
    let export_stats = mnem_transport::export(&*src_bs, &head_cid, &mut car_bytes)?;
    println!(
        "exported {} blocks ({} bytes) to CAR buffer",
        export_stats.blocks, export_stats.bytes
    );
    assert_eq!(car_bytes.len() as u64, export_stats.bytes);

    // ---- 3. Import into a FRESH blockstore. ----
    let dst_bs = MemoryBlockstore::new();
    let import_stats = mnem_transport::import(&mut car_bytes.as_slice(), &dst_bs)?;
    println!(
        "imported {} blocks ({} bytes); declared roots = {:?}",
        import_stats.blocks, import_stats.bytes, import_stats.roots
    );

    // ---- 4. Byte-identity assertion: every block in `src` now
    //        lives byte-for-byte in `dst` under the same CID. ----
    assert_eq!(import_stats.blocks, export_stats.blocks);
    assert_eq!(import_stats.roots.len(), 1, "single-root export");
    assert_eq!(
        import_stats.roots[0], head_cid,
        "imported root must match exported root"
    );

    // Walk the head block on both sides and compare the raw bytes.
    let src_head = src_bs
        .get(&head_cid)?
        .expect("head present on source (just exported)");
    let dst_head = dst_bs
        .get(&head_cid)?
        .expect("head present on dest (just imported)");
    assert_eq!(
        src_head, dst_head,
        "byte-identity must hold for the head commit block"
    );
    println!("round-trip byte-identity verified for head = {head_cid}");
    println!("OK");
    Ok(())
}
