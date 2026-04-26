//! End-to-end CAR export / import round-trip tests.
//!
//! Each test builds a small DAG in an in-memory blockstore, exports it
//! to a byte buffer, imports that buffer into a fresh blockstore, and
//! asserts the two stores are byte-identical on every reachable CID.

use std::collections::BTreeMap;

use bytes::Bytes;
use ipld_core::ipld::Ipld;
use mnem_core::codec::hash_to_cid;
use mnem_core::id::{CODEC_RAW, Cid, Multihash};
use mnem_core::store::{Blockstore, MemoryBlockstore};
use mnem_transport::{TransportError, export, import};

// ---- helpers ----

/// Store an Ipld value and return its CID.
fn put_ipld(store: &MemoryBlockstore, value: &Ipld) -> Cid {
    let (bytes, cid) = hash_to_cid(value).expect("encode");
    store.put(cid.clone(), bytes).expect("put");
    cid
}

/// Convert our own `Cid` into the `ipld_core::cid::Cid` form that
/// `Ipld::Link` requires.
fn as_link(cid: &Cid) -> Ipld {
    let inner = ipld_core::cid::Cid::try_from(cid.to_bytes().as_slice()).expect("cid rt");
    Ipld::Link(inner)
}

/// Deep-compare: every CID reachable from `root` in `a` is present in
/// `b` with byte-identical bytes, and vice-versa.
fn assert_stores_equal(a: &MemoryBlockstore, b: &MemoryBlockstore, root: &Cid) {
    let a_blocks: Result<Vec<_>, _> = a.iter_from_root(root).collect();
    let b_blocks: Result<Vec<_>, _> = b.iter_from_root(root).collect();
    let a_blocks = a_blocks.expect("a walk");
    let b_blocks = b_blocks.expect("b walk");
    let a_map: BTreeMap<Cid, Bytes> = a_blocks.into_iter().collect();
    let b_map: BTreeMap<Cid, Bytes> = b_blocks.into_iter().collect();
    assert_eq!(a_map.len(), b_map.len(), "block count differs");
    for (cid, bytes) in &a_map {
        let other = b_map.get(cid).unwrap_or_else(|| panic!("missing {cid}"));
        assert_eq!(other, bytes, "block bytes differ for {cid}");
    }
}

// ---- tests ----

#[test]
fn round_trip_single_leaf_preserves_root_cid() {
    let src = MemoryBlockstore::new();
    let root_cid = put_ipld(&src, &Ipld::String("hello, transport".into()));

    let mut buf = Vec::new();
    let export_stats = export(&src, &root_cid, &mut buf).unwrap();
    assert_eq!(export_stats.blocks, 1);

    let dst = MemoryBlockstore::new();
    let import_stats = import(&mut &buf[..], &dst).unwrap();
    assert_eq!(import_stats.blocks, 1);
    assert_eq!(import_stats.roots, vec![root_cid.clone()]);

    // Root CID unchanged in the destination store.
    assert!(dst.has(&root_cid).unwrap());
    assert_stores_equal(&src, &dst, &root_cid);
}

#[test]
fn round_trip_multi_block_dag_is_byte_identical() {
    // Build a small DAG with a shared child (c reached from both a
    // and b). This exercises dedup both on the export walk and on
    // the import side (where `put` is idempotent).
    let src = MemoryBlockstore::new();

    let leaf = put_ipld(&src, &Ipld::String("leaf-c".into()));
    // Raw block: shows that non-DAG-CBOR leaves round-trip verbatim.
    let raw_bytes = Bytes::from_static(b"\x00\x01\x02 raw payload");
    let raw_cid = Cid::new(CODEC_RAW, Multihash::sha2_256(&raw_bytes));
    src.put(raw_cid.clone(), raw_bytes).unwrap();

    let a = put_ipld(
        &src,
        &Ipld::Map(
            [
                ("tag".into(), Ipld::String("a".into())),
                ("child".into(), as_link(&leaf)),
                ("blob".into(), as_link(&raw_cid)),
            ]
            .into_iter()
            .collect(),
        ),
    );
    let b = put_ipld(
        &src,
        &Ipld::Map(
            [
                ("tag".into(), Ipld::String("b".into())),
                ("child".into(), as_link(&leaf)),
            ]
            .into_iter()
            .collect(),
        ),
    );
    let root_cid = put_ipld(
        &src,
        &Ipld::List(vec![as_link(&a), as_link(&b), as_link(&a)]),
    );

    let mut buf = Vec::new();
    let export_stats = export(&src, &root_cid, &mut buf).unwrap();
    // Reachable set: {root, a, b, leaf, raw} = 5 blocks.
    assert_eq!(export_stats.blocks, 5);

    let dst = MemoryBlockstore::new();
    let import_stats = import(&mut &buf[..], &dst).unwrap();
    assert_eq!(import_stats.blocks, 5);

    assert_stores_equal(&src, &dst, &root_cid);

    // Re-exporting from the destination produces the same bytes.
    let mut buf2 = Vec::new();
    export(&dst, &root_cid, &mut buf2).unwrap();
    assert_eq!(buf, buf2, "re-export must be byte-identical");
}

#[test]
fn corrupt_input_yields_clean_error() {
    // Case A: empty input - header parse fails before reading blocks.
    let dst = MemoryBlockstore::new();
    let err = import(&mut &[][..], &dst).unwrap_err();
    match err {
        TransportError::Car(_) | TransportError::Io(_) => {}
        other => panic!("expected Car/Io error on empty input, got {other:?}"),
    }

    // Case B: a valid CAR truncated mid-block.
    let src = MemoryBlockstore::new();
    let cid = put_ipld(&src, &Ipld::String("target".into()));
    let mut buf = Vec::new();
    export(&src, &cid, &mut buf).unwrap();
    buf.truncate(buf.len() - 3);

    let err = import(&mut &buf[..], &dst).unwrap_err();
    match err {
        TransportError::Car(_) | TransportError::Io(_) => {}
        other => panic!("expected Car/Io error on truncated CAR, got {other:?}"),
    }
}
