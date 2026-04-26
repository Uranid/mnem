//! Criterion bench: CAR v1 export + import round-trip over a
//! 1000-node seed repo. Ports `examples/export_then_import.rs` onto
//! the statistical harness so regression drift in the transport layer
//! becomes visible commit-over-commit.
//!
//! Phase-B1b (PHASE-A-2 bench plan §2). Run on demand:
//!
//! ```console
//! cargo bench -p mnem-transport --bench car
//! ```

use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use ipld_core::ipld::Ipld;
use mnem_core::id::{Cid, NodeId};
use mnem_core::objects::Node;
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

const N: u64 = 1_000;

fn seed_repo() -> (Arc<MemoryBlockstore>, Cid) {
    let bs = Arc::new(MemoryBlockstore::new());
    let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new());
    let repo = ReadonlyRepo::init(bs.clone() as Arc<dyn Blockstore>, ohs).expect("init");
    let mut tx = repo.start_transaction();
    for i in 0..N {
        let node = Node::new(NodeId::new_v7(), "Person")
            .with_summary(format!("Person-{i}"))
            .with_prop("idx", Ipld::Integer(i as i128));
        tx.add_node(&node).expect("add_node");
    }
    let r = tx.commit("bench", "seed").expect("commit");
    let head = r.view().heads[0].clone();
    (bs, head)
}

fn bench_car(c: &mut Criterion) {
    let (src_bs, head) = seed_repo();

    // --- Export ---
    let mut export_group = c.benchmark_group("car_export");
    export_group.throughput(Throughput::Elements(N));
    export_group.bench_function("1k_nodes", |b| {
        b.iter(|| {
            let mut buf: Vec<u8> = Vec::with_capacity(1 << 20);
            mnem_transport::export(src_bs.as_ref(), &head, &mut buf).expect("export");
            buf
        });
    });
    export_group.finish();

    // Pre-render the CAR bytes once for import runs.
    let mut car_bytes: Vec<u8> = Vec::with_capacity(1 << 20);
    mnem_transport::export(src_bs.as_ref(), &head, &mut car_bytes).expect("prerender");

    // --- Import ---
    let mut import_group = c.benchmark_group("car_import");
    import_group.throughput(Throughput::Bytes(car_bytes.len() as u64));
    import_group.bench_function("1k_nodes", |b| {
        b.iter(|| {
            let dst = MemoryBlockstore::new();
            mnem_transport::import(&mut car_bytes.as_slice(), &dst).expect("import")
        });
    });
    import_group.finish();
}

criterion_group!(benches, bench_car);
criterion_main!(benches);
