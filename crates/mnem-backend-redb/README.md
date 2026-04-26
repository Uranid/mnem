# mnem-backend-redb

Production embedded-KV backend for mnem - redb-backed `Blockstore` and `OpHeadsStore`.

Implements the two storage traits from `mnem-core` against
[`redb`](https://github.com/cberner/redb), a pure-Rust embedded ACID
key-value store. A single `.redb` file holds `objects` (CID bytes to object
bytes) and `op_heads` (CID bytes to unit, presence-as-truth). Writes are
atomic per-call: each put / delete opens a write transaction and commits,
which fsyncs. redb serialises writers inside a process and uses filesystem
locking across processes. Picked as the sole v1 durable backend per
 for its pure-Rust
dependency surface, mmap reads, and ACID semantics.

```rust
use mnem_backend_redb::open_or_init;
use mnem_core::repo::ReadonlyRepo;

let (bs, ohs, _path) = open_or_init("/tmp/agent.redb")?;
let repo = ReadonlyRepo::init(bs, ohs)?;
```

Workspace top: [`../../README.md`](../../README.md). Spec:
[`../../docs/SPEC.md`](../../docs/SPEC.md).

Licensed under Apache-2.0.
