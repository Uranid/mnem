# mnem-core

Content-addressed versioned substrate for AI agent memory - the core of mnem.

`mnem-core` holds the format types, canonical DAG-CBOR encoding, content
hashing (Multihash + CIDv1), Prolly-tree indexes, operation-log machinery,
and the `ReadonlyRepo` / `Transaction` API that every other crate in the
workspace is built on. The crate is deliberately `#![forbid(unsafe_code)]`,
has no `println!` / terminal I/O, no filesystem access (storage lives behind
the `Blockstore` and `OpHeadsStore` traits), and no tokio binding - so the
same source compiles to native binaries, WASM, and FFI-consumed libraries
under Python / Node / Go. See 
for the WASM-first rationale.

```rust
use std::sync::Arc;
use mnem_core::{repo::ReadonlyRepo, store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore}};

let bs:  Arc<dyn Blockstore>   = Arc::new(MemoryBlockstore::new);
let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::new);
let repo = ReadonlyRepo::init(bs, ohs)?;
```

Workspace top: [`../../README.md`](../../README.md). On-wire format and
invariants: [`../../docs/SPEC.md`](../../docs/SPEC.md).

Licensed under Apache-2.0.
