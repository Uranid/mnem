# mnem-transport

Offline transport for mnem: CAR v1 export/import of content-addressed subtrees, plus the frozen shapes of the remote wire protocol.

`mnem-transport` is the WASM-clean, tokio-free half of mnem's
replication story. It implements [CAR v1](https://ipld.io/specs/transport/car/carv1/)
reader / writer over pure `std::io::{Read, Write}`, so a subtree can
be shipped on a USB stick, attached to an email, `scp`'d across an
air-gap, or streamed through a pipe. It also freezes the
`PROTOCOL_VERSION`, `PROTOCOL_HEADER`, and `Capability` vocabulary
that the HTTP remote protocol is built on, plus the `RemoteConfig`
TOML shape and the `HaveSet`
bloom-filter trait used by `fetch-blocks` / `push-blocks`. HTTP
wiring lives in `mnem-http`; CLI glue lives in `mnem-cli`.

```rust
use std::fs::File;
use mnem_transport::{export, import};

// Export a subtree rooted at `head` to a CAR archive.
let mut out = File::create("backup.car")?;
export(&repo.blockstore, head, &mut out)?;

// Import the same archive into a fresh blockstore elsewhere.
let mut inp = File::open("backup.car")?;
import(&target_blockstore, &mut inp)?;
```

Workspace top: [`../../README.md`](../../README.md). Wire protocol:
[`../../`](../../).

Licensed under Apache-2.0.
