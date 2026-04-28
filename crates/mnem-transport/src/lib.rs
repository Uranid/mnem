//! # mnem-transport
//!
//! Offline transport for mnem: export a subtree of the content-addressed
//! DAG to a CAR v1 archive, ship it anywhere (USB stick, email, `scp`,
//! S3), import on the other side.
//!
//! CAR ([Content-Addressable aRchive]) is IPFS's standard bundling
//! format: a stream of `(varint length, CID, bytes)` triples preceded
//! by a varint-framed DAG-CBOR header that lists the root CIDs. It is
//! streamable in both directions; this crate implements the
//! [CAR v1] wire shape exactly.
//!
//! ## Shape
//!
//! - [`fn@export`] walks the [`Blockstore`][mnem_core::store::Blockstore]
//!   from a root CID via the
//!   [`Blockstore::iter_from_root`][mnem_core::store::Blockstore::iter_from_root]
//!   default impl, writing every reachable block to a
//!   [`std::io::Write`].
//! - [`fn@import`] reads a CAR from a [`std::io::Read`] and inserts every
//!   block into a target blockstore, verifying the CID on each block.
//! - [`car`] exposes the lower-level CAR reader / writer used by the
//!   above, for callers that need to interleave CAR parsing with other
//!   work.
//!
//! In addition to the file-format half, this crate is home to the
//! *shapes* of mnem's remote wire protocol:
//!
//! - [`protocol`] freezes the [`protocol::PROTOCOL_VERSION`]
//!   integer, the [`protocol::PROTOCOL_HEADER`] HTTP
//!   header name, and the [`protocol::Capability`]
//!   vocabulary. PR 2 does not ship any wire code; it ships the
//!   agreement surface so PR 3 can add verbs without a version bump.
//! - [`remote`] defines [`remote::RemoteConfig`], the
//!   in-memory type parsed from the `[remote.<name>]` section of
//!   `.mnem/config.toml`.
//! - [`have_set`] defines the [`have_set::HaveSet`] trait
//!   and the [`have_set::BloomHaveSet`] reference
//!   back-end used to summarise "blocks I already have" on `fetch-
//!   blocks` / `push-blocks`.
//!
//! Everything in those three modules is pure data + pure functions.
//! HTTP wiring lives in `mnem-http`; the CLI glue (`mnem remote add`
//! etc.) lives in `mnem-cli`; neither lands until PR 3.
//!
//! ## Constraints
//!
//! - WASM-clean. No tokio, no async. The entire interface is
//!   `std::io::{Read, Write}` with `?Sized` support.
//! - No filesystem helpers; callers open files and pass the handles.
//!   Keeps this crate usable inside HTTP streams, pipes, etc.
//! - Deterministic ordering: [`fn@export`] writes blocks in
//!   `iter_from_root`'s depth-first order so the same root produces a
//!   byte-identical CAR across runs.
//!
//! [Content-Addressable aRchive]: https://ipld.io/specs/transport/car/carv1/
//! [CAR v1]: https://ipld.io/specs/transport/car/carv1/

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod car;
#[cfg(feature = "client")]
pub mod client;
pub mod error;
pub mod export;
pub mod have_set;
pub mod import;
pub mod protocol;
pub mod remote;
pub mod secret_token;

pub use error::{ClientError, TransportError};
pub use export::{ExportStats, export};
pub use have_set::{BloomHaveSet, HaveSet, build_have_set};
pub use import::{ImportStats, import};
pub use protocol::{
    CAPABILITIES_HEADER, Capability, CapabilitySet, PROTOCOL_HEADER, PROTOCOL_VERSION,
    parse_capabilities, serialize_capabilities,
};
pub use remote::{RemoteConfig, RemoteConfigFile, RemoteSection, parse_config};
pub use secret_token::SecretToken;

#[cfg(feature = "client")]
pub use client::{HttpRemoteClient, PushResponse, RefsResponse, RemoteClient};

/// Library version (tracks the workspace package version).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
