//! Shared test fixtures for the mnem workspace.
//!
//! This crate is dev-dependency only; it is never published to crates.io.
//! Anything that needs to be accessible from multiple test binaries across
//! the workspace - canned Nodes, Edges, Trees, deterministic RNG seeds,
//! blockstore mocks instrumented for assertion - lives here.

#![forbid(unsafe_code)]

/// Crate version (tracks workspace package version).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
