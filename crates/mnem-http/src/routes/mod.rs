//! Route modules grouped by surface.
//!
//! The `/v1/*` handlers still live in the crate-level `handlers.rs`
//! for now; this module houses surfaces added after the library was
//! initially factored. `remote` ships the `/remote/v1/*` transport
//! verbs .

pub(crate) mod remote;
pub(crate) mod traverse;
