//! NER provider adapters for mnem.
//!
//! Ships two built-in providers:
//!
//! - [`RuleNer`], capitalized-phrase heuristic (Person / Organization),
//!   zero dependencies, always available.
//! - [`NullNer`], no-op, emits nothing. Selected via
//!   `[ner]\nprovider = "none"`.
//!
//! The [`open`] factory constructs the right implementation from a
//! [`NerConfig`]. Wire [`NerConfig`] into [`mnem_ingest::IngestConfig`]
//! and call [`open`] during pipeline construction.
//!
//! ## Entity label strings
//!
//! Entity labels are free-form strings, there is no closed vocabulary.
//! Each [`NerProvider`] implementation returns whatever label strings it
//! chooses (e.g. `"Entity:Person"`, `"Entity:Chemical"`, etc.). The
//! mnem node graph stores the raw label string as the node's `ntype`.

pub mod config;
pub mod error;
pub mod null;
pub mod provider;
pub mod rule;

pub use config::{NerConfig, open};
pub use error::NerError;
pub use null::NullNer;
pub use provider::{NamedEntity, NerProvider};
pub use rule::RuleNer;
