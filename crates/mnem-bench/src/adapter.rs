//! `BenchAdapter` trait - what every system-under-test (mnem, mem0,
//! MemPalace) implements so the scorers can drive it.

use serde::{Deserialize, Serialize};
use std::error::Error as StdError;

/// One document staged for ingest. The scorer assigns the
/// `external_id` (e.g. session id, dialog id) and reads it back from
/// the retrieved hits via [`Hit::external_id`] so the adapter is
/// free to mint any internal id it likes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IngestDoc {
    /// Stable per-bench identifier the scorer attaches to this doc.
    /// Echoed back unchanged on the matching [`Hit`].
    pub external_id: String,
    /// Scope label used to keep the corpus per-question / per-conv
    /// isolated. mnem encodes this as the `Node.ntype` so the
    /// retriever's label filter scopes to a single question's
    /// haystack.
    pub label: String,
    /// Free-form natural-language text the embedder consumes.
    pub text: String,
    /// Optional structured property bag. Echoed onto the node /
    /// document for downstream filtering. Values are stringified by
    /// adapters that cannot store arbitrary JSON.
    pub props: serde_json::Map<String, serde_json::Value>,
}

/// One hit from a retrieve. `external_id` matches the value the
/// scorer staged on the corresponding [`IngestDoc`]. Adapters that
/// re-rank or filter still preserve `external_id` round-trip.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hit {
    /// External id originally staged with [`IngestDoc::external_id`].
    pub external_id: String,
    /// Adapter-internal ranking score. Higher is better. The scorer
    /// only consumes the rank order, but the score is logged for
    /// debugging.
    pub score: f32,
}

/// Adapter trait. One instance handles a single benchmark run.
pub trait BenchAdapter {
    /// Drop all per-question / per-conv state so the next ingest is
    /// fresh. mnem's in-memory adapter rotates a new `Repo`; HTTP
    /// adapters POST a label-scoped delete.
    ///
    /// # Errors
    ///
    /// Returns the adapter's own error type when the reset fails.
    fn reset(&mut self) -> Result<(), Box<dyn StdError>>;

    /// Ingest the given documents under their staged labels. Caller
    /// is responsible for [`Self::reset`] between unrelated batches.
    ///
    /// # Errors
    ///
    /// Returns adapter-specific errors when the underlying store
    /// rejects a document.
    fn ingest(&mut self, docs: &[IngestDoc]) -> Result<(), Box<dyn StdError>>;

    /// Retrieve the top-K matches for `query` within `label`.
    /// Returned hits MUST be ordered score-desc. The scorer only
    /// looks at the first `top_k` entries.
    ///
    /// # Errors
    ///
    /// Returns adapter-specific errors when the underlying store
    /// rejects the query.
    fn retrieve(
        &mut self,
        label: &str,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<Hit>, Box<dyn StdError>>;

    /// Free-form adapter name for logs + RESULTS.md. e.g. `"mnem"`.
    fn name(&self) -> &str;
}
