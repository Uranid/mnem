//! Optional LLM-backed [`Extractor`] that talks to a local Ollama server.
//!
//! Phase-B5e ships this module behind the `ollama` Cargo feature. The
//! rule-based baseline in [`crate::extract::RuleExtractor`] must always
//! work without any network round-trip; this extractor is strictly
//! opt-in for operators who already run Ollama locally and want a
//! higher-recall NER pass.
//!
//! # Contract
//!
//! The extractor calls Ollama's `/api/generate` endpoint with a
//! JSON-schema-constrained prompt (`format` parameter). The model is
//! asked to return:
//!
//! ```json
//! {
//!   "entities":  [ { "kind": "person", "text": "Alice",
//!                     "start": 17, "end": 22 } ],
//!   "relations": [ { "kind": "acts_on", "subj": 0, "obj": 1 } ]
//! }
//! ```
//!
//! Because LLMs hallucinate spans, every returned entity is
//! re-verified against the section text: if
//! `section.text[start..end] != text`, the span is discarded. The same
//! guard applies to non-UTF-8-boundary offsets.
//!
//! # Confidence scoring
//!
//! LLM entities are stamped with `confidence = 0.75`. LLM relations
//! ride at `0.65`. Operators who want to weight the pipeline by confidence
//! can use these deterministic bands.
//!
//! # Label pass-through
//!
//! The `kind` field from the LLM response is used verbatim (after trimming
//! and empty-string rejection). There is no normalization or remapping, the
//! LLM's own label string goes straight to the graph as the entity ntype.
//!
//! # Failure policy
//!
//! Any HTTP failure, timeout, or schema-invalid response resolves to an
//! empty `Vec` plus a `tracing::warn!`. The extractor never panics and
//! never surfaces a transport error to the caller: the ingest pipeline
//! degrades gracefully to whatever NER signal was found.
//!
//! Heavy deps (`reqwest`) are feature-gated; the default build stays
//! dep-clean.

use std::ops::Range;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::extract::{EntitySpan, Extractor, RelationSpan};
use crate::types::Section;

/// Default Ollama endpoint; matches the out-of-box `ollama serve` bind.
pub const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";

/// Default model used when none is specified. Small, schema-honouring,
/// and widely available in the Ollama registry.
pub const DEFAULT_OLLAMA_MODEL: &str = "llama3.2:3b";

/// Fixed confidence for LLM-derived entities.
pub const LLM_ENTITY_CONFIDENCE: f32 = 0.75;

/// Fixed confidence for LLM-derived relations.
pub const LLM_RELATION_CONFIDENCE: f32 = 0.65;

/// Request timeout; avoids blocking the ingest pipeline on a stalled
/// model load. On timeout the extractor falls back to `Vec::new()`.
const OLLAMA_TIMEOUT: Duration = Duration::from_secs(30);

/// [`Extractor`] implementation that delegates entity + relation
/// detection to a local Ollama server.
///
/// Construct via [`OllamaExtractor::new`]. The struct is cheap to
/// clone in concept (the `reqwest::Client` is Arc-wrapped internally);
/// we expose it as a constructable field layout so tests can inject a
/// `httpmock`-provided base URL.
pub struct OllamaExtractor {
    /// Blocking HTTP client. Kept sync to stay consistent with
    /// mnem-ingest's tokio-free pipeline .
    client: reqwest::blocking::Client,
    /// Base URL of the Ollama server (no trailing slash).
    base_url: String,
    /// Model name, e.g. `"llama3.2:3b"` or `"qwen2.5:7b"`.
    model: String,
}

impl std::fmt::Debug for OllamaExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaExtractor")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl OllamaExtractor {
    /// Build a new extractor pointing at `base_url` running `model`.
    ///
    /// Pass [`DEFAULT_OLLAMA_URL`] / [`DEFAULT_OLLAMA_MODEL`] to use
    /// the conventional local defaults.
    ///
    /// # Errors
    ///
    /// Returns an [`crate::Error::Extractor`] if the underlying
    /// `reqwest::Client` cannot be constructed (this essentially never
    /// fails on native builds; included for symmetry with WASM-bound
    /// builders that can reject timeouts).
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self, crate::Error> {
        let client = reqwest::blocking::Client::builder()
            .timeout(OLLAMA_TIMEOUT)
            .build()
            .map_err(|e| crate::Error::Extractor(format!("ollama client init: {e}")))?;
        Ok(Self {
            client,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
        })
    }

    /// Call `/api/generate` once and return the parsed schema - or an
    /// empty payload on any failure mode (transport, timeout, schema
    /// rejection). All failures are logged at `warn` level.
    fn invoke(&self, section_text: &str) -> LlmPayload {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind":  { "type": "string" },
                            "text":  { "type": "string" },
                            "start": { "type": "integer" },
                            "end":   { "type": "integer" }
                        },
                        "required": ["kind", "text", "start", "end"]
                    }
                },
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": { "type": "string" },
                            "subj": { "type": "integer" },
                            "obj":  { "type": "integer" }
                        },
                        "required": ["kind", "subj", "obj"]
                    }
                }
            },
            "required": ["entities", "relations"]
        });

        let prompt = format!(
            "Extract entities and relations from the following text. \
             Return STRICTLY JSON matching the schema. \
             Entity 'start' and 'end' are byte offsets into the text. \
             Entity 'kind' is a descriptive label for the entity type (any label is valid, \
             e.g. person, organization, location, product, event, chemical, concept). \
             Relation 'kind' is one of: co_occurs_with, acts_on. \
             Relation 'subj' and 'obj' are indices into the entities array.\n\n\
             TEXT:\n{section_text}"
        );

        let body = serde_json::json!({
            "model":  self.model,
            "prompt": prompt,
            "format": schema,
            "stream": false,
        });

        let url = format!("{}/api/generate", self.base_url);
        let resp = match self.client.post(&url).json(&body).send() {
            Ok(r) => r,
            Err(e) => {
                warn!(error = %e, url = %url, "ollama request failed; falling back to empty extract");
                return LlmPayload::default();
            }
        };
        if !resp.status().is_success() {
            warn!(status = %resp.status(), "ollama returned non-2xx; fallback");
            return LlmPayload::default();
        }
        let envelope: OllamaEnvelope = match resp.json() {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "ollama envelope not JSON; fallback");
                return LlmPayload::default();
            }
        };
        match serde_json::from_str::<LlmPayload>(&envelope.response) {
            Ok(p) => {
                debug!(
                    entities = p.entities.len(),
                    relations = p.relations.len(),
                    "ollama payload parsed"
                );
                p
            }
            Err(e) => {
                warn!(error = %e, "ollama payload schema rejected; fallback");
                LlmPayload::default()
            }
        }
    }
}

impl Extractor for OllamaExtractor {
    fn extract_entities(&self, section: &Section) -> Vec<EntitySpan> {
        let payload = self.invoke(&section.text);
        payload
            .entities
            .into_iter()
            .filter_map(|e| verify_entity(e, &section.text))
            .collect()
    }

    fn extract_relations(&self, entities: &[EntitySpan], section: &Section) -> Vec<RelationSpan> {
        // Re-issue a call that carries the surrounding context, then
        // discard any relation whose subj/obj indices fall outside the
        // caller's already-verified entity list. This keeps the public
        // trait contract (take `entities`, return relations) intact
        // while still letting the LLM see the original text.
        let payload = self.invoke(&section.text);
        payload
            .relations
            .into_iter()
            .filter(|r| (r.subj as usize) < entities.len() && (r.obj as usize) < entities.len())
            .map(|r| RelationSpan {
                kind: r.kind,
                subject_span: r.subj as usize,
                object_span: r.obj as usize,
                confidence: LLM_RELATION_CONFIDENCE,
            })
            .collect()
    }
}

// ---------------- Wire schema ----------------

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct LlmPayload {
    #[serde(default)]
    entities: Vec<LlmEntity>,
    #[serde(default)]
    relations: Vec<LlmRelation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LlmEntity {
    kind: String,
    text: String,
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LlmRelation {
    kind: String,
    subj: u32,
    obj: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct OllamaEnvelope {
    /// Ollama's `/api/generate` returns the model's textual reply in
    /// this field; with `format: <schema>` it is guaranteed to be JSON
    /// matching the schema - but we still re-validate on our side.
    response: String,
}

/// Verify that `entity.text == section_text[start..end]` byte-for-byte.
/// Returns `None` for hallucinated spans - this is non-negotiable.
fn verify_entity(e: LlmEntity, section_text: &str) -> Option<EntitySpan> {
    if e.start > e.end || e.end > section_text.len() {
        return None;
    }
    if !section_text.is_char_boundary(e.start) || !section_text.is_char_boundary(e.end) {
        return None;
    }
    let slice = &section_text[e.start..e.end];
    if slice != e.text {
        return None;
    }
    let kind = e.kind.trim().to_string();
    if kind.is_empty() {
        return None;
    }
    Some(EntitySpan {
        kind,
        text: e.text,
        byte_range: Range {
            start: e.start,
            end: e.end,
        },
        confidence: LLM_ENTITY_CONFIDENCE,
    })
}

// ---------------- Tests ----------------

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;

    fn make_section(body: &str) -> Section {
        Section {
            heading: None,
            depth: 0,
            text: body.to_string(),
            byte_range: 0..body.len(),
        }
    }

    fn ok_response(payload: &serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "model": "llama3.2:3b",
            "created_at": "2026-04-24T00:00:00Z",
            "response": payload.to_string(),
            "done": true
        })
    }

    #[test]
    fn valid_schema_round_trips_verified_spans() {
        let server = MockServer::start();
        // Section: "Alice met Bob."  Indices: Alice=0..5, Bob=10..13.
        let payload = serde_json::json!({
            "entities": [
                { "kind": "person", "text": "Alice", "start": 0, "end": 5 },
                { "kind": "person", "text": "Bob",   "start": 10, "end": 13 }
            ],
            "relations": [
                { "kind": "co_occurs_with", "subj": 0, "obj": 1 }
            ]
        });
        let _mock = server.mock(|when, then| {
            when.method(POST).path("/api/generate");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(ok_response(&payload));
        });
        let ex = OllamaExtractor::new(server.base_url(), "t").unwrap();
        let sec = make_section("Alice met Bob.");
        let ents = ex.extract_entities(&sec);
        assert_eq!(ents.len(), 2);
        assert_eq!(ents[0].text, "Alice");
        assert!((ents[0].confidence - LLM_ENTITY_CONFIDENCE).abs() < f32::EPSILON);
        let rels = ex.extract_relations(&ents, &sec);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].kind, "co_occurs_with");
    }

    #[test]
    fn hallucinated_span_is_rejected() {
        let server = MockServer::start();
        // "Charlie" is NOT in the section text - must be dropped.
        let payload = serde_json::json!({
            "entities": [
                { "kind": "person", "text": "Charlie", "start": 0, "end": 7 }
            ],
            "relations": []
        });
        let _mock = server.mock(|when, then| {
            when.method(POST).path("/api/generate");
            then.status(200).json_body(ok_response(&payload));
        });
        let ex = OllamaExtractor::new(server.base_url(), "t").unwrap();
        let sec = make_section("Alice met Bob.");
        assert!(ex.extract_entities(&sec).is_empty());
    }

    #[test]
    fn http_500_falls_back_to_empty() {
        let server = MockServer::start();
        let _mock = server.mock(|when, then| {
            when.method(POST).path("/api/generate");
            then.status(500);
        });
        let ex = OllamaExtractor::new(server.base_url(), "t").unwrap();
        let sec = make_section("whatever");
        assert!(ex.extract_entities(&sec).is_empty());
    }

    #[test]
    fn schema_invalid_response_is_rejected() {
        let server = MockServer::start();
        let _mock = server.mock(|when, then| {
            when.method(POST).path("/api/generate");
            then.status(200).json_body(serde_json::json!({
                "model": "t",
                "response": "not-json-at-all",
                "done": true
            }));
        });
        let ex = OllamaExtractor::new(server.base_url(), "t").unwrap();
        let sec = make_section("Alice met Bob.");
        assert!(ex.extract_entities(&sec).is_empty());
    }
}
