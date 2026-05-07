//! Entity + relation extraction over parsed [`Section`]s.
//!
//! Entity extraction is delegated entirely to the configured
//! [`mnem_ner_providers::NerProvider`]. The default is
//! [`mnem_ner_providers::RuleNer`] (capitalized-phrase heuristic).
//! Swap for [`mnem_ner_providers::NullNer`] or any future provider via
//! [`IngestConfig::ner`]. Provider labels pass through unconditionally,
//! there is no fixed vocabulary.
//!
//! Relations are proximity-based: two entity spans whose start positions
//! are within `window_tokens` of each other in the same [`Section`] get a
//! candidate `"co_occurs_with"` edge (confidence `0.40`). A lightweight
//! verb-between check promotes that to `"acts_on"` (confidence `0.50`)
//! when a token like `"joined"`, `"founded"`, `"acquired"`, `"owns"`, or
//! `"hired"` sits between the two spans.

use std::ops::Range;
use std::sync::Arc;

use mnem_ner_providers::NerProvider;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::types::{ExtractorConfig, Section};

// ---------------- Types ----------------

/// A single entity mention inside a [`Section`].
///
/// `byte_range` refers to offsets within the section's `text` field
/// (not the original source). Downstream commit code combines it with
/// `Section::byte_range` when provenance-accurate source offsets are
/// needed.
///
/// `kind` is the namespaced ntype string (e.g. `"Entity:Person"`).
/// Using a `String` keeps the type open for any NER provider label vocabulary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntitySpan {
    /// Namespaced ntype label string (e.g. `"Entity:Person"`).
    pub kind: String,
    /// Verbatim surface string as it appears in the section text.
    pub text: String,
    /// Byte range within the section's `text`.
    pub byte_range: Range<usize>,
    /// Heuristic confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

/// A candidate relation between two entities in the same section.
///
/// `subject_span` and `object_span` are indices into the entity vector
/// returned by the same extract call. Relation identifiers are plain
/// strings to keep the shape open; callers emit `"co_occurs_with"` or
/// `"acts_on"` today.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelationSpan {
    /// Predicate label (e.g. `"co_occurs_with"`, `"acts_on"`).
    pub kind: String,
    /// Index of the subject entity within the accompanying `Vec<EntitySpan>`.
    pub subject_span: usize,
    /// Index of the object entity within the accompanying `Vec<EntitySpan>`.
    pub object_span: usize,
    /// Heuristic confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

// ---------------- Extractor trait ----------------

/// Pluggable entity + relation extractor.
///
/// Implementations must be `Send + Sync` so the [`crate::Ingester`]
/// façade can hand them across thread boundaries in batch ingest paths
/// scheduled by CLI/HTTP wrappers in later waves.
pub trait Extractor: Send + Sync {
    /// Extract entity mentions from a single section.
    fn extract_entities(&self, section: &Section) -> Vec<EntitySpan>;

    /// Extract candidate relations between already-extracted entities.
    fn extract_relations(&self, entities: &[EntitySpan], section: &Section) -> Vec<RelationSpan>;

    /// Optional pre-extraction hook. Called once per file by
    /// [`crate::pipeline::Ingester::ingest`] BEFORE any
    /// `extract_entities` / `extract_relations` call, with the full
    /// list of sections the file produced. The default implementation
    /// is a no-op, so existing extractors keep their behaviour.
    ///
    /// # Errors
    ///
    /// Returns whatever the implementation chooses; the pipeline
    /// passes the error through.
    fn prepare(&self, _sections: &[Section]) -> Result<(), crate::error::Error> {
        Ok(())
    }
}

// ---------------- Default rule extractor ----------------

/// [`Extractor`] implementation that delegates entity detection to the
/// configured [`NerProvider`] and proximity-based relation detection to an
/// internal verb-window regex.
///
/// Construct via [`RuleExtractor::new`] or [`RuleExtractor::with_default_ner`].
pub struct RuleExtractor {
    cfg: ExtractorConfig,
    verb_window: Regex,
    ner: Arc<dyn NerProvider>,
}

impl std::fmt::Debug for RuleExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuleExtractor")
            .field("cfg", &self.cfg)
            .field("ner", &self.ner.provider_id())
            .finish()
    }
}

impl RuleExtractor {
    /// Build a new extractor from configuration and a NER provider.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn new(cfg: ExtractorConfig, ner: Arc<dyn NerProvider>) -> Self {
        let verb_window = Regex::new(
            r"(?i)\b(?:joined|founded|acquired|owns|hired|created|launched|bought|leads|runs)\b",
        )
        .expect("verb regex compiles");
        Self {
            cfg,
            verb_window,
            ner,
        }
    }

    /// Build with the default [`mnem_ner_providers::RuleNer`] provider.
    #[must_use]
    pub fn with_default_ner(cfg: ExtractorConfig) -> Self {
        Self::new(cfg, Arc::new(mnem_ner_providers::RuleNer))
    }
}

impl Default for RuleExtractor {
    fn default() -> Self {
        Self::with_default_ner(ExtractorConfig::default())
    }
}

impl Extractor for RuleExtractor {
    fn extract_entities(&self, section: &Section) -> Vec<EntitySpan> {
        if !self.cfg.extract_ner {
            return Vec::new();
        }
        let text = section.text.as_str();
        let mut out: Vec<EntitySpan> = self
            .ner
            .extract(text)
            .into_iter()
            .filter_map(|ne| {
                if ne.label.trim().is_empty() {
                    return None;
                }
                let slice = text.get(ne.byte_start..ne.byte_end)?.to_string();
                if slice.is_empty() {
                    return None;
                }
                Some(EntitySpan {
                    kind: ne.label,
                    text: slice,
                    byte_range: ne.byte_start..ne.byte_end,
                    confidence: ne.confidence,
                })
            })
            .collect();

        out.sort_by(|a, b| {
            a.byte_range
                .start
                .cmp(&b.byte_range.start)
                .then_with(|| a.kind.as_str().cmp(b.kind.as_str()))
        });
        out.dedup_by(|a, b| a.byte_range == b.byte_range && a.kind == b.kind);
        out
    }

    fn extract_relations(&self, entities: &[EntitySpan], section: &Section) -> Vec<RelationSpan> {
        if entities.len() < 2 {
            return Vec::new();
        }
        let text = section.text.as_str();
        let window = self.cfg.relation_window_tokens;
        let mut out = Vec::new();

        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let a = &entities[i];
                let b = &entities[j];
                if a.byte_range.end > b.byte_range.start {
                    continue;
                }
                let between = &text[a.byte_range.end..b.byte_range.start];
                let tokens_between = between.split_whitespace().count();
                if tokens_between > window {
                    continue;
                }
                let (kind, conf) = if self.verb_window.is_match(between) {
                    ("acts_on".to_string(), 0.50_f32)
                } else {
                    ("co_occurs_with".to_string(), 0.40_f32)
                };
                out.push(RelationSpan {
                    kind,
                    subject_span: i,
                    object_span: j,
                    confidence: conf,
                });
            }
        }
        out
    }
}

// ---------------- Free helpers ----------------

/// Run [`RuleExtractor::default`] once against a section.
#[must_use]
pub fn extract_entities(section: &Section) -> Vec<EntitySpan> {
    RuleExtractor::default().extract_entities(section)
}

/// Run [`RuleExtractor::default`] once to derive relations.
#[must_use]
pub fn extract_relations(entities: &[EntitySpan], section: &Section) -> Vec<RelationSpan> {
    RuleExtractor::default().extract_relations(entities, section)
}

// ---------------- Tests ----------------

#[cfg(test)]
mod tests {
    use super::*;

    fn section(text: &str) -> Section {
        Section {
            heading: None,
            depth: 0,
            text: text.to_string(),
            byte_range: 0..text.len(),
        }
    }

    #[test]
    fn ner_detects_person() {
        let s = section("Alice Johnson met Bob Lee at the lobby.");
        let ents = extract_entities(&s);
        assert!(
            ents.iter().any(|e| e.text == "Alice Johnson"),
            "got: {ents:?}"
        );
        assert!(ents.iter().any(|e| e.text == "Bob Lee"), "got: {ents:?}");
    }

    #[test]
    fn ner_detects_org() {
        let s = section("Acme Corp and Foo Inc signed the deal.");
        let ents = extract_entities(&s);
        assert!(ents.iter().any(|e| e.text == "Acme Corp"), "got: {ents:?}");
    }

    #[test]
    fn ner_single_token_not_detected() {
        let s = section("Alice then left.");
        let ents = extract_entities(&s);
        assert!(ents.is_empty(), "single-token should not match: {ents:?}");
    }

    #[test]
    fn relations_proximity_co_occurs() {
        let s = section("Alice Johnson met Bob Lee today.");
        let ents = extract_entities(&s);
        let rels = extract_relations(&ents, &s);
        assert!(
            rels.iter().any(|r| r.kind == "co_occurs_with"),
            "got rels: {rels:?}"
        );
    }

    #[test]
    fn relations_verb_between_becomes_acts_on() {
        let s = section("Alice Johnson founded Acme Corp in 2022.");
        let ents = extract_entities(&s);
        let rels = extract_relations(&ents, &s);
        assert!(
            rels.iter().any(|r| r.kind == "acts_on"),
            "got rels: {rels:?}, ents: {ents:?}"
        );
    }

    #[test]
    fn confidence_in_unit_range() {
        let s = section("Alice Johnson and Bob Lee work at Acme Corp.");
        let ents = extract_entities(&s);
        assert!(!ents.is_empty(), "expected at least one entity from NER");
        for e in &ents {
            assert!(
                (0.0..=1.0).contains(&e.confidence),
                "confidence {} out of [0,1] for {:?}",
                e.confidence,
                e
            );
        }
    }

    #[test]
    fn null_ner_produces_no_entities() {
        use mnem_ner_providers::NullNer;
        let ext = RuleExtractor::new(ExtractorConfig::default(), Arc::new(NullNer));
        let s = section("Alice Johnson founded Acme Corp.");
        assert!(
            ext.extract_entities(&s).is_empty(),
            "NullNer must produce nothing"
        );
    }

    #[test]
    fn extract_ner_false_produces_no_entities() {
        let cfg = ExtractorConfig {
            extract_ner: false,
            ..ExtractorConfig::default()
        };
        let ext = RuleExtractor::with_default_ner(cfg);
        let s = section("Alice Johnson founded Acme Corp.");
        assert!(ext.extract_entities(&s).is_empty());
    }
}
