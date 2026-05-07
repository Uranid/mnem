//! Adapter that lets a [`mnem_extract::KeyBertExtractor`] drop into the
//! [`crate::extract::Extractor`] slot on [`crate::pipeline::Ingester`].
//!
//! Two-phase contract:
//!
//! 1. [`Extractor::prepare`] is called once per file with every
//! section the parser produced. The adapter collects unique
//! section texts and runs them through `Embedder::embed_batch` in
//! a single ORT session.run, caching the resulting vectors.
//! 2. [`Extractor::extract_entities`] is then called per (section,
//! chunk) pair by the pipeline; the adapter looks up the cached
//! section embedding and runs KeyBERT candidate ranking + MMR
//! against it. On a cache miss (e.g. a caller that did not invoke
//! `prepare`) the adapter falls back to a single-section
//! `Embedder::embed`, preserving the original drop-in contract.
//!
//! pre-batching the section pass turns the
//! Bible-scale walltime bottleneck (~1 sequential ORT call per
//! section, dominated by long chapters) into a single ORT batch per
//! file. Same vectors land in `Node.embed`; only the wall-time
//! changes.
//!
//! Relations returned by the statistical miner are mapped to the
//! existing [`RelationSpan`] shape with predicate `"co_occurs_with"`.
//!
//! Gated behind the `keybert` cargo feature so callers who ship only
//! the rule-based baseline pay zero compile / binary cost.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use mnem_embed_providers::Embedder;
use mnem_extract::{Extractor as StatisticalExtractor, KeyBertExtractor};

use crate::extract::{EntitySpan, Extractor, RelationSpan};
use crate::types::Section;

/// Predicate emitted for co-occurrence edges by the KeyBERT adapter.
/// Mirrors the string the rule-based extractor uses so downstream
/// graph consumers don't need to learn a new vocabulary.
pub const KEYBERT_RELATION_LABEL: &str = "co_occurs_with";

/// Confidence stamped onto every [`EntitySpan`] emitted by the
/// adapter. Statistical extraction has a genuine score per candidate
/// (cosine post-MMR), but the ingest pipeline's [`EntitySpan::confidence`]
/// is constrained to `[0.0, 1.0]`; we preserve that by clamping.
pub const KEYBERT_MIN_CONFIDENCE: f32 = 0.0;

/// KeyBERT-backed [`Extractor`] adapter.
///
/// Construct with [`KeyBertAdapter::new`]; hand to
/// [`crate::pipeline::Ingester::with_extractor`] in place of the
/// default [`crate::extract::RuleExtractor`].
pub struct KeyBertAdapter {
    embedder: Arc<dyn Embedder>,
    top_k: usize,
    ngram_range: (usize, usize),
    mmr_diversity: f32,
    pmi_threshold: f32,
    /// ntype label stamped on every entity this adapter emits.
    /// Callers set this via [`KeyBertAdapter::with_label`]; there is no
    /// built-in default, the label vocabulary is entirely up to the caller.
    label: String,
    /// Section-text → embedding cache. Populated by [`Extractor::prepare`]
    /// in one batched `Embedder::embed_batch` call per file; queried
    /// by [`Extractor::extract_entities`] on every (section, chunk)
    /// pair the pipeline iterates over. Misses fall back to a single
    /// `Embedder::embed`, so callers who skip `prepare` still get
    /// correct behaviour. Keyed on the literal section text - same
    /// section content across files therefore reuses one entry,
    /// which is the dedup property `prepare` relies on internally.
    section_cache: Mutex<HashMap<String, Vec<f32>>>,
}

impl std::fmt::Debug for KeyBertAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cached = self.section_cache.lock().map(|c| c.len()).unwrap_or(0);
        f.debug_struct("KeyBertAdapter")
            .field("embedder_model", &self.embedder.model())
            .field("embedder_dim", &self.embedder.dim())
            .field("top_k", &self.top_k)
            .field("ngram_range", &self.ngram_range)
            .field("mmr_diversity", &self.mmr_diversity)
            .field("pmi_threshold", &self.pmi_threshold)
            .field("label", &self.label)
            .field("section_cache_len", &cached)
            .finish()
    }
}

impl KeyBertAdapter {
    /// Build an adapter around the supplied embedder with KeyBERT defaults
    /// (`top_k = 10`, `ngram_range = (1, 3)`, `mmr_diversity = 0.5`,
    /// `pmi_threshold = 1.0`).
    ///
    /// `label` is the ntype string stamped on every entity this adapter emits.
    /// The caller owns the vocabulary, pass whatever label fits your graph
    /// (e.g. `"Keyword"`, `"Tag"`, `"Concept"`, or any domain-specific type).
    #[must_use]
    pub fn new(embedder: Arc<dyn Embedder>, label: impl Into<String>) -> Self {
        Self {
            embedder,
            top_k: mnem_extract::keybert::DEFAULT_TOP_K,
            ngram_range: mnem_extract::keybert::DEFAULT_NGRAM_RANGE,
            mmr_diversity: mnem_extract::keybert::DEFAULT_MMR_DIVERSITY,
            pmi_threshold: mnem_extract::cooccurrence::DEFAULT_PMI_THRESHOLD,
            label: label.into(),
            section_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Override the entity label. Returns `self` for chaining.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Override `top_k`. Returns `self` for chaining.
    #[must_use]
    pub const fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Override the PMI threshold used when mining co-occurrence
    /// edges. Returns `self` for chaining.
    #[must_use]
    pub const fn with_pmi_threshold(mut self, t: f32) -> Self {
        self.pmi_threshold = t;
        self
    }
}

impl Extractor for KeyBertAdapter {
    fn prepare(&self, sections: &[Section]) -> Result<(), crate::error::Error> {
        // Collect unique non-empty section texts. Pipelines that
        // re-emit identical content across sections (e.g. boilerplate
        // headers) pay only once.
        let mut unique: Vec<&str> = Vec::with_capacity(sections.len());
        let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for s in sections {
            if s.text.is_empty() {
                continue;
            }
            if seen.insert(s.text.as_str()) {
                unique.push(s.text.as_str());
            }
        }
        if unique.is_empty() {
            return Ok(());
        }

        // Best-effort batch embed. Failure here downgrades silently
        // to the per-section lazy path in `extract_entities` (which
        // has its own error swallow), so a transient embedder hiccup
        // never aborts the whole file ingest. This matches the
        // legacy "skip section on embed failure" behaviour the
        // adapter shipped with before pre-batching landed.
        let vecs = match self.embedder.embed_batch(&unique) {
            Ok(v) => v,
            Err(_e) => return Ok(()),
        };

        if let Ok(mut cache) = self.section_cache.lock() {
            // Store result indexed by the same text key
            // `extract_entities` will look up. `embed_batch`'s
            // contract preserves order, so unique[i] aligns with
            // vecs[i].
            for (text, vec) in unique.into_iter().zip(vecs) {
                cache.entry(text.to_string()).or_insert(vec);
            }
        }
        Ok(())
    }

    fn extract_entities(&self, section: &Section) -> Vec<EntitySpan> {
        let text = &section.text;
        if text.is_empty() {
            return Vec::new();
        }

        // Cache hit path: `prepare` populated this entry in one
        // batched ORT call; we just clone the f32 vector. Cache miss
        // path: caller skipped `prepare` (or the batch failed at
        // prepare time); embed the section in a single call so the
        // adapter still works end-to-end. Either path produces the
        // same vector for the same text on the same embedder.
        let cached = self
            .section_cache
            .lock()
            .ok()
            .and_then(|cache| cache.get(text).cloned());
        let section_embed = match cached {
            Some(v) => v,
            None => match self.embedder.embed(text) {
                Ok(v) => v,
                Err(_) => return Vec::new(),
            },
        };

        let kb = KeyBertExtractor {
            embedder: self.embedder.as_ref(),
            top_k: self.top_k,
            ngram_range: self.ngram_range,
            mmr_diversity: self.mmr_diversity,
        };
        let entities = kb.extract_entities(text, &section_embed);
        entities
            .into_iter()
            .map(|e| EntitySpan {
                kind: self.label.clone(),
                text: e.mention,
                byte_range: e.span.0..e.span.1,
                confidence: e.score.clamp(KEYBERT_MIN_CONFIDENCE, 1.0),
            })
            .collect()
    }

    fn extract_relations(&self, entities: &[EntitySpan], section: &Section) -> Vec<RelationSpan> {
        if entities.len() < 2 {
            return Vec::new();
        }
        // Map EntitySpan → mnem_extract::Entity keeping the original
        // index so we can refer back to it.
        let bridged: Vec<mnem_extract::Entity> = entities
            .iter()
            .map(|e| mnem_extract::Entity {
                mention: e.text.clone(),
                score: e.confidence,
                span: (e.byte_range.start, e.byte_range.end),
            })
            .collect();
        let rels = mnem_extract::mine_relations(
            &section.text,
            &bridged,
            self.pmi_threshold,
            mnem_extract::ExtractionSource::Statistical,
        );

        // Reverse-lookup each Relation.src / .dst back to the
        // EntitySpan index the pipeline expects.
        let index_of =
            |mention: &str| -> Option<usize> { entities.iter().position(|e| e.text == mention) };
        let mut out = Vec::with_capacity(rels.len());
        for r in rels {
            let (Some(si), Some(oi)) = (index_of(&r.src), index_of(&r.dst)) else {
                continue;
            };
            out.push(RelationSpan {
                kind: KEYBERT_RELATION_LABEL.to_string(),
                subject_span: si,
                object_span: oi,
                confidence: r.weight.clamp(0.0, 1.0),
            });
        }
        out
    }
}
