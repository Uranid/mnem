//! Co-occurrence relation miner - PMI-weighted edges between entities
//! that share a sentence.
//!
//! This is the statistical-relation companion to
//! [`crate::keybert::KeyBertExtractor`]: once entity mentions are
//! located, sentence-local co-occurrence tells us which pairs are
//! semantically linked without invoking an LLM.
//!
//! ## Algorithm
//!
//! 1. Segment `text` into sentences via
//!    `unicode_segmentation::UnicodeSentences` (locale-insensitive
//!    but Unicode-correct; handles `…`, `。`, etc).
//! 2. For each sentence, mark the set of entity indices whose `span`
//!    overlaps the sentence byte range.
//! 3. Accumulate per-entity counts `c(e)` and per-pair counts
//!    `c(e_i, e_j)` across sentences.
//! 4. For every pair with `c(e_i, e_j) > 0`, compute
//!    `PMI = log( P(e_i, e_j) / ( P(e_i) * P(e_j) ) )` against the
//!    total sentence count.
//! 5. Emit one [`crate::traits::Relation`] per pair whose PMI exceeds
//!    `threshold` (default `1.0`), with a stable `(src, dst)` ordering
//!    - entities sort by their lower-cased mention so reversed
//!    `(A, B)` and `(B, A)` pairs collapse to the same edge.

use unicode_segmentation::UnicodeSegmentation;

use crate::traits::{Entity, ExtractionSource, Relation};

/// Default PMI threshold. Pairs at or below this are suppressed to
/// keep the emitted edge set readable.
pub const DEFAULT_PMI_THRESHOLD: f32 = 1.0;

/// Reusable miner if a caller wants to pre-bind a threshold + source.
/// The free function [`mine_relations`] covers the common path.
#[derive(Debug, Clone)]
pub struct CoOccurrenceMiner {
    /// Emit only pairs whose PMI exceeds this threshold.
    pub threshold: f32,
    /// Provenance tag applied to every emitted [`Relation`].
    pub source: ExtractionSource,
}

impl Default for CoOccurrenceMiner {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_PMI_THRESHOLD,
            source: ExtractionSource::Statistical,
        }
    }
}

impl CoOccurrenceMiner {
    /// Mine relations with this miner's configured threshold + source.
    #[must_use]
    pub fn mine(&self, text: &str, entities: &[Entity]) -> Vec<Relation> {
        mine_relations(text, entities, self.threshold, self.source.clone())
    }
}

/// Free-function form of the miner.
///
/// Returns relations deterministically sorted by
/// `(src, dst, weight_bits)` so two runs against the same input
/// produce byte-identical output.
#[must_use]
pub fn mine_relations(
    text: &str,
    entities: &[Entity],
    threshold: f32,
    source: ExtractionSource,
) -> Vec<Relation> {
    if entities.len() < 2 || text.is_empty() {
        return Vec::new();
    }

    // Collect sentence byte ranges. Sentences are inclusive of their
    // trailing punctuation, which is fine here - we only use them to
    // scope entity co-occurrence.
    let sentences: Vec<(usize, usize)> = text
        .split_sentence_bound_indices()
        .map(|(start, frag)| (start, start + frag.len()))
        .filter(|(s, e)| e > s)
        .collect();
    if sentences.is_empty() {
        return Vec::new();
    }

    let n = entities.len();
    let mut per_entity: Vec<u32> = vec![0; n];
    // Upper-triangle pair counts keyed by (i, j) with i < j. Using a
    // BTreeMap keeps iteration deterministic without an extra sort.
    let mut per_pair: std::collections::BTreeMap<(usize, usize), u32> =
        std::collections::BTreeMap::new();
    let sent_count = sentences.len() as u32;

    for (s_start, s_end) in &sentences {
        // Set of entity indices whose span overlaps this sentence.
        let mut present: Vec<usize> = Vec::new();
        for (idx, e) in entities.iter().enumerate() {
            let (e_start, e_end) = e.span;
            // Half-open span overlap: [e_start, e_end) ∩ [s_start, s_end)
            if e_start < *s_end && e_end > *s_start {
                present.push(idx);
            }
        }
        if present.is_empty() {
            continue;
        }
        // Entity may appear twice in the same sentence (same mention
        // extracted with different spans); count it once per sentence
        // for the marginal, following standard PMI conventions.
        present.sort_unstable();
        present.dedup();
        for &i in &present {
            per_entity[i] += 1;
        }
        for i in 0..present.len() {
            for j in (i + 1)..present.len() {
                let a = present[i];
                let b = present[j];
                let key = if a < b { (a, b) } else { (b, a) };
                *per_pair.entry(key).or_insert(0) += 1;
            }
        }
    }

    let total = f64::from(sent_count);
    let mut out: Vec<Relation> = Vec::with_capacity(per_pair.len());
    for ((i, j), c_ij) in per_pair {
        let c_i = per_entity[i];
        let c_j = per_entity[j];
        if c_ij == 0 || c_i == 0 || c_j == 0 {
            continue;
        }
        let p_ij = f64::from(c_ij) / total;
        let p_i = f64::from(c_i) / total;
        let p_j = f64::from(c_j) / total;
        let pmi = (p_ij / (p_i * p_j)).ln();
        #[allow(clippy::cast_possible_truncation)]
        let pmi_f32 = pmi as f32;
        if !pmi_f32.is_finite() || pmi_f32 <= threshold {
            continue;
        }
        // Order (src, dst) by lower-cased mention so reciprocal pairs
        // collapse to one canonical direction.
        let m_i = &entities[i].mention;
        let m_j = &entities[j].mention;
        let (src, dst) = if lc(m_i) <= lc(m_j) {
            (m_i.clone(), m_j.clone())
        } else {
            (m_j.clone(), m_i.clone())
        };
        out.push(Relation {
            src,
            dst,
            weight: pmi_f32,
            source: source.clone(),
        });
    }

    // Final sort for deterministic output regardless of BTreeMap
    // iteration order on different toolchains.
    out.sort_by(|a, b| {
        a.src
            .cmp(&b.src)
            .then_with(|| a.dst.cmp(&b.dst))
            .then_with(|| a.weight.to_bits().cmp(&b.weight.to_bits()))
    });
    out
}

fn lc(s: &str) -> String {
    s.chars().flat_map(char::to_lowercase).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ent(mention: &str, span: (usize, usize)) -> Entity {
        Entity {
            mention: mention.to_string(),
            score: 0.5,
            span,
        }
    }

    #[test]
    fn empty_inputs_return_empty() {
        let out = mine_relations("", &[], 0.0, ExtractionSource::Statistical);
        assert!(out.is_empty());
    }

    #[test]
    fn single_entity_returns_empty() {
        let text = "The dog ran fast.";
        let entities = vec![ent("dog", (4, 7))];
        let out = mine_relations(text, &entities, 0.0, ExtractionSource::Statistical);
        assert!(out.is_empty());
    }

    #[test]
    fn cooccurring_pair_emits_positive_pmi() {
        let text = "Alice met Bob. They shook hands.";
        let entities = vec![ent("Alice", (0, 5)), ent("Bob", (10, 13))];
        let out = mine_relations(text, &entities, 0.0, ExtractionSource::Statistical);
        assert_eq!(out.len(), 1);
        assert!(out[0].weight > 0.0);
    }
}
