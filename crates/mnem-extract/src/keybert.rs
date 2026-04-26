//! KeyBERT-style statistical keyword / entity extractor.
//!
//! The algorithm mirrors the canonical KeyBERT pipeline (Grootendorst
//! 2020) but runs entirely against mnem's existing synchronous
//! [`Embedder`] trait - no Python, no sklearn, no ONNX-only path.
//!
//! 1. **Tokenise** the chunk text with `unicode-segmentation` into
//! words and sentence boundaries.
//! 2. **Enumerate n-gram candidates** of length `ngram_range.0 ..=
//! ngram_range.1`, skipping candidates that are pure stop-word
//! sequences.
//! 3. **Deduplicate** candidates to their earliest span; sort the
//! deduped list lexicographically for determinism.
//! 4. **Embed** each candidate via `Embedder::embed`; compute cosine
//! similarity against the caller-supplied `chunk_embed`.
//! 5. **MMR-diversify**: iteratively pick the highest-scoring
//! candidate after subtracting `mmr_diversity * max_sim_to_picked`.
//! Stable lex tiebreaks on exact-tie scores.
//!
//! The implementation allocates once per call and keeps all scoring in
//! `f64` to dodge `f32` summation drift on long inputs.

use mnem_embed_providers::Embedder;
use tracing::trace;
use unicode_segmentation::UnicodeSegmentation;

use crate::traits::{Entity, ExtractionSource, Extractor, Relation};

/// Default KeyBERT extractor parameters. Picked to match the KeyBERT
/// paper's out-of-the-box behaviour: 1–3-grams, top-10 keywords,
/// MMR diversity 0.5.
pub const DEFAULT_NGRAM_RANGE: (usize, usize) = (1, 3);
/// Number of entities returned per call by default.
pub const DEFAULT_TOP_K: usize = 10;
/// Default MMR diversity coefficient (λ in the KeyBERT paper).
/// 0.0 → pure cosine ranking; 1.0 → maximal redundancy penalty.
pub const DEFAULT_MMR_DIVERSITY: f32 = 0.5;

/// KeyBERT-style extractor.
///
/// Holds a borrowed [`Embedder`] reference; the caller owns the
/// concrete provider (Ollama / OpenAI / ONNX / mock) and threads it in
/// for the duration of an ingest run.
pub struct KeyBertExtractor<'a> {
 /// Embedder used to encode candidate n-grams. MUST be the same
 /// provider + model that produced `chunk_embed`, otherwise cosine
 /// similarity is meaningless.
 pub embedder: &'a dyn Embedder,
 /// Number of entities to return per call. See [`DEFAULT_TOP_K`].
 pub top_k: usize,
 /// Inclusive `(min_n, max_n)` n-gram length range.
 pub ngram_range: (usize, usize),
 /// MMR diversity coefficient in `[0.0, 1.0]`.
 pub mmr_diversity: f32,
}

impl std::fmt::Debug for KeyBertExtractor<'_> {
 fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
 // `dyn Embedder` is not Debug; project the relevant fields
 // instead so tracing / assertion output still identifies the
 // configured model.
 f.debug_struct("KeyBertExtractor")
 .field("embedder_model", &self.embedder.model())
 .field("embedder_dim", &self.embedder.dim())
 .field("top_k", &self.top_k)
 .field("ngram_range", &self.ngram_range)
 .field("mmr_diversity", &self.mmr_diversity)
 .finish()
 }
}

impl<'a> KeyBertExtractor<'a> {
 /// Construct a KeyBERT extractor with default parameters.
 #[must_use]
 pub fn new(embedder: &'a dyn Embedder) -> Self {
 Self {
 embedder,
 top_k: DEFAULT_TOP_K,
 ngram_range: DEFAULT_NGRAM_RANGE,
 mmr_diversity: DEFAULT_MMR_DIVERSITY,
 }
 }

 /// Override `top_k`. Returns `self` for chaining.
 #[must_use]
 pub const fn with_top_k(mut self, k: usize) -> Self {
 self.top_k = k;
 self
 }

 /// Override n-gram range. `min` must be >= 1; callers that pass 0
 /// are clamped to 1. Returns `self` for chaining.
 #[must_use]
 pub const fn with_ngram_range(mut self, min: usize, max: usize) -> Self {
 let min = if min == 0 { 1 } else { min };
 let max = if max < min { min } else { max };
 self.ngram_range = (min, max);
 self
 }

 /// Override MMR diversity coefficient. Returns `self` for
 /// chaining. Callers passing values outside `[0.0, 1.0]` get them
 /// clamped.
 #[must_use]
 pub fn with_mmr_diversity(mut self, lambda: f32) -> Self {
 self.mmr_diversity = lambda.clamp(0.0, 1.0);
 self
 }
}

impl Extractor for KeyBertExtractor<'_> {
 fn extract_entities(&self, text: &str, chunk_embed: &[f32]) -> Vec<Entity> {
 // 1. collect word spans.
 let words: Vec<(usize, &str)> = text.unicode_word_indices().collect();
 if words.is_empty() || chunk_embed.is_empty() {
 return Vec::new();
 }

 // 2. enumerate n-gram candidates of length min..=max, deduped
 // to their first occurrence span.
 let (min_n, max_n) = self.ngram_range;
 let mut candidates: Vec<Candidate> = Vec::new();
 let mut seen_keys: std::collections::BTreeMap<String, usize> =
 std::collections::BTreeMap::new();
 for start_idx in 0..words.len() {
 for n in min_n..=max_n {
 if start_idx + n > words.len() {
 break;
 }
 let (first_byte, first_tok) = words[start_idx];
 let (last_byte, last_tok) = words[start_idx + n - 1];
 let end_byte = last_byte + last_tok.len();
 // collect surface form over the exact byte span so
 // punctuation between words is preserved when present.
 let surface = &text[first_byte..end_byte];
 let normalised = normalise(surface);
 if normalised.is_empty() {
 continue;
 }
 // reject pure-stopword n-grams (single-token
 // stopwords are also rejected).
 if (start_idx..start_idx + n).all(|i| is_stopword(words[i].1)) {
 continue;
 }
 // skip candidates that don't have any alphanumeric
 // content (e.g. all-punctuation windows).
 if !normalised.chars().any(char::is_alphanumeric) {
 continue;
 }
 // For short single-word candidates, require length > 1
 // to drop noise like "a", "I".
 if n == 1 && first_tok.chars().count() < 2 {
 continue;
 }
 let key = normalised.clone();
 if let std::collections::btree_map::Entry::Vacant(e) = seen_keys.entry(key.clone())
 {
 e.insert(candidates.len());
 candidates.push(Candidate {
 key,
 surface: surface.to_string(),
 span: (first_byte, end_byte),
 });
 }
 }
 }

 if candidates.is_empty() {
 return Vec::new();
 }

 // 3. sort candidates lexicographically before embedding - this
 // is the determinism anchor even if the enumeration loop
 // order changes.
 candidates.sort_by(|a, b| a.key.cmp(&b.key));

 // 4. embed every candidate via the provider's batch call when
 // available (5-10x faster on ONNX/OpenAI vs the per-text
 // loop; Ollama transparently falls back to sequential
 // `embed` per its `embed_batch` default impl). Compute
 // cosine vs `chunk_embed`. Result vectors line up with
 // `candidates` index-for-index.
 let mut scored: Vec<Scored> = Vec::with_capacity(candidates.len());
 let surfaces: Vec<&str> = candidates.iter().map(|c| c.surface.as_str()).collect();
 match self.embedder.embed_batch(&surfaces) {
 Ok(vecs) => {
 for (c, vec) in candidates.iter().zip(vecs.into_iter()) {
 if vec.len() != chunk_embed.len() {
 trace!(
 cand = %c.key,
 expected = chunk_embed.len(),
 got = vec.len(),
 "dim mismatch, skipping candidate",
 );
 continue;
 }
 let sim = cosine(&vec, chunk_embed);
 scored.push(Scored {
 candidate: c.clone(),
 embed: vec,
 sim,
 });
 }
 }
 Err(batch_err) => {
 // Per-candidate fallback: a single bad input shouldn't
 // wipe a chunk's entire extraction. Keep the same
 // skip-on-error / dim-mismatch contract as the
 // pre-batch implementation.
 trace!(?batch_err, "embed_batch failed, falling back to per-candidate");
 for c in &candidates {
 match self.embedder.embed(&c.surface) {
 Ok(vec) => {
 if vec.len() != chunk_embed.len() {
 trace!(
 cand = %c.key,
 expected = chunk_embed.len(),
 got = vec.len(),
 "dim mismatch, skipping candidate",
 );
 continue;
 }
 let sim = cosine(&vec, chunk_embed);
 scored.push(Scored {
 candidate: c.clone(),
 embed: vec,
 sim,
 });
 }
 Err(err) => {
 trace!(cand = %c.key, ?err, "embed failed, skipping candidate");
 }
 }
 }
 }
 }
 if scored.is_empty() {
 return Vec::new();
 }

 // 5. MMR diversify.
 let picks = mmr_select(&scored, self.top_k, self.mmr_diversity);
 picks
 .into_iter()
 .map(|(s, mmr_score)| Entity {
 mention: s.candidate.surface.clone(),
 #[allow(clippy::cast_possible_truncation)]
 score: (mmr_score as f32).clamp(-1.0, 1.0),
 span: s.candidate.span,
 })
 .collect()
 }

 fn extract_relations(&self, text: &str, entities: &[Entity]) -> Vec<Relation> {
 crate::cooccurrence::mine_relations(
 text,
 entities,
 crate::cooccurrence::DEFAULT_PMI_THRESHOLD,
 ExtractionSource::Statistical,
 )
 }
}

// -------- internals --------

#[derive(Debug, Clone)]
struct Candidate {
 /// Lower-cased, whitespace-normalised lookup key used for dedup
 /// and lex sort. The original surface form is preserved separately
 /// so rendering stays faithful to the source.
 key: String,
 surface: String,
 span: (usize, usize),
}

#[derive(Debug, Clone)]
struct Scored {
 candidate: Candidate,
 embed: Vec<f32>,
 sim: f64,
}

/// Lower-case + collapse whitespace. Pure-Rust, no regex so the crate
/// stays tiny; the extractor runs per candidate so the loop is cheap.
fn normalise(s: &str) -> String {
 let mut out = String::with_capacity(s.len());
 let mut prev_ws = true;
 for ch in s.chars() {
 if ch.is_whitespace() {
 if !prev_ws {
 out.push(' ');
 prev_ws = true;
 }
 } else {
 for lc in ch.to_lowercase() {
 out.push(lc);
 }
 prev_ws = false;
 }
 }
 if out.ends_with(' ') {
 out.pop();
 }
 out
}

/// Minimal English stop-word list. Deliberately short - the goal is
/// to reject pure-stopword n-grams like "the dog" from dominating the
/// top-k, not to replicate NLTK's 180-word list. Callers that need
/// broader coverage should post-filter the returned Entities.
#[rustfmt::skip]
const STOPWORDS: &[&str] = &[
 "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
 "from", "has", "have", "he", "her", "hers", "him", "his", "i",
 "if", "in", "into", "is", "it", "its", "me", "my", "no", "not",
 "of", "on", "or", "our", "ours", "over", "she", "so", "that",
 "the", "their", "theirs", "them", "then", "there", "they",
 "this", "those", "to", "too", "us", "was", "we", "were", "what",
 "when", "where", "which", "while", "who", "whom", "why", "will",
 "with", "you", "your", "yours",
];

fn is_stopword(tok: &str) -> bool {
 let lc: String = tok.chars().flat_map(char::to_lowercase).collect();
 STOPWORDS.binary_search(&lc.as_str()).is_ok()
}

/// Cosine similarity in `f64` to avoid `f32` accumulation drift.
/// Returns 0.0 for zero-magnitude inputs rather than `NaN`.
fn cosine(a: &[f32], b: &[f32]) -> f64 {
 debug_assert_eq!(a.len(), b.len());
 let mut dot = 0.0_f64;
 let mut na = 0.0_f64;
 let mut nb = 0.0_f64;
 for (x, y) in a.iter().zip(b.iter()) {
 let xf = f64::from(*x);
 let yf = f64::from(*y);
 dot += xf * yf;
 na += xf * xf;
 nb += yf * yf;
 }
 if na <= 0.0 || nb <= 0.0 {
 return 0.0;
 }
 dot / (na.sqrt() * nb.sqrt())
}

/// Iteratively select up to `top_k` candidates by MMR.
///
/// Score function: `sim(cand, chunk) - lambda * max_i sim(cand, picked_i)`.
/// Ties on score are broken by the candidate `key` (lex order) for
/// determinism across runs / platforms.
fn mmr_select(scored: &[Scored], top_k: usize, lambda: f32) -> Vec<(Scored, f64)> {
 let lambda = f64::from(lambda);
 let k = top_k.min(scored.len());
 let mut picks: Vec<(Scored, f64)> = Vec::with_capacity(k);
 let mut remaining: Vec<usize> = (0..scored.len()).collect();

 while picks.len() < k && !remaining.is_empty() {
 let mut best_idx_in_remaining: Option<usize> = None;
 let mut best_score: f64 = f64::NEG_INFINITY;
 let mut best_key: Option<&str> = None;
 for (pos, &i) in remaining.iter().enumerate() {
 let c = &scored[i];
 let redundancy = picks
 .iter()
 .map(|(p, _)| cosine(&c.embed, &p.embed))
 .fold(f64::NEG_INFINITY, f64::max)
 .max(0.0_f64);
 let redundancy = if picks.is_empty() { 0.0 } else { redundancy };
 let mmr = c.sim - lambda * redundancy;
 let tiebreak = c.candidate.key.as_str();
 let better = mmr > best_score
 || (approx_eq(mmr, best_score) && best_key.is_none_or(|bk| tiebreak < bk));
 if better {
 best_score = mmr;
 best_idx_in_remaining = Some(pos);
 best_key = Some(tiebreak);
 }
 }
 match best_idx_in_remaining {
 Some(pos) => {
 let i = remaining.swap_remove(pos);
 picks.push((scored[i].clone(), best_score));
 }
 None => break,
 }
 }
 picks
}

/// `f64` equality up to 1e-9; enough for our cosine-derived tiebreaks.
fn approx_eq(a: f64, b: f64) -> bool {
 (a - b).abs() < 1e-9
}

#[cfg(test)]
mod tests {
 use super::*;

 #[test]
 fn normalise_collapses_whitespace_and_lowercases() {
 assert_eq!(normalise(" Hello World "), "hello world");
 assert_eq!(normalise("MixedCase"), "mixedcase");
 }

 #[test]
 fn stopwords_are_sorted_for_binary_search() {
 let mut sorted = STOPWORDS.to_vec();
 sorted.sort_unstable();
 assert_eq!(sorted.as_slice(), STOPWORDS);
 }

 #[test]
 fn cosine_identity() {
 let v = vec![1.0_f32, 2.0, 3.0];
 let c = cosine(&v, &v);
 assert!((c - 1.0).abs() < 1e-9, "cosine(v, v) = {c}");
 }

 #[test]
 fn cosine_zero_magnitude_returns_zero() {
 let a = vec![0.0_f32; 8];
 let b = vec![1.0_f32; 8];
 assert_eq!(cosine(&a, &b), 0.0);
 }
}
