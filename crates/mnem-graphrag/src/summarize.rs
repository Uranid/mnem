//! Centroid + MMR extractive community summarizer.
//!
//! Given a set of sentences (e.g. the text spans belonging to a
//! community of nodes), score each sentence by its distance to the
//! community centroid, optionally to a query vector, and by a
//! graph-centrality fallback (degree today; PPR once E2 lands).
//! Then greedily pick `k` sentences with MMR diversity so the output
//! is not dominated by near-duplicates.
//!
//! # Determinism
//!
//! The function is input-order-insensitive: callers may pass
//! sentences in any order and the resulting [`Summary`] is byte-for-byte
//! identical. This is achieved by sorting the working set by the
//! SHA-256-style content hash of each sentence (we use BLAKE3 for
//! speed and because it is already a workspace dep; the guarantee the
//! caller cares about is stability, not a specific hash family).
//! MMR tie-breaks fall back to lexicographic order.
//!
//! # Weights
//!
//! `alpha = 0.5`, `beta = 0.3`, `gamma = 0.2` (see spec §E4).
//! If `query_embed` is `None`, `beta` is redistributed to `alpha`
//! (effective `alpha = 0.8, gamma = 0.2`).

use mnem_embed_providers::Embedder;

/// A single sentence picked by [`summarize_community`], with the
/// final MMR-adjusted score at the moment it was selected.
#[derive(Debug, Clone, PartialEq)]
pub struct SummaryItem {
    /// The original sentence text.
    pub sentence: String,
    /// The score at selection time (post-MMR penalty).
    pub score: f32,
}

/// The output of [`summarize_community`]: picked sentences in MMR
/// order (first-picked = highest effective score) plus a
/// parallel-indexed score vector for callers that want the numbers
/// without the strings.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    /// Picked sentences in MMR selection order.
    pub sentences: Vec<String>,
    /// Scores aligned with [`Summary::sentences`].
    pub scores: Vec<f32>,
}

impl Summary {
    /// Convenience: zip the parallel vectors into [`SummaryItem`]s.
    #[must_use]
    pub fn items(&self) -> Vec<SummaryItem> {
        self.sentences
            .iter()
            .zip(self.scores.iter())
            .map(|(s, &score)| SummaryItem {
                sentence: s.clone(),
                score,
            })
            .collect()
    }
}

/// Summarize a community of sentences using Centroid + MMR.
///
/// # Arguments
///
/// - `sentences`: all sentences in the community. Order-insensitive.
/// - `embedder`: any [`Embedder`] (typically the MiniLM MCP default,
///   or a mock in tests). Reused from `mnem-embed-providers`.
/// - `query_embed`: optional query vector for query-focused
///   summarization. Must match `embedder.dim()` when provided.
/// - `centrality`: closure returning a non-negative centrality weight
///   for each sentence index. Today this is degree-centrality from
///   the caller; when E2 lands a PPR vector can slot in unchanged.
/// - `k`: maximum number of sentences to return. `min(k, sentences.len())`
///   are actually picked.
/// - `mmr_lambda`: diversity knob in `[0.0, 1.0]`. `0.0` = pure
///   relevance, `1.0` = pure diversity. Values outside the range
///   are clamped. Default from spec: `0.5`.
///
/// # Panics
///
/// Does not panic on empty input; returns an empty [`Summary`].
///
/// # Errors
///
/// Propagates any [`mnem_embed_providers::EmbedError`] from the
/// underlying embedder.
#[allow(clippy::too_many_arguments)]
pub fn summarize_community(
    sentences: &[String],
    embedder: &dyn Embedder,
    query_embed: Option<&[f32]>,
    centrality: &dyn Fn(usize) -> f32,
    k: usize,
    mmr_lambda: f32,
) -> Result<Summary, mnem_embed_providers::EmbedError> {
    if sentences.is_empty() || k == 0 {
        return Ok(Summary {
            sentences: Vec::new(),
            scores: Vec::new(),
        });
    }

    // ------------------------------------------------------------
    // Step 1: stable ordering via content hash.
    //
    // We build an index permutation `perm` so that
    // `sentences[perm[i]]` is the i-th sentence in canonical order.
    // The caller's `centrality` closure is still called with the
    // ORIGINAL index so that a caller-provided degree/PPR vector
    // does not have to be re-permuted.
    // ------------------------------------------------------------
    let mut perm: Vec<usize> = (0..sentences.len()).collect();
    perm.sort_by(|&a, &b| {
        let ha = blake3::hash(sentences[a].as_bytes());
        let hb = blake3::hash(sentences[b].as_bytes());
        ha.as_bytes()
            .cmp(hb.as_bytes())
            .then_with(|| sentences[a].cmp(&sentences[b]))
    });

    // ------------------------------------------------------------
    // Step 2: embed every sentence in canonical order.
    // ------------------------------------------------------------
    let texts: Vec<&str> = perm.iter().map(|&i| sentences[i].as_str()).collect();
    let embeds = embedder.embed_batch(&texts)?;

    // ------------------------------------------------------------
    // Step 3: centroid = mean of sentence embeddings.
    // ------------------------------------------------------------
    let dim = embedder.dim() as usize;
    let mut centroid = vec![0.0_f32; dim];
    for v in &embeds {
        for (c, x) in centroid.iter_mut().zip(v.iter()) {
            *c += *x;
        }
    }
    let n_f = embeds.len() as f32;
    for c in &mut centroid {
        *c /= n_f;
    }

    // Validate query_embed dimension if supplied.
    if let Some(q) = query_embed
        && q.len() != dim
    {
        return Err(mnem_embed_providers::EmbedError::DimMismatch {
            expected: embedder.dim(),
            got: u32::try_from(q.len()).unwrap_or(u32::MAX),
        });
    }

    // ------------------------------------------------------------
    // Step 4: per-sentence base score.
    //
    // Score(s_i) = alpha * cos(s_i, centroid)
    //            + beta  * cos(s_i, query)         (if query)
    //            + gamma * centrality(orig_i)/max_centrality
    //
    // If no query, redistribute beta to alpha (alpha=0.8, gamma=0.2).
    // ------------------------------------------------------------
    let (alpha, beta, gamma) = if query_embed.is_some() {
        (0.5_f32, 0.3_f32, 0.2_f32)
    } else {
        (0.8_f32, 0.0_f32, 0.2_f32)
    };

    // Materialise centralities in canonical order AND find their max.
    let mut centralities_canon: Vec<f32> = Vec::with_capacity(perm.len());
    for &orig_i in &perm {
        let c = centrality(orig_i);
        centralities_canon.push(c.max(0.0));
    }
    let max_centrality = centralities_canon
        .iter()
        .copied()
        .fold(0.0_f32, f32::max)
        .max(f32::EPSILON); // avoid /0

    let base_scores: Vec<f32> = embeds
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let s_cent = cosine(v, &centroid);
            let s_query = query_embed.map_or(0.0, |q| cosine(v, q));
            let s_centrality = centralities_canon[i] / max_centrality;
            alpha * s_cent + beta * s_query + gamma * s_centrality
        })
        .collect();

    // ------------------------------------------------------------
    // Step 5: MMR greedy selection.
    //
    // effective(i) = (1 - lambda) * base_scores[i]
    //              - lambda * max_{j in picked} cos(v_i, v_j)
    //
    // Note: the spec text in the worktree task says the penalty is
    // `lambda * max(cos(..., picked))`. The standard MMR formulation
    // balances relevance and diversity as
    //   MMR = lambda * rel - (1 - lambda) * max_sim
    // We follow the standard interpretation with `mmr_lambda` being
    // the *diversity* weight (high lambda -> strong penalty), which
    // matches the spec's "diversity tradeoff" language and the
    // lambda=0.5 default.
    // ------------------------------------------------------------
    let lambda = mmr_lambda.clamp(0.0, 1.0);
    let k_cap = k.min(embeds.len());
    let mut picked: Vec<usize> = Vec::with_capacity(k_cap);
    let mut picked_set = vec![false; embeds.len()];
    let mut out_sentences: Vec<String> = Vec::with_capacity(k_cap);
    let mut out_scores: Vec<f32> = Vec::with_capacity(k_cap);

    while picked.len() < k_cap {
        let mut best_idx: Option<usize> = None;
        let mut best_score = f32::NEG_INFINITY;

        for i in 0..embeds.len() {
            if picked_set[i] {
                continue;
            }
            // MMR penalty: max cosine similarity to any already-picked
            // sentence. Clamped to [0.0, 1.0] so that anti-correlated
            // vectors (which the MockEmbedder can produce) cannot turn
            // the penalty into a spurious BONUS. With normalized
            // MiniLM embeddings cosines are in [0,1] already, but the
            // clamp keeps the invariant "effective_score is
            // non-increasing across greedy picks" under every embedder.
            let penalty = if picked.is_empty() {
                0.0
            } else {
                picked
                    .iter()
                    .map(|&j| cosine(&embeds[i], &embeds[j]).clamp(0.0, 1.0))
                    .fold(0.0_f32, f32::max)
            };
            let eff = (1.0 - lambda) * base_scores[i] - lambda * penalty;

            // Tie-break: lexicographic on the sentence text.
            let is_better = match best_idx {
                None => true,
                Some(bi) => {
                    if eff > best_score {
                        true
                    } else if (eff - best_score).abs() < f32::EPSILON {
                        texts[i] < texts[bi]
                    } else {
                        false
                    }
                }
            };
            if is_better {
                best_idx = Some(i);
                best_score = eff;
            }
        }

        if let Some(bi) = best_idx {
            picked.push(bi);
            picked_set[bi] = true;
            out_sentences.push(texts[bi].to_owned());
            out_scores.push(best_score);
        } else {
            break;
        }
    }

    Ok(Summary {
        sentences: out_sentences,
        scores: out_scores,
    })
}

/// Cosine similarity. Returns 0.0 when either vector is zero-norm
/// (no panics, no NaN).
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "cosine: dim mismatch");
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom <= f32::EPSILON {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnem_embed_providers::MockEmbedder;

    fn make_mock() -> MockEmbedder {
        MockEmbedder::new("test:mock", 32)
    }

    #[test]
    fn empty_input_returns_empty_summary() {
        let e = make_mock();
        let s = summarize_community(&[], &e, None, &|_| 1.0, 5, 0.5).unwrap();
        assert!(s.sentences.is_empty());
        assert!(s.scores.is_empty());
    }

    #[test]
    fn k_zero_returns_empty() {
        let e = make_mock();
        let xs = vec!["a".to_string(), "b".to_string()];
        let s = summarize_community(&xs, &e, None, &|_| 1.0, 0, 0.5).unwrap();
        assert!(s.sentences.is_empty());
    }

    #[test]
    fn k_larger_than_n_is_clamped() {
        let e = make_mock();
        let xs = vec!["a".to_string(), "b".to_string()];
        let s = summarize_community(&xs, &e, None, &|_| 1.0, 99, 0.5).unwrap();
        assert_eq!(s.sentences.len(), 2);
    }
}
