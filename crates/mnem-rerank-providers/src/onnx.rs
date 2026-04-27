//! In-process cross-encoder reranker via ONNX Runtime.
//!
//! Feature-gated (`onnx`). Gives mnem a fully self-hosted reranker
//! option so the hybrid retrieval stack runs without any SaaS key:
//!
//!     sparse (onnx) + dense (ollama) + graph-expand + rerank (onnx)
//!
//! Models (Apache-2.0, all BERT-family sequence-classification heads):
//!
//! - `cross-encoder/ms-marco-MiniLM-L-6-v2` (via Xenova's ONNX export).
//!   22M params, 384 hidden dim, MS-MARCO fine-tuned. The default
//!   because it is the smallest cross-encoder that still materially
//!   beats bi-encoder RRF on BEIR and runs on laptop CPU at ~5ms/pair.
//! - `BAAI/bge-reranker-v2-m3`. 568M params, XLM-R base, multilingual.
//!   Strongest public BEIR numbers in the Apache-2.0 lineup; use when
//!   quality matters more than footprint.
//! - `BAAI/bge-reranker-base`. 278M params, English-only, faster than
//!   v2-m3 with most of its English quality.
//!
//! Inspired by (but not importing) `fastembed-rs` and the sentence-
//! transformers cross-encoder inference pattern.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use hf_hub::api::sync::{Api, ApiBuilder};
use ndarray::{Array2, ArrayViewD};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use mnem_core::rerank::{RerankError, Reranker};

// ----------------------------------------------------------------------------
// Model registry
// ----------------------------------------------------------------------------

/// Which ONNX cross-encoder to load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RerankerModel {
    /// `cross-encoder/ms-marco-MiniLM-L-6-v2` via `Xenova/ms-marco-MiniLM-L-6-v2`
    /// ONNX export. 22M params, 384 hidden, MS-MARCO fine-tuned.
    /// Apache-2.0.
    MsMarcoMiniLmL6V2,
    /// `BAAI/bge-reranker-v2-m3`. 568M params, XLM-R base,
    /// multilingual. Apache-2.0.
    BgeRerankerV2M3,
    /// `BAAI/bge-reranker-base`. 278M params, English-only. Apache-2.0.
    BgeRerankerBase,
}

impl RerankerModel {
    fn repo_id(self) -> &'static str {
        match self {
            Self::MsMarcoMiniLmL6V2 => "Xenova/ms-marco-MiniLM-L-6-v2",
            Self::BgeRerankerV2M3 => "BAAI/bge-reranker-v2-m3",
            Self::BgeRerankerBase => "BAAI/bge-reranker-base",
        }
    }

    fn onnx_path(self) -> &'static str {
        match self {
            // Xenova puts the ONNX file under `onnx/model.onnx`.
            Self::MsMarcoMiniLmL6V2 => "onnx/model.onnx",
            Self::BgeRerankerV2M3 => "onnx/model.onnx",
            Self::BgeRerankerBase => "onnx/model.onnx",
        }
    }

    /// Canonical wire-id stamped on the `Reranker::model()` string so
    /// logs + `_meta` telemetry can distinguish identical architectures
    /// with different fine-tunes.
    fn wire_id(self) -> &'static str {
        match self {
            Self::MsMarcoMiniLmL6V2 => "onnx:ms-marco-MiniLM-L-6-v2",
            Self::BgeRerankerV2M3 => "onnx:bge-reranker-v2-m3",
            Self::BgeRerankerBase => "onnx:bge-reranker-base",
        }
    }

    /// Default sequence length we feed the tokenizer. Cross-encoder
    /// input is `[CLS] query [SEP] candidate [SEP]`, so the cap
    /// applies to the concatenated pair. We ship a conservative 512
    /// default across the board for predictable compute; callers who
    /// need more can raise it via `with_max_length` / the env var
    /// up to `positional_limit()`.
    pub const fn default_max_length(self) -> usize {
        512
    }

    /// Hard ceiling imposed by the model's positional-embedding table.
    /// Going above this triggers runtime OOB. Callers overriding
    /// `default_max_length` are clamped at this value.
    pub const fn positional_limit(self) -> usize {
        match self {
            // MiniLM-L-6-v2 inherits BERT-base's 512-position table.
            Self::MsMarcoMiniLmL6V2 => 512,
            // bge-reranker-v2-m3 is trained on XLM-R with extended
            // positions up to 8192.
            Self::BgeRerankerV2M3 => 8192,
            // bge-reranker-base uses XLM-RoBERTa-base, 512-capped.
            Self::BgeRerankerBase => 512,
        }
    }
}

// ----------------------------------------------------------------------------
// Model download + tokenizer
// ----------------------------------------------------------------------------

struct ModelFiles {
    model_onnx: PathBuf,
    tokenizer_json: PathBuf,
}

fn hf_api() -> Result<Api, RerankError> {
    ApiBuilder::new()
        .build()
        .map_err(|e| RerankError::Config(format!("hf-hub init: {e}")))
}

fn fetch_files(kind: RerankerModel) -> Result<ModelFiles, RerankError> {
    let api = hf_api()?;
    let repo = api.model(kind.repo_id().to_string());

    let model_onnx = repo
        .get(kind.onnx_path())
        .or_else(|_| repo.get("model.onnx"))
        .map_err(|e| RerankError::Config(format!("download {} onnx: {e}", kind.repo_id())))?;
    let tokenizer_json = repo
        .get("tokenizer.json")
        .map_err(|e| RerankError::Config(format!("download tokenizer.json: {e}")))?;

    Ok(ModelFiles {
        model_onnx,
        tokenizer_json,
    })
}

fn load_tokenizer(path: &Path, max_len: usize) -> Result<Tokenizer, RerankError> {
    let mut tok = Tokenizer::from_file(path)
        .map_err(|e| RerankError::Config(format!("tokenizer.json load: {e}")))?;
    tok.with_truncation(Some(TruncationParams {
        max_length: max_len,
        ..Default::default()
    }))
    .map_err(|e| RerankError::Config(format!("tokenizer truncation: {e}")))?;
    tok.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        ..Default::default()
    }));
    Ok(tok)
}

/// Env var that overrides the tokenizer max_length at construction
/// time. Clamped to `RerankerModel::positional_limit()`. Cross-encoder
/// inputs are the concatenated `[CLS] query [SEP] candidate [SEP]`
/// pair, so raising this lets long candidates survive without
/// upstream chunking.
const ENV_RERANK_MAX_LEN: &str = "MNEM_ONNX_RERANK_MAX_LEN";

fn resolve_max_length(kind: RerankerModel, override_: Option<usize>) -> usize {
    let ceiling = kind.positional_limit();
    let requested = override_
        .or_else(|| {
            std::env::var(ENV_RERANK_MAX_LEN)
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
        })
        .unwrap_or_else(|| kind.default_max_length());
    if requested == 0 {
        eprintln!(
            "mnem-rerank: requested max_length=0 for {}; snapping to default {}",
            kind.wire_id(),
            kind.default_max_length()
        );
        return kind.default_max_length();
    }
    if requested > ceiling {
        eprintln!(
            "mnem-rerank: requested max_length={requested} exceeds {}'s positional limit {ceiling}; clamping",
            kind.wire_id()
        );
        return ceiling;
    }
    requested
}

// ----------------------------------------------------------------------------
// Session setup
// ----------------------------------------------------------------------------

struct OnnxSession {
    session: Session,
    needs_token_type_ids: bool,
}

impl OnnxSession {
    fn open(model_path: &Path) -> Result<Self, RerankError> {
        // Default to SINGLE intra-op thread. Rerank scores OVERWRITE
        // the fused composite score in Retriever::execute, so
        // multi-thread non-determinism would ripple out to the
        // caller-visible top-K ordering. Callers who want the
        // throughput can bump via `MNEM_ORT_INTRA_THREADS`.
        let threads: usize = std::env::var("MNEM_ORT_INTRA_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let session = Session::builder()
            .map_err(|e| RerankError::Config(format!("ort session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| RerankError::Config(format!("ort opt level: {e}")))?
            .with_intra_threads(threads)
            .map_err(|e| RerankError::Config(format!("ort intra threads: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| {
                RerankError::Config(format!("ort commit {}: {e}", model_path.display()))
            })?;
        let needs_token_type_ids = session
            .inputs()
            .iter()
            .any(|i| i.name() == "token_type_ids");
        Ok(Self {
            session,
            needs_token_type_ids,
        })
    }
}

// ----------------------------------------------------------------------------
// Public reranker
// ----------------------------------------------------------------------------

/// In-process ONNX cross-encoder reranker. Implements [`Reranker`].
///
/// Lazy-downloads model + tokenizer from the HuggingFace Hub on first
/// construction; subsequent runs read from the local cache. The
/// session is reused across `rerank()` calls and batches all
/// candidates in a single forward pass per call, padded to the batch
/// longest so short inputs don't burn compute.
pub struct OnnxReranker {
    kind: RerankerModel,
    tokenizer: Tokenizer,
    session: Mutex<OnnxSession>,
    model_fq: String,
    /// Effective tokenizer max_length (post-clamp), covering the
    /// concatenated `[CLS] query [SEP] candidate [SEP]` pair. Stored
    /// so the rerank path can detect tail-truncation.
    max_len: usize,
    /// One-shot latch: first time we see a pair filling the attention
    /// window, emit a single stderr warning per process.
    warned_truncation: AtomicBool,
}

impl std::fmt::Debug for OnnxReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxReranker")
            .field("kind", &self.kind)
            .field("model_fq", &self.model_fq)
            .field("max_len", &self.max_len)
            .finish()
    }
}

static ORT_INIT: OnceLock<()> = OnceLock::new();

fn ensure_ort_init() {
    ORT_INIT.get_or_init(|| {
        // Placeholder for future global ort setup (e.g. custom log
        // routing, CUDA provider init).
    });
}

impl OnnxReranker {
    /// Construct a reranker, lazy-downloading the model + tokenizer
    /// into the HuggingFace cache on first call. Uses
    /// `RerankerModel::default_max_length()` (512) unless
    /// `MNEM_ONNX_RERANK_MAX_LEN` overrides it.
    pub fn new(kind: RerankerModel) -> Result<Self, RerankError> {
        Self::with_max_length(kind, None)
    }

    /// Construct a reranker with an explicit tokenizer `max_length`
    /// applied to the concatenated cross-encoder pair. `None` defers
    /// to the env var / model default. Values above
    /// `RerankerModel::positional_limit()` are clamped with a stderr
    /// warning - bge-reranker-v2-m3 callers who want the full 8192
    /// window can pass `Some(8192)` here.
    pub fn with_max_length(
        kind: RerankerModel,
        max_length: Option<usize>,
    ) -> Result<Self, RerankError> {
        ensure_ort_init();
        let max_len = resolve_max_length(kind, max_length);
        let files = fetch_files(kind)?;
        let tokenizer = load_tokenizer(&files.tokenizer_json, max_len)?;
        let session = OnnxSession::open(&files.model_onnx)?;
        let model_fq = kind.wire_id().to_string();
        Ok(Self {
            kind,
            tokenizer,
            session: Mutex::new(session),
            model_fq,
            max_len,
            warned_truncation: AtomicBool::new(false),
        })
    }

    /// Effective tokenizer max_length (post-clamp).
    pub fn max_length(&self) -> usize {
        self.max_len
    }
}

impl Reranker for OnnxReranker {
    fn model(&self) -> &str {
        &self.model_fq
    }

    fn rerank(&self, query: &str, candidates: &[&str]) -> Result<Vec<f32>, RerankError> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Pair-encode every (query, candidate). `BatchLongest`
        // padding makes the batch rectangular without wasting compute
        // on short inputs.
        let pairs: Vec<(&str, &str)> = candidates.iter().map(|c| (query, *c)).collect();
        let encodings = self
            .tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| RerankError::Inference(format!("tokenize batch: {e}")))?;

        let batch = encodings.len();
        let seq_len = encodings
            .first()
            .map(|e| e.get_ids().len())
            .ok_or_else(|| RerankError::Inference("empty encoding batch".into()))?;

        // Warn-once when any pair in the batch filled the attention
        // window. The default right-truncation silently drops the
        // candidate tail, which is where the relevant text typically
        // lives for long-doc reranking. `BatchLongest` padding means
        // seq_len == max_len iff at least one pair was truncated.
        if seq_len >= self.max_len && !self.warned_truncation.swap(true, Ordering::Relaxed) {
            eprintln!(
                "mnem-rerank: batch filled max_length={} on {}; pair tail truncated. \
                 Raise via MNEM_ONNX_RERANK_MAX_LEN (<= {}) or chunk upstream.",
                self.max_len,
                self.kind.wire_id(),
                self.kind.positional_limit()
            );
        }

        // Flatten ids + mask (+ type_ids where the model asks for it)
        // into row-major (batch, seq_len) i64 tensors.
        let mut ids_flat: Vec<i64> = Vec::with_capacity(batch * seq_len);
        let mut mask_flat: Vec<i64> = Vec::with_capacity(batch * seq_len);
        let mut type_flat: Vec<i64> = Vec::with_capacity(batch * seq_len);
        for enc in &encodings {
            ids_flat.extend(enc.get_ids().iter().map(|&x| x as i64));
            mask_flat.extend(enc.get_attention_mask().iter().map(|&x| x as i64));
            type_flat.extend(enc.get_type_ids().iter().map(|&x| x as i64));
        }
        let ids_arr = Array2::from_shape_vec((batch, seq_len), ids_flat)
            .map_err(|e| RerankError::Inference(format!("ids reshape: {e}")))?;
        let mask_arr = Array2::from_shape_vec((batch, seq_len), mask_flat)
            .map_err(|e| RerankError::Inference(format!("mask reshape: {e}")))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| RerankError::Inference("session mutex poisoned".into()))?;

        let mut inputs: Vec<(&'static str, Value)> = Vec::with_capacity(3);
        inputs.push((
            "input_ids",
            Value::from_array(ids_arr)
                .map_err(|e| RerankError::Inference(format!("ids tensor: {e}")))?
                .into_dyn(),
        ));
        inputs.push((
            "attention_mask",
            Value::from_array(mask_arr)
                .map_err(|e| RerankError::Inference(format!("mask tensor: {e}")))?
                .into_dyn(),
        ));
        if session.needs_token_type_ids {
            let type_arr = Array2::from_shape_vec((batch, seq_len), type_flat)
                .map_err(|e| RerankError::Inference(format!("type_ids reshape: {e}")))?;
            inputs.push((
                "token_type_ids",
                Value::from_array(type_arr)
                    .map_err(|e| RerankError::Inference(format!("type_ids tensor: {e}")))?
                    .into_dyn(),
            ));
        }

        let outputs = session
            .session
            .run(inputs)
            .map_err(|e| RerankError::Inference(format!("ort run: {e}")))?;

        // Cross-encoder classification heads expose their score as
        // `logits` (shape `[batch, 1]` for regression / single-label
        // binary, or `[batch, 2]` for two-class where we take the
        // positive-class logit). Some exports emit shape `[batch]`.
        let value = outputs
            .iter()
            .find(|(name, _)| *name == "logits")
            .map(|(_, v)| v)
            .or_else(|| outputs.iter().next().map(|(_, v)| v))
            .ok_or_else(|| RerankError::Decode("no logits output".into()))?;
        let view: ArrayViewD<'_, f32> = value
            .try_extract_array::<f32>()
            .map_err(|e| RerankError::Decode(format!("extract logits: {e}")))?;
        let shape = view.shape().to_vec();
        let buffer: Vec<f32> = view.iter().copied().collect();

        let scores = extract_pair_scores(&buffer, &shape, batch)?;
        // `session` lock is released naturally at scope end; `outputs`
        // borrows from it, so no explicit `drop(session)` (that would
        // move-out while outputs is still live).
        Ok(scores)
    }
}

/// Convert a raw logits buffer into `Vec<f32>` of length `batch`.
///
/// Shapes handled:
/// - `[batch]` (rank 1) - scalar relevance per pair.
/// - `[batch, 1]` (rank 2, 1 label) - regression head, take the only
///   logit.
/// - `[batch, 2]` (rank 2, 2 labels) - two-class classification
///   (MS-MARCO fine-tunes commonly do this); take the second class's
///   logit, which is the "relevant" class.
///
/// Any other shape is a decode error.
fn extract_pair_scores(
    buffer: &[f32],
    shape: &[usize],
    batch: usize,
) -> Result<Vec<f32>, RerankError> {
    match shape {
        [n] if *n == batch => Ok(buffer.to_vec()),
        [n, 1] if *n == batch => Ok(buffer.to_vec()),
        [n, labels] if *n == batch && *labels >= 2 => {
            // Take the positive-class logit at index `labels - 1`.
            // MS-MARCO two-class heads map 0 -> irrelevant,
            // 1 -> relevant. Generalizes for three-way heads too.
            let mut out = Vec::with_capacity(batch);
            let stride = *labels;
            for n in 0..batch {
                out.push(buffer[n * stride + (stride - 1)]);
            }
            Ok(out)
        }
        _ => Err(RerankError::Decode(format!(
            "unexpected logits shape {shape:?} for batch={batch}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_rank_1_batch_scores() {
        let buf = vec![0.1, 0.2, 0.3];
        let out = extract_pair_scores(&buf, &[3], 3).unwrap();
        assert_eq!(out, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn extract_rank_2_single_label() {
        let buf = vec![0.1, 0.2, 0.3];
        let out = extract_pair_scores(&buf, &[3, 1], 3).unwrap();
        assert_eq!(out, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn extract_rank_2_two_class_takes_positive() {
        // Two classes: [irrelevant, relevant] per row.
        // Row 0: [0.9, 0.1] -> take 0.1
        // Row 1: [0.2, 0.8] -> take 0.8
        let buf = vec![0.9, 0.1, 0.2, 0.8];
        let out = extract_pair_scores(&buf, &[2, 2], 2).unwrap();
        assert_eq!(out, vec![0.1, 0.8]);
    }

    #[test]
    fn extract_rejects_mismatched_batch() {
        let buf = vec![0.1, 0.2];
        let err = extract_pair_scores(&buf, &[3], 2).unwrap_err();
        assert!(matches!(err, RerankError::Decode(_)));
    }

    #[test]
    fn extract_rejects_unknown_shape() {
        let buf = vec![0.1, 0.2, 0.3, 0.4];
        // Rank-3 tensor is not supported by cross-encoders.
        let err = extract_pair_scores(&buf, &[1, 2, 2], 1).unwrap_err();
        assert!(matches!(err, RerankError::Decode(_)));
    }

    #[test]
    fn resolve_max_length_uses_default_when_none() {
        let n = resolve_max_length(RerankerModel::MsMarcoMiniLmL6V2, None);
        assert_eq!(n, 512);
    }

    #[test]
    fn resolve_max_length_passes_through_in_range() {
        // 2048 is above MiniLM's 512 ceiling but within v2-m3's 8192,
        // so requesting it explicitly should stay as 2048 for v2-m3.
        let n = resolve_max_length(RerankerModel::BgeRerankerV2M3, Some(2048));
        assert_eq!(n, 2048);
    }

    #[test]
    fn resolve_max_length_clamps_above_positional_limit() {
        // MiniLM caps at 512; requesting 8192 clamps back down.
        let n = resolve_max_length(RerankerModel::MsMarcoMiniLmL6V2, Some(8192));
        assert_eq!(n, 512);
    }

    #[test]
    fn resolve_max_length_v2_m3_can_unlock_full_window() {
        // bge-reranker-v2-m3 natively supports 8192. The default is
        // 512 for predictable compute, but callers opting in get the
        // full window without clamping.
        let n = resolve_max_length(RerankerModel::BgeRerankerV2M3, Some(8192));
        assert_eq!(n, 8192);
    }

    #[test]
    fn resolve_max_length_zero_snaps_to_default() {
        let n = resolve_max_length(RerankerModel::BgeRerankerBase, Some(0));
        assert_eq!(n, 512);
    }

    #[test]
    fn positional_limits_match_published_windows() {
        assert_eq!(RerankerModel::MsMarcoMiniLmL6V2.positional_limit(), 512);
        assert_eq!(RerankerModel::BgeRerankerV2M3.positional_limit(), 8192);
        assert_eq!(RerankerModel::BgeRerankerBase.positional_limit(), 512);
    }
}
