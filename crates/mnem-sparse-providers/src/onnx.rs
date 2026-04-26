//! Native, in-process learned-sparse encoders.
//!
//! This module owns mnem's sparse-retrieval implementation end-to-end.
//! It is *inspired by* the algorithms published with:
//!
//! - OpenSearch `opensearch-neural-sparse-encoding-doc-v3-distill`
//!   (Apache-2.0): the asymmetric inference-free-query design, the
//!   double-log activation, the ship-an-IDF-table pattern.
//! - `fastembed-rs` (Apache-2.0): the sync `ort::Session` +
//!   `tokenizers` + `hf-hub` plumbing patterns.
//!
//! It **does not** depend on either crate. Every line below is written
//! against the raw `ort` / `tokenizers` / `hf-hub` APIs so the
//! implementation lives inside mnem rather than as a thin wrapper.
//!
//! Feature-gated (`onnx`). WASM builds never see this code because the
//! feature defaults off and the dependencies would not compile.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

use hf_hub::api::sync::{Api, ApiBuilder};
use ndarray::{Array2, ArrayViewD};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use mnem_core::sparse::{SparseEmbed, SparseEncoder, SparseError};

// ----------------------------------------------------------------------------
// Model registry
// ----------------------------------------------------------------------------

/// Which sparse-retrieval model to load.
///
/// Each variant pins (a) the HuggingFace repo id we lazy-download from,
/// (b) the tokenizer + model filenames, (c) the activation we apply to
/// the encoder's MLM logits, and (d) whether the query side runs the
/// network at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// OpenSearch `opensearch-neural-sparse-encoding-doc-v3-distill`.
    /// Apache-2.0. Asymmetric: the document side runs a DistilBERT MLM
    /// head with `log(1 + log(1 + ReLU(logits)))` activation; the
    /// query side is inference-free (tokenise + IDF-table lookup, no
    /// neural forward pass). Primary default for mnem.
    OpensearchDocV3Distill,
    /// OpenSearch `opensearch-neural-sparse-encoding-v2-distill`. Same
    /// DistilBERT backbone but symmetric: both sides run the network
    /// with single-log activation. Kept as a comparison baseline and
    /// for non-English users who want fully-neural symmetry.
    OpensearchBiV2Distill,
}

impl ModelKind {
    /// HuggingFace repo id we fetch weights + tokenizer from.
    pub fn repo_id(self) -> &'static str {
        match self {
            Self::OpensearchDocV3Distill => {
                "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill"
            }
            Self::OpensearchBiV2Distill => {
                "opensearch-project/opensearch-neural-sparse-encoding-v2-distill"
            }
        }
    }

    /// Canonical vocab id written into every `SparseEmbed` we emit.
    /// Callers key the inverted index on this; mixing vocabularies in
    /// one index produces garbage scores, so the id must be stable.
    pub fn vocab_id(self) -> &'static str {
        match self {
            Self::OpensearchDocV3Distill => "opensearch-doc-v3-distill",
            Self::OpensearchBiV2Distill => "opensearch-bi-v2-distill",
        }
    }

    /// `true` if the query side can skip the neural forward pass
    /// entirely and compute weights from tokenise + IDF lookup.
    pub fn query_is_inference_free(self) -> bool {
        matches!(self, Self::OpensearchDocV3Distill)
    }

    /// Logits activation used on the document side. v3-distill applies
    /// `log(1 + log(1 + ReLU(x)))` (double-log saturation); the
    /// symmetric models apply single-log SPLADE-style
    /// `log(1 + ReLU(x))`.
    fn activation(self) -> Activation {
        match self {
            Self::OpensearchDocV3Distill => Activation::DoubleLog,
            Self::OpensearchBiV2Distill => Activation::SingleLog,
        }
    }

    /// Hard ceiling on `max_length`: going above this triggers a
    /// positional-embedding out-of-bounds at forward time. Both
    /// OpenSearch distill variants share DistilBERT's 512-position
    /// table, so `positional_limit()` and `default_max_length()`
    /// currently coincide; kept as separate methods so future XLM-R
    /// / Longformer backbones can lift the default while keeping
    /// the ceiling honest.
    pub const fn positional_limit(self) -> usize {
        match self {
            Self::OpensearchDocV3Distill | Self::OpensearchBiV2Distill => 512,
        }
    }

    /// Default `max_length` we feed the tokenizer when the caller
    /// doesn't pick one. Kept at the positional ceiling so we
    /// preserve the full attention window by default.
    pub const fn default_max_length(self) -> usize {
        self.positional_limit()
    }
}

#[derive(Debug, Clone, Copy)]
enum Activation {
    SingleLog,
    DoubleLog,
}

impl Activation {
    fn apply(self, x: f32) -> f32 {
        // ReLU first: negatives contribute nothing.
        let relu = x.max(0.0);
        // log(1 + y) is monotonic in y; the double-log variant piles
        // a second log on top so very large logits saturate quickly.
        // `ln_1p` is numerically stable (no catastrophic cancellation
        // near y = 0).
        match self {
            Self::SingleLog => relu.ln_1p(),
            Self::DoubleLog => relu.ln_1p().ln_1p(),
        }
    }
}

// ----------------------------------------------------------------------------
// Model download + layout
// ----------------------------------------------------------------------------

/// The three artifacts we lazy-fetch for every model. Paths resolve to
/// the local HuggingFace cache directory (`$HF_HOME` or the platform
/// default).
#[derive(Debug, Clone)]
struct ModelFiles {
    model_onnx: PathBuf,
    tokenizer_json: PathBuf,
    /// Present only for asymmetric models. `None` when the query side
    /// runs the encoder (symmetric models don't need an IDF table).
    idf_json: Option<PathBuf>,
}

fn hf_api() -> Result<Api, SparseError> {
    // Sync API; respects `HF_HOME` automatically via `hf_hub`'s cache
    // resolution. No network on warm cache.
    ApiBuilder::new()
        .build()
        .map_err(|e| SparseError::Config(format!("hf-hub init: {e}")))
}

fn fetch_files(kind: ModelKind) -> Result<ModelFiles, SparseError> {
    let api = hf_api()?;
    let repo = api.model(kind.repo_id().to_string());

    let model_onnx = repo
        .get("onnx/model.onnx")
        .or_else(|_| repo.get("model.onnx"))
        .map_err(|e| {
            SparseError::Config(format!("download model.onnx from {}: {e}", kind.repo_id()))
        })?;
    let tokenizer_json = repo
        .get("tokenizer.json")
        .map_err(|e| SparseError::Config(format!("download tokenizer.json: {e}")))?;

    let idf_json = if kind.query_is_inference_free() {
        Some(
            repo.get("idf.json")
                .map_err(|e| SparseError::Config(format!("download idf.json: {e}")))?,
        )
    } else {
        None
    };

    Ok(ModelFiles {
        model_onnx,
        tokenizer_json,
        idf_json,
    })
}

// ----------------------------------------------------------------------------
// Tokenizer configuration
// ----------------------------------------------------------------------------

/// Env var that overrides the tokenizer max_length at construction
/// time. Clamped to `ModelKind::positional_limit()`. Set to a number
/// of tokens, e.g. `MNEM_ONNX_SPARSE_MAX_LEN=256` to speed up ingest
/// on short docs.
const ENV_SPARSE_MAX_LEN: &str = "MNEM_ONNX_SPARSE_MAX_LEN";

/// Resolve the effective max_length for a model given an optional
/// caller-supplied override and the env var. Clamps at the positional
/// ceiling and emits a one-liner to stderr when clamping happens, so a
/// misconfigured operator notices rather than getting silent OOB at
/// inference time.
fn resolve_max_length(kind: ModelKind, override_: Option<usize>) -> usize {
    let ceiling = kind.positional_limit();
    let requested = override_
        .or_else(|| {
            std::env::var(ENV_SPARSE_MAX_LEN)
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
        })
        .unwrap_or_else(|| kind.default_max_length());
    if requested == 0 {
        eprintln!(
            "mnem-sparse: requested max_length=0 for {}; snapping to default {}",
            kind.vocab_id(),
            kind.default_max_length()
        );
        return kind.default_max_length();
    }
    if requested > ceiling {
        eprintln!(
            "mnem-sparse: requested max_length={requested} exceeds {}'s positional limit {ceiling}; clamping",
            kind.vocab_id()
        );
        return ceiling;
    }
    requested
}

fn load_tokenizer(path: &Path, max_len: usize) -> Result<Tokenizer, SparseError> {
    let mut tok = Tokenizer::from_file(path)
        .map_err(|e| SparseError::Config(format!("tokenizer.json load: {e}")))?;
    // Truncate from the right (head-kept) to `max_len`; pad-batch to the
    // longest in the batch (not to a fixed length) so short inputs
    // don't waste compute. Same pattern fastembed uses in production.
    tok.with_truncation(Some(TruncationParams {
        max_length: max_len,
        ..Default::default()
    }))
    .map_err(|e| SparseError::Config(format!("tokenizer truncation: {e}")))?;
    tok.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        ..Default::default()
    }));
    Ok(tok)
}

// ----------------------------------------------------------------------------
// IDF table (inference-free query side)
// ----------------------------------------------------------------------------

/// Loaded IDF weights keyed by token id. Dense so the query path is a
/// branchless indexed read instead of a HashMap lookup.
#[derive(Debug, Clone)]
struct IdfTable {
    /// Indexed by token id; length == tokenizer vocab size.
    weights: Vec<f32>,
}

impl IdfTable {
    fn load(path: &Path, tokenizer: &Tokenizer) -> Result<Self, SparseError> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| SparseError::Config(format!("read idf.json: {e}")))?;
        // The OpenSearch v3 distill ships `idf.json` as {token_string -> f32}.
        // We materialise it into a dense vocab-sized vector keyed by
        // tokenizer id, zeroing anything not present (defaults to "no
        // IDF boost for this token").
        let map: HashMap<String, f32> = serde_json::from_str(&raw)
            .map_err(|e| SparseError::Config(format!("parse idf.json: {e}")))?;
        let vocab_size = tokenizer.get_vocab_size(true);
        let mut weights = vec![0.0_f32; vocab_size];
        for (tok_str, idf) in map {
            if let Some(id) = tokenizer.token_to_id(&tok_str) {
                let idx = id as usize;
                if idx < weights.len() {
                    weights[idx] = idf;
                }
            }
        }
        Ok(Self { weights })
    }

    /// Query-side encode: tokenise, accumulate the per-token IDF
    /// weight, de-duplicate by token id taking the max. No neural
    /// compute; pure lookup.
    fn encode_query(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        special_ids: &[u32],
    ) -> Result<Vec<(u32, f32)>, SparseError> {
        let encoded = tokenizer
            .encode(text, true)
            .map_err(|e| SparseError::Inference(format!("tokenize query: {e}")))?;
        let ids = encoded.get_ids();

        let mut by_id: HashMap<u32, f32> = HashMap::with_capacity(ids.len());
        for &id in ids {
            if special_ids.contains(&id) {
                continue;
            }
            let idx = id as usize;
            if idx >= self.weights.len() {
                continue;
            }
            let w = self.weights[idx];
            if w > 0.0 {
                // Same token appearing twice in the query collapses via
                // max (consistent with the doc-side per-token reduction).
                let slot = by_id.entry(id).or_insert(0.0);
                if w > *slot {
                    *slot = w;
                }
            }
        }

        let mut out: Vec<(u32, f32)> = by_id.into_iter().collect();
        out.sort_by_key(|&(id, _)| id);
        Ok(out)
    }
}

// ----------------------------------------------------------------------------
// Special-token ids (masked out of the sparse output)
// ----------------------------------------------------------------------------

fn collect_special_ids(tokenizer: &Tokenizer) -> Vec<u32> {
    // DistilBERT ships CLS=101, SEP=102, PAD=0, UNK=100, MASK=103 in
    // the bert-base-uncased vocab. Looking them up by surface form
    // stays correct if the tokenizer rev ever changes IDs.
    let surfaces = ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"];
    surfaces
        .iter()
        .filter_map(|s| tokenizer.token_to_id(s))
        .collect()
}

// ----------------------------------------------------------------------------
// ONNX session + forward pass
// ----------------------------------------------------------------------------

struct OnnxSession {
    session: Session,
    /// Some encoders expect a `token_type_ids` input tensor; others
    /// (BERT-variants trained single-segment) don't. Probed at build
    /// time so we only wire the input when needed.
    needs_token_type_ids: bool,
}

impl OnnxSession {
    fn open(model_path: &Path) -> Result<Self, SparseError> {
        // Default to SINGLE intra-op thread. Multi-thread parallel
        // reductions in ORT are not bit-stable across core counts;
        // running the default on two machines with different thread
        // pools produces slightly different sparse-embed values,
        // which would break the "byte-identical CIDs across machines"
        // property since Node.sparse_embed participates in the CID.
        // Callers who want throughput over reproducibility can bump
        // via the env override once a config knob ships.
        let threads: usize = std::env::var("MNEM_ORT_INTRA_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let session = Session::builder()
            .map_err(|e| SparseError::Config(format!("ort session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| SparseError::Config(format!("ort opt level: {e}")))?
            .with_intra_threads(threads)
            .map_err(|e| SparseError::Config(format!("ort intra threads: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| {
                SparseError::Config(format!("ort commit {}: {e}", model_path.display()))
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

    /// Single-text forward pass. Returns `(logits, seq_len, vocab_size)`
    /// as a flat f32 row-major buffer plus the dimensions the caller
    /// needs to fold over.
    fn forward_single(
        &mut self,
        encoded: &tokenizers::Encoding,
    ) -> Result<(Vec<f32>, usize, usize), SparseError> {
        let seq_len = encoded.get_ids().len();
        let ids: Vec<i64> = encoded.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoded
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();

        let ids_arr: Array2<i64> = Array2::from_shape_vec((1, seq_len), ids)
            .map_err(|e| SparseError::Inference(format!("ids reshape: {e}")))?;
        let mask_arr: Array2<i64> = Array2::from_shape_vec((1, seq_len), mask)
            .map_err(|e| SparseError::Inference(format!("mask reshape: {e}")))?;

        // Build the input map. `Value::from_array` takes an owned
        // array in ort 2.0.0-rc.12 (the `OwnedTensorArrayData` trait
        // bound), so we hand it the reshaped Array2 directly.
        let mut inputs: Vec<(&'static str, Value)> = Vec::with_capacity(3);
        inputs.push((
            "input_ids",
            Value::from_array(ids_arr)
                .map_err(|e| SparseError::Inference(format!("ids tensor: {e}")))?
                .into_dyn(),
        ));
        inputs.push((
            "attention_mask",
            Value::from_array(mask_arr)
                .map_err(|e| SparseError::Inference(format!("mask tensor: {e}")))?
                .into_dyn(),
        ));
        if self.needs_token_type_ids {
            let type_ids_arr: Array2<i64> = Array2::zeros((1, seq_len));
            inputs.push((
                "token_type_ids",
                Value::from_array(type_ids_arr)
                    .map_err(|e| SparseError::Inference(format!("type_ids tensor: {e}")))?
                    .into_dyn(),
            ));
        }

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| SparseError::Inference(format!("ort run: {e}")))?;

        // DistilBertForMaskedLM emits a single `(batch, seq_len, vocab)`
        // logits tensor. Some ONNX exports name it `logits`; others
        // emit it as the only output. Take by name first, fall back
        // to position 0.
        let value = outputs
            .iter()
            .find(|(name, _)| *name == "logits")
            .map(|(_, v)| v)
            .or_else(|| outputs.iter().next().map(|(_, v)| v))
            .ok_or_else(|| SparseError::Inference("no logits output".into()))?;
        let view: ArrayViewD<'_, f32> = value
            .try_extract_array::<f32>()
            .map_err(|e| SparseError::Inference(format!("extract logits: {e}")))?;
        let shape = view.shape().to_vec();
        let buffer: Vec<f32> = view.iter().copied().collect();

        if shape.len() != 3 {
            return Err(SparseError::Inference(format!(
                "expected rank-3 logits, got shape {:?}",
                shape
            )));
        }
        let seq = shape[1];
        let vocab = shape[2];
        Ok((buffer, seq, vocab))
    }
}

// ----------------------------------------------------------------------------
// Doc-side encoder
// ----------------------------------------------------------------------------

fn reduce_doc_logits(
    logits: &[f32],
    seq_len: usize,
    vocab_size: usize,
    attention_mask: &[u32],
    activation: Activation,
    special_ids: &[u32],
) -> Vec<(u32, f32)> {
    // scores[v] = max over s in [0, seq_len) of activated(logits[s, v]) * mask[s]
    let mut scores = vec![0.0_f32; vocab_size];
    for s in 0..seq_len {
        let m = attention_mask.get(s).copied().unwrap_or(0);
        if m == 0 {
            continue;
        }
        let row_start = s * vocab_size;
        let row = &logits[row_start..row_start + vocab_size];
        // Walk once and relax the running max.
        for v in 0..vocab_size {
            let a = activation.apply(row[v]);
            if a > scores[v] {
                scores[v] = a;
            }
        }
    }
    // Mask out special tokens so [CLS]/[SEP]/[PAD]/[UNK]/[MASK] never
    // contribute to scoring.
    for &id in special_ids {
        let idx = id as usize;
        if idx < scores.len() {
            scores[idx] = 0.0;
        }
    }
    // Emit only the non-zero positions; sort by token id for index
    // determinism.
    let mut out: Vec<(u32, f32)> = scores
        .into_iter()
        .enumerate()
        .filter_map(|(i, w)| if w > 0.0 { Some((i as u32, w)) } else { None })
        .collect();
    out.sort_by_key(|&(id, _)| id);
    out
}

// ----------------------------------------------------------------------------
// Public encoder
// ----------------------------------------------------------------------------

/// In-process sparse encoder backed by a local ONNX model.
///
/// One instance can encode either a query OR a document; the public
/// `encode` method picks the right path based on [`ModelKind`]:
///
/// - asymmetric models: documents run the neural network, queries use
///   the IDF table only (zero neural compute per query).
/// - symmetric models: both sides run the network with SPLADE-style
///   activation.
///
/// The `SparseEncoder` trait sees a single `encode(text) ->
/// SparseEmbed` call. If you're embedding a document, call
/// [`Self::encode_document`] directly; if you're building a query
/// vector for retrieval, call [`Self::encode_query`]. The trait-level
/// `encode` defaults to the document path (most ingests go through
/// there first).
pub struct OnnxSparseEncoder {
    kind: ModelKind,
    tokenizer: Tokenizer,
    session: std::sync::Mutex<OnnxSession>,
    idf: Option<Arc<IdfTable>>,
    special_ids: Vec<u32>,
    model_fq: String,
    /// Effective tokenizer max_length (post-clamp). Stored so the
    /// encode path can detect tail-truncation without re-parsing
    /// the tokenizer config.
    max_len: usize,
}

/// Process-wide latch keyed on `(provider, model_fq)`: the truncation
/// warning prints at most once per tuple per process lifetime, even
/// across multiple `OnnxSparseEncoder` instances or threads. Avoids
/// hundreds of duplicate stderr lines on bench runs (LongMemEval-500
/// ingests hundreds of long sessions).
static TOKENIZER_TRUNCATE_WARNED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

fn warn_truncation_once(provider: &str, model: &str, max_len: usize, positional_limit: usize) {
    let key = format!("{provider}:{model}");
    let set = TOKENIZER_TRUNCATE_WARNED.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = match set.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if guard.insert(key) {
        eprintln!(
            "{provider}: input filled max_length={max_len} on {model}; tail truncated. \
             Raise via MNEM_ONNX_SPARSE_MAX_LEN (<= {positional_limit}) or chunk upstream."
        );
    }
}

impl std::fmt::Debug for OnnxSparseEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxSparseEncoder")
            .field("kind", &self.kind)
            .field("model_fq", &self.model_fq)
            .field("has_idf", &self.idf.is_some())
            .field("max_len", &self.max_len)
            .finish()
    }
}

static ORT_INIT: OnceLock<()> = OnceLock::new();

fn ensure_ort_init() {
    ORT_INIT.get_or_init(|| {
        // ort 2.x no longer requires an explicit environment init for
        // basic inference. Placeholder for future global setup (e.g.
        // CUDA provider init, log routing).
    });
}

impl OnnxSparseEncoder {
    /// Construct an encoder, lazy-downloading the model + tokenizer +
    /// (for asymmetric models) IDF table into the HuggingFace cache
    /// on first call. Uses the model's default max_length (the
    /// positional-embedding ceiling) unless `MNEM_ONNX_SPARSE_MAX_LEN`
    /// overrides it.
    pub fn new(kind: ModelKind) -> Result<Self, SparseError> {
        Self::with_max_length(kind, None)
    }

    /// Construct an encoder with an explicit tokenizer `max_length`.
    /// Pass `None` to defer to the env var / model default. Values
    /// above the model's `positional_limit()` are clamped with a
    /// stderr warning.
    pub fn with_max_length(
        kind: ModelKind,
        max_length: Option<usize>,
    ) -> Result<Self, SparseError> {
        ensure_ort_init();
        let max_len = resolve_max_length(kind, max_length);
        let files = fetch_files(kind)?;
        let tokenizer = load_tokenizer(&files.tokenizer_json, max_len)?;
        let special_ids = collect_special_ids(&tokenizer);
        let idf = match &files.idf_json {
            Some(p) => Some(Arc::new(IdfTable::load(p, &tokenizer)?)),
            None => None,
        };
        let session = OnnxSession::open(&files.model_onnx)?;
        let model_fq = format!("onnx:{}", kind.vocab_id());
        Ok(Self {
            kind,
            tokenizer,
            session: std::sync::Mutex::new(session),
            idf,
            special_ids,
            model_fq,
            max_len,
        })
    }

    /// Effective tokenizer max_length (post-clamp). Exposed for
    /// telemetry / debug inspection.
    pub fn max_length(&self) -> usize {
        self.max_len
    }

    /// Encode a document: always runs the neural network.
    pub fn encode_document(&self, text: &str) -> Result<SparseEmbed, SparseError> {
        let encoded = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| SparseError::Inference(format!("tokenize doc: {e}")))?;
        // Warn-once when the encoded sequence fills the window.
        // `encode` returns only the truncated head; the dropped tail
        // is silent by default. We don't know the exact pre-truncation
        // length without a second tokenise pass, so the heuristic is
        // "filled the window" -> likely truncated. Skips false
        // positives for empty/short corpora and caps stderr noise via
        // a per-process latch.
        if encoded.get_ids().len() >= self.max_len {
            warn_truncation_once(
                "mnem-sparse",
                self.kind.vocab_id(),
                self.max_len,
                self.kind.positional_limit(),
            );
        }
        let mask = encoded.get_attention_mask().to_vec();
        let mut session = self
            .session
            .lock()
            .map_err(|_| SparseError::Inference("session mutex poisoned".into()))?;
        let (logits, seq_len, vocab_size) = session.forward_single(&encoded)?;
        drop(session);
        let pairs = reduce_doc_logits(
            &logits,
            seq_len,
            vocab_size,
            &mask,
            self.kind.activation(),
            &self.special_ids,
        );
        pairs_to_sparse(pairs, self.kind.vocab_id())
    }

    /// Encode a query. Asymmetric models (opensearch-doc-v3-distill)
    /// use the IDF table; symmetric models fall back to the neural
    /// forward pass.
    pub fn encode_query(&self, text: &str) -> Result<SparseEmbed, SparseError> {
        if let Some(idf) = &self.idf {
            let pairs = idf.encode_query(&self.tokenizer, text, &self.special_ids)?;
            return pairs_to_sparse(pairs, self.kind.vocab_id());
        }
        self.encode_document(text)
    }
}

fn pairs_to_sparse(mut pairs: Vec<(u32, f32)>, vocab_id: &str) -> Result<SparseEmbed, SparseError> {
    // Both the doc-side reduction and the query-side IDF path emit
    // ascending-by-id already, but defensive sort keeps the
    // `SparseEmbed::new` strict-ascending invariant safe against a
    // future refactor.
    pairs.sort_by_key(|&(id, _)| id);
    let mut indices: Vec<u32> = Vec::with_capacity(pairs.len());
    let mut values: Vec<f32> = Vec::with_capacity(pairs.len());
    for (id, w) in pairs {
        indices.push(id);
        values.push(w);
    }
    SparseEmbed::new(indices, values, vocab_id.to_string())
}

impl SparseEncoder for OnnxSparseEncoder {
    fn model(&self) -> &str {
        &self.model_fq
    }

    fn vocab_id(&self) -> &str {
        self.kind.vocab_id()
    }

    fn encode(&self, text: &str) -> Result<SparseEmbed, SparseError> {
        // Document path: always runs the neural network.
        self.encode_document(text)
    }

    fn encode_query(&self, text: &str) -> Result<SparseEmbed, SparseError> {
        // Query path: asymmetric models (v3-distill) skip the neural
        // forward pass entirely and use the shipped IDF table. Call
        // the inherent `encode_query` method (which handles the split
        // internally) instead of the trait default.
        Self::encode_query(self, text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_max_length_uses_default_when_none() {
        let n = resolve_max_length(ModelKind::OpensearchDocV3Distill, None);
        assert_eq!(n, 512);
    }

    #[test]
    fn resolve_max_length_passes_through_in_range() {
        let n = resolve_max_length(ModelKind::OpensearchDocV3Distill, Some(256));
        assert_eq!(n, 256);
    }

    #[test]
    fn resolve_max_length_clamps_to_positional_limit() {
        // 8192 > DistilBERT's 512 ceiling -> clamped.
        let n = resolve_max_length(ModelKind::OpensearchDocV3Distill, Some(8192));
        assert_eq!(n, 512);
    }

    #[test]
    fn resolve_max_length_zero_snaps_to_default() {
        // max_length=0 is illegal for tokenizers; we must not pass it
        // through. Snap to default and warn.
        let n = resolve_max_length(ModelKind::OpensearchBiV2Distill, Some(0));
        assert_eq!(n, 512);
    }

    #[test]
    fn positional_limit_and_default_coincide_for_distilbert() {
        assert_eq!(
            ModelKind::OpensearchDocV3Distill.positional_limit(),
            ModelKind::OpensearchDocV3Distill.default_max_length()
        );
        assert_eq!(
            ModelKind::OpensearchBiV2Distill.positional_limit(),
            ModelKind::OpensearchBiV2Distill.default_max_length()
        );
    }
}
