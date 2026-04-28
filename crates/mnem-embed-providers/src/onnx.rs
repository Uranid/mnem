//! Native, in-process dense embedder.
//!
//! Feature-gated. Two flavours, mutually exclusive:
//! - `onnx` - ort/load-dynamic. User installs onnxruntime
//! separately at the OS level.
//! - `onnx-bundled` - ort/download-binaries. Onnxruntime fetched at
//! cargo-build time and linked in. Path A
//! () - what
//! `cargo install mnem-cli --features
//! bundled-embedder` activates.
//!
//! Same plumbing pattern as `mnem-sparse-providers::onnx` and
//! `mnem-rerank-providers::onnx`: `ort` session cached behind a
//! `Mutex`, `tokenizers` for WordPiece, `hf-hub` for lazy HF-cache-
//! aware model download, `ndarray` for tensor shaping.
//!
//! Target model: **BAAI/bge-large-en-v1.5** via the Xenova ONNX export
//! (Apache-2.0, 1024-dim, English, mean-pool + L2-normalize). Matches
//! the default dense embedder the rest of mnem uses, just with the
//! Ollama HTTP round-trip removed from the hot path.
//!
//! Also supports **sentence-transformers/all-MiniLM-L6-v2** via the
//! Xenova ONNX export (22M params, 384-dim). Chosen specifically for
//! byte-for-byte parity with ChromaDB's `DefaultEmbeddingFunction`:
//! same weights, same tokenizer, same pooling. Enables head-to-head
//! retrieval-quality comparisons against any ChromaDB-backed memory
//! framework without model-weight confounds.
//!
//! ## Why the round-trip mattered
//!
//! On the LongMemEval bake-off (2026-04-19), mnem's per-question wall
//! time was 26.8s against MemPalace's 0.14s. Profiling pointed at
//! `mnem-embed-providers::ollama`: every `Node::embed` on ingest and
//! every query vector on retrieve went through `POST
//! {ollama}/api/embeddings`. An in-process `Session::run` sits at
//! ~15-40ms for bge-large on laptop CPU, so eliminating the HTTP
//! layer alone buys roughly 10-20x; retrieve then becomes
//! index-bound, which is where the HNSW wiring in a sibling change
//! picks up the remaining slop.
//!
//! ## Output shape
//!
//! BGE-v1.5 sentence transformers produce a `last_hidden_state` of
//! `(batch, seq, hidden)`. Pooling is mean over seq positions weighted
//! by the attention mask, followed by L2-normalisation. The resulting
//! vector is unit-norm, so cosine similarity collapses to dot product
//! downstream.

// Picking BOTH `onnx` and `onnx-bundled` simultaneously would feed
// `ort` two mutually exclusive runtime-source features (load-dynamic
// vs download-binaries), producing a confusing linker error far from
// the actual cause. Reject early with a clear message.
#[cfg(all(feature = "onnx", feature = "onnx-bundled"))]
compile_error!(
    "mnem-embed-providers: enable exactly one of `onnx` or `onnx-bundled` (mutually exclusive)"
);

use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::OnceLock;

use ndarray::{Array2, ArrayViewD};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Value;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::embedder::Embedder;
use crate::error::EmbedError;
use crate::manifest::EmbedderManifest;

// ----------------------------------------------------------------------------
// Model registry
// ----------------------------------------------------------------------------

/// Which ONNX dense embedder to load.
///
/// Each variant pins (a) the HuggingFace repo id we lazy-download from,
/// (b) the tokenizer + model filenames, (c) the output dimension, and
/// (d) the positional-embedding ceiling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    /// `BAAI/bge-large-en-v1.5` via `Xenova/bge-large-en-v1.5`.
    /// 335M params, 1024 hidden, English. Apache-2.0. Default.
    BgeLargeEnV15,
    /// `BAAI/bge-base-en-v1.5` via `Xenova/bge-base-en-v1.5`.
    /// 109M params, 768 hidden, English. Apache-2.0.
    BgeBaseEnV15,
    /// `BAAI/bge-small-en-v1.5` via `Xenova/bge-small-en-v1.5`.
    /// 33M params, 384 hidden, English. Apache-2.0.
    BgeSmallEnV15,
    /// `sentence-transformers/all-MiniLM-L6-v2` via
    /// `Xenova/all-MiniLM-L6-v2`. 22M params, 384 hidden, English.
    /// Apache-2.0. Matches ChromaDB's DefaultEmbeddingFunction weights
    /// byte-for-byte, enabling apples-to-apples MemPalace comparisons.
    AllMiniLmL6V2,
}

impl ModelKind {
    fn repo_id(self) -> &'static str {
        match self {
            Self::BgeLargeEnV15 => "Xenova/bge-large-en-v1.5",
            Self::BgeBaseEnV15 => "Xenova/bge-base-en-v1.5",
            Self::BgeSmallEnV15 => "Xenova/bge-small-en-v1.5",
            Self::AllMiniLmL6V2 => "Xenova/all-MiniLM-L6-v2",
        }
    }

    fn onnx_path(self) -> &'static str {
        // Xenova exports all variants under `onnx/model.onnx`.
        "onnx/model.onnx"
    }

    /// Canonical fq identifier stamped on every Node.embed. MUST stay
    /// stable across mnem versions: two embedders with the same
    /// `model()` output MUST produce vectors in the same semantic
    /// space, and mnem keys the vector index on this string.
    fn wire_id(self) -> &'static str {
        match self {
            Self::BgeLargeEnV15 => "onnx:bge-large-en-v1.5",
            Self::BgeBaseEnV15 => "onnx:bge-base-en-v1.5",
            Self::BgeSmallEnV15 => "onnx:bge-small-en-v1.5",
            Self::AllMiniLmL6V2 => "onnx:all-MiniLM-L6-v2",
        }
    }

    /// Output vector dimension. Matches the HuggingFace release card.
    #[must_use]
    pub const fn dim(self) -> u32 {
        match self {
            Self::BgeLargeEnV15 => 1024,
            Self::BgeBaseEnV15 => 768,
            Self::BgeSmallEnV15 | Self::AllMiniLmL6V2 => 384,
        }
    }

    /// Default tokenizer max_length. BGE-v1.5 uses 512; MiniLM-L6-v2
    /// uses 256 (its sentence-transformers default; positional table
    /// is 512 but the model was trained at 256).
    #[must_use]
    pub const fn default_max_length(self) -> usize {
        match self {
            Self::BgeLargeEnV15 | Self::BgeBaseEnV15 | Self::BgeSmallEnV15 => 512,
            Self::AllMiniLmL6V2 => 256,
        }
    }

    /// Hard ceiling: going above triggers positional OOB at forward
    /// time. All four variants share BERT's 512-position table.
    #[must_use]
    pub const fn positional_limit(self) -> usize {
        512
    }

    /// Empirically-measured noise floor (cosine similarity between
    /// embeddings of unrelated texts) for this model. Gap 15: used by
    /// ingest to gate co-occurrence edges without a global constant.
    ///
    /// Values come from the measurement runs documented in
    /// `research/gap-catalog/15-ingest-no-edges/solution.md`.
    #[must_use]
    pub const fn noise_floor(self) -> f32 {
        match self {
            // MiniLM-L6-v2 is the target model the spec pins to 0.22.
            Self::AllMiniLmL6V2 => 0.22,
            // BGE-v1.5 family ran in the same experiment and landed
            // around 0.31, matching Ollama's BGE-M3 serve path.
            Self::BgeLargeEnV15 | Self::BgeBaseEnV15 | Self::BgeSmallEnV15 => 0.31,
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

/// Resolve the root directory for cached HF artifacts.
///
/// Honours `HF_HOME` when set (same convention huggingface-cli and the
/// Python `transformers` library use), falling back to
/// `$HOME/.cache/huggingface` (Unix) / `$USERPROFILE/.cache/huggingface`
/// (Windows) / `./.mnem-hf-cache` as a last resort. The `hub` suffix
/// matches the upstream HF layout so caches are discoverable by
/// adjacent tools.
fn hf_cache_root() -> PathBuf {
    if let Ok(v) = std::env::var("HF_HOME") {
        return PathBuf::from(v).join("hub");
    }
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE"));
    if let Ok(h) = home {
        return PathBuf::from(h)
            .join(".cache")
            .join("huggingface")
            .join("hub");
    }
    PathBuf::from(".mnem-hf-cache")
}

/// Download a single file from the HuggingFace Hub to the local
/// cache, returning the local path.
///
/// Why we do our own download instead of going through `hf-hub`:
/// the 0.4 release of `hf-hub` returns an error when the requested
/// path contains a `/` separator on some repo layouts (observed
/// empirically on `Xenova/bge-large-en-v1.5`'s `onnx/model.onnx`),
/// even though the underlying URL resolves cleanly via HTTP 302. We
/// bypass the issue by calling the `resolve/{revision}/{path}`
/// endpoint directly via `ureq`, which is already in this crate's
/// dependency tree and follows redirects by default.
///
/// Cache path shape: `{hf_root}/models--{org}--{repo}/resolve/{revision}/{file}`.
/// Deliberately NOT the `blobs/{sha}` + `snapshots/{commit}/{path}`
/// symlink layout that `hf-hub` and `huggingface-cli` use: that
/// layout requires resolving the commit SHA from an etag, adding
/// another round-trip per file for no real benefit to a runtime that
/// only needs (repo, revision, path) -> PathBuf.
fn fetch_to_cache(repo: &str, revision: &str, file: &str) -> Result<PathBuf, EmbedError> {
    let base = hf_cache_root();
    let repo_slug = format!("models--{}", repo.replace('/', "--"));
    let target = base
        .join(&repo_slug)
        .join("resolve")
        .join(revision)
        .join(file);

    // Existing-file short-circuit. Treat a zero-byte file as missing
    // so a previously-crashed download doesn't poison the cache.
    if target.is_file() {
        if let Ok(md) = fs::metadata(&target) {
            if md.len() > 0 {
                return Ok(target);
            }
        }
    }

    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            EmbedError::Config(format!("create cache dir {}: {e}", parent.display()))
        })?;
    }

    let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{file}");
    eprintln!("mnem-embed: downloading {} -> {}", url, target.display());
    let resp = ureq::get(&url)
        .call()
        .map_err(|e| EmbedError::Config(format!("hf download {url}: {e}")))?;
    let status = resp.status();
    if status != 200 {
        return Err(EmbedError::Config(format!(
            "hf download {url}: status {status}"
        )));
    }

    // Stream to a temp path and atomically rename; a crashed mid-
    // download otherwise leaves a truncated "valid" file in cache.
    let tmp = target.with_extension("download-partial");
    {
        let mut reader = resp.into_reader();
        let mut out = fs::File::create(&tmp)
            .map_err(|e| EmbedError::Config(format!("create {}: {e}", tmp.display())))?;
        io::copy(&mut reader, &mut out)
            .map_err(|e| EmbedError::Config(format!("download {}: {e}", target.display())))?;
    }
    fs::rename(&tmp, &target).map_err(|e| {
        EmbedError::Config(format!(
            "rename {} -> {}: {e}",
            tmp.display(),
            target.display()
        ))
    })?;

    Ok(target)
}

fn fetch_files(kind: ModelKind) -> Result<ModelFiles, EmbedError> {
    let revision = "main";
    let model_onnx = fetch_to_cache(kind.repo_id(), revision, kind.onnx_path())?;
    let tokenizer_json = fetch_to_cache(kind.repo_id(), revision, "tokenizer.json")?;
    Ok(ModelFiles {
        model_onnx,
        tokenizer_json,
    })
}

fn load_tokenizer(path: &Path, max_len: usize) -> Result<Tokenizer, EmbedError> {
    let mut tok = Tokenizer::from_file(path)
        .map_err(|e| EmbedError::Config(format!("tokenizer.json load: {e}")))?;
    tok.with_truncation(Some(TruncationParams {
        max_length: max_len,
        ..Default::default()
    }))
    .map_err(|e| EmbedError::Config(format!("tokenizer truncation: {e}")))?;
    tok.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        ..Default::default()
    }));
    Ok(tok)
}

/// Env var that overrides the tokenizer max_length at construction
/// time. Clamped to `ModelKind::positional_limit()`. Mirrors the
/// existing `MNEM_ONNX_SPARSE_MAX_LEN` / `MNEM_ONNX_RERANK_MAX_LEN`
/// pattern for a consistent operator surface across the three ONNX
/// crates.
const ENV_EMBED_MAX_LEN: &str = "MNEM_ONNX_EMBED_MAX_LEN";

fn resolve_max_length(kind: ModelKind, override_: Option<usize>) -> usize {
    let ceiling = kind.positional_limit();
    let requested = override_
        .or_else(|| {
            std::env::var(ENV_EMBED_MAX_LEN)
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
        })
        .unwrap_or_else(|| kind.default_max_length());
    if requested == 0 {
        eprintln!(
            "mnem-embed: requested max_length=0 for {}; snapping to default {}",
            kind.wire_id(),
            kind.default_max_length()
        );
        return kind.default_max_length();
    }
    if requested > ceiling {
        eprintln!(
            "mnem-embed: requested max_length={requested} exceeds {}'s positional limit {ceiling}; clamping",
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
    /// BGE exports typically ship `token_type_ids`; some stripped
    /// community exports drop it. Probed once at session-open time.
    needs_token_type_ids: bool,
}

impl OnnxSession {
    fn open(model_path: &Path) -> Result<Self, EmbedError> {
        // Intra-op thread default depends on which mode this binary
        // ships in:
        //
        // - `onnx` (load-dynamic, mnem-http server build):
        // default = 1. mnem keys its vector index on the raw
        // `Embedding` bytes; ORT reductions are NOT deterministic
        // across thread counts (f32 is non-associative under
        // reordered sums) so a single-threaded default keeps
        // `Node.embed` byte-stable across machines and replay
        // tests. The substrate-level federated-graph contract
        // wants this.
        //
        // - `onnx-bundled` (Path A, single-binary install):
        // default = `available_parallelism()`. The bundled-embedder
        // build is the local-developer / single-machine flow;
        // throughput is the dominant operator concern, and the
        // determinism trade-off is opt-out-able by setting
        // `MNEM_ORT_INTRA_THREADS=1` for users who do care.
        //
        // original default-1 left every
        // multicore CPU running keybert ingest on a single core,
        // turning a 5-min job into a 30-min job for no benefit a
        // local-only operator would notice. Differentiating per
        // feature lets the substrate brand keep its determinism
        // promise where it matters and lets the local CLI run fast
        // out of the box.
        #[cfg(feature = "onnx-bundled")]
        let default_threads: usize = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        #[cfg(not(feature = "onnx-bundled"))]
        let default_threads: usize = 1;
        let threads: usize = std::env::var("MNEM_ORT_INTRA_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(default_threads);
        #[allow(unused_mut)]
        let mut builder = Session::builder()
            .map_err(|e| EmbedError::Config(format!("ort session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| EmbedError::Config(format!("ort opt level: {e}")))?
            .with_intra_threads(threads)
            .map_err(|e| EmbedError::Config(format!("ort intra threads: {e}")))?;
        // GPU execution-provider registration. EPs are tried in the
        // order pushed below; ort dispatches to the first one that
        // supports the session's op set, then falls through to CPU.
        // Without either feature the call site is dead-code-eliminated
        // and the CPU EP (always implicit) handles everything.
        #[cfg(any(feature = "onnx-bundled-cuda", feature = "onnx-bundled-directml"))]
        {
            use ort::execution_providers::ExecutionProviderDispatch;
            #[allow(unused_mut)]
            let mut providers: Vec<ExecutionProviderDispatch> = Vec::new();
            #[cfg(feature = "onnx-bundled-cuda")]
            providers.push(ort::execution_providers::CUDAExecutionProvider::default().build());
            #[cfg(feature = "onnx-bundled-directml")]
            providers.push(ort::execution_providers::DirectMLExecutionProvider::default().build());
            builder = builder
                .with_execution_providers(providers)
                .map_err(|e| EmbedError::Config(format!("ort execution providers: {e}")))?;
        }
        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| EmbedError::Config(format!("ort commit {}: {e}", model_path.display())))?;
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
// Public embedder
// ----------------------------------------------------------------------------

/// In-process dense embedder backed by a local ONNX model.
///
/// BGE-v1.5 family: mean-pool over the last-hidden-state weighted by
/// the attention mask, then L2-normalize. Output is exactly
/// `ModelKind::dim()` f32 values per text.
pub struct OnnxEmbedder {
    kind: ModelKind,
    tokenizer: Tokenizer,
    session: Mutex<OnnxSession>,
    model_fq: String,
    max_len: usize,
}

/// Process-wide latch: first time any input fills the attention window
/// for a given (provider, model) tuple, emit a single stderr warning;
/// subsequent calls (across instances, threads, scorer loops) stay
/// silent. Keeps the truncation warning useful (operator sees it once)
/// without flooding the log on a 500-question bench run.
static TOKENIZER_TRUNCATE_WARNED: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();

fn warn_truncation_once(provider: &str, model: &str, max_len: usize, positional_limit: usize) {
    let key = format!("{provider}:{model}");
    let set = TOKENIZER_TRUNCATE_WARNED.get_or_init(|| Mutex::new(HashSet::new()));
    // Guard against poisoning: a poisoned mutex here is non-fatal -
    // worst case the warning prints twice. We still skip on the
    // happy path.
    let mut guard = match set.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if guard.insert(key) {
        eprintln!(
            "{provider}: input filled max_length={max_len} on {model}; tail truncated. \
 Raise via MNEM_ONNX_EMBED_MAX_LEN (<= {positional_limit}) or chunk upstream."
        );
    }
}

impl std::fmt::Debug for OnnxEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxEmbedder")
            .field("kind", &self.kind)
            .field("model_fq", &self.model_fq)
            .field("dim", &self.kind.dim())
            .field("max_len", &self.max_len)
            .finish()
    }
}

static ORT_INIT: OnceLock<()> = OnceLock::new();

fn ensure_ort_init() {
    ORT_INIT.get_or_init(|| {
        // Placeholder for future global ort setup (custom log routing,
        // execution-provider preferences). Kept here so the three
        // ONNX-using crates share one init point.
    });
}

impl OnnxEmbedder {
    /// Construct an embedder, lazy-downloading model + tokenizer from
    /// the HuggingFace Hub on first call. Uses the model's default
    /// max_length (512) unless `MNEM_ONNX_EMBED_MAX_LEN` overrides.
    ///
    /// # Errors
    ///
    /// [`EmbedError::Config`] on download failure, tokenizer parse
    /// failure, or ORT session-init failure.
    pub fn new(kind: ModelKind) -> Result<Self, EmbedError> {
        Self::with_max_length(kind, None)
    }

    /// Construct an embedder with an explicit tokenizer `max_length`.
    /// `None` defers to the env var / model default. Values above
    /// `ModelKind::positional_limit()` are clamped with a stderr
    /// warning.
    ///
    /// # Errors
    ///
    /// Same as [`Self::new`].
    pub fn with_max_length(kind: ModelKind, max_length: Option<usize>) -> Result<Self, EmbedError> {
        ensure_ort_init();
        let max_len = resolve_max_length(kind, max_length);
        let files = fetch_files(kind)?;
        let tokenizer = load_tokenizer(&files.tokenizer_json, max_len)?;
        let session = OnnxSession::open(&files.model_onnx)?;
        Ok(Self {
            kind,
            tokenizer,
            session: Mutex::new(session),
            model_fq: kind.wire_id().to_string(),
            max_len,
        })
    }

    /// Effective tokenizer max_length (post-clamp).
    #[must_use]
    pub fn max_length(&self) -> usize {
        self.max_len
    }

    /// Tokenize + forward + mean-pool + L2-normalise a single text.
    fn forward_single(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let encoded = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| EmbedError::Decode(format!("tokenize: {e}")))?;

        let seq_len = encoded.get_ids().len();
        if seq_len >= self.max_len {
            warn_truncation_once(
                "mnem-embed",
                self.kind.wire_id(),
                self.max_len,
                self.kind.positional_limit(),
            );
        }

        let ids: Vec<i64> = encoded.get_ids().iter().map(|&x| i64::from(x)).collect();
        let mask: Vec<i64> = encoded
            .get_attention_mask()
            .iter()
            .map(|&x| i64::from(x))
            .collect();
        // Keep a copy for the pooling step; the ndarray move consumes
        // the original.
        let mask_for_pool: Vec<f32> = mask.iter().map(|&x| x as f32).collect();

        let ids_arr = Array2::from_shape_vec((1, seq_len), ids)
            .map_err(|e| EmbedError::Decode(format!("ids reshape: {e}")))?;
        let mask_arr = Array2::from_shape_vec((1, seq_len), mask)
            .map_err(|e| EmbedError::Decode(format!("mask reshape: {e}")))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| EmbedError::Decode("session mutex poisoned".into()))?;

        let mut inputs: Vec<(&'static str, Value)> = Vec::with_capacity(3);
        inputs.push((
            "input_ids",
            Value::from_array(ids_arr)
                .map_err(|e| EmbedError::Decode(format!("ids tensor: {e}")))?
                .into_dyn(),
        ));
        inputs.push((
            "attention_mask",
            Value::from_array(mask_arr)
                .map_err(|e| EmbedError::Decode(format!("mask tensor: {e}")))?
                .into_dyn(),
        ));
        if session.needs_token_type_ids {
            let type_arr: Array2<i64> = Array2::zeros((1, seq_len));
            inputs.push((
                "token_type_ids",
                Value::from_array(type_arr)
                    .map_err(|e| EmbedError::Decode(format!("type_ids tensor: {e}")))?
                    .into_dyn(),
            ));
        }

        let outputs = session
            .session
            .run(inputs)
            .map_err(|e| EmbedError::Decode(format!("ort run: {e}")))?;

        // BGE ONNX exports emit the token-level hidden state as
        // `last_hidden_state` of shape (batch, seq, hidden). Fall back
        // to `token_embeddings` or the first output for stripped
        // exports.
        let value = outputs
            .iter()
            .find(|(name, _)| *name == "last_hidden_state" || *name == "token_embeddings")
            .map(|(_, v)| v)
            .or_else(|| outputs.iter().next().map(|(_, v)| v))
            .ok_or_else(|| EmbedError::Decode("no hidden-state output".into()))?;
        let view: ArrayViewD<'_, f32> = value
            .try_extract_array::<f32>()
            .map_err(|e| EmbedError::Decode(format!("extract hidden state: {e}")))?;
        let shape = view.shape().to_vec();
        if shape.len() != 3 || shape[0] != 1 {
            return Err(EmbedError::Decode(format!(
                "expected (1, seq, hidden) hidden state, got {shape:?}"
            )));
        }
        let seq = shape[1];
        let hidden = shape[2];
        let buffer: Vec<f32> = view.iter().copied().collect();
        drop(outputs);
        drop(session);

        // Mean-pool over sequence positions, weighted by the attention
        // mask. BGE-v1.5 follows the sentence-transformers recipe:
        //
        // pooled = sum_s (mask_s * H_s) / sum_s mask_s
        //
        // Then L2-normalise so cosine equals dot-product downstream.
        let mut pooled = vec![0.0_f32; hidden];
        let mut denom = 0.0_f32;
        for s in 0..seq {
            let m = mask_for_pool.get(s).copied().unwrap_or(0.0);
            if m == 0.0 {
                continue;
            }
            denom += m;
            let row = &buffer[s * hidden..(s + 1) * hidden];
            for (i, v) in row.iter().enumerate() {
                pooled[i] += m * v;
            }
        }
        if denom > 0.0 {
            let inv = 1.0_f32 / denom;
            for v in &mut pooled {
                *v *= inv;
            }
        }
        // L2-normalise. Guard the zero-vector edge case.
        let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            let inv = 1.0_f32 / norm;
            for v in &mut pooled {
                *v *= inv;
            }
        }

        let expected = self.kind.dim() as usize;
        if pooled.len() != expected {
            return Err(EmbedError::DimMismatch {
                expected: self.kind.dim(),
                got: u32::try_from(pooled.len()).unwrap_or(u32::MAX),
            });
        }
        Ok(pooled)
    }
}

impl Embedder for OnnxEmbedder {
    fn model(&self) -> &str {
        &self.model_fq
    }

    fn dim(&self) -> u32 {
        self.kind.dim()
    }

    fn manifest(&self) -> EmbedderManifest {
        EmbedderManifest::new(
            self.model_fq.clone(),
            self.kind.dim(),
            self.kind.noise_floor(),
        )
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.forward_single(text)
    }

    // Truly-batched forward. Tokenises the whole list via
    // `encode_batch` (BatchLongest padding is already configured on
    // the tokenizer), stacks into a `(batch, seq_max)` input, runs
    // ORT once, and pools per row with the per-row attention mask.
    //
    // Byte-identity with the fan-out path is contract-bound: mean-
    // pool + L2-norm of a row in a batched last_hidden_state is the
    // same function of (ids, mask) as the single-item path, modulo
    // f32 rounding on the extra zero-padded positions (which the
    // attention mask zeroes out before they enter the denominator).
    // Pinned by `onnx_batch_matches_fanout` when ONNX is available.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        // Single-text shortcut: keeps the hot path byte-identical for
        // the common "embed one query" caller.
        if texts.len() == 1 {
            return Ok(vec![self.forward_single(texts[0])?]);
        }
        self.forward_batch(texts)
    }
}

impl OnnxEmbedder {
    /// Batched tokenize + forward + mean-pool + L2-normalise. See
    /// [`Embedder::embed_batch`] for the contract.
    fn forward_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        // Tokenizer is configured with PaddingStrategy::BatchLongest at
        // construction time (see `load_tokenizer`), so encode_batch
        // already pads rows to the longest element in the batch.
        let inputs_vec: Vec<tokenizers::EncodeInput<'_>> = texts
            .iter()
            .map(|t| tokenizers::EncodeInput::Single((*t).into()))
            .collect();
        let encoded = self
            .tokenizer
            .encode_batch(inputs_vec, true)
            .map_err(|e| EmbedError::Decode(format!("tokenize batch: {e}")))?;

        let batch = encoded.len();
        let seq_len = encoded.first().map(|e| e.get_ids().len()).unwrap_or(0);
        if seq_len == 0 {
            // Every input was empty. Degrade to the fan-out which
            // handles per-item edge cases (zero-vector output).
            return texts.iter().map(|t| self.forward_single(t)).collect();
        }
        // Sanity: BatchLongest must have padded every row to the same
        // length. If not, fall through to the fan-out rather than
        // emit wrong-shape tensors.
        if encoded.iter().any(|e| e.get_ids().len() != seq_len) {
            return texts.iter().map(|t| self.forward_single(t)).collect();
        }

        // One-shot truncation warning, same as the single path.
        if seq_len >= self.max_len {
            warn_truncation_once(
                "mnem-embed",
                self.kind.wire_id(),
                self.max_len,
                self.kind.positional_limit(),
            );
        }

        let total = batch * seq_len;
        let mut ids: Vec<i64> = Vec::with_capacity(total);
        let mut mask: Vec<i64> = Vec::with_capacity(total);
        for e in &encoded {
            ids.extend(e.get_ids().iter().map(|&x| i64::from(x)));
            mask.extend(e.get_attention_mask().iter().map(|&x| i64::from(x)));
        }
        let mask_for_pool: Vec<f32> = mask.iter().map(|&x| x as f32).collect();

        let ids_arr = Array2::from_shape_vec((batch, seq_len), ids)
            .map_err(|e| EmbedError::Decode(format!("ids reshape: {e}")))?;
        let mask_arr = Array2::from_shape_vec((batch, seq_len), mask)
            .map_err(|e| EmbedError::Decode(format!("mask reshape: {e}")))?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| EmbedError::Decode("session mutex poisoned".into()))?;

        let mut inputs: Vec<(&'static str, Value)> = Vec::with_capacity(3);
        inputs.push((
            "input_ids",
            Value::from_array(ids_arr)
                .map_err(|e| EmbedError::Decode(format!("ids tensor: {e}")))?
                .into_dyn(),
        ));
        inputs.push((
            "attention_mask",
            Value::from_array(mask_arr)
                .map_err(|e| EmbedError::Decode(format!("mask tensor: {e}")))?
                .into_dyn(),
        ));
        if session.needs_token_type_ids {
            let type_arr: Array2<i64> = Array2::zeros((batch, seq_len));
            inputs.push((
                "token_type_ids",
                Value::from_array(type_arr)
                    .map_err(|e| EmbedError::Decode(format!("type_ids tensor: {e}")))?
                    .into_dyn(),
            ));
        }

        let outputs = session
            .session
            .run(inputs)
            .map_err(|e| EmbedError::Decode(format!("ort run: {e}")))?;

        let value = outputs
            .iter()
            .find(|(name, _)| *name == "last_hidden_state" || *name == "token_embeddings")
            .map(|(_, v)| v)
            .or_else(|| outputs.iter().next().map(|(_, v)| v))
            .ok_or_else(|| EmbedError::Decode("no hidden-state output".into()))?;
        let view: ArrayViewD<'_, f32> = value
            .try_extract_array::<f32>()
            .map_err(|e| EmbedError::Decode(format!("extract hidden state: {e}")))?;
        let shape = view.shape().to_vec();
        if shape.len() != 3 || shape[0] != batch || shape[1] != seq_len {
            return Err(EmbedError::Decode(format!(
                "expected ({batch}, {seq_len}, hidden) hidden state, got {shape:?}"
            )));
        }
        let hidden = shape[2];
        let buffer: Vec<f32> = view.iter().copied().collect();
        drop(outputs);
        drop(session);

        let expected = self.kind.dim() as usize;
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(batch);
        let row_stride = seq_len * hidden;
        for b in 0..batch {
            let mut pooled = vec![0.0_f32; hidden];
            let mut denom = 0.0_f32;
            let row_base = b * row_stride;
            let mask_base = b * seq_len;
            for s in 0..seq_len {
                let m = mask_for_pool[mask_base + s];
                if m == 0.0 {
                    continue;
                }
                denom += m;
                let tok_base = row_base + s * hidden;
                let row = &buffer[tok_base..tok_base + hidden];
                for (i, v) in row.iter().enumerate() {
                    pooled[i] += m * v;
                }
            }
            if denom > 0.0 {
                let inv = 1.0_f32 / denom;
                for v in &mut pooled {
                    *v *= inv;
                }
            }
            let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                let inv = 1.0_f32 / norm;
                for v in &mut pooled {
                    *v *= inv;
                }
            }
            if pooled.len() != expected {
                return Err(EmbedError::DimMismatch {
                    expected: self.kind.dim(),
                    got: u32::try_from(pooled.len()).unwrap_or(u32::MAX),
                });
            }
            out.push(pooled);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_max_length_uses_default_when_none() {
        let n = resolve_max_length(ModelKind::BgeLargeEnV15, None);
        assert_eq!(n, 512);
    }

    #[test]
    fn resolve_max_length_clamps_above_positional_limit() {
        let n = resolve_max_length(ModelKind::BgeLargeEnV15, Some(8192));
        assert_eq!(n, 512);
    }

    #[test]
    fn resolve_max_length_zero_snaps_to_default() {
        let n = resolve_max_length(ModelKind::BgeBaseEnV15, Some(0));
        assert_eq!(n, 512);
    }

    #[test]
    fn dims_match_published_sizes() {
        assert_eq!(ModelKind::BgeLargeEnV15.dim(), 1024);
        assert_eq!(ModelKind::BgeBaseEnV15.dim(), 768);
        assert_eq!(ModelKind::BgeSmallEnV15.dim(), 384);
        assert_eq!(ModelKind::AllMiniLmL6V2.dim(), 384);
    }

    #[test]
    fn wire_ids_are_stable_and_namespaced() {
        assert_eq!(ModelKind::BgeLargeEnV15.wire_id(), "onnx:bge-large-en-v1.5");
        assert_eq!(ModelKind::BgeBaseEnV15.wire_id(), "onnx:bge-base-en-v1.5");
        assert_eq!(ModelKind::BgeSmallEnV15.wire_id(), "onnx:bge-small-en-v1.5");
        assert_eq!(ModelKind::AllMiniLmL6V2.wire_id(), "onnx:all-MiniLM-L6-v2");
    }

    #[test]
    fn minilm_default_max_length_is_256() {
        assert_eq!(ModelKind::AllMiniLmL6V2.default_max_length(), 256);
    }
}
