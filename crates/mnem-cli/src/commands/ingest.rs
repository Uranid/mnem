//! `mnem ingest <path>` - parse external source files into the graph.
//!
//! Wires `mnem-ingest`'s [`Ingester`] pipeline behind a CLI surface so
//! an operator can point at a file or directory and get a committed
//! Doc + Chunk + Entity subgraph.
//!
//! ## Supported sources
//!
//! - `.md` / `.markdown` - CommonMark + GFM, paragraph-chunked.
//! - `.txt` - plain text, one section.
//! - `.pdf` - pure-Rust text-layer extraction.
//! - `.json` / `.jsonl` - chat-conversation exports (ChatGPT / Claude / generic).
//! - `.rs`, `.py`, `.js`, `.ts`, `.go`, `.java`, `.c`, `.cpp`, `.rb`, `.cs` and more -
//!   code files parsed by tree-sitter into function/class-level chunks (`SourceKind::Code`).
//! - `.yaml`, `.toml`, `.sql`, `.html`, `.sh`, `.php`, `.swift`, `.kt`, `.lua`, `.zig`
//!   and other structured, script, and code-like formats without a tree-sitter grammar -
//!   routed to `SourceKind::Text` for sentence-aware chunking.
//!
//! Unknown extensions fall back to `SourceKind::Text` so `README`
//! without an extension still ingests cleanly.
//!
//! ## Examples
//!
//! ```text
//! mnem ingest notes.md
//! mnem ingest --chunker recursive --max-tokens 1024 book.pdf
//! mnem ingest --recursive docs/
//! ```

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Args as ClapArgs;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use mnem_ingest::{
    ChunkerKind, IngestConfig, IngestResult, Ingester, Section, SourceKind,
    chunk as chunk_sections, resolve_chunker,
};
use tracing::{info, info_span};

use crate::config;
use crate::repo;

/// `mnem ingest` arguments.
#[derive(ClapArgs, Debug)]
#[command(after_long_help = "\
Examples:
 mnem ingest notes.md
 mnem ingest --text \"The quick brown fox\"
 mnem ingest --chunker recursive --max-tokens 1024 book.pdf
 mnem ingest --recursive docs/
")]
pub(crate) struct Args {
    /// Path to a file, or a directory when `--recursive` is set.
    /// Mutually exclusive with `--text`.
    #[arg(conflicts_with = "text")]
    pub path: Option<PathBuf>,

    /// Inline text to ingest directly (without a file).
    /// Mutually exclusive with `<PATH>` and `--recursive`.
    #[arg(long, conflicts_with = "path", conflicts_with = "recursive")]
    pub text: Option<String>,

    /// Root Doc node label (e.g. `Doc`, `Note`, `Transcript`). Default `Doc`.
    #[arg(long, default_value = "Doc")]
    pub ntype: String,

    /// Chunker strategy. `auto` picks per source kind (default).
    /// Explicit choices: `session` | `paragraph` | `recursive` |
    /// `sentence_recursive` | `structural`.
    #[arg(long, default_value = "auto")]
    pub chunker: String,

    /// Target tokens per chunk (recursive chunker). Clamped at 8192.
    #[arg(long, default_value_t = 512)]
    pub max_tokens: u32,

    /// Overlap tokens between adjacent chunks (recursive chunker).
    #[arg(long, default_value_t = 32)]
    pub overlap: u32,

    /// Walk directory trees. When set, `path` must be a directory and
    /// each supported file under it is ingested as one Doc.
    /// Mutually exclusive with `--text`.
    #[arg(long, conflicts_with = "text")]
    pub recursive: bool,

    /// Commit message. Overridable per-run; the default embeds the
    /// ingested file count so the op-log stays self-describing.
    #[arg(long, short = 'm')]
    pub message: Option<String>,

    /// Entity / relation extractor. `none` (default) keeps the built-in
    /// rule-based [`RuleExtractor`]. `keybert` swaps in the statistical
    /// KeyBERT adapter - requires the `keybert` feature on the mnem-cli
    /// build; an error is emitted if the binary was compiled without it.
    #[arg(long, default_value = "none")]
    pub extractor: String,

    /// Skip the auto-index phase that runs after a successful ingest.
    /// `mnem ingest` previously committed
    /// chunks without vectors, forcing a manual `mnem reindex` before
    /// `mnem retrieve` would surface anything semantic. With this
    /// flag absent, the CLI now invokes the reindex driver
    /// automatically when an `[embed]` provider is configured, so
    /// ingest → retrieve works in one shot. Pass `--no-embed` to
    /// preserve the legacy two-step flow (e.g. for bulk imports
    /// where you'll batch reindex later).
    #[arg(long)]
    pub no_embed: bool,
}

/// Upper bound on `--max-tokens`. Mirrors the MCP + HTTP clamps so a
/// caller that migrates between surfaces sees the same ceiling.
const MAX_TOKENS_CAP: u32 = 8192;

/// Extensions we recurse into when `--recursive` is set. Anything else
/// is skipped silently.
// NOTE: The following extensions are recognised by `CodeLanguage::from_extension`
// (and therefore ingestible via single-file `mnem ingest <file>`) but are NOT listed
// here, so `--recursive` walks silently skip them. Add them here if recursive support
// is needed: .pyi (Python stubs), .mjs/.cjs (ESM/CJS), .c++ (alt C++ extension).
// See also: mnem-ingest/src/types.rs CodeLanguage::from_extension.
const SUPPORTED_EXTS: &[&str] = &[
    // Documents and prose
    "md", "markdown", "txt", "pdf", "json", "jsonl",
    // Structured text / data (routed to Text; sentence-aware chunking)
    "yaml", "yml", "toml", "xml", "html", "htm", "csv", "sql",
    // Code: tree-sitter parsed (function-level chunks)
    "rs", "py", "js", "ts", "tsx", "mts", "cts", "go", "java", "c", "cpp", "cc", "cxx", "h", "hpp", "hxx",
    "rb", "gemspec", "rake", "erb", "cs", "csx",
    // Code: no tree-sitter grammar, routed to Text
    "sh", "bash", "zsh", "fish",
    "php", "swift", "kt", "kts", "scala", "lua",
    "ex", "exs", "hs", "lhs", "r", "zig",
    // Config / script formats also routed to Text
    "ini", "conf", "env",
];

/// Run `mnem ingest`.
///
/// # Errors
///
/// Returns an error if the repo cannot be opened, the source path
/// does not exist, a file cannot be read, or the pipeline rejects
/// the payload (e.g. malformed UTF-8 on a Markdown file).
pub(crate) fn run(override_path: Option<&Path>, a: Args) -> Result<()> {
    let _span = info_span!("mnem-ingest").entered();

    if a.max_tokens > MAX_TOKENS_CAP {
        bail!(
            "--max-tokens {} exceeds the {MAX_TOKENS_CAP} cap; lower it \
 or raise the ceiling in code",
            a.max_tokens
        );
    }

    let data_dir = repo::locate_data_dir(override_path)?;
    let cfg = config::load(&data_dir)?;
    let r = repo::open_repo(Some(data_dir.as_path()))?;

    // Resolve the source: --text supplies bytes directly as SourceKind::Text;
    // <PATH> reads from the filesystem (file or recursive directory walk).
    // Exactly one must be provided; clap's `conflicts_with` catches the
    // "both supplied" case, and we bail here for "neither supplied".
    let text_bytes: Option<Vec<u8>> = a.text.as_deref().map(|s| s.as_bytes().to_vec());

    let files: Vec<PathBuf> = if text_bytes.is_some() {
        // Bypass file collection entirely; we'll synthesise one PreReadFile
        // entry below from the inline text.
        Vec::new()
    } else if let Some(ref p) = a.path {
        if a.recursive {
            collect_files(p)?
        } else {
            if !p.is_file() {
                bail!(
                    "{} is not a file; pass --recursive to walk a directory",
                    p.display()
                );
            }
            vec![p.clone()]
        }
    } else {
        bail!("either <PATH> or --text must be provided");
    };

    if text_bytes.is_none() && files.is_empty() {
        bail!(
            "no ingestable files found under {}",
            a.path
                .as_deref()
                .unwrap_or_else(|| std::path::Path::new("<unknown>"))
                .display()
        );
    }

    // Warn when the caller passes --max-tokens with a chunker that ignores it.
    // `auto` is excluded: it maps Text/Pdf to SentenceRecursive, so the flag
    // has a real effect. Only `paragraph` and `session` genuinely ignore it.
    const DEFAULT_MAX_TOKENS: u32 = 512;
    if a.max_tokens != DEFAULT_MAX_TOKENS
        && matches!(
            a.chunker.to_ascii_lowercase().as_str(),
            "paragraph" | "session"
        )
    {
        eprintln!(
            "warning: --max-tokens has no effect with --chunker {}; \
             use --chunker recursive to enable token-based splitting",
            a.chunker
        );
    }

    // Extractor selector. Default (`none`) keeps the built-in
    // `RuleExtractor`. `keybert` (C3 FIX-3) wires the statistical
    // KeyBERT adapter; it needs an embedder opened from the
    // operator's `[embed]` config, so we resolve one upfront and
    // share it across every file in this ingest run.
    let keybert_embedder: Option<std::sync::Arc<dyn mnem_embed_providers::Embedder>> =
        match a.extractor.as_str() {
            "none" | "" => None,
            "keybert" => {
                // resolve via `config::resolve_embedder`
                // so MNEM_EMBED_* env vars and the user-global
                // `~/.mnem/config.toml` both work as fallbacks. Per-repo
                // `.mnem/config.toml` still wins when set; precedence
                // matches `mnem retrieve` so behaviour is symmetric
                // across the two embedder consumers.
                let pc = config::resolve_embedder(&cfg).ok_or_else(|| {
                    anyhow::anyhow!(
                        "--extractor keybert requires an `[embed]` provider; checked \
 MNEM_EMBED_PROVIDER env, per-repo `.mnem/config.toml`, and \
 the user-global `~/.mnem/config.toml`. Configure one with \
 `mnem config set embed.provider ollama` (per-repo) or write \
 a `[embed]` section to `~/.mnem/config.toml` (global)."
                    )
                })?;
                let boxed = mnem_embed_providers::open(&pc)
                    .map_err(|e| anyhow::anyhow!("opening embed provider for keybert: {e}"))?;
                // Box<dyn Embedder> -> Arc<dyn Embedder> via the
                // conversion trait Rust ships for unsized pointees.
                Some(std::sync::Arc::from(boxed))
            }
            other => bail!("unknown --extractor {other}; expected one of: none, keybert"),
        };

    // Pre-walk: read every file once, parse to sections, run the
    // chunker to count the chunks each file will produce. Two reasons:
    // (1) `Ingester::ingest` re-parses internally so re-reading later
    // costs only a sub-millisecond per file vs the embedder cost we
    // are trying to surface, and (2) summing the per-file chunk counts
    // gives the progress bar a chunk-level total instead of a 3-of-3
    // file-level total - single huge files (Bible books, long PDFs)
    // would otherwise show "0/N" for the entire keybert pass.
    //
    // Memory: holds every file's bytes in RAM during phase 1 so the
    // ingest phase can reuse them without a re-read. Within typical
    // agent-memory corpora this is megabytes, not gigabytes; truly
    // large book ingests should pre-split files.
    struct PreReadFile {
        path: PathBuf,
        bytes: Vec<u8>,
        kind: SourceKind,
        chunker: ChunkerKind,
        chunk_count: u64,
    }
    let mut pre: Vec<PreReadFile> = Vec::with_capacity(files.len().max(1));
    if let Some(bytes) = text_bytes {
        // Inline text: treat as a single plain-text source named "<inline>".
        let kind = SourceKind::Text;
        let chunker =
            resolve_chunker(&a.chunker, kind, a.max_tokens, a.overlap).context("--chunker")?;
        let chunk_count = count_chunks_for(&bytes, kind, &chunker).unwrap_or(0);
        pre.push(PreReadFile {
            path: PathBuf::from("<inline>"),
            bytes,
            kind,
            chunker,
            chunk_count,
        });
    } else {
        for file in &files {
            let kind = Ingester::source_kind_for_path(file);
            let bytes =
                std::fs::read(file).with_context(|| format!("reading {}", file.display()))?;
            let chunker =
                resolve_chunker(&a.chunker, kind, a.max_tokens, a.overlap).context("--chunker")?;
            // count_chunks_for is best-effort; on parse failure we fall
            // back to a per-file bar (chunk_count = 0 marks "unknown").
            let chunk_count = count_chunks_for(&bytes, kind, &chunker).unwrap_or(0);
            pre.push(PreReadFile {
                path: file.clone(),
                bytes,
                kind,
                chunker,
                chunk_count,
            });
        }
    }
    let total_chunks: u64 = pre.iter().map(|f| f.chunk_count).sum();
    // If pre-count succeeded for every file, drive the bar in chunks;
    // otherwise fall back to per-file ticks so the bar still moves.
    let use_chunk_progress = total_chunks > 0 && pre.iter().all(|f| f.chunk_count > 0);

    let started = Instant::now();
    let mut totals = Totals::default();
    let mut tx = r.start_transaction();

    // Single in-place progress bar. Total is chunk-level when the
    // pre-walk produced counts (so a 4000-chunk Bible run shows
    // "1234/4000" mid-Genesis instead of "0/3"); per-file otherwise.
    // `enable_steady_tick` redraws the bar every 120ms even when the
    // position has not moved, so long single-file embedding passes
    // keep ticking elapsed/spinner instead of looking frozen.
    let pb_total = if use_chunk_progress {
        total_chunks
    } else {
        pre.len() as u64
    };
    let pb = ProgressBar::new(pb_total);
    pb.set_style(
        ProgressStyle::with_template(
            " [{elapsed_precise}] {bar:32.cyan/blue} {pos}/{len} ({percent}%) ETA {eta} {msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    // Route bar to stdout so a stderr-side warning from the embedder
    // (e.g. `mnem-embed: input filled max_length=...`) cannot break
    // the bar's in-place ANSI redraw and leave a duplicate frozen
    // line above the live one. Terminal still receives both streams;
    // they just stop competing for the same cursor anchor.
    pb.set_draw_target(ProgressDrawTarget::stdout());
    pb.enable_steady_tick(Duration::from_millis(120));

    // Per-chunk progress callback (). Fires from
    // inside `Ingester::ingest` after every chunk is written, so the
    // bar moves smoothly even mid-Genesis.md instead of waiting for
    // the whole file to commit.
    let progress_cb: Option<std::sync::Arc<dyn Fn() + Send + Sync>> = if use_chunk_progress {
        let pb_cb = pb.clone();
        Some(std::sync::Arc::new(move || {
            pb_cb.inc(1);
        }))
    } else {
        None
    };

    for f in &pre {
        let config = IngestConfig {
            chunker: f.chunker.clone(),
            ntype: a.ntype.clone(),
            max_tokens: a.max_tokens,
            overlap: a.overlap,
            ner: config::resolve_ner(&cfg),
        };
        let mut ing = Ingester::new(config);
        if let Some(emb) = &keybert_embedder {
            ing = ing.with_extractor(Box::new(mnem_ingest::KeyBertAdapter::new(
                emb.clone(),
                "Keyword",
            )));
        }
        if let Some(cb) = &progress_cb {
            ing = ing.with_progress(cb.clone());
        }

        let display_name = f
            .path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<unnamed>")
            .to_string();
        pb.set_message(display_name);
        info!(path = %f.path.display(), kind = ?f.kind, "ingesting");
        let result = ing
            .ingest(&mut tx, &f.bytes, f.kind)
            .with_context(|| format!("ingest failed on {}", f.path.display()))?;
        totals.add(&result);
        // When chunk-level progress is on, the callback already inc'd
        // the bar once per chunk inside the ingest loop; nothing more
        // to do here. Per-file fallback still ticks one unit per file.
        if !use_chunk_progress {
            pb.inc(1);
        }
    }
    pb.finish_and_clear();

    let file_count = pre.len();
    let default_msg = format!("mnem ingest: {file_count} file(s)");
    let msg = a.message.as_deref().unwrap_or(&default_msg);
    let new_r = tx.commit(&config::author_string(&cfg), msg)?;

    let elapsed_ms = started.elapsed().as_millis();
    println!(
        "ingested {file_count} files, {} chunks, {} nodes, {} edges in {}ms",
        totals.chunk_count, totals.node_count, totals.edge_count, elapsed_ms
    );
    println!(" op_id {}", new_r.op_id());
    if let Some(head) = new_r.view().heads.first() {
        println!(" commit_cid {head}");
    }

    // Drop ALL ingest-side handles so the reindex driver can reopen
    // the redb file under its own RW lock. Both `r` (original open)
    // and `new_r` (post-commit handle) own Arc<Database> clones -
    // dropping just one is not enough; redb refuses a second
    // `Database::open` while any clone lives. Also drop the keybert
    // adapter's embedder Arc so a future ORT-EP-locking provider
    // does not race with reindex's own embedder open. `tx` was
    // consumed by `tx.commit(...)`.
    drop(new_r);
    drop(r);
    drop(keybert_embedder);

    // auto-index newly-committed nodes when an
    // embedder is configured and the operator did not opt out.
    // Symmetric with `mnem add node`'s auto-embed contract from
    // commit c68a6b2. Two visible phases (ingest → reindex)
    // each with their own progress bar + completion message; the
    // existing reindex driver provides both. When no embedder is
    // configured the call is skipped silently - the legacy
    // ingest-then-manual-reindex path still works unchanged.
    if !a.no_embed && config::resolve_embedder(&cfg).is_some() {
        println!();
        super::reindex::run(
            override_path,
            super::reindex::Args {
                force: false,
                label: None,
                since: None,
                dry_run: false,
                message: None,
                lift_legacy_extra: false,
                lift_legacy_sparse: false,
            },
        )?;
    }
    Ok(())
}

/// Accumulated counters across a multi-file ingest run. `elapsed_ms`
/// on the emitted summary is wall-clock for the whole CLI invocation,
/// not the sum of per-file pipelines.
#[derive(Default)]
struct Totals {
    node_count: u64,
    chunk_count: u64,
    entity_count: u64,
    relation_count: u64,
    edge_count: u64,
}

impl Totals {
    fn add(&mut self, r: &IngestResult) {
        self.node_count = self.node_count.saturating_add(r.node_count);
        self.chunk_count = self.chunk_count.saturating_add(r.chunk_count);
        self.entity_count = self.entity_count.saturating_add(r.entity_count);
        self.relation_count = self.relation_count.saturating_add(r.relation_count);
        self.edge_count = self.edge_count.saturating_add(r.edge_count);
    }
}


/// Best-effort estimate of the chunk count the ingest pipeline will
/// produce for a given source file. Used by the CLI's progress bar to
/// drive a chunk-level total (so a 4000-chunk Bible-by-book run shows
/// real granularity instead of "0/3"). Errors are mapped to the
/// caller, which falls back to a per-file bar.
fn count_chunks_for(bytes: &[u8], kind: SourceKind, chunker: &ChunkerKind) -> Result<u64> {
    let sections: Vec<Section> = match kind {
        SourceKind::Markdown => {
            let s = std::str::from_utf8(bytes).with_context(|| "non-utf8 markdown source")?;
            mnem_ingest::md::parse_markdown(s).map_err(|e| anyhow::anyhow!(e.to_string()))?
        }
        SourceKind::Text => {
            let s = std::str::from_utf8(bytes).with_context(|| "non-utf8 text source")?;
            mnem_ingest::text::parse_text(s).map_err(|e| anyhow::anyhow!(e.to_string()))?
        }
        SourceKind::Pdf => {
            mnem_ingest::pdf::parse_pdf(bytes).map_err(|e| anyhow::anyhow!(e.to_string()))?
        }
        SourceKind::Conversation => mnem_ingest::conversation::parse_conversation(bytes)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?,
        SourceKind::Code(lang) => {
            let s = std::str::from_utf8(bytes).with_context(|| "non-utf8 code source")?;
            mnem_ingest::code::parse_code(s, lang).map_err(|e| anyhow::anyhow!(e.to_string()))?
        }
    };
    Ok(u64::try_from(chunk_sections(&sections, chunker).len()).unwrap_or(u64::MAX))
}

/// Recursively collect every file under `root` whose extension matches
/// [`SUPPORTED_EXTS`]. Symlinks are followed; hidden files are not
/// skipped (same policy as `mnem add` on a file path - the caller's
/// shell decides).
fn collect_files(root: &Path) -> Result<Vec<PathBuf>> {
    if !root.is_dir() {
        bail!(
            "{} is not a directory; drop --recursive to ingest a single file",
            root.display()
        );
    }
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries =
            std::fs::read_dir(&dir).with_context(|| format!("reading dir {}", dir.display()))?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if let Some(ext) = path
                .extension()
                .and_then(|e| e.to_str())
                .map(str::to_ascii_lowercase)
                && SUPPORTED_EXTS.contains(&ext.as_str())
            {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}
