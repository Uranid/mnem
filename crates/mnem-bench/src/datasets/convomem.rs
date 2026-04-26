//! ConvoMem (Salesforce) dataset spec + loader.
//!
//! Source: HuggingFace `Salesforce/ConvoMem`,
//! `core_benchmark/evidence_questions/<category>/1_evidence/<file>.json`.
//!
//! Layout differs from LongMemEval / LoCoMo: there is no single
//! download. Instead the bench-harness ships a small bundled manifest
//! listing one shard URL per (category, evidence_file). On `fetch`
//! we walk the manifest, download each shard into the cache, and
//! merge the `evidence_items` arrays into a single
//! `convomem_evidence.json` blob the scorer consumes.
//!
//! Cache layout:
//!
//! ```text
//! ~/.mnem/bench-data/convomem/
//!   convomem_evidence.json      <- merged blob, what the scorer loads
//!   shards/<category>/<file>    <- per-shard cache (idempotent fetches)
//! ```
//!
//! The merged blob is treated as the canonical artefact for
//! [`crate::datasets::is_cached`] / [`crate::datasets::cached_path`].

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use super::DatasetSpec;
use crate::bench::Bench;

/// HuggingFace base URL for the ConvoMem evidence_questions tree.
pub const HF_BASE: &str =
    "https://huggingface.co/datasets/Salesforce/ConvoMem/resolve/main/core_benchmark/evidence_questions";

/// Five headline categories used by the MemPalace headline numbers.
/// Ordering matches `convomem.py`'s default; identical to the
/// upstream `CATEGORIES` keys minus `changing_evidence` (which is
/// unstable across MemPalace runs and excluded from the headline).
pub const HEADLINE_CATEGORIES: &[&str] = &[
    "assistant_facts_evidence",
    "implicit_connection_evidence",
    "preference_evidence",
    "user_evidence",
    "abstention_evidence",
];

/// HuggingFace tree API base. The fetcher hits
/// `<TREE_API>/<category>/1_evidence` to discover the per-role
/// shard filenames, then downloads each via [`HF_BASE`].
pub const TREE_API: &str =
    "https://huggingface.co/api/datasets/Salesforce/ConvoMem/tree/main/core_benchmark/evidence_questions";

/// Default per-category shard cap. The headline 50/cat slice fits
/// well under this; raise via `MNEM_BENCH_CONVOMEM_PER_CAT` for the
/// full sweep.
pub const DEFAULT_PER_CATEGORY_CAP: usize = 50;

/// Static spec. The ConvoMem fetcher does NOT use this URL directly
/// (it walks the HF tree API instead) but keeping the field
/// non-empty keeps the [`crate::datasets::DatasetSpec`] surface
/// uniform with LongMemEval / LoCoMo. `sha256` is empty because the
/// merged blob is composed at fetch-time.
pub const SPEC: DatasetSpec = DatasetSpec {
    bench: Bench::Convomem,
    filename: "convomem_evidence.json",
    url: "https://huggingface.co/datasets/Salesforce/ConvoMem",
    sha256: "",
    bytes: 5 * 1024 * 1024,
};

/// One conversation message inside an evidence item.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Message {
    /// Speaker tag. Free-form, usually `"user"` or `"assistant"`.
    #[serde(default)]
    pub speaker: String,
    /// Message body.
    #[serde(default)]
    pub text: String,
}

/// One conversation (a chat thread).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Conversation {
    /// Messages within the conversation.
    #[serde(default)]
    pub messages: Vec<Message>,
}

/// One piece of gold evidence: a substring expected to appear in
/// retrieved candidates.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MessageEvidence {
    /// Substring that must appear in any retrieved candidate (either
    /// direction) for the evidence to count as "found".
    #[serde(default)]
    pub text: String,
}

/// One evidence item the scorer processes.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvidenceItem {
    /// Question text.
    pub question: String,
    /// Conversation history to ingest.
    #[serde(default)]
    pub conversations: Vec<Conversation>,
    /// Gold evidence substrings.
    #[serde(default)]
    pub message_evidences: Vec<MessageEvidence>,
    /// Category bucket (filled in by the loader; never present in
    /// the upstream JSON).
    #[serde(default, rename = "_category_key")]
    pub category_key: String,
}

/// File format on disk. Mirrors the merged-blob shape produced by
/// [`fetch_into`].
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvidenceFile {
    /// Items belonging to this file. Each carries its own
    /// `_category_key` (set at fetch time).
    pub evidence_items: Vec<EvidenceItem>,
}

/// Load + parse the merged blob at `path`.
pub fn load(path: &Path) -> Result<Vec<EvidenceItem>> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let parsed: EvidenceFile = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing {}", path.display()))?;
    Ok(parsed.evidence_items)
}

/// One discovered shard.
#[derive(Clone, Debug, Deserialize)]
struct TreeEntry {
    #[serde(default)]
    path: String,
    #[serde(default, rename = "type")]
    entry_type: String,
}

/// Walk the HF tree API to discover `1_evidence/*.json` filenames
/// for `category`. Cached at `cache_dir/<category>_filelist.json` so
/// reruns skip the API hop.
fn discover_files(cache_dir: &Path, category: &str) -> Result<Vec<String>> {
    let cache_path = cache_dir.join(format!("{category}_filelist.json"));
    if cache_path.is_file() {
        let bytes = fs::read(&cache_path)
            .with_context(|| format!("reading {}", cache_path.display()))?;
        let v: Vec<String> = serde_json::from_slice(&bytes)
            .with_context(|| format!("parsing {}", cache_path.display()))?;
        if !v.is_empty() {
            return Ok(v);
        }
    }
    let url = format!("{TREE_API}/{category}/1_evidence");
    let resp = ureq::get(&url).call().with_context(|| format!("GET {url}"))?;
    let mut body = String::new();
    resp.into_reader()
        .read_to_string(&mut body)
        .context("read tree body")?;
    let entries: Vec<TreeEntry> = serde_json::from_str(&body)
        .with_context(|| format!("parsing tree response for {category}"))?;
    let mut out = Vec::new();
    for e in entries {
        if e.entry_type == "file" && e.path.ends_with(".json") {
            // Path looks like `<...>/1_evidence/<file>`; we only need
            // the filename.
            if let Some(name) = e.path.rsplit('/').next() {
                out.push(name.to_string());
            }
        }
    }
    let bytes = serde_json::to_vec(&out).context("serialize filelist")?;
    fs::write(&cache_path, &bytes)
        .with_context(|| format!("writing {}", cache_path.display()))?;
    Ok(out)
}

/// Fetch ConvoMem shards for every headline category, merge into a
/// single canonical blob under `cache_dir`, return the path.
///
/// Discovery is dynamic (HF tree API) so we never have to ship a
/// stale manifest. Per-category cap defaults to
/// [`DEFAULT_PER_CATEGORY_CAP`]; override via the
/// `MNEM_BENCH_CONVOMEM_PER_CAT` env var.
pub fn fetch_into(cache_dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(cache_dir.join("shards"))
        .with_context(|| format!("mkdir {}", cache_dir.display()))?;
    let cap = std::env::var("MNEM_BENCH_CONVOMEM_PER_CAT")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_PER_CATEGORY_CAP);

    let mut merged = EvidenceFile { evidence_items: Vec::new() };
    for cat in HEADLINE_CATEGORIES {
        let files = discover_files(cache_dir, cat)
            .with_context(|| format!("discover {cat}"))?;
        if files.is_empty() {
            eprintln!("[convomem] no shards discovered for {cat}; skipping");
            continue;
        }
        let mut taken = 0usize;
        for fname in &files {
            if taken >= cap {
                break;
            }
            let shard_path = cache_dir.join("shards").join(cat).join(fname);
            if !shard_path.is_file() {
                fs::create_dir_all(shard_path.parent().unwrap()).ok();
                let url = format!("{HF_BASE}/{cat}/1_evidence/{fname}");
                let resp = match ureq::get(&url).call() {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("[convomem] GET {url}: {e}; skipping");
                        continue;
                    }
                };
                let mut body: Vec<u8> = Vec::new();
                if let Err(e) = resp.into_reader().read_to_end(&mut body) {
                    eprintln!("[convomem] read body for {fname}: {e}; skipping");
                    continue;
                }
                if let Err(e) = fs::write(&shard_path, &body) {
                    eprintln!("[convomem] write {}: {e}; skipping", shard_path.display());
                    continue;
                }
            }
            let bytes = match fs::read(&shard_path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("[convomem] reading {}: {e}", shard_path.display());
                    continue;
                }
            };
            let parsed: EvidenceFile = match serde_json::from_slice(&bytes) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("[convomem] parsing {}: {e}", shard_path.display());
                    continue;
                }
            };
            for mut it in parsed.evidence_items {
                it.category_key = (*cat).to_string();
                merged.evidence_items.push(it);
                taken += 1;
                if taken >= cap {
                    break;
                }
            }
        }
        eprintln!("[convomem] {cat}: {taken} items");
    }
    if merged.evidence_items.is_empty() {
        return Err(anyhow!(
            "convomem fetch yielded zero evidence_items; check network reachability for {HF_BASE}"
        ));
    }
    let dst = cache_dir.join("convomem_evidence.json");
    let bytes = serde_json::to_vec(&merged).context("serialize merged blob")?;
    fs::write(&dst, &bytes)
        .with_context(|| format!("writing {}", dst.display()))?;
    Ok(dst)
}

// Pull `Read` into scope for `into_reader().read_to_end/string`.
use std::io::Read;
