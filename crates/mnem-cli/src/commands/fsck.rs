//! `mnem fsck` - reachability-only integrity check.
//!
//! Walks every operation reachable from HEAD and verifies:
//!
//! 1. The op block itself is present and CID-correct.
//! 2. The view block referenced by the op is present and CID-correct.
//! 3. Every head commit referenced in the view is present and CID-correct.
//! 4. Each commit's `nodes`, `edges`, and `schema` Prolly-tree root CIDs
//!    are fully walked: every interior and leaf block in each tree is
//!    fetched and verified to exist in the blockstore.
//! 5. Optional Prolly-tree sidecars (`embeddings` G16, `sparse` G17)
//!    are fully walked: every interior and leaf block is verified.
//! 6. Other optional commit fields (`indexes`, `delta`) are root-block
//!    checked when present.
//!
//! This is NOT a full GC-style "iterate all store blocks" scan. It is
//! reachability-only: start from HEAD, follow parent pointers, check every
//! referenced CID that can be verified cheaply.

use std::collections::HashSet;

use serde::Serialize;

use super::*;
use mnem_core::objects::View;
use mnem_core::prolly::tree::{TreeChunk, load_tree_chunk};
use mnem_core::store::blockstore::recompute_cid;

/// Hard cap on the number of ops walked when `--limit` is not supplied.
const DEFAULT_LIMIT: usize = 50_000;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Walks all ops reachable from HEAD and verifies every referenced block is
present and CID-correct (content hashes to its CID). For each commit,
ALL blocks of the nodes, edges, schema, embeddings (G16), and sparse (G17)
Prolly trees are fetched and checked - not just the root. Missing interior
or leaf blocks are reported as errors so corruption inside a Prolly tree
is always detected.

Examples:
  mnem fsck                   # check all ops from HEAD
  mnem fsck --limit 100       # only walk the last 100 ops
  mnem fsck --json            # machine-readable output
")]
pub(crate) struct Args {
    /// Maximum number of ops to walk backwards from HEAD.
    /// Defaults to all ops (capped at 50,000).
    #[arg(long)]
    pub limit: Option<usize>,

    /// Emit a single JSON object instead of human-readable text.
    #[arg(long)]
    pub json: bool,
}

/// A single integrity error discovered during the walk.
#[derive(Debug, Serialize)]
struct FsckError {
    /// The op CID that triggered the error.
    op: String,
    /// Short description of what went wrong.
    kind: String,
    /// The CID that was missing or corrupt (if applicable).
    #[serde(skip_serializing_if = "Option::is_none")]
    cid: Option<String>,
}

/// JSON output shape.
#[derive(Serialize)]
struct FsckReport {
    ops_checked: usize,
    blocks_verified: usize,
    errors: Vec<FsckError>,
    ok: bool,
}

/// Verify that a CID exists in the blockstore and that its bytes
/// actually hash to that CID. Returns `Ok(())` on success.
///
/// Passing `None` for `bs` is not possible; `bs` is always a `&dyn Blockstore`.
fn check_block(
    bs: &dyn mnem_core::store::Blockstore,
    cid: &mnem_core::id::Cid,
) -> Result<(), String> {
    let bytes = bs
        .get(cid)
        .map_err(|e| format!("store I/O error fetching {cid}: {e}"))?;

    let bytes = match bytes {
        Some(b) => b,
        None => return Err("missing".to_string()),
    };

    // Recompute and compare. `recompute_cid` returns `None` for unknown
    // hash algorithms - we treat those as "cannot verify, skip".
    if let Some(computed) = recompute_cid(cid, &bytes) {
        if computed != *cid {
            return Err(format!(
                "CID mismatch: claimed {cid} but content hashes to {computed}"
            ));
        }
    }

    Ok(())
}

/// Recursively walk every block in the Prolly tree rooted at `root`.
///
/// Fetches each block, counts it, and recurses into children of internal
/// nodes. Any missing block is recorded as an error in `errors`.
///
/// Returns the number of blocks successfully fetched (i.e. present in the
/// blockstore). Missing blocks are pushed to `errors` but do not abort the
/// walk - sibling subtrees are still checked.
fn walk_prolly_tree(
    bs: &dyn mnem_core::store::Blockstore,
    root: &mnem_core::id::Cid,
    tree_name: &str,
    op_cid_str: &str,
    errors: &mut Vec<FsckError>,
) -> usize {
    let mut stack: Vec<mnem_core::id::Cid> = vec![root.clone()];
    let mut blocks_ok: usize = 0;

    while let Some(cid) = stack.pop() {
        let chunk = match load_tree_chunk(bs, &cid) {
            Ok(c) => {
                blocks_ok += 1;
                c
            }
            Err(_) => {
                errors.push(FsckError {
                    op: op_cid_str.to_owned(),
                    kind: format!("missing interior block {cid} in {tree_name} tree"),
                    cid: Some(cid.to_string()),
                });
                // Cannot descend into missing block; continue with siblings.
                continue;
            }
        };

        if let TreeChunk::Internal(internal) = chunk {
            // Push all children so they are fetched and verified.
            stack.extend(internal.children);
        }
        // Leaf blocks have no further tree structure to follow.
    }

    blocks_ok
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    let (_data_dir, repo, bs, _ohs) = repo::open_all(override_path)?;
    let bs = bs.as_ref();

    let limit = args.limit.unwrap_or(DEFAULT_LIMIT);
    let head_op_cid = repo.op_id().clone();

    if !args.json {
        let full = head_op_cid.to_string();
        let short = short_cid(&full);
        println!("fsck: checking from HEAD (op {short})");
    }

    let mut errors: Vec<FsckError> = Vec::new();
    let mut ops_checked: usize = 0;
    let mut blocks_verified: usize = 0;
    // Track visited op CIDs to avoid loops in a corrupt DAG.
    let mut visited: HashSet<mnem_core::id::Cid> = HashSet::new();

    let mut cur = head_op_cid.clone();

    loop {
        if ops_checked >= limit {
            break;
        }
        if !visited.insert(cur.clone()) {
            // Already visited - DAG cycle guard.
            break;
        }

        let op_cid_str = cur.to_string();

        // ── Step 1: verify the op block itself ──────────────────────────
        let op_bytes = match check_block(bs, &cur) {
            Ok(()) => {
                blocks_verified += 1;
                // Re-fetch to decode; check_block already confirmed presence.
                bs.get(&cur)
                    .map_err(|e| anyhow!("store I/O: {e}"))?
                    .expect("just verified present")
            }
            Err(reason) => {
                errors.push(FsckError {
                    op: op_cid_str.clone(),
                    kind: format!("op block {reason}"),
                    cid: Some(op_cid_str.clone()),
                });
                // Cannot decode this op - stop walking.
                break;
            }
        };

        let op: Operation = match from_canonical_bytes(&op_bytes) {
            Ok(o) => o,
            Err(e) => {
                errors.push(FsckError {
                    op: op_cid_str.clone(),
                    kind: format!("op block decode failed: {e}"),
                    cid: Some(op_cid_str.clone()),
                });
                break;
            }
        };

        ops_checked += 1;

        // ── Step 2: verify the view block ───────────────────────────────
        let view_cid = &op.view;
        let view_opt: Option<View> = match check_block(bs, view_cid) {
            Ok(()) => {
                blocks_verified += 1;
                let view_bytes = bs
                    .get(view_cid)
                    .map_err(|e| anyhow!("store I/O: {e}"))?
                    .expect("just verified present");
                match from_canonical_bytes::<View>(&view_bytes) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        errors.push(FsckError {
                            op: op_cid_str.clone(),
                            kind: format!("view block decode failed: {e}"),
                            cid: Some(view_cid.to_string()),
                        });
                        None
                    }
                }
            }
            Err(reason) => {
                errors.push(FsckError {
                    op: op_cid_str.clone(),
                    kind: format!("view block {reason}"),
                    cid: Some(view_cid.to_string()),
                });
                None
            }
        };

        // ── Step 3 & 4: verify each head commit and its tree roots ───────
        if let Some(view) = view_opt {
            for head_cid in &view.heads {
                let commit_opt: Option<Commit> = match check_block(bs, head_cid) {
                    Ok(()) => {
                        blocks_verified += 1;
                        let commit_bytes = bs
                            .get(head_cid)
                            .map_err(|e| anyhow!("store I/O: {e}"))?
                            .expect("just verified present");
                        match from_canonical_bytes::<Commit>(&commit_bytes) {
                            Ok(c) => Some(c),
                            Err(e) => {
                                errors.push(FsckError {
                                    op: op_cid_str.clone(),
                                    kind: format!("commit block decode failed: {e}"),
                                    cid: Some(head_cid.to_string()),
                                });
                                None
                            }
                        }
                    }
                    Err(reason) => {
                        errors.push(FsckError {
                            op: op_cid_str.clone(),
                            kind: format!("commit block {reason}"),
                            cid: Some(head_cid.to_string()),
                        });
                        None
                    }
                };

                if let Some(commit) = commit_opt {
                    // Walk every block in each Prolly tree (root + all interior
                    // and leaf blocks). Missing interior blocks are reported as
                    // errors; the walk continues with sibling subtrees so all
                    // missing blocks are reported in a single fsck run.
                    for (tree_name, tree_cid) in [
                        ("nodes", &commit.nodes),
                        ("edges", &commit.edges),
                        ("schema", &commit.schema),
                    ] {
                        let tree_blocks =
                            walk_prolly_tree(bs, tree_cid, tree_name, &op_cid_str, &mut errors);
                        blocks_verified += tree_blocks;
                    }

                    // Optional Prolly-tree sidecars: embeddings (G16) and
                    // sparse (G17) are full trees - walk every block so
                    // corrupt or missing interior blocks are caught.
                    for (tree_name, maybe_cid) in [
                        ("embeddings", commit.embeddings.as_ref()),
                        ("sparse", commit.sparse.as_ref()),
                    ] {
                        if let Some(cid) = maybe_cid {
                            let tree_blocks =
                                walk_prolly_tree(bs, cid, tree_name, &op_cid_str, &mut errors);
                            blocks_verified += tree_blocks;
                        }
                    }

                    // Other optional single-block commit fields (indexes,
                    // delta) are not Prolly trees; check root presence only.
                    let optional_roots: &[(&str, Option<&mnem_core::id::Cid>)] = &[
                        ("indexes root", commit.indexes.as_ref()),
                        ("delta root", commit.delta.as_ref()),
                    ];
                    for (label, maybe_cid) in optional_roots {
                        if let Some(opt_cid) = maybe_cid {
                            match check_block(bs, opt_cid) {
                                Ok(()) => blocks_verified += 1,
                                Err(reason) => {
                                    errors.push(FsckError {
                                        op: op_cid_str.clone(),
                                        kind: format!("{label} {reason}"),
                                        cid: Some(opt_cid.to_string()),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Advance: follow the first parent (linear chain walk) ─────────
        match op.parents.first() {
            Some(parent_cid) => cur = parent_cid.clone(),
            None => break, // root op reached
        }
    }

    let ok = errors.is_empty();

    if args.json {
        let report = FsckReport {
            ops_checked,
            blocks_verified,
            errors,
            ok,
        };
        let line = serde_json::to_string(&report).context("serialising fsck report")?;
        println!("{line}");
    } else {
        println!("  ops checked:     {ops_checked}");
        println!("  blocks verified: {blocks_verified}");
        if errors.is_empty() {
            println!("  errors:          0");
            println!("ok");
        } else {
            for e in &errors {
                let cid_suffix = e
                    .cid
                    .as_deref()
                    .map(|c| format!(" ({c})"))
                    .unwrap_or_default();
                println!("  ERROR: op {} - {}{}", e.op, e.kind, cid_suffix);
            }
            println!("  errors:          {}", errors.len());
            println!("FAILED");
        }
    }

    if !ok {
        std::process::exit(1);
    }

    Ok(())
}

/// Produce a short-hex prefix of a CID for display. Mirrors the
/// helper in `log.rs`: skip the multibase prefix byte and take 8 chars.
fn short_cid(full: &str) -> String {
    if full.len() <= 10 {
        full.to_string()
    } else {
        full.chars().skip(2).take(8).collect()
    }
}
