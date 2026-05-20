use super::*;

use mnem_core::prolly::Cursor;

pub(crate) fn run(override_path: Option<&Path>) -> Result<()> {
    let (_dir, r, bs, _ohs) = repo::open_all(override_path)?;
    let commit = r.head_commit();
    let idx = load_index_set(&bs, commit)?;

    let label_names: Vec<String> = idx
        .as_ref()
        .map(|i| i.nodes_by_label.keys().cloned().collect())
        .unwrap_or_default();
    let label_count = label_names.len();
    let edge_count = if let Some(commit) = commit {
        let cursor = Cursor::new(&*bs, &commit.edges)?;
        let mut count = 0_usize;
        for entry in cursor {
            let _ = entry?;
            count += 1;
        }
        count
    } else {
        0
    };
    // `refs=` preserves the pre-fix one-line output shape for tooling that
    // grepped the old `view().refs.len()` value; the new `edges=` slot is the
    // real Prolly edge count and is what consumers should read going forward.
    let ref_count = r.view().refs.len();
    let head_commit_str = r
        .view()
        .heads
        .first()
        .map_or_else(|| "<none>".into(), ToString::to_string);

    // audit-2026-04-25 P0-1 (partial): expose `content_cid` -- a CID
    // computed from only the data-DAG roots of the head commit
    // (nodes/edges/schema/indexes/parents). Two ingest runs against
    // byte-identical input agree on `content_cid` even when their
    // `commit_cid` differs (the latter still embeds wall-clock +
    // UUIDv7 metadata for audit trail). See Commit::content_cid.
    let content_cid_str = match commit {
        Some(c) => c
            .content_cid()
            .map(|cid| cid.to_string())
            .unwrap_or_else(|_| "<encode-error>".into()),
        None => "<none>".into(),
    };

    // One-line machine-friendly form, followed by a human summary that tells the
    // user whether the repo actually contains anything. Per-label node counts
    // require walking a Prolly subtree per label; skip here and surface in a
    // future `mnem stats --verbose` instead.
    println!(
        "op={} commit={} content={} refs={} edges={} labels={}",
        r.op_id(),
        head_commit_str,
        content_cid_str,
        ref_count,
        edge_count,
        label_count
    );
    if label_count == 0 {
        println!("  (no nodes yet - run `mnem add node --summary \"...\"` to start)");
    } else {
        let preview = label_names
            .iter()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let more = if label_count > 5 {
            format!(" (+{} more)", label_count - 5)
        } else {
            String::new()
        };
        println!("  labels: {preview}{more}");
    }
    Ok(())
}
