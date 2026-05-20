use super::*;
use mnem_core::prolly::{DiffEntry, diff as prolly_diff};
use mnem_core::store::Blockstore;
use serde::Serialize;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem diff <op-a-cid> <op-b-cid>
  # common flow: find two ops via `mnem log`, then diff:
  mnem log -n 2
  mnem diff <older-op> <newer-op>
  mnem diff <older-op> <newer-op> --json
")]
pub(crate) struct Args {
    pub op_a: String,
    pub op_b: String,
    /// Output the diff as structured JSON instead of human-readable text.
    #[arg(long)]
    pub json: bool,
}

// ---- JSON output structs ----

#[derive(Serialize)]
struct RefDeltaAdded {
    name: String,
    target: String,
}

#[derive(Serialize)]
struct RefDeltaRemoved {
    name: String,
    target: String,
}

#[derive(Serialize)]
struct RefDeltaChanged {
    name: String,
    from: String,
    to: String,
}

#[derive(Serialize)]
struct RefDeltas {
    added: Vec<RefDeltaAdded>,
    removed: Vec<RefDeltaRemoved>,
    changed: Vec<RefDeltaChanged>,
}

#[derive(Serialize)]
struct NodeBeforeState {
    ntype: String,
    summary: Option<String>,
}

#[derive(Serialize)]
struct EdgeBeforeState {
    label: String,
    src: String,
    dst: String,
}

#[derive(Serialize)]
struct NodeDelta {
    #[serde(rename = "type")]
    delta_type: String,
    id: String,
    ntype: String,
    summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    before: Option<NodeBeforeState>,
}

#[derive(Serialize)]
struct EdgeDelta {
    #[serde(rename = "type")]
    delta_type: String,
    label: String,
    src: String,
    dst: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    before: Option<EdgeBeforeState>,
}

#[derive(Serialize)]
struct DiffOutput {
    op_a: String,
    op_b: String,
    commit_a: Option<String>,
    commit_b: Option<String>,
    ref_deltas: RefDeltas,
    node_deltas: Vec<NodeDelta>,
    edge_deltas: Vec<EdgeDelta>,
}

pub(crate) fn run(override_path: Option<&Path>, args: Args) -> Result<()> {
    // audit-2026-04-25 R2 (Stage E re-fix): `mnem diff` decodes each
    // side as an Operation, so we MUST resolve to op-CIDs. The
    // generic `resolve_commitish` returns commit-CIDs, which then
    // failed to decode as Operations with the V4 error
    // `decode: Msg("missing field 'view'")`. The dedicated op-CID
    // resolver accepts HEAD (-> r.op_id()) and raw op CIDs; named
    // refs intentionally are not supported here because they target
    // commits, not ops.
    let (_dir, r, bs, _ohs) = repo::open_all(override_path)?;

    let a_cid = super::resolve_op_commitish(&r, &args.op_a)
        .with_context(|| format!("resolving op_a `{}`", args.op_a))?;
    let b_cid = super::resolve_op_commitish(&r, &args.op_b)
        .with_context(|| format!("resolving op_b `{}`", args.op_b))?;

    let op_a: Operation =
        from_canonical_bytes(&bs.get(&a_cid)?.ok_or_else(|| anyhow!("op_a missing"))?)?;
    let op_b: Operation =
        from_canonical_bytes(&bs.get(&b_cid)?.ok_or_else(|| anyhow!("op_b missing"))?)?;
    let view_a: mnem_core::objects::View = from_canonical_bytes(
        &bs.get(&op_a.view)?
            .ok_or_else(|| anyhow!("view_a missing"))?,
    )?;
    let view_b: mnem_core::objects::View = from_canonical_bytes(
        &bs.get(&op_b.view)?
            .ok_or_else(|| anyhow!("view_b missing"))?,
    )?;

    // Ref deltas.
    let mut added_names: Vec<&String> = Vec::new();
    let mut removed_names: Vec<&String> = Vec::new();
    let mut changed_names: Vec<&String> = Vec::new();
    for (name, target) in &view_b.refs {
        match view_a.refs.get(name) {
            None => added_names.push(name),
            Some(prev) if prev != target => changed_names.push(name),
            _ => {}
        }
    }
    for name in view_a.refs.keys() {
        if !view_b.refs.contains_key(name) {
            removed_names.push(name);
        }
    }

    // Commit CIDs.
    let head_a = view_a.heads.first();
    let head_b = view_b.heads.first();

    if args.json {
        // ---- JSON output path ----
        let ref_deltas = RefDeltas {
            added: added_names
                .iter()
                .map(|name| RefDeltaAdded {
                    name: (*name).clone(),
                    target: view_b
                        .refs
                        .get(*name)
                        .map(ref_target_str)
                        .unwrap_or_default(),
                })
                .collect(),
            removed: removed_names
                .iter()
                .map(|name| RefDeltaRemoved {
                    name: (*name).clone(),
                    target: view_a
                        .refs
                        .get(*name)
                        .map(ref_target_str)
                        .unwrap_or_default(),
                })
                .collect(),
            changed: changed_names
                .iter()
                .map(|name| RefDeltaChanged {
                    name: (*name).clone(),
                    from: view_a
                        .refs
                        .get(*name)
                        .map(ref_target_str)
                        .unwrap_or_default(),
                    to: view_b
                        .refs
                        .get(*name)
                        .map(ref_target_str)
                        .unwrap_or_default(),
                })
                .collect(),
        };

        let mut node_deltas: Vec<NodeDelta> = Vec::new();
        let mut edge_deltas: Vec<EdgeDelta> = Vec::new();

        if let (Some(ha), Some(hb)) = (head_a, head_b) {
            let commit_a: Commit = from_canonical_bytes(
                &bs.get(ha)?
                    .ok_or_else(|| anyhow!("commit_a block missing"))?,
            )?;
            let commit_b: Commit = from_canonical_bytes(
                &bs.get(hb)?
                    .ok_or_else(|| anyhow!("commit_b block missing"))?,
            )?;

            let node_changes = prolly_diff(&*bs, &commit_a.nodes, &commit_b.nodes)?;
            let edge_changes = prolly_diff(&*bs, &commit_a.edges, &commit_b.edges)?;

            for entry in &node_changes {
                if let Some(delta) = node_delta_json(&*bs, entry) {
                    node_deltas.push(delta);
                }
            }
            for entry in &edge_changes {
                if let Some(delta) = edge_delta_json(&*bs, entry) {
                    edge_deltas.push(delta);
                }
            }
        }

        let output = DiffOutput {
            op_a: a_cid.to_string(),
            op_b: b_cid.to_string(),
            commit_a: head_a.map(ToString::to_string),
            commit_b: head_b.map(ToString::to_string),
            ref_deltas,
            node_deltas,
            edge_deltas,
        };
        println!("{}", serde_json::to_string_pretty(&output)?);
    } else {
        // ---- Human-readable output path (unchanged) ----
        println!("op_a {a_cid}");
        println!("op_b {b_cid}");
        println!();

        println!(
            "ref deltas: +{} -{} ~{}",
            added_names.len(),
            removed_names.len(),
            changed_names.len()
        );
        for r in &added_names {
            println!("  +{r}");
        }
        for r in &removed_names {
            println!("  -{r}");
        }
        for r in &changed_names {
            println!("  ~{r}");
        }

        println!();
        println!(
            "commit deltas: a={} -> b={}",
            head_a.map_or_else(|| "<none>".to_string(), ToString::to_string),
            head_b.map_or_else(|| "<none>".to_string(), ToString::to_string)
        );

        if let (Some(ha), Some(hb)) = (head_a, head_b) {
            let commit_a: Commit = from_canonical_bytes(
                &bs.get(ha)?
                    .ok_or_else(|| anyhow!("commit_a block missing"))?,
            )?;
            let commit_b: Commit = from_canonical_bytes(
                &bs.get(hb)?
                    .ok_or_else(|| anyhow!("commit_b block missing"))?,
            )?;

            let node_changes = prolly_diff(&*bs, &commit_a.nodes, &commit_b.nodes)?;
            let edge_changes = prolly_diff(&*bs, &commit_a.edges, &commit_b.edges)?;

            let (na, nr, nc) = tally(&node_changes);
            println!();
            println!("node deltas: +{na} -{nr} ~{nc}");
            for entry in &node_changes {
                print_node_entry(&*bs, entry);
            }

            let (ea, er, ec) = tally(&edge_changes);
            println!();
            println!("edge deltas: +{ea} -{er} ~{ec}");
            for entry in &edge_changes {
                print_edge_entry(&*bs, entry);
            }
        }
    }

    Ok(())
}

fn ref_target_str(t: &mnem_core::objects::RefTarget) -> String {
    use mnem_core::objects::RefTarget;
    match t {
        RefTarget::Normal { target } => target.to_string(),
        RefTarget::Conflicted { adds, .. } => adds
            .first()
            .map_or_else(|| "<conflicted>".to_string(), ToString::to_string),
    }
}

fn tally(entries: &[DiffEntry]) -> (usize, usize, usize) {
    entries.iter().fold((0, 0, 0), |(a, r, c), e| match e {
        DiffEntry::Added { .. } => (a + 1, r, c),
        DiffEntry::Removed { .. } => (a, r + 1, c),
        DiffEntry::Changed { .. } => (a, r, c + 1),
    })
}

fn node_from_blockstore(bs: &dyn Blockstore, value_cid: &mnem_core::id::Cid) -> Option<Node> {
    let bytes = bs.get(value_cid).ok()??;
    from_canonical_bytes::<Node>(&bytes).ok()
}

fn edge_from_blockstore(bs: &dyn Blockstore, value_cid: &mnem_core::id::Cid) -> Option<Edge> {
    let bytes = bs.get(value_cid).ok()??;
    from_canonical_bytes::<Edge>(&bytes).ok()
}

fn node_delta_json(bs: &dyn Blockstore, entry: &DiffEntry) -> Option<NodeDelta> {
    match entry {
        DiffEntry::Added { value, .. } => {
            let node = node_from_blockstore(bs, value)?;
            Some(NodeDelta {
                delta_type: "added".to_string(),
                id: node.id.to_string(),
                ntype: node.ntype.clone(),
                summary: node.summary.clone(),
                before: None,
            })
        }
        DiffEntry::Removed { value, .. } => {
            let node = node_from_blockstore(bs, value)?;
            Some(NodeDelta {
                delta_type: "removed".to_string(),
                id: node.id.to_string(),
                ntype: node.ntype.clone(),
                summary: node.summary.clone(),
                before: None,
            })
        }
        DiffEntry::Changed { before, after, .. } => {
            let node_after = node_from_blockstore(bs, after)?;
            let before_state = node_from_blockstore(bs, before).map(|n| NodeBeforeState {
                ntype: n.ntype.clone(),
                summary: n.summary.clone(),
            });
            Some(NodeDelta {
                delta_type: "changed".to_string(),
                id: node_after.id.to_string(),
                ntype: node_after.ntype.clone(),
                summary: node_after.summary.clone(),
                before: before_state,
            })
        }
    }
}

fn edge_delta_json(bs: &dyn Blockstore, entry: &DiffEntry) -> Option<EdgeDelta> {
    match entry {
        DiffEntry::Added { value, .. } => {
            let edge = edge_from_blockstore(bs, value)?;
            Some(EdgeDelta {
                delta_type: "added".to_string(),
                label: edge.etype.clone(),
                src: edge.src.to_string(),
                dst: edge.dst.to_string(),
                before: None,
            })
        }
        DiffEntry::Removed { value, .. } => {
            let edge = edge_from_blockstore(bs, value)?;
            Some(EdgeDelta {
                delta_type: "removed".to_string(),
                label: edge.etype.clone(),
                src: edge.src.to_string(),
                dst: edge.dst.to_string(),
                before: None,
            })
        }
        DiffEntry::Changed { before, after, .. } => {
            let edge_after = edge_from_blockstore(bs, after)?;
            let before_state = edge_from_blockstore(bs, before).map(|e| EdgeBeforeState {
                label: e.etype.clone(),
                src: e.src.to_string(),
                dst: e.dst.to_string(),
            });
            Some(EdgeDelta {
                delta_type: "changed".to_string(),
                label: edge_after.etype.clone(),
                src: edge_after.src.to_string(),
                dst: edge_after.dst.to_string(),
                before: before_state,
            })
        }
    }
}

fn node_summary(bs: &dyn Blockstore, value_cid: &mnem_core::id::Cid) -> String {
    let Ok(Some(bytes)) = bs.get(value_cid) else {
        return format!("<cid:{value_cid}>");
    };
    let Ok(node) = from_canonical_bytes::<Node>(&bytes) else {
        return format!("<cid:{value_cid}>");
    };
    let summary_part = match &node.summary {
        Some(s) if !s.is_empty() => {
            let preview: String = s.chars().take(60).collect();
            if s.len() > 60 {
                format!(" \"{preview}...\"")
            } else {
                format!(" \"{preview}\"")
            }
        }
        _ => String::new(),
    };
    format!("{} [{}]{summary_part}", node.id, node.ntype)
}

fn edge_summary(bs: &dyn Blockstore, value_cid: &mnem_core::id::Cid) -> String {
    let Ok(Some(bytes)) = bs.get(value_cid) else {
        return format!("<cid:{value_cid}>");
    };
    let Ok(edge) = from_canonical_bytes::<Edge>(&bytes) else {
        return format!("<cid:{value_cid}>");
    };
    format!("{} -[{}]-> {}", edge.src, edge.etype, edge.dst)
}

fn print_node_entry(bs: &dyn Blockstore, entry: &DiffEntry) {
    match entry {
        DiffEntry::Added { value, .. } => println!("  + {}", node_summary(bs, value)),
        DiffEntry::Removed { value, .. } => println!("  - {}", node_summary(bs, value)),
        DiffEntry::Changed { before, after, .. } => {
            println!(
                "  ~ {} -> {}",
                node_summary(bs, before),
                node_summary(bs, after)
            );
        }
    }
}

fn print_edge_entry(bs: &dyn Blockstore, entry: &DiffEntry) {
    match entry {
        DiffEntry::Added { value, .. } => println!("  + {}", edge_summary(bs, value)),
        DiffEntry::Removed { value, .. } => println!("  - {}", edge_summary(bs, value)),
        DiffEntry::Changed { before, after, .. } => {
            println!(
                "  ~ {} -> {}",
                edge_summary(bs, before),
                edge_summary(bs, after)
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnem_core::codec::hash_to_cid;
    use mnem_core::id::{CODEC_RAW, Multihash};
    use mnem_core::prolly::constants::ProllyKey;
    use mnem_core::store::MemoryBlockstore;

    // ---- test helpers ----

    fn raw_cid(n: u8) -> mnem_core::id::Cid {
        mnem_core::id::Cid::new(CODEC_RAW, Multihash::sha2_256(&[n]))
    }

    fn test_key(n: u8) -> ProllyKey {
        let mut k = [0u8; 16];
        k[15] = n;
        ProllyKey(k)
    }

    fn store_node(bs: &MemoryBlockstore, node: &Node) -> mnem_core::id::Cid {
        let (bytes, cid) = hash_to_cid(node).unwrap();
        bs.put_trusted(cid.clone(), bytes).unwrap();
        cid
    }

    fn store_edge(bs: &MemoryBlockstore, edge: &Edge) -> mnem_core::id::Cid {
        let (bytes, cid) = hash_to_cid(edge).unwrap();
        bs.put_trusted(cid.clone(), bytes).unwrap();
        cid
    }

    // ---- tally ----

    #[test]
    fn tally_empty_slice() {
        assert_eq!(tally(&[]), (0, 0, 0));
    }

    #[test]
    fn tally_counts_each_variant() {
        let entries = vec![
            DiffEntry::Added {
                key: test_key(0),
                value: raw_cid(0),
            },
            DiffEntry::Added {
                key: test_key(1),
                value: raw_cid(1),
            },
            DiffEntry::Removed {
                key: test_key(2),
                value: raw_cid(2),
            },
            DiffEntry::Changed {
                key: test_key(3),
                before: raw_cid(3),
                after: raw_cid(4),
            },
        ];
        assert_eq!(tally(&entries), (2, 1, 1));
    }

    #[test]
    fn tally_all_added() {
        let entries: Vec<_> = (0u8..5)
            .map(|i| DiffEntry::Added {
                key: test_key(i),
                value: raw_cid(i),
            })
            .collect();
        assert_eq!(tally(&entries), (5, 0, 0));
    }

    // ---- node_summary ----

    #[test]
    fn node_summary_includes_id_ntype_and_summary() {
        let bs = MemoryBlockstore::new();
        let id = NodeId::from_bytes_raw([7u8; 16]);
        let node = Node::new(id, "Person").with_summary("Alice, software engineer");
        let cid = store_node(&bs, &node);

        let s = node_summary(&bs, &cid);

        assert!(s.contains(&id.to_string()), "should contain the node UUID");
        assert!(s.contains("[Person]"), "should contain ntype in brackets");
        assert!(
            s.contains("\"Alice, software engineer\""),
            "should contain quoted summary"
        );
    }

    #[test]
    fn node_summary_no_summary_omits_quotes() {
        let bs = MemoryBlockstore::new();
        let node = Node::new(NodeId::from_bytes_raw([1u8; 16]), "Thing");
        let cid = store_node(&bs, &node);

        let s = node_summary(&bs, &cid);

        assert!(s.contains("[Thing]"));
        assert!(!s.contains('"'), "no quotes when summary is absent");
    }

    #[test]
    fn node_summary_long_summary_truncated_with_ellipsis() {
        let bs = MemoryBlockstore::new();
        let long_summary = "x".repeat(100);
        let node = Node::new(NodeId::from_bytes_raw([2u8; 16]), "Doc").with_summary(long_summary);
        let cid = store_node(&bs, &node);

        let s = node_summary(&bs, &cid);

        assert!(s.contains("...\""), "long summaries must end with ...\"");
        // The preview is 60 chars + "..." inside quotes, total is well under 100.
        assert!(
            s.len() < 200,
            "output must not reproduce the full 100-char summary verbatim"
        );
    }

    #[test]
    fn node_summary_exactly_60_chars_not_truncated() {
        let bs = MemoryBlockstore::new();
        let exactly_60 = "y".repeat(60);
        let node =
            Node::new(NodeId::from_bytes_raw([3u8; 16]), "Chunk").with_summary(exactly_60.clone());
        let cid = store_node(&bs, &node);

        let s = node_summary(&bs, &cid);

        assert!(!s.contains("..."), "exactly-60 must not be truncated");
        assert!(s.contains(&exactly_60));
    }

    #[test]
    fn node_summary_missing_block_returns_cid_placeholder() {
        let bs = MemoryBlockstore::new();
        // Compute a valid CID but don't actually store the block.
        let (_, phantom_cid) =
            hash_to_cid(&Node::new(NodeId::from_bytes_raw([4u8; 16]), "Ghost")).unwrap();

        let s = node_summary(&bs, &phantom_cid);

        assert!(
            s.starts_with("<cid:"),
            "missing block must return <cid:...> placeholder"
        );
    }

    // ---- edge_summary ----

    #[test]
    fn edge_summary_formats_src_etype_dst() {
        let bs = MemoryBlockstore::new();
        let src = NodeId::from_bytes_raw([10u8; 16]);
        let dst = NodeId::from_bytes_raw([20u8; 16]);
        let edge = Edge::new(EdgeId::from_bytes_raw([1u8; 16]), "knows", src, dst);
        let cid = store_edge(&bs, &edge);

        let s = edge_summary(&bs, &cid);

        assert!(s.contains(&src.to_string()), "should contain src UUID");
        assert!(s.contains("-[knows]->"), "should contain edge type");
        assert!(s.contains(&dst.to_string()), "should contain dst UUID");
    }

    #[test]
    fn edge_summary_missing_block_returns_cid_placeholder() {
        let bs = MemoryBlockstore::new();
        let (_, phantom_cid) = hash_to_cid(&Edge::new(
            EdgeId::from_bytes_raw([5u8; 16]),
            "ghost",
            NodeId::from_bytes_raw([6u8; 16]),
            NodeId::from_bytes_raw([7u8; 16]),
        ))
        .unwrap();

        let s = edge_summary(&bs, &phantom_cid);

        assert!(
            s.starts_with("<cid:"),
            "missing block must return <cid:...> placeholder"
        );
    }

    // ---- edge_delta_json: Changed branch ----
    //
    // DiffEntry::Changed for edges is ARCHITECTURALLY UNREACHABLE through the
    // CLI because `add edge` always generates a fresh EdgeId (UUID-v7), so the
    // same prolly key can never appear in two different commits with two
    // different value CIDs.  The only way to exercise the Changed arm is to
    // call edge_delta_json directly with a synthetic DiffEntry::Changed.

    #[test]
    fn edge_delta_json_changed_entry_produces_changed_delta() {
        let bs = MemoryBlockstore::new();
        let src = NodeId::from_bytes_raw([0xA0u8; 16]);
        let dst = NodeId::from_bytes_raw([0xB0u8; 16]);

        // "Before" edge: label "old_label"
        let edge_before = Edge::new(EdgeId::from_bytes_raw([0x01u8; 16]), "old_label", src, dst);
        let cid_before = store_edge(&bs, &edge_before);

        // "After" edge: same logical edge but label changed to "new_label"
        let edge_after = Edge::new(EdgeId::from_bytes_raw([0x01u8; 16]), "new_label", src, dst);
        let cid_after = store_edge(&bs, &edge_after);

        let entry = DiffEntry::Changed {
            key: test_key(1),
            before: cid_before,
            after: cid_after,
        };

        let delta = edge_delta_json(&bs, &entry).expect("Changed edge must produce a delta");

        assert_eq!(delta.delta_type, "changed", "delta_type must be 'changed'");
        assert_eq!(
            delta.label, "new_label",
            "label must reflect the after state"
        );
        assert_eq!(delta.src, src.to_string(), "src must match");
        assert_eq!(delta.dst, dst.to_string(), "dst must match");

        let before_state = delta
            .before
            .expect("Changed delta must have a before state");
        assert_eq!(
            before_state.label, "old_label",
            "before.label must reflect the before state"
        );
        assert_eq!(before_state.src, src.to_string(), "before.src must match");
        assert_eq!(before_state.dst, dst.to_string(), "before.dst must match");
    }

    // ---- prolly_diff wiring smoke test ----
    //
    // Verifies that a node added between two commits appears as a single
    // Added entry when the trees are diffed. This exercises the wiring
    // in `run` without requiring a real repo on disk.

    #[test]
    fn added_node_shows_up_in_prolly_diff() {
        use mnem_core::prolly::{build_tree, diff as prolly_diff};

        let bs = MemoryBlockstore::new();

        // Build commit_a's node tree: one node.
        let node_a =
            Node::new(NodeId::from_bytes_raw([0xAAu8; 16]), "Fact").with_summary("initial node");
        let (node_a_bytes, node_a_cid) = hash_to_cid(&node_a).unwrap();
        bs.put_trusted(node_a_cid.clone(), node_a_bytes).unwrap();
        let key_a: ProllyKey = node_a.id.into();

        let root_a = build_tree(&bs, vec![(key_a, node_a_cid.clone())]).unwrap();

        // Build commit_b's node tree: same node plus a new one.
        let node_b =
            Node::new(NodeId::from_bytes_raw([0xBBu8; 16]), "Fact").with_summary("added node");
        let (node_b_bytes, node_b_cid) = hash_to_cid(&node_b).unwrap();
        bs.put_trusted(node_b_cid.clone(), node_b_bytes).unwrap();
        let key_b: ProllyKey = node_b.id.into();

        let root_b = build_tree(
            &bs,
            vec![(key_a, node_a_cid.clone()), (key_b, node_b_cid.clone())],
        )
        .unwrap();

        let changes = prolly_diff(&bs, &root_a, &root_b).unwrap();

        assert_eq!(changes.len(), 1, "exactly one delta: the added node");
        assert!(
            matches!(&changes[0], DiffEntry::Added { value, .. } if value == &node_b_cid),
            "the delta must be Added with node_b's CID"
        );

        // Verify node_summary resolves the added entry correctly.
        if let DiffEntry::Added { value, .. } = &changes[0] {
            let s = node_summary(&bs, value);
            assert!(s.contains("[Fact]"));
            assert!(s.contains("\"added node\""));
        }
    }
}
