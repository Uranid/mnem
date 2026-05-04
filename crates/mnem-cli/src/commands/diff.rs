use super::*;
use mnem_core::prolly::{DiffEntry, diff as prolly_diff};
use mnem_core::store::Blockstore;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
  mnem diff <op-a-cid> <op-b-cid>
  # common flow: find two ops via `mnem log`, then diff:
  mnem log -n 2
  mnem diff <older-op> <newer-op>
")]
pub(crate) struct Args {
    pub op_a: String,
    pub op_b: String,
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

    println!("op_a {a_cid}");
    println!("op_b {b_cid}");
    println!();

    // Ref deltas.
    let mut added: Vec<&String> = Vec::new();
    let mut removed: Vec<&String> = Vec::new();
    let mut changed: Vec<&String> = Vec::new();
    for (name, target) in &view_b.refs {
        match view_a.refs.get(name) {
            None => added.push(name),
            Some(prev) if prev != target => changed.push(name),
            _ => {}
        }
    }
    for name in view_a.refs.keys() {
        if !view_b.refs.contains_key(name) {
            removed.push(name);
        }
    }
    println!(
        "ref deltas: +{} -{} ~{}",
        added.len(),
        removed.len(),
        changed.len()
    );
    for r in &added {
        println!("  +{r}");
    }
    for r in &removed {
        println!("  -{r}");
    }
    for r in &changed {
        println!("  ~{r}");
    }

    // Commit CIDs.
    let head_a = view_a.heads.first();
    let head_b = view_b.heads.first();
    println!();
    println!(
        "commit deltas: a={} -> b={}",
        head_a.map_or_else(|| "<none>".to_string(), ToString::to_string),
        head_b.map_or_else(|| "<none>".to_string(), ToString::to_string)
    );

    // Node and edge structural diffs (requires both views to have a head commit).
    if let (Some(ha), Some(hb)) = (head_a, head_b) {
        let commit_a: Commit = from_canonical_bytes(
            &bs.get(ha)?.ok_or_else(|| anyhow!("commit_a block missing"))?,
        )?;
        let commit_b: Commit = from_canonical_bytes(
            &bs.get(hb)?.ok_or_else(|| anyhow!("commit_b block missing"))?,
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

    Ok(())
}

fn tally(entries: &[DiffEntry]) -> (usize, usize, usize) {
    entries.iter().fold((0, 0, 0), |(a, r, c), e| match e {
        DiffEntry::Added { .. } => (a + 1, r, c),
        DiffEntry::Removed { .. } => (a, r + 1, c),
        DiffEntry::Changed { .. } => (a, r, c + 1),
    })
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
            DiffEntry::Added { key: test_key(0), value: raw_cid(0) },
            DiffEntry::Added { key: test_key(1), value: raw_cid(1) },
            DiffEntry::Removed { key: test_key(2), value: raw_cid(2) },
            DiffEntry::Changed { key: test_key(3), before: raw_cid(3), after: raw_cid(4) },
        ];
        assert_eq!(tally(&entries), (2, 1, 1));
    }

    #[test]
    fn tally_all_added() {
        let entries: Vec<_> = (0u8..5)
            .map(|i| DiffEntry::Added { key: test_key(i), value: raw_cid(i) })
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
        assert!(s.contains("\"Alice, software engineer\""), "should contain quoted summary");
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
        let node = Node::new(NodeId::from_bytes_raw([2u8; 16]), "Doc")
            .with_summary(long_summary);
        let cid = store_node(&bs, &node);

        let s = node_summary(&bs, &cid);

        assert!(s.contains("...\""), "long summaries must end with ...\"");
        // The preview is 60 chars + "..." inside quotes — total is well under 100.
        assert!(s.len() < 200, "output must not reproduce the full 100-char summary verbatim");
    }

    #[test]
    fn node_summary_exactly_60_chars_not_truncated() {
        let bs = MemoryBlockstore::new();
        let exactly_60 = "y".repeat(60);
        let node = Node::new(NodeId::from_bytes_raw([3u8; 16]), "Chunk")
            .with_summary(exactly_60.clone());
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

        assert!(s.starts_with("<cid:"), "missing block must return <cid:...> placeholder");
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

        assert!(s.starts_with("<cid:"), "missing block must return <cid:...> placeholder");
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
        let node_a = Node::new(NodeId::from_bytes_raw([0xAAu8; 16]), "Fact")
            .with_summary("initial node");
        let (node_a_bytes, node_a_cid) = hash_to_cid(&node_a).unwrap();
        bs.put_trusted(node_a_cid.clone(), node_a_bytes).unwrap();
        let key_a: ProllyKey = node_a.id.into();

        let root_a = build_tree(&bs, vec![(key_a, node_a_cid.clone())]).unwrap();

        // Build commit_b's node tree: same node plus a new one.
        let node_b = Node::new(NodeId::from_bytes_raw([0xBBu8; 16]), "Fact")
            .with_summary("added node");
        let (node_b_bytes, node_b_cid) = hash_to_cid(&node_b).unwrap();
        bs.put_trusted(node_b_cid.clone(), node_b_bytes).unwrap();
        let key_b: ProllyKey = node_b.id.into();

        let root_b =
            build_tree(&bs, vec![(key_a, node_a_cid.clone()), (key_b, node_b_cid.clone())])
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
