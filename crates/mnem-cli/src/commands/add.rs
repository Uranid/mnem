use super::*;

#[derive(clap::Subcommand, Debug)]
pub(crate) enum AddCmd {
    /// Add a node and commit it.
    Node(NodeArgs),
    /// Add an edge between two nodes (by UUID) and commit it.
    Edge(EdgeArgs),
}

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
 mnem add node -s \"Alice lives in Berlin\"
 mnem add node --label Person --prop name=Alice --prop city=Berlin \\
-s \"Alice is a climber\"
 echo 'full text content here' | mnem add node -s \"my notes\" --content @-
")]
pub(crate) struct NodeArgs {
    /// audit-2026-04-25 P3-6: `mnem add node Person` (positional
    /// label) previously failed with clap's generic "unexpected
    /// argument" error. Accept a hidden positional so we can surface
    /// an explicit, actionable hint pointing at `--label <LABEL>`.
    #[arg(hide = true)]
    pub positional: Option<String>,
    /// Node type (filterable via `mnem retrieve --label X` or the
    /// `label` Query predicate). See `docs/guide/ntype-vocab.md`
    /// for the recommended vocabulary.
    #[arg(long, alias = "ntype")]
    pub label: Option<String>,
    /// Short LLM-facing summary. Indexed by `mnem retrieve` via
    /// the dense embedder.
    #[arg(long, short = 's')]
    pub summary: Option<String>,
    /// Property: repeatable. `--prop name=Alice --prop age=30`.
    /// Values parse as JSON when possible, else as strings.
    #[arg(long = "prop")]
    pub props: Vec<String>,
    /// Opaque content body (UTF-8). If set to `@-`, read from stdin.
    #[arg(long)]
    pub content: Option<String>,
    /// Skip the embedder for this node even if one is configured.
    /// Useful for bulk imports where you'll `mnem embed` later.
    #[arg(long)]
    pub no_embed: bool,
    /// audit-2026-04-25 P0-1: caller-supplied node UUID. When
    /// present, the new node's NodeId is set from this string instead
    /// of being freshly generated as a UUIDv7. Lets distributed
    /// agents + replay pipelines pin node identity so two machines
    /// ingesting the same logical event produce the same Node CID
    /// (and therefore the same content_cid). Must parse as a UUID
    /// (any version) accepted by NodeId::parse_uuid. Mirrors the HTTP
    /// `POST /v1/nodes` `id` field.
    #[arg(long = "id", value_name = "UUID")]
    pub id: Option<String>,
    /// audit-2026-04-25 C3-2 (Cycle-3, partial): derive the node
    /// UUID deterministically from `(label, sorted props)` via
    /// blake3 truncation instead of generating a fresh UUIDv7.
    /// Two fresh sandboxes that pass the same `--label` and
    /// `--prop` set produce identical node CIDs (and therefore
    /// identical content_cids), which is the property required by
    /// distributed-replay and content-addressable archive flows.
    /// The legacy random-UUID path remains the default to avoid
    /// breaking callers that rely on time-ordering; a default flip
    /// is tracked for v0.5. Conflicts with `--id`.
    #[arg(long = "deterministic", conflicts_with = "id")]
    pub deterministic: bool,
    /// Resolve-or-create mode: anchor the node on this property
    /// instead of always creating a new one. Format: `key=value`.
    /// If a node with `(label, key=value)` already exists in the
    /// graph its UUID is reused; otherwise a new node is created
    /// with that property. Additional `--prop` flags set extra
    /// properties on the resolved or newly-created node.
    /// Conflicts with `--id` and `--deterministic`.
    #[arg(
        long = "canonical",
        value_name = "KEY=VALUE",
        conflicts_with_all = &["id", "deterministic"]
    )]
    pub canonical: Option<String>,
    /// Also resolve-or-create the entity in the global knowledge graph
    /// (~/.mnemglobal/.mnem/). Only valid when --canonical is also provided.
    #[arg(long, requires = "canonical")]
    pub global: bool,
    /// Commit message.
    #[arg(long, short = 'm', default_value = "mnem add node")]
    pub message: String,
}

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:
 mnem add edge --from <src-uuid> --to <dst-uuid> --label knows
 mnem add edge --from 019... --to 018... --label authored_by \\
--prop confidence=0.9
")]
pub(crate) struct EdgeArgs {
    #[arg(long = "from")]
    pub src: String,
    #[arg(long = "to")]
    pub dst: String,
    #[arg(long)]
    pub label: String,
    #[arg(long = "prop")]
    pub props: Vec<String>,
    #[arg(long, short = 'm', default_value = "mnem add edge")]
    pub message: String,
}

pub(crate) fn run(override_path: Option<&Path>, cmd: AddCmd) -> Result<()> {
    match cmd {
        AddCmd::Node(a) => add_node(override_path, a),
        AddCmd::Edge(a) => add_edge(override_path, a),
    }
}

fn add_node(override_path: Option<&Path>, a: NodeArgs) -> Result<()> {
    if let Some(p) = &a.positional {
        anyhow::bail!(
            "positional argument `{p}` is not supported by `mnem add node`\n\
 hint: use `--label {p}` (or drop it entirely to fall back to Node::DEFAULT_NTYPE)"
        );
    }
    let data_dir = repo::locate_data_dir(override_path)?;
    let cfg = config::load(&data_dir)?;
    let r = repo::open_repo(Some(data_dir.as_path()))?;

    // --canonical KEY=VALUE: resolve-or-create path.
    //
    // Find an existing node with (label, key=value); if absent create
    // it. Then apply any additional --prop flags and --summary on top,
    // mirroring the MCP `mnem_resolve_or_create` tool so the CLI has
    // full feature parity.
    if let Some(ref canonical_arg) = a.canonical {
        let (prop_name, anchor_value) = parse_prop(canonical_arg).with_context(|| {
            format!("--canonical expects KEY=VALUE format, got `{canonical_arg}`")
        })?;
        let label = a
            .label
            .as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or(Node::DEFAULT_NTYPE);

        let mut tx = r.start_transaction();
        // Resolve or create the anchor node.
        let node_id = tx.resolve_or_create_node(label, &prop_name, anchor_value.clone())?;

        // Build the full node: start from any existing props (so we
        // don't lose them), then layer the anchor prop + extra --prop +
        // --summary on top.  New values win on key conflict; existing
        // props that are not mentioned in this call are preserved.
        //
        // Clone prop_name and anchor_value before moving them into the
        // node builder; we need them again below if --global is set.
        let prop_name_for_global = prop_name.clone();
        let anchor_value_for_global = anchor_value.clone();

        // Load the committed base for this node (if it already existed).
        // `resolve_or_create_node` may have just created a brand-new
        // node (in which case lookup returns None) or found an existing
        // one.  Either way we start building from a clean slate and
        // then re-apply the existing props before layering the new ones.
        let mut node = match tx.base().lookup_node(&node_id)? {
            Some(existing) => existing,
            None => Node::new(node_id, label),
        };
        // Ensure the ntype is (re-)set to the caller's label in case the
        // existing node had a different label somehow (shouldn't happen,
        // but defensive).
        node.ntype = label.to_string();
        // Layer: anchor prop (always wins).
        node = node.with_prop(prop_name, anchor_value);
        if let Some(s) = &a.summary {
            node = node.with_summary(s);
        }
        for p in &a.props {
            let (k, v) = parse_prop(p)?;
            node = node.with_prop(k, v);
        }
        if let Some(c) = a.content {
            let data = if c == "@-" {
                use std::io::Read;
                let mut buf = String::new();
                std::io::stdin().read_to_string(&mut buf)?;
                buf
            } else {
                c
            };
            node = node.with_content(bytes::Bytes::from(data.into_bytes()));
        }

        // Embed (same warn-but-commit semantics as the plain path).
        let mut pending_embed: Option<(String, mnem_core::objects::node::Embedding)> = None;
        let mut embedded_dim: Option<usize> = None;
        let mut embedded_model: Option<String> = None;
        if !a.no_embed
            && let Some(pc) = config::resolve_embedder(&cfg)
            && let Some(text) = embed_text_of(&node)
        {
            match mnem_embed_providers::open(&pc) {
                Ok(embedder) => match embedder.embed(&text) {
                    Ok(v) => {
                        let model = embedder.model().to_string();
                        let emb = mnem_embed_providers::to_embedding(&model, &v);
                        embedded_dim = Some(v.len());
                        embedded_model = Some(model.clone());
                        pending_embed = Some((model, emb));
                    }
                    Err(e) => {
                        eprintln!("{}", format_embed_failure(&e, &pc, "embedding"));
                        eprintln!(
                            " note: [embed] unreachable; node added without dense_embed. \
 Run `mnem reindex` later to backfill, or use --no-embed to silence."
                        );
                    }
                },
                Err(e) => {
                    eprintln!("{}", format_embed_failure(&e, &pc, "embedding"));
                    eprintln!(
                        " note: [embed] unreachable; node added without dense_embed. \
 Run `mnem reindex` later to backfill, or use --no-embed to silence."
                    );
                }
            }
        }

        let node_cid = tx.add_node(&node)?;
        if let Some((model, emb)) = pending_embed {
            tx.set_embedding(node_cid, model, emb)?;
        }
        let new_r = tx.commit(&config::author_string(&cfg), &a.message)?;
        println!("resolved node {}", node_id.to_uuid_string());
        if let (Some(dim), Some(model)) = (embedded_dim, embedded_model.as_ref()) {
            println!(" embedded (dim={dim}) via {model}");
        }
        println!(" op_id {}", new_r.op_id());

        // --global: also resolve-or-create the same entity in
        // ~/.mnemglobal/.mnem/. Mirrors the `global: true` parameter of
        // the MCP `mnem_resolve_or_create` tool. Best-effort: if the
        // global graph is not initialised (or errors), we warn to stderr
        // and exit 0 -- the local operation already succeeded.
        if a.global {
            let global_dir = crate::global::default_dir();
            let global_data_dir = global_dir.join(crate::repo::MNEM_DIR);
            if !global_data_dir.is_dir() {
                eprintln!(
                    " note: global graph not found at {}; skipping global stamp. \
                     Run `mnem integrate` to create it.",
                    global_data_dir.display()
                );
            } else {
                match repo::open_repo(Some(&global_dir)) {
                    Err(e) => {
                        eprintln!(" note: could not open global graph: {e}; skipping global stamp");
                    }
                    Ok(global_r) => {
                        let mut global_tx = global_r.start_transaction();
                        match global_tx.resolve_or_create_node(
                            label,
                            &prop_name_for_global,
                            anchor_value_for_global,
                        ) {
                            Err(e) => {
                                eprintln!(
                                    " note: global resolve_or_create failed: {e}; \
                                     skipping global stamp"
                                );
                            }
                            Ok(global_id) => {
                                match global_tx.commit(
                                    &config::author_string(&cfg),
                                    "mnem add node --canonical (global stamp)",
                                ) {
                                    Err(e) => {
                                        eprintln!(
                                            " note: global commit failed: {e}; \
                                             global stamp not written"
                                        );
                                    }
                                    Ok(_) => {
                                        println!("global node {}", global_id.to_uuid_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return Ok(());
    }

    // Plain add-node path (no --canonical): always create a new node.

    // audit-2026-04-25 P0-1: honour `--id` so callers can pin node
    // identity for deterministic content_cid. Fresh UUIDv7 otherwise.
    //
    // audit-2026-04-25 C3-2 (Cycle-3, partial): when `--deterministic`
    // is passed, derive the UUID from a blake3 hash of
    // `(label, sorted_props_canonical)` so two fresh sandboxes
    // produce byte-identical NodeIds for the same logical input.
    // This is opt-in for a future release; the default flip to deterministic
    // IDs is tracked separately (CHANGELOG entry).
    let node_id = match (a.id.as_deref(), a.deterministic) {
        (Some(s), _) => {
            NodeId::parse_uuid(s).map_err(|e| anyhow::anyhow!("invalid --id `{s}`: {e}"))?
        }
        (None, true) => derive_deterministic_node_id(&a)?,
        (None, false) => NodeId::new_v7(),
    };
    let mut node = match &a.label {
        Some(l) if !l.is_empty() => Node::new(node_id, l),
        _ => Node::new_default(node_id),
    };
    if let Some(s) = &a.summary {
        node = node.with_summary(s);
    }
    for p in &a.props {
        let (k, v) = parse_prop(p)?;
        node = node.with_prop(k, v);
    }
    if let Some(c) = a.content {
        let data = if c == "@-" {
            use std::io::Read;
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            buf
        } else {
            c
        };
        node = node.with_content(bytes::Bytes::from(data.into_bytes()));
    }

    // Auto-embed if a provider is configured and the node has text
    // worth embedding. Provider failures are warned, not fatal:
    // commits are user-authoritative and never blocked on provider
    // uptime. `mnem embed` later backfills anything missed.
    // audit-2026-04-25 C7-4: surface a positive confirmation when the
    // dense embed actually lands on the node. The earlier silent path
    // ("just commit; no output") was a UX trap because operators
    // could not tell whether `mnem retrieve "..."` would have a
    // vector to match against. Print `embedded (dim=N) via <model>`
    // on success; warn + commit-without-vector on failure (Ollama
    // unreachable etc.). The unreachable warning text now points
    // at `mnem reindex` (C7-5) as the explicit recovery step.
    // defer embedding to a sidecar attachment via
    // `Transaction::set_embedding`. The Node body itself is no longer
    // mutated with the vector; we compute the dense embedding (if any)
    // up front so we can keep the existing UX (warn-but-commit on
    // provider failure), then attach it after `add_node` returns the
    // freshly-hashed CID. The legacy `Node::with_embed` mutation is
    // gone from this write path; removes the field entirely.
    let mut pending_embed: Option<(String, mnem_core::objects::node::Embedding)> = None;
    let mut embedded_dim: Option<usize> = None;
    let mut embedded_model: Option<String> = None;
    if !a.no_embed
        && let Some(pc) = config::resolve_embedder(&cfg)
        && let Some(text) = embed_text_of(&node)
    {
        match mnem_embed_providers::open(&pc) {
            Ok(embedder) => match embedder.embed(&text) {
                Ok(v) => {
                    let model = embedder.model().to_string();
                    let emb = mnem_embed_providers::to_embedding(&model, &v);
                    embedded_dim = Some(v.len());
                    embedded_model = Some(model.clone());
                    pending_embed = Some((model, emb));
                }
                Err(e) => {
                    eprintln!("{}", format_embed_failure(&e, &pc, "embedding"));
                    eprintln!(
                        " note: [embed] unreachable; node added without dense_embed. \
 Run `mnem reindex` later to backfill, or use --no-embed to silence."
                    );
                }
            },
            Err(e) => {
                eprintln!("{}", format_embed_failure(&e, &pc, "embedding"));
                eprintln!(
                    " note: [embed] unreachable; node added without dense_embed. \
 Run `mnem reindex` later to backfill, or use --no-embed to silence."
                );
            }
        }
    }

    let mut tx = r.start_transaction();
    let node_cid = tx.add_node(&node)?;
    if let Some((model, emb)) = pending_embed {
        tx.set_embedding(node_cid, model, emb)?;
    }
    let new_r = tx.commit(&config::author_string(&cfg), &a.message)?;
    println!("added node {}", node.id.to_uuid_string());
    if let (Some(dim), Some(model)) = (embedded_dim, embedded_model.as_ref()) {
        println!(" embedded (dim={dim}) via {model}");
    }
    println!(" op_id {}", new_r.op_id());
    Ok(())
}

fn add_edge(override_path: Option<&Path>, a: EdgeArgs) -> Result<()> {
    let data_dir = repo::locate_data_dir(override_path)?;
    let cfg = config::load(&data_dir)?;
    let r = repo::open_repo(Some(data_dir.as_path()))?;

    let src = NodeId::parse_uuid(&a.src).context("parsing --from")?;
    let dst = NodeId::parse_uuid(&a.dst).context("parsing --to")?;

    // C8: validate both endpoints exist before writing, so we never
    // commit a dangling edge that points at a non-existent node.
    if r.lookup_node(&src)?.is_none() {
        anyhow::bail!("no node with id={src} (--from)");
    }
    if r.lookup_node(&dst)?.is_none() {
        anyhow::bail!("no node with id={dst} (--to)");
    }

    let mut edge = Edge::new(EdgeId::new_v7(), &a.label, src, dst);
    for p in &a.props {
        let (k, v) = parse_prop(p)?;
        edge = edge.with_prop(k, v);
    }

    let mut tx = r.start_transaction();
    tx.add_edge(&edge)?;
    let new_r = tx.commit(&config::author_string(&cfg), &a.message)?;
    println!("added edge {}", edge.id.to_uuid_string());
    println!(" op_id {}", new_r.op_id());
    Ok(())
}

/// audit-2026-04-25 C3-2 (Cycle-3, partial): derive a stable
/// `NodeId` from `(label, sorted props)` via blake3 truncation.
///
/// Two callers passing the same `--label` and the same
/// `--prop K=V` set produce byte-identical NodeIds (and therefore
/// the same content_cid once the node is committed). The hash
/// input is:
///
/// ```text
/// "mnem-c3-2:node:v1\0" || label || "\0" ||
/// for (k, v) in sort_by_key(props):
/// k || "=" || dag-cbor(v) || "\0"
/// ```
///
/// The `mnem-c3-2:node:v1` prefix domain-separates this hash from
/// any other blake3 use in mnem (multihash on object bytes,
/// prolly chunker rolling hash, etc.). The literal `v1` lets us
/// version the derivation if the prop-canonicalisation rule
/// changes; existing IDs computed under v1 stay valid because the
/// caller has already pinned them in the op log.
///
/// `--summary` and `--content` are intentionally NOT folded into
/// the hash: those are mutable narration fields, while the
/// `(label, props)` pair is the identity contract. Including them
/// would defeat the dedup goal that the verification target
/// (`Two fresh sandboxes ... identical content_cid`) is checking.
fn derive_deterministic_node_id(a: &NodeArgs) -> Result<NodeId> {
    use mnem_core::codec::to_canonical_bytes;
    use mnem_core::id::Multihash;

    let label = a.label.as_deref().unwrap_or(Node::DEFAULT_NTYPE);

    // Parse + sort props by key so input order does not affect
    // the derived ID. Duplicate keys: keep the last occurrence,
    // matching the `with_prop` overwrite semantics applied below.
    let mut kv: std::collections::BTreeMap<String, ipld_core::ipld::Ipld> =
        std::collections::BTreeMap::new();
    for raw in &a.props {
        let (k, v) = parse_prop(raw)?;
        kv.insert(k, v);
    }

    let mut buf: Vec<u8> = Vec::with_capacity(64 + label.len() + 16 * kv.len());
    buf.extend_from_slice(b"mnem-c3-2:node:v1\0");
    buf.extend_from_slice(label.as_bytes());
    buf.push(0);
    for (k, v) in &kv {
        buf.extend_from_slice(k.as_bytes());
        buf.push(b'=');
        let cbor = to_canonical_bytes(v).context("canonicalising prop value for det-id")?;
        buf.extend_from_slice(&cbor);
        buf.push(0);
    }

    // blake3 -> 32 bytes; UUIDs are 16 bytes. Take the first 16
    // bytes of the digest; this is the same truncation pattern
    // used by content-addressing tools that need a UUID-shaped
    // ID from a longer hash. Collision probability is 2^-64 over
    // a population of N nodes (N ~= 2^32 for a million-node graph);
    // safe for the foreseeable graph sizes mnem targets.
    let mh = Multihash::blake3_256(&buf);
    let digest = mh.digest();
    let mut bytes16 = [0u8; 16];
    bytes16.copy_from_slice(&digest[..16]);
    Ok(NodeId::from_random_bytes(bytes16))
}

#[cfg(test)]
mod c3_2_deterministic_node_id_tests {
    use super::*;

    fn args(label: Option<&str>, props: &[&str]) -> NodeArgs {
        NodeArgs {
            positional: None,
            label: label.map(String::from),
            summary: None,
            props: props.iter().map(|s| (*s).to_string()).collect(),
            content: None,
            no_embed: true,
            id: None,
            deterministic: true,
            canonical: None,
            global: false,
            message: "t".into(),
        }
    }

    // C3-2 verification: two fresh sandboxes, same inputs ->
    // identical node UUID. Captured here as a unit test so the
    // contract is locked in CI.
    #[test]
    fn same_label_and_props_yield_same_id() {
        let a1 = args(Some("Person"), &["name=Alice", "city=Berlin"]);
        let a2 = args(Some("Person"), &["name=Alice", "city=Berlin"]);
        let id1 = derive_deterministic_node_id(&a1).expect("derive 1");
        let id2 = derive_deterministic_node_id(&a2).expect("derive 2");
        assert_eq!(id1, id2);
    }

    // Prop-order independence: Pass-2 found callers re-asserting
    // the same fact in different prop order. The derivation must
    // sort props before hashing.
    #[test]
    fn prop_order_does_not_matter() {
        let a1 = args(Some("Person"), &["name=Alice", "city=Berlin"]);
        let a2 = args(Some("Person"), &["city=Berlin", "name=Alice"]);
        let id1 = derive_deterministic_node_id(&a1).expect("derive 1");
        let id2 = derive_deterministic_node_id(&a2).expect("derive 2");
        assert_eq!(id1, id2);
    }

    #[test]
    fn different_labels_yield_different_ids() {
        let a1 = args(Some("Person"), &["name=Alice"]);
        let a2 = args(Some("Org"), &["name=Alice"]);
        let id1 = derive_deterministic_node_id(&a1).expect("derive 1");
        let id2 = derive_deterministic_node_id(&a2).expect("derive 2");
        assert_ne!(id1, id2);
    }

    #[test]
    fn different_props_yield_different_ids() {
        let a1 = args(Some("Person"), &["name=Alice"]);
        let a2 = args(Some("Person"), &["name=Bob"]);
        let id1 = derive_deterministic_node_id(&a1).expect("derive 1");
        let id2 = derive_deterministic_node_id(&a2).expect("derive 2");
        assert_ne!(id1, id2);
    }
}
