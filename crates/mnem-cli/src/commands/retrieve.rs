use super::*;

#[derive(clap::Args, Debug)]
#[command(after_long_help = "\
Examples:

  # Positional form: one-shot query, embedder/rerank use whatever is
  # configured (`mnem config set embed.provider ollama` once).
  mnem retrieve \"Alice Berlin\"

  # Persistent defaults via config (git-shaped \"set once\" UX):
  mnem config set retrieve.limit 5
  mnem config set retrieve.graph_expand 20
  mnem retrieve \"climbing in Berlin\"

  # Property filter + per-call override.
  mnem retrieve --where name=Alice -n 5 \"climbing\"

  # Property filter only (no ranker).
  mnem retrieve --where name=Alice --limit 10

  # Force text-only (skip semantic even with embedder configured).
  mnem retrieve --no-vector \"Alice\"
")]
// Many independent CLI flags: community-filter (E1), graph-mode/ppr (E2),
// summarize (E4), plus pre-existing no_vector/explain. A state machine
// does not apply: these are orthogonal user toggles, not phases.
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct Args {
    /// Query text (positional). Preferred over `-t`/`--text`; both
    /// forms cannot be set at once. Embedded via the configured
    /// embedder and retrieved through the dense vector lane;
    /// also passed to a cross-encoder reranker when installed.
    #[arg(value_name = "QUERY")]
    pub query: Option<String>,
    /// Single property-equality gate as `KEY=VALUE`.
    /// VALUE parses as JSON when possible, otherwise as a string.
    #[arg(long = "where")]
    pub where_eq: Option<String>,
    /// Explicit text-query flag. Equivalent to the positional
    /// `QUERY`; use the positional form when possible.
    #[arg(long, short = 't', conflicts_with = "query")]
    pub text: Option<String>,
    /// Max rendered tokens in the combined output. Default: unlimited.
    #[arg(long, short = 'b')]
    pub budget: Option<u32>,
    /// Max items to return, independent of budget.
    #[arg(long, short = 'n')]
    pub limit: Option<usize>,
    /// Disable the cosine-vector ranker for this call, even when
    /// an embedder is configured.
    #[arg(long)]
    pub no_vector: bool,
    /// Override the configured embedding model for this call only.
    #[arg(long)]
    pub embed_model: Option<String>,
    /// Print diagnostic information about retrieval. Each retrieved item
    /// gets a `lanes: <name>=<score> ...` line on stdout showing which
    /// ranker lanes contributed and their native scores. Retrieval
    /// decisions (multi-query variants, HyDE preview, community-filter
    /// status) are printed to stderr. Off by default so scripts see
    /// clean output; turn on when debugging why a result appeared or
    /// did not appear.
    #[arg(long = "explain")]
    pub explain: bool,
    /// Cross-encoder reranker to apply after fusion (tier 3; see
    /// ). `PROVIDER:MODEL`, e.g. `cohere:rerank-v3.5`
    /// or `voyage:rerank-2` or `jina:jina-reranker-v3`.
    /// Requires the provider's API key in the env var named by
    /// `rerank.api_key_env` (defaults: `COHERE_API_KEY`,
    /// `VOYAGE_API_KEY`, `JINA_API_KEY`). Overrides any persistent
    /// `[rerank]` section in `.mnem/config.toml` for this call. On
    /// any adapter error the retriever silently falls back to the
    /// fused order (never propagates rerank failures to stdout).
    #[arg(long = "rerank", value_name = "PROVIDER:MODEL")]
    pub rerank: Option<String>,
    /// Top-K of the fused list to re-score with `--rerank`. Default
    /// 25. Larger K means more per-query rerank cost; smaller K
    /// means the reranker has less to work with.
    #[arg(long = "rerank-top-k", value_name = "N", requires = "rerank")]
    pub rerank_top_k: Option<usize>,
    /// Experiment E1: community-filter stage between fusion and
    /// rerank. Requires a pre-computed `CommunityAssignment` to be
    /// wired in at the caller level; passing `--community-filter`
    /// with no assignment is a no-op (the retriever pipeline remains
    /// byte-identical to the `--community-filter=false` path). Off
    /// by default while we gather benchmark deltas in E1 T3.
    #[arg(long = "community-filter")]
    pub community_filter: bool,
    /// Minimum fraction of top-K fused weight that surviving
    /// communities must cover when `--community-filter` is enabled.
    /// Defaults to `0.5` (keep the top communities whose combined
    /// weight covers half of the total fused top-K score).
    #[arg(
        long = "community-min-coverage",
        value_name = "F",
        requires = "community_filter"
    )]
    pub community_min_coverage: Option<f32>,
    /// HyDE (Hypothetical Document Embeddings): ask an LLM to
    /// generate a hypothetical answer to the query, embed the
    /// answer instead of the query, use that vector for the
    /// semantic ranker. Works best when the user's query phrasing
    /// diverges from stored phrasing. Requires `[llm]` config
    /// (`mnem config set llm.provider openai|ollama`). Opt-in
    /// while we gather benchmark deltas; default off. Accepts an
    /// optional `PROVIDER:MODEL` one-shot override.
    #[arg(long = "hyde", value_name = "PROVIDER:MODEL", num_args = 0..=1, default_missing_value = "")]
    pub hyde: Option<String>,
    /// Multi-query / RAG-Fusion: ask an LLM to generate N query
    /// variations, retrieve top-K for each, RRF-fuse. Produces
    /// better recall on queries with sharp phrasing mismatch to
    /// stored summaries. Requires `[llm]` config. `N` defaults
    /// to 4 (plus the original, so 5 lanes total). Mutually
    /// exclusive with `--hyde` for now (future: compose them).
    #[arg(long = "multi-query", value_name = "N", num_args = 0..=1, default_missing_value = "4", conflicts_with = "hyde")]
    pub multi_query: Option<usize>,
    /// Enable graph-expand (tier 2): after the hybrid fusion produces
    /// a top-K, traverse outgoing edges 1 hop from each seed and
    /// add neighbors as candidates with a decay-weighted score. The
    /// expanded list is what the reranker (if any) re-scores.
    /// Structural advantage over chunk-bag competitors: mnem's
    /// graph is authored, not extracted, so expansion carries
    /// clean agent-authored relationship signal. `N` caps the
    /// total expanded neighbors; default 20. Nothing in mnem-core
    /// hard-limits this; override with `--graph-expand 200` or
    /// more when the corpus warrants it.
    #[arg(long = "graph-expand", value_name = "N", num_args = 0..=1, default_missing_value = "20")]
    pub graph_expand: Option<usize>,
    /// Score decay factor for expanded neighbors (graph-expand).
    /// `decay=1.0` treats neighbors as equals to seeds; `0.7`
    /// (default) ranks them below. Any float in (0, infinity) is
    /// accepted; values >1 would up-weight neighbors (unusual but
    /// allowed, not clamped).
    #[arg(long = "graph-decay", value_name = "F", requires = "graph_expand")]
    pub graph_decay: Option<f32>,
    /// Restrict graph-expand to edges with these etypes. Repeat
    /// for multiple: `--graph-etype sibling_of --graph-etype parent_of`.
    /// When unset, every outgoing edge is traversed.
    #[arg(long = "graph-etype", value_name = "ETYPE", requires = "graph_expand")]
    pub graph_etype: Vec<String>,
    /// Graph-expand strategy. `decay` (default) runs the historical
    /// multi-hop BFS with `decay^depth` scoring; `ppr` switches to
    /// personalised PageRank over the hybrid adjacency index (E2+).
    /// PPR requires a wired adjacency index (added in E3); without
    /// one it silently falls through to the decay walk for
    /// backward compatibility.
    ///
    /// When `--graph-mode` is set without `--graph-expand`, the
    /// expand budget defaults to 20 (same default as
    /// `--graph-expand` with no value). Pass an explicit
    /// `--graph-expand N` to override.
    #[arg(
        long = "graph-mode",
        value_name = "MODE",
        value_parser = ["decay", "ppr"]
    )]
    pub graph_mode: Option<String>,
    /// PPR damping factor (`d` in `(1 - d) * p + d * M^T r`). Default
    /// 0.85. Only honored when `--graph-mode ppr`.
    #[arg(long = "ppr-damping", value_name = "F", requires = "graph_mode")]
    pub ppr_damping: Option<f32>,
    /// PPR power-iteration cap. Default 15. Only honored when
    /// `--graph-mode ppr`.
    #[arg(long = "ppr-iter", value_name = "N", requires = "graph_mode")]
    pub ppr_iter: Option<u32>,
    /// Maximum vector-ranker candidates fed into fusion. Default
    /// 256. Corpora with >256 strongly-matching docs should raise
    /// this; the previous silent cap was an audit finding.
    #[arg(long = "vector-cap", value_name = "N")]
    pub vector_cap: Option<usize>,
    /// Max tokens in the HyDE hypothetical passage. Default 200.
    /// Raise for long-form prompts; the default caps 2-4 sentences.
    #[arg(long = "hyde-max-tokens", value_name = "N", requires = "hyde")]
    pub hyde_max_tokens: Option<u32>,
    /// HyDE sampling temperature. Default 0.7. Lower (0.1-0.3)
    /// when hallucinated specifics are hurting retrieval.
    #[arg(long = "hyde-temperature", value_name = "F", requires = "hyde")]
    pub hyde_temperature: Option<f32>,
    /// E4 T2: opt-in Centroid + MMR extractive summarization over
    /// the retrieved items' `summary` fields. Off by default so the
    /// flag is zero-impact when absent (same posture as the HTTP
    /// `summarize` body field and the MCP `mnem_community_summarize`
    /// tool). Requires a configured embedder; otherwise surfaces a
    /// one-line stderr hint and skips.
    #[arg(long = "summarize")]
    pub summarize: bool,
    /// Number of summary sentences to pick when `--summarize` is
    /// set. Default 3; ignored when `--summarize` is off.
    #[arg(long = "summarize-k", value_name = "N", requires = "summarize")]
    pub summarize_k: Option<usize>,
    /// Filter results by node type (ntype/label). May be specified multiple times.
    ///
    /// Note: this is a post-retrieval filter - results are fetched first, then filtered.
    /// The retrieval budget (--top-k / --limit) applies before filtering.
    #[arg(long = "label", short = 'l')]
    pub labels: Vec<String>,
}

pub(crate) fn run(override_path: Option<&Path>, mut args: Args) -> Result<()> {
    // Merge positional QUERY with explicit --text. Clap
    // `conflicts_with` already prevents both being set, so at most
    // one of args.query / args.text is Some here.
    if args.text.is_none() && args.query.is_some() {
        args.text = args.query.take();
    }
    let data_dir = repo::locate_data_dir(override_path)?;
    let cfg = config::load(&data_dir)?;
    // Fold persistent `[retrieve]` defaults from config.toml into
    // the flag values. CLI flags always win over config defaults:
    // `.or(...)` on None means "use config only when flag unset".
    // This is the "git-shaped set-once" UX - e.g.
    //   mnem config set retrieve.limit 20
    //   mnem retrieve "query"          # picks up limit=20
    //   mnem retrieve -n 5 "query"     # one-shot override
    if let Some(rd) = cfg.retrieve.as_ref() {
        args.budget = args.budget.or(rd.budget);
        args.limit = args.limit.or(rd.limit);
        args.vector_cap = args.vector_cap.or(rd.vector_cap);
        args.graph_expand = args.graph_expand.or(rd.graph_expand);
        args.graph_decay = args.graph_decay.or(rd.graph_decay);
        args.rerank_top_k = args.rerank_top_k.or(rd.rerank_top_k);
        args.hyde_max_tokens = args.hyde_max_tokens.or(rd.hyde_max_tokens);
        args.hyde_temperature = args.hyde_temperature.or(rd.hyde_temperature);
    }
    // audit-2026-04-25 P1-6: `--graph-mode` previously `requires`d
    // `--graph-expand`, so `mnem retrieve --graph-mode ppr` failed with
    // a clap error even though the budget has a sensible default.
    // Supply the same default (20) that `--graph-expand` (no value)
    // uses; explicit `--graph-expand N` still overrides.
    if args.graph_mode.is_some() && args.graph_expand.is_none() {
        args.graph_expand = Some(20);
    }
    let r = repo::open_repo(Some(data_dir.as_path()))?;
    let mut ret = r.retrieve();
    if let Some(w) = &args.where_eq {
        let (k, v) = parse_prop(w)?;
        ret = ret.where_prop(k, PropPredicate::Eq(v));
    }

    // Retain the original text so a cross-encoder reranker (if
    // installed) can read the `(query, candidate)` pair jointly.
    // The retrieve itself is driven by the embedder-produced
    // vector below; mnem-core no longer ships a lexical ranker.
    if let Some(t) = &args.text {
        ret = ret.query_text(t.clone());
    }
    if let Some(b) = args.budget {
        ret = ret.token_budget(b);
    }
    if let Some(n) = args.limit {
        ret = ret.limit(n);
    }
    if let Some(n) = args.vector_cap {
        ret = ret.vector_cap(n);
    }

    // Embedder input: verbatim user text (HyDE may replace this
    // below when `--hyde` is set).
    let mut embedder_text: Option<String> = args.text.clone();

    // Multi-query / RAG-Fusion: if `--multi-query N` is set AND
    // we have a text query AND an LLM is configured, ask the LLM
    // for N paraphrases, embed each + the original, run N+1
    // retrievals, and RRF-fuse the ranked lists. Returns the
    // fused result directly (early exit from the normal flow).
    if args.multi_query.is_some_and(|n| n > 0) && config::resolve_llm(&cfg, None).is_none() {
        eprintln!(
            "warning: --multi-query requires an LLM provider \
             (set [llm] in config); falling back to plain retrieve"
        );
    }
    if let Some(n_variants) = args.multi_query
        && n_variants > 0
        && let Some(q) = args.text.as_deref()
        && let Some(lc) = config::resolve_llm(&cfg, None)
        && let Some(pc) = config::resolve_embedder(&cfg)
    {
        match run_multi_query(&r, &cfg, &args, q, n_variants, &lc, &pc) {
            Ok(Some(result)) => {
                // Per-lane scores are not propagated through RRF fusion:
                // each sub-retrieval produces Vector lane scores but the
                // fused list is built from node-ID rank order, discarding
                // the sub-query native scores. Items therefore have empty
                // `lane_scores` and `print_retrieval_result` will not
                // print a `lanes:` line for them.
                if args.explain {
                    eprintln!(
                        "(multi-query: per-lane scores not propagated \
                         through RRF fusion; use plain retrieve for lane diagnostics)"
                    );
                }
                let result = filter_by_label(result, &args.labels);
                print_retrieval_result(&result, &args, &cfg);
                return Ok(());
            }
            Ok(None) => {
                if args.explain {
                    eprintln!(
                        "(multi-query produced empty variants; falling back to plain retrieve)"
                    );
                }
            }
            Err(e) => {
                if args.explain {
                    eprintln!("(multi-query error: {e}; falling back to plain retrieve)");
                }
            }
        }
    }

    // HyDE: if `--hyde` is set AND we have a text query AND an LLM
    // is configured, ask the LLM to generate a hypothetical answer
    // and REPLACE the embedder input with it. LLM failures fall
    // back to the plain embedder text.
    if args.hyde.is_some() && config::resolve_llm(&cfg, args.hyde.as_deref()).is_none() {
        eprintln!(
            "warning: --hyde requires an LLM provider \
             (set [llm] in config); falling back to plain retrieve"
        );
    }
    if args.hyde.is_some()
        && let Some(q) = args.text.as_deref()
        && let Some(lc) = config::resolve_llm(&cfg, args.hyde.as_deref())
    {
        use mnem_core::llm::{GenOptions, HYDE_PROMPT_TEMPLATE, fill_template};
        match mnem_llm_providers::open(&lc) {
            Ok(llm) => {
                let prompt = fill_template(HYDE_PROMPT_TEMPLATE, q);
                let opts = GenOptions {
                    n: 1,
                    max_tokens: Some(args.hyde_max_tokens.unwrap_or(200)),
                    temperature: Some(args.hyde_temperature.unwrap_or(0.7)),
                    ..Default::default()
                };
                match llm.generate(&prompt, &opts) {
                    Ok(mut passages) if !passages.is_empty() => {
                        let passage = passages.remove(0);
                        if args.explain {
                            eprintln!(
                                "(hyde via {}): {}",
                                llm.model(),
                                passage
                                    .lines()
                                    .next()
                                    .unwrap_or("")
                                    .chars()
                                    .take(120)
                                    .collect::<String>()
                            );
                        }
                        embedder_text = Some(format!("{q}\n{passage}"));
                    }
                    Ok(_) | Err(_) => {
                        if args.explain {
                            eprintln!("(hyde disabled: empty or failed completion)");
                        }
                    }
                }
            }
            Err(e) => {
                if args.explain {
                    eprintln!("(hyde disabled: {e})");
                }
            }
        }
    }

    // Embed when an embedder is configured and we have a text
    // query. After there is no lexical fallback in
    // mnem-core; embedder failures are surfaced to stderr but the
    // retrieve still proceeds with whatever filters the user set.
    let mut vector_attached = false;
    let mut mock_fallback_used = false;
    if !args.no_vector
        && let Some(text) = &embedder_text
        && let Some(mut pc) = config::resolve_embedder(&cfg)
    {
        if let Some(m) = &args.embed_model {
            override_model(&mut pc, m);
        }
        match mnem_embed_providers::open(&pc) {
            Ok(embedder) => match embedder.embed(text) {
                Ok(qvec) => {
                    ret = ret.vector(embedder.model().to_string(), qvec);
                    vector_attached = true;
                }
                Err(e) => {
                    eprintln!("{}", format_embed_failure(&e, &pc, "query embedding"));
                }
            },
            Err(e) => {
                eprintln!("{}", format_embed_failure(&e, &pc, "query embedding"));
            }
        }
    }

    // audit-2026-04-25 C7-3b: CLI-side mock-embedder fallback. Mirrors
    // the BENCH-1 cold-start fallback already wired into the HTTP
    // `/v1/retrieve` handler (mnem-http/src/handlers.rs:653-671 and
    // :958-978). Three trigger conditions, all-of:
    //   1. user passed a text query (positional or `--text`),
    //   2. user did NOT pass `--no-vector` (explicit opt-out),
    //   3. no real embedder was attached (either no `[embed]` config
    //      OR the configured provider failed: Ollama unreachable,
    //      OpenAI key missing/rejected, etc.).
    // Opt-out via `MNEM_DISABLE_MOCK_FALLBACK=1` for CI / strict-mode
    // tests that want the original "no semantic ranker" behaviour.
    // The mock is deterministic but NOT semantically meaningful, so
    // we emit a stderr warn note so users see the degradation.
    if !vector_attached
        && !args.no_vector
        && embedder_text
            .as_deref()
            .is_some_and(|t| !t.trim().is_empty())
        && !mock_fallback_disabled()
    {
        use mnem_embed_providers::Embedder as _;
        let text = embedder_text.as_deref().unwrap_or("");
        let mock = mnem_embed_providers::MockEmbedder::new("mock:cold-start-384", 384);
        match mock.embed(text) {
            Ok(qvec) => {
                ret = ret.vector(mock.model().to_string(), qvec);
                vector_attached = true;
                mock_fallback_used = true;
                eprintln!(
                    "warn: using mock embedder for retrieval - semantic results unreliable. \
                     Configure [embed] in .mnem/config.toml for production."
                );
            }
            Err(e) => {
                eprintln!("(mock fallback failed: {e})");
            }
        }
    }

    // Attach graph-expand if requested. Defaults are overridable
    // by the caller; nothing in mnem-core clamps max_expand or
    // decay. (planned).
    if let Some(max_expand) = args.graph_expand {
        let mut ge = mnem_core::retrieve::GraphExpand {
            max_expand,
            decay: args
                .graph_decay
                .unwrap_or(mnem_core::retrieve::GraphExpand::DEFAULT_DECAY),
            etype_filter: None,
            ..Default::default()
        };
        if !args.graph_etype.is_empty() {
            ge.etype_filter = Some(args.graph_etype.clone());
        }
        // No CLI flag for graph_depth yet; pick it up from
        // `[retrieve].graph_depth` in config.toml when set so
        // multi-hop MuSiQue runs don't need a flag redesign.
        if let Some(depth) = cfg.retrieve.as_ref().and_then(|r| r.graph_depth) {
            ge = ge.with_depth(depth);
        }
        // E2: PPR mode dispatch via `--graph-mode ppr`. Defaults
        // (0.85 damping, 15 iterations, 1e-6 eps) match the mnem-core
        // [`ppr`] module constants.
        if let Some(mode) = args.graph_mode.as_deref()
            && mode == "ppr"
        {
            let damping = args.ppr_damping.unwrap_or(mnem_core::ppr::DEFAULT_DAMPING);
            let iter = args.ppr_iter.unwrap_or(mnem_core::ppr::DEFAULT_MAX_ITER);
            ge = ge.with_ppr(damping, iter, mnem_core::ppr::DEFAULT_EPS);
        }
        ret = ret.with_graph_expand(ge);
    }

    // Attach reranker if requested. Precedence:
    //   1. --rerank PROVIDER:MODEL (per-call override)
    //   2. `[rerank]` section in config.toml (or MNEM_RERANK_* env)
    // Adapter construction failures become a stderr warning and
    // the retriever proceeds without a reranker.
    let rerank_cfg: Option<mnem_rerank_providers::ProviderConfig> = match &args.rerank {
        Some(spec) => Some(config::parse_rerank_override(spec)?),
        None => config::resolve_reranker(&cfg),
    };
    if let Some(rcfg) = rerank_cfg {
        match mnem_rerank_providers::open(&rcfg) {
            Ok(rr) => {
                ret = ret.with_reranker(rr);
                if let Some(k) = args.rerank_top_k {
                    ret = ret.rerank_top_k(k);
                }
            }
            Err(e) => {
                eprintln!("(rerank disabled: {e})");
            }
        }
    }

    // audit-2026-04-25 C4-3: wire `--community-filter` on the CLI
    // side. The HTTP `/v1/retrieve` path already installs a
    // `CommunityLookup` via `with_community_filter` (handlers.rs);
    // the CLI parsed the flag but never read it, so passing
    // `--community-filter` was a silent no-op. Build authored-edge
    // adjacency from the repo head commit, compute Leiden
    // communities, and install the lookup. Empty adjacency yields
    // an empty assignment (`community_of` returns `None` for every
    // node), which makes the expander a passthrough -- byte-
    // identical to the off path.
    if args.community_filter {
        match collect_authored_edges_for_community(&r) {
            Ok(edges) => {
                let adj = mnem_core::index::AuthoredSliceAdjacency::new(&edges);
                let assignment =
                    std::sync::Arc::new(mnem_graphrag::community::compute_communities(&adj, 0));
                let fwd = assignment.clone();
                let inv = assignment.clone();
                let lookup =
                    std::sync::Arc::new(mnem_core::retrieve::CommunityLookup::new_with_members(
                        move |nid: &mnem_core::id::NodeId| fwd.community_of(*nid),
                        move |cid| inv.members_of(cid).to_vec(),
                    ));
                let cfg_cf = mnem_core::retrieve::CommunityFilterCfg {
                    enabled: true,
                    expand_seeds: 3,
                    max_per_community: 10,
                    decay: 0.85,
                    min_coverage: args.community_min_coverage.unwrap_or(0.5).clamp(0.0, 1.0),
                };
                if args.explain && assignment.community_count() == 0 {
                    eprintln!(
                        "(community-filter: no_community_assignment -- authored adjacency empty)"
                    );
                }
                ret = ret.with_community_filter(cfg_cf, lookup);
            }
            Err(e) => {
                if args.explain {
                    eprintln!("(community-filter disabled: {e})");
                }
            }
        }
    }

    let result = ret.execute()?;
    let result = filter_by_label(result, &args.labels);
    print_retrieval_result(&result, &args, &cfg);

    // E4 T2: optional Centroid + MMR extractive summarization over
    // the retrieved items' `summary` fields. Strictly opt-in via
    // `--summarize`; off means the print path ends above, unchanged
    // from the pre-E4 shape. Mirrors the HTTP `summarize: true`
    // field and the MCP `mnem_community_summarize` tool so all
    // three transports agree on the math.
    if args.summarize {
        print_summarize_section(&result, &args, &cfg);
    }

    // One-line footer hint on stderr so it doesn't pollute stdout
    // (which scripts pipe and parse). Suppressed when the mock
    // fallback already emitted its own warn (otherwise the user
    // sees two competing notes).
    #[allow(clippy::collapsible_if)]
    if !vector_attached && !args.no_vector && args.text.is_some() && !mock_fallback_used {
        if config::resolve_embedder(&cfg).is_none() {
            eprintln!(
                "(semantic search off - run `mnem config set embed.provider <openai|ollama>` \
                 to enable)"
            );
        }
    }
    Ok(())
}

/// audit-2026-04-25 C7-3b: opt-out switch for the mock-embedder
/// fallback. CI and strict-mode tests set `MNEM_DISABLE_MOCK_FALLBACK=1`
/// to get back the pre-fallback behaviour (retrieve proceeds without
/// a vector ranker when no real embedder is configured / available).
/// Any non-empty value other than `0` disables the fallback; this
/// matches the convention used by `MNEM_DISABLE_GLOBAL_CONFIG`.
fn mock_fallback_disabled() -> bool {
    std::env::var("MNEM_DISABLE_MOCK_FALLBACK")
        .ok()
        .is_some_and(|v| !v.is_empty() && v != "0")
}

/// Print the `--summarize` section of a retrieve output. Off-path
/// equivalent of the HTTP `summary` field: collects each retrieved
/// item's `node.summary`, opens the configured embedder, and runs
/// `mnem_graphrag::summarize_community` with degree-centrality
/// fallback and the caller-supplied `k` (default 3, lambda 0.5).
///
/// Prints exactly what the task spec asks for:
/// ```text
/// Summary:
///   1. <sentence> (score: 0.87)
///   2. <sentence> (score: 0.74)
/// ```
///
/// All "not enough data / no embedder / provider failed" paths emit
/// a one-line stderr hint and skip without failing the retrieve
/// itself. Parity with how HyDE / rerank handle their own miss
/// cases (stdout stays clean; scripts don't see a half-result).
fn print_summarize_section(
    result: &mnem_core::retrieve::RetrievalResult,
    args: &Args,
    cfg: &crate::config::Config,
) {
    let k = args.summarize_k.unwrap_or(3);
    let sentences: Vec<String> = result
        .items
        .iter()
        .filter_map(|it| it.node.summary.clone())
        .collect();
    if sentences.is_empty() {
        eprintln!("(summarize skipped: no retrieved items have a `summary` field)");
        return;
    }
    let Some(pc) = config::resolve_embedder(cfg) else {
        eprintln!(
            "(summarize skipped: no embedder configured - run \
             `mnem config set embed.provider <openai|ollama>` to enable)"
        );
        return;
    };
    let embedder = match mnem_embed_providers::open(&pc) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("(summarize skipped: embed provider open failed: {e})");
            return;
        }
    };
    let centrality = |_: usize| 1.0_f32;
    // Pass the user's text query through as the query vector so the
    // summary is query-focused when a text query was supplied;
    // otherwise fall back to unsupervised centroid-only mode.
    let query_embed: Option<Vec<f32>> = args
        .text
        .as_deref()
        .filter(|q| !q.is_empty())
        .and_then(|q| embedder.embed(q).ok());
    let summary = match mnem_graphrag::summarize_community(
        &sentences,
        embedder.as_ref(),
        query_embed.as_deref(),
        &centrality,
        k,
        0.5,
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("(summarize skipped: {e})");
            return;
        }
    };
    if summary.sentences.is_empty() {
        return;
    }
    println!("Summary:");
    for (i, (sentence, score)) in summary
        .sentences
        .iter()
        .zip(summary.scores.iter())
        .enumerate()
    {
        println!("  {}. {} (score: {:.2})", i + 1, sentence, score);
    }
}

/// Shared print path for a `RetrievalResult`. Both the normal
/// retrieve flow and the multi-query branch emit the same shape.
fn print_retrieval_result(
    result: &mnem_core::retrieve::RetrievalResult,
    args: &Args,
    cfg: &crate::config::Config,
) {
    let budget_str = if result.tokens_budget == u32::MAX {
        "unlimited".to_string()
    } else {
        result.tokens_budget.to_string()
    };
    println!(
        "# {} item(s), {}/{} tokens, {} dropped, {} candidates",
        result.items.len(),
        result.tokens_used,
        budget_str,
        result.dropped,
        result.candidates_seen,
    );
    for (i, item) in result.items.iter().enumerate() {
        println!(
            "---\n[{i}] score={:.4} tokens={} id={} {}",
            item.score,
            item.tokens,
            item.node.id.to_uuid_string(),
            item.node.ntype,
        );
        if args.explain && !item.lane_scores.is_empty() {
            let lanes: Vec<String> = item
                .lane_scores
                .iter()
                .map(|(lane, score)| format!("{}={:.4}", lane_name(*lane), score))
                .collect();
            println!("  lanes: {}", lanes.join(" "));
        }
        for line in item.rendered.lines() {
            println!("  {line}");
        }
    }
    // Stderr hint when the text query is the only ranker.
    // Suppressed when the C7-3b mock fallback would handle this
    // case (i.e. fallback is enabled and we have a non-empty text
    // query) so the caller doesn't see contradictory advice.
    if !args.no_vector
        && args.text.is_some()
        && config::resolve_embedder(cfg).is_none()
        && (mock_fallback_disabled() || args.text.as_deref().is_none_or(|t| t.trim().is_empty()))
    {
        eprintln!(
            "(semantic search off - run `mnem config set embed.provider <openai|ollama>` \
             to enable)"
        );
    }
}

/// Multi-query / RAG-Fusion retrieval. Ask the LLM for N
/// paraphrase variations of the query, embed each (plus the
/// original), run N+1 retrievals, and RRF-fuse the ranked lists.
///
/// Returns `Ok(None)` when the LLM produced zero usable variants
/// so the caller can fall back to plain retrieve. Returns
/// `Err(...)` on catastrophic failure; transient LLM / embed
/// errors return `Ok(None)` instead so the CLI always produces
/// some answer.
#[allow(clippy::too_many_arguments)]
fn run_multi_query(
    r: &mnem_core::repo::ReadonlyRepo,
    cfg: &crate::config::Config,
    args: &Args,
    query: &str,
    n_variants: usize,
    llm_cfg: &mnem_llm_providers::ProviderConfig,
    embed_cfg: &mnem_embed_providers::ProviderConfig,
) -> Result<Option<mnem_core::retrieve::RetrievalResult>> {
    use mnem_core::llm::{GenOptions, MULTI_QUERY_PROMPT_TEMPLATE, fill_multi_query_template};

    let llm = mnem_llm_providers::open(llm_cfg).map_err(|e| anyhow!("llm open failed: {e}"))?;
    let embedder =
        mnem_embed_providers::open(embed_cfg).map_err(|e| anyhow!("embed open failed: {e}"))?;

    let prompt = fill_multi_query_template(MULTI_QUERY_PROMPT_TEMPLATE, query, n_variants);
    let opts = GenOptions {
        n: 1,
        max_tokens: Some(512),
        temperature: Some(0.7),
        ..Default::default()
    };
    let completions = llm
        .generate(&prompt, &opts)
        .map_err(|e| anyhow!("llm generate failed: {e}"))?;
    if completions.is_empty() {
        return Ok(None);
    }
    let mut variants: Vec<String> = completions[0]
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(ToString::to_string)
        .collect();
    // Drop any variant identical to the query; cap to requested N.
    variants.retain(|v| v.as_str() != query);
    variants.truncate(n_variants);
    if variants.is_empty() {
        return Ok(None);
    }
    if args.explain {
        eprintln!("(multi-query variants: {} generated)", variants.len());
        for (i, v) in variants.iter().enumerate() {
            eprintln!("  [{i}] {v}");
        }
    }

    // Build all N+1 queries (original + variants). Run each
    // through a retrieve with the SAME filters and fusion knobs,
    // collect ranked NodeId lists, then RRF-fuse.
    let mut all_queries: Vec<String> = vec![query.to_string()];
    all_queries.extend(variants);

    let mut ranked_lists: Vec<(Vec<mnem_core::id::NodeId>, f32)> = Vec::new();
    for q in &all_queries {
        let mut ret = r.retrieve();
        if let Some(w) = &args.where_eq {
            let (k, v) = parse_prop(w)?;
            ret = ret.where_prop(k, mnem_core::index::PropPredicate::Eq(v));
        }
        ret = ret.query_text(q.clone());
        if let Some(n) = args.limit {
            // Over-retrieve slightly so RRF has material to rank.
            ret = ret.limit(n.saturating_mul(3).max(30));
        }
        if let Some(n) = args.vector_cap {
            ret = ret.vector_cap(n);
        }
        if !args.no_vector
            && let Ok(qvec) = embedder.embed(q)
        {
            ret = ret.vector(embedder.model().to_string(), qvec);
        }
        let sub = ret.execute()?;
        let ids: Vec<mnem_core::id::NodeId> = sub.items.iter().map(|i| i.node.id).collect();
        ranked_lists.push((ids, 1.0));
    }

    let fused = mnem_core::retrieve::weighted_reciprocal_rank_fusion(
        &ranked_lists,
        mnem_core::retrieve::Retriever::DEFAULT_RRF_K,
    );

    // Look up each fused node to produce a RetrievalResult. This
    // bypasses the reranker pass for now; the simpler multi-query
    // flow is useful without it, and reranking over fused lists is
    // a follow-up.
    let cap = args.limit.unwrap_or(usize::MAX);
    let budget = args.budget.unwrap_or(u32::MAX);
    let estimator: std::sync::Arc<dyn mnem_core::retrieve::TokenEstimator> =
        std::sync::Arc::new(mnem_core::retrieve::HeuristicEstimator);
    let mut items: Vec<mnem_core::retrieve::RetrievedItem> = Vec::new();
    let mut tokens_used: u32 = 0;
    let mut dropped: u32 = 0;
    let candidates_seen = u32::try_from(fused.len()).unwrap_or(u32::MAX);
    for (nid, score) in fused {
        if items.len() >= cap {
            dropped = dropped.saturating_add(1);
            continue;
        }
        let Some(node) = r.lookup_node(&nid)? else {
            continue;
        };
        let rendered = mnem_core::retrieve::render_node(&node);
        let tokens = estimator.estimate(&rendered);
        let next = tokens_used.saturating_add(tokens);
        if next > budget {
            dropped = dropped.saturating_add(1);
            continue;
        }
        tokens_used = next;
        items.push(mnem_core::retrieve::RetrievedItem::new(
            node, rendered, tokens, score,
        ));
    }
    let _ = cfg; // reserved for future per-tier config reads
    Ok(Some(mnem_core::retrieve::RetrievalResult::new(
        items,
        tokens_used,
        budget,
        dropped,
        candidates_seen,
    )))
}

/// audit-2026-04-25 C4-3 helper: walk the head commit's edges Prolly
/// tree once and collect every authored `(src, dst)` pair. Mirrors
/// the HTTP-side `collect_authored_edges` in
/// `crates/mnem-http/src/state.rs` so CLI `--community-filter` and
/// HTTP `community_filter: true` see byte-identical assignments.
/// Returns an empty vec when the repo has no head commit, which
/// makes the community filter a no-op (passthrough).
fn collect_authored_edges_for_community(
    repo: &mnem_core::repo::ReadonlyRepo,
) -> Result<Vec<(NodeId, NodeId)>> {
    let Some(commit) = repo.head_commit() else {
        return Ok(Vec::new());
    };
    let bs = repo.blockstore().clone();
    let cursor = mnem_core::prolly::Cursor::new(&*bs, &commit.edges)
        .map_err(|e| anyhow!("opening edge cursor: {e}"))?;
    let mut edges: Vec<(NodeId, NodeId)> = Vec::new();
    for entry in cursor {
        let (_key, edge_cid) = entry.map_err(|e| anyhow!("walking edge tree: {e}"))?;
        let bytes = bs
            .get(&edge_cid)
            .map_err(|e| anyhow!("fetching edge block: {e}"))?
            .ok_or_else(|| anyhow!("edge block {edge_cid} missing"))?;
        let edge: mnem_core::objects::Edge = mnem_core::codec::from_canonical_bytes(&bytes)
            .map_err(|e| anyhow!("decoding edge: {e}"))?;
        edges.push((edge.src, edge.dst));
    }
    Ok(edges)
}

/// Post-retrieval label filter. When `labels` is empty, returns the
/// result unchanged. Otherwise keeps only items whose `node.ntype`
/// appears in the label list. The count/token header reflects the
/// filtered set, not the raw retrieval.
fn filter_by_label(
    mut result: mnem_core::retrieve::RetrievalResult,
    labels: &[String],
) -> mnem_core::retrieve::RetrievalResult {
    if labels.is_empty() {
        return result;
    }
    result
        .items
        .retain(|item| labels.contains(&item.node.ntype));
    // Recalculate token count to reflect the filtered set
    result.tokens_used = result.items.iter().map(|i| i.tokens).sum();
    result
}

/// Maps a [`Lane`][mnem_core::retrieve::Lane] variant to its stable lowercase
/// string label as it appears in `--explain` output.
fn lane_name(lane: mnem_core::retrieve::Lane) -> &'static str {
    use mnem_core::retrieve::Lane;
    match lane {
        Lane::Vector => "vector",
        Lane::Sparse => "sparse",
        Lane::GraphExpand => "graph_expand",
        Lane::Rerank => "rerank",
        // Lane is #[non_exhaustive]; new variants fall here until an explicit
        // arm is added above. The debug_assert fires in dev/test builds so
        // the gap is caught immediately without changing release behaviour.
        _ => {
            debug_assert!(false, "unhandled Lane variant: add a new arm to lane_name()");
            "unknown"
        }
    }
}

/// Per-call model override: swap the `model` field on whichever
/// provider variant the config resolved to.
fn override_model(pc: &mut mnem_embed_providers::ProviderConfig, model: &str) {
    use mnem_embed_providers::ProviderConfig;
    match pc {
        ProviderConfig::Openai(c) => c.model = model.to_string(),
        ProviderConfig::Ollama(c) => c.model = model.to_string(),
        ProviderConfig::Onnx(c) => c.model = model.to_string(),
    }
}
