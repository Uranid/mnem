//! Static MCP tool description table.
//!
//! Extracted from `tools.rs` in R3. `all_tools(allow_labels)` is the
//! single entry point the MCP server calls to advertise its tool list.

use serde_json::json;

use crate::protocol::ToolDef;

/// Build the tool list.
///
/// The advertised schemas are **stable**: they do NOT mutate based on
/// `allow_labels` / `MNEM_BENCH`. This is a post-audit guarantee - a
/// public API surface that changes shape based on a runtime env var
/// is not a public API. Every schema always exposes the full set of
/// fields (including `label` / `ntype`). The `MNEM_BENCH` gate is
/// still enforced at the **handler** layer (see `handlers/*.rs`): when
/// the gate is off, caller-supplied `label` / `ntype` is silently
/// coerced to `Node::DEFAULT_NTYPE`. Schema introspection therefore
/// always shows the full surface; the handler side is the boundary.
///
/// The `allow_labels` parameter is retained as `pub fn` signature for
/// source + binary compat with callers that already thread it through;
/// the value is ignored at schema-build time.
pub fn all_tools(allow_labels: bool) -> Vec<ToolDef> {
    // Retained to preserve the public signature post-audit; handlers
    // are where the label/ntype gate is enforced.
    let _ = allow_labels;

    let search_schema = json!({
        "type": "object",
        "properties": {
            "label":         { "type": "string", "description": "Node label (e.g. 'Person'). Honoured by default. Set MNEM_LABELS=0 (or legacy MNEM_BENCH=0) at server launch to force every label to Node::DEFAULT_NTYPE." },
            "where":         { "type": "object", "description": "Optional prop-equality filter, e.g. {\"name\": \"Alice\"}. Single property only in this version." },
            "with_outgoing": { "type": "array", "items": { "type": "string" }, "description": "Edge labels to include on each hit." },
            "limit":         { "type": "integer", "minimum": 1, "maximum": 500, "default": 10 }
        },
        "additionalProperties": false
    });

    let commit_nodes_item_schema = json!({
        "type": "object",
        "properties": {
            "ntype":   { "type": "string", "description": "Node type / label. Honoured by default. Set MNEM_LABELS=0 (or legacy MNEM_BENCH=0) at server launch to force the handler to substitute Node::DEFAULT_NTYPE." },
            "summary": { "type": "string", "description": "Short LLM-facing summary. Indexed by text + retrieve." },
            "props":   { "type": "object" },
            "content": { "type": "string", "description": "Optional text/markdown body (UTF-8)." }
        },
        "additionalProperties": false
    });

    let list_nodes_schema = json!({
        "type": "object",
        "properties": {
            "label":  { "type": "string", "description": "Optional label (ntype) filter. Honoured by default. Set MNEM_LABELS=0 (or legacy MNEM_BENCH=0) at server launch to force the filter to be silently dropped." },
            "limit":  { "type": "integer", "minimum": 1, "maximum": 1000, "default": 50 },
            "offset": { "type": "integer", "minimum": 0, "default": 0 }
        },
        "additionalProperties": false
    });

    // `mnem_resolve_or_create`: `label` is load-bearing for the tool's
    // semantics ("find-or-create by (label, prop_name) == value"). We
    // always advertise it. When the server is not launched under
    // MNEM_BENCH=1, the handler substitutes Node::DEFAULT_NTYPE for any
    // caller-supplied label.
    // audit-2026-04-25 C3-10: accept a friendly `{name, kind}`
    // shape as an alias for `{prop_name: "name", value: <name>,
    // label: <kind>}`. Most agent callers think in (entity-name,
    // entity-type) terms; the canonical (label, prop_name, value)
    // shape stays available for callers that anchor on a different
    // property (e.g. `email`, `slug`). `agent_id` defaults to
    // "mnem-mcp" so the alias path is callable end-to-end without
    // extra fields.
    let resolve_or_create_schema = json!({
        "type": "object",
        "properties": {
            "label":     { "type": "string", "description": "Node label / kind. Honoured by default. Set MNEM_LABELS=0 (or legacy MNEM_BENCH=0) at server launch to force the handler to substitute Node::DEFAULT_NTYPE." },
            "kind":      { "type": "string", "description": "Alias for `label`. Pick one." },
            "prop_name": { "type": "string", "description": "Property to anchor the find-or-create on. Defaults to `name` when the `name` alias is used." },
            "name":      { "type": "string", "description": "Alias for the natural-language entity name. When set, `prop_name` defaults to \"name\" and `value` defaults to this string." },
            "value":     { "description": "String, number, bool, or JSON object/array. Canonicalised before indexing." },
            "agent_id":  { "type": "string", "description": "Commit author. Defaults to 'mnem-mcp' when absent." },
            "task_id":   { "type": "string" },
            "extra_props": { "type": "object", "description": "Additional properties to set if the node has to be created." }
        },
        "anyOf": [
            { "required": ["prop_name", "value"] },
            { "required": ["name"] }
        ],
        "additionalProperties": false
    });

    let retrieve_schema = json!({
        "type": "object",
        "properties": {
            "label":        { "type": "string", "description": "Label filter. Honoured by default. Set MNEM_LABELS=0 (or legacy MNEM_BENCH=0) at server launch to force the filter to be silently dropped." },
            "where":        { "type": "object", "description": "Optional single-property equality gate, e.g. {\"team\": \"eng\"}." },
            "text":         { "type": "string", "description": "Query text. Retained so a reranker can read (query, candidate) pairs jointly. For retrieval proper, pass a `vector` in the matching embed model or configure the sparse lane separately." },
            "vector":       {
                "type": "object",
                "properties": {
                    "model":  { "type": "string", "minLength": 1 },
                    "values": { "type": "array", "items": { "type": "number" }, "minItems": 1 }
                },
                "required": ["model", "values"],
                "additionalProperties": false
            },
            "token_budget": { "type": "integer", "minimum": 0, "description": "Max rendered-text tokens to return. Default: unlimited." },
            "limit":        { "type": "integer", "minimum": 1, "description": "Max items to return, independent of the token budget. No hard ceiling; callers own back-pressure." },
            "vector_cap":   { "type": "integer", "minimum": 1, "description": "Override the per-lane cap on vector candidates (default: retriever-built-in). Raising it lets rerank / graph-expand see more of the long tail." },
            "rerank_top_k": { "type": "integer", "minimum": 1, "description": "If a reranker is wired in via the host config, how many fused candidates to rerank. Has no effect without a reranker." },
            "fusion":       { "type": "string", "enum": ["convex_min_max", "rrf"], "description": "Rank-fusion strategy over the lane outputs. `convex_min_max` (default) per Bruch 2023; `rrf` for the classic Reciprocal Rank Fusion baseline." },
            "graph_expand": { "type": "integer", "minimum": 1, "description": "Enable graph-expand: after hybrid fusion produces a top-K, traverse authored edges up to this many frontier nodes. Disables when absent." },
            "graph_decay":  { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Score decay applied per hop during graph-expand. Default preserves retriever built-in." },
            "graph_depth":  { "type": "integer", "minimum": 1, "maximum": 4, "description": "Multi-hop traversal depth. 1 = single-hop; 2+ for MuSiQue-style compositional queries. Clamped to [1, 4]." },
            "graph_etype":  { "type": "array", "items": { "type": "string" }, "description": "Edge-type allowlist for graph-expand. Empty / absent means all edge types." },
            "graph_max_per_seed": { "type": "integer", "minimum": 1, "description": "Per-seed outgoing-edge cap: prevents a hot-seed node from starving siblings in the global graph_expand budget." },
            "graph_mode":   { "type": "string", "enum": ["decay", "ppr"], "description": "Graph-expand strategy. `decay` (default) = historical BFS with decay^depth scoring; `ppr` = personalised PageRank over the hybrid adjacency index (E2+). PPR falls through to decay when no adjacency index is wired." },
            "ppr_damping":  { "type": "number", "minimum": 0.0, "maximum": 0.999, "description": "PPR damping factor. Default 0.85. Ignored unless graph_mode = \"ppr\"." },
            "ppr_iter":     { "type": "integer", "minimum": 1, "description": "PPR power-iteration cap. Default 15. Ignored unless graph_mode = \"ppr\"." }
        },
        "additionalProperties": false
    });

    #[cfg_attr(not(feature = "summarize"), allow(unused_mut))]
    let mut tools: Vec<ToolDef> = vec![
        ToolDef {
            name: "mnem_stats",
            description: "Repository overview: op-head, head commit, ref summary, known labels. \
                          Cheap; call this first to discover what a repo contains.",
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            }),
        },
        ToolDef {
            name: "mnem_schema",
            description: "List every node label and edge label present in the current commit, \
                          along with the property names the IndexSet has built for each label. \
                          Agents use this to write well-scoped queries.",
            input_schema: json!({
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": false
            }),
        },
        ToolDef {
            name: "mnem_search",
            description: "Search for nodes. Uses the indexed path when a label + exact property \
                          match is specified; falls back to label-scoped scan or full scan \
                          otherwise. Optionally include each hit's outgoing edges of named \
                          labels.",
            input_schema: search_schema,
        },
        ToolDef {
            name: "mnem_get_node",
            description: "Fetch a single node by UUID (as returned by mnem_search / mnem_commit). \
                          Returns full props + content size + outgoing edge count.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Node UUID (hyphenated form)." }
                },
                "required": ["id"],
                "additionalProperties": false
            }),
        },
        ToolDef {
            name: "mnem_traverse",
            description: "From a start node, list outgoing neighbours reachable via specified \
                          edge labels. One-hop only in this version; deeper traversal lands in a future version.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "start":       { "type": "string", "description": "Start node UUID." },
                    "edge_labels": { "type": "array", "items": { "type": "string" }, "description": "Edge labels to follow." },
                    "limit":       { "type": "integer", "minimum": 1, "maximum": 200, "default": 25 }
                },
                "required": ["start"],
                "additionalProperties": false
            }),
        },
        ToolDef {
            name: "mnem_commit",
            description: "Add nodes and/or edges as a single commit. `agent_id` (required) is \
                          stored as the Commit author. `task_id` is accepted and reserved for \
                          future Operation.task_id plumbing (tracked in ); today it is \
                          not persisted. Returns the new op-id, commit CID, and created node UUIDs.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "agent_id": { "type": "string", "description": "Required. Stored as the Commit author." },
                    "task_id":  { "type": "string", "description": "Reserved. Accepted but not yet persisted ." },
                    "message":  { "type": "string", "default": "" },
                    "nodes":    {
                        "type": "array",
                        "items": commit_nodes_item_schema
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "etype": { "type": "string" },
                                "src":   { "type": "string", "description": "Source node UUID." },
                                "dst":   { "type": "string", "description": "Destination node UUID." },
                                "props": { "type": "object" }
                            },
                            "required": ["etype", "src", "dst"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["agent_id"],
                "additionalProperties": false
            }),
        },
        ToolDef {
            name: "mnem_commit_relation",
            description: "Compound write: resolve-or-create a subject node, resolve-or-create an \
                          object node, and connect them with a typed edge - all in one commit. \
                          Audit fix G6 (2026-04-25): collapses the 3-tool dance \
                          (resolve_or_create + resolve_or_create + commit-edge) that an LLM under \
                          no specific instruction was unlikely to perform fully, leaving the graph \
                          flat. Anchor property defaults to `name`; pass `anchor` to switch to \
                          `email` / `slug` / `id`. Typical call: \
                          {\"subject\": \"Alice\", \"subject_kind\": \"Entity:Person\", \
                          \"predicate\": \"works_at\", \"object\": \"Globex\", \
                          \"object_kind\": \"Entity:Organization\"}.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "subject":      { "type": "string", "description": "Subject entity natural-language name (the value of the anchor property)." },
                    "subject_kind": { "type": "string", "description": "Subject ntype (e.g. 'Entity:Person'). Honoured when labels are enabled (default); otherwise the handler substitutes Node::DEFAULT_NTYPE." },
                    "predicate":    { "type": "string", "description": "Edge type (e.g. 'works_at', 'lives_in', 'has_preference')." },
                    "object":       { "type": "string", "description": "Object entity natural-language name (the value of the anchor property)." },
                    "object_kind":  { "type": "string", "description": "Object ntype (e.g. 'Entity:Organization'). Honoured when labels are enabled (default); otherwise the handler substitutes Node::DEFAULT_NTYPE." },
                    "anchor":       { "type": "string", "default": "name", "description": "Property name to anchor the resolve_or_create on. Defaults to `name`." },
                    "subject_props":{ "type": "object", "description": "Optional extra props to set on the subject node." },
                    "object_props": { "type": "object", "description": "Optional extra props to set on the object node." },
                    "edge_props":   { "type": "object", "description": "Optional props to set on the edge." },
                    "agent_id":     { "type": "string", "description": "Commit author. Defaults to 'mnem-mcp' when absent." },
                    "message":      { "type": "string", "default": "mnem_mcp commit_relation" }
                },
                "required": ["subject", "predicate", "object"],
                "additionalProperties": false
            }),
        },
        ToolDef {
            name: "mnem_delete_node",
            description: "Remove a node from the current head. Commits a new op with the removal. \
                          The node is no longer reachable from the new commit's node tree, but its \
                          prior CID and any prior commits that referenced it remain addressable \
                          (mnem's history is append-only). Edges incident to the node are NOT \
                          auto-removed; delete them explicitly or via a future cascade flag.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id":        { "type": "string", "description": "Node UUID to remove." },
                    "agent_id":  { "type": "string", "description": "Required. Stored as the Commit author." },
                    "message":   { "type": "string", "default": "mnem_mcp delete" }
                },
                "required": ["id", "agent_id"],
                "additionalProperties": false
            }),
        },
        ToolDef {
            name: "mnem_tombstone_node",
            description: "Logically \"forget\" a node without deleting its content. Unlike \
                          mnem_delete_node this does NOT remove the node from the node tree - the \
                          node's CID stays stable and any prior edges / commits that reference \
                          it remain intact. What changes is that subsequent retrieves filter the \
                          node out by default (agent can no longer see the memory). Use this when \
                          a user says \"forget X\" or revokes consent; use mnem_delete_node only \
                          when the goal is to free storage, not memory hygiene. Errors if the \
                          node does not exist or has already been tombstoned.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id":        { "type": "string", "description": "Node UUID to tombstone." },
                    "reason":    { "type": "string", "description": "Free-form reason recorded on the tombstone (e.g. the user's own phrasing)." },
                    "agent_id":  { "type": "string", "description": "Required. Stored as the Commit author." },
                    "message":   { "type": "string", "default": "mnem_mcp tombstone" }
                },
                "required": ["id", "agent_id"],
                "additionalProperties": false
            }),
        },
        ToolDef {
            name: "mnem_list_nodes",
            description: "Enumerate nodes at the current head, optionally filtered by label. \
                          Returns UUID + label + optional summary per node. Cheap discovery tool \
                          an agent can call before composing a retrieval: lets it see what's in \
                          the repo without a text-search guess.",
            input_schema: list_nodes_schema,
        },
        ToolDef {
            name: "mnem_resolve_or_create",
            description: "Find-or-create a node by a primary-key property. Accepts EITHER the \
                          friendly `{name: \"Alice\", kind: \"Person\"}` shape (anchors on the \
                          `name` property) OR the canonical \
                          `{prop_name: \"email\", value: \"a@x\", label: \"Person\"}` shape \
                          (anchors on whatever property you choose). If a node with the same \
                          (label, anchor-property) == value already exists, its UUID is \
                          returned; otherwise a new node is committed. Prevents the duplicate-\
                          entity problem agents hit when the same fact is re-asserted across \
                          tool calls. audit-2026-04-25 C3-10: `name`/`kind` aliases added.",
            input_schema: resolve_or_create_schema,
        },
        ToolDef {
            name: "mnem_recent",
            description: "Walk the op-log from the current head backwards. Returns the last N \
                          operations with time, author, agent_id, task_id, and one-line message.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "minimum": 1, "maximum": 100, "default": 10 }
                },
                "additionalProperties": false
            }),
        },
        ToolDef {
            name: "mnem_vector_search",
            description: "Cosine-similarity nearest-neighbour search over stored node embeddings. \
                          Pass the embedding-model identifier and a query vector; receive the \
                          top-k matches. Nodes whose embedding.model differs from the query are \
                          silently skipped - each index binds to one (model, dim).",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "model":  { "type": "string", "minLength": 1 },
                    "vector": { "type": "array", "items": { "type": "number" }, "minItems": 1 },
                    "k":      { "type": "integer", "minimum": 1, "maximum": 500, "default": 10 }
                },
                "required": ["model", "vector"],
                "additionalProperties": false
            }),
        },
        ToolDef {
            name: "mnem_retrieve",
            description: "Composite retrieval: combines label + prop-eq filters with cosine \
                          vector search and (optionally) the learned-sparse lane, fuses ranked \
                          lists via min-max convex combination (Bruch 2023) or RRF, optionally \
                          runs multi-hop graph expansion over the authored edges, and greedily \
                          packs rendered nodes under a token budget. Use this as the default \
                          tool when assembling LLM context: it returns nodes pre-rendered to \
                          text plus tokens_used / dropped / candidates_seen metadata so you \
                          know whether the budget was tight. All retrieval knobs exposed by \
                          POST /v1/retrieve are available here so MCP callers reach parity \
                          with the HTTP surface.",
            input_schema: retrieve_schema,
        },
        ToolDef {
            name: "mnem_ingest",
            description: "Ingest a source as a Doc + Chunk + Entity subgraph. Accepts EITHER \
                          {path: \"<file>\"} (server reads the file from disk) OR \
                          {text: \"...\", source?: \"label\"} (caller has already buffered the \
                          document). Runs parse + chunk + rule-based-NER and commits in one \
                          transaction. Chunker choice: 'auto' (picks per source kind), \
                          'paragraph' (blank-line split, best for markdown), 'recursive' \
                          (token-budgeted sliding window, best for PDFs), 'session' (groups \
                          conversation messages). Typical calls: \
                          {\"path\": \"notes.md\"}, \
                          {\"path\": \"book.pdf\", \"chunker\": \"recursive\", \"max_tokens\": 1024}, \
                          {\"text\": \"Alice met Bob.\", \"source\": \"convo-2026-04-25\"}. \
                          File / text size is capped at 32 MiB and max_tokens at 8192 for DoS \
                          resistance. Returns commit_cid plus per-run node / chunk / entity / \
                          relation counts. audit-2026-04-25 C3-8: schema accepts both shapes.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path":       { "type": "string", "description": "Absolute or relative path to the source file on the MCP server's filesystem. Mutually exclusive with `text`." },
                    "text":       { "type": "string", "description": "Inline document body. Use this when the caller already has the bytes; mutually exclusive with `path`." },
                    "source":     { "type": "string", "description": "Cosmetic label rendered as the `path:` field in the output when ingesting via `text`. Defaults to 'inline-text'." },
                    "ntype":      { "type": "string", "description": "Root Doc node label (default 'Doc').", "default": "Doc" },
                    "chunker":    { "type": "string", "enum": ["auto", "paragraph", "recursive", "session"], "default": "auto" },
                    "max_tokens": { "type": "integer", "minimum": 1, "maximum": 8192, "default": 512 },
                    "overlap":    { "type": "integer", "minimum": 0, "maximum": 8192, "default": 32 },
                    "agent_id":   { "type": "string", "description": "Commit author. Defaults to 'mnem-mcp' when absent." },
                    "message":    { "type": "string", "default": "mnem_mcp ingest" }
                },
                "anyOf": [
                    { "required": ["path"] },
                    { "required": ["text"] }
                ],
                "additionalProperties": false
            }),
        },
    ];
    // C3 FIX-5: community_summarize is the only tool that pulls the
    // embed-providers tree into the MCP binary. Hide it behind the
    // `summarize` feature so default builds stay lean (~2.3 MiB saving).
    #[cfg(feature = "summarize")]
    {
        // E4 T2: extractive community summarizer. No LLM, no BM25;
        // reuses the embedder the server already uses for retrieve.
        tools.push(ToolDef {
            name: "mnem_community_summarize",
            description: "Extractive Centroid + MMR summarizer over a caller-supplied set of node \
                          UUIDs. Looks up each node's `summary` field, embeds the collected \
                          sentences through the server's configured embedder (MNEM_EMBED_* env \
                          vars or `[embed]` in <repo>/config.toml), and picks `k` sentences \
                          balancing proximity to the community centroid against MMR diversity. \
                          No LLM call, no rewrite: the returned sentences are verbatim slices \
                          from the input summaries. Optional `query` biases selection toward \
                          query-relevant sentences. This is the MCP mirror of POST /v1/retrieve \
                          with `summarize: true`, except you choose the node set directly \
                          (typical callers: a Leiden-community node list, or a hand-curated \
                          subgraph). Degree-centrality fallback is uniform today; PPR slots in \
                          unchanged once E2 lands.",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "node_ids":   {
                        "type": "array",
                        "items": { "type": "string" },
                        "minItems": 1,
                        "maxItems": 10000,
                        "description": "Node UUIDs (as produced by other tool outputs)."
                    },
                    "query":      {
                        "type": "string",
                        "description": "Optional query text. When set, biases sentence selection toward query-relevance (beta=0.3 in the Centroid+MMR weighting)."
                    },
                    "k":          {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1000,
                        "default": 3,
                        "description": "Max number of sentences to return. Clamped to min(k, sentences)."
                    },
                    "mmr_lambda": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.5,
                        "description": "MMR diversity weight. 0 = pure relevance, 1 = pure diversity."
                    }
                },
                "required": ["node_ids"],
                "additionalProperties": false
            }),
        });
    }
    tools
}
