//! In-process mnem adapter. Spins up a fresh `ReadonlyRepo` backed
//! by `MemoryBlockstore` + `MemoryOpHeadsStore`, ingests through a
//! single transaction per batch, and queries via the
//! `repo.retrieve()` builder.
//!
//! The adapter holds a [`BenchEmbedder`] which covers both the toy
//! bag-of-tokens flavour (offline) and the real `all-MiniLM-L6-v2`
//! ONNX flavour via `mnem-embed-providers` (gated on the
//! `onnx-minilm` feature, default-on). The scoring path is
//! variant-blind.

use std::error::Error as StdError;
use std::sync::Arc;

use bytes::Bytes;
use mnem_core::id::NodeId;
use mnem_core::objects::{Dtype, Embedding, Node};
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

use crate::adapter::{BenchAdapter, Hit, IngestDoc};
use crate::embed::BenchEmbedder;

/// In-process mnem adapter (cpu-local mode).
///
/// One adapter instance owns one `ReadonlyRepo`. `reset` rotates a
/// fresh repo so per-question / per-conversation runs do not leak
/// across each other.
pub struct MnemAdapter {
    repo: ReadonlyRepo,
    embedder: BenchEmbedder,
    /// Track id -> external_id so retrieve hits can be projected
    /// back to bench-defined ids without a Node decode. mnem's
    /// retriever already returns the Node, but we keep the side map
    /// because the bench scoring loop is hot.
    id_to_external: std::collections::HashMap<NodeId, String>,
}

impl MnemAdapter {
    /// Construct a fresh adapter with a freshly-initialised
    /// in-memory mnem repo and the toy bag-of-tokens embedder of
    /// dimension `dim`. Kept for backward compatibility with 0.1.0
    /// callers; new code should prefer
    /// [`MnemAdapter::with_embedder`].
    ///
    /// # Errors
    ///
    /// Surfaces blockstore / repo init errors verbatim.
    pub fn new(dim: u32) -> Result<Self, Box<dyn StdError>> {
        Self::with_embedder(BenchEmbedder::bag_of_tokens(dim))
    }

    /// Construct a fresh adapter wrapping the supplied embedder.
    /// The adapter takes ownership and uses it for both ingest and
    /// retrieve so the model id stamped on every `Embedding`
    /// matches the lane the retriever queries.
    ///
    /// # Errors
    ///
    /// Surfaces blockstore / repo init errors verbatim.
    pub fn with_embedder(embedder: BenchEmbedder) -> Result<Self, Box<dyn StdError>> {
        let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::default());
        let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::default());
        let repo = ReadonlyRepo::init(bs, ohs).map_err(|e| Box::new(e) as Box<dyn StdError>)?;
        Ok(Self {
            repo,
            embedder,
            id_to_external: std::collections::HashMap::new(),
        })
    }

    /// Borrow the embedder so callers can inspect the active model
    /// or share it across non-adapter call sites.
    pub fn embedder(&self) -> &BenchEmbedder {
        &self.embedder
    }

    /// The model id reported on Embedding objects. Same string is
    /// passed to `Retriever::vector` so the lookup matches.
    pub fn model_id(&self) -> &str {
        self.embedder.model()
    }
}

impl BenchAdapter for MnemAdapter {
    fn reset(&mut self) -> Result<(), Box<dyn StdError>> {
        let bs: Arc<dyn Blockstore> = Arc::new(MemoryBlockstore::default());
        let ohs: Arc<dyn OpHeadsStore> = Arc::new(MemoryOpHeadsStore::default());
        self.repo = ReadonlyRepo::init(bs, ohs).map_err(|e| Box::new(e) as Box<dyn StdError>)?;
        self.id_to_external.clear();
        Ok(())
    }

    fn ingest(&mut self, docs: &[IngestDoc]) -> Result<(), Box<dyn StdError>> {
        if docs.is_empty() {
            return Ok(());
        }
        let mut tx = self.repo.start_transaction();
        for d in docs {
            let id = NodeId::new_v7();
            let mut node = Node::new(id, d.label.as_str()).with_summary(d.text.as_str());
            // Echo the external_id as a property so the retrieve
            // path can recover it without a second lookup.
            node = node.with_prop("external_id", ipld_string(d.external_id.as_str()));
            for (k, v) in &d.props {
                if let Some(s) = v.as_str() {
                    node = node.with_prop(k.as_str(), ipld_string(s));
                }
            }
            // Optional opaque content - scorers don't need it but
            // keeping the field aligned with the HTTP wire contract
            // keeps mnem-bench's adapter compatible with mnem-http
            // for future subprocess-mode wiring.
            if d.text.len() < 1 << 16 {
                node = node.with_content(Bytes::from(d.text.clone().into_bytes()));
            }
            // Embed + attach the dense vector via the sidecar.
            let vec = self.embedder.embed_text(&d.text)?;
            let emb = to_embedding_f32(self.embedder.model(), &vec);
            let cid = tx
                .add_node(&node)
                .map_err(|e| Box::new(e) as Box<dyn StdError>)?;
            tx.set_embedding(cid, self.embedder.model().to_string(), emb)
                .map_err(|e| Box::new(e) as Box<dyn StdError>)?;
            self.id_to_external.insert(id, d.external_id.clone());
        }
        let next = tx
            .commit("mnem-bench", "bench ingest")
            .map_err(|e| Box::new(e) as Box<dyn StdError>)?;
        self.repo = next;
        Ok(())
    }

    fn retrieve(
        &mut self,
        label: &str,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<Hit>, Box<dyn StdError>> {
        let qvec = self.embedder.embed_text(query)?;
        let result = self
            .repo
            .retrieve()
            .label(label)
            .vector(self.embedder.model().to_string(), qvec)
            .limit(top_k.max(1))
            .execute()
            .map_err(|e| Box::new(e) as Box<dyn StdError>)?;

        let mut out = Vec::with_capacity(result.items.len());
        for item in result.items {
            // Recover external_id either from the side-map (fast
            // path) or fall back to the node's `external_id` prop
            // when the side-map missed (e.g. a node created by an
            // adapter we did not author).
            let ext = if let Some(e) = self.id_to_external.get(&item.node.id) {
                e.clone()
            } else {
                node_external_id(&item.node).unwrap_or_default()
            };
            out.push(Hit {
                external_id: ext,
                score: item.score,
            });
        }
        Ok(out)
    }

    fn name(&self) -> &str {
        "mnem"
    }
}

// ============================================================
// Helpers
// ============================================================

/// Shorthand: build a string-typed `Ipld` value.
fn ipld_string(s: &str) -> ipld_core::ipld::Ipld {
    ipld_core::ipld::Ipld::String(s.to_string())
}

/// Read the `external_id` property off a Node. Returns `None` if
/// missing or non-string.
fn node_external_id(node: &Node) -> Option<String> {
    let v = node.props.get("external_id")?;
    match v {
        ipld_core::ipld::Ipld::String(s) => Some(s.clone()),
        _ => None,
    }
}

/// Build an `Embedding` from a `&[f32]`. Mirrors
/// `mnem_embed_providers::to_embedding` but lives here to avoid
/// pulling the full provider feature surface for what is a 4-line
/// helper. Endianness: native, matching the workspace convention
/// (`f32::to_ne_bytes` round-trips on the same machine and the
/// vector is regenerated rather than persisted across machines).
fn to_embedding_f32(model: &str, v: &[f32]) -> Embedding {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for f in v {
        bytes.extend_from_slice(&f.to_ne_bytes());
    }
    Embedding {
        model: model.to_string(),
        dtype: Dtype::F32,
        dim: v.len() as u32,
        vector: Bytes::from(bytes),
    }
}
