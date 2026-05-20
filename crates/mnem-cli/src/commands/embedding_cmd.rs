//! `mnem embedding` subcommand group.
//!
//! Exposes two subcommands:
//!
//! * `get <node-id> <model>` - print the embedding vector for a node.
//! * `ls <node-id>`          - list every model stored for a node.

use std::path::Path;

use anyhow::{Result, anyhow};
use clap::Subcommand;
use mnem_core::id::NodeId;

use super::*;

// ---- Args / Cmd ----

/// `mnem embedding` subcommand group.
#[derive(clap::Args, Debug)]
pub(crate) struct EmbeddingArgs {
    #[command(subcommand)]
    pub cmd: EmbeddingCmd,
}

/// Subcommands under `mnem embedding`.
#[derive(Subcommand, Debug)]
pub(crate) enum EmbeddingCmd {
    /// Fetch and print the embedding vector for a node.
    ///
    /// Prints the vector as space-separated floats to stdout and prints
    /// `model=<model> dim=<dim> dtype=<dtype>` to stderr on success.
    /// Exits non-zero when no embedding exists for the requested model.
    Get(GetArgs),
    /// List all embedding models stored for a node.
    ///
    /// Prints one model identifier per line on stdout. Exits non-zero when
    /// the node has no embeddings at all (or does not exist).
    Ls(LsArgs),
}

/// Arguments for `mnem embedding get`.
#[derive(clap::Args, Debug)]
pub(crate) struct GetArgs {
    /// UUID of the node whose embedding to fetch.
    pub node_id: String,
    /// Model identifier string (e.g. `onnx:all-MiniLM-L6-v2`).
    pub model: String,
}

/// Arguments for `mnem embedding ls`.
#[derive(clap::Args, Debug)]
pub(crate) struct LsArgs {
    /// UUID of the node whose embedding models to list.
    pub node_id: String,
}

// ---- Dispatch ----

pub(crate) fn run(override_path: Option<&Path>, args: EmbeddingArgs) -> Result<()> {
    match args.cmd {
        EmbeddingCmd::Get(a) => run_get(override_path, a),
        EmbeddingCmd::Ls(a) => run_ls(override_path, a),
    }
}

// ---- `mnem embedding get` ----

fn run_get(override_path: Option<&Path>, args: GetArgs) -> Result<()> {
    let id = NodeId::parse_uuid(&args.node_id).map_err(|e| anyhow!("invalid UUID: {e}"))?;

    let (_dir, r, _bs, _ohs) = repo::open_all(override_path)?;

    let node = r
        .lookup_node(&id)?
        .ok_or_else(|| anyhow!("no node with id={}", args.node_id))?;

    let (_, node_cid) =
        mnem_core::codec::hash_to_cid(&node).map_err(|e| anyhow!("hash node: {e}"))?;

    let emb = r.embedding_for(&node_cid, &args.model)?.ok_or_else(|| {
        anyhow!(
            "no embedding for model={} on node {}",
            args.model,
            args.node_id
        )
    })?;

    // Decode f32 vector from little-endian bytes.
    let bytes = emb.vector.as_ref();
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // Print vector to stdout (space-separated).
    let line: Vec<String> = floats.iter().map(ToString::to_string).collect();
    println!("{}", line.join(" "));

    // Print metadata to stderr.
    let dtype_str = match emb.dtype {
        mnem_core::objects::Dtype::F32 => "f32",
        mnem_core::objects::Dtype::F16 => "f16",
        mnem_core::objects::Dtype::F64 => "f64",
        mnem_core::objects::Dtype::I8 => "i8",
    };
    eprintln!("model={} dim={} dtype={}", emb.model, emb.dim, dtype_str);

    Ok(())
}

// ---- `mnem embedding ls` ----

fn run_ls(override_path: Option<&Path>, args: LsArgs) -> Result<()> {
    let id = NodeId::parse_uuid(&args.node_id).map_err(|e| anyhow!("invalid UUID: {e}"))?;

    let (_dir, r, _bs, _ohs) = repo::open_all(override_path)?;

    let node = r
        .lookup_node(&id)?
        .ok_or_else(|| anyhow!("no node with id={}", args.node_id))?;

    let (_, node_cid) =
        mnem_core::codec::hash_to_cid(&node).map_err(|e| anyhow!("hash node: {e}"))?;

    let models = r.embedding_models_for(&node_cid)?;

    if models.is_empty() {
        anyhow::bail!("no embeddings for node {}", args.node_id);
    }

    for m in models {
        println!("{m}");
    }

    Ok(())
}
