//! Agent memory under a token budget - the Phase-2 retrieval pitch in one file.
//!
//! The shape every agent coder recognises: a chat history has
//! accumulated facts about the user; the assistant is about to issue
//! the next prompt and has, say, 500 tokens of context window to spend
//! on memory. Naive path: dump every remembered fact, eat 5-10k tokens.
//! mnem path: `retrieve().vector(model, embedding).token_budget(500)`
//! returns the nodes that fit, in ranked order, with explicit `dropped`
//! and `tokens_used` metadata so the agent knows whether the budget was
//! tight.
//!
//! For simplicity this example uses a hand-rolled keyword-bag vector
//! so it has no external dependency on an embedding provider. Real
//! agents call `mnem-embed-providers` on write and on query.
//!
//! Run:
//! ```console
//! cargo run --release -p mnem-core --example agent_retrieve_under_budget
//! ```

use std::sync::Arc;

use bytes::Bytes;
use mnem_core::id::NodeId;
use mnem_core::objects::{Dtype, Embedding, Node};
use mnem_core::repo::ReadonlyRepo;
use mnem_core::retrieve::{HeuristicEstimator, TokenEstimator, render_node};
use mnem_core::store::{Blockstore, MemoryBlockstore, MemoryOpHeadsStore, OpHeadsStore};

fn stores() -> (Arc<dyn Blockstore>, Arc<dyn OpHeadsStore>) {
    (
        Arc::new(MemoryBlockstore::new()),
        Arc::new(MemoryOpHeadsStore::new()),
    )
}

/// Deterministic 8-dim embedding: hash each whitespace token into one
/// of 8 slots and bump its weight. Just enough to give the cosine
/// ranker something to rank; nowhere near a real embedder.
const DIM: usize = 8;
const MODEL: &str = "demo:keyword-bag-8";

/// FNV-1a 32-bit offset basis (Fowler-Noll-Vo hash, standard seed).
const FNV_OFFSET_BASIS_32: u32 = 2_166_136_261;
/// FNV-1a 32-bit prime (Fowler-Noll-Vo hash, standard multiplier).
const FNV_PRIME_32: u32 = 16_777_619;

fn keyword_bag_embed(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    for tok in text.to_lowercase().split_whitespace() {
        let mut h = FNV_OFFSET_BASIS_32;
        for b in tok.bytes() {
            h = h.wrapping_mul(FNV_PRIME_32).wrapping_add(u32::from(b));
        }
        v[(h as usize) % DIM] += 1.0;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn to_embedding(v: &[f32]) -> Embedding {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    Embedding {
        model: MODEL.to_string(),
        dtype: Dtype::F32,
        dim: v.len() as u32,
        vector: Bytes::from(bytes),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Conversation history the agent has committed over several turns.
    // Each fact is one node with a natural-language summary - the shape
    // most agents already want (one "thought" per memory).
    let facts: &[&str] = &[
        "User's name is Alice and she lives in Berlin.",
        "Alice works as a machine-learning engineer at Globex.",
        "Alice prefers Python for rapid prototyping but ships in Rust.",
        "Alice's partner Bob moved to Paris last month for a new role.",
        "Alice is planning a trip to Tokyo in June.",
        "Alice's hobby is climbing; she visits a bouldering gym weekly.",
        "Alice's favourite colour is olive green.",
        "Alice dislikes overly formal email greetings.",
        "Alice has two cats named Nyx and Kafka.",
        "Alice recently finished reading Anna Karenina.",
        "Alice is allergic to penicillin.",
        "Alice's manager at Globex is Carol.",
        "Alice's timezone is CET.",
        "Alice prefers concise answers and source links.",
        "Alice asked the agent to remind her to renew her passport.",
    ];

    let (bs, ohs) = stores();
    let repo = ReadonlyRepo::init(bs, ohs)?;
    let mut tx = repo.start_transaction();
    for (i, fact) in facts.iter().enumerate() {
        let id = NodeId::from_bytes_raw({
            let mut b = [0u8; 16];
            b[..8].copy_from_slice(&(i as u64).to_be_bytes());
            b
        });
        let emb = to_embedding(&keyword_bag_embed(fact));
        let node = Node::new(id, "Memory").with_summary(*fact);
        let cid = tx.add_node(&node)?;
        tx.set_embedding(cid, emb.model.clone(), emb)?;
    }
    let repo = tx.commit("agent", "seed conversation memory")?;

    // --- Path 1: naive dump. Every memory enters the next prompt. ---
    let est = HeuristicEstimator;
    let naive: String = facts.iter().map(|f| format!("- {f}\n")).collect::<String>();
    let naive_tokens = est.estimate(&naive);

    println!("# Agent context assembly: naive vs mnem budget-aware\n");
    println!("## Path 1 - naive dump (every remembered fact)");
    println!("   items in prompt: {}", facts.len());
    println!("   prompt tokens:   {naive_tokens}\n");

    // --- Path 2: mnem retrieve. The agent is about to answer a user
    //     question about travel; it embeds the question and asks for
    //     the top memories under a 120-token budget.
    let keywords = "Tokyo trip";
    let qvec = keyword_bag_embed(keywords);
    let result = repo
        .retrieve()
        .query_text(keywords.to_string())
        .vector(MODEL, qvec)
        .token_budget(120)
        .execute()?;

    println!("## Path 2 - mnem retrieve (question-aware, budget=120)");
    println!("   keywords:        {keywords:?}");
    println!("   items returned:  {}", result.items.len());
    println!("   tokens used:     {} / {}", result.tokens_used, 120);
    println!("   candidates seen: {}", result.candidates_seen);
    println!("   dropped:         {}\n", result.dropped);
    println!("   packed context:");
    for item in &result.items {
        for line in item.rendered.lines() {
            println!("     {line}");
        }
    }

    // --- Summary row ---
    let savings = if result.tokens_used == 0 {
        f32::INFINITY
    } else {
        (naive_tokens as f32) / (result.tokens_used as f32)
    };
    println!("\n## Savings");
    println!(
        "   mnem fit the right fact(s) in {} tokens vs {} naive  ({:.2}x reduction)",
        result.tokens_used, naive_tokens, savings
    );

    // Quick sanity: render_node alone is a pure function, re-exported
    // so other tools (MCP, CLI) can preview what will be packed without
    // running the full retrieve.
    if let Some(top) = result.items.first() {
        debug_assert_eq!(render_node(&top.node), top.rendered);
    }
    Ok(())
}
