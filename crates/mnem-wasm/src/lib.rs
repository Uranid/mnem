#![allow(missing_docs)]
#![allow(unreachable_pub)]

use mnem_core::store::{MemoryBlockstore, MemoryOpHeadsStore};
use mnem_core::ReadonlyRepo;
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use js_sys::{Array, Object, Reflect};

struct LogEntry {
    hash: String,
    label: String,
    summary: String,
}

/// In-browser mnem knowledge graph backed by MemoryBlockstore.
///
/// Wraps a real mnem ReadonlyRepo/Transaction cycle with a parallel
/// Vec<LogEntry> for fast log/retrieve display (no Prolly tree walk needed).
#[wasm_bindgen]
pub struct MnemGraph {
    repo: ReadonlyRepo,
    log: Vec<LogEntry>,
}

#[wasm_bindgen]
impl MnemGraph {
    /// Create and initialize a fresh in-memory knowledge graph.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<MnemGraph, JsError> {
        console_error_panic_hook::set_once();
        let bs = Arc::new(MemoryBlockstore::new());
        let oh = Arc::new(MemoryOpHeadsStore::new());
        let repo = ReadonlyRepo::init(bs, oh)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(MnemGraph { repo, log: vec![] })
    }

    /// Commit a node and return the first 7 hex chars of its UUID as a hash.
    pub fn commit_node(&mut self, summary: &str, label: &str) -> Result<String, JsError> {
        let mut tx = self.repo.start_transaction();
        let node_id = tx
            .commit_memory(label, summary, std::iter::empty())
            .map_err(|e| JsError::new(&e.to_string()))?;
        self.repo = tx
            .commit("demo", "add node")
            .map_err(|e| JsError::new(&e.to_string()))?;
        let uuid_str = node_id.to_uuid_string();
        let hash: String = uuid_str.replace('-', "").chars().take(7).collect();
        self.log.push(LogEntry {
            hash: hash.clone(),
            label: label.to_string(),
            summary: summary.to_string(),
        });
        Ok(hash)
    }

    /// Keyword search over stored nodes; returns up to 3 hits as a JS Array of objects.
    pub fn retrieve_nodes(&self, query: &str) -> JsValue {
        let terms: Vec<String> = query
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        let hits: Vec<&LogEntry> = self
            .log
            .iter()
            .filter(|e| {
                let sl = e.summary.to_lowercase();
                terms.iter().any(|t| sl.contains(t.as_str()))
            })
            .rev()
            .take(3)
            .collect();
        let arr = Array::new();
        for (i, e) in hits.iter().enumerate() {
            let obj = Object::new();
            let score = 0.97 - (i as f64 * 0.09);
            let _ = Reflect::set(&obj, &"hash".into(), &e.hash.as_str().into());
            let _ = Reflect::set(&obj, &"label".into(), &e.label.as_str().into());
            let _ = Reflect::set(&obj, &"summary".into(), &e.summary.as_str().into());
            let _ = Reflect::set(&obj, &"score".into(), &score.into());
            arr.push(&obj);
        }
        arr.into()
    }

    /// Return the most recent `limit` nodes as a JS Array of objects.
    pub fn log_nodes(&self, limit: usize) -> JsValue {
        let arr = Array::new();
        for e in self.log.iter().rev().take(limit) {
            let obj = Object::new();
            let _ = Reflect::set(&obj, &"hash".into(), &e.hash.as_str().into());
            let _ = Reflect::set(&obj, &"label".into(), &e.label.as_str().into());
            let _ = Reflect::set(&obj, &"summary".into(), &e.summary.as_str().into());
            arr.push(&obj);
        }
        arr.into()
    }

    /// Total number of committed nodes.
    pub fn node_count(&self) -> usize {
        self.log.len()
    }
}
