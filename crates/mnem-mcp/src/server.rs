//! Server state + JSON-RPC request dispatch.
//!
//! State model: the server holds `Arc<dyn Blockstore>` and
//! `Arc<dyn OpHeadsStore>` lazily initialised on first use. Each tool
//! call loads a fresh `ReadonlyRepo` via `ReadonlyRepo::open` - this
//! transparently runs the 3-way view/commit merge if concurrent
//! op-heads exist, so readers always see a coherent state.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use mnem_backend_redb::open_or_init;
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{Blockstore, OpHeadsStore};
use serde_json::{Value, json};

use crate::protocol::{MCP_PROTOCOL_VERSION, Request, Response, error_code};
use crate::tools;

/// Parsed-line JSON-RPC dispatcher. Holds the lazily initialised
/// blockstore / op-heads-store pair and the `MNEM_BENCH` gate.
pub struct Server {
    repo_path: PathBuf,
    bs: Option<Arc<dyn Blockstore>>,
    ohs: Option<Arc<dyn OpHeadsStore>>,
    /// Whether the server accepts caller-supplied `label` values on
    /// ingest and `label` filters on retrieve.
    ///
    /// **Defaults to `true`** as of the 2026-04-25 audit fix (G3): the
    /// previous default-off behaviour silently coerced every node's
    /// `ntype` to `Node::DEFAULT_NTYPE`, breaking the typed knowledge
    /// graph for every customer who didn't know to set `MNEM_BENCH=1`.
    ///
    /// Resolution order at startup:
    ///   1. `MNEM_LABELS` if set: `0`/`false`/`no`/`off` → false,
    ///      anything else → true. Preferred name.
    ///   2. else `MNEM_BENCH` if set: same parsing. Legacy name kept
    ///      for benchmark harnesses.
    ///   3. else: `true`.
    ///
    /// Parity with `mnem-http`'s `AppState::allow_labels`.
    pub allow_labels: bool,
}

impl Server {
    /// Build a server bound to `repo_path`. Stores are opened lazily on
    /// the first tool call that requires them.
    pub fn new(repo_path: PathBuf) -> Self {
        Self {
            repo_path,
            bs: None,
            ohs: None,
            allow_labels: Self::resolve_allow_labels_from_env(),
        }
    }

    /// Resolve the labels-on flag from the process environment.
    ///
    /// Default: `true`. `MNEM_LABELS` is the preferred opt-out (`0`/
    /// `false`/`no`/`off`); `MNEM_BENCH` is kept as a legacy alias for
    /// benchmark harnesses that already set it.
    #[must_use]
    pub(crate) fn resolve_allow_labels_from_env() -> bool {
        Self::resolve_allow_labels_with(|k| std::env::var(k).ok())
    }

    /// Pure resolver injectable with an arbitrary env-var getter.
    /// Lets tests cover the (MNEM_LABELS, MNEM_BENCH) precedence
    /// matrix without mutating process-global state - the crate is
    /// `#![forbid(unsafe_code)]`, so `std::env::set_var` (which is
    /// `unsafe` since 2024 edition) is not available.
    #[must_use]
    pub(crate) fn resolve_allow_labels_with<F>(get: F) -> bool
    where
        F: Fn(&str) -> Option<String>,
    {
        if let Some(v) = get("MNEM_LABELS") {
            return Self::parse_truthy_env(Some(&v));
        }
        if let Some(v) = get("MNEM_BENCH") {
            return Self::parse_truthy_env(Some(&v));
        }
        true
    }

    /// Pure parser for a truthy/falsy env-var value.
    ///
    /// `None` (unset) is `false` (caller decides what unset means).
    /// Falsy strings (`"0"`, `"false"`, `"no"`, `"off"`, empty, all
    /// case-insensitive) parse `false`. Anything else parses `true`.
    #[must_use]
    pub(crate) fn parse_truthy_env(val: Option<&str>) -> bool {
        match val {
            None => false,
            Some(s) => {
                let t = s.trim();
                if t.is_empty() {
                    return false;
                }
                let l = t.to_ascii_lowercase();
                !matches!(l.as_str(), "0" | "false" | "no" | "off")
            }
        }
    }

    /// Read-only accessor for the backing repo path. Used by handlers
    /// that need to resolve sibling config files (e.g.
    /// `mnem_community_summarize` reading `<repo>/config.toml` for
    /// the `[embed]` section when no `MNEM_EMBED_*` env vars are
    /// set).
    #[cfg(feature = "summarize")]
    pub(crate) fn repo_path(&self) -> &std::path::Path {
        &self.repo_path
    }

    /// Ensure the backing redb file exists and the store handles are
    /// cached. Safe to call repeatedly; no-op after first success.
    fn ensure_stores(&mut self) -> anyhow::Result<()> {
        if self.bs.is_some() {
            return Ok(());
        }
        std::fs::create_dir_all(&self.repo_path)?;
        let redb_path = self.repo_path.join("repo.redb");
        let (bs, ohs, _file) = open_or_init(&redb_path)?;
        self.bs = Some(bs);
        self.ohs = Some(ohs);
        Ok(())
    }

    /// Load a fresh `ReadonlyRepo` for this tool call. If the store is
    /// uninitialised (no root op yet), creates a root op so the tool
    /// has something to talk to. Any *other* error (store corruption,
    /// codec failure, broken op-DAG) is propagated instead of silently
    /// auto-reinitialising - an agent should see "something's wrong",
    /// not find itself looking at an empty repo.
    pub(crate) fn load_repo(&mut self) -> anyhow::Result<ReadonlyRepo> {
        self.ensure_stores()?;
        let bs = self.bs.as_ref().unwrap().clone();
        let ohs = self.ohs.as_ref().unwrap().clone();
        match ReadonlyRepo::open(bs.clone(), ohs.clone()) {
            Ok(r) => Ok(r),
            Err(e) if e.is_uninitialized() => ReadonlyRepo::init(bs, ohs).map_err(Into::into),
            Err(e) => Err(e.into()),
        }
    }

    /// Accessor for tools that need direct access to the underlying
    /// stores (e.g. to start a transaction).
    pub(crate) fn stores(
        &mut self,
    ) -> anyhow::Result<(Arc<dyn Blockstore>, Arc<dyn OpHeadsStore>)> {
        self.ensure_stores()?;
        Ok((
            self.bs.as_ref().unwrap().clone(),
            self.ohs.as_ref().unwrap().clone(),
        ))
    }

    /// Parse one line of stdin; return the response line (or `None`
    /// for a notification, which produces no output).
    pub fn handle_line(&mut self, line: &str) -> Option<String> {
        let req: Request = match serde_json::from_str(line) {
            Ok(r) => r,
            Err(e) => {
                let resp =
                    Response::err(None, error_code::PARSE_ERROR, format!("parse error: {e}"));
                return Some(serde_json::to_string(&resp).unwrap());
            }
        };

        // JSON-RPC 2.0 §4.1: the `jsonrpc` field MUST be exactly the
        // string "2.0". Clients that send anything else get a clean
        // INVALID_REQUEST rather than a silent success.
        if req.jsonrpc != "2.0" {
            let resp = Response::err(
                req.id.clone(),
                error_code::INVALID_REQUEST,
                format!(
                    "invalid jsonrpc field: expected '2.0', got {:?}",
                    req.jsonrpc
                ),
            );
            return Some(serde_json::to_string(&resp).unwrap());
        }

        // Notifications (no id) never produce a response body.
        let is_notification = req.id.is_none();
        let id = req.id.clone();

        let resp = match req.method.as_str() {
            "initialize" => Self::handle_initialize(id),
            "tools/list" => self.handle_tools_list(id),
            "tools/call" => self.handle_tools_call(id, req.params),
            // Spec utility: either peer MAY send `ping` and the receiver
            // MUST respond with an empty result. Claude Desktop and other
            // hosts use this as a liveness check.
            "ping" => Response::ok(id, json!({})),
            // Notifications we accept silently. The outer `is_notification`
            // check would drop the response anyway, but returning `None`
            // here avoids constructing one.
            "notifications/initialized" | "notifications/cancelled" => return None,
            other => Response::err(
                id,
                error_code::METHOD_NOT_FOUND,
                format!("method not found: {other}"),
            ),
        };

        if is_notification {
            None
        } else {
            Some(serde_json::to_string(&resp).unwrap())
        }
    }

    fn handle_initialize(id: Option<Value>) -> Response {
        Response::ok(
            id,
            json!({
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "mnem-mcp",
                    "version": env!("CARGO_PKG_VERSION"),
                }
            }),
        )
    }

    fn handle_tools_list(&self, id: Option<Value>) -> Response {
        Response::ok(id, json!({ "tools": tools::all_tools(self.allow_labels) }))
    }

    fn handle_tools_call(&mut self, id: Option<Value>, params: Value) -> Response {
        // Missing or non-string `name` is an RPC-level INVALID_PARAMS,
        // not a tool error - the call never reaches dispatch.
        let Some(name) = params.get("name").and_then(Value::as_str) else {
            return Response::err(
                id,
                error_code::INVALID_PARAMS,
                "tools/call: `name` field is missing or not a string",
            );
        };
        let name = name.to_string();
        let args = params.get("arguments").cloned().unwrap_or(Value::Null);

        let start = Instant::now();
        let outcome = tools::dispatch(self, &name, args);
        let latency_micros = start.elapsed().as_micros() as u64;

        let text = match outcome {
            Ok(v) => v,
            Err(e) => {
                // Tool-level failure: surface as MCP tool-error content,
                // not a JSON-RPC protocol error. `_meta` schema is kept
                // parity-stable with the success path so clients can
                // bucket by shape without a success/error branch.
                let err_text = format!("mnem_mcp tool error: {e}");
                let bytes = err_text.len();
                let tokens_estimate = bytes / 4;
                return Response::ok(
                    id,
                    json!({
                        "content": [{ "type": "text", "text": err_text }],
                        "isError": true,
                        "_meta": {
                            "bytes": bytes,
                            "latency_micros": latency_micros,
                            "tokens_estimate": tokens_estimate
                        }
                    }),
                );
            }
        };

        let bytes = text.len();
        // Coarse GPT-style approximation: ~4 bytes per token on
        // English/code mixes. A real tokenizer (via the `TokenEstimator` trait)
        // the token-budget query primitive.
        let tokens_estimate = bytes / 4;

        Response::ok(
            id,
            json!({
                "content": [{ "type": "text", "text": text }],
                "_meta": {
                    "bytes": bytes,
                    "latency_micros": latency_micros,
                    "tokens_estimate": tokens_estimate
                }
            }),
        )
    }
}

#[cfg(test)]
mod env_parse_tests {
    use super::Server;

    #[test]
    fn unset_value_parses_false() {
        // `parse_truthy_env(None)` is the unset-string sentinel, not the
        // unset-env sentinel. The full env resolution is exercised in
        // `resolve_allow_labels_default_is_true_when_both_unset`.
        assert!(!Server::parse_truthy_env(None));
    }

    #[test]
    fn falsy_strings_parse_false() {
        for v in [
            "", "0", "false", "FALSE", "False", "no", "No", "NO", "off", "Off", "OFF", "  ", "  0 ",
        ] {
            assert!(
                !Server::parse_truthy_env(Some(v)),
                "expected `{v:?}` to parse false"
            );
        }
    }

    #[test]
    fn truthy_strings_parse_true() {
        for v in ["1", "true", "yes", "on", "YES", "benchmark", "anything"] {
            assert!(
                Server::parse_truthy_env(Some(v)),
                "expected `{v:?}` to parse true"
            );
        }
    }

    #[test]
    fn resolve_allow_labels_default_is_true_when_both_unset() {
        // audit-2026-04-25 G3 fix: the default is now ON. Pure
        // resolver, no env mutation: `#![forbid(unsafe_code)]` rules
        // out `std::env::set_var` (unsafe since 2024 edition), so
        // we exercise the precedence matrix via the injected getter.
        assert!(Server::resolve_allow_labels_with(|_| None));
    }

    #[test]
    fn resolve_allow_labels_honours_explicit_off_via_mnem_labels() {
        let off = Server::resolve_allow_labels_with(|k| match k {
            "MNEM_LABELS" => Some("0".into()),
            _ => None,
        });
        assert!(!off);
    }

    #[test]
    fn resolve_allow_labels_honours_legacy_mnem_bench_off() {
        let off = Server::resolve_allow_labels_with(|k| match k {
            "MNEM_BENCH" => Some("0".into()),
            _ => None,
        });
        assert!(!off);
    }

    #[test]
    fn mnem_labels_takes_precedence_over_mnem_bench() {
        // LABELS=1 wins over BENCH=0.
        let on = Server::resolve_allow_labels_with(|k| match k {
            "MNEM_LABELS" => Some("1".into()),
            "MNEM_BENCH" => Some("0".into()),
            _ => None,
        });
        assert!(on);
        // LABELS=0 wins over BENCH=1.
        let off = Server::resolve_allow_labels_with(|k| match k {
            "MNEM_LABELS" => Some("0".into()),
            "MNEM_BENCH" => Some("1".into()),
            _ => None,
        });
        assert!(!off);
    }

    #[test]
    fn legacy_mnem_bench_alone_still_enables() {
        // No MNEM_LABELS, MNEM_BENCH=1 → on (back-compat with
        // benchmark harnesses already setting MNEM_BENCH).
        let on = Server::resolve_allow_labels_with(|k| match k {
            "MNEM_BENCH" => Some("1".into()),
            _ => None,
        });
        assert!(on);
    }
}
