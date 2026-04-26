//! JSON-RPC 2.0 and minimal MCP wire types.
//!
//! Hand-rolled because the spec is small and the official Rust SDK
//! pulls in a large tokio dep graph we don't need for stdio. If we
//! grow to HTTP transport later we can layer `rmcp` on top without
//! changing these types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// MCP protocol version this server implements. Newer clients that
/// negotiate a different version are still served (we log and proceed);
/// breaking protocol changes would warrant bumping this constant.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// JSON-RPC 2.0 request.
#[derive(Deserialize, Debug)]
pub(crate) struct Request {
    #[serde(default)]
    pub jsonrpc: String,
    /// Missing id = notification; no response sent.
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

/// JSON-RPC 2.0 response.
#[derive(Serialize, Debug)]
pub(crate) struct Response {
    pub jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<RpcError>,
}

/// JSON-RPC 2.0 error object.
#[derive(Serialize, Debug)]
pub(crate) struct RpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl Response {
    pub(crate) fn ok(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    pub(crate) fn err(id: Option<Value>, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(RpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

/// The five JSON-RPC 2.0 error codes (§5.1). We export all five even
/// though the current server only returns `PARSE_ERROR`,
/// `INVALID_REQUEST`, and `METHOD_NOT_FOUND`; the other two are the
/// natural exit codes for "malformed tool arguments" and "handler
/// panicked", and they're referenced by future error branches as the
/// surface grows.
pub(crate) mod error_code {
    /// Invalid JSON was received by the server.
    pub(crate) const PARSE_ERROR: i32 = -32700;
    /// The JSON sent is not a valid Request object.
    pub(crate) const INVALID_REQUEST: i32 = -32600;
    /// The requested method does not exist / is not available.
    pub(crate) const METHOD_NOT_FOUND: i32 = -32601;
    /// Invalid method parameter(s).
    #[allow(
        dead_code,
        reason = "standard JSON-RPC code; referenced by future error paths"
    )]
    pub(crate) const INVALID_PARAMS: i32 = -32602;
    /// Internal JSON-RPC error.
    #[allow(
        dead_code,
        reason = "standard JSON-RPC code; referenced by future error paths"
    )]
    pub(crate) const INTERNAL_ERROR: i32 = -32603;
}

/// MCP tool definition, as emitted in `tools/list` responses. Matches
/// the wire format the MCP spec defines (name, description, JSON
/// Schema input).
#[derive(Serialize, Debug)]
pub struct ToolDef {
    /// Tool name (exact string the client uses in `tools/call`).
    pub name: &'static str,
    /// Human-readable description surfaced to the model.
    pub description: &'static str,
    /// JSON Schema describing the `arguments` object for this tool.
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}
