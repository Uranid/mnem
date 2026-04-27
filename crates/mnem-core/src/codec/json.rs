//! Canonical `serde_json::Value` -> [`Ipld`] conversion for untrusted input.
//!
//! Three surfaces feed untrusted JSON into mnem: the CLI (`mnem ...
//! --prop key=value`), `mnem-http` (request bodies on `/v1/*`), and
//! `mnem-mcp` (tool-call `arguments` objects). Before `json_to_ipld`
//! lived here, each of those crates carried its own near-identical
//! implementation, each with its own copy of [`IPLD_MAX_DEPTH`], its
//! own `u64 > i64::MAX` rejection path, and its own error type
//! (`anyhow::Result`, `Result<_, String>`, `anyhow::Result`). Every
//! future hardening change had to be replicated across three files,
//! and the three were already out-of-sync in subtle ways (error
//! message wording drift, different comment wording).
//!
//! This module is the canonical implementation. All three callers
//! re-export [`json_to_ipld`] and adapt [`JsonIpldError`] to their
//! local error boundary:
//!
//! - `mnem-cli`: `?` through `anyhow::Error` (the library `Display`
//!   impl threads directly).
//! - `mnem-http`: `map_err` to `mnem_http::error::Error::BadRequest` so
//!   a malformed JSON body returns HTTP 400 with a specific reason.
//! - `mnem-mcp`: `map_err` to an MCP `error.invalid_params` response
//!   carrying the same `Display` string as a structured field.
//!
//! ## Hardening
//!
//! Two concrete attacker-controlled inputs motivate this module's
//! shape:
//!
//! 1. **Deeply-nested arrays/objects.** A stock recursive-descent
//!    converter stack-overflows on `[[[[[[[...]]]]]]]` with a few
//!    thousand levels of nesting. [`IPLD_MAX_DEPTH`] caps the
//!    traversal at 64 levels, matching [`crate::codec::dagcbor::WALK_IPLD_MAX_DEPTH`]
//!    so a payload cannot pass this check and then fail further down
//!    the pipeline.
//! 2. **Unsigned ids above `i64::MAX`.** Silently demoting such a
//!    value to [`Ipld::Float`] loses precision above 2^53 (a 19-digit
//!    id becomes a rounded double). Reject instead: callers that
//!    really need a 64-bit unsigned id must send it as a string.

use std::collections::BTreeMap;

use ipld_core::ipld::Ipld;
use serde_json::Value;
use thiserror::Error;

/// Maximum depth of nested JSON objects / arrays [`json_to_ipld`]
/// will walk. Beyond this, the conversion returns an error rather
/// than recursing. Picked at 64 because legitimate agent-memory
/// props rarely nest past ~6, while a malicious payload can cheaply
/// ship arbitrary depth and stack-overflow the process.
///
/// This MUST stay equal to
/// [`crate::codec::dagcbor::WALK_IPLD_MAX_DEPTH`]: a payload that
/// clears the input-layer cap must also clear the decode-layer cap
/// so there is no gap between "accepted on the wire" and "decodable
/// after a round-trip through DAG-CBOR".
pub const IPLD_MAX_DEPTH: usize = 64;

/// Failure modes for [`json_to_ipld`].
///
/// Deliberately coarse-grained; each variant carries enough detail
/// for a caller to render a user-facing error without string-parsing
/// the `Display` output.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum JsonIpldError {
    /// The input exceeded [`IPLD_MAX_DEPTH`] levels of nesting.
    #[error("json_to_ipld: nesting exceeds depth cap of {cap}")]
    DepthExceeded {
        /// The cap that was exceeded; always equals [`IPLD_MAX_DEPTH`].
        cap: usize,
    },
    /// A JSON `Number` was an unsigned integer greater than
    /// `i64::MAX`. Such values cannot round-trip through
    /// [`Ipld::Integer`] (which is `i128` but DAG-CBOR encodes only
    /// `i64` / `u64`) without ambiguity; the old "demote to
    /// [`Ipld::Float`]" path silently lost precision above 2^53.
    #[error("json_to_ipld: unsigned integer {value} exceeds i64::MAX; send as a string if id-like")]
    UnsignedOverflow {
        /// The rejected value, rendered as it appeared in the input.
        value: String,
    },
    /// A JSON `Number` was neither an `i64`, a `u64`, nor a finite
    /// `f64`. In practice this cannot happen from `serde_json` today
    /// (the `Number` variants exhaust the space) but is kept as a
    /// defensive catch-all.
    #[error("json_to_ipld: unsupported JSON number {value}")]
    UnsupportedNumber {
        /// The rejected value, rendered as it appeared in the input.
        value: String,
    },
}

/// Convert a [`serde_json::Value`] into an [`Ipld`] value, rejecting
/// deeply-nested or precision-losing inputs.
///
/// # Errors
///
/// Returns [`JsonIpldError::DepthExceeded`] if the input nests past
/// [`IPLD_MAX_DEPTH`]; [`JsonIpldError::UnsignedOverflow`] if a
/// numeric field is `> i64::MAX`; [`JsonIpldError::UnsupportedNumber`]
/// for any other unhandled numeric shape.
pub fn json_to_ipld(v: &Value) -> Result<Ipld, JsonIpldError> {
    json_to_ipld_at(v, 0)
}

fn json_to_ipld_at(v: &Value, depth: usize) -> Result<Ipld, JsonIpldError> {
    if depth >= IPLD_MAX_DEPTH {
        return Err(JsonIpldError::DepthExceeded {
            cap: IPLD_MAX_DEPTH,
        });
    }
    Ok(match v {
        Value::Null => Ipld::Null,
        Value::Bool(b) => Ipld::Bool(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ipld::Integer(i128::from(i))
            } else if n.is_u64() {
                return Err(JsonIpldError::UnsignedOverflow {
                    value: n.to_string(),
                });
            } else if let Some(f) = n.as_f64() {
                Ipld::Float(f)
            } else {
                return Err(JsonIpldError::UnsupportedNumber {
                    value: n.to_string(),
                });
            }
        }
        Value::String(s) => Ipld::String(s.clone()),
        Value::Array(xs) => Ipld::List(
            xs.iter()
                .map(|x| json_to_ipld_at(x, depth + 1))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Value::Object(m) => {
            let mut out = BTreeMap::new();
            for (k, v) in m {
                out.insert(k.clone(), json_to_ipld_at(v, depth + 1)?);
            }
            Ipld::Map(out)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn null_bool_string_roundtrip() {
        assert_eq!(json_to_ipld(&Value::Null).unwrap(), Ipld::Null);
        assert_eq!(json_to_ipld(&json!(true)).unwrap(), Ipld::Bool(true));
        assert_eq!(
            json_to_ipld(&json!("hello")).unwrap(),
            Ipld::String("hello".to_string())
        );
    }

    #[test]
    fn i64_as_integer() {
        assert_eq!(
            json_to_ipld(&json!(42_i64)).unwrap(),
            Ipld::Integer(42_i128)
        );
        assert_eq!(
            json_to_ipld(&json!(i64::MIN)).unwrap(),
            Ipld::Integer(i128::from(i64::MIN))
        );
        assert_eq!(
            json_to_ipld(&json!(i64::MAX)).unwrap(),
            Ipld::Integer(i128::from(i64::MAX))
        );
    }

    #[test]
    fn u64_gt_i64_max_rejected() {
        let err = json_to_ipld(&json!(u64::MAX)).unwrap_err();
        assert!(matches!(err, JsonIpldError::UnsignedOverflow { .. }));
    }

    #[test]
    fn float_preserved() {
        assert_eq!(json_to_ipld(&json!(1.5_f64)).unwrap(), Ipld::Float(1.5));
    }

    #[test]
    fn deeply_nested_rejected() {
        // Build 128 levels of array nesting - well past the 64 cap.
        let mut v = Value::Null;
        for _ in 0..128 {
            v = Value::Array(vec![v]);
        }
        let err = json_to_ipld(&v).unwrap_err();
        assert!(matches!(
            err,
            JsonIpldError::DepthExceeded {
                cap: IPLD_MAX_DEPTH
            }
        ));
    }

    #[test]
    fn nested_map_respects_cap() {
        // 65 nested objects: {a: {a: {a: ... {a: null}}}}
        let mut v = Value::Null;
        for _ in 0..65 {
            let mut m = serde_json::Map::new();
            m.insert("a".into(), v);
            v = Value::Object(m);
        }
        let err = json_to_ipld(&v).unwrap_err();
        assert!(matches!(err, JsonIpldError::DepthExceeded { .. }));
    }

    #[test]
    fn shallow_nesting_ok() {
        // 10 levels of nesting: comfortably under the 64 cap.
        let mut v = Value::Null;
        for _ in 0..10 {
            v = Value::Array(vec![v]);
        }
        let _ = json_to_ipld(&v).unwrap();
    }

    #[test]
    fn array_and_object_mixed() {
        let v = json!({
            "name": "a",
            "xs": [1, 2, 3],
            "meta": { "kind": "note", "active": true }
        });
        let out = json_to_ipld(&v).unwrap();
        let Ipld::Map(m) = out else {
            panic!("expected map");
        };
        assert!(m.contains_key("name"));
        assert!(m.contains_key("xs"));
        assert!(m.contains_key("meta"));
    }
}
