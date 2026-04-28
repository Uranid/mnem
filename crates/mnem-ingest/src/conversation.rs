//! Conversation-export parser.
//!
//! Accepts `ChatGPT`, Claude, and generic `[{role, content, timestamp?}]`
//! JSON payloads and emits one [`Section`] per message.
//!
//! Message order is preserved; `heading` is formatted as `[{role}]` with
//! `depth = 2` so the downstream chunker (`Session`, `Recursive`, or
//! `Paragraph`) can build sensible `section_path` breadcrumbs.
//!
//! Format detection is a cheap JSON schema peek - no full-document
//! deserialisation happens until the format is pinned. Unknown shapes
//! fall through to [`ConversationFormat::Generic`].

use serde::Deserialize;
use serde_json::Value;

use crate::{
    error::Error,
    types::{ConversationFormat, Message, Section},
};

/// Parse a conversation-export JSON blob into a flat `Vec<Section>`.
///
/// One [`Section`] per turn, in source order. Each section has:
///
/// - `heading = Some("[{role}]")`
/// - `depth = 2`
/// - `text = content`
/// - `byte_range = 0..content.len()` (not meaningful for JSON input)
///
/// The wrapper also returns [`ConversationFormat`] implicitly by
/// dispatching on schema shape; callers who need the detected format
/// directly should use [`detect_format`] first.
///
/// # Errors
///
/// Returns [`Error::ParseFailed`] if the bytes are not valid UTF-8 JSON
/// or if no known conversation shape can be extracted.
pub fn parse_conversation(json: &[u8]) -> Result<Vec<Section>, Error> {
    if json.iter().all(u8::is_ascii_whitespace) {
        return Ok(Vec::new());
    }

    let value: Value = serde_json::from_slice(json).map_err(|e| Error::ParseFailed {
        what: "conversation".into(),
        detail: e.to_string(),
    })?;

    let format = detect_format(&value);
    let messages = match format {
        ConversationFormat::ChatGpt => parse_chatgpt(&value)?,
        ConversationFormat::Claude => parse_claude(&value)?,
        ConversationFormat::Generic => parse_generic(&value)?,
    };

    Ok(messages_to_sections(&messages))
}

/// Classify a parsed JSON value by shape.
///
/// The detector only inspects the outermost container plus one level of
/// keys. It is deliberately tolerant: any shape it does not recognise
/// maps to [`ConversationFormat::Generic`], which gracefully rejects
/// nonsense during the actual parse.
#[must_use]
pub fn detect_format(value: &Value) -> ConversationFormat {
    // Claude: {"conversation": [...]}
    if let Some(obj) = value.as_object()
        && obj.contains_key("conversation")
    {
        return ConversationFormat::Claude;
    }

    // ChatGPT: top-level array where the first element has a `mapping`
    // object whose children look like message nodes.
    if let Some(arr) = value.as_array()
        && let Some(first) = arr.first()
        && first.as_object().is_some_and(|o| o.contains_key("mapping"))
    {
        return ConversationFormat::ChatGpt;
    }

    // Everything else: treat as generic `[{role, content, ...}]`.
    ConversationFormat::Generic
}

/// Decode a `ChatGPT` conversation export.
///
/// The schema is `[{"mapping": {msg_id: {"message": {"author":
/// {"role":...}, "content": {"parts": [...]}}, "create_time": f64}}}]`.
/// We walk `mapping` in insertion order (`serde_json` preserves this
/// under the `preserve_order` feature - without it we fall back to
/// `create_time` sorting as a best effort).
fn parse_chatgpt(value: &Value) -> Result<Vec<Message>, Error> {
    let outer = value.as_array().ok_or_else(|| Error::ParseFailed {
        what: "conversation".into(),
        detail: "ChatGPT export must be a top-level array".into(),
    })?;

    let mut messages: Vec<(Option<f64>, Message)> = Vec::new();
    for convo in outer {
        let Some(mapping) = convo.get("mapping").and_then(Value::as_object) else {
            continue;
        };
        for node in mapping.values() {
            let Some(msg) = node.get("message") else {
                continue;
            };
            // Null `message` nodes are the system-root placeholders -
            // skip silently.
            if msg.is_null() {
                continue;
            }
            let role = msg
                .get("author")
                .and_then(|a| a.get("role"))
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();

            let content = extract_chatgpt_parts(msg);
            if content.trim().is_empty() {
                continue;
            }

            let timestamp = msg.get("create_time").and_then(Value::as_f64);
            messages.push((
                timestamp,
                Message {
                    role,
                    content,
                    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                    timestamp: timestamp.map(|t| t.max(0.0) as u64),
                },
            ));
        }
    }

    // Stable sort by create_time when present; ties keep original order.
    messages.sort_by(|a, b| match (a.0, b.0) {
        (Some(x), Some(y)) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
        _ => std::cmp::Ordering::Equal,
    });

    Ok(messages.into_iter().map(|(_, m)| m).collect())
}

/// Flatten a `ChatGPT` `content` node into plain text.
///
/// `content` may be `{"parts": ["...", {...}, ...]}` or the bare string
/// encoding used by older exports. We concatenate string parts with
/// `"\n\n"` separators and drop non-string parts (tool calls, images)
/// rather than inventing stringified placeholders.
fn extract_chatgpt_parts(message: &Value) -> String {
    let Some(content) = message.get("content") else {
        return String::new();
    };

    if let Some(s) = content.as_str() {
        return s.to_string();
    }

    let Some(parts) = content.get("parts").and_then(Value::as_array) else {
        return String::new();
    };

    let mut out = String::new();
    for (idx, part) in parts.iter().enumerate() {
        if let Some(s) = part.as_str() {
            if idx > 0 && !out.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str(s);
        }
    }
    out
}

/// Decode a Claude conversation export.
#[derive(Debug, Deserialize)]
struct ClaudeEnvelope {
    conversation: Vec<ClaudeTurn>,
}

#[derive(Debug, Deserialize)]
struct ClaudeTurn {
    role: String,
    content: String,
    #[serde(default)]
    timestamp: Option<u64>,
}

fn parse_claude(value: &Value) -> Result<Vec<Message>, Error> {
    let env: ClaudeEnvelope =
        serde_json::from_value(value.clone()).map_err(|e| Error::ParseFailed {
            what: "conversation".into(),
            detail: format!("claude shape mismatch: {e}"),
        })?;

    Ok(env
        .conversation
        .into_iter()
        .map(|t| Message {
            role: t.role,
            content: t.content,
            timestamp: t.timestamp,
        })
        .collect())
}

/// Decode a generic `[{role, content, timestamp?}]` array.
#[derive(Debug, Deserialize)]
struct GenericTurn {
    role: String,
    content: String,
    #[serde(default)]
    timestamp: Option<u64>,
}

fn parse_generic(value: &Value) -> Result<Vec<Message>, Error> {
    let turns: Vec<GenericTurn> =
        serde_json::from_value(value.clone()).map_err(|e| Error::ParseFailed {
            what: "conversation".into(),
            detail: format!("generic shape mismatch: {e}"),
        })?;

    Ok(turns
        .into_iter()
        .map(|t| Message {
            role: t.role,
            content: t.content,
            timestamp: t.timestamp,
        })
        .collect())
}

/// Convert a sequence of decoded [`Message`]s into [`Section`]s with
/// deterministic `heading` / `depth` / `byte_range` fields.
fn messages_to_sections(messages: &[Message]) -> Vec<Section> {
    let mut sections = Vec::with_capacity(messages.len());
    let mut cursor = 0usize;
    for m in messages {
        let len = m.content.len();
        sections.push(Section {
            heading: Some(format!("[{}]", m.role)),
            depth: 2,
            text: m.content.clone(),
            byte_range: cursor..(cursor + len),
        });
        cursor = cursor.saturating_add(len).saturating_add(1);
    }
    sections
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHATGPT_FIXTURE: &str = r#"[
      {
        "title": "Demo",
        "mapping": {
          "a": {
            "message": {
              "author": {"role": "user"},
              "content": {"parts": ["Hello"]},
              "create_time": 1.0
            }
          },
          "b": {
            "message": {
              "author": {"role": "assistant"},
              "content": {"parts": ["Hi there", "how can I help?"]},
              "create_time": 2.0
            }
          },
          "root": { "message": null }
        }
      }
    ]"#;

    const CLAUDE_FIXTURE: &str = r#"{
      "conversation": [
        {"role": "user", "content": "What is 2+2?", "timestamp": 1000},
        {"role": "assistant", "content": "Four."}
      ]
    }"#;

    const GENERIC_FIXTURE: &str = r#"[
      {"role": "user", "content": "ping"},
      {"role": "assistant", "content": "pong", "timestamp": 42}
    ]"#;

    #[test]
    fn detects_chatgpt_format() {
        let v: Value = serde_json::from_str(CHATGPT_FIXTURE).unwrap();
        assert_eq!(detect_format(&v), ConversationFormat::ChatGpt);
    }

    #[test]
    fn detects_claude_format() {
        let v: Value = serde_json::from_str(CLAUDE_FIXTURE).unwrap();
        assert_eq!(detect_format(&v), ConversationFormat::Claude);
    }

    #[test]
    fn detects_generic_format() {
        let v: Value = serde_json::from_str(GENERIC_FIXTURE).unwrap();
        assert_eq!(detect_format(&v), ConversationFormat::Generic);
    }

    #[test]
    fn parses_chatgpt_export() {
        let sections = parse_conversation(CHATGPT_FIXTURE.as_bytes()).unwrap();
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].heading.as_deref(), Some("[user]"));
        assert_eq!(sections[0].text, "Hello");
        assert_eq!(sections[1].heading.as_deref(), Some("[assistant]"));
        assert!(sections[1].text.contains("Hi there"));
        assert!(sections[1].text.contains("how can I help?"));
        for s in &sections {
            assert_eq!(s.depth, 2);
        }
    }

    #[test]
    fn parses_claude_export() {
        let sections = parse_conversation(CLAUDE_FIXTURE.as_bytes()).unwrap();
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].heading.as_deref(), Some("[user]"));
        assert_eq!(sections[0].text, "What is 2+2?");
        assert_eq!(sections[1].heading.as_deref(), Some("[assistant]"));
        assert_eq!(sections[1].text, "Four.");
    }

    #[test]
    fn parses_generic_export() {
        let sections = parse_conversation(GENERIC_FIXTURE.as_bytes()).unwrap();
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].text, "ping");
        assert_eq!(sections[1].text, "pong");
    }

    #[test]
    fn empty_input_yields_no_sections() {
        assert!(parse_conversation(b"").unwrap().is_empty());
        assert!(parse_conversation(b"   \n  ").unwrap().is_empty());
    }

    #[test]
    fn malformed_json_is_error_not_panic() {
        let result = parse_conversation(b"{not json");
        assert!(matches!(result, Err(Error::ParseFailed { .. })));
    }

    #[test]
    fn chatgpt_null_message_is_skipped() {
        // The `root` node with `message: null` must not appear in output.
        let sections = parse_conversation(CHATGPT_FIXTURE.as_bytes()).unwrap();
        assert!(
            sections.iter().all(|s| !s.text.trim().is_empty()),
            "null-message placeholders must be skipped",
        );
    }

    #[test]
    fn snapshot_generic_parse() {
        let sections = parse_conversation(GENERIC_FIXTURE.as_bytes()).unwrap();
        insta::assert_yaml_snapshot!("conversation_generic", sections);
    }

    #[test]
    fn snapshot_claude_parse() {
        let sections = parse_conversation(CLAUDE_FIXTURE.as_bytes()).unwrap();
        insta::assert_yaml_snapshot!("conversation_claude", sections);
    }
}
