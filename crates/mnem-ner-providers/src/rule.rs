//! Rule-based NER provider: capitalized-phrase heuristic for Person/Organization.
//!
//! This is the logic previously embedded in `mnem-ingest/src/extract.rs`.
//! Confidence 0.60 for all emitted entities.

use crate::provider::{NamedEntity, NerProvider};

/// Common leading words that should not start an entity span.
const COMMON_DENYLIST: &[&str] = &[
    "The", "This", "That", "These", "Those", "A", "An", "And", "Or", "But", "If", "In", "On", "At",
    "To", "From", "With", "By", "For", "Of", "As", "Is", "Was", "Are", "Were", "Be", "Been",
    "Being", "I", "We", "You", "He", "She", "It", "They", "My", "Our", "Your", "His", "Her",
    "Their", "Mr", "Mrs", "Ms", "Dr",
];

/// Suffix tokens that promote a capitalized run to Organization.
const ORG_SUFFIXES: &[&str] = &[
    "Inc",
    "Inc.",
    "LLC",
    "Ltd",
    "Ltd.",
    "Corp",
    "Corp.",
    "Corporation",
    "Company",
    "Co",
    "Co.",
    "GmbH",
    "AG",
    "SA",
    "BV",
    "PLC",
];

/// Rule-based [`NerProvider`] that detects Person and Organization spans
/// via a capitalized-phrase heuristic (two or more consecutive capitalized
/// tokens, denylist filtered, org-suffix promoted).
#[derive(Debug, Default, Clone)]
pub struct RuleNer;

impl NerProvider for RuleNer {
    fn extract(&self, text: &str) -> Vec<NamedEntity> {
        let mut out = Vec::new();
        for (label, range) in capitalized_phrases(text) {
            let slice = match text.get(range.clone()) {
                Some(s) if !s.is_empty() => s.to_string(),
                _ => continue,
            };
            out.push(NamedEntity {
                text: slice,
                label,
                byte_start: range.start,
                byte_end: range.end,
                confidence: 0.60,
            });
        }
        out
    }

    fn provider_id(&self) -> &str {
        "rule"
    }
}

fn capitalized_phrases(text: &str) -> Vec<(String, std::ops::Range<usize>)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    let len = bytes.len();

    while i < len {
        if !is_ascii_upper(bytes[i]) {
            i += 1;
            continue;
        }
        let start = i;
        let mut last_end = i;
        let mut token_count = 0;
        let mut saw_org_suffix = false;

        while i < len && is_ascii_upper(bytes[i]) {
            let tok_start = i;
            i += 1;
            while i < len && (bytes[i].is_ascii_alphabetic() || bytes[i] == b'.') {
                i += 1;
            }
            let tok = &text[tok_start..i];
            if COMMON_DENYLIST.contains(&tok) && token_count == 0 {
                token_count = 0;
                last_end = i;
                break;
            }
            token_count += 1;
            last_end = i;
            if ORG_SUFFIXES.contains(&tok) {
                saw_org_suffix = true;
            }
            if i < len && bytes[i] == b' ' && i + 1 < len && is_ascii_upper(bytes[i + 1]) {
                i += 1;
                continue;
            }
            break;
        }

        if token_count >= 2 {
            let label = if saw_org_suffix {
                "Entity:Organization".to_string()
            } else {
                "Entity:Person".to_string()
            };
            out.push((label, start..last_end));
        }
        while i < len && !is_ascii_upper(bytes[i]) {
            i += 1;
        }
    }
    out
}

const fn is_ascii_upper(b: u8) -> bool {
    b.is_ascii_uppercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_person() {
        let ner = RuleNer;
        let ents = ner.extract("Alice Johnson met Bob Lee at the lobby.");
        assert!(
            ents.iter()
                .any(|e| e.label == "Entity:Person" && e.text == "Alice Johnson"),
            "got: {ents:?}"
        );
        assert!(
            ents.iter()
                .any(|e| e.label == "Entity:Person" && e.text == "Bob Lee"),
            "got: {ents:?}"
        );
    }

    #[test]
    fn detects_org_suffix() {
        let ner = RuleNer;
        let ents = ner.extract("Acme Corp and Foo Inc signed the deal.");
        assert!(
            ents.iter()
                .any(|e| e.label == "Entity:Organization" && e.text == "Acme Corp"),
            "got: {ents:?}"
        );
    }

    #[test]
    fn rejects_single_token() {
        let ner = RuleNer;
        let ents = ner.extract("Alice then left.");
        assert!(
            !ents.iter().any(|e| e.label == "Entity:Person"),
            "got: {ents:?}"
        );
    }

    #[test]
    fn null_ner_emits_nothing() {
        use crate::null::NullNer;
        let ner = NullNer;
        assert!(ner.extract("Alice Johnson founded Acme Corp.").is_empty());
    }
}
