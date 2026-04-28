//! Rule-based entity + relation extraction over parsed [`Section`]s.
//!
//! Phase-B5c ships a deterministic, dependency-light NER baseline. Three
//! layers cooperate:
//!
//! 1. **Regex** for structured surface forms (URLs, emails, ISO-8601 dates,
//! "`Mon DD, YYYY`" dates). Confidence `0.95`.
//! 2. **Aho-Corasick** for a caller-supplied keyword list. Confidence `0.90`.
//! 3. **Capitalized-phrase heuristic** for Person / Organization (2+
//! consecutive capitalized tokens, denylist filtered). Confidence `0.60`.
//!
//! Relations are proximity-based: two entity spans whose start positions
//! are within `window_tokens` of each other in the same [`Section`] get a
//! candidate `"co_occurs_with"` edge (confidence `0.40`). A lightweight
//! verb-between check promotes that to `"acts_on"` (confidence `0.50`)
//! when a token like `"joined"`, `"founded"`, `"acquired"`, `"owns"`, or
//! `"hired"` sits between the two spans.
//!
//! LLM-driven extraction is explicitly deferred to Phase-B5e. The public
//! [`Extractor`] trait is the extension point; the default
//! [`RuleExtractor`] ships here so every downstream call-site has a
//! working implementation on day one.
//!
//! Every public item is documented - the crate carries
//! `#![deny(missing_docs)]` and `#![forbid(unsafe_code)]`.

use std::ops::Range;

use aho_corasick::{AhoCorasick, MatchKind};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::types::{ExtractorConfig, Section};

// ---------------- Types ----------------

/// Coarse category assigned to an [`EntitySpan`].
///
/// The set is intentionally small; richer ontologies ride in later
/// waves on top of an LLM or gazetteer-driven extractor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityKind {
    /// A natural person ("Alice", "Barack Obama"). Produced by the
    /// capitalized-phrase heuristic; never by regex.
    Person,
    /// An organization (`"Acme Corp"`, `"OpenAI"`). Produced by the
    /// capitalized-phrase heuristic with an org-hint suffix list.
    Organization,
    /// A place ("Berlin", "New York"). Reserved - the baseline does not
    /// emit these, but the variant exists so the extension point is stable.
    Location,
    /// A calendar date (ISO-8601 `YYYY-MM-DD` or `Mon DD, YYYY`).
    Date,
    /// A URL (`http`/`https` schemes).
    Url,
    /// An email address (`local@domain.tld`).
    Email,
    /// A user-supplied keyword (see [`ExtractorConfig::keywords`]).
    Keyword,
}

impl EntityKind {
    /// Short lower-snake-case string used as the `Node::ntype` for the
    /// entity node committed downstream. Stable wire identifier.
    #[must_use]
    pub const fn ntype(self) -> &'static str {
        match self {
            Self::Person => "person",
            Self::Organization => "organization",
            Self::Location => "location",
            Self::Date => "date",
            Self::Url => "url",
            Self::Email => "email",
            Self::Keyword => "keyword",
        }
    }
}

/// A single entity mention inside a [`Section`].
///
/// `byte_range` refers to offsets within the section's `text` field
/// (not the original source). Downstream commit code combines it with
/// `Section::byte_range` when provenance-accurate source offsets are
/// needed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntitySpan {
    /// Category assigned by the extractor.
    pub kind: EntityKind,
    /// Verbatim surface string as it appears in the section text.
    pub text: String,
    /// Byte range within the section's `text`.
    pub byte_range: Range<usize>,
    /// Heuristic confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

/// A candidate relation between two entities in the same section.
///
/// `subject_span` and `object_span` are indices into the entity vector
/// returned by the same extract call. Relation identifiers are plain
/// strings to keep the shape open; callers emit `"co_occurs_with"` or
/// `"acts_on"` today.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelationSpan {
    /// Predicate label (e.g. `"co_occurs_with"`, `"acts_on"`).
    pub kind: String,
    /// Index of the subject entity within the accompanying `Vec<EntitySpan>`.
    pub subject_span: usize,
    /// Index of the object entity within the accompanying `Vec<EntitySpan>`.
    pub object_span: usize,
    /// Heuristic confidence in `[0.0, 1.0]`.
    pub confidence: f32,
}

// ---------------- Extractor trait ----------------

/// Pluggable entity + relation extractor.
///
/// Implementations must be `Send + Sync` so the [`crate::Ingester`]
/// façade can hand them across thread boundaries in batch ingest paths
/// scheduled by CLI/HTTP wrappers in later waves.
pub trait Extractor: Send + Sync {
    /// Extract entity mentions from a single section.
    fn extract_entities(&self, section: &Section) -> Vec<EntitySpan>;

    /// Extract candidate relations between already-extracted entities.
    fn extract_relations(&self, entities: &[EntitySpan], section: &Section) -> Vec<RelationSpan>;

    /// Optional pre-extraction hook. Called once per file by
    /// [`crate::pipeline::Ingester::ingest`] BEFORE any
    /// `extract_entities` / `extract_relations` call, with the full
    /// list of sections the file produced. The default implementation
    /// is a no-op, so existing extractors keep their behaviour.
    ///
    /// the [`crate::extract_keybert::KeyBertAdapter`]
    /// override pre-batches every section's embedding through
    /// `Embedder::embed_batch` and stashes the vectors in an internal
    /// cache, so subsequent `extract_entities` calls hit the cache
    /// instead of issuing one ORT session.run per section. Bible-
    /// scale ingest drops from "single-threaded sequential section
    /// embed dominates wall time" to "one batched session.run per
    /// file."
    ///
    /// Implementations MUST be idempotent: callers may invoke
    /// `prepare` multiple times across re-uses of the same extractor
    /// without changing entity output. Errors should be swallowed
    /// internally where possible (the default fall-back to lazy
    /// per-section embed must remain correct); a hard error here
    /// aborts the whole file ingest.
    ///
    /// # Errors
    ///
    /// Returns whatever the implementation chooses; the pipeline
    /// passes the error through.
    fn prepare(&self, _sections: &[Section]) -> Result<(), crate::error::Error> {
        Ok(())
    }
}

// ---------------- Default rule extractor ----------------

/// Default rule-based [`Extractor`] implementation shipped with
/// mnem-ingest.
///
/// Construct via [`RuleExtractor::new`] with an [`ExtractorConfig`].
/// Internally caches compiled regex + Aho-Corasick matchers so calls
/// across many sections reuse the automaton.
#[derive(Debug)]
pub struct RuleExtractor {
    cfg: ExtractorConfig,
    url: Regex,
    email: Regex,
    iso_date: Regex,
    long_date: Regex,
    keywords: Option<AhoCorasick>,
    verb_window: Regex,
}

impl RuleExtractor {
    /// Build a new extractor from configuration.
    ///
    /// # Errors
    ///
    /// Returns a `regex::Error` if any of the embedded patterns fails to
    /// compile. The patterns are fixed at compile time, so callers can
    /// treat this as infallible in practice; the error is surfaced only
    /// for symmetry with the aho-corasick builder.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn new(cfg: ExtractorConfig) -> Self {
        // Patterns are fixed, and panicking here on a miscompiled regex
        // would be a compile-time surprise, not a runtime one. The
        // `expect` messages are intentional; they cannot fire.
        let url = Regex::new(r"https?://[^\s<>()\[\]]+[A-Za-z0-9/]").expect("url regex compiles");
        let email = Regex::new(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
            .expect("email regex compiles");
        let iso_date = Regex::new(r"\b\d{4}-\d{2}-\d{2}\b").expect("iso date regex compiles");
        let long_date = Regex::new(
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b",
        )
        .expect("long date regex compiles");
        let verb_window = Regex::new(
            r"(?i)\b(?:joined|founded|acquired|owns|hired|created|launched|bought|leads|runs)\b",
        )
        .expect("verb regex compiles");

        let keywords = if cfg.keywords.is_empty() {
            None
        } else {
            AhoCorasick::builder()
                .match_kind(MatchKind::LeftmostLongest)
                .ascii_case_insensitive(true)
                .build(&cfg.keywords)
                .ok()
        };

        Self {
            cfg,
            url,
            email,
            iso_date,
            long_date,
            keywords,
            verb_window,
        }
    }
}

impl Default for RuleExtractor {
    fn default() -> Self {
        Self::new(ExtractorConfig::default())
    }
}

impl Extractor for RuleExtractor {
    fn extract_entities(&self, section: &Section) -> Vec<EntitySpan> {
        let text = section.text.as_str();
        let mut out: Vec<EntitySpan> = Vec::new();

        if self.cfg.emit_kinds.contains(&EntityKind::Url) {
            collect_regex(&mut out, &self.url, text, EntityKind::Url, 0.95);
        }
        if self.cfg.emit_kinds.contains(&EntityKind::Email) {
            collect_regex(&mut out, &self.email, text, EntityKind::Email, 0.95);
        }
        if self.cfg.emit_kinds.contains(&EntityKind::Date) {
            collect_regex(&mut out, &self.iso_date, text, EntityKind::Date, 0.95);
            collect_regex(&mut out, &self.long_date, text, EntityKind::Date, 0.95);
        }

        if self.cfg.emit_kinds.contains(&EntityKind::Keyword)
            && let Some(ac) = &self.keywords
        {
            for m in ac.find_iter(text) {
                push_span(
                    &mut out,
                    EntityKind::Keyword,
                    text,
                    m.start()..m.end(),
                    0.90,
                );
            }
        }

        let want_person = self.cfg.emit_kinds.contains(&EntityKind::Person);
        let want_org = self.cfg.emit_kinds.contains(&EntityKind::Organization);
        if want_person || want_org {
            for (kind, range) in capitalized_phrases(text) {
                let keep = match kind {
                    EntityKind::Organization => want_org,
                    EntityKind::Person => want_person,
                    _ => false,
                };
                if keep {
                    push_span(&mut out, kind, text, range, 0.60);
                }
            }
        }

        // Deterministic ordering: primary by start offset, secondary by
        // kind so dedup is stable across runs.
        out.sort_by(|a, b| {
            a.byte_range
                .start
                .cmp(&b.byte_range.start)
                .then_with(|| a.kind.ntype().cmp(b.kind.ntype()))
        });
        out.dedup_by(|a, b| a.byte_range == b.byte_range && a.kind == b.kind);
        out
    }

    fn extract_relations(&self, entities: &[EntitySpan], section: &Section) -> Vec<RelationSpan> {
        if entities.len() < 2 {
            return Vec::new();
        }
        let text = section.text.as_str();
        let window = self.cfg.relation_window_tokens;
        let mut out = Vec::new();

        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let a = &entities[i];
                let b = &entities[j];
                if a.byte_range.end > b.byte_range.start {
                    // Overlapping - skip to avoid a self-relation artifact.
                    continue;
                }
                let between = &text[a.byte_range.end..b.byte_range.start];
                let tokens_between = between.split_whitespace().count();
                if tokens_between > window {
                    continue;
                }
                let (kind, conf) = if self.verb_window.is_match(between) {
                    ("acts_on".to_string(), 0.50_f32)
                } else {
                    ("co_occurs_with".to_string(), 0.40_f32)
                };
                out.push(RelationSpan {
                    kind,
                    subject_span: i,
                    object_span: j,
                    confidence: conf,
                });
            }
        }
        out
    }
}

// ---------------- Free helpers ----------------

/// Run [`RuleExtractor::default`] once against a section.
///
/// Thin convenience for callers that don't care to configure the
/// extractor (tests, `mnem ingest --auto`, ad-hoc scripts).
#[must_use]
pub fn extract_entities(section: &Section) -> Vec<EntitySpan> {
    RuleExtractor::default().extract_entities(section)
}

/// Run [`RuleExtractor::default`] once to derive relations.
///
/// Expects `entities` to have been produced by the same extractor; the
/// relation indices only make sense against that exact list.
#[must_use]
pub fn extract_relations(entities: &[EntitySpan], section: &Section) -> Vec<RelationSpan> {
    RuleExtractor::default().extract_relations(entities, section)
}

fn collect_regex(
    out: &mut Vec<EntitySpan>,
    re: &Regex,
    text: &str,
    kind: EntityKind,
    confidence: f32,
) {
    for m in re.find_iter(text) {
        push_span(out, kind, text, m.start()..m.end(), confidence);
    }
}

fn push_span(
    out: &mut Vec<EntitySpan>,
    kind: EntityKind,
    text: &str,
    range: Range<usize>,
    confidence: f32,
) {
    let slice = text.get(range.clone()).unwrap_or("").to_string();
    if slice.is_empty() {
        return;
    }
    out.push(EntitySpan {
        kind,
        text: slice,
        byte_range: range,
        confidence,
    });
}

/// Common words a capitalized-phrase run is likely to swallow but that
/// carry no entity meaning. Kept deliberately small - false positives at
/// this layer are filtered downstream by the LLM extractor in B5e.
const COMMON_DENYLIST: &[&str] = &[
    "The", "This", "That", "These", "Those", "A", "An", "And", "Or", "But", "If", "In", "On", "At",
    "To", "From", "With", "By", "For", "Of", "As", "Is", "Was", "Are", "Were", "Be", "Been",
    "Being", "I", "We", "You", "He", "She", "It", "They", "My", "Our", "Your", "His", "Her",
    "Their", "Mr", "Mrs", "Ms", "Dr",
];

/// Suffix tokens that promote a capitalized run from Person to
/// Organization (e.g. "`Acme Corp`", "`Foo Inc`"). Case-sensitive on the
/// suffix but matched against the trimmed token.
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

/// Walk the text, emitting `(kind, byte_range)` for every run of two or
/// more capitalized tokens. Heuristic, not linguistic.
fn capitalized_phrases(text: &str) -> Vec<(EntityKind, Range<usize>)> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    let len = bytes.len();

    while i < len {
        // Skip non-letter.
        if !is_ascii_upper(bytes[i]) {
            i += 1;
            continue;
        }
        let start = i;
        let mut last_end = i;
        let mut token_count = 0;
        let mut saw_org_suffix = false;

        while i < len && is_ascii_upper(bytes[i]) {
            // Consume a capitalized token: upper then [A-Za-z.]*.
            let tok_start = i;
            i += 1;
            while i < len && (bytes[i].is_ascii_alphabetic() || bytes[i] == b'.') {
                i += 1;
            }
            let tok = &text[tok_start..i];
            if COMMON_DENYLIST.contains(&tok) && token_count == 0 {
                // Leading common word - bail and restart after it.
                token_count = 0;
                last_end = i;
                break;
            }
            token_count += 1;
            last_end = i;
            if ORG_SUFFIXES.contains(&tok) {
                saw_org_suffix = true;
            }
            // Expect a single space before the next capitalized token.
            if i < len && bytes[i] == b' ' && i + 1 < len && is_ascii_upper(bytes[i + 1]) {
                i += 1;
                continue;
            }
            break;
        }

        if token_count >= 2 {
            let kind = if saw_org_suffix {
                EntityKind::Organization
            } else {
                EntityKind::Person
            };
            out.push((kind, start..last_end));
        }
        // Advance past whitespace so the outer loop makes progress.
        while i < len && !is_ascii_upper(bytes[i]) {
            i += 1;
        }
    }
    out
}

const fn is_ascii_upper(b: u8) -> bool {
    b.is_ascii_uppercase()
}

// ---------------- Tests ----------------

#[cfg(test)]
mod tests {
    use super::*;

    fn section(text: &str) -> Section {
        Section {
            heading: None,
            depth: 0,
            text: text.to_string(),
            byte_range: 0..text.len(),
        }
    }

    #[test]
    fn extracts_urls() {
        let s = section("See https://example.com/x and http://foo.io for details.");
        let ents = extract_entities(&s);
        let urls: Vec<_> = ents.iter().filter(|e| e.kind == EntityKind::Url).collect();
        assert_eq!(urls.len(), 2);
        assert!(
            urls.iter()
                .any(|e| e.text.starts_with("https://example.com"))
        );
        assert!(urls.iter().any(|e| e.text.starts_with("http://foo.io")));
    }

    #[test]
    fn extracts_emails() {
        let s = section("Contact alice@example.com or bob.smith+x@corp.co.uk today.");
        let ents = extract_entities(&s);
        let emails: Vec<_> = ents
            .iter()
            .filter(|e| e.kind == EntityKind::Email)
            .collect();
        assert_eq!(emails.len(), 2);
        assert!(emails.iter().any(|e| e.text == "alice@example.com"));
    }

    #[test]
    fn rejects_non_email_atsign() {
        let s = section("the @handle tag is not email, nor is foo@.");
        let ents = extract_entities(&s);
        assert!(!ents.iter().any(|e| e.kind == EntityKind::Email));
    }

    #[test]
    fn extracts_iso_and_long_dates() {
        let s = section("Filed on 2026-04-24; rescheduled to Apr 30, 2026.");
        let ents = extract_entities(&s);
        let dates = ents.iter().filter(|e| e.kind == EntityKind::Date).count();
        assert_eq!(dates, 2);
    }

    #[test]
    fn ignores_bogus_date() {
        let s = section("version 1.2.3 released last year");
        let ents = extract_entities(&s);
        assert!(!ents.iter().any(|e| e.kind == EntityKind::Date));
    }

    #[test]
    fn extracts_keyword_matches() {
        let cfg = ExtractorConfig {
            keywords: vec!["rustls".into(), "tokio".into()],
            ..ExtractorConfig::default()
        };
        let ext = RuleExtractor::new(cfg);
        let s = section("Built on rustls and Tokio for async I/O.");
        let ents = ext.extract_entities(&s);
        let kw = ents
            .iter()
            .filter(|e| e.kind == EntityKind::Keyword)
            .count();
        assert_eq!(kw, 2, "got: {ents:?}");
    }

    #[test]
    fn no_keyword_when_denied() {
        let cfg = ExtractorConfig::default();
        let ext = RuleExtractor::new(cfg);
        let s = section("This body has no keyword configured at all.");
        let ents = ext.extract_entities(&s);
        assert!(!ents.iter().any(|e| e.kind == EntityKind::Keyword));
    }

    #[test]
    fn capitalized_phrase_detects_person() {
        let s = section("Alice Johnson met Bob Lee at the lobby.");
        let ents = extract_entities(&s);
        assert!(
            ents.iter()
                .any(|e| e.kind == EntityKind::Person && e.text == "Alice Johnson"),
            "got: {ents:?}"
        );
        assert!(
            ents.iter()
                .any(|e| e.kind == EntityKind::Person && e.text == "Bob Lee"),
            "got: {ents:?}"
        );
    }

    #[test]
    fn capitalized_phrase_detects_org_suffix() {
        let s = section("Acme Corp and Foo Inc signed the deal.");
        let ents = extract_entities(&s);
        assert!(
            ents.iter()
                .any(|e| e.kind == EntityKind::Organization && e.text == "Acme Corp"),
            "got: {ents:?}"
        );
    }

    #[test]
    fn capitalized_rejects_single_token() {
        let s = section("Alice then left.");
        let ents = extract_entities(&s);
        assert!(!ents.iter().any(|e| e.kind == EntityKind::Person));
    }

    #[test]
    fn relations_proximity_co_occurs() {
        let s = section("Alice Johnson met Bob Lee today.");
        let ents = extract_entities(&s);
        let rels = extract_relations(&ents, &s);
        assert!(
            rels.iter().any(|r| r.kind == "co_occurs_with"),
            "got rels: {rels:?}"
        );
    }

    #[test]
    fn relations_verb_between_becomes_acts_on() {
        let s = section("Alice Johnson founded Acme Corp in 2022.");
        let ents = extract_entities(&s);
        let rels = extract_relations(&ents, &s);
        assert!(
            rels.iter().any(|r| r.kind == "acts_on"),
            "got rels: {rels:?}, ents: {ents:?}"
        );
    }

    #[test]
    fn confidence_tiers_respected() {
        let s = section("Alice Johnson visited https://example.com on 2026-04-24.");
        let ents = extract_entities(&s);
        for e in &ents {
            match e.kind {
                EntityKind::Url | EntityKind::Date | EntityKind::Email => {
                    assert!((e.confidence - 0.95).abs() < f32::EPSILON);
                }
                EntityKind::Person | EntityKind::Organization | EntityKind::Location => {
                    assert!((e.confidence - 0.60).abs() < f32::EPSILON);
                }
                EntityKind::Keyword => {
                    assert!((e.confidence - 0.90).abs() < f32::EPSILON);
                }
            }
        }
    }
}
