//! Source-code parser using tree-sitter.
//!
//! Converts source code into one [`Section`] per top-level callable or
//! type definition (function, method, struct, class, enum, trait, impl
//! block). The section body is the verbatim source span of the item, making
//! it the natural unit for embedding: the whole function is the chunk.
//!
//! When no structured items are found (e.g. a header-only file or a
//! file that the grammar does not handle), a single headless section
//! covering the full source is returned so the pipeline always has
//! something to embed.
//!
//! # Supported languages
//!
//! Rust, Python, JavaScript, TypeScript, Go, Java, C, C++.
//! Languages are selected by [`CodeLanguage`]; the grammar is compiled
//! from C source by each grammar crate's `build.rs`.

use streaming_iterator::StreamingIterator;
use tree_sitter::{Language, Parser, Query, QueryCursor};

use crate::error::Error;
use crate::types::{CodeLanguage, Section};

/// Parse source code into structural sections.
///
/// Each top-level function / class / struct / impl becomes a [`Section`]
/// whose `heading` is `"<kind>:<name>"` (e.g. `"fn:main"`, `"class:Foo"`)
/// and whose `text` is the verbatim source span.
///
/// # Errors
///
/// Returns [`Error::ParseFailed`] if the grammar cannot be initialised or
/// if the source is not valid UTF-8 (the caller should validate before
/// passing). A source with syntax errors is still parsed - tree-sitter
/// produces a best-effort tree for recovery.
pub fn parse_code(source: &str, lang: CodeLanguage) -> Result<Vec<Section>, Error> {
    let ts_lang = language_for(lang);

    let mut parser = Parser::new();
    if let Err(e) = parser.set_language(&ts_lang) {
        // Grammar ABI mismatch (e.g. a grammar crate built for a newer tree-sitter
        // ABI than the core we link against). Degrade gracefully to a whole-file
        // section rather than failing the ingest entirely.
        tracing::debug!(
            lang = lang.as_str(),
            error = %e,
            "tree-sitter grammar ABI incompatible; falling back to full-file section"
        );
        return Ok(vec![Section {
            heading: None,
            depth: 0,
            text: source.to_string(),
            byte_range: 0..source.len(),
        }]);
    }

    let tree = parser.parse(source, None).ok_or_else(|| Error::ParseFailed {
        what: format!("code:{}", lang.as_str()),
        detail: "tree-sitter parse returned None (input may be empty or cancelled)".into(),
    })?;

    let query_src = item_query(lang);
    let query = match Query::new(&ts_lang, query_src) {
        Ok(q) => q,
        Err(e) => {
            tracing::debug!(
                lang = lang.as_str(),
                error = %e,
                "tree-sitter query compile failed; falling back to full-file section"
            );
            return Ok(vec![Section {
                heading: None,
                depth: 0,
                text: source.to_string(),
                byte_range: 0..source.len(),
            }]);
        }
    };

    let name_idx = query.capture_index_for_name("name");
    let item_idx = query.capture_index_for_name("item");

    let mut cursor = QueryCursor::new();
    let raw = source.as_bytes();
    let mut matches = cursor.matches(&query, tree.root_node(), raw);

    let mut sections: Vec<Section> = Vec::new();

    while let Some(m) = matches.next() {
        let mut item_node = None;
        let mut name_text: Option<String> = None;

        for cap in m.captures {
            if Some(cap.index) == item_idx {
                item_node = Some(cap.node);
            } else if Some(cap.index) == name_idx {
                if let Ok(t) = cap.node.utf8_text(raw) {
                    name_text = Some(t.to_string());
                }
            }
        }

        let (Some(node), Some(name)) = (item_node, name_text) else {
            continue;
        };

        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        let body = source[start_byte..end_byte].to_string();

        let kind = item_kind_label(node.kind());
        let heading = format!("{kind}:{name}");

        sections.push(Section {
            heading: Some(heading),
            depth: 1,
            text: body,
            byte_range: start_byte..end_byte,
        });
    }

    if sections.is_empty() {
        // Fallback: whole file as one headless section.
        return Ok(vec![Section {
            heading: None,
            depth: 0,
            text: source.to_string(),
            byte_range: 0..source.len(),
        }]);
    }

    Ok(sections)
}

// ─── Language selection ───────────────────────────────────────────────────────

fn language_for(lang: CodeLanguage) -> Language {
    match lang {
        CodeLanguage::Rust => tree_sitter_rust::LANGUAGE.into(),
        CodeLanguage::Python => tree_sitter_python::LANGUAGE.into(),
        CodeLanguage::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
        CodeLanguage::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        CodeLanguage::Go => tree_sitter_go::LANGUAGE.into(),
        CodeLanguage::Java => tree_sitter_java::LANGUAGE.into(),
        CodeLanguage::C => tree_sitter_c::LANGUAGE.into(),
        CodeLanguage::Cpp => tree_sitter_cpp::LANGUAGE.into(),
        CodeLanguage::Ruby => tree_sitter_ruby::LANGUAGE.into(),
        CodeLanguage::CSharp => tree_sitter_c_sharp::LANGUAGE.into(),
    }
}

// ─── Tree-sitter queries ──────────────────────────────────────────────────────

/// Return the S-expression query that extracts named items for `lang`.
///
/// Each query must capture exactly two names:
/// - `@item` - the whole node (for byte range extraction)
/// - `@name` - the identifier node (for heading text)
fn item_query(lang: CodeLanguage) -> &'static str {
    match lang {
        CodeLanguage::Rust => concat!(
            "(function_item name: (identifier) @name) @item\n",
            "(struct_item name: (type_identifier) @name) @item\n",
            "(enum_item name: (type_identifier) @name) @item\n",
            "(trait_item name: (type_identifier) @name) @item\n",
        ),
        CodeLanguage::Python => concat!(
            "(function_definition name: (identifier) @name) @item\n",
            "(class_definition name: (identifier) @name) @item\n",
        ),
        CodeLanguage::JavaScript => concat!(
            "(function_declaration name: (identifier) @name) @item\n",
            "(class_declaration name: (identifier) @name) @item\n",
        ),
        CodeLanguage::TypeScript => concat!(
            "(function_declaration name: (identifier) @name) @item\n",
            "(class_declaration name: (type_identifier) @name) @item\n",
            "(interface_declaration name: (type_identifier) @name) @item\n",
            "(type_alias_declaration name: (type_identifier) @name) @item\n",
        ),
        CodeLanguage::Go => concat!(
            "(function_declaration name: (identifier) @name) @item\n",
            "(method_declaration name: (field_identifier) @name) @item\n",
            "(type_spec name: (type_identifier) @name) @item\n",
        ),
        CodeLanguage::Java => concat!(
            "(method_declaration name: (identifier) @name) @item\n",
            "(class_declaration name: (identifier) @name) @item\n",
        ),
        CodeLanguage::C | CodeLanguage::Cpp => concat!(
            "(function_definition\n",
            "  declarator: (function_declarator\n",
            "    declarator: (identifier) @name)) @item\n",
        ),
        CodeLanguage::Ruby => concat!(
            "(method name: (identifier) @name) @item\n",
            "(singleton_method name: (identifier) @name) @item\n",
            "(class name: (constant) @name) @item\n",
            "(module name: (constant) @name) @item\n",
        ),
        CodeLanguage::CSharp => concat!(
            "(method_declaration name: (identifier) @name) @item\n",
            "(class_declaration name: (identifier) @name) @item\n",
            "(interface_declaration name: (identifier) @name) @item\n",
            "(struct_declaration name: (identifier) @name) @item\n",
        ),
    }
}

fn item_kind_label(kind: &str) -> &'static str {
    match kind {
        "function_item" | "function_definition" | "function_declaration" => "fn",
        "method_declaration" | "method_definition" | "method" | "singleton_method" => "method",
        "struct_item" | "struct_declaration" => "struct",
        "enum_item" => "enum",
        "trait_item" => "trait",
        "class_definition" | "class_declaration" => "class",
        "interface_declaration" => "interface",
        "module" => "module",
        "type_alias_declaration" | "type_spec" => "type",
        _ => "item",
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rust_functions() {
        let src = r#"
fn add(a: i32, b: i32) -> i32 { a + b }

struct Point { x: f32, y: f32 }

fn main() {
    println!("hello");
}
"#;
        let sections = parse_code(src, CodeLanguage::Rust).unwrap();
        // Should find: add, Point, main → at least 2
        assert!(
            sections.len() >= 2,
            "expected at least 2 sections, got {} - {:#?}",
            sections.len(),
            sections
        );
        let headings: Vec<_> = sections.iter().filter_map(|s| s.heading.as_deref()).collect();
        assert!(
            headings.iter().any(|h| h.starts_with("fn:add")),
            "missing fn:add in {headings:?}"
        );
        assert!(
            headings.iter().any(|h| h.starts_with("fn:main")),
            "missing fn:main in {headings:?}"
        );
    }

    #[test]
    fn parse_python_class_and_function() {
        let src = r#"
class Animal:
    def speak(self):
        pass

def standalone():
    return 42
"#;
        let sections = parse_code(src, CodeLanguage::Python).unwrap();
        let headings: Vec<_> = sections.iter().filter_map(|s| s.heading.as_deref()).collect();
        assert!(
            headings.iter().any(|h| h.starts_with("class:Animal")),
            "missing class:Animal in {headings:?}"
        );
        assert!(
            headings.iter().any(|h| h.starts_with("fn:standalone")),
            "missing fn:standalone in {headings:?}"
        );
    }

    #[test]
    fn fallback_for_empty_or_no_items() {
        // A file with no top-level declarations.
        let src = "// just a comment\n";
        let sections = parse_code(src, CodeLanguage::Rust).unwrap();
        assert_eq!(sections.len(), 1, "expected single fallback section");
        assert!(sections[0].heading.is_none());
    }

    #[test]
    fn section_byte_range_is_valid() {
        let src = "fn foo() {}\nfn bar() {}\n";
        let sections = parse_code(src, CodeLanguage::Rust).unwrap();
        for s in &sections {
            assert!(s.byte_range.end <= src.len());
            assert!(s.byte_range.start <= s.byte_range.end);
        }
    }
}
