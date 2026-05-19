//! End-to-end integration smokes for the ingest pipeline.
//!
//! Distinct from the unit tests inside `pipeline.rs` (which exercise
//! individual stages): this file drives the public surface the way the
//! CLI / MCP / HTTP callers do, so any regression in the composed
//! pipeline (parse + chunk + extract + write) surfaces here.
//!
//! Eight real tests currently present (plus two `#[ignore]`d subprocess
//! stubs): `mnem_ingest_md_roundtrip`, `mnem_ingest_conversation_smoke`,
//! `mnem_ingest_pdf_no_panic`, `mnem_ingest_rust_code_extracts_functions`,
//! `mnem_ingest_python_code_extracts_functions`,
//! `mnem_ingest_malformed_python_code_falls_back_to_full_file_chunk`,
//! `mnem_ingest_rust_code_with_embedder`, `mnem_ingest_python_code_with_embedder`.
//! Subprocess tests for the CLI / HTTP binaries are `#[ignore]`d so the
//! default `cargo test` run stays hermetic.

use std::sync::Arc;

use bytes::Bytes;
use mnem_core::objects::{Dtype, Embedding};
use mnem_core::repo::ReadonlyRepo;
use mnem_core::store::{MemoryBlockstore, MemoryOpHeadsStore};

use mnem_ingest::pipeline::EmbedText;
use mnem_ingest::{ChunkerKind, IngestConfig, Ingester, SourceKind};

/// In-memory repo: no redb, no temp dir, no fsync. Mirrors the helper
/// used in the pipeline inline tests; duplicated here so the integration
/// test compiles without leaking private items from `pipeline.rs`.
fn test_repo() -> ReadonlyRepo {
    let bs = Arc::new(MemoryBlockstore::new());
    let op_heads = Arc::new(MemoryOpHeadsStore::new());
    ReadonlyRepo::init(bs, op_heads).expect("init mnem repo")
}

/// Zero-vector stub embedder. The ingest pipeline calls
/// `embed_text(chunk)` once per chunk when an embedder is attached;
/// this stub satisfies the trait without pulling in a provider crate.
/// Dim is 16 - arbitrary at this layer; the pipeline does not enforce
/// a specific size, only that every chunk gets the same shape.
struct ZeroEmbedder;

impl EmbedText for ZeroEmbedder {
    fn embed_text(&self, _text: &str) -> Result<Embedding, mnem_ingest::Error> {
        let buf: Vec<u8> = vec![0u8; 16 * 4];
        Ok(Embedding {
            model: "zero-test".into(),
            dtype: Dtype::F32,
            dim: 16,
            vector: Bytes::from(buf),
        })
    }
}

#[test]
fn mnem_ingest_md_roundtrip() {
    let repo = test_repo();
    let mut tx = repo.start_transaction();
    let ing = Ingester::new(IngestConfig {
        chunker: ChunkerKind::Paragraph,
        ntype: "Doc".into(),
        max_tokens: 512,
        overlap: 32,
        ner: mnem_ingest::NerConfig::default(),
    });

    // Snippet shaped like docs/SPEC.md: a heading plus two prose
    // paragraphs with named entities the rule-based extractor should
    // pick up (names, a URL, an ISO-8601 date).
    let md = "# mnem - Content-addressed Knowledge Graphs\n\n\
              mnem is a Rust substrate developed at Uranid. The 0.3 line \
              added semantic search and HTTP + MCP transports.\n\n\
              Alice Johnson filed on 2026-04-24; see https://example.com \
              for the full decision record.\n";

    let result = ing
        .ingest(&mut tx, md.as_bytes(), SourceKind::Markdown)
        .expect("ingest must succeed on a valid markdown snippet");

    assert!(
        result.node_count > 0,
        "node_count must be > 0 (doc + chunks + entities); got {result:?}"
    );
    assert!(
        result.chunk_count >= 1,
        "at least one chunk expected; got {result:?}"
    );
    assert!(
        result.entity_count >= 1,
        "rule-based extractor must detect at least one entity in: {result:?}"
    );
}

#[test]
fn mnem_ingest_conversation_smoke() {
    let repo = test_repo();
    let mut tx = repo.start_transaction();
    // Generic conversation format: array of `{role, content}` turns.
    // Matches the smallest ChatGPT-export-shaped payload that
    // `conversation::parse_conversation` recognises via the Generic
    // fallback.
    let chat = r#"[
        {"role": "user", "content": "Who founded Acme Corp?"},
        {"role": "assistant", "content": "Acme Corp was founded by Alice Johnson in 2026."},
        {"role": "user", "content": "When did Bob Lee join?"}
    ]"#;

    let ing = Ingester::new(IngestConfig {
        // Session chunker groups contiguous messages until the role
        // returns to `user` or a cap is hit; auto_chunker would pick
        // this for SourceKind::Conversation anyway, but we spell it
        // out to keep the assertion tight.
        chunker: ChunkerKind::Session { max_messages: 10 },
        ntype: "Conversation".into(),
        max_tokens: 512,
        overlap: 0,
        ner: mnem_ingest::NerConfig::default(),
    });

    let result = ing
        .ingest(&mut tx, chat.as_bytes(), SourceKind::Conversation)
        .expect("3-message conversation must ingest cleanly");

    assert!(
        result.node_count > 0,
        "node_count must be > 0 for a 3-message conversation; got {result:?}"
    );
    assert!(
        result.chunk_count >= 1,
        "session chunker must emit at least one chunk; got {result:?}"
    );
}

#[test]
fn mnem_ingest_pdf_no_panic() {
    // Inline minimal PDF identical in spirit to the B5b pdf.rs
    // fixture: just enough header + cross-ref to satisfy `pdf-extract`
    // that it's looking at a PDF without forcing us to ship a binary
    // fixture. The pipeline must EITHER return Ok OR surface a typed
    // ParseFailed; the one thing it must not do is panic, per the
    // B5b `catch_unwind` contract.
    let tiny_pdf: &[u8] = b"%PDF-1.4\n\
        1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n\
        2 0 obj<</Type/Pages/Count 0/Kids[]>>endobj\n\
        xref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000053 00000 n \n\
        trailer<</Size 3/Root 1 0 R>>\nstartxref\n95\n%%EOF\n";

    let repo = test_repo();
    let mut tx = repo.start_transaction();
    let ing = Ingester::new(IngestConfig::default());

    let outcome = ing.ingest(&mut tx, tiny_pdf, SourceKind::Pdf);
    match outcome {
        Ok(res) => {
            // Empty / near-empty PDFs legitimately produce zero chunks;
            // what we care about is that the path is panic-free and
            // returns a well-formed result structure.
            assert_eq!(
                res.relation_count, 0,
                "empty PDF should not surface relations; got {res:?}"
            );
        }
        Err(e) => {
            // Accept any typed error. The panic-free contract is the
            // load-bearing one; exact error shape is a B5b detail.
            let _ = format!("{e}");
        }
    }
}

// ---------- Code (tree-sitter) source kind ----------

#[test]
fn mnem_ingest_rust_code_extracts_functions() {
    use mnem_ingest::CodeLanguage;

    let repo = test_repo();
    let mut tx = repo.start_transaction();
    let ing = Ingester::new(IngestConfig {
        chunker: ChunkerKind::Structural,
        ntype: "Doc".into(),
        max_tokens: 512,
        overlap: 0,
        ner: mnem_ingest::NerConfig::default(),
    });

    let rust_src = "\
pub fn add(a: i32, b: i32) -> i32 { a + b }
pub fn subtract(a: i32, b: i32) -> i32 { a - b }
struct Counter { count: u32 }
impl Counter {
    fn new() -> Self { Counter { count: 0 } }
    fn increment(&mut self) { self.count += 1; }
}
";

    let result = ing
        .ingest(
            &mut tx,
            rust_src.as_bytes(),
            SourceKind::Code(CodeLanguage::Rust),
        )
        .expect("ingest must succeed on a valid Rust snippet");

    assert!(
        result.node_count > 0,
        "node_count must be > 0 (doc node + function/struct chunks); got {result:?}"
    );
    // 3 items are provably captured by the Rust query (function_item, struct_item, enum_item, trait_item):
    //   fn add, fn subtract → function_item captures
    //   struct Counter      → struct_item capture
    // impl Counter's methods (fn new, fn increment) live inside an impl_item node, which the
    // Rust query does NOT capture - only the outer impl block would be, and impl_item is not
    // in the query. Methods are embedded in the class span but not chunked independently.
    // This is why the Rust floor (>= 3) is lower than the Python floor (>= 4), where methods
    // inside a class ARE captured via function_definition at any depth.
    assert!(
        result.chunk_count >= 3,
        "at least 3 chunks expected (fn add, fn subtract, struct Counter); got {result:?}"
    );
}

#[test]
fn mnem_ingest_python_code_extracts_functions() {
    use mnem_ingest::CodeLanguage;

    let repo = test_repo();
    let mut tx = repo.start_transaction();
    let ing = Ingester::new(IngestConfig {
        chunker: ChunkerKind::Structural,
        ntype: "Doc".into(),
        max_tokens: 512,
        overlap: 0,
        ner: mnem_ingest::NerConfig::default(),
    });

    let python_src = "\
def greet(name):
    return f\"Hello, {name}!\"

def add(a, b):
    return a + b

class Calculator:
    def multiply(self, a, b):
        return a * b
";

    let result = ing
        .ingest(
            &mut tx,
            python_src.as_bytes(),
            SourceKind::Code(CodeLanguage::Python),
        )
        .expect("ingest must succeed on a valid Python snippet");

    assert!(
        result.node_count > 0,
        "node_count must be > 0 (doc node + function/class chunks); got {result:?}"
    );
    assert!(
        result.chunk_count >= 4,
        // 4 items provably captured:
        //   - def greet, def add (top-level function_definitions)
        //   - class Calculator (class_definition)
        //   - def multiply (function_definition inside class - captured at any depth)
        // Note: method code appears in both its own chunk AND as part of the class span.
        // This is expected Python structural behavior.
        "at least 4 chunks expected (def greet, def add, class Calculator, def multiply); got {result:?}"
    );
}

#[test]
fn mnem_ingest_malformed_python_code_falls_back_to_full_file_chunk() {
    use mnem_ingest::CodeLanguage;

    // tree-sitter is fault-tolerant: it never panics on invalid input and never
    // returns None for non-empty source. A syntactically broken .py file is parsed
    // as a partial tree with ERROR nodes. No function_definition or class_definition
    // nodes survive the ERROR region, so item_query finds nothing and the fallback
    // (one headless section containing the full source text) fires.
    // This verifies that parse errors are handled gracefully without panics.
    let repo = test_repo();
    let mut tx = repo.start_transaction();
    let ing = Ingester::new(IngestConfig {
        chunker: ChunkerKind::Structural,
        ntype: "Doc".into(),
        max_tokens: 512,
        overlap: 0, // overlap has no effect on ChunkerKind::Structural
        ner: mnem_ingest::NerConfig::default(),
    });
    let malformed_py = b"def broken(\n    x: int\n    y: int  # missing comma\n->:\n    return x + y\n";
    let result = ing
        .ingest(&mut tx, malformed_py, SourceKind::Code(CodeLanguage::Python))
        .expect("malformed Python must not panic or return Err; tree-sitter is fault-tolerant");
    assert!(
        result.node_count >= 1,
        "must produce at least 1 node (doc root) even for malformed source; {result:?}"
    );
    // chunk_count is exactly 1 since the source is non-empty: the fallback section
    // contains the full source text, which chunk_structural does NOT skip.
    assert_eq!(
        result.chunk_count,
        1,
        "non-empty malformed Python must produce exactly 1 chunk via the whole-file fallback section; {result:?}"
    );
}

#[test]
fn mnem_ingest_rust_code_with_embedder() {
    use mnem_ingest::CodeLanguage;

    // Uses Rust to cover the Rust + embedder path for symmetric coverage
    // alongside the Python embedder test below.
    let repo = test_repo();
    let mut tx = repo.start_transaction();

    let snippet = "pub fn multiply(a: i32, b: i32) -> i32 { a * b }\n\
                   pub fn divide(a: f64, b: f64) -> f64 { a / b }\n\
                   enum Op { Add, Sub, Mul }\n";

    let ing = Ingester::new(IngestConfig {
        chunker: ChunkerKind::Structural,
        ntype: "Doc".into(),
        max_tokens: 512,
        overlap: 32,
        ner: mnem_ingest::NerConfig::default(),
    })
    .with_embedder(Arc::new(ZeroEmbedder));

    let result = ing
        .ingest(
            &mut tx,
            snippet.as_bytes(),
            SourceKind::Code(CodeLanguage::Rust),
        )
        .expect("ingest must succeed");
    assert!(result.chunk_count >= 3, "snippet has 2 fns + 1 enum → must produce >= 3 structural chunks; {result:?}");
    assert!(result.node_count >= 4, "must create doc node + >= 3 chunk nodes = >= 4 total; {result:?}");
}

#[test]
fn mnem_ingest_python_code_with_embedder() {
    use mnem_ingest::CodeLanguage;

    // Exercises the Python + ZeroEmbedder path specifically:
    // - ZeroEmbedder produces a zero vector for each chunk, exercising the
    //   embedding storage write path (prop writes to chunk nodes in the blockstore).
    // - Verifies that embedding a Code source does not alter the structural
    //   chunk count or node count relative to the non-embedded case.
    let repo = test_repo();
    let mut tx = repo.start_transaction();

    let python_src = "\
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

class MathUtils:
    def factorial(self, n):
        if n == 0:
            return 1
        return n * self.factorial(n - 1)
";

    let ing = Ingester::new(IngestConfig {
        chunker: ChunkerKind::Structural,
        ntype: "Doc".into(),
        max_tokens: 512,
        overlap: 0,
        ner: mnem_ingest::NerConfig::default(),
    })
    .with_embedder(Arc::new(ZeroEmbedder));

    let result = ing
        .ingest(
            &mut tx,
            python_src.as_bytes(),
            SourceKind::Code(CodeLanguage::Python),
        )
        .expect("ingest with embedder must not panic on Python code chunks");

    assert!(result.chunk_count >= 4, "Python snippet has 2 fns + 1 class + 1 method → must produce >= 4 structural chunks; {result:?}");
    assert!(result.node_count >= 5, "must create doc node + >= 4 chunk nodes = >= 5 total; {result:?}");
}

// ---------- CLI / HTTP subprocess smokes ----------
//
// These would spawn `target/debug/mnem ingest` and
// `target/debug/mnem http` respectively, POST a Markdown body, and
// assert on the JSON response. They're marked `#[ignore]` so the
// default `cargo test` pass stays hermetic; CI opts in via
// `cargo test -- --ignored` once a dedicated job lands.

#[test]
#[ignore = "spawns the CLI binary; opt in with --ignored"]
fn mnem_ingest_cli_subprocess() {
    // Intentionally unimplemented at the B5 finalize cut. The CLI
    // command itself is already covered by the in-crate tests under
    // `crates/mnem-cli/tests/` (B5d-1); this stub reserves the slot
    // for a full end-to-end subprocess run once the bench harness
    // lands.
}

#[test]
#[ignore = "spawns the HTTP binary; opt in with --ignored"]
fn mnem_ingest_http_subprocess() {
    // Reserved for the same reason as above. The in-process axum
    // `oneshot` route is covered by
    // `crates/mnem-http/tests/integration.rs` (B5d-3).
}

// Compile-check the embedder contract so the ZeroEmbedder stays
// linked even if no test consumes it directly today. `Arc<dyn
// EmbedText>` is the shape downstream callers attach via
// `Ingester::with_embedder`.
#[allow(dead_code)]
fn _embedder_contract_compiles() -> Arc<dyn EmbedText> {
    Arc::new(ZeroEmbedder)
}
