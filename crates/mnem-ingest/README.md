# mnem-ingest

Ingest pipeline for [mnem]: converts external source artifacts (Markdown,
plain text, and - in later sub-waves - PDFs and chat conversations) into
content-addressed memory graphs.

## Scope

This crate ships in five sub-waves (B5a – B5e). The current release
(**B5a**) provides the pure, deterministic half of the pipeline:

- **Markdown parser** (`md::parse_markdown`) - CommonMark + GFM with
  heading hierarchy preserved and code blocks kept atomic.
- **Text parser** (`text::parse_text`) - single-section pass-through.
- **Chunkers** (`chunk::chunk`):
  - `Paragraph` - splits on blank lines.
  - `Recursive` - token-budgeted sliding window with overlap.

The repository-commit path (`Ingester::run`) is stubbed until **B5c**.

## Quick start

```rust
use mnem_ingest::{md::parse_markdown, chunk::{chunk, ChunkerKind}};

let md = "# Title\n\nFirst para.\n\n## Sub\n\nSecond para.";
let sections = parse_markdown(md).unwrap;
let chunks = chunk(
    &sections,
    &ChunkerKind::Recursive { max_tokens: 64, overlap: 8 },
);

for c in chunks {
    println!("{:?} → {}", c.section_path, c.text);
}
```

## License

Apache-2.0. See `../../LICENSE`.

[mnem]: https://github.com/Uranid/mnem
