# Ingest pipeline

`mnem ingest` is the only path content takes into the graph. The pipeline:

```
parse -> chunk -> extract -> embed -> commit
```

## Sources

- file path (`mnem ingest README.md`)
- structured JSON (`mnem ingest data.json`) — JSON/JSONL detected by extension
- code files (`mnem ingest src/lib.rs`)
- directory (`mnem ingest src/ --recursive`)
- inline text (`mnem ingest --text "The quick brown fox"`)

## Code file ingest

mnem uses Tree-sitter to parse source files into function- and class-level
chunks. These extensions are fully supported including in `--recursive` mode:

| Language | Extensions |
|----------|------------|
| Rust | `.rs` |
| Python | `.py` |
| JavaScript | `.js` |
| TypeScript | `.ts` |
| Go | `.go` |
| Java | `.java` |
| C | `.c` |
| C++ | `.cpp` |
| Ruby | `.rb` |
| C# | `.cs` |

Config files (`.yaml`, `.toml`, `.sql`, `.html`, `.sh`, `.php`, `.swift`,
`.kt`, `.lua`, `.zig`) are also supported and use sentence-aware chunking
instead of Tree-sitter.

```bash
mnem ingest src/lib.rs                    # single Rust file
mnem ingest src/ --recursive              # all supported files under src/
mnem ingest . --recursive                 # whole project
```

> **Note:** `.pyi`, `.mjs`, `.cjs`, and `.c++` files can be ingested
> individually but are not picked up by `--recursive`.

## Chunking

The `auto` chunker (default) picks a strategy by file type: paragraph splits
for Markdown, Tree-sitter for code, sentence-recursive for plain text and PDFs,
and session chunking for JSON/JSONL conversation exports. The `--max-tokens`
flag (default: 512) and `--overlap` flag (default: 32) apply when the recursive
or sentence-recursive strategy is active. Override both flags explicitly:

```bash
mnem ingest notes.md --max-tokens 256 --overlap 50
```

Use `--chunker` to force a specific strategy regardless of file type. Valid
values: `auto`, `paragraph`, `recursive`, `sentence_recursive`, `session`,
`structural`.

## Extractors

Optional ingest-time enrichment:

| Extractor | What it does |
|-----------|--------------|
| `none` (default) | raw text only |
| `keybert` | KeyBERT keyphrase extraction; phrases stored in node metadata |

Enable via flag:

```bash
mnem ingest README.md --extractor keybert
```

## Labels

Pass `--ntype <str>` to tag ingested nodes with a custom type:

```bash
mnem ingest user-42-chat.json --ntype user-42
```

## Idempotency

Ingesting the same content twice produces the same CID; the second commit is
a no-op (parent points at the same tree). Edit-and-reingest produces a new
CID and a child commit.
