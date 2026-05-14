# Rich Ingest Pipeline

mnem produces semantically coherent chunks from any source file - not arbitrary text windows.

```bash
mnem ingest README.md          # Markdown
mnem ingest report.pdf         # PDF
mnem ingest legal.txt          # Plain text
mnem ingest main.rs            # Rust source
mnem ingest service.py         # Python source
mnem ingest --recursive src/   # All supported formats, auto-detected
```

## Supported formats

| Format | Extensions | Chunk granularity |
|---|---|---|
| Markdown | `.md`, `.markdown` | Per heading section |
| PDF | `.pdf` | Sentence-aware windows |
| Plain text | `.txt` | Sentence-aware windows; detects document headings |
| Conversation | `.json`, `.jsonl` | Per turn/session |
| Rust | `.rs` | `fn`, `struct`, `enum`, `trait` |
| Python | `.py`, `.pyi` | `def`, `class` |
| JavaScript | `.js`, `.mjs`, `.cjs` | `function`, `class` |
| TypeScript | `.ts`, `.tsx`, `.mts`, `.cts` | `function`, `class`, `interface`, `type alias` |
| Go | `.go` | `func`, `method`, `type` |
| Java | `.java` | `method`, `class` |
| C | `.c`, `.h` | `function` |
| C++ | `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hxx` | `function` |
| Ruby | `.rb`, `.gemspec`, `.rake`, `.erb` | `method`, `class`, `module` |
| C# | `.cs`, `.csx` | `method`, `class`, `interface`, `struct` |
| Other code/config | `.sh`, `.yaml`, `.toml`, `.html`, `.xml`, `.csv`, `.sql`, `.php`, `.swift`, `.kt`, … | Sentence-aware windows |

## Content hash for deduplication

Every Doc node carries a `mnem:content_hash`. Re-ingest an unchanged file and get zero new nodes.

```bash
mnem show --prop mnem:content_hash <doc-cid>
```

## Choosing a chunker manually

The default can be overridden per run:

```bash
mnem ingest notes.txt --chunker paragraph           # double-newline split
mnem ingest notes.txt --chunker recursive           # word-count window
mnem ingest notes.txt --chunker sentence_recursive  # sentence window
mnem ingest code.rs   --chunker structural          # one chunk per section (default for code)
mnem ingest chat.json --chunker session             # group by conversation turn
```

## See also

- [Ingest pipeline guide](../src/guides/ingest.md)
- [mnem ingest CLI reference](../src/cli.md)
- [Deterministic ingest](deterministic-ingest.md)
