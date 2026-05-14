# Deterministic Ingest

mnem ingests documents without an LLM. The same bytes always produce the same nodes, on any machine.

```bash
mnem ingest architecture.md        # parses into Doc + Chunk + Entity nodes
mnem ingest main.rs                # code: one chunk per function/struct
mnem ingest --recursive src/       # walk a directory, auto-detect all types
```

## No LLM at ingest

The pipeline is fully local and offline. No network calls, no API keys, no non-determinism. The `mnem:content_hash` on each Doc node lets callers skip files whose content has not changed.

## Supported input formats

| Format | Extensions | Chunking |
|---|---|---|
| Markdown | `.md`, `.markdown` | One chunk per heading section |
| PDF | `.pdf` | Sentence-aware windows |
| Plain text | `.txt` | Sentence-aware windows; auto-splits on document headings |
| Conversation | `.json`, `.jsonl` | One chunk per conversation turn/session |
| Rust | `.rs` | One chunk per function, struct, enum, trait |
| Python | `.py`, `.pyi` | One chunk per def/class |
| JavaScript | `.js`, `.mjs`, `.cjs` | One chunk per function/class |
| TypeScript | `.ts`, `.tsx`, `.mts`, `.cts` | One chunk per function/class/interface/type alias |
| Go | `.go` | One chunk per func/method/type declaration |
| Java | `.java` | One chunk per method/class |
| C | `.c`, `.h` | One chunk per function |
| C++ | `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hxx` | One chunk per function |
| Ruby | `.rb`, `.gemspec`, `.rake`, `.erb` | One chunk per method/class/module |
| C# | `.cs`, `.csx` | One chunk per method/class/interface/struct |
| Other code/config | `.sh`, `.yaml`, `.toml`, `.html`, `.xml`, `.csv`, `.sql`, `.php`, `.swift`, `.kt`, … | Sentence-aware windows |

## Optional keyphrase enrichment

For stronger sparse retrieval signal, KeyBERT extraction is available at ingest time:

```bash
mnem ingest --extractor keybert notes.md
```

KeyBERT uses a local model - no LLM call, no network required.

## Fuzz testing

The ingest parsers are fuzz-harnessed. Malformed or adversarial input is handled safely without panics.

## See also

- [Rich ingest pipeline](rich-ingest.md) - supported formats, content hash, chunker options
- [Ingest pipeline guide](../src/guides/ingest.md)
- [mnem ingest CLI reference](../src/cli.md)
