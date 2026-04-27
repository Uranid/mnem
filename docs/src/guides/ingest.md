# Ingest pipeline

`mnem ingest` is the only path content takes into the graph. The pipeline:

```
parse -> chunk -> extract -> embed -> commit
```

## Sources

- file path (`mnem ingest README.md`)
- glob (`mnem ingest 'docs/**/*.md'`)
- stdin (`cat data.txt | mnem ingest -`)
- structured JSON (`mnem ingest data.json --json`)

## Chunking

Default: ~1k-token chunks with sentence-boundary alignment. Override via
config:

```toml
[ingest]
chunk_size_tokens = 512
chunk_overlap_tokens = 50
```

Document-aware chunkers exist for code (Tree-sitter) and for Markdown
(heading-aware). Auto-detected by file extension.

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

Pass `--label <str>` to scope the ingested nodes:

```bash
mnem ingest user-42-chat.json --label user-42 --json
```

Subsequent `retrieve` calls with `--label user-42` will see only this scope.

## Idempotency

Ingesting the same content twice produces the same CID; the second commit is
a no-op (parent points at the same tree). Edit-and-reingest produces a new
CID and a child commit.
