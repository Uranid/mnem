# Embedding providers

mnem decouples embedder from store. Switch providers without re-ingesting.

## Built-in providers

| Provider | Model | Network? | Notes |
|----------|-------|----------|-------|
| `onnx` | `all-MiniLM-L6-v2` (bundled) | no | default; in-process; fastest cold-start |
| `ollama` | any pulled model | local HTTP | e.g. `bge-large`, `nomic-embed-text` |
| `openai` | `text-embedding-3-small`/`-large` | yes | needs `OPENAI_API_KEY` |
| `cohere` | `embed-english-v3.0` | yes | needs `COHERE_API_KEY` |
| `voyage` | `voyage-3` | yes | needs `VOYAGE_API_KEY` |
| `mock` | deterministic blake3 | no | tests / smoke |

## Switching

Edit `<repo>/.mnem/config.toml`:

```toml
[embed]
provider = "ollama"
model = "bge-large"
base_url = "http://127.0.0.1:11434"
```

Or override per-process:

```bash
MNEM_EMBED_PROVIDER=ollama MNEM_EMBED_MODEL=bge-large mnem retrieve "..."
```

After switching, run `mnem reindex` to regenerate the per-commit
embedding sidecar. Node CIDs are unchanged (they don't carry embeddings);
only the sidecar changes.

## Sidecar layout

```
.mnem/
  store.redb              # nodes + commits
  sidecars/
    <embedder-id>/        # one dir per (provider, model) pair
      <commit-cid>.bin    # embedding bucket for that commit
```

Multiple sidecars co-exist. `retrieve` picks the sidecar matching the active
embedder; if missing, it builds on-demand.

## Adding a provider

Implement the `Embedder` trait in `mnem-embed-providers/src/<your>.rs`,
gate behind a feature flag, register in the provider registry. See
 for the
contract.
