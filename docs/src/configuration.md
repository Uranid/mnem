# Configuration

mnem reads config from three sources, in priority order:

1. **Environment variables** - `MNEM_*` (highest precedence)
2. **Per-repo config** - `<repo>/.mnem/config.toml`
3. **User-global config** - `~/.mnem/config.toml`

## Defaults

```toml
# .mnem/config.toml
[embed]
provider = "onnx"
model = "all-MiniLM-L6-v2"

[store]
backend = "redb"        # "redb" | "in-memory"

[retrieve]
top_k = 10
vector_cap = 256
```

## Common environment overrides

| Variable | Effect |
|----------|--------|
| `MNEM_EMBED_PROVIDER` | `onnx` / `ollama` / `openai` / `mock` |
| `MNEM_EMBED_MODEL` | model name (e.g. `all-MiniLM-L6-v2`) |
| `MNEM_EMBED_BASE_URL` | for `ollama` / `openai` providers |
| `MNEM_EMBED_API_KEY_ENV` | name of env var holding the API key |
| `MNEM_ORT_INTRA_THREADS` | pin ONNX runtime thread count (bench harness) |
| `MNEM_BENCH` | enable bench-only label scoping |
| `MNEM_HTTP_ALLOW_NON_LOOPBACK` | allow `mnem-http` to bind 0.0.0.0 (Docker) |

## Provider switching

Embedder, sparse encoder, reranker, and LLM are all configured via
`provider:model` strings - no code change to switch from local ONNX to
hosted Cohere.

```toml
[embed]
provider = "cohere"
model = "embed-english-v3.0"
api_key_env = "COHERE_API_KEY"
```

See [Embedding providers](./guides/embed-providers.md) for the full provider
matrix.
