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
model = "bge-large-en-v1.5"
```

Retrieval returns all matching results by default (no hard cap). The vector candidate pool defaults to 256 entries. Set persistent overrides with `mnem config set retrieve.limit N` or use `--limit N` per call.

## Common environment overrides

| Variable | Effect |
|----------|--------|
| `MNEM_EMBED_PROVIDER` | `onnx` / `ollama` / `openai` / `mock` |
| `MNEM_EMBED_MODEL` | model name (e.g. `all-MiniLM-L6-v2`) |
| `MNEM_EMBED_BASE_URL` | for `ollama` / `openai` providers |
| `MNEM_EMBED_API_KEY_ENV` | name of env var holding the API key |
| `MNEM_ORT_INTRA_THREADS` | pin ONNX runtime thread count (bench harness) |
| `MNEM_BENCH` | enable bench-only label scoping |
| `MNEM_HTTP_ALLOW_NON_LOOPBACK` | allow `mnem http` to bind 0.0.0.0 (Docker) |
| `MNEM_DISABLE_GLOBAL_CONFIG` | set to `1` to skip reading `~/.mnem/config.toml`; useful in tests for isolation |

## Provider switching

Embedder, sparse encoder, reranker, and LLM are all configured via
`provider:model` strings - no code change to switch from local ONNX to
hosted Cohere.

```toml
[embed]
provider = "cohere"
model = "embed-english-v3.0"
api_key_env = "COHERE_API_KEY"

[rerank]
provider = "cohere"
model = "rerank-english-v3.0"
api_key_env = "COHERE_API_KEY"
```

See [Embedding providers](./guides/embed-providers.md) for the full provider
matrix.

## API key guardrail

mnem rejects config values that look like raw secrets to prevent accidental
credential leakage:

- The key names `embed.api_key` and `rerank.api_key` are always rejected. Use
  `api_key_env` (the name of an environment variable) instead.
- Any value under the `embed.*` or `rerank.*` namespace that starts with `sk-`
  is also rejected regardless of key name.

```bash
# WRONG - will error
mnem config set embed.api_key sk-abc123

# CORRECT - store the key name, not the value
mnem config set embed.api_key_env OPENAI_API_KEY
export OPENAI_API_KEY=sk-abc123
```

## Per-repo isolation

`mnem config set` requires an initialised `.mnem/` repo. Running it outside a
repo (or before `mnem init`) returns an error. This is intentional - per-repo
config only makes sense inside a repo.
