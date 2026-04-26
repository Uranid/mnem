# mnem-embed-providers

Embedding-provider adapters for mnem (OpenAI, Ollama). Sync, TLS-via-rustls, tokio-free.

Turns a user-configured provider into a concrete `Embedder` that the mnem
CLI, MCP server, HTTP API, and Python bindings use to auto-embed node
summaries on write and query strings on retrieve. All adapters are sync and
built on [`ureq`] with rustls; mnem cannot afford to drag an async runtime
into the CLI or the MCP server. API keys are never stored on disk - config
records the name of the env var (`api_key_env`) and the key is read at
adapter construction. The sibling-crate layout keeps `mnem-core` off any
HTTP dependency, preserving the WASM-embeddability promise from
; see also
 for the trait-surface
choice.

```rust
use mnem_embed_providers::{open, ProviderConfig, OpenAiConfig};

let cfg = ProviderConfig::Openai(OpenAiConfig {
    model: "text-embedding-3-small".into,
    ..Default::default
});
let embedder = open(&cfg)?;
```

Workspace top: [`../../README.md`](../../README.md).

Licensed under Apache-2.0.
