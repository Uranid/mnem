# mnem-llm-providers

Text-generation adapters for mnem (OpenAI chat, Ollama chat) for HyDE, multi-query, and future LLM-in-the-loop features. Sync, TLS-via-rustls, tokio-free.

Provides production adapters for the `TextGenerator` trait defined in
`mnem-core` (`mnem_core::llm`). Today's primary user is
`mnem retrieve --hyde`, which asks the configured LLM to hypothesise an
answer and embeds that answer for denser recall. The planned multi-query /
RAG-Fusion variant shares the same trait, and future features (query
rewriting, answer synthesis, retrieval grading) will build on top of the
same surface. All adapters are sync over `ureq` + rustls, matching the
discipline in `mnem-embed-providers` and `mnem-rerank-providers`. API keys
stay in env vars; config only records the env-var name.

```rust
use mnem_llm_providers::{open, ProviderConfig, OpenAiLlmConfig};

let cfg = ProviderConfig::OpenAi(OpenAiLlmConfig {
    model: "gpt-4o-mini".into,
    ..Default::default
});
let gen = open(&cfg)?;
```

Workspace top: [`../../README.md`](../../README.md).

Licensed under Apache-2.0.
