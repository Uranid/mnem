# mnem-rerank-providers

Cross-encoder reranker adapters for mnem (Cohere, Voyage, Jina). Sync, TLS-via-rustls, tokio-free.

Provides production adapters for the `Reranker` trait defined in
`mnem-core` (`mnem_core::rerank`). Per
, the retrieve pipeline
invokes a reranker as an optional post-fusion pass over the top-K. The
cross-encoders here read `(query, candidate)` pairs jointly, which is what
lets them handle compositional paraphrase that dense + sparse bi-encoder
fusion misses ("father's sister" vs "aunt"). All adapters are sync over
`ureq` + rustls, matching the discipline in `mnem-embed-providers`. API
keys stay in env vars; config only records the env-var name.

```rust
use std::sync::Arc;
use mnem_rerank_providers::{open, ProviderConfig, CohereConfig};
use mnem_core::rerank::Reranker;

let cfg = ProviderConfig::Cohere(CohereConfig {
    model: "rerank-v3.5".into,
    ..Default::default
});
let rr: Arc<dyn Reranker> = open(&cfg)?;
```

Workspace top: [`../../README.md`](../../README.md).

Licensed under Apache-2.0.
