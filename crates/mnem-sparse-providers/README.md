# mnem-sparse-providers

Learned-sparse encoder adapters for mnem (SPLADE, BGE-M3-sparse, opensearch-doc-v3-distill). Sync, TLS-via-rustls, tokio-free.

Implements `mnem_core::sparse::SparseEncoder` for three backends per
. The
**sidecar** transport (always available) POSTs to a local Python service
running the reference SPLADE / BGE-M3 implementation, which keeps the
install light and defers weights + tokenization to whatever Python ML
infra the user already runs. The **ONNX** backend (feature `onnx`) runs
inference in-process via `ort` + `tokenizers`; fastest, but pulls a heavy
dep tree (onnxruntime + SIMD tokenizers) and so is feature-gated to keep
WASM targets clean. The **mock** backend re-exports
`mnem_core::sparse::MockSparseEncoder` for tests and dry-run benchmarks.
Learned-sparse is the primary sparse lane after
 retired the
in-tree BM25 lane.

Workspace top: [`../../README.md`](../../README.md).

Licensed under Apache-2.0.
