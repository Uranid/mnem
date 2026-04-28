# mnem-http

HTTP JSON API for mnem - REST surface over the core repo operations.

Library + binary pair. The library exposes `app(repo_dir)` which builds an
axum `Router` that wraps an open `ReadonlyRepo` on `repo_dir/.mnem`,
auto-initialising if needed. The v1 surface is `GET /v1/healthz`,
`GET /v1/stats`, `POST /v1/nodes`, `GET /v1/nodes/{id}`,
`DELETE /v1/nodes/{id}`, and `GET /v1/retrieve?text=&budget=&limit=` (dense
vector retrieval, embedder required when `text` is set). Per
 this crate is the
only place tokio enters the workspace; `mnem-core` stays WASM-clean. The
binary binds to loopback by default and emits a loud stderr warning when
exposed to a network interface, since v1 has no auth layer.

```bash
mnem-http --repo /path/to/project --bind 127.0.0.1:9876
```

Workspace top: [`../../README.md`](../../README.md). Spec:
[`../../docs/SPEC.md`](../../docs/SPEC.md).

Licensed under Apache-2.0.
