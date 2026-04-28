# Storage

Content-addressed objects, commit-scoped sidecars, one durable backend.

## Object types

| Object | Carries | Identity |
|--------|---------|----------|
| Node | text + label + metadata | content-derived CID |
| Commit | parent CID + tree CID + sidecar CID + author | CID over canonical bytes |
| Tree | sorted node-CID list | CID |
| Sidecar | per-commit embedding bucket | CID |

CIDs use [IPLD DAG-CBOR](https://ipld.io) canonical bytes + BLAKE3.
Determinism is enforced via canonical sort + canonical encoding.

## Backends

| Backend | When to use |
|---------|-------------|
| `redb` | persistent on disk (default); single-file, ACID |
| `in-memory` | bench harnesses, ephemeral agent sessions |

Backend is chosen at startup via config or `--in-memory` flag on
`mnem-http`. The two backends share the same object layer; switching
does not change CIDs.

## Vector sidecar

As of 0.1.0, embeddings are no longer part of node identity. The commit
holds a pointer to an `EmbeddingBucket` keyed by node CID. Switching
embedders does not invalidate node CIDs; only the sidecar CID changes.

This means:
- swapping embedder = re-embed only, no re-ingest
- two embedders = two sidecars sharing one node-tree
- sparse / dense / community signals all live alongside as separate sidecars

## Garbage collection

Tombstones are soft. A `tombstone` op writes a marker; the node remains
reachable from prior commits. A `gc` pass walks unreachable nodes and
sidecars from the head and prunes them. Not yet automated in 0.1.0;
incremental GC is queued for v1.1.
