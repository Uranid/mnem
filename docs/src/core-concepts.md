# Core concepts

Three primitives. Everything else is composed from these.

## Node

A node is content + metadata, addressed by its **CID** (content identifier
derived from a hash of canonical bytes). Two nodes with identical content
collapse to one CID. Nodes carry:

- `text` - the unit of content (a sentence, a chunk, a fact)
- `label` - string scope; queries can filter to a label
- `metadata` - opaque JSON map for caller-defined tags

The embedding lives in a per-commit **sidecar bucket**, not on the node, so
two nodes with the same text but different embedders share one CID.

## Commit

A commit is a snapshot of the graph at a point in time. Every ingest, every
edit, every tombstone produces a new commit. Commits chain by parent CID;
the head commit is the working tree's "current state". Older commits are
immutable and reachable.

## Label

A label is an opt-in namespace string attached to nodes at ingest time. Used
for:

- per-user / per-conversation isolation in agent memory
- bench harness scoping (per-question, per-document)
- coarse multi-tenancy

A query without a label sees the whole repo; a query with a label sees only
nodes carrying that label.

## Retrieval lanes

Every `retrieve` call fans out across three lanes and fuses the results:

1. **Vector** - HNSW over the per-commit sidecar embeddings
2. **Sparse** - BM25 / SPLADE (optional, feature-gated)
3. **Graph** - n-hop traversal over authored edges, optionally PPR-scored

Lanes are configurable. Vector-only is the default and is what the 0.1.0
benchmarks measure.
