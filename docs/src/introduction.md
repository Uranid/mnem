# Introduction

**mnem** is a knowledge-graph substrate. It stores nodes as content-addressed
objects, retrieves them with vector + sparse + graph signals, and exposes
the result over CLI, HTTP, and MCP surfaces.

## What it does

- **Content-addressed nodes** - every node has a CID; identical content collapses to one node.
- **Versioned commits** - every change is a commit with a parent chain (Git-style for graphs).
- **Hybrid retrieval** - vector (HNSW), sparse (BM25 / SPLADE), and graph traversal in one query.
- **In-process embedder** - bundled ONNX MiniLM-L6-v2 (no Ollama / API keys required).
- **MCP-native** - drop-in memory layer for Claude / Cursor / any MCP client.
- **WASM target** - same core compiles to wasm32 for in-browser use.

## What it is not

- A vector database (it's a graph; vectors are one signal among several).
- An LLM (mnem holds memory; the LLM uses it).
- A finished product. 0.1.0 is the first public cut. See [roadmap](./architecture/overview.md#roadmap).

## Where to next

- [Install](./install.md) - single command per platform.
- [Quickstart](./quickstart.md) - five minutes from zero to retrieve.
- [Core concepts](./core-concepts.md) - what's a CID, what's a commit, what's a label.
