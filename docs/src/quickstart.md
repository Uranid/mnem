# Quickstart

Five minutes from zero to retrieve.

## 1. Install

```bash
cargo install --locked mnem-cli
```

(See [Install](./install.md) for other platforms.)

## 2. Initialise a repo

```bash
mkdir my-graph && cd my-graph
mnem init
```

This creates `.mnem/` with default config (in-process MiniLM embedder, redb store).

## 3. Ingest

```bash
mnem ingest README.md
mnem ingest docs/*.md
mnem ingest <(echo '{"text": "the cat sat on the mat", "label": "demo"}') --json
```

## 4. Retrieve

```bash
mnem retrieve "what does this project do"
mnem retrieve "what is X" --label demo --top-k 5
```

## 5. Serve over HTTP (optional)

```bash
mnem-http --bind 127.0.0.1:9876 --repo .
curl http://127.0.0.1:9876/v1/retrieve -d '{"text": "what does this do"}'
```

## 6. Wire into Claude / Cursor (optional)

```bash
mnem mcp install
```

Adds an MCP server entry to your client config; subsequent agent turns can
call `mnem_retrieve` and `mnem_ingest` natively.

## Next steps

- [CLI reference](./cli.md) for every flag.
- [MCP server](./mcp.md) for agent integrations.
- [Retrieval tuning](./guides/retrieval-tuning.md) for top-K, hybrid, and graph traversal options.
