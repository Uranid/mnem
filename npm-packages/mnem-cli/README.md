# mnem-cli

**Git for knowledge graphs** — versioned, content-addressed, embeddable agent memory.

## Install

```bash
npm install -g mnem-cli
```

Or with bundled embedder (recommended):

```bash
cargo install --locked mnem-cli --features bundled-embedder
```

## Quick Start

```bash
# Initialize a new memory graph
mnem init

# Add a memory node
mnem add node --label Person --summary "Alice works at Acme"

# Query your memory
mnem retrieve --text "who works at Acme"
```

## MCP Server

Wire mnem into Claude Desktop, Cursor, or any MCP client:

```bash
mnem mcp --repo /path/to/your/project
```

Then configure your client to connect via stdio to the `mnem mcp` command.

## HTTP API

Start the HTTP server:

```bash
mnem http --bind 127.0.0.1:9876 --repo /path/to/project
```

Then query via REST:

```bash
curl -X POST http://127.0.0.1:9876/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{"text": "your query", "budget": 500}'
```

## Docs

- [CLI Reference](https://github.com/Uranid/mnem/blob/main/docs/src/cli.md)
- [MCP Integration](https://github.com/Uranid/mnem/blob/main/docs/src/mcp.md)
- [Configuration](https://github.com/Uranid/mnem/blob/main/docs/src/configuration.md)

## License

Apache-2.0