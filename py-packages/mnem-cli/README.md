# mnem-cli (pip)

Install the `mnem` CLI via pip:

```bash
pip install mnem-cli
mnem --version
```

On first run `mnem` downloads the correct prebuilt binary for your platform
from the [GitHub release](https://github.com/Uranid/mnem/releases) and caches
it in `~/.mnem_cli/`. Subsequent calls run the cached binary directly.

## Supported platforms

| Platform | Architecture |
|---|---|
| Linux | x86_64, aarch64 |
| macOS | arm64 (Apple Silicon), x86_64 (via Rosetta 2) |
| Windows | x86_64 |

## Alternatives

- **Cargo**: `cargo install --locked mnem-cli --features bundled-embedder`
- **npm**: `npm install -g mnem-cli`
- **Prebuilt binary**: download from [Releases](https://github.com/Uranid/mnem/releases)

## Python bindings

For the Python API (`import pymnem`), install the companion package:

```bash
pip install mnem-py
```

---

Part of the [mnem](https://github.com/Uranid/mnem) project - Git for AI Agent Knowledge.
