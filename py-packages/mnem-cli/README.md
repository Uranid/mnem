# mnem-cli (pip)

Install the `mnem` CLI via pip:

```bash
pip install mnem-cli
mnem --version
```

`pip` resolves to the per-platform wheel for your OS and architecture; the
prebuilt `mnem` binary and its bundled onnxruntime ship inside the wheel and
are invoked directly. No first-run download, no extra cache directory, no
network required after `pip install`.

## Supported wheels

| Platform | Architecture | Wheel tag |
|---|---|---|
| Windows | x86_64 | `win_amd64` |
| Linux   | x86_64 | `manylinux_2_17_x86_64.manylinux2014_x86_64` |
| Linux   | aarch64 | `manylinux_2_17_aarch64.manylinux2014_aarch64` |
| macOS   | arm64 (Apple Silicon) | `macosx_11_0_arm64` |

Installing from sdist (or on an unsupported platform) succeeds, but invoking
`mnem` exits with a hint to use `cargo install --locked mnem-cli` or download
a prebuilt binary from [Releases](https://github.com/Uranid/mnem/releases).

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
