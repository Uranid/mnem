# Install

mnem ships a single `mnem` binary plus optional Python and HTTP daemons. Pick
the source that matches your platform.

## From Cargo (any platform with Rust toolchain)

```bash
cargo install --locked mnem-cli
mnem --version
```

Requires Rust 1.95+ (see `rust-toolchain.toml`).

## From npm (Node.js users)

```bash
npm install -g mnem-cli
mnem --version

# or one-shot via npx
npx mnem-cli --version
```

Downloads the prebuilt native binary for your platform at install time. Node 18+ required. No Rust toolchain needed.

## From PyPI (Python users)

```bash
pip install mnem-cli
mnem --version
```

The PyPI package ships the same `mnem` binary as a manylinux / macOS / Windows wheel.

## From a release binary

Download the platform tarball from the latest [GitHub release](https://github.com/Uranid/mnem/releases/latest):

```bash
curl -L https://github.com/Uranid/mnem/releases/latest/download/mnem-linux-x86_64.tar.gz | tar xz
sudo mv mnem /usr/local/bin/
mnem --version
```

Replace `linux-x86_64` with `linux-aarch64` / `macos-x86_64` / `macos-aarch64` / `windows-x86_64.zip` as appropriate.

## Per-OS package managers

After v0.2.0, mnem ships only via **Cargo** and **PyPI**. The Homebrew
tap, AUR, Nix, winget, and scoop channels have been dropped in favour
of a lean three-channel model (cargo / PyPI / npm). The Cargo channel
supports `bundled-embedder`, `bundled-embedder-cuda`,
`bundled-embedder-directml` feature flags.

<details>
<summary>macOS / Linux / Windows</summary>

```bash
# npm (Node 18+, no Rust toolchain needed)
npm install -g mnem-cli

# Cargo (any platform with Rust 1.95+)
cargo install --locked mnem-cli --features bundled-embedder

# or via cargo-binstall (faster, downloads prebuilt)
cargo binstall mnem-cli

# PyPI (Python users)
pip install mnem-cli
```

</details>

<details>
<summary>Docker</summary>

```bash
docker run --rm -p 9876:9876 ghcr.io/uranid/mnem:latest http serve
```

</details>

<details>
<summary>WASM (in-browser)</summary>

```bash
cargo build --release --target wasm32-unknown-unknown -p mnem-core
```

See [`crates/mnem-core/README.md`](https://github.com/Uranid/mnem/blob/main/crates/mnem-core/README.md) for embedding examples.

</details>

## Verify

```bash
mnem --version
mnem doctor
```

`mnem doctor` probes embedder, store, and config - useful first command after install.
