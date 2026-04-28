# Install

mnem ships a single `mnem` binary plus optional Python and HTTP daemons. Pick
the source that matches your platform.

## From Cargo (any platform with Rust toolchain)

```bash
cargo install --locked mnem-cli
mnem --version
```

Requires Rust 1.95+ (see `rust-toolchain.toml`).

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

<details>
<summary>macOS</summary>

```bash
brew install mnem      # tap added at 0.1.0+
```

</details>

<details>
<summary>Linux</summary>

```bash
# Arch
yay -S mnem
# Nix
nix-env -iA nixpkgs.mnem
# Cargo (works everywhere)
cargo install --locked mnem-cli
```

</details>

<details>
<summary>Windows</summary>

```powershell
winget install mnem
# or
scoop install mnem
```

</details>

<details>
<summary>Docker</summary>

```bash
docker run --rm -p 9876:9876 ghcr.io/uranid/mnem-http:latest
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
