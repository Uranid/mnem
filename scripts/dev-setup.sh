#!/usr/bin/env sh
# mnem dev-setup bootstrap (POSIX).
#
# For new contributors: one command, then `cargo test` works. Use this
# instead of `install.sh` (which is the END-USER installer). Safe to
# re-run: every step checks "already done" before acting.
#
# What it does:
#   1. Verifies `rustup` is installed; prints the install link if not.
#   2. Installs the toolchain pinned in rust-toolchain.toml (1.95 +
#      rustfmt + clippy + wasm32-unknown-unknown target) via rustup.
#   3. Installs the git pre-commit hook if .git/hooks/pre-commit is
#      missing.
#   4. Pulls crate metadata with `cargo fetch` so a later offline
#      `cargo build` works.
#   5. Emits a "ready to go" message with the next-step commands.
#
# No Rust deps are compiled here -- that's what the contributor's first
# `cargo test` does. We just make sure the toolchain + submodules are
# in place.

set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
cd "$here"

log() { printf '%s\n' "dev-setup: $*"; }
warn() { printf '%s\n' "dev-setup: WARN: $*" >&2; }
die() { printf '%s\n' "dev-setup: FATAL: $*" >&2; exit 1; }

# ---- 1. rustup ----
if ! command -v rustup >/dev/null 2>&1; then
    cat <<EOF >&2
dev-setup: FATAL: rustup not on PATH.
dev-setup: install rustup first:
dev-setup:   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
dev-setup: then re-run this script.
EOF
    exit 1
fi

# ---- 2. toolchain ----
# rust-toolchain.toml pins channel / components / targets. rustup
# reads it transparently on any cargo invocation; we force the install
# up-front so the first `cargo test` isn't stuck downloading a fresh
# rustc at random.
log "installing toolchain from rust-toolchain.toml..."
rustup show active-toolchain >/dev/null 2>&1 || true
# `rustup show` both prints and installs what the toolchain file asks
# for; it is the idiomatic way to materialise `rust-toolchain.toml`.
rustup show >/dev/null

# ---- 3. submodules (belt-and-braces; mnem has none today) ----
if [ -f .gitmodules ]; then
    log "updating git submodules..."
    git submodule update --init --recursive
fi

# ---- 4. pre-commit hook ----
hook=".git/hooks/pre-commit"
if [ ! -e "$hook" ]; then
    log "installing pre-commit hook at $hook..."
    cat > "$hook" <<'HOOK'
#!/usr/bin/env sh
# mnem pre-commit hook: cargo fmt + clippy on staged Rust files.
# Skip with `git commit --no-verify` if you really mean to.
set -e
if git diff --cached --name-only --diff-filter=ACM | grep -q '\.rs$'; then
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
fi
HOOK
    chmod +x "$hook"
else
    log "pre-commit hook already present; leaving alone"
fi

# ---- 5. cargo fetch ----
log "running cargo fetch (populates the local registry cache)..."
cargo fetch --locked >/dev/null

# ---- done ----
cat <<'EOF'

dev-setup: ready to go.

Next steps:
  cargo test --workspace --tests --lib           # run unit+integration tests
  cargo fmt --all -- --check                     # formatting gate
  cargo clippy --workspace --all-targets         # lint gate
  cargo doc --no-deps --workspace                # rustdoc (stays 0 warnings)

CONTRIBUTING.md has the full contribution flow.
EOF
