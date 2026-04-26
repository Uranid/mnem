# mnem dev-setup bootstrap (PowerShell).
#
# Windows parity for scripts/dev-setup.sh. For new contributors on
# Windows: one command, then `cargo test` works. Use this INSTEAD of
# `install.ps1` (which is the end-user installer). Safe to re-run.
#
# What it does:
#   1. Verifies `rustup` is installed; prints the install link if not.
#   2. Installs the toolchain pinned in rust-toolchain.toml.
#   3. Installs a git pre-commit hook if one is not already present.
#   4. Runs `cargo fetch` so a later offline build works.
#   5. Prints "ready to go" with the next-step commands.

$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $PSScriptRoot
if (-not $here) { $here = Split-Path -Parent $PSCommandPath }
$root = Join-Path $here ''
Set-Location -Path $root\..

function Info([string]$m) { Write-Host "dev-setup: $m" }
function Fail([string]$m) { Write-Host "dev-setup: FATAL: $m" -ForegroundColor Red; exit 1 }

# ---- 1. rustup ----
if (-not (Get-Command rustup -ErrorAction SilentlyContinue)) {
    Write-Host "dev-setup: FATAL: rustup not on PATH." -ForegroundColor Red
    Write-Host "dev-setup: install rustup first:" -ForegroundColor Red
    Write-Host "dev-setup:   https://rustup.rs" -ForegroundColor Red
    Write-Host "dev-setup: then re-run this script." -ForegroundColor Red
    exit 1
}

# ---- 2. toolchain ----
# rust-toolchain.toml pins channel + components + wasm target. `rustup
# show` both prints the active toolchain and materialises anything the
# file asks for that is not already installed.
Info "installing toolchain from rust-toolchain.toml..."
& rustup show | Out-Null

# ---- 3. submodules (none today, belt-and-braces) ----
if (Test-Path .gitmodules) {
    Info "updating git submodules..."
    & git submodule update --init --recursive
}

# ---- 4. pre-commit hook ----
$hook = '.git/hooks/pre-commit'
if (-not (Test-Path $hook)) {
    Info "installing pre-commit hook at $hook..."
    $hookBody = @'
#!/usr/bin/env sh
# mnem pre-commit hook: cargo fmt + clippy on staged Rust files.
# Skip with `git commit --no-verify` if you really mean to.
set -e
if git diff --cached --name-only --diff-filter=ACM | grep -q '\.rs$'; then
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
fi
'@
    Set-Content -Path $hook -Value $hookBody -Encoding utf8
} else {
    Info "pre-commit hook already present; leaving alone"
}

# ---- 5. cargo fetch ----
Info "running cargo fetch (populates the local registry cache)..."
& cargo fetch --locked | Out-Null

# ---- done ----
Write-Host ""
Write-Host "dev-setup: ready to go."
Write-Host ""
Write-Host "Next steps:"
Write-Host "  cargo test --workspace --tests --lib           # run unit+integration tests"
Write-Host "  cargo fmt --all -- --check                     # formatting gate"
Write-Host "  cargo clippy --workspace --all-targets         # lint gate"
Write-Host "  cargo doc --no-deps --workspace                # rustdoc (stays 0 warnings)"
Write-Host ""
Write-Host "CONTRIBUTING.md has the full contribution flow."
