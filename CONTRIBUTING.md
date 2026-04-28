# Contributing to mnem

Thanks for your interest. mnem is an early-stage project, and the format
spec is deliberately locked down - the whole value proposition is that two
independent implementations produce byte-identical objects. This guide
exists to keep contributions aligned with that bar.

## Before you open anything

1. **Read [SPEC.md](docs/SPEC.md)** if the change touches the wire format,
   object types, or semantics. An engineer reading only SPEC should still
   be able to build a compatible implementation after your change lands.
2. **Check [ROADMAP.md](docs/ROADMAP.md)** - the work may already be
   planned for a specific phase.

## Types of contribution

### Bug fix

Open an issue describing the bug (minimal reproduction preferred), then
send a PR. Small self-contained fixes can skip the issue step, but a one-
line "what broke and how the fix is local" in the PR body saves review
time.

### New feature

- **Core library feature** (anything in `mnem-core`): open an issue
  first. If the change affects SPEC, the issue conversation should
  converge on a design before code lands.
- **Backend crate** (`mnem-backend-*`): open an issue; note any SPEC §7
  implications.
- **CLI or tooling**: open an issue with the UX sketch.

### Documentation

PRs welcome directly.

### Design changes

For non-trivial design changes, open an issue first to discuss the approach.

## Development setup

```bash
git clone https://github.com/Uranid/mnem.git
cd mnem
rustup show                                  # installs 1.95 from rust-toolchain.toml
cargo test --workspace --tests --lib         # full suite (279 tests today)
cargo run --release -p mnem-core --example agent_memory_killer
```

Windows developers: the workspace is Windows-clean; no WSL required.
Release builds on Windows MSVC use the normal Rust toolchain.

## PR workflow

1. **Fork + branch.** Branch name should reflect scope: `fix/...`,
   `feat/...`, `docs/...`.
2. **Commit messages**: one short header (`<type>: <imperative summary>`,
   max ~72 chars) followed by a body explaining *why*, not *what*. Match
   the tone of the existing git log.
3. **Run the pre-flight**:
   ```bash
   cargo fmt --all
   cargo clippy --workspace --all-targets
   cargo test --workspace --tests --lib
   ```
4. **Update tests**: new public behavior needs a test. Prefer integration
   tests under `crates/<crate>/tests/` over `#[cfg(test)] mod tests`
   when exercising the repo facade end-to-end.
5. **Update docs**: if a public API, SPEC field, or on-disk shape changed,
   update SPEC.md and the relevant rustdoc.
6. **Open the PR** using the template. Link the issue, describe what
   broke vs what changed, list breaking changes if any.

## What we look for in review

- **SPEC conformance**: every byte that leaves the process must match
  the canonical form.
- **No hidden I/O in `mnem-core`**: the crate is deliberately
  filesystem-free, terminal-free, and network-free. I/O lives in
  backends or binaries.
- **`#![forbid(unsafe_code)]`**: unsafe is not accepted in `mnem-core`.
  In backend crates, every `unsafe` block must have a `// SAFETY:`
  comment.
- **Determinism**: if your code emits bytes, those bytes must be a pure
  function of the inputs. Wall-clock times, random UUIDs outside their
  allowed scopes, and non-deterministic iteration orders all break the
  "same input, same hash" invariant.
- **Tests green on Linux, macOS, Windows**: the CI matrix catches
  platform drift; assume PRs that pass locally on one OS may still need
  fixes for the others.

## Releases

Releases are cut by pushing a bumped `[workspace.package].version` to the
`release` branch. `.github/workflows/release.yml` runs the matrix, tags
`v<version>`, and publishes a GitHub release. Pre-release suffixes
(`-alpha.N`, `-beta.N`, `-rc.N`) are auto-detected and flagged.

## Security

See [SECURITY.md](SECURITY.md). Do not file security issues in the public
issue tracker.

## Code of conduct

All contributors are expected to uphold the
[Contributor Covenant](CODE_OF_CONDUCT.md).

## Questions

Open a GitHub issue with the `question` label. The maintainer reads every
one.
