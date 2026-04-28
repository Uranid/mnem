# mnem-core-testutils

Shared test fixtures for the mnem workspace. Internal use only; never published to crates.io (`publish = false` in `Cargo.toml`).

Holds anything that multiple test binaries across the workspace need to
reach for: canned `Node`, `Edge`, `Tree` values, deterministic RNG seeds,
blockstore mocks instrumented for assertion, helper constructors for
`Commit` / `Operation` / `View` shapes. Keeping these in a dev-dependency
crate lets downstream integration tests reuse the same fixtures without
duplicating them across `tests/` trees, and keeps `mnem-core`'s own
production code free of test-only scaffolding.

```rust
// In a downstream crate's Cargo.toml:
// [dev-dependencies]
// mnem-core-testutils = { path = "../mnem-core-testutils" }
```

Workspace top: [`../../README.md`](../../README.md).

Licensed under Apache-2.0.
