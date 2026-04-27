# mnem-cli

Command-line interface for mnem - git for knowledge graphs.

Ships the `mnem` binary: the Git-shaped porcelain that most users reach for
first. The v0.2 surface covers `init / status / log / show / add / query /
diff / ref / config / stats`, plus the onboarding trio `integrate / doctor`
and first-run wizard. Branching and remote operations (push / pull / clone /
remote) land in v0.1.0. The binary walks up to the nearest `.mnem/` like `git`
does, and persists defaults in `config.toml` so common retrieve knobs
(`retrieve.budget`, `retrieve.limit`) can be set once and forgotten.

```bash
mnem init
mnem add node --summary "Alice lives in Berlin and works at Globex"
mnem retrieve "Alice Berlin" --budget 200
```

Workspace top and full quickstart: [`../../README.md`](../../README.md),
[`../../docs/guide/getting-started.md`](../../docs/guide/getting-started.md).

Licensed under Apache-2.0.
