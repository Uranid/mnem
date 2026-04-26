# Fuzz crash-regression policy

When a fuzz run hits a crash (locally or in nightly CI), the input
must land as a permanent regression seed so the same shape can never
re-crash silently. The flow is short and mechanical.

## 1. Locate the crash input

Local run:

```bash
ls fuzz/artifacts/<target>/        # crash-<sha1>, leak-<sha1>, timeout-<sha1>
```

Nightly CI: download the `fuzz-crash-<target>` artifact from the
failing workflow run page on GitHub Actions.

## 2. Minimise

`cargo fuzz tmin` strips bytes from the input while keeping the
crash reproducible. Always commit the *minimised* bytes, never the
raw one-shot crash file - smaller inputs read as regression tests.

```bash
cargo +nightly fuzz tmin <target> fuzz/artifacts/<target>/crash-<sha1>
# minimised bytes land at fuzz/artifacts/<target>/minimized-from-*
```

## 3. Commit to the regressions corpus

Rename the minimised file by its sha256 (so the name is
content-addressed and collisions are impossible) and drop it in
the target's `regressions/` sub-directory:

```bash
HASH=$(sha256sum fuzz/artifacts/<target>/minimized-from-XYZ | cut -c1-16)
cp fuzz/artifacts/<target>/minimized-from-XYZ \
   fuzz/corpus/<target>/regressions/${HASH}.seed
git add fuzz/corpus/<target>/regressions/${HASH}.seed
git commit -m "fuzz(<target>): regression seed for <short crash description>"
```

libFuzzer walks `fuzz/corpus/<target>/**` recursively; the
`regressions/` sub-directory is picked up automatically on the next
run. Nothing else to wire up.

## 4. File the issue

Open a GitHub issue titled `fuzz: <target> crash on <summary>`
with:

- the minimised input (attached or pasted as hex for < 256-byte seeds);
- the stack from `RUST_BACKTRACE=1 cargo +nightly fuzz run <target> <crash>`;
- a link to the nightly CI run that surfaced it (if any).

Tag `A-fuzz` and the relevant component label (`A-transport`,
`A-codec`).

## 5. Fix, then verify

Once the fix lands, run the full corpus locally to confirm the
regression seed no longer crashes:

```bash
cargo +nightly fuzz run <target> -- -runs=0  # replays every seed once, exits
```

A green `-runs=0` run is the merge gate for the fix PR.

## 6. Corpus minimisation (optional, weekly)

`cargo fuzz cmin` compacts the corpus by dropping seeds whose
coverage is subsumed by others. Too much compaction kills the
benefit of a growing seed bank, so this is run weekly at most and
only by a maintainer:

```bash
# Weekly maintenance, optional
cargo +nightly fuzz cmin car_import
cargo +nightly fuzz cmin walk_ipld
```

A maintainer bot could open this as a PR on a weekly cron; it
isn't wired up on this branch - document-only pattern. If we see
corpus growth past ~1 MiB per target, spin it up then.

## Invariants

- Regression seeds are **never removed** except by replacement
  with a smaller minimisation of the same crash class.
- The `regressions/` directory is **not** corpus-minimised by
  `cargo fuzz cmin` runs (pass `--` and explicit corpus paths
  excluding `regressions/` if the compaction would touch them).
- Every fuzz-target PR adds at least a compile-check through
  `fuzz/`; the nightly CI workflow is the only continuous
  signal and it does not block PR merges.
