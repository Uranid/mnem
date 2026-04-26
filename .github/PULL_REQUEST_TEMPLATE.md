<!--
Thanks for the PR. A short form is fine; the goal is to make review fast,
not paperwork-y. Delete sections that don't apply.
-->

## What this PR does

<!-- One or two sentences. Focus on WHY, not just WHAT. -->

## Linked issue

<!-- e.g. "Fixes #42" -->

## Type of change

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (API or wire-format)
- [ ] Documentation only
- [ ] CI / tooling only
- [ ] Refactor (no behavior change)

## Testing done

<!--
- What new tests did you add?
- Did you run `cargo test --workspace --tests --lib` locally?
- Did you exercise the relevant example (e.g. agent_memory_killer)?
-->

## Checklist

- [ ] `cargo fmt --all` is clean
- [ ] `cargo clippy --workspace --all-targets` produces no new warnings
- [ ] `cargo test --workspace --tests --lib` passes locally
- [ ] New public API has rustdoc with `# Errors` if it returns `Result`
- [ ] SPEC.md updated if the wire format changed
- [ ] No em dashes introduced (project style: hyphens only)
- [ ] No `unsafe` added to `mnem-core` (the crate is `#![forbid(unsafe_code)]`)

## Breaking changes

<!--
If the PR breaks a public API or the on-disk format, call it out here.
Note the migration path and the version bump expected.
-->
