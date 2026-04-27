# mnem fuzz harness

Coverage-guided fuzz targets for mnem's highest-value parsers.

## Why

Three input surfaces handle untrusted bytes on every deployment:

- **CAR import** (`mnem_transport::import`): offline transport boundary.
  Anything that round-trips through a `.car` file or the future remote
  `/fetch-blocks` surface must survive arbitrary byte input without a
  panic or an OOM.
- **DAG-CBOR decode + link walk** (`mnem_core::codec::extract_links`):
  the canonical on-wire encoding. Every block pulled from a blockstore,
  every REST body that parses into `Ipld`, every MCP tool arg after
  round-trip goes through this path.
- **Ingest PDF parser** (`mnem_ingest::pdf::parse_pdf`): every
  `SourceKind::Pdf` ingest routes user-supplied bytes through
  `pdf-extract`. Phase-B5b already wraps the call in `catch_unwind`;
  the fuzz target regression-guards that contract.

All three surfaces already have hand-written unit tests for their
defined-behaviour corners (`import_rejects_tampered_cid`,
`walk_ipld_rejects_deeply_nested_structure`,
`malformed_bytes_are_error_not_panic`). Fuzzing catches the
*undefined-behaviour* corners - input shapes a human test author
never thought of.

## Targets

| Target | What it exercises | Expected invariant |
|--------|-------------------|---------------------|
| `car_import` | `import_with_limit` against a `MemoryBlockstore` with a 256 KiB cap | never panics; rejects malformed with `Err` |
| `walk_ipld`  | `from_canonical_bytes` + `extract_links`                            | never panics; bounded depth via `WALK_IPLD_MAX_DEPTH` |
| `ingest_pdf` | `mnem_ingest::pdf::parse_pdf` on arbitrary bytes                   | never panics; malformed input returns `Err(ParseFailed)` |

## Prerequisites

Fuzzing requires the nightly Rust toolchain and `cargo-fuzz`:

```bash
rustup install nightly
cargo install cargo-fuzz
```

On Windows, libFuzzer links through the MSVC toolchain; on Linux / macOS
the default system linker is sufficient.

## Running

From the repository root:

```bash
# 60-second run of the CAR import target
cargo +nightly fuzz run car_import -- -max_total_time=60

# 60-second run of the DAG-CBOR walk target
cargo +nightly fuzz run walk_ipld -- -max_total_time=60

# 60-second run of the ingest-PDF target
cargo +nightly fuzz run ingest_pdf -- -max_total_time=60

# Extended run (10 minutes per target) for a nightly CI sweep
cargo +nightly fuzz run car_import -- -max_total_time=600
cargo +nightly fuzz run walk_ipld  -- -max_total_time=600
cargo +nightly fuzz run ingest_pdf -- -max_total_time=600
```

Without `cargo-fuzz` installed you can still drive the targets:

```bash
# Compile-check only (runs on any toolchain that accepts nightly deps)
cd fuzz && cargo +nightly check

# Build the libFuzzer binary directly (nightly required for the
# SanitizerCoverage LLVM pass)
cd fuzz && cargo +nightly build --release --bin car_import
./target/release/car_import -max_total_time=60
```

## Corpora

Seed corpora live under `fuzz/corpus/<target>/` and ship
**checked-in** on this branch. Both corpora mix round-trip-valid
seeds (hand-built CAR v1 archives and DAG-CBOR blocks with correct
CIDs / canonical encoding) with deliberately adversarial seeds
(truncations, bad varints, wrong multihash, non-canonical CBOR,
oversized headers, depth bombs, unknown tags, indefinite-length
forms, trailing garbage).

Current seed inventory:

| Target       | Round-trip valid | Adversarial | Total |
|--------------|------------------|-------------|-------|
| `car_import` | 30               | 27          | 57    |
| `walk_ipld`  | 39               | 31          | 70    |

Seeds were generated deterministically from the CAR v1 spec
(<https://ipld.io/specs/transport/car/carv1/>) and the DAG-CBOR spec
(<https://ipld.io/specs/codecs/dag-cbor/spec/>); they are not
pulled from any external dataset and carry no PII. The seed files
are small (median < 256 bytes, max < 4 KiB) so the repo footprint
stays well under 64 KiB for the whole corpus.

### Adding more seeds

Drop any CAR or DAG-CBOR bytes that exercise an interesting shape
into the matching directory:

```bash
# From your own mnem-exported CAR bundle
cp ~/path/to/export.car fuzz/corpus/car_import/

# From raw DAG-CBOR blocks extracted via `mnem export` + `mnem inspect`
cp ~/path/to/block.cbor fuzz/corpus/walk_ipld/

# For ingest_pdf: drop any small PDFs you already have
cp ~/Documents/*.pdf fuzz/corpus/ingest_pdf/
```

libFuzzer picks up every file in the corpus directory automatically
on the next `cargo +nightly fuzz run`.

## Nightly CI

The `.github/workflows/fuzz.yml` workflow runs both targets nightly
at 05:00 UTC for 600 seconds each, with a 2-target matrix. It caches
the corpus directories across runs so the coverage map accumulates
week-over-week. On a crash, it uploads the crash input **and** the
full corpus-plus-regressions directory as artifacts for post-mortem.

Badge (optional; requires Gist endpoint - see below):

```markdown
![fuzz](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/<user>/<gist-id>/raw/fuzz.json)
```

The badge relies on a shields.io endpoint-JSON hosted on a user
Gist that nightly CI updates on success. Setting that up is a user
task (needs a Gist plus a `GIST_TOKEN` secret with `gist` scope);
until the Gist exists the badge URL returns a 404 and the line can
simply be omitted from the README. The workflow itself works
end-to-end without the Gist; only the badge render is gated on it.

## Findings triage

If a run hits a crash, follow the procedure in
[`REGRESSIONS.md`](REGRESSIONS.md): minimise with
`cargo fuzz tmin`, commit the minimised input to
`fuzz/corpus/<target>/regressions/`, open an issue linking the
crash artifact from the CI run.

## Workspace isolation

The `fuzz/` directory is deliberately NOT a member of the main
Cargo workspace. `cargo build --workspace` on the rest of the repo
stays stable-toolchain clean; only `cd fuzz && cargo +nightly ...`
pulls in the libFuzzer runtime. This keeps the nightly-only
dependency out of the default contributor build path.
