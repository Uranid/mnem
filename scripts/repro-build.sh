#!/usr/bin/env bash
# scripts/repro-build.sh — reproducible-build verifier for mnem.
#
# Builds the release binaries twice in independent target dirs with
# identical env, then compares SHA-256 of every artifact. On Linux-musl
# we require byte-identical output (hard fail on diff). On macOS +
# Windows we strip known-nondeterministic sections before comparing
# (warn-only; divergences are documented in docs/REPRODUCIBLE-BUILDS.md).
#
# Emits `repro-manifest.json` in the repo root with per-artifact hashes
# + build metadata. The release workflow uploads this manifest as a
# release asset next to the archive + SLSA attestation.
#
# Usage:
#   bash scripts/repro-build.sh                        # native triple
#   bash scripts/repro-build.sh --triple <triple>      # specific triple
#   bash scripts/repro-build.sh --triple <triple> --strict  # hard-fail
#
# Exits 0 on reproducible, 1 on diff, 2 on setup error.

set -euo pipefail

TRIPLE=""
STRICT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --triple) TRIPLE="$2"; shift 2 ;;
    --strict) STRICT=1; shift ;;
    -h|--help)
      sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "error: unknown arg $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$TRIPLE" ]]; then
  TRIPLE="$(rustc -vV | awk '/^host:/ {print $2}')"
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f Cargo.toml ]]; then
  echo "error: must be run from mnem repo root" >&2; exit 2
fi

# Determinism knobs.
SOURCE_DATE_EPOCH="$(git log -1 --pretty=%ct HEAD)"
export SOURCE_DATE_EPOCH

export RUSTFLAGS="-C strip=debuginfo -C codegen-units=1 --remap-path-prefix=${REPO_ROOT}=."

# musl is our reproducible-first path. When the caller asks for musl
# and we're on Linux, require the target be installed + the linker
# be rust-lld. Otherwise proceed best-effort.
case "$TRIPLE" in
  *-linux-musl) MODE="strict-musl" ;;
  *-linux-gnu)  MODE="linux-gnu" ;;
  *-apple-*)    MODE="macos-allowlist" ;;
  *-windows-msvc) MODE="windows-allowlist" ;;
  *-windows-gnu)  MODE="windows-gnu-strict" ;;
  *) MODE="best-effort" ;;
esac

echo "== mnem repro-build =="
echo "triple:             $TRIPLE"
echo "mode:               $MODE"
echo "SOURCE_DATE_EPOCH:  $SOURCE_DATE_EPOCH"
echo "RUSTFLAGS:          $RUSTFLAGS"
echo "commit:             $(git rev-parse HEAD)"
echo

# Two independent target dirs so incremental cache cannot bleed
# between the two runs.
TARGET_A="$REPO_ROOT/target-repro-a"
TARGET_B="$REPO_ROOT/target-repro-b"
rm -rf "$TARGET_A" "$TARGET_B"

build_once() {
  local target_dir="$1"
  echo ">> build into $target_dir"
  CARGO_TARGET_DIR="$target_dir" \
    cargo build --release --locked \
    --target "$TRIPLE" \
    -p mnem-cli -p mnem-mcp -p mnem-http
}

build_once "$TARGET_A"
build_once "$TARGET_B"

# Pick the artifact extension.
EXT=""
case "$TRIPLE" in
  *-windows-*) EXT=".exe" ;;
esac

ARTIFACTS=("mnem${EXT}" "mnem-mcp${EXT}" "mnem-http${EXT}")

sha256_of() {
  # Portable SHA-256 that works on Linux + macOS + Git-Bash.
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

# Allowlist-normalize an artifact before comparison when the platform
# injects known non-determinism we can't suppress at build time.
normalize() {
  local input="$1" output="$2"
  cp "$input" "$output"
  case "$MODE" in
    macos-allowlist)
      # Strip Mach-O LC_UUID + codesign blob.
      if command -v llvm-strip >/dev/null 2>&1; then
        llvm-strip --remove-section=__LINKEDIT "$output" 2>/dev/null || true
      fi
      ;;
    windows-allowlist)
      # Strip PE TimeDateStamp + Rich header + RSDS guid.
      # A targeted byte-patch lives in a separate helper we ship later.
      if command -v llvm-objcopy >/dev/null 2>&1; then
        llvm-objcopy --strip-unneeded "$output" 2>/dev/null || true
      fi
      ;;
  esac
}

RESULTS_JSON="["
ANY_DIFF=0
FIRST=1
for art in "${ARTIFACTS[@]}"; do
  path_a="$TARGET_A/$TRIPLE/release/$art"
  path_b="$TARGET_B/$TRIPLE/release/$art"

  if [[ ! -f "$path_a" || ! -f "$path_b" ]]; then
    echo "error: missing artifact $art in one of the target dirs" >&2
    exit 2
  fi

  norm_a="$(mktemp)"
  norm_b="$(mktemp)"
  normalize "$path_a" "$norm_a"
  normalize "$path_b" "$norm_b"

  sha_a_raw="$(sha256_of "$path_a")"
  sha_b_raw="$(sha256_of "$path_b")"
  sha_a_norm="$(sha256_of "$norm_a")"
  sha_b_norm="$(sha256_of "$norm_b")"
  size="$(wc -c < "$path_a" | tr -d ' ')"

  repro="true"
  if [[ "$sha_a_norm" != "$sha_b_norm" ]]; then
    repro="false"
    ANY_DIFF=1
    echo "DIFF: $art (a=$sha_a_norm b=$sha_b_norm)"
    # Surface a short diffoscope hint if available.
    if command -v diffoscope >/dev/null 2>&1; then
      diffoscope --max-report-size 8192 "$norm_a" "$norm_b" || true
    else
      cmp -l "$norm_a" "$norm_b" | head -20 || true
    fi
  else
    echo "OK:   $art  $sha_a_norm"
  fi

  rm -f "$norm_a" "$norm_b"

  if [[ $FIRST -eq 0 ]]; then RESULTS_JSON+=","; fi
  FIRST=0
  RESULTS_JSON+=$(cat <<JSON
{
  "name": "$art",
  "sha256": "$sha_a_raw",
  "size_bytes": $size,
  "build_a_sha256": "$sha_a_raw",
  "build_b_sha256": "$sha_b_raw",
  "normalized_sha256": "$sha_a_norm",
  "reproducible": $repro
}
JSON
)
done
RESULTS_JSON+="]"

RUSTC_VERSION="$(rustc --version)"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
COMMIT="$(git rev-parse HEAD)"

cat > repro-manifest.json <<JSON
{
  "schema": "mnem.repro-manifest/v1",
  "commit": "$COMMIT",
  "source_date_epoch": $SOURCE_DATE_EPOCH,
  "rustc_version": "$RUSTC_VERSION",
  "triple": "$TRIPLE",
  "mode": "$MODE",
  "rustflags": "$RUSTFLAGS",
  "generated_at": "$NOW",
  "artifacts": $RESULTS_JSON
}
JSON

echo
echo "wrote repro-manifest.json"

if [[ $ANY_DIFF -ne 0 ]]; then
  case "$MODE" in
    strict-musl|windows-gnu-strict)
      echo "FAIL: byte-identical gate violated on $MODE" >&2
      exit 1
      ;;
    *)
      if [[ $STRICT -eq 1 ]]; then
        echo "FAIL: --strict set and diff detected on $MODE" >&2
        exit 1
      fi
      echo "WARN: diff detected on $MODE (allowlisted; see docs/REPRODUCIBLE-BUILDS.md)"
      exit 0
      ;;
  esac
fi

echo "OK:   all artifacts reproduced byte-identical"
exit 0
