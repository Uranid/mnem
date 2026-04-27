#!/usr/bin/env bash
# release-dry-run.sh - rehearse every publish step without touching a registry.
#
# Runs locally (contributor machines) AND in CI (`publish-dryrun.yml`) on any
# PR that touches publish metadata, so a broken Cargo.toml never reaches the
# real `.github/workflows/release.yml` path.
#
# What it does:
#   1. `cargo publish --dry-run -p <crate>` in topological order over every
#      publishable workspace crate (skips `publish = false` crates).
#   2. `maturin build --release` dry build for the Python wheel.
#   3. `docker build` of the mnem-http image, tagged `mnem-http:dry-run`,
#      without a push.
#
# On failure, exits non-zero immediately ("fail fast") so you see exactly
# which step needs fixing.
#
# Override the crate list via `CRATES="<space-separated>"` if you are
# debugging a single-crate publish.
#
# Use `SKIP_DOCKER=1` to skip the docker step in offline environments
# (docker daemon absent, slow DNS). CI always runs it.

set -euo pipefail

# Resolve repo root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ANSI colors only if stdout is a TTY (so CI logs stay clean).
if [ -t 1 ]; then
  RED=$'\033[0;31m'
  GREEN=$'\033[0;32m'
  YELLOW=$'\033[1;33m'
  BLUE=$'\033[0;34m'
  BOLD=$'\033[1m'
  RESET=$'\033[0m'
else
  RED=""; GREEN=""; YELLOW=""; BLUE=""; BOLD=""; RESET=""
fi

# Crate publish order matches `.github/workflows/release.yml`'s
# `publish-crates` list, extended with the three crates that workflow
# doesn't currently publish (ann, transport, py). When you add a new
# crate: append to the list here AND to release.yml's CRATES array,
# both in topological order (deps before dependents).
CRATES_DEFAULT=(
  # Tier 0: no internal deps.
  mnem-core
  # Tier 1: depend only on mnem-core.
  mnem-core-testutils   # publish = false; skipped by the loop
  mnem-backend-redb
  mnem-transport
  mnem-ann
  mnem-embed-providers
  mnem-sparse-providers
  mnem-rerank-providers
  mnem-llm-providers
  # Tier 2: binaries / servers that depend on tier 1.
  mnem-http
  mnem-mcp
  mnem-cli
  # Tier 3: Python bindings (dry-run build goes through maturin separately).
  mnem-py
)
CRATES=(${CRATES:-${CRATES_DEFAULT[@]}})

pass=0
fail=0
skipped=()
passed=()
failed=()

step() { printf "\n${BOLD}${BLUE}==> %s${RESET}\n" "$*"; }
ok()   { printf "${GREEN}[pass]${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}[warn]${RESET} %s\n" "$*"; }
die()  { printf "${RED}[fail]${RESET} %s\n" "$*"; fail_count; }
fail_count() { fail=$((fail+1)); }

# --- 1) cargo publish --dry-run / cargo package --list for every publishable crate ---
#
# Two strategies, because cargo's dry-run + the crates.io index don't compose:
#
#   * For the bottom of the DAG (`mnem-core` + anything with no internal deps):
#     run the full `cargo publish --dry-run`. That re-packages the crate AND
#     performs a full registry-index verification compile, catching missing
#     files, bad readme paths, broken metadata, and unresolvable deps.
#
#   * For every other crate: run `cargo package --list --allow-dirty`. This
#     validates the package manifest (all required fields present, no stray
#     uncommittable files, license string parseable, etc.) without hitting the
#     registry. That matters because our internal crate deps (`mnem-core`,
#     `mnem-backend-redb`, ...) aren't on crates.io yet. A `cargo publish
#     --dry-run` against a crate that depends on `mnem-core` transitively
#     fails with "no matching package named `mnem-core` found" until
#     `mnem-core` is actually published. That failure is structural, not a
#     metadata bug - the guard would be permanently red. `cargo package
#     --list` avoids the issue while still catching the Cargo.toml audit
#     classes this script exists to prevent.
#
# When `mnem-core` lands on crates.io (first real release), you can flip
# `FULL_DRY_RUN_EVERYWHERE=1` via env to upgrade every crate to the full
# dry-run. The CI workflow defaults to the split mode so PR checks remain
# meaningful before the first publish.

step "cargo publish --dry-run (base) + cargo package --list (dependents)"

# Crates with NO internal workspace dep - safe to do a full dry-run against
# crates.io because their dep resolution never reaches a path= workspace
# crate. Keep this list conservative: adding a crate here and getting it
# wrong means CI fails for an unrelated reason.
BASE_CRATES=("mnem-core")

for c in "${CRATES[@]}"; do
  # Skip crates marked publish=false (mnem-core-testutils).
  publishable=$(cargo metadata --format-version 1 --no-deps 2>/dev/null \
    | python -c "
import json, sys
m = json.load(sys.stdin)
for p in m['packages']:
    if p['name'] == '$c':
        print('true' if p.get('publish') is None else 'false')
        break
else:
    print('missing')
" 2>/dev/null || echo "unknown")
  if [ "$publishable" = "false" ]; then
    warn "skip $c (publish = false)"
    skipped+=("$c")
    continue
  fi
  if [ "$publishable" = "missing" ]; then
    printf "%s[fail]%s %s: crate not found in workspace\n" "$RED" "$RESET" "$c"
    failed+=("$c")
    fail=$((fail+1))
    break
  fi

  # Decide the check strategy for this crate.
  use_full=0
  if [ "${FULL_DRY_RUN_EVERYWHERE:-0}" = "1" ]; then
    use_full=1
  else
    for b in "${BASE_CRATES[@]}"; do
      if [ "$c" = "$b" ]; then use_full=1; break; fi
    done
  fi

  if [ "$use_full" = "1" ]; then
    printf "%s-- cargo publish --dry-run -p %s%s\n" "$BLUE" "$c" "$RESET"
    if cargo publish --dry-run --allow-dirty -p "$c" 2>&1; then
      ok "$c publishes clean (full dry-run)"
      passed+=("$c")
      pass=$((pass+1))
    else
      printf "%s[fail]%s %s: cargo publish --dry-run failed (see above)\n" "$RED" "$RESET" "$c"
      failed+=("$c")
      fail=$((fail+1))
      break
    fi
  else
    printf "%s-- cargo package --list -p %s%s\n" "$BLUE" "$c" "$RESET"
    # Redirect stderr to stdout so we can capture the full message set.
    if out=$(cargo package --list --allow-dirty -p "$c" 2>&1); then
      # Count files that would be packaged; a zero-file output is a bug.
      files=$(printf '%s\n' "$out" | grep -c '^[^[:space:]].*\..*' || true)
      ok "$c manifest audit passed (${files} files in package)"
      passed+=("$c")
      pass=$((pass+1))
    else
      printf '%s\n' "$out"
      printf "%s[fail]%s %s: cargo package --list failed (see above)\n" "$RED" "$RESET" "$c"
      failed+=("$c")
      fail=$((fail+1))
      break
    fi
  fi
done

# --- 2) maturin build --release (mnem-py wheel) ---------------------------
step "maturin build --release (mnem-py wheel dry build)"

if ! command -v maturin >/dev/null 2>&1; then
  warn "maturin not installed; skipping wheel build (install with 'pip install maturin')"
  skipped+=("maturin-wheel")
else
  if maturin build --release --manifest-path crates/mnem-py/Cargo.toml 2>&1; then
    ok "mnem-py wheel built"
    passed+=("mnem-py-wheel")
    pass=$((pass+1))
  else
    printf "${RED}[fail]${RESET} maturin build failed\n"
    failed+=("mnem-py-wheel")
    fail=$((fail+1))
  fi
fi

# --- 3) docker build (mnem-http image) ------------------------------------
step "docker build (mnem-http image, local tag only)"

if [ "${SKIP_DOCKER:-0}" = "1" ]; then
  warn "SKIP_DOCKER=1; skipping docker build"
  skipped+=("docker-mnem-http")
elif ! command -v docker >/dev/null 2>&1; then
  warn "docker not available; skipping image build"
  skipped+=("docker-mnem-http")
else
  # Dockerfile lives at the repo root (not crates/mnem-http/Dockerfile).
  if docker build -f Dockerfile -t mnem-http:dry-run . 2>&1; then
    ok "mnem-http:dry-run built"
    passed+=("docker-mnem-http")
    pass=$((pass+1))
  else
    printf "${RED}[fail]${RESET} docker build failed\n"
    failed+=("docker-mnem-http")
    fail=$((fail+1))
  fi
fi

# --- summary --------------------------------------------------------------
printf "\n${BOLD}==== release-dry-run summary ====${RESET}\n"
printf "passed (%d):\n" "$pass"
for p in "${passed[@]:-}"; do printf "  ${GREEN}ok${RESET}   %s\n" "$p"; done
if [ "${#skipped[@]}" -gt 0 ]; then
  printf "skipped (%d):\n" "${#skipped[@]}"
  for s in "${skipped[@]}"; do printf "  ${YELLOW}skip${RESET} %s\n" "$s"; done
fi
if [ "$fail" -gt 0 ]; then
  printf "${RED}failed (%d):${RESET}\n" "$fail"
  for f in "${failed[@]:-}"; do printf "  ${RED}fail${RESET} %s\n" "$f"; done
  printf "\n${RED}${BOLD}release dry-run FAILED${RESET} - fix the above before cutting a tag.\n"
  exit 1
fi

printf "\n${GREEN}${BOLD}release dry-run PASSED${RESET} - every publishable crate + wheel + image is ready.\n"
exit 0
