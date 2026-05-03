"""
bump-version.py <new_version>
------------------------------
Single entry point for version bumps.  Updates every place that records
the package version, then exits 0 so callers (CI, humans) get a clean
signal.

Files touched:
  1. Cargo.toml                             [workspace.package] version
                                            + all internal mnem-* entries
                                            in [workspace.dependencies]
  2. crates/mnem-py/pyproject.toml          version = "..."
  3. py-packages/mnem-cli/pyproject.toml    version = "..."
  4. py-packages/mnem-cli/mnem_cli/__init__.py  _VERSION = "..."
  5. npm-packages/mnem-cli/package.json     "version": "..."

Usage (human):
    python scripts/bump-version.py 0.2.0

Usage (CI, release.yml):
    python scripts/bump-version.py "${TAG_VERSION}"
    # TAG_VERSION should NOT have a leading 'v'; strip it if needed.

After running, commit all changed files and tag the release.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def validate_version(v: str) -> None:
    if not re.fullmatch(r'\d+\.\d+\.\d+(?:[-+].+)?', v):
        die(f"version must be semver (e.g. 0.2.0), got: {v!r}")


def bump_cargo_toml(new: str) -> None:
    path = REPO / 'Cargo.toml'
    text = path.read_text(encoding='utf-8')

    # [workspace.package] version = "..." - the single Cargo source of truth
    text, n1 = re.subn(
        r'(^\[workspace\.package\][^\[]*?version\s*=\s*")[^"]+(")',
        lambda m: m.group(1) + new + m.group(2),
        text,
        flags=re.MULTILINE | re.DOTALL,
        count=1,
    )
    if n1 == 0:
        die('Could not find [workspace.package] version in Cargo.toml')

    print(f"  Cargo.toml: workspace.package version -> {new}")
    path.write_text(text, encoding='utf-8')


def bump_pyproject(rel_path: str, new: str) -> None:
    path = REPO / rel_path
    text = path.read_text(encoding='utf-8')
    new_text, n = re.subn(
        r'(^version\s*=\s*")[^"]+(")',
        lambda m: m.group(1) + new + m.group(2),
        text,
        flags=re.MULTILINE,
        count=1,
    )
    if n == 0:
        die(f'Could not find version = "..." in {rel_path}')
    print(f"  {rel_path}: version -> {new}")
    path.write_text(new_text, encoding='utf-8')


def bump_init_py(new: str) -> None:
    rel = 'py-packages/mnem-cli/mnem_cli/__init__.py'
    path = REPO / rel
    text = path.read_text(encoding='utf-8')
    new_text, n = re.subn(
        r'(_VERSION\s*=\s*")[^"]+(")',
        lambda m: m.group(1) + new + m.group(2),
        text,
        count=1,
    )
    if n == 0:
        die(f'Could not find _VERSION = "..." in {rel}')
    print(f"  {rel}: _VERSION -> {new}")
    path.write_text(new_text, encoding='utf-8')


def bump_package_json(new: str) -> None:
    rel = 'npm-packages/mnem-cli/package.json'
    path = REPO / rel
    text = path.read_text(encoding='utf-8')
    new_text, n = re.subn(
        r'("version"\s*:\s*")[^"]+(")',
        lambda m: m.group(1) + new + m.group(2),
        text,
        count=1,
    )
    if n == 0:
        die(f'Could not find "version": "..." in {rel}')
    print(f"  {rel}: version -> {new}")
    path.write_text(new_text, encoding='utf-8')


def main() -> None:
    if len(sys.argv) != 2:
        die(f'Usage: python {sys.argv[0]} <new_version>')
    new = sys.argv[1].lstrip('v')
    validate_version(new)

    print(f"Bumping all version references to {new} ...")
    bump_cargo_toml(new)
    bump_pyproject('crates/mnem-py/pyproject.toml', new)
    bump_pyproject('py-packages/mnem-cli/pyproject.toml', new)
    bump_init_py(new)
    bump_package_json(new)
    print("Done.  Review the diff, commit, and tag.")


if __name__ == '__main__':
    main()
