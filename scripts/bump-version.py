"""
bump-version.py <new_version>
------------------------------
Updates every file that records the package version.

Files touched:
  1. Cargo.toml                              [workspace.package] version
  2. Cargo.lock                              workspace member versions
  3. crates/mnem-py/pyproject.toml           version = "..."
  4. py-packages/mnem-cli/pyproject.toml     version = "..."
  5. py-packages/mnem-cli/mnem_cli/__init__.py  _VERSION = "..."
  6. npm-packages/mnem-cli/package.json      "version": "..."

Usage (human, before committing):
    python scripts/bump-version.py 0.2.0

Usage (CI release.yml - derives version from the pushed tag):
    python scripts/bump-version.py "${{ needs.release.outputs.tag }}"
    # Leading 'v' is stripped automatically (v0.2.0 -> 0.2.0)

The tag is the single source of truth.  CI calls this script in every
job that needs an accurate version before building or publishing.
Cargo.lock is patched alongside Cargo.toml so --locked builds remain
valid without a separate cargo update step.
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
    text, n = re.subn(
        r'(^\[workspace\.package\][^\[]*?version\s*=\s*")[^"]+(")',
        lambda m: m.group(1) + new + m.group(2),
        text,
        flags=re.MULTILINE | re.DOTALL,
        count=1,
    )
    if n == 0:
        die('Could not find [workspace.package] version in Cargo.toml')
    print(f"  Cargo.toml: workspace.package version -> {new}")
    path.write_text(text, encoding='utf-8')


def bump_cargo_lock(new: str) -> None:
    path = REPO / 'Cargo.lock'
    if not path.exists():
        return
    text = path.read_text(encoding='utf-8')
    # Cargo.lock structure: each [[package]] block has name then version on
    # consecutive lines.  Match only mnem-* workspace members.
    new_text, n = re.subn(
        r'(name = "mnem-[^"]+"\r?\nversion = ")[^"]+(")',
        lambda m: m.group(1) + new + m.group(2),
        text,
    )
    if n:
        print(f"  Cargo.lock: {n} workspace member version(s) -> {new}")
        path.write_text(new_text, encoding='utf-8')


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
    bump_cargo_lock(new)
    bump_pyproject('crates/mnem-py/pyproject.toml', new)
    bump_pyproject('py-packages/mnem-cli/pyproject.toml', new)
    bump_init_py(new)
    bump_package_json(new)
    print("Done.")


if __name__ == '__main__':
    main()
