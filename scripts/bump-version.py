"""
bump-version.py <new_version>
------------------------------
Updates every file that records the package version.

Files touched:
  1. Cargo.toml                              [workspace.package] version
                                             + [workspace.dependencies] internal-dep
                                               `version = "..."` pins (gotcha 9.2
                                               from the v0.1.6 release playbook)
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

    # Cargo 1.95+ rejects `cargo publish` when an internal path-only dep is
    # missing a `version = "..."`. The `[workspace.dependencies]` block in
    # Cargo.toml pins each `mnem-*` internal dep at the current release; we
    # roll them all forward in one sweep so the publish loop doesn't have
    # to.  Pre-fix, this was the manual edit recorded as gotcha 9.2 in the
    # release playbook.  The regex anchors on the `path = "crates/mnem-*"`
    # left-hand side so we only touch internal deps (third-party deps with
    # the same `version =` shape aren't matched).
    new_text, dep_n = re.subn(
        r'(path\s*=\s*"crates/mnem-[^"]+"\s*,\s*version\s*=\s*")[^"]+(")',
        lambda m: m.group(1) + new + m.group(2),
        text,
    )
    if dep_n:
        print(
            f"  Cargo.toml: {dep_n} workspace.dependencies internal-dep version pin(s) -> {new}"
        )
        text = new_text

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
    # The pre-v0.1.8 wrapper held an `_VERSION = "..."` constant used to
    # build the GitHub-release download URL at runtime. After the
    # cross-platform refactor, the binary ships inside the wheel and no
    # URL needs constructing, so the constant is gone. Keep this
    # function as a no-op-if-absent helper rather than removing it
    # outright; that way an older wrapper checked out at an old commit
    # still bumps cleanly when this script runs against it.
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
        return  # no _VERSION present in the new wrapper, nothing to bump
    print(f"  {rel}: _VERSION -> {new}")
    path.write_text(new_text, encoding='utf-8')


def bump_package_json(new: str) -> None:
    """Bump npm package versions.

    Touches both the umbrella `mnem-cli/package.json` and every
    per-platform sub-package directory `mnem-cli-<plat>-<arch>/`.
    The umbrella additionally carries an `optionalDependencies` block
    that pins each sub-package to an EXACT version; those pins must
    track the workspace version too, or a partial publish would leave
    the umbrella resolving to a stale sub-package. The exact-version
    regex anchors on the `mnem-cli-` prefix to avoid touching any
    unrelated third-party dep that happens to have the same shape.
    """
    npm_dir = REPO / 'npm-packages'
    pkg_dirs = sorted(d for d in npm_dir.iterdir() if d.is_dir())
    if not pkg_dirs:
        die(f'No npm packages found under {npm_dir}')
    for pkg_dir in pkg_dirs:
        path = pkg_dir / 'package.json'
        if not path.exists():
            continue
        text = path.read_text(encoding='utf-8')

        # 1. Top-level `"version": "..."`. count=1 anchors on the first
        #    occurrence, which is the package's own version (a future
        #    "version" key under another object would not be hit).
        new_text, n_top = re.subn(
            r'("version"\s*:\s*")[^"]+(")',
            lambda m: m.group(1) + new + m.group(2),
            text,
            count=1,
        )
        if n_top == 0:
            die(f'Could not find "version": "..." in {path.relative_to(REPO)}')

        # 2. optionalDependencies / dependencies pins on the sibling
        #    sub-packages: `"mnem-cli-linux-x64": "0.1.7"` -> `... "0.1.8"`.
        #    The `mnem-cli-` name anchor keeps this from touching any
        #    unrelated dep that happens to have a matching string shape.
        new_text, n_dep = re.subn(
            r'("mnem-cli-[a-z0-9-]+"\s*:\s*")[^"]+(")',
            lambda m: m.group(1) + new + m.group(2),
            new_text,
        )

        path.write_text(new_text, encoding='utf-8')
        rel = path.relative_to(REPO).as_posix()
        if n_dep:
            print(
                f"  {rel}: version -> {new} "
                f"({n_dep} sub-package pin(s) also bumped)"
            )
        else:
            print(f"  {rel}: version -> {new}")


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
