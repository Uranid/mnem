"""
migrate-internal-deps.py
------------------------
One-shot migration script: rewrite all crate Cargo.toml files so that
internal mnem-* path deps use { workspace = true } instead of hardcoded
path+version strings.

Run once from the repo root:
    python scripts/migrate-internal-deps.py
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def rewrite_dep_value(m: re.Match) -> str:
    inner = m.group(1)
    # Strip path = "..." and version = "..." fields
    inner = re.sub(r'path\s*=\s*"[^"]*"\s*,?\s*', '', inner)
    inner = re.sub(r'version\s*=\s*"[^"]*"\s*,?\s*', '', inner)
    inner = inner.strip().strip(',').strip()
    if inner:
        return '{ workspace = true, ' + inner + ' }'
    return '{ workspace = true }'


def transform_line(line: str) -> str:
    if not re.search(r'mnem-[\w-]+\s*=\s*\{[^}]*path\s*=\s*"\.\.',  line):
        return line
    return re.sub(r'\{([^{}]*)\}', rewrite_dep_value, line)


def migrate_file(path: Path) -> bool:
    original = path.read_text(encoding='utf-8')
    lines = original.splitlines(keepends=True)
    new_lines = [transform_line(l) for l in lines]
    new_text = ''.join(new_lines)
    if new_text == original:
        return False
    path.write_text(new_text, encoding='utf-8')
    return True


def main() -> None:
    targets = sorted(REPO_ROOT.glob('crates/*/Cargo.toml'))
    changed = []
    for t in targets:
        if migrate_file(t):
            changed.append(t.relative_to(REPO_ROOT))
    if changed:
        print(f"Migrated {len(changed)} file(s):")
        for p in changed:
            print(f"  {p}")
    else:
        print("Nothing to migrate - all files already up to date.")


if __name__ == '__main__':
    main()
