"""Binary launcher for mnem-cli.

This module is the console-script entry point installed as `mnem`. The actual
mnem binary plus any required shared libraries are vendored inside this
package under `_vendor/` by the per-platform wheel build (see
release.yml + hatch_build.py). At runtime we locate the vendored binary
via importlib.resources, set the right dynamic-loader path so the bundled
onnxruntime libs are found, and exec it with the caller's argv.

If `_vendor/bin/mnem(.exe)` is missing (the sdist case), we exit with a
clean message pointing the user at cargo or the GitHub releases page.
"""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from typing import Optional


def _exe_name() -> str:
    return "mnem.exe" if sys.platform == "win32" else "mnem"


def _vendor_root() -> Path:
    # importlib.resources.files returns a Traversable; for a regular
    # filesystem-backed package this is a Path. We treat it as Path
    # since wheels are unpacked on install.
    return Path(str(files("mnem_cli"))) / "_vendor"


def _find_binary() -> Optional[Path]:
    candidate = _vendor_root() / "bin" / _exe_name()
    if candidate.is_file():
        return candidate
    return None


def _augment_library_path(env: dict[str, str], lib_dir: Path) -> None:
    if not lib_dir.is_dir():
        return
    lib_str = str(lib_dir)
    if sys.platform.startswith("linux"):
        var = "LD_LIBRARY_PATH"
    elif sys.platform == "darwin":
        # macOS System Integrity Protection strips `DYLD_LIBRARY_PATH`
        # from processes spawned from SIP-protected binaries (most of
        # /usr/bin, including the Apple-shipped python3). Use the
        # `DYLD_FALLBACK_*` variant: it's checked only when the binary
        # didn't already resolve the library via @rpath / install-name,
        # and SIP leaves it intact across exec boundaries.
        var = "DYLD_FALLBACK_LIBRARY_PATH"
    else:
        # Windows resolves DLLs from the binary's own directory; no env tweak needed.
        return
    existing = env.get(var, "")
    env[var] = os.pathsep.join(p for p in (lib_str, existing) if p)


def main() -> None:
    binary = _find_binary()
    if binary is None:
        sys.stderr.write(
            "mnem: no prebuilt mnem-cli wheel for this platform.\n"
            "Install via: cargo install --locked mnem-cli\n"
            "Or download from: https://github.com/Uranid/mnem/releases\n"
        )
        sys.exit(1)

    env = os.environ.copy()
    _augment_library_path(env, _vendor_root() / "lib")

    result = subprocess.run([str(binary), *sys.argv[1:]], env=env)
    sys.exit(result.returncode)
