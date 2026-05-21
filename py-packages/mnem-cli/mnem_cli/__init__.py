"""Binary wrapper for mnem-cli: downloads the prebuilt binary on first use."""

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

try:
    from importlib.metadata import version as _pkg_version
    _VERSION = _pkg_version("mnem-cli")
except Exception:
    _VERSION = "0.1.7"

# Maps (system, machine) to (triple, archive-ext, exe-name).
# darwin/x86_64 falls back to the aarch64 binary which runs via Rosetta 2.
_TRIPLES = {
    ("darwin",  "arm64"):  ("aarch64-apple-darwin",      "tar.gz", "mnem"),
    ("darwin",  "x86_64"): ("aarch64-apple-darwin",      "tar.gz", "mnem"),
    ("linux",   "aarch64"): ("aarch64-unknown-linux-gnu", "tar.gz", "mnem"),
    ("linux",   "x86_64"): ("x86_64-unknown-linux-gnu",  "tar.gz", "mnem"),
    ("windows", "amd64"):  ("x86_64-pc-windows-msvc",    "zip",    "mnem.exe"),
    ("windows", "x86_64"): ("x86_64-pc-windows-msvc",    "zip",    "mnem.exe"),
}

_BASE_URL = "https://github.com/Uranid/mnem/releases/download/v{version}"
_BIN_DIR = Path.home() / ".mnem_cli" / "bin"


def _target():
    sys_name = platform.system().lower()
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        machine = "arm64"
    elif machine in ("amd64", "x86_64"):
        machine = "x86_64"
    return _TRIPLES.get((sys_name, machine))


def _ensure_binary():
    # type: () -> Optional[Path]
    target = _target()
    if target is None:
        return None
    triple, ext, exe = target
    bin_path = _BIN_DIR / exe
    if bin_path.exists():
        return bin_path

    base = _BASE_URL.format(version=_VERSION)
    archive_name = f"mnem-{triple}.{ext}"
    archive_url = f"{base}/{archive_name}"
    sha_url = f"{archive_url}.sha256"

    sys.stderr.write(f"mnem: downloading {archive_name}...\n")

    with tempfile.TemporaryDirectory() as tmp:
        archive_path = os.path.join(tmp, archive_name)
        urllib.request.urlretrieve(archive_url, archive_path)

        with urllib.request.urlopen(sha_url) as resp:
            sha_line = resp.read().decode().strip()
        expected = sha_line.split()[0]

        with open(archive_path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch: expected {expected}, got {actual}")

        extract_dir = os.path.join(tmp, "extracted")
        os.makedirs(extract_dir)
        if ext == "tar.gz":
            with tarfile.open(archive_path) as tf:
                extract_kwargs = {}
                if sys.version_info >= (3, 12):
                    extract_kwargs["filter"] = "data"
                tf.extractall(extract_dir, **extract_kwargs)
        else:
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(extract_dir)

        _BIN_DIR.mkdir(parents=True, exist_ok=True)
        src = os.path.join(extract_dir, f"mnem-{triple}", "bin", exe)
        shutil.copy2(src, bin_path)
        if sys.platform != "win32":
            os.chmod(bin_path, 0o755)

        # Copy the bundled onnxruntime so `mnem http serve` works out of the box.
        lib_src = os.path.join(extract_dir, f"mnem-{triple}", "lib")
        lib_dir = _BIN_DIR.parent / "lib"
        lib_dir.mkdir(parents=True, exist_ok=True)
        if os.path.isdir(lib_src):
            for name in os.listdir(lib_src):
                shutil.copy2(os.path.join(lib_src, name), lib_dir / name)

    sys.stderr.write(f"mnem: installed to {bin_path}\n")
    return bin_path


def main():
    # type: () -> None
    bin_path = _ensure_binary()
    if bin_path is None:
        sys.stderr.write(
            "mnem: unsupported platform.\n"
            "Install via: cargo install --locked mnem-cli\n"
            "Or download from: https://github.com/Uranid/mnem/releases\n"
        )
        sys.exit(1)

    env = os.environ.copy()
    lib_dir = str(_BIN_DIR.parent / "lib")
    if sys.platform.startswith("linux"):
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(x for x in [lib_dir, existing] if x)
    elif sys.platform == "darwin":
        existing = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = ":".join(x for x in [lib_dir, existing] if x)

    result = subprocess.run([str(bin_path)] + sys.argv[1:], env=env)
    sys.exit(result.returncode)
