"""Hatchling build hook for mnem-cli.

Drives two things at wheel-build time:

1. The wheel's PEP 425 platform tag, read from the env var
   `MNEM_CLI_PLAT_TAG`. release.yml's binaries matrix sets this per arm
   (e.g. `win_amd64`, `manylinux_2_17_x86_64.manylinux2014_x86_64`,
   `manylinux_2_17_aarch64.manylinux2014_aarch64`, `macosx_11_0_arm64`).
   If unset, we fall back to `any`, which is appropriate for the sdist-like
   smoke-test wheel that ships no binary.

2. Whether the wheel is "pure python". A wheel carrying a native binary
   is NOT pure python; we flip `pure_python` off whenever a non-`any`
   plat tag was supplied so hatchling tags the wheel correctly and
   skips its purity warning.

The actual vendored binary + shared libs are staged by release.yml into
`mnem_cli/_vendor/` before invoking `python -m build --wheel`; the
inclusion of that directory in the wheel is configured statically in
pyproject.toml via [tool.hatch.build.targets.wheel].
"""

from __future__ import annotations

import os
from typing import Any, Dict

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: Dict[str, Any]) -> None:
        plat_tag = os.environ.get("MNEM_CLI_PLAT_TAG", "").strip()
        if plat_tag and plat_tag != "any":
            # py3-none-<plat>: still pure-python-compatible interpreter-wise,
            # but platform-specific because of the bundled native binary.
            build_data["tag"] = f"py3-none-{plat_tag}"
            build_data["pure_python"] = False
        else:
            # Smoke-test / sdist-equivalent build: no binary, runs anywhere
            # but `mnem` will exit with the cargo-install hint.
            build_data["tag"] = "py3-none-any"
            build_data["pure_python"] = True
