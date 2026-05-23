#!/usr/bin/env node
'use strict';

// mnem umbrella launcher.
//
// At install time, npm picks exactly one of the four `mnem-cli-<plat>-<arch>`
// optionalDependencies based on the host's `os` / `cpu` filters and skips the
// rest. This script resolves that sub-package at runtime, locates the native
// binary it ships, and forwards argv / stdio to it.
//
// Why `require.resolve(<sub>/package.json)` instead of a hard-coded
// `node_modules/...` path: when mnem-cli is hoisted, depended on from a
// monorepo, or installed under pnpm, the sub-package may not live directly
// under our own node_modules. Letting Node's resolver find it handles all of
// those layouts the same way.

const { spawnSync } = require('child_process');
const path = require('path');
const os = require('os');

const PLATFORM = process.platform;
const ARCH = process.arch;
const SUBPKG = `mnem-cli-${PLATFORM}-${ARCH}`;

let subPkgDir;
try {
  // Resolve the sub-package's package.json, then strip the filename to get
  // its root directory. `require.resolve` follows the same lookup rules as a
  // normal `require`, so it works under hoisting and pnpm.
  subPkgDir = path.dirname(require.resolve(`${SUBPKG}/package.json`));
} catch (err) {
  if (err && err.code === 'MODULE_NOT_FOUND') {
    process.stderr.write(
      `mnem: no prebuilt binary for ${PLATFORM}-${ARCH} (looked for "${SUBPKG}").\n` +
      'Either this platform is unsupported, or `npm install` skipped the\n' +
      'platform-specific package. To install from source:\n' +
      '  cargo install --locked mnem-cli\n' +
      'Or grab a prebuilt archive from:\n' +
      '  https://github.com/Uranid/mnem/releases\n'
    );
    process.exit(1);
  }
  throw err;
}

const exeName = PLATFORM === 'win32' ? 'mnem.exe' : 'mnem';
const native = path.join(subPkgDir, 'bin', exeName);
const lib = path.join(subPkgDir, 'lib');

// Point the dynamic linker at the sub-package's `lib/` so the binary can find
// its sibling shared libraries (e.g. onnxruntime). Windows uses the directory
// the .exe lives in by default, so no equivalent env var is needed.
//
// macOS uses DYLD_FALLBACK_LIBRARY_PATH (not DYLD_LIBRARY_PATH): SIP strips
// DYLD_* from processes spawned from SIP-protected binaries, including the
// Apple-shipped node at /usr/bin/node. The FALLBACK variant survives the SIP
// scrub and is only consulted when the binary couldn't resolve the lib via
// its own @rpath / install-name, so the load order remains correct.
const env = { ...process.env };
if (PLATFORM === 'linux') {
  env.LD_LIBRARY_PATH = [lib, env.LD_LIBRARY_PATH].filter(Boolean).join(':');
}
if (PLATFORM === 'darwin') {
  env.DYLD_FALLBACK_LIBRARY_PATH = [lib, env.DYLD_FALLBACK_LIBRARY_PATH].filter(Boolean).join(':');
}

const r = spawnSync(native, process.argv.slice(2), { env, stdio: 'inherit' });
process.exit(r.status ?? 1);
