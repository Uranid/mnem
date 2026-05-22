#!/usr/bin/env node
'use strict';

const { createHash } = require('crypto');
const { createWriteStream, existsSync, mkdirSync, copyFileSync, readdirSync, chmodSync, rmSync } = require('fs');
const { join } = require('path');
const { tmpdir, platform } = require('os');
const { spawnSync } = require('child_process');

const VERSION = process.env.npm_package_version;
const PKG_DIR = join(__dirname, '..');

const TRIPLES = {
  'darwin-arm64': { triple: 'aarch64-apple-darwin',      ext: 'tar.gz', exe: 'mnem' },
  'darwin-x64':   { triple: 'x86_64-apple-darwin',       ext: 'tar.gz', exe: 'mnem' },
  'linux-arm64':  { triple: 'aarch64-unknown-linux-gnu', ext: 'tar.gz', exe: 'mnem' },
  'linux-x64':    { triple: 'x86_64-unknown-linux-gnu',  ext: 'tar.gz', exe: 'mnem' },
  'win32-x64':    { triple: 'x86_64-pc-windows-msvc',    ext: 'zip',    exe: 'mnem.exe' },
};

const BASE_URL = `https://github.com/Uranid/mnem/releases/download/v${VERSION}`;

async function main() {
  const key = `${platform()}-${process.arch}`;
  const target = TRIPLES[key];
  if (!target) {
    warn(`Unsupported platform: ${key}. Install manually: cargo install --locked mnem-cli`);
    return;
  }

  const { triple, ext, exe } = target;
  const archiveName = `mnem-${triple}.${ext}`;
  const archiveURL  = `${BASE_URL}/${archiveName}`;
  const sha256URL   = `${archiveURL}.sha256`;

  const tmpDir = join(tmpdir(), `mnem-install-${Date.now()}`);
  mkdirSync(tmpDir, { recursive: true });
  const archivePath = join(tmpDir, archiveName);

  try {
    process.stdout.write(`mnem postinstall: downloading ${archiveName}…\n`);

    await download(archiveURL, archivePath);

    const shaResp = await fetch(sha256URL);
    if (!shaResp.ok) throw new Error(`SHA256 fetch failed: ${shaResp.status}`);
    const shaLine = await shaResp.text();
    const expected = shaLine.trim().split(/\s+/)[0];
    const actual = fileHash(archivePath);
    if (actual !== expected) throw new Error(`SHA256 mismatch: expected ${expected}, got ${actual}`);

    const extractDir = join(tmpDir, 'extracted');
    mkdirSync(extractDir, { recursive: true });
    extract(archivePath, extractDir);

    const binSrc = join(extractDir, `mnem-${triple}`, 'bin', exe);
    const binDir = join(PKG_DIR, 'bin');
    mkdirSync(binDir, { recursive: true });
    const binDst = join(binDir, exe);
    copyFileSync(binSrc, binDst);
    if (process.platform !== 'win32') chmodSync(binDst, 0o755);

    const libSrc = join(extractDir, `mnem-${triple}`, 'lib');
    const libDir = join(PKG_DIR, 'lib');
    mkdirSync(libDir, { recursive: true });
    if (existsSync(libSrc)) {
      for (const f of readdirSync(libSrc)) {
        copyFileSync(join(libSrc, f), join(libDir, f));
      }
    }

    process.stdout.write(`mnem postinstall: installed ${binDst}\n`);
  } catch (err) {
    warn(`Download failed: ${err.message}\nFallback: cargo install --locked mnem-cli`);
  } finally {
    try { rmSync(tmpDir, { recursive: true, force: true }); } catch (_) {}
  }
}

async function download(url, dest) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);
  const writer = createWriteStream(dest);
  const reader = resp.body.getReader();
  await new Promise((resolve, reject) => {
    writer.on('error', reject);
    writer.on('finish', resolve);
    (async () => {
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) { writer.end(); break; }
          writer.write(Buffer.from(value));
        }
      } catch (e) { writer.destroy(e); }
    })();
  });
}

function fileHash(path) {
  const data = require('fs').readFileSync(path);
  return createHash('sha256').update(data).digest('hex');
}

function extract(archive, dest) {
  // Windows always gets a .zip per the TRIPLES map. PATH on a user's
  // machine can resolve `tar` to two very different binaries:
  //
  //   * native bsdtar (C:\Windows\System32\tar.exe, Win10 1803+) which
  //     does auto-detect zips - but interprets a leading drive letter
  //     like `C:\...` as `host:path` and errors out with
  //     `tar: Cannot connect to C: resolve failed`; needs --force-local.
  //   * GNU tar (1.35+) from MSYS2 / MINGW / Git Bash which has the
  //     opposite problem - it doesn't treat `C:` as a host but also
  //     doesn't read zip format at all (`tar: This does not look like
  //     a tar archive`).
  //
  // Rather than detect which tar we have, always shell out to
  // PowerShell on Windows. Expand-Archive ships in every Win10+ and
  // doesn't care about PATH ordering. Linux / macOS still get .tar.gz
  // which tar handles uniformly across both GNU and BSD variants.
  if (process.platform === 'win32' && archive.toLowerCase().endsWith('.zip')) {
    const r = spawnSync('powershell', [
      '-NoProfile', '-NonInteractive', '-Command',
      `Expand-Archive -Path '${archive}' -DestinationPath '${dest}' -Force`,
    ], { stdio: 'inherit' });
    if (r.status !== 0) throw new Error(`Expand-Archive failed (status ${r.status})`);
    return;
  }
  const r = spawnSync('tar', ['-xf', archive, '-C', dest], { stdio: 'inherit' });
  if (r.status !== 0) throw new Error(`Extraction failed (status ${r.status})`);
}

function warn(msg) {
  process.stderr.write(`mnem postinstall WARNING: ${msg}\n`);
}

main().catch(err => {
  warn(`Unexpected error: ${err.message}`);
});
