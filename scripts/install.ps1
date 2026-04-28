# mnem one-line installer for Windows (PowerShell 5+).
#
# Usage (from the repo raw URL; host behind whatever CDN you run):
#   iwr -useb https://raw.githubusercontent.com/Uranid/mnem/main/scripts/install.ps1 | iex
#
# Env vars:
#   $env:MNEM_INSTALL_DIR   target dir (default: $env:USERPROFILE\.mnem\bin)
#   $env:MNEM_VERSION       tag to install (default: latest)
#   $env:MNEM_NO_MODIFY_PATH=1  skip user-PATH update

$ErrorActionPreference = 'Stop'

$Repo       = 'Uranid/mnem'
$InstallDir = if ($env:MNEM_INSTALL_DIR) { $env:MNEM_INSTALL_DIR } else { "$env:USERPROFILE\.mnem\bin" }
$Version    = if ($env:MNEM_VERSION)     { $env:MNEM_VERSION }     else { 'latest' }

function Say([string]$msg) { Write-Host "mnem-install: $msg" }

function Detect-Triple {
    if ([System.Environment]::Is64BitOperatingSystem) {
        # Windows ARM64 support is not in the first release cut; add
        # aarch64-pc-windows-msvc here when we ship it.
        return 'x86_64-pc-windows-msvc'
    } else {
        throw "32-bit Windows is not supported. Build from source: https://github.com/$Repo"
    }
}

$Triple  = Detect-Triple
$Archive = "mnem-$Triple.zip"
$Url = if ($Version -eq 'latest') {
    "https://github.com/$Repo/releases/latest/download/$Archive"
} else {
    "https://github.com/$Repo/releases/download/$Version/$Archive"
}

Say "triple=$Triple version=$Version"
Say "install_dir=$InstallDir"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

$Tmp = Join-Path $env:TEMP "mnem-install-$([guid]::NewGuid().ToString('N'))"
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
$Zip = Join-Path $Tmp 'mnem.zip'

Say "downloading $Url"
try {
    Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing
} catch {
    Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
    throw "download failed. Check that a release for $Triple exists at $Url."
}

Say "extracting..."
Expand-Archive -Path $Zip -DestinationPath $Tmp -Force

foreach ($bin in @('mnem.exe', 'mnem-mcp.exe')) {
    $src = Join-Path $Tmp $bin
    if (-not (Test-Path $src)) {
        Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
        throw "archive does not contain $bin; cannot continue"
    }
    Copy-Item -Force $src (Join-Path $InstallDir $bin)
}
Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue

if (-not $env:MNEM_NO_MODIFY_PATH) {
    # Refuse to patch user PATH with a path that contains separators or
    # control characters (most likely an attacker-controlled env var).
    if ($InstallDir -match '[;\r\n\0]' -or $InstallDir -like '\\?\*') {
        throw "MNEM_INSTALL_DIR contains illegal characters; refusing to touch user PATH"
    }
    $userPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
    if (-not $userPath) { $userPath = '' }
    if ($userPath -notlike "*$InstallDir*") {
        $sep = if ($userPath.Length -gt 0) { ';' } else { '' }
        [Environment]::SetEnvironmentVariable('PATH', "$userPath$sep$InstallDir", 'User')
        Say "added $InstallDir to user PATH"
    } else {
        Say "user PATH already includes $InstallDir"
    }
}

Say 'done.'
Write-Host
Write-Host 'Next:'
Write-Host '  1. Open a fresh PowerShell window.'
Write-Host '  2. mnem --version'
Write-Host '  3. mnem integrate      (wire Claude Desktop / Cursor / Continue / Zed)'
Write-Host '  4. mnem doctor         (health check)'
