#!/usr/bin/env sh
# mnem one-line installer for Linux and macOS.
#
# Usage (from the repo raw URL; host behind whatever CDN you run):
#   curl -fsSL https://raw.githubusercontent.com/Uranid/mnem/main/scripts/install.sh | sh
#
# Env vars:
#   MNEM_INSTALL_DIR   target dir for binaries (default: $HOME/.mnem/bin)
#   MNEM_VERSION       tag to install (default: latest)
#   MNEM_NO_MODIFY_PATH=1  skip shell-rc patching
#
# Downloads the matching release archive from GitHub Releases, extracts
# `mnem` + `mnem-mcp` to MNEM_INSTALL_DIR, and appends that directory to
# PATH via the user's shell rc.

set -eu

REPO="Uranid/mnem"
INSTALL_DIR="${MNEM_INSTALL_DIR:-$HOME/.mnem/bin}"
VERSION="${MNEM_VERSION:-latest}"

say() { printf '%s\n' "$*"; }
err() { say "mnem-install: $*" >&2; exit 1; }

detect_triple() {
    os=$(uname -s | tr '[:upper:]' '[:lower:]')
    arch=$(uname -m)
    case "$os-$arch" in
        linux-x86_64)             echo "x86_64-unknown-linux-gnu" ;;
        linux-aarch64|linux-arm64) echo "aarch64-unknown-linux-gnu" ;;
        darwin-x86_64)            echo "x86_64-apple-darwin" ;;
        darwin-arm64|darwin-aarch64) echo "aarch64-apple-darwin" ;;
        *) err "unsupported platform: $os-$arch. Build from source: https://github.com/$REPO" ;;
    esac
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || err "required: $1"
}

need_cmd curl
need_cmd tar
need_cmd uname
need_cmd mkdir

TRIPLE=$(detect_triple)
ARCHIVE="mnem-${TRIPLE}.tar.gz"

if [ "$VERSION" = "latest" ]; then
    URL="https://github.com/${REPO}/releases/latest/download/${ARCHIVE}"
else
    URL="https://github.com/${REPO}/releases/download/${VERSION}/${ARCHIVE}"
fi

say "mnem-install: triple=$TRIPLE version=$VERSION"
say "mnem-install: install_dir=$INSTALL_DIR"

mkdir -p "$INSTALL_DIR"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT INT TERM

say "mnem-install: downloading $URL"
if ! curl -fsSL "$URL" -o "$TMP/mnem.tar.gz"; then
    err "download failed. Check that a release for $TRIPLE exists at $URL."
fi

say "mnem-install: extracting..."
tar -xzf "$TMP/mnem.tar.gz" -C "$TMP"

# The release archive ships `mnem` and `mnem-mcp` at its root.
for bin in mnem mnem-mcp; do
    if [ ! -f "$TMP/$bin" ]; then
        err "archive does not contain $bin; cannot continue"
    fi
    install -m 0755 "$TMP/$bin" "$INSTALL_DIR/$bin"
done

# PATH hook. INSTALL_DIR is double-quoted so paths with spaces are fine;
# reject paths containing shell metacharacters before we emit the hook
# so we cannot accidentally write `eval`-able content into an rc file.
if [ -z "${MNEM_NO_MODIFY_PATH:-}" ]; then
    case "$INSTALL_DIR" in
        *'$'*|*'`'*|*$'\n'*|*';'*|*'|'*|*'&'*)
            err "MNEM_INSTALL_DIR contains shell metacharacters; refusing to patch rc files"
            ;;
    esac
    hook="export PATH=\"${INSTALL_DIR}:\$PATH\""
    for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
        if [ -f "$rc" ] && ! grep -qs ".mnem/bin" "$rc"; then
            {
                echo
                echo "# added by mnem-install $(date -u +%Y-%m-%dT%H:%M:%SZ)"
                echo "$hook"
            } >> "$rc"
            say "mnem-install: patched $rc"
        fi
    done
fi

say "mnem-install: done."
say
say "Next:"
say "  1. Restart your shell, or run: export PATH=\"$INSTALL_DIR:\$PATH\""
say "  2. mnem --version"
say "  3. mnem integrate      (wire Claude Desktop / Cursor / Continue / Zed)"
say "  4. mnem doctor         (health check)"
