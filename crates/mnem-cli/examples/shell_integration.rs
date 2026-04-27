//! How to spawn the `mnem` binary from Rust and capture its output.
//!
//! Many integration builders (build scripts, AI-agent shims, custom
//! wrappers) don't want to depend on the library surface of
//! `mnem-core` directly; they want the friendly, stable `mnem` CLI as
//! a subprocess and a way to feed it flags + parse its stdout.
//!
//! This example shows the pattern: locate the freshly-built binary,
//! invoke it with `--help`, and print the first few lines. Under
//! `cargo run --example shell_integration -p mnem-cli` the binary is
//! at `target/debug/mnem[.exe]` next to the example. In production
//! the caller should resolve it via `$PATH` or pass an absolute path
//! the operator supplied.
//!
//! See also:
//! - `docs/guide/cli.md` - user-facing command reference.
//! - `docs/guide/installation.md` - how to get `mnem` on `$PATH`.
//! - `mnem completions <shell>` - bash/zsh/fish/powershell/elvish.
//!
//! Run:
//! ```console
//! cargo run --example shell_integration -p mnem-cli
//! ```

use std::path::PathBuf;
use std::process::Command;

fn locate_mnem() -> PathBuf {
    // `cargo run --example` drops examples alongside the primary
    // binary, so resolving "../mnem[.exe]" from the current exe's
    // directory is the reliable Windows + POSIX trick.
    let ext = if cfg!(windows) { ".exe" } else { "" };
    let exe = std::env::current_exe().expect("current_exe");
    exe.parent()
        .expect("examples/ parent")
        .parent()
        .expect("debug/ parent")
        .join(format!("mnem{ext}"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mnem = locate_mnem();
    if !mnem.exists() {
        eprintln!(
            "error: {} not found. Run `cargo build -p mnem-cli` before this example.",
            mnem.display()
        );
        std::process::exit(1);
    }
    println!("locating mnem at: {}", mnem.display());

    // `mnem --help` is a harmless, read-only call. No repo required.
    let output = Command::new(&mnem).arg("--help").output()?;
    if !output.status.success() {
        eprintln!(
            "mnem --help failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        std::process::exit(1);
    }
    let help = String::from_utf8_lossy(&output.stdout);
    println!("---- mnem --help (first 6 lines) ----");
    for line in help.lines().take(6) {
        println!("{line}");
    }
    println!("---- ...");
    println!("OK");
    Ok(())
}
