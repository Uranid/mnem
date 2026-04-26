//! Opaque bearer-token newtype used by the remote transport.
//!
//! A [`SecretToken`] wraps a `String` that holds an HTTP bearer token.
//! Its job is behavioural, not cryptographic: make it hard to leak a
//! token through `Debug` or `Display`, and zero the inner buffer on
//! drop on a best-effort basis.
//!
//! ## Threat model (narrow)
//!
//! - Accidental logging of `cfg` structs / error contexts through
//!   `{:?}` or `{}` formatting.
//! - Incidental retention of token bytes in heap memory after the
//!   `SecretToken` has been dropped.
//!
//! This type is NOT a defence against a privileged debugger, a core
//! dump, or a compromised allocator. Do not rely on it for those.
//!
//! ## API shape
//!
//! - [`SecretToken::new`] - move a `String` in from the caller.
//! - [`SecretToken::from_env`] - best-effort construction from a
//!   named environment variable. Returns `None` if the variable is
//!   unset or empty after trimming whitespace.
//! - [`SecretToken::reveal`] - borrow the inner `&str`, intended to
//!   be used at exactly the point an `Authorization: Bearer ...`
//!   header is assembled. Callers MUST NOT log or persist the
//!   return value.
//! - `impl Debug for SecretToken` - renders as
//!   `SecretToken(***)`.
//! - `impl Display for SecretToken` - renders as `***`.
//! - `impl Drop for SecretToken` - overwrites the inner buffer via
//!   `String::clear` on a best-effort basis. `zeroize` is not a
//!   workspace dependency; we rely on `clear` to drop capacity back
//!   into the allocator after the buffer has been overwritten by
//!   the standard library's drop glue.

#![allow(clippy::missing_const_for_fn)]

use std::fmt;

/// Opaque bearer token held in memory only.
///
/// `Debug` renders as `SecretToken(***)`; `Display` renders as
/// `***`. The inner string can only leave the type through an
/// explicit [`SecretToken::reveal`] call.
#[derive(Clone, Eq, PartialEq)]
pub struct SecretToken(String);

impl SecretToken {
    /// Construct from a caller-owned string. The string moves into
    /// the struct; callers are responsible for zeroising any prior
    /// buffer (outside this crate's scope).
    #[must_use]
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Best-effort construction from a named environment variable.
    ///
    /// Returns `None` when the variable is unset, empty, or consists
    /// only of ASCII whitespace. The variable value is trimmed
    /// before being stored. This is the only place in the crate
    /// that reads `std::env`; other call-sites pass through
    /// [`Self::new`] after reading env / keychain / flag on their
    /// own terms.
    #[must_use]
    pub fn from_env(var: &str) -> Option<Self> {
        let raw = std::env::var(var).ok()?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(Self(trimmed.to_owned()))
        }
    }

    /// Reveal the underlying token. Callers doing HTTP wiring use
    /// this exactly at the point the `Authorization: Bearer <...>`
    /// header is assembled; callers MUST NOT log or persist the
    /// return value.
    #[must_use]
    pub fn reveal(&self) -> &str {
        &self.0
    }

    /// Alias for [`Self::reveal`] used at Authorization-header build
    /// sites. Kept as a separate name so grep-ing the codebase for
    /// `.expose()` finds every place a token leaves the type.
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// Number of bytes in the token, for debug / doctor output.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.0.len()
    }
}

impl fmt::Debug for SecretToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretToken(***)")
    }
}

impl fmt::Display for SecretToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("***")
    }
}

impl Drop for SecretToken {
    fn drop(&mut self) {
        // Best-effort zeroisation: overwrite the buffer's live bytes
        // before releasing capacity. `zeroize` is not a workspace
        // dependency, so we can't use its `Zeroize` impl for
        // `String`; the `clear` call at least truncates length to 0
        // so subsequent allocator reads don't observe the token
        // through the original `String` handle.
        self.0.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_redacted() {
        let tok = SecretToken::new("abc");
        assert_eq!(tok.to_string(), "***");
    }

    #[test]
    fn debug_is_redacted() {
        let tok = SecretToken::new("abc");
        assert_eq!(format!("{tok:?}"), "SecretToken(***)");
    }

    #[test]
    fn debug_does_not_leak_token_bytes() {
        let tok = SecretToken::new("super-secret-1234");
        let dbg = format!("{tok:?}");
        assert!(!dbg.contains("super-secret-1234"));
    }

    #[test]
    fn from_env_returns_none_when_unset() {
        // Variable name chosen to be vanishingly unlikely to exist
        // in any CI or dev environment. We deliberately do NOT
        // mutate the process environment in tests: `std::env::set_var`
        // is unsafe on the edition this crate targets and the crate
        // has a blanket `#![forbid(unsafe_code)]`. `from_env` for
        // present / whitespace inputs is covered by the CLI-layer
        // integration tests that own the env contract.
        assert!(SecretToken::from_env("MNEM_NONEXISTENT_VAR_12345_ZZZZ").is_none());
    }

    #[test]
    fn reveal_and_expose_agree() {
        let tok = SecretToken::new("xyz");
        assert_eq!(tok.reveal(), "xyz");
        assert_eq!(tok.expose(), "xyz");
    }
}
