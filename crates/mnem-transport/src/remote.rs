//! [`RemoteConfig`] and the `.mnem/config.toml` `[remote.<name>]`
//! schema.
//!
//! Scope of this module is deliberately narrow: a `RemoteConfig` is a
//! pure data record describing where a remote mnem repository lives
//! and which capabilities the local peer will advertise for it. There
//! is no network code here. The on-disk `config.toml` section is
//! parsed into a [`RemoteConfigFile`] and serialised back out again;
//! that map is how `mnem-cli`'s future `remote add / remove / list`
//! verbs will talk to this crate.
//!
//! ## Config file schema (v0)
//!
//! ```toml
//! [remote.origin]
//! url = "https://example.com/repo/alice/notes"
//! # Optional capability overrides. When omitted, the client advertises
//! # every capability this build supports.
//! capabilities = ["have-set-bloom", "atomic-push"]
//! # Optional: name of an environment variable holding the bearer
//! # token. The token itself is NEVER stored in config.toml.
//! token_env = "MNEM_ORIGIN_TOKEN"
//!
//! [remote.backup]
//! url = "file:///srv/mnem-mirrors/notes.car"
//! ```
//!
//! ## Security model
//!
//! `RemoteConfig::token` is populated at run time from the process
//! environment (or, later, a platform keychain via `mnem remote
//! set-token`). It MUST NOT appear on disk. A `RemoteConfig` parsed
//! from a `config.toml` always starts with `token = None`; callers
//! wire up authentication by reading `token_env`, looking up the
//! environment variable, and calling
//! [`RemoteConfig::with_token`]. That step is a CLI-layer
//! responsibility; this crate deliberately does not touch `std::env`.

// Pedantic doc-length + closure warnings on design-heavy modules
// are opinionated; prose and explicit match arms are deliberate.
#![allow(
    clippy::too_long_first_doc_paragraph,
    clippy::missing_const_for_fn,
    clippy::option_if_let_else,
    clippy::needless_collect
)]

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::protocol::Capability;

/// On-disk shape of a single `[remote.<name>]` section.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RemoteConfigFile {
    /// Remote URL. Any URL scheme is accepted at the config layer;
    /// interpretation (`https://`, `file://`, `mnem+ssh://`, ...) is
    /// up to the transport driver. PR 2 does not implement any
    /// drivers.
    pub url: String,
    /// Optional capability allow-list. When `None`, the client
    /// advertises every capability in [`Capability::all`] that this
    /// build knows. When `Some`, only these capabilities are
    /// advertised (useful for interop-testing against older servers
    /// or for opting out of `filter-spec`).
    ///
    /// Unknown capability strings on the wire are tolerated and
    /// silently dropped (forward-compat). A `config.toml` file that
    /// lists an unknown string will likewise parse into this
    /// capability set with the unknown dropped. This matches the
    /// forward-compat rule in [`crate::protocol::parse_capabilities`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<String>>,
    /// Optional name of an environment variable holding the bearer
    /// token. When set, the CLI layer reads that variable at request
    /// time and injects the token via [`RemoteConfig::with_token`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_env: Option<String>,
}

/// In-memory representation of a single remote. Unlike
/// [`RemoteConfigFile`] this type holds the parsed capability set,
/// optionally holds a runtime-only bearer token, and is what the
/// network layer actually consumes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemoteConfig {
    /// Short name for this remote (`origin`, `backup`, ...). Matches
    /// the `[remote.<NAME>]` section header the config was parsed
    /// from. Used as a key in [`View::remote_refs`][remote_refs] so
    /// tracking refs render as `origin/main`, `backup/release`, ...
    ///
    /// [remote_refs]: mnem_core::objects::View::remote_refs
    pub name: String,
    /// Remote URL. See [`RemoteConfigFile::url`].
    pub url: String,
    /// Capability allow-list advertised by the local peer when
    /// talking to this remote. Empty means "advertise every built-in
    /// capability"; non-empty restricts the ad.
    pub capabilities: HashSet<Capability>,
    /// Optional environment variable name from which the CLI layer
    /// will load the bearer token at run time. Persisted in
    /// `config.toml`; the token itself never is.
    pub token_env: Option<String>,
    /// Optional bearer token. Populated from the environment (or a
    /// future platform keychain) by the CLI / HTTP client, never by
    /// reading `config.toml`. `Debug` intentionally redacts this
    /// field to keep accidental `println!("{cfg:?}")` calls safe.
    pub token: Option<SecretToken>,
}

// `SecretToken` lives in [`crate::secret_token`]. Re-exported at the
// crate root and from this module so historic `use
// crate::remote::SecretToken` paths keep compiling.
pub use crate::secret_token::SecretToken;

impl RemoteConfig {
    /// Build a fresh [`RemoteConfig`] with no capabilities and no
    /// token. Callers typically go through [`Self::from_file`] or a
    /// top-level [`parse_config`] call; this constructor is for tests
    /// and programmatic use.
    #[must_use]
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            capabilities: HashSet::new(),
            token_env: None,
            token: None,
        }
    }

    /// Attach a bearer token at run time. Intended to be called
    /// exactly once by the CLI after reading `token_env` out of the
    /// process environment.
    #[must_use]
    pub fn with_token(mut self, token: SecretToken) -> Self {
        self.token = Some(token);
        self
    }

    /// Add a capability to the local peer's advertised set.
    #[must_use]
    pub fn with_capability(mut self, cap: Capability) -> Self {
        self.capabilities.insert(cap);
        self
    }

    /// Merge a parsed `[remote.<name>]` file section with its name
    /// (from the section header) into a runtime [`RemoteConfig`].
    ///
    /// Unknown capability strings are silently dropped
    /// (forward-compat). An empty / absent capability list in the
    /// file becomes the full built-in capability set at runtime.
    #[must_use]
    pub fn from_file(name: impl Into<String>, file: RemoteConfigFile) -> Self {
        let caps = match file.capabilities {
            None => Capability::all().iter().copied().collect(),
            Some(list) => list
                .iter()
                .filter_map(|s| s.parse::<Capability>().ok())
                .collect(),
        };
        Self {
            name: name.into(),
            url: file.url,
            capabilities: caps,
            token_env: file.token_env,
            token: None,
        }
    }

    /// Project back into the `[remote.<name>]` on-disk shape, suitable
    /// for round-tripping through `toml::to_string_pretty`. Tokens are
    /// never written out.
    ///
    /// Capability ordering in the output is wire-string ascending, so
    /// two configs with the same logical capability set round-trip to
    /// byte-identical TOML.
    #[must_use]
    pub fn to_file(&self) -> RemoteConfigFile {
        let mut caps: Vec<String> = self
            .capabilities
            .iter()
            .map(|c| c.as_wire_str().to_owned())
            .collect();
        caps.sort();
        RemoteConfigFile {
            url: self.url.clone(),
            capabilities: if caps.is_empty() { None } else { Some(caps) },
            token_env: self.token_env.clone(),
        }
    }
}

/// Parsed `.mnem/config.toml` fragment carrying a `[remote.<name>]`
/// table. This is just the remote section; the real config in
/// `mnem-cli` is a superset that flattens over the top.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct RemoteSection {
    /// One entry per `[remote.<name>]` section. `BTreeMap` for
    /// deterministic iteration and output ordering.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub remote: BTreeMap<String, RemoteConfigFile>,
}

impl RemoteSection {
    /// Convert every parsed `[remote.<name>]` into a runtime
    /// [`RemoteConfig`]. Iterator order matches `BTreeMap` iteration
    /// (ascending by remote name).
    pub fn into_runtime(self) -> impl Iterator<Item = RemoteConfig> {
        self.remote
            .into_iter()
            .map(|(name, file)| RemoteConfig::from_file(name, file))
    }
}

/// Parse a `.mnem/config.toml` payload and return every
/// `[remote.<name>]` section it contains. Other top-level tables are
/// ignored, so this function is safe to call on the full
/// `config.toml` and can be combined with other parsers that pick
/// different sections off the same payload.
///
/// # Errors
///
/// Returns the underlying [`toml::de::Error`] on malformed TOML.
pub fn parse_config(s: &str) -> Result<RemoteSection, toml::de::Error> {
    toml::from_str(s)
}

/// Inverse of [`parse_config`]: serialise a [`RemoteSection`] back to
/// TOML. The output is pretty-printed and sorted deterministically
/// (because `BTreeMap`).
///
/// # Errors
///
/// Returns the underlying [`toml::ser::Error`] on serialisation
/// failure; this should not happen for the shapes defined here.
pub fn serialize_config(section: &RemoteSection) -> Result<String, toml::ser::Error> {
    toml::to_string_pretty(section)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_TOML: &str = r#"
[remote.origin]
url = "https://example.com/repo/alice/notes"
capabilities = ["have-set-bloom", "atomic-push"]
token_env = "MNEM_ORIGIN_TOKEN"

[remote.backup]
url = "file:///srv/mnem-mirrors/notes.car"
"#;

    #[test]
    fn parse_config_extracts_named_remotes() {
        let parsed = parse_config(SAMPLE_TOML).unwrap();
        assert_eq!(parsed.remote.len(), 2);
        let origin = &parsed.remote["origin"];
        assert_eq!(origin.url, "https://example.com/repo/alice/notes");
        assert_eq!(
            origin.capabilities.as_ref().unwrap(),
            &vec!["have-set-bloom".to_owned(), "atomic-push".to_owned()],
        );
        assert_eq!(origin.token_env.as_deref(), Some("MNEM_ORIGIN_TOKEN"));

        let backup = &parsed.remote["backup"];
        assert_eq!(backup.url, "file:///srv/mnem-mirrors/notes.car");
        assert!(backup.capabilities.is_none());
        assert!(backup.token_env.is_none());
    }

    #[test]
    fn from_file_resolves_capabilities() {
        let file = RemoteConfigFile {
            url: "https://example.com".into(),
            capabilities: Some(vec![
                "have-set-bloom".into(),
                "atomic-push".into(),
                "no-such-capability".into(),
            ]),
            token_env: None,
        };
        let cfg = RemoteConfig::from_file("origin", file);
        assert_eq!(cfg.name, "origin");
        // Unknown string is silently dropped; known ones survive.
        assert_eq!(cfg.capabilities.len(), 2);
        assert!(cfg.capabilities.contains(&Capability::HaveSetBloom));
        assert!(cfg.capabilities.contains(&Capability::AtomicPush));
    }

    #[test]
    fn from_file_missing_capabilities_means_all() {
        let file = RemoteConfigFile {
            url: "https://example.com".into(),
            capabilities: None,
            token_env: None,
        };
        let cfg = RemoteConfig::from_file("origin", file);
        // Missing capability list == advertise everything built-in.
        assert_eq!(cfg.capabilities.len(), Capability::all().len());
    }

    #[test]
    fn remote_config_round_trips_through_toml() {
        // Parse sample, convert to runtime, back to file shape, emit
        // TOML, re-parse, and check semantic equality.
        let parsed = parse_config(SAMPLE_TOML).unwrap();
        let runtime: Vec<RemoteConfig> = parsed.clone().into_runtime().collect();
        let round_tripped = RemoteSection {
            remote: runtime
                .into_iter()
                .map(|r| (r.name.clone(), r.to_file()))
                .collect(),
        };
        let emitted = serialize_config(&round_tripped).unwrap();
        let re_parsed = parse_config(&emitted).unwrap();
        assert_eq!(re_parsed.remote.len(), parsed.remote.len());
        // URLs survive exactly.
        for (name, file) in &parsed.remote {
            assert_eq!(re_parsed.remote[name].url, file.url);
        }
        // origin's explicit capability list survives (unknown strings
        // are never produced by the runtime so nothing is dropped here).
        let origin_caps = re_parsed.remote["origin"].capabilities.as_ref().unwrap();
        assert!(origin_caps.contains(&"have-set-bloom".to_owned()));
        assert!(origin_caps.contains(&"atomic-push".to_owned()));
    }

    #[test]
    fn secret_token_debug_redacts() {
        let tok = SecretToken::new("super-secret-1234");
        let dbg = format!("{tok:?}");
        assert!(
            !dbg.contains("super-secret-1234"),
            "debug leaked token: {dbg}"
        );
        // New Debug shape (see `crate::secret_token`): fixed mask so
        // even the byte-length doesn't leak.
        assert_eq!(dbg, "SecretToken(***)");
        assert_eq!(tok.reveal(), "super-secret-1234");
    }

    #[test]
    fn remote_config_debug_redacts_token() {
        let cfg = RemoteConfig::new("origin", "https://example.com")
            .with_token(SecretToken::new("abc123"));
        let dbg = format!("{cfg:?}");
        assert!(!dbg.contains("abc123"), "token leaked in debug: {dbg}");
    }

    #[test]
    fn token_never_serialised_to_toml() {
        let cfg = RemoteConfig::new("origin", "https://example.com")
            .with_token(SecretToken::new("abc123"));
        let file = cfg.to_file();
        let s = toml::to_string_pretty(&file).unwrap();
        assert!(!s.contains("abc123"), "token leaked through to_file: {s}");
    }

    #[test]
    fn empty_config_parses_to_empty_section() {
        let parsed = parse_config("").unwrap();
        assert!(parsed.remote.is_empty());
    }
}
