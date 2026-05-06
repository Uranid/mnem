//! Async HTTP `RemoteClient`.
//!
//! Behind the `client` feature so the default `mnem-transport` build
//! stays WASM-clean (no tokio, no reqwest). This module ships the
//! full four-verb surface: `list_refs` + capability negotiation, plus
//! the CAR-body verbs `fetch_blocks`, `push_blocks`, `advance_head`
//! (B3.3).
//!
//! ## Trait
//!
//! [`RemoteClient`] is the async surface mnem-cli / mnem http talk to.
//! It is async-trait free: the associated-type-position `-> impl
//! Future` shape needs `trait-return-impl-trait` which is stable but
//! noisy on older toolchains; we use a concrete `Pin<Box<dyn
//! Future>>` return shape and an inherent-method-per-verb pattern on
//! [`HttpRemoteClient`] until async-fn-in-trait stabilises in the
//! workspace MSRV.
//!
//! ## HTTP semantics (frozen here so B3.3 can fill bodies without
//! drift)
//!
//! | Verb | Method + path | Auth required? | `mnem-protocol` | `mnem-capabilities` |
//! |---|---|---|---|---|
//! | `list_refs`    | `GET /remote/v1/refs`              | no  | 1 | advertised |
//! | `fetch_blocks` | `POST /remote/v1/fetch-blocks`     | no  | 1 | agreed |
//! | `push_blocks`  | `POST /remote/v1/push-blocks`      | yes | 1 | agreed |
//! | `advance_head` | `POST /remote/v1/advance-head`     | yes | 1 | agreed |
//!
//! Bearer tokens are injected ONLY on push endpoints. The `Authorization`
//! header never goes out on `list_refs` / `fetch_blocks`. This is a
//! defence-in-depth choice: read-side requests MUST stay usable from
//! unauthenticated contexts (caches, mirrors) and leaking a write
//! token on a GET is a known mis-use of bearer auth.

#![cfg(feature = "client")]
#![allow(
    clippy::missing_errors_doc,
    clippy::module_name_repetitions,
    clippy::too_long_first_doc_paragraph
)]

use std::future::Future;
use std::pin::Pin;

use bytes::Bytes;
use mnem_core::id::Cid;
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};

use crate::error::ClientError;
use crate::have_set::{BloomHaveSet, HaveSet};
use crate::protocol::{
    CAPABILITIES_HEADER, Capability, CapabilitySet, PROTOCOL_HEADER, PROTOCOL_VERSION,
};
use crate::remote::RemoteConfig;
use crate::secret_token::SecretToken;

/// Boxed-future alias so the trait methods stay object-safe on
/// stable Rust without pulling in `async-trait`.
type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Async surface mnem-cli / mnem http talk to when speaking to a
/// remote peer. Every method corresponds to one of the four wire
/// verbs documented in the module-level docs.
///
/// This trait is deliberately narrow: it carries only the bytes the
/// caller needs and folds all error kinds into [`ClientError`]. The
/// CAR body on the wire is a `Bytes` blob for now; streaming framing
/// lands under B3.3 when the CAR framing module is spec-pinned.
pub trait RemoteClient: Send + Sync {
    /// `GET /remote/v1/refs` - enumerate the server's current refs
    /// and the capability set it will negotiate against.
    fn list_refs(&self) -> BoxFuture<'_, Result<RefsResponse, ClientError>>;

    /// `POST /remote/v1/fetch-blocks` - request a CAR body containing
    /// every block in `wants` and its transitive graph, minus
    /// anything in `have_set`.
    fn fetch_blocks(
        &self,
        wants: Vec<Cid>,
        have_set: BloomHaveSet,
    ) -> BoxFuture<'_, Result<Bytes, ClientError>>;

    /// `POST /remote/v1/push-blocks` - upload a CAR body. Requires
    /// the bearer token.
    fn push_blocks(&self, car_body: Bytes) -> BoxFuture<'_, Result<PushResponse, ClientError>>;

    /// `POST /remote/v1/advance-head` - atomic compare-and-swap on a
    /// named ref. Requires the bearer token. Returns
    /// [`ClientError::CasMismatch`] on 409.
    fn advance_head(
        &self,
        old: Cid,
        new: Cid,
        ref_name: String,
    ) -> BoxFuture<'_, Result<(), ClientError>>;
}

/// `GET /remote/v1/refs` response body.
///
/// The on-wire shape matches the server's [`RefsResponse`] in
/// `crates/mnem-http/src/routes/remote.rs`:
///
/// ```json
/// {
///   "head": "bafy..." | null,
///   "refs": { "HEAD": "bafy...", "main": "bafy..." },
///   "capabilities": ["have-set-bloom", "atomic-push", ...]
/// }
/// ```
///
/// The client deserialises via a private DTO (`RefsWireBody`) so
/// invalid CIDs become a `Protocol` error instead of a `Deserialize`
/// panic, and unknown capability strings are silently dropped via
/// [`parse_wire_capabilities`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefsResponse {
    /// Canonical default-branch head, or `None` on a fresh server.
    /// Mirrored under the `"HEAD"` key of `refs` when present; this
    /// top-level field exists so clients that only care about the
    /// default branch can index a single well-known slot.
    pub head: Option<Cid>,
    /// Map of ref name -> current head CID, as the server sees it.
    pub refs: std::collections::BTreeMap<String, Cid>,
    /// Capabilities the server is willing to speak for this session.
    /// The client intersects with its own advertisement; the result
    /// is the agreed set. Any unknown capability strings the server
    /// advertised are dropped here - see [`parse_wire_capabilities`].
    pub capabilities: Vec<Capability>,
}

/// Private on-wire DTO for `GET /remote/v1/refs`. Kept separate from
/// the public [`RefsResponse`] so we can (a) tolerate unknown
/// capability strings without failing the deserialise, and (b)
/// surface invalid CIDs as a `Protocol` error rather than a generic
/// deserialise failure.
#[derive(Debug, Deserialize)]
struct RefsWireBody {
    #[serde(default)]
    head: Option<String>,
    #[serde(default)]
    refs: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    capabilities: Vec<String>,
}

/// Parse a slice of wire-form capability strings into `Capability`s,
/// silently dropping any value unknown to this build. This is the
/// forward-compat policy requires: the server may advertise
/// capabilities added in a minor release; older clients MUST ignore
/// them rather than failing the handshake.
#[must_use]
pub fn parse_wire_capabilities(raw: &[String]) -> Vec<Capability> {
    let mut out: Vec<Capability> = raw
        .iter()
        .filter_map(|s| s.parse::<Capability>().ok())
        .collect();
    // Sort-dedupe for determinism; the server already emits sorted
    // but nothing in the wire contract guarantees it.
    out.sort_by_key(Capability::as_wire_str);
    out.dedup();
    out
}

/// `POST /remote/v1/push-blocks` response body. The server echoes
/// the CID count it accepted; the client cross-checks against the
/// CAR roots it sent.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PushResponse {
    /// Number of blocks the server stored from this push.
    pub accepted: u64,
    /// The CID of the root block after the push, echoed for the
    /// caller's optimistic cache invalidation.
    pub root: Cid,
}

/// Reference HTTP [`RemoteClient`] implementation. A single instance
/// is tied to exactly one [`RemoteConfig`] and one base URL.
#[derive(Debug)]
pub struct HttpRemoteClient {
    client: Client,
    base_url: String,
    token: Option<SecretToken>,
    /// Capabilities advertised by the local peer. After a successful
    /// [`Self::negotiate_capabilities`] this is narrowed to the
    /// intersection with the server's advertised set.
    capabilities: CapabilitySet,
}

impl HttpRemoteClient {
    /// Build a client from a parsed [`RemoteConfig`]. A fresh
    /// [`reqwest::Client`] is created per call; callers who want to
    /// pool connections across remotes should build one client
    /// explicitly and share it via the future `with_client`
    /// constructor (deferred, tracked in B3.3).
    #[must_use]
    pub fn new(cfg: RemoteConfig) -> Self {
        let capabilities = if cfg.capabilities.is_empty() {
            CapabilitySet::all_known()
        } else {
            CapabilitySet::with_caps(cfg.capabilities.iter().copied())
        };
        Self {
            client: Client::new(),
            base_url: cfg.url.trim_end_matches('/').to_owned(),
            token: cfg.token,
            capabilities,
        }
    }

    /// Negotiate capabilities with the remote by calling `list_refs`
    /// and intersecting the server's advertised set with the local
    /// one. After this call, [`Self::capabilities`] returns the
    /// agreed set that all subsequent verbs must operate under.
    pub async fn negotiate_capabilities(&mut self) -> Result<(), ClientError> {
        let refs = self.list_refs_impl().await?;
        let server_caps = CapabilitySet::with_caps(refs.capabilities.iter().copied());
        self.capabilities = self.capabilities.intersect(&server_caps);
        Ok(())
    }

    /// The agreed-upon capability set. Before
    /// [`Self::negotiate_capabilities`] has been called this is the
    /// local advertisement; after, it's the intersection with the
    /// server.
    #[must_use]
    pub const fn capabilities(&self) -> &CapabilitySet {
        &self.capabilities
    }

    // -- inherent impls per verb -----------------------------------------

    async fn list_refs_impl(&self) -> Result<RefsResponse, ClientError> {
        // `list_refs` is read-side: token MUST NOT be attached.
        let url = format!("{}/remote/v1/refs", self.base_url);
        let req = self
            .client
            .get(&url)
            .header(PROTOCOL_HEADER, PROTOCOL_VERSION.to_string())
            .header(CAPABILITIES_HEADER, self.capabilities.serialize());
        let resp = req.send().await?;
        let status = resp.status();
        if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
            return Err(ClientError::Auth(format!(
                "list_refs rejected with {status}"
            )));
        }
        if !status.is_success() {
            return Err(ClientError::Protocol(format!(
                "list_refs: unexpected status {status}"
            )));
        }
        let body = resp.bytes().await?;
        let wire: RefsWireBody = serde_json::from_slice(&body)?;
        // Parse the wire strings into strongly-typed values. Invalid
        // CIDs are a protocol error; unknown capability strings are
        // dropped forward-compat rules.
        let head =
            match wire.head {
                None => None,
                Some(ref s) => Some(Cid::parse_str(s).map_err(|e| {
                    ClientError::Protocol(format!("list_refs: invalid head CID: {e}"))
                })?),
            };
        let mut refs = std::collections::BTreeMap::new();
        for (name, cid_str) in wire.refs {
            let cid = Cid::parse_str(&cid_str).map_err(|e| {
                ClientError::Protocol(format!("list_refs: invalid CID for ref `{name}`: {e}"))
            })?;
            refs.insert(name, cid);
        }
        let capabilities = parse_wire_capabilities(&wire.capabilities);
        Ok(RefsResponse {
            head,
            refs,
            capabilities,
        })
    }

    /// Build the bearer-auth `Authorization` header value, if a
    /// token is configured. Returns `None` when the client holds no
    /// token; callers MUST NOT fall through to an unauthenticated
    /// push in that case.
    fn bearer_header(&self) -> Option<String> {
        self.token
            .as_ref()
            .map(|t| format!("Bearer {}", t.expose()))
    }
}

impl HttpRemoteClient {
    /// `POST /remote/v1/fetch-blocks`. Read-side; no bearer.
    async fn fetch_blocks_impl(
        &self,
        wants: Vec<Cid>,
        have_set: BloomHaveSet,
    ) -> Result<Bytes, ClientError> {
        let url = format!("{}/remote/v1/fetch-blocks", self.base_url);
        let wants_str: Vec<String> = wants.iter().map(Cid::to_string).collect();
        let body = serde_json::json!({
            "wants": wants_str,
            "have_set": have_set.serialize(),
        });
        let resp = self
            .client
            .post(&url)
            .header(PROTOCOL_HEADER, PROTOCOL_VERSION.to_string())
            .header(CAPABILITIES_HEADER, self.capabilities.serialize())
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
            return Err(ClientError::Auth(format!(
                "fetch_blocks rejected with {status}"
            )));
        }
        if !status.is_success() {
            return Err(ClientError::Protocol(format!(
                "fetch_blocks: unexpected status {status}"
            )));
        }
        let bytes = resp.bytes().await?;
        Ok(bytes)
    }

    /// `POST /remote/v1/push-blocks`. Bearer-required.
    async fn push_blocks_impl(&self, car_body: Bytes) -> Result<PushResponse, ClientError> {
        let url = format!("{}/remote/v1/push-blocks", self.base_url);
        let auth = self
            .bearer_header()
            .ok_or_else(|| ClientError::Auth("push_blocks: no bearer token configured".into()))?;
        let resp = self
            .client
            .post(&url)
            .header(PROTOCOL_HEADER, PROTOCOL_VERSION.to_string())
            .header(CAPABILITIES_HEADER, self.capabilities.serialize())
            .header(reqwest::header::AUTHORIZATION, auth)
            .header(reqwest::header::CONTENT_TYPE, "application/vnd.ipld.car")
            .body(car_body)
            .send()
            .await?;
        let status = resp.status();
        if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
            return Err(ClientError::Auth(format!(
                "push_blocks rejected with {status}"
            )));
        }
        if !status.is_success() {
            return Err(ClientError::Protocol(format!(
                "push_blocks: unexpected status {status}"
            )));
        }
        // Server response carries `{staged, blocks_accepted}`; we
        // project into `PushResponse { accepted, root }` for the
        // client shape. `staged` is optional server-side but the
        // importer rejects empty-root CARs, so in practice it's
        // always present.
        #[derive(Deserialize)]
        struct Wire {
            staged: Option<String>,
            blocks_accepted: u64,
        }
        let body = resp.bytes().await?;
        let wire: Wire = serde_json::from_slice(&body)?;
        let root_str = wire.staged.ok_or_else(|| {
            ClientError::Protocol("push_blocks: server returned null staged root".into())
        })?;
        let root = Cid::parse_str(&root_str).map_err(|e| {
            ClientError::Protocol(format!("push_blocks: server staged root parse: {e}"))
        })?;
        Ok(PushResponse {
            accepted: wire.blocks_accepted,
            root,
        })
    }

    /// `POST /remote/v1/advance-head`. Bearer-required. Maps 409 to
    /// [`ClientError::CasMismatch`].
    async fn advance_head_impl(
        &self,
        old: Cid,
        new: Cid,
        ref_name: String,
    ) -> Result<(), ClientError> {
        let url = format!("{}/remote/v1/advance-head", self.base_url);
        let auth = self
            .bearer_header()
            .ok_or_else(|| ClientError::Auth("advance_head: no bearer token configured".into()))?;
        let body = serde_json::json!({
            "old": old.to_string(),
            "new": new.to_string(),
            "ref": ref_name,
        });
        let resp = self
            .client
            .post(&url)
            .header(PROTOCOL_HEADER, PROTOCOL_VERSION.to_string())
            .header(CAPABILITIES_HEADER, self.capabilities.serialize())
            .header(reqwest::header::AUTHORIZATION, auth)
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        if status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN {
            return Err(ClientError::Auth(format!(
                "advance_head rejected with {status}"
            )));
        }
        if status == StatusCode::CONFLICT {
            // Server replies with `{current: <cid>}` on CAS
            // mismatch. If parsing fails we still surface a
            // mismatch with the client's `old` echoed back, since
            // that is what the caller needs to retry.
            #[derive(Deserialize)]
            struct CurrentBody {
                current: Option<String>,
            }
            let bytes = resp.bytes().await.unwrap_or_default();
            let actual = serde_json::from_slice::<CurrentBody>(&bytes)
                .ok()
                .and_then(|c| c.current)
                .and_then(|s| Cid::parse_str(&s).ok())
                .unwrap_or_else(|| old.clone());
            return Err(ClientError::CasMismatch {
                ref_name,
                expected: old,
                actual,
            });
        }
        if !status.is_success() {
            return Err(ClientError::Protocol(format!(
                "advance_head: unexpected status {status}"
            )));
        }
        let _ = new;
        Ok(())
    }
}

impl RemoteClient for HttpRemoteClient {
    fn list_refs(&self) -> BoxFuture<'_, Result<RefsResponse, ClientError>> {
        Box::pin(self.list_refs_impl())
    }

    fn fetch_blocks(
        &self,
        wants: Vec<Cid>,
        have_set: BloomHaveSet,
    ) -> BoxFuture<'_, Result<Bytes, ClientError>> {
        Box::pin(self.fetch_blocks_impl(wants, have_set))
    }

    fn push_blocks(&self, car_body: Bytes) -> BoxFuture<'_, Result<PushResponse, ClientError>> {
        Box::pin(self.push_blocks_impl(car_body))
    }

    fn advance_head(
        &self,
        old: Cid,
        new: Cid,
        ref_name: String,
    ) -> BoxFuture<'_, Result<(), ClientError>> {
        Box::pin(self.advance_head_impl(old, new, ref_name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;

    #[tokio::test]
    async fn list_refs_omits_authorization_header() {
        let server = MockServer::start_async().await;
        let mock = server
            .mock_async(|when, then| {
                // Authorization MUST NOT be present on
                // list_refs: read-side verb, no token.
                when.method(GET)
                    .path("/remote/v1/refs")
                    .header_missing("authorization");
                then.status(200)
                    .header("content-type", "application/json")
                    .body(r#"{"refs":{},"capabilities":["have-set-bloom","atomic-push"]}"#);
            })
            .await;

        let cfg = RemoteConfig::new("origin", server.base_url())
            .with_token(SecretToken::new("unit-test-token"));
        let client = HttpRemoteClient::new(cfg);
        let refs = client.list_refs_impl().await.expect("list_refs ok");
        assert!(refs.capabilities.contains(&Capability::HaveSetBloom));
        assert!(refs.capabilities.contains(&Capability::AtomicPush));
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn negotiate_capabilities_intersects() {
        let server = MockServer::start_async().await;
        let _mock = server
            .mock_async(|when, then| {
                when.method(GET).path("/remote/v1/refs");
                then.status(200)
                    .header("content-type", "application/json")
                    // Server knows have-set-bloom + atomic-push.
                    .body(r#"{"refs":{},"capabilities":["have-set-bloom","atomic-push"]}"#);
            })
            .await;

        // Local advertises have-set-bloom + push-negotiate.
        let cfg = RemoteConfig::new("origin", server.base_url())
            .with_capability(Capability::HaveSetBloom)
            .with_capability(Capability::PushNegotiate);
        let mut client = HttpRemoteClient::new(cfg);
        client.negotiate_capabilities().await.expect("negotiate ok");
        // Intersection is {have-set-bloom}. atomic-push was
        // server-only, push-negotiate was client-only.
        let agreed = client.capabilities();
        assert!(agreed.contains(Capability::HaveSetBloom));
        assert!(!agreed.contains(Capability::AtomicPush));
        assert!(!agreed.contains(Capability::PushNegotiate));
    }

    #[test]
    fn bearer_header_includes_token_when_present() {
        let cfg = RemoteConfig::new("origin", "https://example.com")
            .with_token(SecretToken::new("tok-abc"));
        let client = HttpRemoteClient::new(cfg);
        assert_eq!(client.bearer_header().as_deref(), Some("Bearer tok-abc"));
    }

    #[test]
    fn bearer_header_none_when_no_token() {
        let cfg = RemoteConfig::new("origin", "https://example.com");
        let client = HttpRemoteClient::new(cfg);
        assert!(client.bearer_header().is_none());
    }
}
