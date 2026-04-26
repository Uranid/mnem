//! `Link<T>` - a phantom-typed [`Cid`].
//!
//! A bare [`Cid`] points at "some content." A [`Link<T>`] points at "content
//! that is a `T`." The generic parameter is never materialized; it exists
//! solely to make `fn parents(&self) -> &[Link<Commit>]` refuse a
//! `Link<Node>` at compile time, closing a large category of
//! reference-mixing bugs.
//!
//! On the wire a `Link<T>` is identical to a [`Cid`] - same bytes, same
//! CBOR tag. The phantom type is a pure Rust-level convenience.

use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::id::cid::Cid;

/// A typed content reference.
///
/// `Link<T>` is a [`Cid`] annotated at the type level with the kind of
/// object it addresses. Construct via [`Link::new`]; extract via
/// [`Link::cid`].
pub struct Link<T: ?Sized> {
    cid: Cid,
    _target: PhantomData<fn() -> T>,
}

impl<T: ?Sized> Link<T> {
    /// Construct from a raw CID.
    ///
    /// No runtime validation is performed - the caller is responsible for
    /// ensuring the CID actually addresses a `T`. To validate, read the
    /// content via the object store and decode into `T`.
    #[must_use]
    pub const fn new(cid: Cid) -> Self {
        Self {
            cid,
            _target: PhantomData,
        }
    }

    /// Borrow the underlying CID.
    #[must_use]
    pub const fn cid(&self) -> &Cid {
        &self.cid
    }

    /// Consume and return the underlying CID, dropping the phantom type.
    #[must_use]
    pub const fn into_cid(self) -> Cid {
        self.cid
    }

    /// Reinterpret this link as pointing at a different type without any
    /// runtime check. Use only when converting between representations of
    /// the same logical object.
    #[must_use]
    pub const fn transmute<U: ?Sized>(self) -> Link<U> {
        Link::new(self.cid)
    }
}

// Manual trait impls that delegate to `cid`, avoiding derive-bound
// propagation on the phantom type parameter (same pattern as StableId).
impl<T: ?Sized> Clone for Link<T> {
    fn clone(&self) -> Self {
        Self::new(self.cid.clone())
    }
}

impl<T: ?Sized> PartialEq for Link<T> {
    fn eq(&self, other: &Self) -> bool {
        self.cid == other.cid
    }
}

impl<T: ?Sized> Eq for Link<T> {}

impl<T: ?Sized> PartialOrd for Link<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: ?Sized> Ord for Link<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.cid.cmp(&other.cid)
    }
}

impl<T: ?Sized> Hash for Link<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.cid.hash(state);
    }
}

impl<T: ?Sized> fmt::Debug for Link<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Link<{}>({})", core::any::type_name::<T>(), self.cid)
    }
}

impl<T: ?Sized> fmt::Display for Link<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cid, f)
    }
}

impl<T: ?Sized> Serialize for Link<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.cid.serialize(s)
    }
}

impl<'de, T: ?Sized> Deserialize<'de> for Link<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Cid::deserialize(d).map(Self::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::cid::CODEC_DAG_CBOR;
    use crate::id::multihash::Multihash;

    // Phantom type tags for testing. Real modules (Node, Edge, Commit, Tree)
    // become the target types in M4+.
    struct TestNode;
    struct TestEdge;

    #[test]
    fn link_distinguishes_target_types_at_compile_time() {
        let cid = Cid::new(CODEC_DAG_CBOR, Multihash::sha2_256(b"x"));
        let node_link: Link<TestNode> = Link::new(cid.clone());
        let edge_link: Link<TestEdge> = Link::new(cid);
        // These are different types; comparing them is a compile error
        // and the commented line below demonstrates it:
        // let _ = node_link == edge_link; // <- compile error
        let node_link2 = node_link.clone();
        assert_eq!(node_link, node_link2);
        assert_eq!(edge_link.cid(), node_link2.cid());
    }

    #[test]
    fn link_round_trip_cid_equality() {
        let cid = Cid::new(CODEC_DAG_CBOR, Multihash::sha2_256(b"round"));
        let link: Link<TestNode> = Link::new(cid.clone());
        assert_eq!(link.cid(), &cid);
        let taken = link.into_cid();
        assert_eq!(taken, cid);
    }
}
