//! CAR v1 reader + writer.
//!
//! Wire shape (from <https://ipld.io/specs/transport/car/carv1/>):
//!
//! ```text
//! | varint(len) | DAG-CBOR header | varint(len) | CID | bytes | varint(len) | CID | bytes | ...
//! ```
//!
//! - The **header** is a DAG-CBOR map with two fields: `version: 1`
//!   and `roots: [Cid]`. CIDs in the roots list are in their usual
//!   IPLD-tagged form (tag 42).
//! - Each **block section** is `varint(len) | cid_bytes | payload_bytes`
//!   where `len` is the byte count of `cid_bytes + payload_bytes`
//!   combined, and `cid_bytes` is the *binary* `CIDv1` form (not the
//!   multibase string). A `CIDv0` (legacy, codec-less) is also permitted
//!   by the spec but not emitted by this writer.
//!
//! This module stays at the byte level: it produces / consumes CARs,
//! but knows nothing about which blocks belong in a particular DAG.
//! The [`mod@super::export`] / [`mod@super::import`] modules do the walking
//! and the blockstore wiring.

use std::collections::BTreeMap;
use std::io::{Read, Write};

use ipld_core::ipld::Ipld;
use mnem_core::codec::{from_canonical_bytes, to_canonical_bytes};
use mnem_core::id::Cid;

use crate::error::TransportError;

/// Maximum header length we will accept on import.
///
/// A well-formed `CARv1` header is ~50 bytes plus `36 * n_roots`. Cap at
/// 8 MiB so a malicious sender cannot force us to allocate the
/// universe just by sending a varint that says "next, four gigabytes".
const MAX_HEADER_BYTES: usize = 8 * 1024 * 1024;

/// Maximum single-block payload we will accept on import.
///
/// Real mnem blocks are well under 4 MiB (Prolly chunks target ~4 KiB;
/// embeddings are the largest single blocks at 3-6 KiB for typical
/// vectors). Cap at 32 MiB: generous enough to never bite a legitimate
/// sender, small enough to bound peak memory.
const MAX_BLOCK_BYTES: usize = 32 * 1024 * 1024;

// ---------------- varint helpers ----------------
//
// We use `unsigned-varint`'s buffer-based `encode::u64` for writing
// and roll our own byte-at-a-time reader for decoding so the `io`
// feature (which otherwise pulls in dependencies we don't need) isn't
// required. Both match the LEB128-style encoding the CAR spec defers
// to via multiformats.

fn write_varint_u64<W: Write + ?Sized>(w: &mut W, n: u64) -> std::io::Result<()> {
    let mut buf = unsigned_varint::encode::u64_buffer();
    let slice = unsigned_varint::encode::u64(n, &mut buf);
    w.write_all(slice)
}

fn read_varint_u64<R: Read + ?Sized>(r: &mut R) -> Result<u64, TransportError> {
    // At most 10 bytes for u64 under LEB128. We stop as soon as a
    // byte with MSB clear is seen.
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    for i in 0..10usize {
        let mut b = [0u8; 1];
        let n = r.read(&mut b)?;
        if n == 0 {
            return Err(TransportError::Car(format!("truncated varint at byte {i}")));
        }
        let byte = b[0];
        let val = u64::from(byte & 0x7F);
        // Shift-left before OR; an overflow here means the encoded
        // number is larger than u64.
        let Some(shifted) = val.checked_shl(shift) else {
            return Err(TransportError::Car("varint overflow".into()));
        };
        result |= shifted;
        if byte & 0x80 == 0 {
            // Reject non-minimal encodings: any trailing 0x00 byte (a
            // varint "continue with zero") is disallowed by the
            // multiformats spec.
            if i > 0 && byte == 0 {
                return Err(TransportError::Car("non-minimal varint".into()));
            }
            return Ok(result);
        }
        shift += 7;
    }
    Err(TransportError::Car("varint too long".into()))
}

// ---------------- header ----------------

/// Canonical in-memory CAR header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CarHeader {
    /// Always `1` for `CARv1`. Other versions are rejected on read.
    pub version: u64,
    /// Root CIDs the CAR declares. The export writer always produces a
    /// single root; readers accept zero or more.
    pub roots: Vec<Cid>,
}

impl CarHeader {
    /// Construct a single-root `CARv1` header.
    #[must_use]
    pub fn single_root(root: Cid) -> Self {
        Self {
            version: 1,
            roots: vec![root],
        }
    }

    /// Encode the header to canonical DAG-CBOR. The caller is
    /// responsible for the outer varint length prefix (written by
    /// [`write_header`]).
    fn to_dagcbor(&self) -> Result<Vec<u8>, TransportError> {
        let mut map = BTreeMap::new();
        map.insert(
            "version".to_string(),
            Ipld::Integer(i128::from(self.version)),
        );
        let root_ipld: Vec<Ipld> = self
            .roots
            .iter()
            .map(|c| {
                // Round-trip through binary form to translate
                // `mnem_core::id::Cid` into the `ipld_core::cid::Cid`
                // that `Ipld::Link` requires.
                let inner = ipld_core::cid::Cid::try_from(c.to_bytes().as_slice())
                    .expect("own cid round-trips");
                Ipld::Link(inner)
            })
            .collect();
        map.insert("roots".to_string(), Ipld::List(root_ipld));
        let bytes = to_canonical_bytes(&Ipld::Map(map))?;
        Ok(bytes.to_vec())
    }

    /// Decode a header from its DAG-CBOR body bytes (without the
    /// surrounding varint length).
    fn from_dagcbor(bytes: &[u8]) -> Result<Self, TransportError> {
        let value: Ipld = from_canonical_bytes(bytes)?;
        let Ipld::Map(map) = value else {
            return Err(TransportError::Car("header is not a map".into()));
        };

        let version = match map.get("version") {
            Some(Ipld::Integer(n)) => u64::try_from(*n)
                .map_err(|_| TransportError::Car(format!("header version out of range: {n}")))?,
            _ => return Err(TransportError::Car("missing 'version' field".into())),
        };
        if version != 1 {
            return Err(TransportError::Car(format!(
                "unsupported CAR version: {version}"
            )));
        }

        let roots = match map.get("roots") {
            Some(Ipld::List(xs)) => {
                let mut out = Vec::with_capacity(xs.len());
                for x in xs {
                    match x {
                        Ipld::Link(c) => {
                            let ours = Cid::from_bytes(&c.to_bytes())
                                .map_err(|e| TransportError::Car(format!("root: {e}")))?;
                            out.push(ours);
                        }
                        _ => return Err(TransportError::Car("non-link in roots".into())),
                    }
                }
                out
            }
            _ => return Err(TransportError::Car("missing 'roots' list".into())),
        };

        Ok(Self { version, roots })
    }
}

// ---------------- writer ----------------

/// Write a CAR v1 header: `varint(len) || DAG-CBOR(header)`.
///
/// # Errors
///
/// [`TransportError::Io`] on write failure, [`TransportError::Codec`]
/// if header encoding fails (should not happen in practice).
pub fn write_header<W: Write + ?Sized>(
    w: &mut W,
    header: &CarHeader,
) -> Result<(), TransportError> {
    let body = header.to_dagcbor()?;
    write_varint_u64(w, usize_to_u64(body.len()))?;
    w.write_all(&body)?;
    Ok(())
}

/// Widen a `usize` to `u64`. On every platform mnem targets (32- and
/// 64-bit), this is lossless; we open-code the cast in one spot so
/// the clippy annotation is centralised instead of sprinkled across
/// every call-site.
#[inline]
#[allow(clippy::cast_lossless, clippy::cast_possible_truncation)]
pub(crate) const fn usize_to_u64(n: usize) -> u64 {
    n as u64
}

/// Write a single block section: `varint(cid_len + data_len) || cid || data`.
///
/// # Errors
///
/// [`TransportError::Io`] on write failure.
pub fn write_block<W: Write + ?Sized>(
    w: &mut W,
    cid: &Cid,
    data: &[u8],
) -> Result<(), TransportError> {
    let cid_bytes = cid.to_bytes();
    let total = cid_bytes.len() + data.len();
    write_varint_u64(w, usize_to_u64(total))?;
    w.write_all(&cid_bytes)?;
    w.write_all(data)?;
    Ok(())
}

// ---------------- reader ----------------

/// Read and validate a CAR v1 header from a byte stream.
///
/// # Errors
///
/// [`TransportError::Car`] on malformed / oversized header,
/// [`TransportError::Io`] on read failure.
pub fn read_header<R: Read + ?Sized>(r: &mut R) -> Result<CarHeader, TransportError> {
    let len = read_varint_u64(r)?;
    let len_usize: usize = len
        .try_into()
        .map_err(|_| TransportError::Car("header length exceeds usize".into()))?;
    if len_usize > MAX_HEADER_BYTES {
        return Err(TransportError::Car(format!(
            "header too large: {len_usize} > {MAX_HEADER_BYTES}"
        )));
    }
    let mut buf = vec![0u8; len_usize];
    r.read_exact(&mut buf)?;
    CarHeader::from_dagcbor(&buf)
}

/// A single parsed block: `(cid, payload_bytes)`.
pub type CarBlock = (Cid, Vec<u8>);

/// Streaming reader for CAR block sections.
///
/// Call [`CarBlockReader::next_block`] repeatedly. Returns
/// `Ok(None)` on clean EOF between block sections, `Ok(Some(..))` on
/// a decoded block, `Err(..)` on a malformed or truncated block.
pub struct CarBlockReader<R: Read + ?Sized> {
    r: R,
}

impl<R: Read> CarBlockReader<R> {
    /// Wrap a reader. The caller should have already consumed the
    /// CAR header with [`read_header`]; this reader starts at the
    /// first block section.
    pub const fn new(r: R) -> Self {
        Self { r }
    }
}

impl<R: Read + ?Sized> CarBlockReader<R> {
    /// Read the next block section from the stream. Returns `Ok(None)`
    /// when EOF is cleanly reached at a block boundary.
    ///
    /// # Errors
    ///
    /// - [`TransportError::Car`] if the section is malformed (bad
    ///   varint, oversized, truncated mid-CID).
    /// - [`TransportError::Io`] on read failure.
    pub fn next_block(&mut self) -> Result<Option<CarBlock>, TransportError> {
        // Peek one byte to distinguish clean EOF from a truncated
        // varint in the middle of a section.
        let mut first = [0u8; 1];
        let n = self.r.read(&mut first)?;
        if n == 0 {
            return Ok(None);
        }

        // Finish reading the varint, using the byte we just peeked as
        // the first continuation byte.
        let mut len: u64 = 0;
        let mut shift: u32 = 0;
        let mut byte = first[0];
        let mut i: usize = 0;
        loop {
            let Some(shifted) = u64::from(byte & 0x7F).checked_shl(shift) else {
                return Err(TransportError::Car("block-length varint overflow".into()));
            };
            len |= shifted;
            if byte & 0x80 == 0 {
                if i > 0 && byte == 0 {
                    return Err(TransportError::Car(
                        "non-minimal block-length varint".into(),
                    ));
                }
                break;
            }
            shift += 7;
            if shift >= 64 {
                return Err(TransportError::Car("block-length varint too long".into()));
            }
            let mut next = [0u8; 1];
            let m = self.r.read(&mut next)?;
            if m == 0 {
                return Err(TransportError::Car("truncated block-length varint".into()));
            }
            byte = next[0];
            i += 1;
        }

        let len_usize: usize = len
            .try_into()
            .map_err(|_| TransportError::Car("block length exceeds usize".into()))?;
        if len_usize > MAX_BLOCK_BYTES {
            return Err(TransportError::Car(format!(
                "block too large: {len_usize} > {MAX_BLOCK_BYTES}"
            )));
        }
        if len_usize == 0 {
            return Err(TransportError::Car("empty block section".into()));
        }

        let mut buf = vec![0u8; len_usize];
        self.r.read_exact(&mut buf)?;

        // Parse the CID off the front. `cid::CidGeneric::read_bytes`
        // is exactly what we want, but our `Cid::from_bytes` wraps it
        // via the `TryFrom<&[u8]>` impl - which consumes only the CID
        // prefix, returning an error on trailing bytes. Instead, walk
        // the multihash manually.
        let cid_len = cid_binary_length(&buf)
            .ok_or_else(|| TransportError::Car("cannot determine CID length".into()))?;
        if cid_len > buf.len() {
            return Err(TransportError::Car("cid overruns block length".into()));
        }
        let (cid_bytes, data) = buf.split_at(cid_len);
        let cid =
            Cid::from_bytes(cid_bytes).map_err(|e| TransportError::Car(format!("cid: {e}")))?;
        Ok(Some((cid, data.to_vec())))
    }
}

/// Compute the binary length of a `CIDv1` prefix embedded at the start
/// of `buf`.
///
/// `CIDv1` wire shape: `varint(version) | varint(codec) | multihash`
/// where multihash = `varint(code) | varint(digest_len) | digest[digest_len]`.
/// We decode the four varints and sum their byte lengths with the
/// digest size.
fn cid_binary_length(buf: &[u8]) -> Option<usize> {
    let mut cursor = 0usize;
    // version
    let (version, n) = decode_inline_varint(buf.get(cursor..)?)?;
    cursor += n;
    if version != 1 {
        // CIDv0 has no version / codec prefix (starts with multihash
        // code 0x12). We don't emit them and won't accept them on
        // import, so signal "bad shape" to the caller instead of
        // trying to parse.
        return None;
    }
    // codec
    let (_codec, n) = decode_inline_varint(buf.get(cursor..)?)?;
    cursor += n;
    // multihash code
    let (_mh_code, n) = decode_inline_varint(buf.get(cursor..)?)?;
    cursor += n;
    // digest size
    let (digest_size, n) = decode_inline_varint(buf.get(cursor..)?)?;
    cursor += n;
    let digest_size_usize: usize = digest_size.try_into().ok()?;
    cursor = cursor.checked_add(digest_size_usize)?;
    Some(cursor)
}

/// Minimal inline varint decoder: returns `(value, bytes_read)` or
/// `None` on truncation / overflow. Used only by
/// [`cid_binary_length`] where we already hold the full block payload
/// in a slice.
fn decode_inline_varint(buf: &[u8]) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    for (i, &byte) in buf.iter().enumerate().take(10) {
        let val = u64::from(byte & 0x7F);
        let shifted = val.checked_shl(shift)?;
        result |= shifted;
        if byte & 0x80 == 0 {
            return Some((result, i + 1));
        }
        shift += 7;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnem_core::id::{CODEC_DAG_CBOR, Multihash};

    fn sample_cid(seed: u8) -> Cid {
        Cid::new(CODEC_DAG_CBOR, Multihash::sha2_256(&[seed]))
    }

    #[test]
    fn varint_round_trip_small() {
        let mut buf = Vec::new();
        write_varint_u64(&mut buf, 0).unwrap();
        write_varint_u64(&mut buf, 1).unwrap();
        write_varint_u64(&mut buf, 127).unwrap();
        write_varint_u64(&mut buf, 128).unwrap();
        write_varint_u64(&mut buf, 1_000_000).unwrap();

        let mut cursor: &[u8] = &buf;
        assert_eq!(read_varint_u64(&mut cursor).unwrap(), 0);
        assert_eq!(read_varint_u64(&mut cursor).unwrap(), 1);
        assert_eq!(read_varint_u64(&mut cursor).unwrap(), 127);
        assert_eq!(read_varint_u64(&mut cursor).unwrap(), 128);
        assert_eq!(read_varint_u64(&mut cursor).unwrap(), 1_000_000);
    }

    #[test]
    fn header_round_trip_single_root() {
        let header = CarHeader::single_root(sample_cid(42));
        let mut buf = Vec::new();
        write_header(&mut buf, &header).unwrap();

        let mut cursor: &[u8] = &buf;
        let decoded = read_header(&mut cursor).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn block_section_round_trip() {
        let cid = sample_cid(7);
        let payload = b"hello, car".to_vec();
        let mut buf = Vec::new();
        write_block(&mut buf, &cid, &payload).unwrap();

        let mut reader = CarBlockReader::new(&buf[..]);
        let (got_cid, got_bytes) = reader.next_block().unwrap().expect("one block");
        assert_eq!(got_cid, cid);
        assert_eq!(got_bytes, payload);
        assert!(
            reader.next_block().unwrap().is_none(),
            "clean EOF after one"
        );
    }

    #[test]
    fn truncated_block_is_car_error() {
        let cid = sample_cid(11);
        let payload = b"truncate-me".to_vec();
        let mut buf = Vec::new();
        write_block(&mut buf, &cid, &payload).unwrap();
        // Drop the last few bytes of the payload.
        buf.truncate(buf.len() - 3);

        let mut reader = CarBlockReader::new(&buf[..]);
        let err = reader.next_block().unwrap_err();
        match err {
            TransportError::Io(_) | TransportError::Car(_) => {}
            other => panic!("expected Io/Car error, got {other:?}"),
        }
    }
}
