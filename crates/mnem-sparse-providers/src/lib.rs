//! # mnem-sparse-providers
//!
//! Learned-sparse encoder adapters for mnem. Implements the
//! [`mnem_core::sparse::SparseEncoder`] trait for the three shipping
//! backends :
//!
//! - **Sidecar** (always available): HTTP POST to a local Python
//!   service running the reference SPLADE / BGE-M3 implementation.
//!   Lightest install; the sidecar handles weights + tokenization.
//! - **ONNX** (feature `onnx`): in-process inference via `ort` +
//!   `tokenizers`. Pulls an ~80MB onnxruntime binary and a ~70MB
//!   quantized model weights file. Fastest but heaviest dep.
//! - **Mock** (always available, re-exports
//!   [`mnem_core::sparse::MockSparseEncoder`]): deterministic
//!   length-inverse hash, for tests and dry-run benchmarks.
//!
//! ## Why ONNX is feature-gated
//!
//! `ort` wraps a C++ shared library; `tokenizers` carries internal
//! `unsafe` for its SIMD fast paths. Both compile fine on native
//! but do NOT compile to wasm32. Keeping them optional means a
//! wasm-target build of mnem + this crate stays clean; and users who
//! don't want the onnxruntime dep skip it entirely.
//!
//! ## Why sidecar is the default
//!
//! Most real users will run SPLADE inside their existing Python ML
//! infra. The sidecar transport keeps mnem-sparse-providers a
//! single binary with one HTTP dep (ureq + rustls), deferring the
//! heavy lifting to whatever the user already has running.
//!
//! ## Invariants
//!
//! - **No tokio.** All adapters are sync. Matches every other mnem
//!   provider crate.
//! - **No API keys on disk.** Sidecar URLs are configurable; auth
//!   (when needed) comes from env vars.
//! - **`mnem-core` stays zero-network.** This crate carries the
//!   HTTP + optional ML runtime deps alone.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod config;
pub mod sidecar;

#[cfg(feature = "onnx")]
pub mod onnx;

// Re-export the mock so callers get a one-line swap for adapter
// tests without pulling mnem-core directly.
pub use mnem_core::sparse::MockSparseEncoder;

pub use config::{ProviderConfig, SidecarConfig, open};
