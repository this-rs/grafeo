//! C FFI bindings for the Obrain graph database.
//!
//! This crate exposes a C-compatible API that can be consumed by any language
//! with C interop support (Go via CGO, Ruby via FFI, Java via JNI, etc.).
//!
//! # Memory Management
//!
//! All opaque pointers returned by `obrain_*` functions must be freed by their
//! corresponding `obrain_free_*` function. Strings returned by the API are owned
//! by the Rust side and must be freed with [`obrain_free_string`].
//!
//! # Error Handling
//!
//! Functions return [`ObrainStatus`] codes. On error, call [`obrain_last_error`]
//! to retrieve a human-readable error message. The error is thread-local and
//! valid until the next FFI call on the same thread.
//!
//! # Thread Safety
//!
//! A [`ObrainDatabase`] handle can be shared across threads. Internally it uses
//! `Arc<RwLock<ObrainDB>>` for safe concurrent access.

#![allow(unsafe_code)]

mod database;
mod error;
mod types;
