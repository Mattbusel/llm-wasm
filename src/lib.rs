//! # llm-wasm
//!
//! LLM primitives designed to compile to `wasm32-unknown-unknown` and the host
//! target alike.
//!
//! All modules are pure Rust with no platform-specific dependencies. WASM
//! bindings (wasm-bindgen, js-sys) are gated behind `#[cfg(target_arch = "wasm32")]`
//! so that `cargo test -p llm-wasm` works on any host without a WASM toolchain.
//!
//! ## Modules
//! - [`cache`] — FNV-1a keyed TTL response cache
//! - [`cost`] — static pricing table and USD budget ledger
//! - [`error`] — unified error type
//! - [`format`] — JSON extraction and Markdown fence stripping
//! - [`guard`] — composable request guards (content, length)
//! - [`retry`] — exponential-backoff retry policy
//! - [`routing`] — rule-based model routing
//! - [`template`] — `{{variable}}` / `{{>partial}}` template engine
//! - [`types`] — shared data types (`ChatMessage`, `ChatRequest`, etc.)

pub mod cache;
pub mod cost;
pub mod error;
pub mod format;
pub mod guard;
pub mod retry;
pub mod routing;
pub mod template;
pub mod types;

pub use error::LlmWasmError;
