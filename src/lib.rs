#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]

pub mod cache;
pub mod cost;
pub mod error;
pub mod format;
pub mod guard;
#[cfg(feature = "js")]
#[cfg_attr(docsrs, doc(cfg(feature = "js")))]
pub mod js;
pub mod retry;
pub mod routing;
#[cfg(feature = "secrets")]
#[cfg_attr(docsrs, doc(cfg(feature = "secrets")))]
pub mod secrets;
#[cfg(feature = "stream")]
#[cfg_attr(docsrs, doc(cfg(feature = "stream")))]
pub mod stream;
pub mod template;
pub mod types;

pub use error::LlmWasmError;
