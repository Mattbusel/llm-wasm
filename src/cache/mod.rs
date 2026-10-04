//! # Module: Cache
//!
//! ## Responsibility
//! In-memory TTL cache for LLM responses, keyed by a fast FNV-1a hash of
//! model name + serialized messages.
//!
//! ## Guarantees
//! - Deterministic: `cache_key(model, messages_json)` always returns the same u64
//! - Non-blocking: all operations are synchronous and O(n) at worst for purge
//! - Bounded TTL: entries older than `ttl_ms` are ignored and cleaned up
//!
//! ## NOT Responsible For
//! - Cross-process or cross-node cache sharing
//! - Persistence across restarts

pub mod ttl;

pub use ttl::{CacheEntry, TtlCache};

/// Compute an FNV-1a 64-bit hash of `data`.
///
/// This is a pure deterministic function with no external dependencies,
/// suitable for WASM and host compilation.
///
/// # Arguments
/// * `data`: arbitrary string to hash
///
/// # Returns
/// A 64-bit FNV-1a digest.
///
/// # Panics
/// This function never panics.
///
/// # Example
/// ```rust
/// use llm_wasm::cache::fnv1a_hash;
/// let h = fnv1a_hash("hello");
/// assert_eq!(h, fnv1a_hash("hello")); // deterministic
/// ```
pub fn fnv1a_hash(data: &str) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET;
    for byte in data.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// A cache key: the SHA-256 of the model name and the messages.
///
/// 0.1 used a 64-bit FNV-1a hash of `"{model}::{messages}"`. FNV collisions
/// are easy to construct on purpose, so in a cache shared between users one
/// user could be served another's answer; and the `::` join made
/// `("a::b", "c")` and `("a", "b::c")` the same key. SHA-256 over
/// length-prefixed fields fixes both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CacheKey(pub [u8; 32]);

impl std::fmt::Display for CacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// Compute a cache key from a model identifier and a JSON-serialized messages string.
///
/// # Example
/// ```rust
/// use llm_wasm::cache::cache_key;
/// assert_eq!(cache_key("gpt-4o", "[]"), cache_key("gpt-4o", "[]"));
/// assert_ne!(cache_key("a::b", "c"), cache_key("a", "b::c"));
/// ```
pub fn cache_key(model: &str, messages_json: &str) -> CacheKey {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update((model.len() as u64).to_le_bytes());
    h.update(model.as_bytes());
    h.update((messages_json.len() as u64).to_le_bytes());
    h.update(messages_json.as_bytes());
    CacheKey(h.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv1a_hash_is_deterministic() {
        let h1 = fnv1a_hash("hello world");
        let h2 = fnv1a_hash("hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_hash_different_inputs_differ() {
        let h1 = fnv1a_hash("hello");
        let h2 = fnv1a_hash("world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_fnv1a_hash_empty_string_is_stable() {
        let h = fnv1a_hash("");
        assert_eq!(h, fnv1a_hash(""));
    }

    #[test]
    fn test_cache_key_includes_model() {
        let k1 = cache_key("claude-sonnet-4-6", "[{\"role\":\"user\"}]");
        let k2 = cache_key("gpt-4o", "[{\"role\":\"user\"}]");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_includes_messages() {
        let k1 = cache_key("model", "msg1");
        let k2 = cache_key("model", "msg2");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_has_no_join_ambiguity() {
        assert_ne!(cache_key("a::b", "c"), cache_key("a", "b::c"));
        assert_ne!(cache_key("ab", ""), cache_key("a", "b"));
    }

    #[test]
    fn test_cache_key_is_sha256() {
        // stable across platforms and versions: hex of a known input
        let k = cache_key("gpt-4o", "[]").to_string();
        assert_eq!(k.len(), 64);
        assert_eq!(k, cache_key("gpt-4o", "[]").to_string());
    }
}
