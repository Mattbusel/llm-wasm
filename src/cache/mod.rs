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
/// * `data` — arbitrary string to hash
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

/// Compute a cache key from a model identifier and a JSON-serialized messages string.
///
/// # Arguments
/// * `model` — model name string
/// * `messages_json` — JSON representation of the messages array
///
/// # Returns
/// A `u64` cache key.
///
/// # Panics
/// This function never panics.
pub fn cache_key(model: &str, messages_json: &str) -> u64 {
    let combined = format!("{model}::{messages_json}");
    fnv1a_hash(&combined)
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
}
