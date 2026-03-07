//! TTL cache implementation using wall-clock timestamps in milliseconds.
//!
//! On WASM, `inserted_at_ms` would be populated from `js_sys::Date::now()`.
//! On host targets the caller supplies the timestamp directly, keeping this
//! module free of platform-specific dependencies.

use std::collections::HashMap;

/// A single cached value with its insertion timestamp.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The cached string value (e.g. serialized [`ChatResponse`]).
    pub value: String,
    /// Unix-epoch timestamp in milliseconds at insertion time.
    pub inserted_at_ms: f64,
}

/// An in-memory TTL cache keyed by `u64` FNV-1a hashes.
///
/// Entries older than `ttl_ms` milliseconds are treated as expired.
///
/// # Example
/// ```rust
/// use llm_wasm::cache::ttl::TtlCache;
/// let mut cache = TtlCache::new(5_000.0); // 5-second TTL
/// cache.set(42, "value".into(), 0.0);
/// assert_eq!(cache.get(42, 1_000.0), Some("value".to_string()));
/// ```
pub struct TtlCache {
    entries: HashMap<u64, CacheEntry>,
    /// Time-to-live in milliseconds.
    pub ttl_ms: f64,
}

impl TtlCache {
    /// Create a new cache with the given TTL.
    ///
    /// # Arguments
    /// * `ttl_ms` — maximum entry age in milliseconds before expiry
    pub fn new(ttl_ms: f64) -> Self {
        Self { entries: HashMap::new(), ttl_ms }
    }

    /// Retrieve a value if it exists and has not expired.
    ///
    /// # Arguments
    /// * `key` — FNV-1a cache key
    /// * `now_ms` — current time in milliseconds (caller-supplied for testability)
    ///
    /// # Returns
    /// `Some(value)` if found and not expired, `None` otherwise.
    pub fn get(&mut self, key: u64, now_ms: f64) -> Option<String> {
        match self.entries.get(&key) {
            Some(entry) if now_ms - entry.inserted_at_ms < self.ttl_ms => {
                Some(entry.value.clone())
            }
            Some(_) => {
                // Expired — lazy evict
                self.entries.remove(&key);
                None
            }
            None => None,
        }
    }

    /// Insert or replace a cache entry.
    ///
    /// # Arguments
    /// * `key` — FNV-1a cache key
    /// * `value` — string to cache
    /// * `now_ms` — current time in milliseconds
    pub fn set(&mut self, key: u64, value: String, now_ms: f64) {
        self.entries.insert(key, CacheEntry { value, inserted_at_ms: now_ms });
    }

    /// Remove all expired entries.
    ///
    /// # Arguments
    /// * `now_ms` — current time in milliseconds
    ///
    /// # Returns
    /// Number of entries removed.
    pub fn purge_expired(&mut self, now_ms: f64) -> u32 {
        let ttl = self.ttl_ms;
        let before = self.entries.len();
        self.entries.retain(|_, e| now_ms - e.inserted_at_ms < ttl);
        (before - self.entries.len()) as u32
    }

    /// Return the number of entries currently in the cache (including expired ones
    /// not yet purged).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ttl_cache_get_missing_returns_none() {
        let mut cache = TtlCache::new(1_000.0);
        assert!(cache.get(99, 0.0).is_none());
    }

    #[test]
    fn test_ttl_cache_set_and_get_returns_value() {
        let mut cache = TtlCache::new(1_000.0);
        cache.set(1, "hello".into(), 0.0);
        assert_eq!(cache.get(1, 500.0), Some("hello".into()));
    }

    #[test]
    fn test_ttl_cache_expired_returns_none() {
        let mut cache = TtlCache::new(1_000.0);
        cache.set(1, "val".into(), 0.0);
        // Now at 1001ms — beyond TTL
        assert!(cache.get(1, 1_001.0).is_none());
    }

    #[test]
    fn test_ttl_cache_len_tracks_entries() {
        let mut cache = TtlCache::new(5_000.0);
        assert_eq!(cache.len(), 0);
        cache.set(1, "a".into(), 0.0);
        cache.set(2, "b".into(), 0.0);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_ttl_cache_purge_expired_removes_old_entries() {
        let mut cache = TtlCache::new(1_000.0);
        cache.set(1, "a".into(), 0.0);
        cache.set(2, "b".into(), 500.0);
        // At t=1500, key 1 is expired (1500ms > 1000ms TTL), key 2 is not (1000ms == TTL, not < TTL... let's use 900ms inserted)
        cache.set(3, "c".into(), 600.0);
        let removed = cache.purge_expired(1_001.0);
        assert_eq!(removed, 1); // only key 1 expired
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_ttl_cache_is_empty_initially() {
        let cache = TtlCache::new(1_000.0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_ttl_cache_overwrite_updates_timestamp() {
        let mut cache = TtlCache::new(1_000.0);
        cache.set(1, "old".into(), 0.0);
        cache.set(1, "new".into(), 500.0);
        // At t=1100, original insertion (0ms) would have expired but updated one (500ms) hasn't
        assert_eq!(cache.get(1, 1_100.0), Some("new".into()));
    }
}
