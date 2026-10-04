//! TTL cache with least-recently-used eviction (via the `lru` crate).
//!
//! The caller supplies the current time in milliseconds on every call, so the
//! cache needs no clock and behaves the same on host and wasm32 targets.

use std::hash::Hash;
use std::num::NonZeroUsize;

use lru::LruCache;

use super::CacheKey;

/// A single cached value with its insertion timestamp.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The cached string value (e.g. serialized [`ChatResponse`](crate::types::ChatResponse)).
    pub value: String,
    /// Unix-epoch timestamp in milliseconds at insertion time.
    pub inserted_at_ms: f64,
}

/// An in-memory cache whose entries expire `ttl_ms` milliseconds after they
/// were stored, optionally bounded in size.
///
/// Keys default to [`CacheKey`] (a SHA-256 of model and messages, see
/// [`cache_key`](super::cache_key)); any `Hash + Eq` type works.
///
/// # Example
/// ```rust
/// use llm_wasm::cache::{cache_key, TtlCache};
/// let mut cache = TtlCache::new(5_000.0); // 5-second TTL
/// let key = cache_key("gpt-4o", "[]");
/// cache.set(key, "value".into(), 0.0);
/// assert_eq!(cache.get(&key, 1_000.0), Some("value".to_string()));
/// assert_eq!(cache.get(&key, 6_000.0), None);
/// ```
pub struct TtlCache<K: Hash + Eq = CacheKey> {
    entries: LruCache<K, CacheEntry>,
    /// Time-to-live in milliseconds.
    pub ttl_ms: f64,
}

impl<K: Hash + Eq> std::fmt::Debug for TtlCache<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TtlCache").field("len", &self.entries.len()).field("ttl_ms", &self.ttl_ms).finish()
    }
}

impl<K: Hash + Eq> TtlCache<K> {
    /// A cache with no size limit.
    ///
    /// # Arguments
    /// * `ttl_ms`: maximum entry age in milliseconds before expiry
    pub fn new(ttl_ms: f64) -> Self {
        Self { entries: LruCache::unbounded(), ttl_ms }
    }

    /// A cache that holds at most `max_entries` values. When it is full,
    /// `set` drops the least recently used entry (`get` counts as a use).
    /// `max_entries` of 0 is treated as 1.
    pub fn with_capacity(ttl_ms: f64, max_entries: usize) -> Self {
        let cap = NonZeroUsize::new(max_entries).unwrap_or(NonZeroUsize::MIN);
        Self { entries: LruCache::new(cap), ttl_ms }
    }

    fn fresh(&self, e: &CacheEntry, now_ms: f64) -> bool {
        now_ms - e.inserted_at_ms < self.ttl_ms
    }

    /// The value for `key` if present and not expired; expired entries are
    /// removed on the way.
    pub fn get(&mut self, key: &K, now_ms: f64) -> Option<String> {
        let ttl = self.ttl_ms;
        match self.entries.get(key) {
            Some(e) if now_ms - e.inserted_at_ms < ttl => Some(e.value.clone()),
            Some(_) => {
                self.entries.pop(key);
                None
            }
            None => None,
        }
    }

    /// Insert or replace a cache entry.
    pub fn set(&mut self, key: K, value: String, now_ms: f64) {
        self.entries.put(key, CacheEntry { value, inserted_at_ms: now_ms });
    }

    /// Remove all expired entries. Returns how many were removed.
    pub fn purge_expired(&mut self, now_ms: f64) -> u32 {
        let before = self.entries.len();
        // Drain from least to most recently used and re-insert the live
        // entries in the same order, so recency is preserved.
        let mut kept = LruCache::unbounded();
        while let Some((k, e)) = self.entries.pop_lru() {
            if self.fresh(&e, now_ms) {
                kept.put(k, e);
            }
        }
        if let Some(cap) = self.entries_cap() {
            kept.resize(cap);
        }
        self.entries = kept;
        u32::try_from(before - self.entries.len()).unwrap_or(u32::MAX)
    }

    fn entries_cap(&self) -> Option<NonZeroUsize> {
        let cap = self.entries.cap();
        (cap.get() != usize::MAX).then_some(cap)
    }

    /// Number of entries held (including expired ones not yet purged).
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
        let mut cache: TtlCache<u64> = TtlCache::new(1_000.0);
        assert!(cache.get(&99, 0.0).is_none());
    }

    #[test]
    fn test_ttl_cache_set_and_get_returns_value() {
        let mut cache = TtlCache::new(1_000.0);
        cache.set(1, "hello".into(), 0.0);
        assert_eq!(cache.get(&1, 500.0), Some("hello".into()));
    }

    #[test]
    fn test_ttl_cache_expired_returns_none() {
        let mut cache = TtlCache::new(1_000.0);
        cache.set(1, "val".into(), 0.0);
        // Now at 1001ms: beyond TTL
        assert!(cache.get(&1, 1_001.0).is_none());
    }

    #[test]
    fn test_ttl_cache_len_tracks_entries() {
        let mut cache: TtlCache<u64> = TtlCache::new(5_000.0);
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
        let cache: TtlCache<u64> = TtlCache::new(1_000.0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_ttl_cache_overwrite_updates_timestamp() {
        let mut cache = TtlCache::new(1_000.0);
        cache.set(1, "old".into(), 0.0);
        cache.set(1, "new".into(), 500.0);
        // At t=1100, original insertion (0ms) would have expired but updated one (500ms) hasn't
        assert_eq!(cache.get(&1, 1_100.0), Some("new".into()));
    }

    #[test]
    fn test_capacity_evicts_oldest() {
        let mut cache = TtlCache::with_capacity(10_000.0, 2);
        cache.set(1, "a".into(), 0.0);
        cache.set(2, "b".into(), 1.0);
        cache.set(3, "c".into(), 2.0);
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&1, 3.0).is_none());
        assert_eq!(cache.get(&3, 3.0), Some("c".into()));
        // Overwriting an existing key does not evict anything.
        cache.set(2, "b2".into(), 4.0);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&3, 5.0), Some("c".into()));
    }

    #[test]
    fn test_capacity_prefers_dropping_expired() {
        let mut cache = TtlCache::with_capacity(100.0, 2);
        cache.set(1, "old".into(), 0.0);
        cache.set(2, "fresh".into(), 150.0);
        cache.set(3, "new".into(), 160.0); // key 1 expired at 100
        assert_eq!(cache.get(&2, 170.0), Some("fresh".into()));
        assert_eq!(cache.get(&3, 170.0), Some("new".into()));
    }

    #[test]
    fn test_lru_keeps_recently_read_entries() {
        // 0.2.0-dev evicted the oldest insert even if it was read constantly.
        let mut cache = TtlCache::with_capacity(10_000.0, 2);
        cache.set(1u64, "hot".into(), 0.0);
        cache.set(2, "cold".into(), 1.0);
        assert!(cache.get(&1, 2.0).is_some()); // 1 is now most recently used
        cache.set(3, "new".into(), 3.0);
        assert_eq!(cache.get(&1, 4.0), Some("hot".into()));
        assert!(cache.get(&2, 4.0).is_none());
    }

    #[test]
    fn test_purge_keeps_capacity_and_order() {
        let mut cache = TtlCache::with_capacity(100.0, 3);
        cache.set(1u64, "a".into(), 0.0);
        cache.set(2, "b".into(), 90.0);
        cache.set(3, "c".into(), 95.0);
        assert_eq!(cache.purge_expired(150.0), 1);
        cache.set(4, "d".into(), 150.0);
        cache.set(5, "e".into(), 151.0); // evicts 2, the least recently used
        assert!(cache.get(&2, 152.0).is_none());
        assert_eq!(cache.len(), 3);
    }
}
