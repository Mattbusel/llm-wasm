// Integration tests: Cache + retry pipeline scenarios.

use llm_wasm::cache::{cache_key, fnv1a_hash, TtlCache};
use llm_wasm::retry::RetryPolicy;

// ── FNV-1a hash properties ────────────────────────────────────────────────

#[test]
fn hash_empty_string_is_stable() {
    let h = fnv1a_hash("");
    assert_eq!(h, fnv1a_hash(""));
}

#[test]
fn hash_single_char_differs_from_empty() {
    assert_ne!(fnv1a_hash(""), fnv1a_hash("a"));
}

#[test]
fn hash_case_sensitive() {
    assert_ne!(fnv1a_hash("Hello"), fnv1a_hash("hello"));
}

#[test]
fn hash_long_string_stable() {
    let s = "a".repeat(1000);
    assert_eq!(fnv1a_hash(&s), fnv1a_hash(&s));
}

#[test]
fn hash_different_strings_likely_differ() {
    let hashes: std::collections::HashSet<u64> = (0..20)
        .map(|i| fnv1a_hash(&format!("prompt_{}", i)))
        .collect();
    assert_eq!(hashes.len(), 20, "all 20 hashes should be distinct");
}

#[test]
fn cache_key_model_separator_matters() {
    let k1 = cache_key("gpt-4o", "msg");
    let k2 = cache_key("gpt-4o-mini", "msg");
    assert_ne!(k1, k2);
}

#[test]
fn cache_key_symmetric_variation() {
    // Swapping model and messages produces different key
    let k1 = cache_key("model", "messages");
    let k2 = cache_key("messages", "model");
    assert_ne!(k1, k2);
}

#[test]
fn cache_key_deterministic_across_calls() {
    for _ in 0..5 {
        let k = cache_key("claude-sonnet-4-6", "[{\"role\":\"user\",\"content\":\"hi\"}]");
        assert_eq!(k, cache_key("claude-sonnet-4-6", "[{\"role\":\"user\",\"content\":\"hi\"}]"));
    }
}

// ── TtlCache ─────────────────────────────────────────────────────────────

#[test]
fn ttl_cache_basic_set_get() {
    let mut cache = TtlCache::new(1000.0);
    cache.set(1, "value".into(), 0.0);
    assert_eq!(cache.get(1, 500.0), Some("value".into()));
}

#[test]
fn ttl_cache_miss_returns_none() {
    let mut cache = TtlCache::new(1000.0);
    assert!(cache.get(99, 0.0).is_none());
}

#[test]
fn ttl_cache_exact_ttl_boundary_is_expired() {
    let mut cache = TtlCache::new(1000.0);
    cache.set(1, "v".into(), 0.0);
    // At exactly ttl_ms the condition is `now - inserted < ttl`, so 1000 < 1000 = false → expired
    assert!(cache.get(1, 1000.0).is_none());
}

#[test]
fn ttl_cache_just_under_ttl_is_fresh() {
    let mut cache = TtlCache::new(1000.0);
    cache.set(1, "v".into(), 0.0);
    assert_eq!(cache.get(1, 999.0), Some("v".into()));
}

#[test]
fn ttl_cache_expired_removes_entry() {
    let mut cache = TtlCache::new(500.0);
    cache.set(1, "v".into(), 0.0);
    cache.get(1, 600.0); // triggers lazy eviction
    assert_eq!(cache.len(), 0);
}

#[test]
fn ttl_cache_multiple_entries_independent_ttls() {
    let mut cache = TtlCache::new(1000.0);
    cache.set(1, "early".into(), 0.0);
    cache.set(2, "late".into(), 800.0);
    // At t=1100: key 1 expired, key 2 still fresh (1100-800=300 < 1000)
    assert!(cache.get(1, 1100.0).is_none());
    assert_eq!(cache.get(2, 1100.0), Some("late".into()));
}

#[test]
fn ttl_cache_overwrite_refreshes_timestamp() {
    let mut cache = TtlCache::new(500.0);
    cache.set(1, "old".into(), 0.0);
    cache.set(1, "new".into(), 400.0); // re-insert at t=400
    // At t=600: 600-400=200 < 500 → fresh
    assert_eq!(cache.get(1, 600.0), Some("new".into()));
    // At t=901: 901-400=501 > 500 → expired
    assert!(cache.get(1, 901.0).is_none());
}

#[test]
fn ttl_cache_purge_expired_removes_correct_count() {
    let mut cache = TtlCache::new(500.0);
    cache.set(1, "a".into(), 0.0);  // expires at 500
    cache.set(2, "b".into(), 0.0);  // expires at 500
    cache.set(3, "c".into(), 400.0); // expires at 900
    let removed = cache.purge_expired(600.0);
    assert_eq!(removed, 2);
    assert_eq!(cache.len(), 1);
}

#[test]
fn ttl_cache_purge_on_empty_is_zero() {
    let mut cache = TtlCache::new(1000.0);
    assert_eq!(cache.purge_expired(99999.0), 0);
}

#[test]
fn ttl_cache_is_empty_after_purge_all() {
    let mut cache = TtlCache::new(100.0);
    cache.set(1, "v".into(), 0.0);
    cache.purge_expired(200.0);
    assert!(cache.is_empty());
}

// ── RetryPolicy ───────────────────────────────────────────────────────────

#[test]
fn retry_policy_custom_params() {
    let p = RetryPolicy::new(5, 200, 10_000).unwrap();
    assert_eq!(p.max_attempts(), 5);
}

#[test]
fn retry_policy_delay_at_attempt_1_is_base() {
    let p = RetryPolicy::new(5, 100, 10_000).unwrap();
    // Attempt 1: base * 2^0 = 100, jitter = 75+1%50 = 76, 100*76/100 = 76
    let d = p.delay_for_attempt(1);
    assert!(d <= 100, "delay at attempt 1 should be <= base: {}", d);
    assert!(d > 0);
}

#[test]
fn retry_policy_delay_caps_at_max() {
    let p = RetryPolicy::new(10, 100, 500).unwrap();
    for attempt in 1..=10 {
        assert!(p.delay_for_attempt(attempt) <= 500);
    }
}

#[test]
fn retry_policy_all_retryable_codes() {
    let p = RetryPolicy::exponential();
    for code in [429u16, 500, 502, 503, 504] {
        assert!(p.should_retry(1, code), "code {} should be retryable", code);
    }
}

#[test]
fn retry_policy_non_retryable_codes() {
    let p = RetryPolicy::exponential();
    for code in [200u16, 201, 400, 401, 403, 404, 422] {
        assert!(!p.should_retry(1, code), "code {} should not retry", code);
    }
}

#[test]
fn retry_policy_exhausted_after_max_attempts() {
    let p = RetryPolicy::new(3, 10, 1000).unwrap();
    assert!(p.should_retry(1, 500));
    assert!(p.should_retry(2, 500));
    assert!(!p.should_retry(3, 500)); // attempt 3 = max_attempts, no more
}

#[test]
fn retry_policy_invalid_zero_attempts() {
    assert!(RetryPolicy::new(0, 100, 1000).is_err());
}

#[test]
fn retry_policy_base_exceeds_max_is_invalid() {
    assert!(RetryPolicy::new(3, 2000, 1000).is_err());
}

#[test]
fn retry_policy_base_equals_max_is_valid() {
    assert!(RetryPolicy::new(3, 1000, 1000).is_ok());
}

// ── Cache + Retry combined scenario ──────────────────────────────────────

#[test]
fn cached_response_skips_retry_simulation() {
    let mut cache = TtlCache::new(5000.0);
    let policy = RetryPolicy::exponential();
    let key = cache_key("gpt-4o", "[{\"role\":\"user\",\"content\":\"hello\"}]");

    // Simulate: first request misses cache, retry on 503
    assert!(cache.get(key, 0.0).is_none());
    assert!(policy.should_retry(1, 503));
    assert!(policy.should_retry(2, 503));
    assert!(!policy.should_retry(3, 503)); // exhausted

    // Success on 3rd attempt — store in cache
    cache.set(key, "{\"content\":\"Hi there!\"}".into(), 1000.0);

    // Second request hits cache
    let cached = cache.get(key, 2000.0);
    assert!(cached.is_some());
    assert!(cached.unwrap().contains("Hi there"));
}

#[test]
fn cache_stores_multiple_model_responses() {
    let mut cache = TtlCache::new(10_000.0);
    let models = ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-6"];
    let msg = "[{\"role\":\"user\",\"content\":\"what is 2+2?\"}]";

    for (i, model) in models.iter().enumerate() {
        let key = cache_key(model, msg);
        cache.set(key, format!("response_{}", i), 0.0);
    }

    for (i, model) in models.iter().enumerate() {
        let key = cache_key(model, msg);
        let v = cache.get(key, 5000.0).unwrap();
        assert_eq!(v, format!("response_{}", i));
    }
}
