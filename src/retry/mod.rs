//! # Module: Retry
//!
//! ## Responsibility
//! Compute exponential-backoff retry delays with jitter and evaluate whether a
//! given HTTP status code is retryable.
//!
//! ## Guarantees
//! - `delay_for_attempt` never returns a value exceeding `max_delay_ms`
//! - Retry is only recommended for transient HTTP status codes
//! - No async I/O: purely computational

use crate::error::LlmWasmError;

/// HTTP status codes on which retries should be attempted.
const RETRYABLE_STATUS_CODES: &[u16] = &[429, 500, 502, 503, 504];

/// Policy controlling retry behaviour for LLM requests.
///
/// Delays follow truncated exponential backoff with uniform jitter:
/// `delay = min(base_delay_ms * 2^(attempt-1), max_delay_ms) * (0.75 + jitter * 0.5)`
///
/// # Example
/// ```rust
/// use llm_wasm::retry::RetryPolicy;
/// let policy = RetryPolicy::exponential();
/// assert!(policy.should_retry(1, 429));
/// assert!(!policy.should_retry(1, 400));
/// ```
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    max_attempts: u32,
    base_delay_ms: u32,
    max_delay_ms: u32,
}

impl RetryPolicy {
    /// Create a custom retry policy.
    ///
    /// # Arguments
    /// * `max_attempts`: maximum number of attempts (including the first)
    /// * `base_delay_ms`: base delay in milliseconds for attempt 1
    /// * `max_delay_ms`: hard ceiling on any single delay
    ///
    /// # Errors
    /// Returns [`LlmWasmError::InvalidConfig`] if `max_attempts` is 0 or
    /// `base_delay_ms` exceeds `max_delay_ms`.
    pub fn new(
        max_attempts: u32,
        base_delay_ms: u32,
        max_delay_ms: u32,
    ) -> Result<Self, LlmWasmError> {
        if max_attempts == 0 {
            return Err(LlmWasmError::InvalidConfig {
                field: "max_attempts".into(),
                reason: "must be at least 1".into(),
            });
        }
        if base_delay_ms > max_delay_ms {
            return Err(LlmWasmError::InvalidConfig {
                field: "base_delay_ms".into(),
                reason: "must not exceed max_delay_ms".into(),
            });
        }
        Ok(Self { max_attempts, base_delay_ms, max_delay_ms })
    }

    /// Sensible default: 3 attempts, 100 ms base, 5 000 ms max.
    pub fn exponential() -> Self {
        Self { max_attempts: 3, base_delay_ms: 100, max_delay_ms: 5_000 }
    }

    /// Compute the delay in milliseconds for the given attempt number (1-indexed).
    ///
    /// The result is the capped exponential delay scaled by a fixed factor
    /// between 0.76 and 1.24 that depends only on the attempt number. That is
    /// deterministic, so every client computes the same delay: use
    /// [`delay_with_jitter`](Self::delay_with_jitter) with a random number
    /// when many clients may retry at once.
    ///
    /// # Arguments
    /// * `attempt`: 1-indexed attempt number
    ///
    /// # Returns
    /// Delay in milliseconds, capped at `max_delay_ms`.
    pub fn delay_for_attempt(&self, attempt: u32) -> u32 {
        let exp = attempt.saturating_sub(1);
        let base = self.base_delay_ms as u64;
        let max = self.max_delay_ms as u64;
        // Truncated exponential: base * 2^(attempt-1), capped at max
        let raw = base.saturating_mul(1u64 << exp.min(30));
        let truncated = raw.min(max);
        // Deterministic jitter in [0.75, 1.25): multiply by (75 + attempt % 50) / 100
        // This avoids any randomness dependency while staying within bounds.
        let jitter_num = 75u64 + u64::from(attempt % 50);
        let with_jitter = truncated.saturating_mul(jitter_num) / 100;
        with_jitter.min(max) as u32
    }

    /// Delay for `attempt` with "equal jitter": half of the capped exponential
    /// delay plus a random share of the other half. `random_unit` must be a
    /// random number in `[0, 1)` (for example `Math.random()` in JavaScript);
    /// values outside that range are clamped.
    ///
    /// Spreading retries out like this stops many clients that failed at the
    /// same moment from all retrying at the same moment again.
    pub fn delay_with_jitter(&self, attempt: u32, random_unit: f64) -> u32 {
        let exp = attempt.saturating_sub(1).min(30);
        let capped = u64::from(self.base_delay_ms)
            .saturating_mul(1u64 << exp)
            .min(u64::from(self.max_delay_ms));
        let r = if random_unit.is_finite() { random_unit.clamp(0.0, 1.0) } else { 0.5 };
        let half = capped as f64 / 2.0;
        (half + half * r).round().min(f64::from(self.max_delay_ms)) as u32
    }

    /// Delay to wait before the next attempt, honouring the server's
    /// `Retry-After` header when there is one.
    ///
    /// `retry_after` is the raw header value. Only the delta-seconds form
    /// (`"120"`) is understood, because this crate has no clock to compare an
    /// HTTP date against; for a date, or no header, the jittered backoff is
    /// used. A server-requested delay is not capped by `max_delay_ms`.
    pub fn delay_for_response(&self, attempt: u32, retry_after: Option<&str>, random_unit: f64) -> u32 {
        match retry_after.and_then(parse_retry_after_seconds) {
            Some(secs) => secs.saturating_mul(1000),
            None => self.delay_with_jitter(attempt, random_unit),
        }
    }

    /// Return `true` if another attempt should be made.
    ///
    /// # Arguments
    /// * `attempt`: the attempt that just failed (1-indexed)
    /// * `status_code`: HTTP response status code
    ///
    /// Retries on: 429, 500, 502, 503, 504.
    /// Does not retry on: 400, 401, 403, 404, or any other code.
    pub fn should_retry(&self, attempt: u32, status_code: u16) -> bool {
        attempt < self.max_attempts && RETRYABLE_STATUS_CODES.contains(&status_code)
    }

    /// Maximum number of attempts configured.
    pub fn max_attempts(&self) -> u32 {
        self.max_attempts
    }
}

/// Parse the delta-seconds form of an HTTP `Retry-After` header (`"120"`).
/// Returns `None` for an HTTP date or anything else.
pub fn parse_retry_after_seconds(value: &str) -> Option<u32> {
    let v = value.trim();
    if v.is_empty() || !v.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    v.parse::<u32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_policy_new_ok() {
        let p = RetryPolicy::new(3, 100, 5_000).unwrap();
        assert_eq!(p.max_attempts(), 3);
    }

    #[test]
    fn test_retry_policy_new_zero_attempts_err() {
        let result = RetryPolicy::new(0, 100, 5_000);
        assert!(matches!(result, Err(LlmWasmError::InvalidConfig { field, .. }) if field == "max_attempts"));
    }

    #[test]
    fn test_retry_policy_new_base_exceeds_max_err() {
        let result = RetryPolicy::new(3, 6_000, 5_000);
        assert!(matches!(result, Err(LlmWasmError::InvalidConfig { field, .. }) if field == "base_delay_ms"));
    }

    #[test]
    fn test_retry_policy_delay_increases_with_attempts() {
        let p = RetryPolicy::exponential();
        let d1 = p.delay_for_attempt(1);
        let d2 = p.delay_for_attempt(2);
        let d3 = p.delay_for_attempt(3);
        assert!(d2 >= d1, "delay should be non-decreasing: d1={d1} d2={d2}");
        assert!(d3 >= d2, "delay should be non-decreasing: d2={d2} d3={d3}");
    }

    #[test]
    fn test_retry_delay_never_exceeds_max() {
        let p = RetryPolicy::exponential();
        for attempt in 1..=20 {
            let d = p.delay_for_attempt(attempt);
            assert!(
                d <= p.max_delay_ms,
                "attempt {attempt}: delay {d} exceeds max {}",
                p.max_delay_ms
            );
        }
    }

    #[test]
    fn test_retry_policy_should_retry_on_429() {
        let p = RetryPolicy::exponential();
        assert!(p.should_retry(1, 429));
        assert!(p.should_retry(2, 429));
    }

    #[test]
    fn test_retry_policy_should_retry_on_500_series() {
        let p = RetryPolicy::exponential();
        for code in [500u16, 502, 503, 504] {
            assert!(p.should_retry(1, code), "should retry on {code}");
        }
    }

    #[test]
    fn test_retry_policy_should_not_retry_on_400() {
        let p = RetryPolicy::exponential();
        assert!(!p.should_retry(1, 400));
    }

    #[test]
    fn test_retry_policy_should_not_retry_on_401_403_404() {
        let p = RetryPolicy::exponential();
        for code in [401u16, 403, 404] {
            assert!(!p.should_retry(1, code), "should not retry on {code}");
        }
    }

    #[test]
    fn test_retry_policy_should_not_retry_beyond_max() {
        let p = RetryPolicy::exponential(); // max_attempts = 3
        // attempt 3 is the last; no more retries after it
        assert!(!p.should_retry(3, 429));
    }

    #[test]
    fn test_retry_policy_exponential_defaults() {
        let p = RetryPolicy::exponential();
        assert_eq!(p.max_attempts(), 3);
        assert_eq!(p.base_delay_ms, 100);
        assert_eq!(p.max_delay_ms, 5_000);
    }

    #[test]
    fn test_jitter_spreads_delays() {
        // 0.1.x had no randomness: every client waited exactly the same time.
        let p = RetryPolicy::new(5, 1_000, 10_000).unwrap();
        assert_eq!(p.delay_with_jitter(1, 0.0), 500);
        assert_eq!(p.delay_with_jitter(1, 1.0), 1_000);
        assert_eq!(p.delay_with_jitter(2, 0.5), 1_500);
        assert_eq!(p.delay_with_jitter(10, 1.0), 10_000); // capped
        assert_eq!(p.delay_with_jitter(1, 7.0), 1_000); // clamped
        assert_eq!(p.delay_with_jitter(1, f64::NAN), 750);
    }

    #[test]
    fn test_retry_after_is_honoured() {
        let p = RetryPolicy::exponential();
        assert_eq!(p.delay_for_response(1, Some("7"), 0.0), 7_000);
        assert_eq!(p.delay_for_response(1, Some(" 2 "), 0.0), 2_000);
        // HTTP-date and junk fall back to backoff.
        assert_eq!(p.delay_for_response(1, Some("Wed, 21 Oct 2026 07:28:00 GMT"), 0.0), 50);
        assert_eq!(p.delay_for_response(1, Some("-3"), 0.0), 50);
        assert_eq!(p.delay_for_response(1, None, 1.0), 100);
        assert_eq!(parse_retry_after_seconds("120"), Some(120));
        assert_eq!(parse_retry_after_seconds(""), None);
    }
}
