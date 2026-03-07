//! Error types for the llm-wasm crate.
//!
//! All fallible operations in this crate return [`LlmWasmError`]. Every variant
//! is named, typed, and propagatable via `?`.

/// The unified error type for all llm-wasm operations.
///
/// # Variants
/// Each variant covers a distinct failure domain so callers can match precisely.
#[derive(Debug, thiserror::Error)]
pub enum LlmWasmError {
    /// A value could not be serialized or deserialized.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// A configuration field contains an invalid value.
    #[error("Invalid configuration: {field} — {reason}")]
    InvalidConfig { field: String, reason: String },

    /// A guard rejected the request.
    #[error("Request blocked by guard '{guard}': {reason}")]
    GuardBlocked { guard: String, reason: String },

    /// All retry attempts were consumed without success.
    #[error("Retry budget exhausted after {attempts} attempts")]
    RetryExhausted { attempts: u32 },

    /// The accumulated cost exceeded the configured budget.
    #[error("Cost budget exceeded: used ${used:.4}, limit ${limit:.4}")]
    BudgetExceeded { used: f64, limit: f64 },

    /// A template could not be rendered.
    #[error("Template render error: {0}")]
    TemplateError(String),

    /// No routing rule matched the request and no fallback was available.
    #[error("No route matched for request")]
    NoRouteMatched,

    /// A cache operation failed.
    #[error("Cache error: {0}")]
    CacheError(String),
}

impl From<serde_json::Error> for LlmWasmError {
    fn from(e: serde_json::Error) -> Self {
        LlmWasmError::Serialization(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization_error_display() {
        let e = LlmWasmError::Serialization("bad json".into());
        assert!(e.to_string().contains("bad json"));
    }

    #[test]
    fn test_invalid_config_display() {
        let e = LlmWasmError::InvalidConfig {
            field: "temperature".into(),
            reason: "must be in [0,2]".into(),
        };
        let s = e.to_string();
        assert!(s.contains("temperature"));
        assert!(s.contains("must be in [0,2]"));
    }

    #[test]
    fn test_guard_blocked_display() {
        let e = LlmWasmError::GuardBlocked {
            guard: "content".into(),
            reason: "blocklist hit".into(),
        };
        let s = e.to_string();
        assert!(s.contains("content"));
        assert!(s.contains("blocklist hit"));
    }

    #[test]
    fn test_retry_exhausted_display() {
        let e = LlmWasmError::RetryExhausted { attempts: 3 };
        assert!(e.to_string().contains("3"));
    }

    #[test]
    fn test_budget_exceeded_display() {
        let e = LlmWasmError::BudgetExceeded { used: 1.5, limit: 1.0 };
        let s = e.to_string();
        assert!(s.contains("1.5000"));
        assert!(s.contains("1.0000"));
    }

    #[test]
    fn test_template_error_display() {
        let e = LlmWasmError::TemplateError("missing var".into());
        assert!(e.to_string().contains("missing var"));
    }

    #[test]
    fn test_no_route_matched_display() {
        let e = LlmWasmError::NoRouteMatched;
        assert!(e.to_string().contains("No route"));
    }

    #[test]
    fn test_cache_error_display() {
        let e = LlmWasmError::CacheError("expired".into());
        assert!(e.to_string().contains("expired"));
    }

    #[test]
    fn test_from_serde_json_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("{bad}").unwrap_err();
        let e = LlmWasmError::from(json_err);
        assert!(matches!(e, LlmWasmError::Serialization(_)));
    }
}
