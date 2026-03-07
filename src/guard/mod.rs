//! # Module: Guard
//!
//! ## Responsibility
//! Inspect [`ChatRequest`]s before they are dispatched and either allow,
//! block, or modify them. Guards compose via [`GuardChain`].
//!
//! ## Guarantees
//! - `GuardChain::check` stops at the first `Block` result
//! - `GuardChain::check` returns `Ok(None)` if every guard allows the request
//! - No I/O — all evaluation is in-memory

use crate::error::LlmWasmError;
use crate::types::ChatRequest;

/// The outcome of a single guard evaluation.
pub enum GuardResult {
    /// The request may proceed unchanged.
    Allow,
    /// The request should be rejected with the given reason.
    Block {
        /// Human-readable explanation.
        reason: String,
    },
    /// The request should be replaced with this modified version.
    Modify(ChatRequest),
}

/// A composable check applied to a [`ChatRequest`].
pub trait Guard {
    /// Short name identifying this guard (used in error messages).
    fn name(&self) -> &str;

    /// Evaluate the guard against the request.
    fn check(&self, request: &ChatRequest) -> GuardResult;
}

/// Rejects requests whose messages contain any blocklisted term.
///
/// Matching is case-insensitive substring search.
///
/// # Example
/// ```rust
/// use llm_wasm::guard::{ContentGuard, Guard, GuardResult};
/// use llm_wasm::types::{ChatRequest, ChatMessage, Role};
///
/// let guard = ContentGuard::new(vec!["spam".into()]);
/// let req = ChatRequest::new("m", vec![ChatMessage::new(Role::User, "buy spam now")]);
/// assert!(matches!(guard.check(&req), GuardResult::Block { .. }));
/// ```
pub struct ContentGuard {
    blocklist: Vec<String>,
}

impl ContentGuard {
    /// Create a guard with the given blocklist terms.
    pub fn new(blocklist: Vec<String>) -> Self {
        Self { blocklist }
    }
}

impl Guard for ContentGuard {
    fn name(&self) -> &str {
        "content"
    }

    fn check(&self, request: &ChatRequest) -> GuardResult {
        for message in &request.messages {
            let lower = message.content.to_lowercase();
            for term in &self.blocklist {
                if lower.contains(term.to_lowercase().as_str()) {
                    return GuardResult::Block {
                        reason: format!("message contains blocked term '{term}'"),
                    };
                }
            }
        }
        GuardResult::Allow
    }
}

/// Rejects requests whose total message character count exceeds a limit.
///
/// # Example
/// ```rust
/// use llm_wasm::guard::{LengthGuard, Guard, GuardResult};
/// use llm_wasm::types::{ChatRequest, ChatMessage, Role};
///
/// let guard = LengthGuard::new(10);
/// let req = ChatRequest::new("m", vec![ChatMessage::new(Role::User, "hello world!")]);
/// assert!(matches!(guard.check(&req), GuardResult::Block { .. }));
/// ```
pub struct LengthGuard {
    max_total_chars: usize,
}

impl LengthGuard {
    /// Create a guard with the given maximum total character count.
    pub fn new(max_total_chars: usize) -> Self {
        Self { max_total_chars }
    }
}

impl Guard for LengthGuard {
    fn name(&self) -> &str {
        "length"
    }

    fn check(&self, request: &ChatRequest) -> GuardResult {
        let total: usize = request.messages.iter().map(|m| m.content.len()).sum();
        if total > self.max_total_chars {
            GuardResult::Block {
                reason: format!(
                    "total message length {total} exceeds limit {}",
                    self.max_total_chars
                ),
            }
        } else {
            GuardResult::Allow
        }
    }
}

/// Runs a sequence of guards in order, stopping at the first `Block`.
///
/// # Example
/// ```rust
/// use llm_wasm::guard::{GuardChain, ContentGuard, LengthGuard};
/// use llm_wasm::types::{ChatRequest, ChatMessage, Role};
///
/// let chain = GuardChain::new()
///     .add(ContentGuard::new(vec!["spam".into()]))
///     .add(LengthGuard::new(1000));
/// let req = ChatRequest::new("m", vec![ChatMessage::new(Role::User, "hello")]);
/// assert!(chain.check(&req).unwrap().is_none()); // allowed
/// ```
pub struct GuardChain {
    guards: Vec<Box<dyn Guard>>,
}

impl Default for GuardChain {
    fn default() -> Self {
        Self::new()
    }
}

impl GuardChain {
    /// Create an empty guard chain.
    pub fn new() -> Self {
        Self { guards: Vec::new() }
    }

    /// Append a guard and return `self` for chaining.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, guard: impl Guard + 'static) -> Self {
        self.guards.push(Box::new(guard));
        self
    }

    /// Run all guards in order against `request`.
    ///
    /// # Returns
    /// - `Ok(None)` — every guard allowed the request
    /// - `Ok(Some(modified))` — a guard returned a `Modify` result
    /// - `Err(LlmWasmError::GuardBlocked)` — a guard blocked the request
    ///
    /// Stops at the first `Block` or `Modify` result.
    ///
    /// # Panics
    /// This function never panics.
    pub fn check(&self, request: &ChatRequest) -> Result<Option<ChatRequest>, LlmWasmError> {
        let current = request;
        // We need an owned copy only if a Modify guard fires
        let mut owned: Option<ChatRequest> = None;

        for guard in &self.guards {
            let target = owned.as_ref().unwrap_or(current);
            match guard.check(target) {
                GuardResult::Allow => {}
                GuardResult::Block { reason } => {
                    return Err(LlmWasmError::GuardBlocked {
                        guard: guard.name().to_string(),
                        reason,
                    });
                }
                GuardResult::Modify(modified) => {
                    owned = Some(modified);
                }
            }
        }
        Ok(owned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, Role};

    fn req(content: &str) -> ChatRequest {
        ChatRequest::new("model", vec![ChatMessage::new(Role::User, content)])
    }

    #[test]
    fn test_content_guard_blocks_blocklisted_term() {
        let guard = ContentGuard::new(vec!["spam".into()]);
        assert!(matches!(guard.check(&req("buy spam now")), GuardResult::Block { .. }));
    }

    #[test]
    fn test_content_guard_case_insensitive() {
        let guard = ContentGuard::new(vec!["spam".into()]);
        assert!(matches!(guard.check(&req("Buy SPAM Now")), GuardResult::Block { .. }));
    }

    #[test]
    fn test_content_guard_allows_clean_request() {
        let guard = ContentGuard::new(vec!["spam".into()]);
        assert!(matches!(guard.check(&req("hello world")), GuardResult::Allow));
    }

    #[test]
    fn test_length_guard_blocks_over_limit() {
        let guard = LengthGuard::new(5);
        assert!(matches!(guard.check(&req("hello world")), GuardResult::Block { .. }));
    }

    #[test]
    fn test_length_guard_allows_under_limit() {
        let guard = LengthGuard::new(100);
        assert!(matches!(guard.check(&req("hi")), GuardResult::Allow));
    }

    #[test]
    fn test_length_guard_allows_at_exact_limit() {
        let guard = LengthGuard::new(5);
        assert!(matches!(guard.check(&req("hello")), GuardResult::Allow));
    }

    #[test]
    fn test_guard_chain_allows_clean_request() {
        let chain = GuardChain::new()
            .add(ContentGuard::new(vec!["bad".into()]))
            .add(LengthGuard::new(1000));
        let result = chain.check(&req("good content")).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_guard_chain_stops_at_first_block() {
        let chain = GuardChain::new()
            .add(ContentGuard::new(vec!["spam".into()]))
            .add(LengthGuard::new(3)); // also would block, but shouldn't be reached
        let result = chain.check(&req("spam"));
        assert!(matches!(
            result,
            Err(LlmWasmError::GuardBlocked { guard, .. }) if guard == "content"
        ));
    }

    #[test]
    fn test_guard_chain_empty_allows_everything() {
        let chain = GuardChain::new();
        assert!(chain.check(&req("anything")).unwrap().is_none());
    }

    #[test]
    fn test_guard_chain_length_blocks() {
        let chain = GuardChain::new().add(LengthGuard::new(2));
        let result = chain.check(&req("too long"));
        assert!(matches!(result, Err(LlmWasmError::GuardBlocked { .. })));
    }
}
