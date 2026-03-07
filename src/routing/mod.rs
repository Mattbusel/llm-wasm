//! # Module: Routing
//!
//! ## Responsibility
//! Select a target model for a [`ChatRequest`] by evaluating an ordered list
//! of [`RoutingRule`]s. Returns the first matching rule's target, or a
//! fallback model when no rule matches.
//!
//! ## Guarantees
//! - `route()` always returns a non-empty string (falls back to the configured default)
//! - Rules are evaluated in insertion order; first match wins
//! - No I/O — purely in-memory evaluation

use crate::types::ChatRequest;

/// A condition that must hold for a routing rule to fire.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum RoutingCondition {
    /// Always matches.
    Always,
    /// Matches when the conversation has more than N messages.
    MessageCountExceeds(usize),
    /// Matches when the request's model string contains the given substring.
    ModelNameContains(String),
    /// Matches when `max_tokens` is set and below the given threshold.
    MaxTokensBelow(u32),
}

impl RoutingCondition {
    /// Evaluate this condition against a [`ChatRequest`].
    pub fn matches(&self, request: &ChatRequest) -> bool {
        match self {
            RoutingCondition::Always => true,
            RoutingCondition::MessageCountExceeds(n) => request.messages.len() > *n,
            RoutingCondition::ModelNameContains(substr) => {
                request.model.contains(substr.as_str())
            }
            RoutingCondition::MaxTokensBelow(threshold) => {
                request.max_tokens.is_some_and(|t| t < *threshold)
            }
        }
    }
}

/// A pairing of a condition with the model to route to when it fires.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RoutingRule {
    /// The condition to evaluate.
    pub condition: RoutingCondition,
    /// The model identifier to use when `condition` is true.
    pub target_model: String,
}

impl RoutingRule {
    /// Construct a new rule.
    pub fn new(condition: RoutingCondition, target_model: impl Into<String>) -> Self {
        Self { condition, target_model: target_model.into() }
    }
}

/// Routes requests to model targets based on an ordered rule list.
///
/// # Example
/// ```rust
/// use llm_wasm::routing::{Router, RoutingRule, RoutingCondition};
/// let mut router = Router::new("gpt-4o-mini");
/// router.add_rule(RoutingRule::new(RoutingCondition::Always, "claude-sonnet-4-6"));
/// // The Always rule fires first
/// let req = llm_wasm::types::ChatRequest::new("gpt-4o", vec![]);
/// assert_eq!(router.route(&req), "claude-sonnet-4-6");
/// ```
pub struct Router {
    rules: Vec<RoutingRule>,
    fallback: String,
}

impl Router {
    /// Create a router with no rules and the given fallback model.
    pub fn new(fallback: impl Into<String>) -> Self {
        Self { rules: Vec::new(), fallback: fallback.into() }
    }

    /// Append a rule to the end of the evaluation list.
    pub fn add_rule(&mut self, rule: RoutingRule) {
        self.rules.push(rule);
    }

    /// Evaluate rules in order and return the target model string.
    ///
    /// Returns the fallback model if no rule matches.
    pub fn route(&self, request: &ChatRequest) -> &str {
        for rule in &self.rules {
            if rule.condition.matches(request) {
                return &rule.target_model;
            }
        }
        &self.fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, Role};

    fn make_request(model: &str, msg_count: usize, max_tokens: Option<u32>) -> ChatRequest {
        let messages = (0..msg_count)
            .map(|_| ChatMessage::new(Role::User, "test"))
            .collect();
        ChatRequest { model: model.into(), messages, max_tokens, temperature: None }
    }

    #[test]
    fn test_router_returns_fallback_when_no_rules() {
        let router = Router::new("gpt-4o-mini");
        let req = make_request("anything", 1, None);
        assert_eq!(router.route(&req), "gpt-4o-mini");
    }

    #[test]
    fn test_router_always_condition_matches() {
        let mut router = Router::new("fallback");
        router.add_rule(RoutingRule::new(RoutingCondition::Always, "target-model"));
        let req = make_request("anything", 0, None);
        assert_eq!(router.route(&req), "target-model");
    }

    #[test]
    fn test_router_first_matching_rule_wins() {
        let mut router = Router::new("fallback");
        router.add_rule(RoutingRule::new(RoutingCondition::Always, "first"));
        router.add_rule(RoutingRule::new(RoutingCondition::Always, "second"));
        let req = make_request("m", 0, None);
        assert_eq!(router.route(&req), "first");
    }

    #[test]
    fn test_router_message_count_condition() {
        let mut router = Router::new("fallback");
        router.add_rule(RoutingRule::new(
            RoutingCondition::MessageCountExceeds(3),
            "long-context-model",
        ));
        let short = make_request("m", 2, None);
        let long = make_request("m", 5, None);
        assert_eq!(router.route(&short), "fallback");
        assert_eq!(router.route(&long), "long-context-model");
    }

    #[test]
    fn test_router_model_name_contains_condition() {
        let mut router = Router::new("fallback");
        router.add_rule(RoutingRule::new(
            RoutingCondition::ModelNameContains("sonnet".into()),
            "rerouted",
        ));
        let sonnet_req = make_request("claude-sonnet-4-6", 1, None);
        let other_req = make_request("gpt-4o", 1, None);
        assert_eq!(router.route(&sonnet_req), "rerouted");
        assert_eq!(router.route(&other_req), "fallback");
    }

    #[test]
    fn test_router_max_tokens_below_condition() {
        let mut router = Router::new("fallback");
        router.add_rule(RoutingRule::new(
            RoutingCondition::MaxTokensBelow(512),
            "mini-model",
        ));
        let small = make_request("m", 0, Some(100));
        let large = make_request("m", 0, Some(1024));
        let none = make_request("m", 0, None);
        assert_eq!(router.route(&small), "mini-model");
        assert_eq!(router.route(&large), "fallback");
        assert_eq!(router.route(&none), "fallback"); // None doesn't satisfy MaxTokensBelow
    }

    #[test]
    fn test_routing_condition_serialization_roundtrip() {
        let cond = RoutingCondition::MessageCountExceeds(5);
        let json = serde_json::to_string(&cond).unwrap();
        let back: RoutingCondition = serde_json::from_str(&json).unwrap();
        assert!(matches!(back, RoutingCondition::MessageCountExceeds(5)));
    }
}
