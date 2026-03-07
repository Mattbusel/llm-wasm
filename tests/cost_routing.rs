//! Integration test: route by model, record cost, verify ledger.

use llm_wasm::cost::CostLedger;
use llm_wasm::routing::{Router, RoutingCondition, RoutingRule};
use llm_wasm::types::{ChatMessage, ChatRequest, Role};

fn make_request(model: &str, msg_count: usize, max_tokens: Option<u32>) -> ChatRequest {
    let messages = (0..msg_count)
        .map(|_| ChatMessage::new(Role::User, "test message content"))
        .collect();
    ChatRequest { model: model.into(), messages, max_tokens, temperature: None }
}

#[test]
fn test_route_then_record_cost() {
    // Router: if request has more than 2 messages use opus, otherwise use mini
    let mut router = Router::new("gpt-4o-mini");
    router.add_rule(RoutingRule::new(
        RoutingCondition::MessageCountExceeds(2),
        "claude-opus-4-6",
    ));

    let short_req = make_request("gpt-4o", 1, None);
    let long_req = make_request("gpt-4o", 5, None);

    let model_a = router.route(&short_req);
    let model_b = router.route(&long_req);

    assert_eq!(model_a, "gpt-4o-mini");
    assert_eq!(model_b, "claude-opus-4-6");

    let mut ledger = CostLedger::new();
    ledger.record(model_a, 1_000, 500).unwrap();
    ledger.record(model_b, 2_000, 1_000).unwrap();

    assert_eq!(ledger.entry_count(), 2);
    assert!(ledger.total_usd() > 0.0);
}

#[test]
fn test_budget_enforced_after_routing() {
    let mut router = Router::new("claude-opus-4-6"); // expensive default
    router.add_rule(RoutingRule::new(
        RoutingCondition::MaxTokensBelow(100),
        "gpt-4o-mini",
    ));

    let req = make_request("m", 1, Some(50)); // will route to mini
    let target = router.route(&req);
    assert_eq!(target, "gpt-4o-mini");

    // Tight budget: $0.0001
    let mut ledger = CostLedger::with_budget(0.0001);
    // gpt-4o-mini: $0.15/M input. 1M tokens = $0.15 >> $0.0001
    let result = ledger.record(target, 1_000_000, 0);
    assert!(result.is_err(), "should exceed budget");
}

#[test]
fn test_ledger_accumulates_across_multiple_routed_requests() {
    let router = Router::new("gpt-4o-mini");

    let mut ledger = CostLedger::new();
    for _ in 0..5 {
        let req = make_request("m", 1, None);
        let model = router.route(&req);
        ledger.record(model, 100, 50).unwrap();
    }

    assert_eq!(ledger.entry_count(), 5);
    let total = ledger.total_usd();
    // 5 * (100 * 0.15/1M + 50 * 0.60/1M) = 5 * (0.000015 + 0.00003) = 5 * 0.000045 = 0.000225
    assert!(total > 0.0);
    assert!(!ledger.exceeded_budget());
}
