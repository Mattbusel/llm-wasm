//! Integration test: route by model, record cost, verify ledger.

use llm_wasm::cost::{context_window, pricing_for_model, CostLedger, ModelPricing};
use llm_wasm::routing::{Router, RoutingCondition, RoutingRule};
use llm_wasm::types::{ChatMessage, ChatRequest, Role, Usage};
use llm_wasm::LlmWasmError;

fn make_request(model: &str, msg_count: usize, max_tokens: Option<u32>) -> ChatRequest {
    let messages = (0..msg_count)
        .map(|_| ChatMessage::new(Role::User, "test message content"))
        .collect();
    ChatRequest {
        model: model.into(),
        messages,
        max_tokens,
        temperature: None,
    }
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

// Cache-aware cost, checked against rows of the built-in price snapshot.

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

#[test]
fn opus_5_5_mixed_cache_usage() {
    // claude-opus-5-5: $4 input, $20 output, $0.20 cache read, $5 cache write per million.
    let p = pricing_for_model("claude-opus-5-5").unwrap();
    assert_eq!(
        (
            p.input_per_million,
            p.output_per_million,
            p.cache_read_per_million,
            p.cache_write_per_million
        ),
        (4.0, 20.0, Some(0.2), Some(5.0))
    );
    let usage = Usage::new(3_000, 1_200)
        .with_cache_read(180_000)
        .with_cache_write(20_000);
    // 3k x 4 + 180k x 0.2 + 20k x 5 + 1.2k x 20 = 12_000 + 36_000 + 100_000 + 24_000 (per million)
    assert!(close(p.cost(&usage), 0.172));
    // Without the cache split the same tokens cost 203k x 4 + 1.2k x 20 = $0.836.
    assert!(close(p.cost_usd(203_000, 1_200), 0.836));
}

#[test]
fn second_turn_cache_hit_is_cheaper_than_list_price() {
    let p = pricing_for_model("claude-sonnet-4-5").unwrap();
    let first = Usage::new(500, 400).with_cache_write(100_000);
    let second = Usage::new(500, 400).with_cache_read(100_000);
    // 100k at $3.75 (write) vs 100k at $0.30 (read)
    assert!(close(p.cost(&first) - p.cost(&second), 0.345));
}

#[test]
fn openai_cached_input_and_reasoning() {
    // gpt-5: $1.25 input, $10 output, $0.125 cached input; no cache-write price.
    let p = pricing_for_model("openai/gpt-5").unwrap();
    assert_eq!((p.provider, p.cache_write_per_million), ("openai", None));
    // prompt_tokens 50_000 with 40_000 cached; completion_tokens 3_000 of which 2_000 reasoning
    let usage = Usage::new(10_000, 1_000)
        .with_cache_read(40_000)
        .with_reasoning(2_000);
    assert!(close(p.cost(&usage), 0.0475)); // 0.0125 + 0.005 + 0.03
                                            // cache writes on a provider without a write price are billed as input
    assert!(close(
        p.cost(&Usage::default().with_cache_write(1_000_000)),
        1.25
    ));
}

#[test]
fn ledger_record_usage_enforces_budget() {
    let mut ledger = CostLedger::with_budget(0.20);
    let usage = Usage::new(3_000, 1_200)
        .with_cache_read(180_000)
        .with_cache_write(20_000);
    assert!(close(
        ledger.record_usage("claude-opus-5-5", &usage).unwrap(),
        0.172
    ));
    let err = ledger.record_usage("claude-opus-5-5", &usage).unwrap_err();
    assert!(matches!(err, LlmWasmError::BudgetExceeded { .. }));
    assert_eq!(ledger.entry_count(), 1);
    assert!(close(ledger.total_usd(), 0.172));
    assert!(matches!(
        ledger.record_usage("no-such-model", &usage),
        Err(LlmWasmError::UnknownModel(m)) if m == "no-such-model"
    ));
    // a custom price for a model missing from the table
    let custom = ModelPricing::new(1.0, 2.0).with_cache_read(0.1);
    assert!(close(
        ledger
            .record_usd(custom.cost(&Usage::new(1_000, 1_000)))
            .unwrap(),
        0.175
    ));
}

#[test]
fn context_windows_route_long_prompts() {
    assert_eq!(context_window("claude-opus-5-5"), Some(1_000_000));
    assert_eq!(context_window("gpt-4o"), Some(128_000));
    assert_eq!(context_window("anthropic/claude-haiku-4-5"), Some(200_000));
    assert_eq!(context_window("unknown"), None);
    // pick the first candidate whose window fits the prompt
    let prompt_tokens = 150_000;
    let pick = ["gpt-4o", "claude-haiku-4-5", "claude-sonnet-4-5"]
        .into_iter()
        .find(|m| context_window(m).is_some_and(|w| w >= prompt_tokens));
    assert_eq!(pick, Some("claude-haiku-4-5"));
}
