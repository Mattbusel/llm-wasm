//! The README quickstart. Run with `cargo run --example quickstart`.

use std::collections::HashMap;
use llm_wasm::cache::{cache_key, TtlCache};
use llm_wasm::cost::{pricing_for_model, CostLedger};
use llm_wasm::format::JsonFormatter;
use llm_wasm::guard::{ContentGuard, GuardChain, LengthGuard};
use llm_wasm::retry::RetryPolicy;
use llm_wasm::routing::{Router, RoutingCondition, RoutingRule};
use llm_wasm::template::TemplateEngine;
use llm_wasm::types::{ChatMessage, ChatRequest, Role};

fn main() -> Result<(), llm_wasm::LlmWasmError> {
    // 1. Build the prompt.
    let mut vars = HashMap::new();
    vars.insert("city".to_string(), "Paris".to_string());
    let prompt = TemplateEngine::new().render("Return JSON with the population of {{city}}.", &vars)?;
    let request = ChatRequest::new("gpt-4o", vec![ChatMessage::new(Role::User, prompt)]);

    // 2. Guard it: blocked words, maximum length in characters.
    let guards = GuardChain::new()
        .add(ContentGuard::whole_words(vec!["password".into()]))
        .add(LengthGuard::new(20_000));
    guards.check(&request)?; // Err(GuardBlocked) if a guard says no

    // 3. Pick a model.
    let mut router = Router::new("gpt-4o-mini");
    router.add_rule(RoutingRule::new(RoutingCondition::MessageCountExceeds(10), "claude-sonnet-4-5"));
    let model = router.route(&request).to_string();

    // 4. Check the cache (you supply the clock, e.g. Date.now() in JS).
    let now_ms = 1_700_000_000_000.0;
    let mut cache = TtlCache::with_capacity(60_000.0, 1_000); // 60 s, at most 1,000 entries
    let key = cache_key(&model, &serde_json::to_string(&request.messages)?);
    if cache.get(&key, now_ms).is_none() {
        // 5. Call the model yourself. If it fails, ask the policy what to do.
        let policy = RetryPolicy::exponential();
        if policy.should_retry(1, 429) {
            // Pass a random number in [0, 1) and the Retry-After header, if any.
            let wait_ms = policy.delay_for_response(1, Some("2"), 0.37);
            assert_eq!(wait_ms, 2_000);
        }
        let reply = "Sure! Here it is: {\"population\": 2102650}".to_string();
        cache.set(key, reply, now_ms);
    }

    // 6. Account for it (real per-model prices) and parse the answer.
    println!("{model} costs ${} per million input tokens", pricing_for_model(&model)?.input_per_million);
    let mut ledger = CostLedger::with_budget(1.00);
    ledger.record(&model, 40, 20)?; // Err(BudgetExceeded) past $1.00
    let reply = cache.get(&key, now_ms).unwrap_or_default();
    let json = JsonFormatter::extract_json(&reply)?;
    println!("{model}: {json} (spent ${:.6})", ledger.total_usd());
    Ok(())
}
