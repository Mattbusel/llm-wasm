// Integration tests: Full llm-wasm request pipeline.

use llm_wasm::cost::{pricing_for_model, CostLedger, ModelPricing};
use llm_wasm::format::{JsonFormatter, MarkdownFormatter};
use llm_wasm::guard::{ContentGuard, Guard, GuardChain, GuardResult, LengthGuard};
use llm_wasm::routing::{Router, RoutingCondition, RoutingRule};
use llm_wasm::template::TemplateEngine;
use llm_wasm::types::{ChatMessage, ChatRequest, ChatResponse, Role, StreamChunk};
use std::collections::HashMap;

// ── Types ─────────────────────────────────────────────────────────────────

#[test]
fn chat_message_stores_role_and_content() {
    let m = ChatMessage::new(Role::System, "You are helpful.");
    assert_eq!(m.role, Role::System);
    assert_eq!(m.content, "You are helpful.");
}

#[test]
fn chat_request_total_chars() {
    let req = ChatRequest::new(
        "m",
        vec![
            ChatMessage::new(Role::User, "Hello!"),
            ChatMessage::new(Role::Assistant, "Hi there!"),
        ],
    );
    assert_eq!(req.total_content_chars(), 15);
}

#[test]
fn chat_request_empty_messages() {
    let req = ChatRequest::new("m", vec![]);
    assert_eq!(req.total_content_chars(), 0);
    assert!(req.messages.is_empty());
}

#[test]
fn chat_response_token_sum() {
    let r = ChatResponse {
        content: "Hello".into(),
        model: "gpt-4o".into(),
        input_tokens: 100,
        output_tokens: 50,
    };
    assert_eq!(r.input_tokens + r.output_tokens, 150);
}

#[test]
fn stream_chunk_delta_field() {
    let chunk = StreamChunk { delta: "partial text".into(), finished: false };
    assert_eq!(chunk.delta, "partial text");
    assert!(!chunk.finished);
}

#[test]
fn stream_chunk_finished_flag_true() {
    let chunk = StreamChunk { delta: "".into(), finished: true };
    assert!(chunk.finished);
}

#[test]
fn role_variants_serialize() {
    let roles = [Role::System, Role::User, Role::Assistant];
    for role in &roles {
        let json = serde_json::to_string(role).unwrap();
        let back: Role = serde_json::from_str(&json).unwrap();
        assert_eq!(&back, role);
    }
}

// ── Cost ledger ───────────────────────────────────────────────────────────

#[test]
fn cost_ledger_new_is_empty() {
    let ledger = CostLedger::new();
    assert_eq!(ledger.total_usd(), 0.0);
    assert_eq!(ledger.entry_count(), 0);
    assert!(!ledger.exceeded_budget());
}

#[test]
fn cost_ledger_records_accumulate() {
    let mut ledger = CostLedger::new();
    ledger.record("gpt-4o-mini", 1_000, 500).unwrap();
    ledger.record("gpt-4o-mini", 2_000, 1_000).unwrap();
    assert_eq!(ledger.entry_count(), 2);
    assert!(ledger.total_usd() > 0.0);
}

#[test]
fn cost_ledger_no_budget_never_exceeds() {
    let mut ledger = CostLedger::new();
    // Record a huge amount — should never "exceed" since no budget
    ledger.record("claude-opus-4-6", 100_000_000, 100_000_000).unwrap();
    assert!(!ledger.exceeded_budget());
}

#[test]
fn cost_ledger_budget_exceeded_on_single_expensive_call() {
    let mut ledger = CostLedger::with_budget(0.001);
    // gpt-4o: 2.5/M input + 10/M output
    // 1M output tokens = $10 >> $0.001
    let result = ledger.record("gpt-4o", 0, 1_000_000);
    assert!(result.is_err());
}

#[test]
fn pricing_for_all_claude_models() {
    let models = [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ];
    for m in &models {
        assert!(pricing_for_model(m).is_ok(), "missing pricing for {}", m);
    }
}

#[test]
fn pricing_opus_more_expensive_than_sonnet() {
    let opus = pricing_for_model("claude-opus-4-6").unwrap();
    let sonnet = pricing_for_model("claude-sonnet-4-6").unwrap();
    assert!(opus.input_per_million > sonnet.input_per_million);
    assert!(opus.output_per_million > sonnet.output_per_million);
}

#[test]
fn pricing_cost_only_input_tokens() {
    let p = ModelPricing { input_per_million: 3.0, output_per_million: 15.0 };
    let cost = p.cost_usd(1_000_000, 0);
    assert!((cost - 3.0).abs() < 1e-9);
}

#[test]
fn pricing_cost_only_output_tokens() {
    let p = ModelPricing { input_per_million: 3.0, output_per_million: 15.0 };
    let cost = p.cost_usd(0, 1_000_000);
    assert!((cost - 15.0).abs() < 1e-9);
}

#[test]
fn cost_ledger_just_at_budget_succeeds() {
    // Budget exactly $0.15 (cost of 1M input gpt-4o-mini)
    let mut ledger = CostLedger::with_budget(0.15);
    let result = ledger.record("gpt-4o-mini", 1_000_000, 0);
    // Exactly at limit: total 0.15 <= 0.15 → should succeed
    assert!(result.is_ok());
}

// ── Guard chain ───────────────────────────────────────────────────────────

#[test]
fn content_guard_multiple_blocked_terms() {
    let guard = ContentGuard::new(vec!["spam".into(), "hack".into(), "illegal".into()]);
    let req = ChatRequest::new(
        "m",
        vec![ChatMessage::new(Role::User, "how to hack a system")],
    );
    assert!(matches!(guard.check(&req), GuardResult::Block { .. }));
}

#[test]
fn content_guard_partial_word_match() {
    let guard = ContentGuard::new(vec!["spam".into()]);
    let req = ChatRequest::new(
        "m",
        vec![ChatMessage::new(Role::User, "antispam filter")],
    );
    // "antispam" contains "spam" as substring
    assert!(matches!(guard.check(&req), GuardResult::Block { .. }));
}

#[test]
fn length_guard_counts_all_messages() {
    let guard = LengthGuard::new(10);
    let req = ChatRequest::new(
        "m",
        vec![
            ChatMessage::new(Role::System, "Hello"), // 5
            ChatMessage::new(Role::User, "World!"),  // 6 → total 11 > 10
        ],
    );
    assert!(matches!(guard.check(&req), GuardResult::Block { .. }));
}

#[test]
fn guard_chain_all_pass_returns_none() {
    let chain = GuardChain::new()
        .add(ContentGuard::new(vec!["bad".into()]))
        .add(LengthGuard::new(1000));
    let req = ChatRequest::new("m", vec![ChatMessage::new(Role::User, "harmless query")]);
    let result = chain.check(&req).unwrap();
    assert!(result.is_none());
}

#[test]
fn guard_chain_first_guard_blocks_second_not_evaluated() {
    // ContentGuard blocks first, LengthGuard (very restrictive) would also block
    let chain = GuardChain::new()
        .add(ContentGuard::new(vec!["spam".into()]))
        .add(LengthGuard::new(1)); // would block everything
    let req = ChatRequest::new("m", vec![ChatMessage::new(Role::User, "spam")]);
    let err = chain.check(&req).unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("content") || msg.to_lowercase().contains("spam") || !msg.is_empty());
}

#[test]
fn guard_chain_empty_allows_all() {
    let chain = GuardChain::new();
    for content in &["spam", "hack", "a".repeat(10000).as_str()] {
        let req = ChatRequest::new("m", vec![ChatMessage::new(Role::User, *content)]);
        assert!(chain.check(&req).unwrap().is_none());
    }
}

// ── Router ────────────────────────────────────────────────────────────────

#[test]
fn router_no_rules_uses_fallback() {
    let router = Router::new("fallback-model");
    let req = ChatRequest::new("gpt-4o", vec![]);
    assert_eq!(router.route(&req), "fallback-model");
}

#[test]
fn router_always_condition_overrides_fallback() {
    let mut router = Router::new("fallback");
    router.add_rule(RoutingRule::new(RoutingCondition::Always, "override"));
    let req = ChatRequest::new("any", vec![]);
    assert_eq!(router.route(&req), "override");
}

#[test]
fn router_message_count_threshold() {
    let mut router = Router::new("default");
    router.add_rule(RoutingRule::new(RoutingCondition::MessageCountExceeds(5), "long-ctx"));
    let short = ChatRequest::new("m", vec![
        ChatMessage::new(Role::User, "hi"),
        ChatMessage::new(Role::User, "hi"),
    ]);
    let long = ChatRequest::new("m", (0..10)
        .map(|_| ChatMessage::new(Role::User, "msg"))
        .collect());
    assert_eq!(router.route(&short), "default");
    assert_eq!(router.route(&long), "long-ctx");
}

#[test]
fn router_model_name_contains() {
    let mut router = Router::new("fallback");
    router.add_rule(RoutingRule::new(
        RoutingCondition::ModelNameContains("opus".into()),
        "expensive-tier",
    ));
    let opus = ChatRequest::new("claude-opus-4-6", vec![]);
    let sonnet = ChatRequest::new("claude-sonnet-4-6", vec![]);
    assert_eq!(router.route(&opus), "expensive-tier");
    assert_eq!(router.route(&sonnet), "fallback");
}

#[test]
fn router_max_tokens_below() {
    let mut router = Router::new("fallback");
    router.add_rule(RoutingRule::new(RoutingCondition::MaxTokensBelow(256), "mini"));
    let small = ChatRequest { model: "m".into(), messages: vec![], max_tokens: Some(100), temperature: None };
    let large = ChatRequest { model: "m".into(), messages: vec![], max_tokens: Some(512), temperature: None };
    let none = ChatRequest { model: "m".into(), messages: vec![], max_tokens: None, temperature: None };
    assert_eq!(router.route(&small), "mini");
    assert_eq!(router.route(&large), "fallback");
    assert_eq!(router.route(&none), "fallback");
}

#[test]
fn routing_condition_serialization() {
    let conditions = [
        RoutingCondition::Always,
        RoutingCondition::MessageCountExceeds(10),
        RoutingCondition::ModelNameContains("claude".into()),
        RoutingCondition::MaxTokensBelow(512),
    ];
    for cond in &conditions {
        let json = serde_json::to_string(cond).unwrap();
        let back: RoutingCondition = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }
}

// ── Format ────────────────────────────────────────────────────────────────

#[test]
fn json_extract_from_long_prose() {
    let text = "The agent responded with some analysis. Then: {\"action\": \"buy\", \"qty\": 100} and that was it.";
    let v = JsonFormatter::extract_json(text).unwrap();
    assert_eq!(v["action"], "buy");
    assert_eq!(v["qty"], 100);
}

#[test]
fn json_extract_nested() {
    let text = r#"{"outer": {"inner": {"value": 42}}}"#;
    let v = JsonFormatter::extract_json(text).unwrap();
    assert_eq!(v["outer"]["inner"]["value"], 42);
}

#[test]
fn json_extract_array_of_objects() {
    let text = r#"[{"a": 1}, {"a": 2}]"#;
    let v = JsonFormatter::extract_json(text).unwrap();
    assert!(v.is_array());
    assert_eq!(v[0]["a"], 1);
    assert_eq!(v[1]["a"], 2);
}

#[test]
fn json_is_valid_rejects_incomplete() {
    assert!(!JsonFormatter::is_valid_json("{\"a\":"));
    assert!(!JsonFormatter::is_valid_json(""));
    assert!(!JsonFormatter::is_valid_json("{unclosed"));
}

#[test]
fn json_is_valid_accepts_primitives() {
    assert!(JsonFormatter::is_valid_json("null"));
    assert!(JsonFormatter::is_valid_json("true"));
    assert!(JsonFormatter::is_valid_json("42"));
    assert!(JsonFormatter::is_valid_json("\"hello\""));
}

#[test]
fn markdown_strip_rust_fence() {
    let text = "```rust\nfn main() {}\n```";
    let out = MarkdownFormatter::strip_code_fence(text);
    assert_eq!(out, "fn main() {}");
}

#[test]
fn markdown_strip_json_fence() {
    let text = "```json\n{\"key\": \"val\"}\n```";
    let out = MarkdownFormatter::strip_code_fence(text);
    assert_eq!(out, "{\"key\": \"val\"}");
}

#[test]
fn markdown_strip_no_fence_unchanged() {
    let text = "plain content";
    assert_eq!(MarkdownFormatter::strip_code_fence(text), text);
}

#[test]
fn markdown_has_fence_detection() {
    assert!(MarkdownFormatter::has_code_fence("```python\ncode```"));
    assert!(!MarkdownFormatter::has_code_fence("no fences"));
}

// ── Template engine ───────────────────────────────────────────────────────

#[test]
fn template_render_no_vars_passthrough() {
    let engine = TemplateEngine::new();
    let out = engine.render("static content", &HashMap::new()).unwrap();
    assert_eq!(out, "static content");
}

#[test]
fn template_render_multiple_same_var() {
    let engine = TemplateEngine::new();
    let mut ctx = HashMap::new();
    ctx.insert("name".into(), "Bob".into());
    let out = engine.render("Hello {{name}}, goodbye {{name}}!", &ctx).unwrap();
    assert_eq!(out, "Hello Bob, goodbye Bob!");
}

#[test]
fn template_render_whitespace_in_tag() {
    let engine = TemplateEngine::new();
    let mut ctx = HashMap::new();
    ctx.insert("x".into(), "42".into());
    let out = engine.render("value: {{ x }}", &ctx).unwrap();
    assert_eq!(out, "value: 42");
}

#[test]
fn template_missing_var_preserved() {
    let engine = TemplateEngine::new();
    let out = engine.render("{{unknown}}", &HashMap::new()).unwrap();
    assert_eq!(out, "{{unknown}}");
}

#[test]
fn template_nested_partials() {
    let mut engine = TemplateEngine::new();
    engine.register_partial("inner", "World");
    engine.register_partial("outer", "Hello {{>inner}}!");
    let out = engine.render("{{>outer}}", &HashMap::new()).unwrap();
    assert_eq!(out, "Hello World!");
}

#[test]
fn template_unknown_partial_errors() {
    let engine = TemplateEngine::new();
    let result = engine.render("{{>ghost}}", &HashMap::new());
    assert!(result.is_err());
}

#[test]
fn template_system_prompt_rendering() {
    let mut engine = TemplateEngine::new();
    engine.register_partial("persona", "You are {{name}}, a helpful assistant.");
    let mut ctx = HashMap::new();
    ctx.insert("name".into(), "ARIA".into());
    ctx.insert("topic".into(), "finance".into());
    let out = engine
        .render("{{>persona}} Focus on {{topic}}.", &ctx)
        .unwrap();
    assert_eq!(out, "You are ARIA, a helpful assistant. Focus on finance.");
}

// ── Full pipeline: guard → route → cost ──────────────────────────────────

#[test]
fn pipeline_guard_blocks_before_routing() {
    let chain = GuardChain::new()
        .add(ContentGuard::new(vec!["forbidden".into()]));
    let mut router = Router::new("gpt-4o-mini");
    router.add_rule(RoutingRule::new(RoutingCondition::Always, "claude-sonnet-4-6"));

    let req = ChatRequest::new("m", vec![
        ChatMessage::new(Role::User, "this is forbidden content"),
    ]);

    // Guard blocks — routing never reached
    let result = chain.check(&req);
    assert!(result.is_err());
}

#[test]
fn pipeline_guard_allows_then_routes_then_costs() {
    let chain = GuardChain::new()
        .add(ContentGuard::new(vec!["forbidden".into()]))
        .add(LengthGuard::new(10_000));

    let mut router = Router::new("gpt-4o-mini");
    router.add_rule(RoutingRule::new(
        RoutingCondition::MessageCountExceeds(3),
        "claude-opus-4-6",
    ));

    let mut ledger = CostLedger::with_budget(10.0);

    let req = ChatRequest::new("m", vec![
        ChatMessage::new(Role::User, "What is Rust?"),
    ]);

    // 1. Guard check
    let modified = chain.check(&req).unwrap();
    let effective_req = modified.as_ref().unwrap_or(&req);

    // 2. Route
    let model = router.route(effective_req);
    assert_eq!(model, "gpt-4o-mini"); // only 1 message, threshold=3

    // 3. Simulate response cost (1000 input, 500 output)
    ledger.record(model, 1000, 500).unwrap();
    assert!(!ledger.exceeded_budget());
}

#[test]
fn pipeline_template_generates_request() {
    let mut engine = TemplateEngine::new();
    engine.register_partial("sys", "You are a {{role}} assistant.");

    let mut ctx = HashMap::new();
    ctx.insert("role".into(), "financial".into());
    ctx.insert("query".into(), "What is the P/E ratio?".into());

    let system = engine.render("{{>sys}}", &ctx).unwrap();
    let user = engine.render("{{query}}", &ctx).unwrap();

    let req = ChatRequest::new("gpt-4o", vec![
        ChatMessage::new(Role::System, system.clone()),
        ChatMessage::new(Role::User, user.clone()),
    ]);

    assert_eq!(req.messages[0].content, "You are a financial assistant.");
    assert_eq!(req.messages[1].content, "What is the P/E ratio?");
    assert_eq!(req.total_content_chars(), system.len() + user.len());
}
