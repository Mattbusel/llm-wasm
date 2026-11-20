//! `rig::CostLedgerHook` against rig's mock model.
//!
//! Needs rig-agent's test utilities (and Rust 1.97.1):
//! `cargo test --features __rig-test-utils --test rig_hook` (or `--all-features`).

#![allow(clippy::unwrap_used)]

use std::sync::{Arc, Mutex};

use llm_wasm::cost::{cost_for_model, CostLedger, ModelPricing};
use llm_wasm::rig::CostLedgerHook;
use llm_wasm::types::Usage;
use rig_agent::agent::{AgentBuilder, AgentHook, HookContext, ModelTurnAction, ModelTurnFinished};
use rig_agent::completion::{Cost, Usage as RigUsage};
use rig_agent::test_utils::{MockAddTool, MockCompletionModel, MockTurn};
use serde_json::json;

fn tool_turn(id: &str, usage: RigUsage) -> MockTurn {
    MockTurn::tool_call(id, "add", json!({"x": 1, "y": 1})).with_usage(usage)
}

/// 10,000 prompt + 500 completion tokens on gpt-4o-mini, whatever the price
/// source: priced through the same function the hook uses.
fn turn_usage() -> RigUsage {
    RigUsage::new().input_tokens(10_000).output_tokens(500)
}

fn turn_usd(model: &str) -> f64 {
    cost_for_model(model, &Usage::from(&turn_usage())).unwrap()
}

fn spent(ledger: &Arc<Mutex<CostLedger>>) -> f64 {
    ledger.lock().unwrap().total_usd()
}

#[tokio::test]
async fn the_turn_that_crosses_the_budget_is_booked_then_the_run_stops() {
    let one = turn_usd("gpt-4o");
    // room for one and a half turns
    let hook = CostLedgerHook::new("gpt-4o", one * 1.5);
    let ledger = hook.ledger();
    let model = MockCompletionModel::from_turns([
        tool_turn("a", turn_usage()),
        tool_turn("b", turn_usage()),
        tool_turn("c", turn_usage()),
        MockTurn::text("done"),
    ]);
    let err = AgentBuilder::new(model)
        .tool(MockAddTool)
        .add_hook(hook)
        .build()
        .prompt("go")
        .max_turns(10)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("USD budget reached"), "{err}");
    // Both paid turns are on the books; the third call was never made.
    assert!(
        (spent(&ledger) - 2.0 * one).abs() < 1e-12,
        "spent {}",
        spent(&ledger)
    );
    assert_eq!(ledger.lock().unwrap().entry_count(), 2);
    assert!(ledger.lock().unwrap().exceeded_budget());
}

#[tokio::test]
async fn a_spent_budget_stops_before_the_first_call() {
    let ledger = Arc::new(Mutex::new(CostLedger::with_budget(0.01)));
    ledger.lock().unwrap().record_paid_usd(0.01).unwrap();
    let model =
        MockCompletionModel::from_turns([MockTurn::text("never sent").with_usage(turn_usage())]);
    let err = AgentBuilder::new(model)
        .add_hook(CostLedgerHook::with_ledger(ledger.clone(), "gpt-4o"))
        .build()
        .prompt("go")
        .await
        .unwrap_err();
    assert!(err.to_string().contains("USD budget reached"), "{err}");
    assert_eq!(ledger.lock().unwrap().entry_count(), 1);
}

#[tokio::test]
async fn stop_below_keeps_the_last_call_from_being_made() {
    let one = turn_usd("gpt-4o-mini");
    // After one turn, 0.9 of a turn is left: below the 1-turn threshold.
    let hook = CostLedgerHook::new("gpt-4o-mini", one * 1.9).stop_below_usd(one);
    let ledger = hook.ledger();
    let model = MockCompletionModel::from_turns([
        tool_turn("a", turn_usage()),
        tool_turn("b", turn_usage()),
        MockTurn::text("done"),
    ]);
    let err = AgentBuilder::new(model)
        .tool(MockAddTool)
        .add_hook(hook)
        .build()
        .prompt("go")
        .max_turns(10)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("USD budget reached"), "{err}");
    assert!((spent(&ledger) - one).abs() < 1e-12);
    assert!(!ledger.lock().unwrap().exceeded_budget(), "never went over");
}

#[tokio::test]
async fn an_over_budget_final_answer_is_returned() {
    let one = turn_usd("gpt-4o");
    let hook = CostLedgerHook::new("gpt-4o", one * 1.5);
    let ledger = hook.ledger();
    let model = MockCompletionModel::from_turns([
        tool_turn("a", turn_usage()),
        MockTurn::text("the answer").with_usage(turn_usage()),
    ]);
    let answer = AgentBuilder::new(model)
        .tool(MockAddTool)
        .add_hook(hook)
        .build()
        .prompt("go")
        .max_turns(10)
        .await
        .unwrap();
    assert!(format!("{answer:?}").contains("the answer"), "{answer:?}");
    assert!((spent(&ledger) - 2.0 * one).abs() < 1e-12);
}

#[tokio::test]
async fn cache_reads_and_writes_are_priced_at_their_own_rates() {
    let usage = RigUsage::new()
        .input_tokens(103_000)
        .cached_input_tokens(100_000)
        .cache_creation_input_tokens(2_000)
        .output_tokens(500);
    let hook = CostLedgerHook::new("claude-opus-5-5", 10.0);
    let ledger = hook.ledger();
    let model = MockCompletionModel::from_turns([MockTurn::text("ok").with_usage(usage)]);
    AgentBuilder::new(model)
        .add_hook(hook)
        .build()
        .prompt("go")
        .await
        .unwrap();
    let ours = Usage::new(1_000, 500)
        .with_cache_read(100_000)
        .with_cache_write(2_000);
    assert_eq!(Usage::from(&usage), ours);
    let expected = cost_for_model("claude-opus-5-5", &ours).unwrap();
    let uncached = cost_for_model("claude-opus-5-5", &Usage::new(103_000, 500)).unwrap();
    assert!((spent(&ledger) - expected).abs() < 1e-12);
    assert!(
        spent(&ledger) < uncached / 3.0,
        "cache-aware {} vs uncached {uncached}",
        spent(&ledger)
    );
}

#[tokio::test]
async fn custom_pricing_wins_over_the_table() {
    let hook = CostLedgerHook::new("my-finetune", 1.0).with_pricing(ModelPricing::new(1.0, 2.0));
    let ledger = hook.ledger();
    let model = MockCompletionModel::from_turns([MockTurn::text("ok").with_usage(turn_usage())]);
    AgentBuilder::new(model)
        .add_hook(hook)
        .build()
        .prompt("go")
        .await
        .unwrap();
    // 10,000 x $1/M + 500 x $2/M
    assert!((spent(&ledger) - 0.011).abs() < 1e-12);
}

#[tokio::test]
async fn an_unknown_model_uses_rigs_reported_cost() {
    let usage = turn_usage().cost(Cost::from_parts(0.01, 0.002, 0.0, 0.0));
    let hook = CostLedgerHook::new("model-in-no-table", 1.0);
    let ledger = hook.ledger();
    let model = MockCompletionModel::from_turns([MockTurn::text("ok").with_usage(usage)]);
    AgentBuilder::new(model)
        .add_hook(hook)
        .build()
        .prompt("go")
        .await
        .unwrap();
    assert!((spent(&ledger) - 0.012).abs() < 1e-12);
}

#[tokio::test]
async fn an_unpriceable_turn_stops_the_run() {
    let hook = CostLedgerHook::new("model-in-no-table", 1.0);
    let ledger = hook.ledger();
    let model =
        MockCompletionModel::from_turns([tool_turn("a", turn_usage()), MockTurn::text("done")]);
    let err = AgentBuilder::new(model)
        .tool(MockAddTool)
        .add_hook(hook)
        .build()
        .prompt("go")
        .max_turns(10)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("cannot price"), "{err}");
    assert_eq!(spent(&ledger), 0.0);
}

#[tokio::test]
async fn agents_on_different_models_share_one_budget() {
    let ledger = Arc::new(Mutex::new(CostLedger::with_budget(1.0)));
    for model_name in ["gpt-4o", "gpt-4o-mini"] {
        let model =
            MockCompletionModel::from_turns([MockTurn::text("ok").with_usage(turn_usage())]);
        AgentBuilder::new(model)
            .add_hook(CostLedgerHook::with_ledger(ledger.clone(), model_name))
            .build()
            .prompt("go")
            .await
            .unwrap();
    }
    let expected = turn_usd("gpt-4o") + turn_usd("gpt-4o-mini");
    assert!((spent(&ledger) - expected).abs() < 1e-12);
    assert_eq!(ledger.lock().unwrap().entry_count(), 2);
}

/// Stops every finished turn, like a token cap that has been reached.
struct StopEveryTurn;

impl AgentHook for StopEveryTurn {
    async fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        ModelTurnAction::Stop("token cap reached".into())
    }
}

#[tokio::test]
async fn added_first_it_books_a_turn_a_later_hook_stops() {
    let hook = CostLedgerHook::new("gpt-4o", 1.0);
    let ledger = hook.ledger();
    let model =
        MockCompletionModel::from_turns([tool_turn("a", turn_usage()), MockTurn::text("done")]);
    let err = AgentBuilder::new(model)
        .tool(MockAddTool)
        .add_hook(hook)
        .add_hook(StopEveryTurn)
        .build()
        .prompt("go")
        .max_turns(10)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("token cap reached"), "{err}");
    assert!((spent(&ledger) - turn_usd("gpt-4o")).abs() < 1e-12);
}
