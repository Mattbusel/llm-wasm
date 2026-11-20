//! A USD budget on a rig agent, run offline against rig's mock model.
//!
//! ```text
//! cargo run --example rig_usd_budget --features __rig-test-utils
//! ```
//!
//! With a real provider the only change is the model you hand to
//! `AgentBuilder::new`: the hook, the ledger and the numbers work the same.
//!
//! What it shows:
//! 1. Every model turn is priced at the model's real rates, prompt-cache
//!    reads at the cache price, and booked in a `CostLedger`.
//! 2. The turn that crosses the budget is still booked, because it was paid
//!    for, and the run stops before its tool calls run.
//! 3. A second agent on another model shares the ledger and is stopped
//!    before it calls the provider at all.

use std::sync::{Arc, Mutex, PoisonError};

use llm_wasm::cost::CostLedger;
use llm_wasm::rig::CostLedgerHook;
use rig_agent::agent::AgentBuilder;
use rig_agent::completion::Usage;
use rig_agent::test_utils::{MockAddTool, MockCompletionModel, MockTurn};

/// A tool-calling turn: 40,000 prompt tokens, 30,000 of them read from the
/// prompt cache, and 800 completion tokens.
fn paid_turn(id: &str) -> MockTurn {
    MockTurn::tool_call(id, "add", serde_json::json!({"x": 1, "y": 2})).with_usage(
        Usage::new()
            .input_tokens(40_000)
            .cached_input_tokens(30_000)
            .output_tokens(800),
    )
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // $0.15 for the whole team.
    let ledger = Arc::new(Mutex::new(CostLedger::with_budget(0.15)));
    let spent = || {
        ledger
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .total_usd()
    };

    // Agent 1 wants six paid turns on claude-sonnet-4-5; the budget covers fewer.
    let researcher = AgentBuilder::new(MockCompletionModel::from_turns([
        paid_turn("a"),
        paid_turn("b"),
        paid_turn("c"),
        paid_turn("d"),
        paid_turn("e"),
        paid_turn("f"),
        MockTurn::text("done"),
    ]))
    .tool(MockAddTool)
    .add_hook(CostLedgerHook::with_ledger(
        ledger.clone(),
        "claude-sonnet-4-5",
    ))
    .build();

    match researcher.prompt("research the topic").max_turns(10).await {
        Ok(answer) => println!("researcher finished: {answer:?}"),
        Err(e) => println!("researcher stopped: {e}"),
    }
    println!("spent ${:.4} of $0.15 after the researcher", spent());

    // Agent 2 shares the ledger on another model. The budget is spent, so it
    // is stopped before its first call and costs nothing.
    let writer = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::text("draft")]))
        .add_hook(CostLedgerHook::with_ledger(ledger.clone(), "gpt-4o-mini"))
        .build();
    if let Err(e) = writer.prompt("write it up").await {
        println!("writer stopped before calling the model: {e}");
    }

    let guard = ledger.lock().unwrap_or_else(PoisonError::into_inner);
    println!(
        "{} paid turns, ${:.4} total, over budget: {}",
        guard.entry_count(),
        guard.total_usd(),
        guard.exceeded_budget()
    );
}
