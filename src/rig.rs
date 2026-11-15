//! A USD budget for [rig](https://crates.io/crates/rig-agent) agents (feature `rig`).
//!
//! [`CostLedgerHook`] is an [`AgentHook`] for rig-agent 0.46. Attach it to an
//! agent and every finished model turn is priced at the model's real rates
//! (prompt-cache reads and writes at their own prices, reasoning at the
//! output price) and booked in a [`CostLedger`]. When the ledger reaches its
//! budget, the run stops.
//!
//! ```
//! use llm_wasm::rig::CostLedgerHook;
//!
//! // $2.00 for this agent, priced as claude-sonnet-4-5
//! let hook = CostLedgerHook::new("claude-sonnet-4-5", 2.00);
//! let ledger = hook.ledger(); // read the spend while or after it runs
//! // AgentBuilder::new(model).add_hook(hook).build()
//! assert_eq!(ledger.lock().unwrap().total_usd(), 0.0);
//! ```
//!
//! A complete, offline run with rig's mock model is in
//! `examples/rig_usd_budget.rs`.
//!
//! The feature needs Rust 1.97.1 (rig-agent 0.46's minimum); the rest of
//! the crate still builds on 1.88.
//!
//! # What happens on each call
//!
//! - **Before a model call** (`on_completion_call`): if the ledger has a
//!   budget and less than [`stop_below_usd`](CostLedgerHook::stop_below_usd)
//!   (default $0) is left, the run stops and the provider is never called.
//! - **After a model turn** (`on_model_turn_finished`): the turn's usage is
//!   priced with [`cost_for_model`] (the
//!   built-in table, or llm-prices 1.2 with feature `llm-prices`), or with
//!   your own [`ModelPricing`] from
//!   [`with_pricing`](CostLedgerHook::with_pricing). A model no table knows
//!   is priced at the cost rig reports for the turn, when it has a complete
//!   one; otherwise the run stops, since a budget cannot be enforced on calls
//!   it cannot price. The cost is booked with
//!   [`CostLedger::record_paid_usd`], even past the budget, because the turn
//!   has been paid for. Then, if the budget is spent and the turn asked for
//!   tool calls, the run stops before the tools run. A turn without tool
//!   calls is the run's final answer: it is returned, since it is paid for.
//!
//! So the ledger matches what the provider bills, and the budget is
//! overshot by at most one turn.
//!
//! # Sharing a budget
//!
//! The ledger is an `Arc<Mutex<CostLedger>>`. Pass the same one to several
//! hooks with [`CostLedgerHook::with_ledger`] to put many runs or agents
//! (with different models) under one budget.
//!
//! # Hook order: add `CostLedgerHook` first
//!
//! rig runs hooks in the order they were added, and for
//! `on_model_turn_finished` it stops at the first hook that does not return
//! `Continue`. A hook added before this one that stops or retries a turn
//! keeps that turn from being booked. This hook returns `Continue` unless
//! the budget is spent, so the hooks after it still see every turn.

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use rig_agent::agent::{
    AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, ModelTurnAction, ModelTurnFinished,
};
use rig_agent::completion::AssistantContent;

use crate::cost::{cost_for_model, CostLedger, ModelPricing};
use crate::error::LlmWasmError;
use crate::types::Usage;

impl From<&rig_agent::completion::Usage> for Usage {
    /// rig counts every prompt token in `input_tokens` (cache reads and
    /// writes included) and every generated token in `output_tokens`
    /// (reasoning included). [`Usage`] keeps each billed kind apart, so the
    /// parts are taken out of the totals (never below zero).
    fn from(u: &rig_agent::completion::Usage) -> Self {
        let input = u.input_tokens.unwrap_or(0);
        let read = u.cached_input_tokens.unwrap_or(0).min(input);
        let written = u.cache_creation_input_tokens.unwrap_or(0).min(input - read);
        let output = u.output_tokens.unwrap_or(0);
        let reasoning = u.reasoning_tokens.unwrap_or(0).min(output);
        Usage::new(input - read - written, output - reasoning)
            .with_cache_read(read)
            .with_cache_write(written)
            .with_reasoning(reasoning)
    }
}

/// Prices every rig model turn into a [`CostLedger`] and stops the run at
/// the ledger's USD budget. See the [module docs](self).
///
/// Add it with `add_hook` before any other hook (see
/// [Hook order](self#hook-order-add-costledgerhook-first)).
#[derive(Debug, Clone)]
pub struct CostLedgerHook {
    ledger: Arc<Mutex<CostLedger>>,
    model: String,
    pricing: Option<ModelPricing>,
    stop_below_usd: f64,
}

impl CostLedgerHook {
    /// A hook with its own ledger capped at `budget_usd`, pricing every turn
    /// as `model` (a name from the price table, such as `"gpt-4o"` or
    /// `"anthropic/claude-sonnet-4-5"`).
    pub fn new(model: impl Into<String>, budget_usd: f64) -> Self {
        Self::with_ledger(Arc::new(Mutex::new(CostLedger::with_budget(budget_usd))), model)
    }

    /// A hook that books into an existing ledger, so several runs or agents
    /// can share one budget. A ledger without a budget only counts.
    pub fn with_ledger(ledger: Arc<Mutex<CostLedger>>, model: impl Into<String>) -> Self {
        Self { ledger, model: model.into(), pricing: None, stop_below_usd: 0.0 }
    }

    /// Price turns with these rates instead of looking `model` up, for a
    /// model missing from the tables or a negotiated price.
    pub fn with_pricing(mut self, pricing: ModelPricing) -> Self {
        self.pricing = Some(pricing);
        self
    }

    /// Stop before a model call when less than `usd` of the budget is left
    /// (default 0: stop only once the budget is spent). Set it to about what
    /// one call costs to keep the overshoot near zero.
    pub fn stop_below_usd(mut self, usd: f64) -> Self {
        self.stop_below_usd = if usd.is_finite() { usd.max(0.0) } else { 0.0 };
        self
    }

    /// The ledger this hook books into.
    pub fn ledger(&self) -> Arc<Mutex<CostLedger>> {
        Arc::clone(&self.ledger)
    }

    /// The model name turns are priced as.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Total spend booked in the ledger so far, in USD.
    pub fn spent_usd(&self) -> f64 {
        self.lock().total_usd()
    }

    /// What one turn with this usage costs in USD: priced with
    /// [`with_pricing`](Self::with_pricing) rates when set, else from the
    /// price table for the hook's model, else rig's reported cost when it
    /// is complete.
    ///
    /// # Errors
    /// [`LlmWasmError::UnknownModel`] when the model is in no table and rig
    /// reported no complete cost.
    pub fn turn_cost_usd(&self, usage: &rig_agent::completion::Usage) -> Result<f64, LlmWasmError> {
        let ours = Usage::from(usage);
        if let Some(pricing) = &self.pricing {
            return Ok(pricing.cost(&ours));
        }
        match cost_for_model(&self.model, &ours) {
            Err(LlmWasmError::UnknownModel(model)) => usage
                .cost
                .as_ref()
                .filter(|c| c.is_complete() && c.total.is_finite() && c.total >= 0.0)
                .map(|c| c.total)
                .ok_or(LlmWasmError::UnknownModel(model)),
            other => other,
        }
    }

    fn lock(&self) -> MutexGuard<'_, CostLedger> {
        self.ledger.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn stop_reason(&self, ledger: &CostLedger) -> String {
        format!(
            "USD budget reached for {}: spent ${:.6} of ${:.6}",
            self.model,
            ledger.total_usd(),
            ledger.budget_usd().unwrap_or(f64::INFINITY)
        )
    }
}

impl AgentHook for CostLedgerHook {
    fn name(&self) -> Option<String> {
        Some(format!("llm_wasm::CostLedgerHook({}, stop_below={})", self.model, self.stop_below_usd))
    }

    async fn on_completion_call(&self, _ctx: &HookContext, _event: CompletionCallEvent<'_>) -> CompletionCallAction {
        let ledger = self.lock();
        match ledger.remaining_usd() {
            Some(left) if left <= self.stop_below_usd || ledger.exceeded_budget() => {
                CompletionCallAction::Stop(self.stop_reason(&ledger))
            }
            _ => CompletionCallAction::Continue,
        }
    }

    async fn on_model_turn_finished(&self, _ctx: &HookContext, event: ModelTurnFinished<'_>) -> ModelTurnAction {
        let cost = match self.turn_cost_usd(&event.usage) {
            Ok(cost) => cost,
            Err(e) => return ModelTurnAction::Stop(format!("cannot price this turn, so the budget cannot hold: {e}")),
        };
        let mut ledger = self.lock();
        if let Err(e) = ledger.record_paid_usd(cost) {
            return ModelTurnAction::Stop(e.to_string());
        }
        let spent = match ledger.remaining_usd() {
            Some(left) => left <= 0.0 || ledger.exceeded_budget(),
            None => false,
        };
        // A turn without tool calls is the final answer, already paid for:
        // return it. Otherwise stop before the tools run.
        if spent && event.content.iter().any(|c| matches!(c, AssistantContent::ToolCall(_))) {
            ModelTurnAction::Stop(self.stop_reason(&ledger))
        } else {
            ModelTurnAction::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_agent::completion::Usage as RigUsage;

    #[test]
    fn rig_usage_is_split_by_billed_kind() {
        let u = RigUsage::new()
            .input_tokens(10_000)
            .cached_input_tokens(8_000)
            .cache_creation_input_tokens(1_000)
            .output_tokens(500)
            .reasoning_tokens(200);
        assert_eq!(Usage::from(&u), Usage::new(1_000, 300).with_cache_read(8_000).with_cache_write(1_000).with_reasoning(200));
        // nonsense counts never underflow
        let odd = RigUsage::new().input_tokens(10).cached_input_tokens(8).cache_creation_input_tokens(8).reasoning_tokens(5);
        assert_eq!(Usage::from(&odd), Usage::new(0, 0).with_cache_read(8).with_cache_write(2));
        assert_eq!(Usage::from(&RigUsage::new()), Usage::default());
    }

    #[test]
    fn builder_and_accessors() {
        let hook = CostLedgerHook::new("gpt-4o", 1.0).stop_below_usd(f64::NAN);
        assert_eq!(hook.model(), "gpt-4o");
        assert_eq!(hook.stop_below_usd, 0.0);
        assert_eq!(hook.spent_usd(), 0.0);
        assert_eq!(hook.ledger().lock().unwrap().budget_usd(), Some(1.0));
        assert!(hook.name().unwrap().contains("gpt-4o"));
    }
}
