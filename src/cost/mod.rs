//! # Module: Cost
//!
//! ## Responsibility
//! Track accumulated token spend against a USD budget using a static pricing
//! table. No network calls — all prices are hardcoded.
//!
//! ## Guarantees
//! - `total_usd()` is always non-negative
//! - `record()` returns an error for unknown models rather than silently
//!   ignoring them
//! - `exceeded_budget()` is accurate after every `record()` call

use crate::error::LlmWasmError;

/// USD per-million-token pricing for one model.
#[derive(Debug, Clone, Copy)]
pub struct ModelPricing {
    /// Cost per 1 000 000 input tokens, in USD.
    pub input_per_million: f64,
    /// Cost per 1 000 000 output tokens, in USD.
    pub output_per_million: f64,
}

impl ModelPricing {
    /// Compute the USD cost for a given token usage.
    pub fn cost_usd(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        let input_cost = self.input_per_million * f64::from(input_tokens) / 1_000_000.0;
        let output_cost = self.output_per_million * f64::from(output_tokens) / 1_000_000.0;
        input_cost + output_cost
    }
}

/// Look up pricing for a known model.
///
/// # Arguments
/// * `model` — model identifier string
///
/// # Returns
/// `Ok(ModelPricing)` for known models, `Err(LlmWasmError::InvalidConfig)` otherwise.
///
/// # Panics
/// This function never panics.
pub fn pricing_for_model(model: &str) -> Result<ModelPricing, LlmWasmError> {
    match model {
        "claude-opus-4-6" => Ok(ModelPricing {
            input_per_million: 15.00,
            output_per_million: 75.00,
        }),
        "claude-sonnet-4-6" => Ok(ModelPricing {
            input_per_million: 3.00,
            output_per_million: 15.00,
        }),
        "claude-haiku-4-5-20251001" => Ok(ModelPricing {
            input_per_million: 0.80,
            output_per_million: 4.00,
        }),
        "gpt-4o" => Ok(ModelPricing {
            input_per_million: 2.50,
            output_per_million: 10.00,
        }),
        "gpt-4o-mini" => Ok(ModelPricing {
            input_per_million: 0.15,
            output_per_million: 0.60,
        }),
        other => Err(LlmWasmError::InvalidConfig {
            field: "model".into(),
            reason: format!("unknown model '{other}' — no pricing data available"),
        }),
    }
}

/// A single cost accounting entry.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CostEntry {
    model: String,
    input_tokens: u32,
    output_tokens: u32,
    cost_usd: f64,
}

/// Accumulates per-request costs and enforces an optional USD budget.
///
/// # Example
/// ```rust
/// use llm_wasm::cost::CostLedger;
/// let mut ledger = CostLedger::with_budget(1.00);
/// ledger.record("gpt-4o-mini", 1_000, 500).unwrap();
/// assert!(!ledger.exceeded_budget());
/// ```
#[derive(Debug)]
pub struct CostLedger {
    entries: Vec<CostEntry>,
    budget_usd: Option<f64>,
}

impl Default for CostLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl CostLedger {
    /// Create an unbounded ledger (no budget limit).
    pub fn new() -> Self {
        Self { entries: Vec::new(), budget_usd: None }
    }

    /// Create a ledger with a USD budget ceiling.
    ///
    /// # Arguments
    /// * `budget_usd` — maximum allowed spend in USD
    pub fn with_budget(budget_usd: f64) -> Self {
        Self { entries: Vec::new(), budget_usd: Some(budget_usd) }
    }

    /// Record a completed request and its token usage.
    ///
    /// # Arguments
    /// * `model` — model identifier; must be in the known pricing table
    /// * `input_tokens` — number of prompt tokens
    /// * `output_tokens` — number of completion tokens
    ///
    /// # Errors
    /// Returns [`LlmWasmError::InvalidConfig`] for unknown models.
    /// Returns [`LlmWasmError::BudgetExceeded`] if recording this usage would
    /// push total spend over the budget.
    pub fn record(
        &mut self,
        model: &str,
        input_tokens: u32,
        output_tokens: u32,
    ) -> Result<(), LlmWasmError> {
        let pricing = pricing_for_model(model)?;
        let cost_usd = pricing.cost_usd(input_tokens, output_tokens);
        let new_total = self.total_usd() + cost_usd;
        if let Some(budget) = self.budget_usd {
            if new_total > budget {
                return Err(LlmWasmError::BudgetExceeded {
                    used: new_total,
                    limit: budget,
                });
            }
        }
        self.entries.push(CostEntry {
            model: model.to_string(),
            input_tokens,
            output_tokens,
            cost_usd,
        });
        Ok(())
    }

    /// Return the sum of all recorded costs in USD.
    pub fn total_usd(&self) -> f64 {
        self.entries.iter().map(|e| e.cost_usd).sum()
    }

    /// Return `true` if the total spend exceeds the configured budget.
    ///
    /// Always `false` when no budget was set.
    pub fn exceeded_budget(&self) -> bool {
        match self.budget_usd {
            Some(budget) => self.total_usd() > budget,
            None => false,
        }
    }

    /// Number of recorded entries.
    pub fn entry_count(&self) -> u32 {
        self.entries.len() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_ledger_record_known_model_ok() {
        let mut ledger = CostLedger::new();
        assert!(ledger.record("gpt-4o-mini", 1_000, 500).is_ok());
    }

    #[test]
    fn test_cost_ledger_record_unknown_model_err() {
        let mut ledger = CostLedger::new();
        let result = ledger.record("unknown-model-xyz", 100, 50);
        assert!(matches!(result, Err(LlmWasmError::InvalidConfig { .. })));
    }

    #[test]
    fn test_cost_ledger_total_accumulates() {
        let mut ledger = CostLedger::new();
        ledger.record("gpt-4o-mini", 1_000_000, 0).unwrap(); // $0.15
        ledger.record("gpt-4o-mini", 1_000_000, 0).unwrap(); // $0.15
        let total = ledger.total_usd();
        assert!((total - 0.30).abs() < 1e-9, "expected 0.30, got {total}");
    }

    #[test]
    fn test_cost_ledger_total_always_non_negative() {
        let ledger = CostLedger::new();
        assert!(ledger.total_usd() >= 0.0);

        let mut ledger2 = CostLedger::new();
        ledger2.record("gpt-4o-mini", 100, 50).unwrap();
        assert!(ledger2.total_usd() >= 0.0);
    }

    #[test]
    fn test_cost_ledger_exceeded_budget_false_under_limit() {
        let mut ledger = CostLedger::with_budget(1.00);
        ledger.record("gpt-4o-mini", 1_000, 500).unwrap();
        assert!(!ledger.exceeded_budget());
    }

    #[test]
    fn test_cost_ledger_exceeded_budget_true_over_limit() {
        // Budget $0.001, but recording 1M output tokens of gpt-4o ($10/M) = $10
        let mut ledger = CostLedger::with_budget(0.001);
        let result = ledger.record("gpt-4o", 0, 1_000_000);
        assert!(matches!(result, Err(LlmWasmError::BudgetExceeded { .. })));
    }

    #[test]
    fn test_cost_ledger_no_budget_never_exceeded() {
        let mut ledger = CostLedger::new();
        // Record 100M tokens of the most expensive model
        ledger.record("claude-opus-4-6", 100_000_000, 100_000_000).unwrap();
        assert!(!ledger.exceeded_budget());
    }

    #[test]
    fn test_cost_ledger_entry_count() {
        let mut ledger = CostLedger::new();
        assert_eq!(ledger.entry_count(), 0);
        ledger.record("gpt-4o-mini", 100, 50).unwrap();
        ledger.record("gpt-4o", 200, 100).unwrap();
        assert_eq!(ledger.entry_count(), 2);
    }

    #[test]
    fn test_pricing_for_all_known_models() {
        for model in &[
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
            "gpt-4o",
            "gpt-4o-mini",
        ] {
            assert!(pricing_for_model(model).is_ok(), "expected pricing for {model}");
        }
    }

    #[test]
    fn test_pricing_cost_usd_calculation() {
        let p = ModelPricing { input_per_million: 3.0, output_per_million: 15.0 };
        let cost = p.cost_usd(1_000_000, 1_000_000);
        assert!((cost - 18.0).abs() < 1e-9);
    }
}
