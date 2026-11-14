//! # Module: Cost
//!
//! Track token spend in US dollars against an optional budget.
//!
//! Prices come from a table compiled into the crate, generated from LiteLLM's
//! maintained `model_prices_and_context_window.json` (MIT license) by
//! `scripts/update_prices.py`: 232 chat models from OpenAI, Anthropic, Google
//! Gemini, Mistral, DeepSeek and xAI. [`PRICES_SNAPSHOT_DATE`] says when it was
//! generated. No network calls; the lookup is a binary search over a static
//! array, so it costs nothing at startup in a WASM module.
//!
//! Each row carries the provider, the input and output prices, the prompt
//! cache read and write prices, and the context window. [`ModelPricing::cost`]
//! prices a [`Usage`] with all of them, so prompt-cached traffic (often 10% of
//! the list input price on a cache hit) is not overstated.
//!
//! ```rust
//! use llm_wasm::cost::{context_window, pricing_for_model};
//! use llm_wasm::types::Usage;
//!
//! let p = pricing_for_model("gpt-4o-mini").unwrap();
//! assert_eq!(p.input_per_million, 0.15);
//! assert!(pricing_for_model("openai/gpt-4o").is_ok()); // provider prefix is accepted
//!
//! // An Anthropic call that mostly hit the prompt cache:
//! let opus = pricing_for_model("claude-opus-5-5").unwrap();
//! assert_eq!(opus.provider, "anthropic");
//! let usage = Usage::new(2_000, 500).with_cache_read(100_000);
//! let usd = opus.cost(&usage); // 2k x $4/M + 100k x $0.20/M + 500 x $20/M
//! assert!((usd - 0.038).abs() < 1e-12);
//! assert_eq!(context_window("claude-opus-5-5"), Some(1_000_000));
//! ```

use crate::error::LlmWasmError;
use crate::types::Usage;

#[path = "../prices_data.rs"]
mod data;

/// Date the built-in price table was generated (YYYY-MM-DD).
pub const PRICES_SNAPSHOT_DATE: &str = data::SNAPSHOT_DATE;

/// USD per-million-token pricing for one model.
///
/// `#[non_exhaustive]` since 0.3: get one from [`pricing_for_model`], or build
/// your own with [`ModelPricing::new`] and the `with_*` setters.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ModelPricing {
    /// Provider name from the price table (`"anthropic"`, `"openai"`,
    /// `"gemini"`, ...); empty for pricing built with [`ModelPricing::new`].
    pub provider: &'static str,
    /// Cost per 1 000 000 input tokens, in USD.
    pub input_per_million: f64,
    /// Cost per 1 000 000 output tokens, in USD.
    pub output_per_million: f64,
    /// Cost per 1 000 000 input tokens read from the prompt cache, in USD.
    /// `None` when the provider lists no cache-read price.
    pub cache_read_per_million: Option<f64>,
    /// Cost per 1 000 000 input tokens written to the prompt cache, in USD.
    /// `None` when the provider lists no cache-write price.
    pub cache_write_per_million: Option<f64>,
    /// Context window (maximum input tokens), when known.
    pub max_input_tokens: Option<u32>,
}

impl ModelPricing {
    /// Pricing with only input and output rates (no provider, cache prices or
    /// context window), for a model missing from the table.
    pub fn new(input_per_million: f64, output_per_million: f64) -> Self {
        Self {
            provider: "",
            input_per_million,
            output_per_million,
            cache_read_per_million: None,
            cache_write_per_million: None,
            max_input_tokens: None,
        }
    }

    /// Set the provider name.
    pub fn with_provider(mut self, provider: &'static str) -> Self {
        self.provider = provider;
        self
    }

    /// Set the prompt cache read price per million tokens.
    pub fn with_cache_read(mut self, usd_per_million: f64) -> Self {
        self.cache_read_per_million = Some(usd_per_million);
        self
    }

    /// Set the prompt cache write price per million tokens.
    pub fn with_cache_write(mut self, usd_per_million: f64) -> Self {
        self.cache_write_per_million = Some(usd_per_million);
        self
    }

    /// Set the context window.
    pub fn with_max_input_tokens(mut self, tokens: u32) -> Self {
        self.max_input_tokens = Some(tokens);
        self
    }

    /// Compute the USD cost for plain input and output token counts (no
    /// prompt caching). Same as `cost(&Usage::new(input, output))`.
    pub fn cost_usd(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        self.cost(&Usage::new(u64::from(input_tokens), u64::from(output_tokens)))
    }

    /// Compute the USD cost of a call from its [`Usage`].
    ///
    /// - `input` at the input price; `output` and `reasoning` at the output price;
    /// - `cache_read` at the cache-read price, or the input price when none is listed;
    /// - `cache_write` at the cache-write price, or the input price when none is
    ///   listed (providers without a write surcharge bill those tokens as input).
    pub fn cost(&self, usage: &Usage) -> f64 {
        let per = |tokens: u64, usd_per_million: f64| tokens as f64 * usd_per_million / 1_000_000.0;
        per(usage.input, self.input_per_million)
            + per(usage.cache_read, self.cache_read_per_million.unwrap_or(self.input_per_million))
            + per(usage.cache_write, self.cache_write_per_million.unwrap_or(self.input_per_million))
            + per(usage.output, self.output_per_million)
            + per(usage.reasoning, self.output_per_million)
    }
}

fn from_row(row: &data::Row) -> ModelPricing {
    let &(_, provider, input, output, cache_read, cache_write, max_input) = row;
    let listed = |v: f64| (!v.is_nan()).then_some(v);
    ModelPricing {
        provider,
        input_per_million: input,
        output_per_million: output,
        cache_read_per_million: listed(cache_read),
        cache_write_per_million: listed(cache_write),
        max_input_tokens: (max_input != 0).then_some(max_input),
    }
}

fn lookup(model: &str) -> Option<ModelPricing> {
    let rows = data::ROWS;
    let find = |name: &str| rows.binary_search_by(|row| row.0.cmp(name)).ok().map(|i| from_row(&rows[i]));
    find(model).or_else(|| model.split_once('/').and_then(|(_, bare)| find(bare)))
}

/// Look up pricing for a model in the built-in table.
///
/// Accepts the provider's model name (`gpt-4o`, `claude-sonnet-4-5`) or the
/// same name with a `provider/` prefix.
///
/// # Errors
/// [`LlmWasmError::UnknownModel`] if the model is not in the table.
pub fn pricing_for_model(model: &str) -> Result<ModelPricing, LlmWasmError> {
    lookup(model).ok_or_else(|| LlmWasmError::UnknownModel(model.to_string()))
}

/// Context window (maximum input tokens) of a model in the built-in table.
///
/// `None` if the model is unknown or the table does not list a window.
///
/// ```
/// assert_eq!(llm_wasm::cost::context_window("gpt-4o"), Some(128_000));
/// assert_eq!(llm_wasm::cost::context_window("no-such-model"), None);
/// ```
pub fn context_window(model: &str) -> Option<u32> {
    lookup(model).and_then(|p| p.max_input_tokens)
}

/// Names of every model in the built-in price table, sorted.
pub fn known_models() -> impl Iterator<Item = &'static str> {
    data::ROWS.iter().map(|row| row.0)
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
#[derive(Debug, Default)]
pub struct CostLedger {
    total_usd: f64,
    entries: u32,
    budget_usd: Option<f64>,
}

impl CostLedger {
    /// Create an unbounded ledger (no budget limit).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a ledger with a USD budget ceiling.
    ///
    /// # Arguments
    /// * `budget_usd`: maximum allowed spend in USD
    pub fn with_budget(budget_usd: f64) -> Self {
        Self { budget_usd: Some(budget_usd), ..Self::default() }
    }

    /// Record a completed request and its token usage. Returns the new total.
    ///
    /// # Errors
    /// Returns [`LlmWasmError::UnknownModel`] for models not in the table.
    /// Returns [`LlmWasmError::BudgetExceeded`] if recording this usage would
    /// push total spend over the budget (nothing is recorded in that case).
    pub fn record(&mut self, model: &str, input_tokens: u32, output_tokens: u32) -> Result<f64, LlmWasmError> {
        let cost = pricing_for_model(model)?.cost_usd(input_tokens, output_tokens);
        self.record_usd(cost)
    }

    /// Record a completed call from its full [`Usage`] (prompt cache reads
    /// and writes, reasoning tokens), priced with [`ModelPricing::cost`].
    /// Returns the new total.
    ///
    /// ```
    /// use llm_wasm::cost::CostLedger;
    /// use llm_wasm::types::Usage;
    /// let mut ledger = CostLedger::with_budget(0.05);
    /// let total = ledger.record_usage("claude-opus-5-5", &Usage::new(2_000, 500).with_cache_read(100_000))?;
    /// assert!((total - 0.038).abs() < 1e-12);
    /// # Ok::<(), llm_wasm::LlmWasmError>(())
    /// ```
    ///
    /// # Errors
    /// [`LlmWasmError::UnknownModel`] for models not in the table (record
    /// those with [`CostLedger::record_usd`]), [`LlmWasmError::BudgetExceeded`]
    /// if it would cross the budget (nothing is recorded in that case).
    pub fn record_usage(&mut self, model: &str, usage: &Usage) -> Result<f64, LlmWasmError> {
        let cost = pricing_for_model(model)?.cost(usage);
        self.record_usd(cost)
    }

    /// Record a cost you already know in USD (for example one your provider
    /// reported, or a model missing from the table). Returns the new total.
    ///
    /// # Errors
    /// [`LlmWasmError::InvalidConfig`] for a negative or non-finite amount,
    /// [`LlmWasmError::BudgetExceeded`] if it would cross the budget.
    pub fn record_usd(&mut self, cost_usd: f64) -> Result<f64, LlmWasmError> {
        if !cost_usd.is_finite() || cost_usd < 0.0 {
            return Err(LlmWasmError::InvalidConfig {
                field: "cost_usd".into(),
                reason: format!("must be a non-negative number, got {cost_usd}"),
            });
        }
        let new_total = self.total_usd + cost_usd;
        if let Some(budget) = self.budget_usd {
            if new_total > budget {
                return Err(LlmWasmError::BudgetExceeded { used: new_total, limit: budget });
            }
        }
        self.total_usd = new_total;
        self.entries = self.entries.saturating_add(1);
        Ok(new_total)
    }

    /// Return the sum of all recorded costs in USD.
    pub fn total_usd(&self) -> f64 {
        self.total_usd
    }

    /// Budget left in USD, or `None` when no budget was set.
    pub fn remaining_usd(&self) -> Option<f64> {
        self.budget_usd.map(|b| (b - self.total_usd).max(0.0))
    }

    /// Return `true` if the total spend exceeds the configured budget.
    ///
    /// Always `false` when no budget was set.
    pub fn exceeded_budget(&self) -> bool {
        match self.budget_usd {
            Some(budget) => self.total_usd > budget,
            None => false,
        }
    }

    /// Number of recorded entries.
    pub fn entry_count(&self) -> u32 {
        self.entries
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
        assert!(matches!(result, Err(LlmWasmError::UnknownModel(m)) if m == "unknown-model-xyz"));
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
        let p = ModelPricing::new(3.0, 15.0);
        let cost = p.cost_usd(1_000_000, 1_000_000);
        assert!((cost - 18.0).abs() < 1e-9);
    }

    #[test]
    fn test_rows_map_to_pricing_fields() {
        let opus = pricing_for_model("claude-opus-5-5").unwrap();
        assert_eq!(opus.provider, "anthropic");
        assert_eq!((opus.input_per_million, opus.output_per_million), (4.0, 20.0));
        assert_eq!((opus.cache_read_per_million, opus.cache_write_per_million), (Some(0.2), Some(5.0)));
        assert_eq!(opus.max_input_tokens, Some(1_000_000));
        // NaN in the table means "not listed"
        let gpt = pricing_for_model("gpt-4o").unwrap();
        assert_eq!((gpt.provider, gpt.cache_read_per_million, gpt.cache_write_per_million), ("openai", Some(1.25), None));
        // every row converts without NaN leaking out
        for m in known_models() {
            let p = pricing_for_model(m).unwrap();
            assert!(p.input_per_million.is_finite() && p.output_per_million.is_finite(), "{m}");
            assert!(p.cache_read_per_million.is_none_or(f64::is_finite), "{m}");
            assert!(p.cache_write_per_million.is_none_or(f64::is_finite), "{m}");
            assert!(!p.provider.is_empty(), "{m}");
        }
    }

    #[test]
    fn test_cost_fallbacks_and_reasoning() {
        let plain = ModelPricing::new(2.0, 8.0);
        // no cache prices listed: cached tokens are billed as input
        let u = Usage::new(0, 0).with_cache_read(1_000_000).with_cache_write(1_000_000);
        assert!((plain.cost(&u) - 4.0).abs() < 1e-12);
        // reasoning is billed at the output price
        assert!((plain.cost(&Usage::new(0, 1_000_000).with_reasoning(1_000_000)) - 16.0).abs() < 1e-12);
        let cached = plain.with_cache_read(0.5).with_cache_write(2.5).with_provider("x").with_max_input_tokens(9);
        assert!((cached.cost(&u) - 3.0).abs() < 1e-12);
        assert_eq!((cached.provider, cached.max_input_tokens), ("x", Some(9)));
        assert_eq!(plain.cost(&Usage::default()), 0.0);
    }

    #[test]
    fn test_prices_match_published_rates() {
        // 0.1.x listed claude-opus-4-6 at $15/$75 and claude-haiku-4-5 at
        // $0.80/$4 per million tokens; Anthropic's prices are $5/$25 and $1/$5.
        let opus = pricing_for_model("claude-opus-4-6").unwrap();
        assert_eq!((opus.input_per_million, opus.output_per_million), (5.0, 25.0));
        let haiku = pricing_for_model("claude-haiku-4-5-20251001").unwrap();
        assert_eq!((haiku.input_per_million, haiku.output_per_million), (1.0, 5.0));
        let gpt = pricing_for_model("gpt-4o").unwrap();
        assert_eq!((gpt.input_per_million, gpt.output_per_million), (2.5, 10.0));
    }

    #[test]
    fn test_table_is_sorted_and_large() {
        let names: Vec<&str> = known_models().collect();
        assert!(names.len() > 100);
        assert!(names.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(PRICES_SNAPSHOT_DATE.len(), 10);
    }

    #[test]
    fn test_provider_prefix_and_other_vendors() {
        assert_eq!(pricing_for_model("anthropic/claude-sonnet-4-5").unwrap(), pricing_for_model("claude-sonnet-4-5").unwrap());
        assert!(known_models().any(|m| m.starts_with("gemini-")));
        assert!(known_models().any(|m| m.starts_with("mistral-")));
        assert!(pricing_for_model("nobody/nothing").is_err());
    }

    #[test]
    fn test_record_usd_and_remaining() {
        let mut ledger = CostLedger::with_budget(1.0);
        assert!((ledger.record_usd(0.4).unwrap() - 0.4).abs() < 1e-12);
        assert!(ledger.record_usd(-1.0).is_err());
        assert!(ledger.record_usd(f64::NAN).is_err());
        assert!(matches!(ledger.record_usd(0.7), Err(LlmWasmError::BudgetExceeded { .. })));
        assert!((ledger.remaining_usd().unwrap() - 0.6).abs() < 1e-12);
        assert_eq!(ledger.entry_count(), 1);
        assert_eq!(CostLedger::new().remaining_usd(), None);
    }
}
