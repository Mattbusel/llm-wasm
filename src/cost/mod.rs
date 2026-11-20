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
//!
//! # Feature `llm-prices`
//!
//! With the `llm-prices` feature, prices come from the
//! [llm-prices](https://crates.io/crates/llm-prices) 1.2 table first: every
//! row is dated and cites the provider's pricing page, model ids are resolved
//! provider-aware (dated snapshots, `-latest`, Bedrock and Vertex spellings),
//! and models with a long-context tier are billed at that tier by
//! [`cost_for_model`], [`CostLedger::record`] and
//! [`CostLedger::record_usage`] when the prompt is over the threshold.
//! Models llm-prices does not list are still priced from the built-in table,
//! and [`context_window`] and [`known_models`] always use the built-in table
//! (llm-prices carries no context windows).
//!
//! The built-in table stays compiled in either way: llm-prices lists fewer
//! models (136 rows against 232) and no context windows, so dropping the
//! table would make known models unknown. That is why the feature is off by
//! default: it adds about 50 KB to a stripped, `opt-level = "z"`, LTO
//! release build of the `js` module (599 KB to 651 KB), for the dated,
//! source-cited rows and the long-context tiers.

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

    /// Pricing from an [llm-prices](https://crates.io/crates/llm-prices) row
    /// (feature `llm-prices`).
    ///
    /// The cache-write price is the 5-minute lifetime's (Anthropic's default),
    /// or the row's first listed lifetime when it has no 5-minute price. An
    /// embedding row without an output price gets an output price of 0.
    /// `max_input_tokens` is `None`: llm-prices lists no context windows.
    /// Long-context tiers do not fit in a flat `ModelPricing`; use
    /// [`cost_for_model`] to have them applied.
    ///
    /// Returns `None` when the row lists no input price.
    ///
    /// ```
    /// # use llm_wasm::cost::ModelPricing;
    /// let row = llm_prices::lookup("claude-sonnet-4-6").unwrap();
    /// let p = ModelPricing::from_llm_prices(row).unwrap();
    /// assert_eq!((p.input_per_million, p.output_per_million), (3.0, 15.0));
    /// assert_eq!(p.provider, "anthropic");
    /// ```
    #[cfg(feature = "llm-prices")]
    #[cfg_attr(docsrs, doc(cfg(feature = "llm-prices")))]
    pub fn from_llm_prices(price: &llm_prices::Price) -> Option<Self> {
        Some(Self {
            provider: price.provider,
            input_per_million: price.input?,
            output_per_million: price.output.unwrap_or(0.0),
            cache_read_per_million: price.cache_read,
            cache_write_per_million: price
                .cache_write("5m")
                .or_else(|| price.cache_write.first().map(|&(_, usd)| usd)),
            max_input_tokens: None,
        })
    }

    /// Compute the USD cost for plain input and output token counts (no
    /// prompt caching). Same as `cost(&Usage::new(input, output))`.
    pub fn cost_usd(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        self.cost(&Usage::new(
            u64::from(input_tokens),
            u64::from(output_tokens),
        ))
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
            + per(
                usage.cache_read,
                self.cache_read_per_million
                    .unwrap_or(self.input_per_million),
            )
            + per(
                usage.cache_write,
                self.cache_write_per_million
                    .unwrap_or(self.input_per_million),
            )
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

/// The built-in table's row for `model`.
fn builtin(model: &str) -> Option<ModelPricing> {
    let rows = data::ROWS;
    let find = |name: &str| {
        rows.binary_search_by(|row| row.0.cmp(name))
            .ok()
            .map(|i| from_row(&rows[i]))
    };
    find(model).or_else(|| model.split_once('/').and_then(|(_, bare)| find(bare)))
}

/// The llm-prices row for `model`, when it lists one with an input price,
/// with the context window filled in from the built-in table.
#[cfg(feature = "llm-prices")]
fn llm_prices_row(model: &str) -> Option<(&'static llm_prices::Price, ModelPricing)> {
    let row = llm_prices::lookup(model)?;
    let mut pricing = ModelPricing::from_llm_prices(row)?;
    pricing.max_input_tokens = builtin(model).and_then(|p| p.max_input_tokens);
    Some((row, pricing))
}

fn lookup(model: &str) -> Option<ModelPricing> {
    #[cfg(feature = "llm-prices")]
    if let Some((_, pricing)) = llm_prices_row(model) {
        return Some(pricing);
    }
    builtin(model)
}

/// Look up pricing for a model.
///
/// Accepts the provider's model name (`gpt-4o`, `claude-sonnet-4-5`) or the
/// same name with a `provider/` prefix. Uses the built-in table; with the
/// `llm-prices` feature, llm-prices first (see the
/// [module docs](self#feature-llm-prices)).
///
/// # Errors
/// [`LlmWasmError::UnknownModel`] if no table lists the model.
pub fn pricing_for_model(model: &str) -> Result<ModelPricing, LlmWasmError> {
    lookup(model).ok_or_else(|| LlmWasmError::UnknownModel(model.to_string()))
}

/// The USD cost of one call to `model` from its [`Usage`].
///
/// Without the `llm-prices` feature this is
/// `pricing_for_model(model)?.cost(usage)`. With it, a model llm-prices lists
/// is priced by llm-prices itself, so a prompt over the model's long-context
/// threshold (input, cache reads and cache writes together) bills at the
/// long-context rates; `cache_write` counts as 5-minute cache writes and
/// `reasoning` bills at the output rate. A row that cannot price the usage
/// that way (cache writes on a row with no 5-minute write price) falls back
/// to its flat [`ModelPricing`].
///
/// [`CostLedger::record`] and [`CostLedger::record_usage`] price calls with
/// this function.
///
/// ```
/// use llm_wasm::cost::cost_for_model;
/// use llm_wasm::types::Usage;
/// let usd = cost_for_model("gpt-4o", &Usage::new(1_000_000, 100_000))?;
/// assert!((usd - 3.5).abs() < 1e-9); // $2.50 + $1.00
/// # Ok::<(), llm_wasm::LlmWasmError>(())
/// ```
///
/// # Errors
/// [`LlmWasmError::UnknownModel`] if no table lists the model.
pub fn cost_for_model(model: &str, usage: &Usage) -> Result<f64, LlmWasmError> {
    #[cfg(feature = "llm-prices")]
    if let Some((row, pricing)) = llm_prices_row(model) {
        let lp = llm_prices::Usage {
            input: usage.input,
            output: usage.output.saturating_add(usage.reasoning),
            cache_read: usage.cache_read,
            cache_write_5m: usage.cache_write,
            ..llm_prices::Usage::default()
        };
        return Ok(row
            .cost_usage(&lp)
            .map_or_else(|| pricing.cost(usage), |c| c.total));
    }
    Ok(pricing_for_model(model)?.cost(usage))
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
    builtin(model).and_then(|p| p.max_input_tokens)
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
        Self {
            budget_usd: Some(budget_usd),
            ..Self::default()
        }
    }

    /// Record a completed request and its token usage. Returns the new total.
    ///
    /// # Errors
    /// Returns [`LlmWasmError::UnknownModel`] for models not in the table.
    /// Returns [`LlmWasmError::BudgetExceeded`] if recording this usage would
    /// push total spend over the budget (nothing is recorded in that case).
    pub fn record(
        &mut self,
        model: &str,
        input_tokens: u32,
        output_tokens: u32,
    ) -> Result<f64, LlmWasmError> {
        let cost = cost_for_model(
            model,
            &Usage::new(u64::from(input_tokens), u64::from(output_tokens)),
        )?;
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
        let cost = cost_for_model(model, usage)?;
        self.record_usd(cost)
    }

    /// Record a cost you already know in USD (for example one your provider
    /// reported, or a model missing from the table). Returns the new total.
    ///
    /// # Errors
    /// [`LlmWasmError::InvalidConfig`] for a negative or non-finite amount,
    /// [`LlmWasmError::BudgetExceeded`] if it would cross the budget.
    pub fn record_usd(&mut self, cost_usd: f64) -> Result<f64, LlmWasmError> {
        check_cost(cost_usd)?;
        let new_total = self.total_usd + cost_usd;
        if let Some(budget) = self.budget_usd {
            if new_total > budget {
                return Err(LlmWasmError::BudgetExceeded {
                    used: new_total,
                    limit: budget,
                });
            }
        }
        self.total_usd = new_total;
        self.entries = self.entries.saturating_add(1);
        Ok(new_total)
    }

    /// Record a cost that has already been paid, even when it takes the
    /// total past the budget. Returns the new total.
    ///
    /// [`record_usd`](Self::record_usd) refuses an amount that would cross
    /// the budget, which is the right check before a request is sent. Once
    /// the provider has answered, the money is spent: book it here so the
    /// ledger matches the bill, then check
    /// [`exceeded_budget`](Self::exceeded_budget) or
    /// [`remaining_usd`](Self::remaining_usd) before the next call. The rig
    /// hook (`rig::CostLedgerHook`, feature `rig`) books turns this way.
    ///
    /// ```
    /// use llm_wasm::cost::CostLedger;
    /// let mut ledger = CostLedger::with_budget(0.05);
    /// ledger.record_paid_usd(0.04)?;
    /// assert!((ledger.record_paid_usd(0.03)? - 0.07).abs() < 1e-12);
    /// assert!(ledger.exceeded_budget());
    /// assert_eq!(ledger.remaining_usd(), Some(0.0));
    /// # Ok::<(), llm_wasm::LlmWasmError>(())
    /// ```
    ///
    /// # Errors
    /// [`LlmWasmError::InvalidConfig`] for a negative or non-finite amount
    /// (nothing is recorded).
    pub fn record_paid_usd(&mut self, cost_usd: f64) -> Result<f64, LlmWasmError> {
        check_cost(cost_usd)?;
        self.total_usd += cost_usd;
        self.entries = self.entries.saturating_add(1);
        Ok(self.total_usd)
    }

    /// The budget ceiling in USD, or `None` when no budget was set.
    pub fn budget_usd(&self) -> Option<f64> {
        self.budget_usd
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

fn check_cost(cost_usd: f64) -> Result<(), LlmWasmError> {
    if cost_usd.is_finite() && cost_usd >= 0.0 {
        return Ok(());
    }
    Err(LlmWasmError::InvalidConfig {
        field: "cost_usd".into(),
        reason: format!("must be a non-negative number, got {cost_usd}"),
    })
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
        ledger
            .record("claude-opus-4-6", 100_000_000, 100_000_000)
            .unwrap();
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
            assert!(
                pricing_for_model(model).is_ok(),
                "expected pricing for {model}"
            );
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
        assert_eq!(
            (opus.input_per_million, opus.output_per_million),
            (4.0, 20.0)
        );
        assert_eq!(
            (opus.cache_read_per_million, opus.cache_write_per_million),
            (Some(0.2), Some(5.0))
        );
        assert_eq!(opus.max_input_tokens, Some(1_000_000));
        // NaN in the table means "not listed"
        let gpt = pricing_for_model("gpt-4o").unwrap();
        assert_eq!(
            (
                gpt.provider,
                gpt.cache_read_per_million,
                gpt.cache_write_per_million
            ),
            ("openai", Some(1.25), None)
        );
        // every row converts without NaN leaking out
        for m in known_models() {
            let p = pricing_for_model(m).unwrap();
            assert!(
                p.input_per_million.is_finite() && p.output_per_million.is_finite(),
                "{m}"
            );
            assert!(p.cache_read_per_million.is_none_or(f64::is_finite), "{m}");
            assert!(p.cache_write_per_million.is_none_or(f64::is_finite), "{m}");
            assert!(!p.provider.is_empty(), "{m}");
        }
    }

    #[test]
    fn test_cost_fallbacks_and_reasoning() {
        let plain = ModelPricing::new(2.0, 8.0);
        // no cache prices listed: cached tokens are billed as input
        let u = Usage::new(0, 0)
            .with_cache_read(1_000_000)
            .with_cache_write(1_000_000);
        assert!((plain.cost(&u) - 4.0).abs() < 1e-12);
        // reasoning is billed at the output price
        assert!(
            (plain.cost(&Usage::new(0, 1_000_000).with_reasoning(1_000_000)) - 16.0).abs() < 1e-12
        );
        let cached = plain
            .with_cache_read(0.5)
            .with_cache_write(2.5)
            .with_provider("x")
            .with_max_input_tokens(9);
        assert!((cached.cost(&u) - 3.0).abs() < 1e-12);
        assert_eq!((cached.provider, cached.max_input_tokens), ("x", Some(9)));
        assert_eq!(plain.cost(&Usage::default()), 0.0);
    }

    #[test]
    fn test_prices_match_published_rates() {
        // 0.1.x listed claude-opus-4-6 at $15/$75 and claude-haiku-4-5 at
        // $0.80/$4 per million tokens; Anthropic's prices are $5/$25 and $1/$5.
        let opus = pricing_for_model("claude-opus-4-6").unwrap();
        assert_eq!(
            (opus.input_per_million, opus.output_per_million),
            (5.0, 25.0)
        );
        let haiku = pricing_for_model("claude-haiku-4-5-20251001").unwrap();
        assert_eq!(
            (haiku.input_per_million, haiku.output_per_million),
            (1.0, 5.0)
        );
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
        assert_eq!(
            pricing_for_model("anthropic/claude-sonnet-4-5").unwrap(),
            pricing_for_model("claude-sonnet-4-5").unwrap()
        );
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
        assert!(matches!(
            ledger.record_usd(0.7),
            Err(LlmWasmError::BudgetExceeded { .. })
        ));
        assert!((ledger.remaining_usd().unwrap() - 0.6).abs() < 1e-12);
        assert_eq!(ledger.entry_count(), 1);
        assert_eq!(CostLedger::new().remaining_usd(), None);
    }

    #[test]
    fn test_record_paid_usd_books_past_the_budget() {
        let mut ledger = CostLedger::with_budget(0.10);
        assert_eq!(ledger.budget_usd(), Some(0.10));
        ledger.record_paid_usd(0.08).unwrap();
        // record_usd refuses what would cross the budget...
        assert!(matches!(
            ledger.record_usd(0.05),
            Err(LlmWasmError::BudgetExceeded { .. })
        ));
        // ...record_paid_usd books it, because it was already paid
        assert!((ledger.record_paid_usd(0.05).unwrap() - 0.13).abs() < 1e-12);
        assert!(ledger.exceeded_budget());
        assert_eq!(ledger.remaining_usd(), Some(0.0));
        assert_eq!(ledger.entry_count(), 2);
        assert!(ledger.record_paid_usd(-0.01).is_err());
        assert!(ledger.record_paid_usd(f64::INFINITY).is_err());
        assert_eq!(ledger.entry_count(), 2);
        assert_eq!(CostLedger::new().budget_usd(), None);
    }

    #[test]
    fn test_cost_for_model_matches_ledger_and_errors() {
        let u = Usage::new(3_000, 1_200)
            .with_cache_read(180_000)
            .with_cache_write(20_000);
        let usd = cost_for_model("claude-opus-5-5", &u).unwrap();
        assert!((usd - pricing_for_model("claude-opus-5-5").unwrap().cost(&u)).abs() < 1e-12);
        let mut ledger = CostLedger::new();
        assert!((ledger.record_usage("claude-opus-5-5", &u).unwrap() - usd).abs() < 1e-12);
        assert!(matches!(
            cost_for_model("no-such-model", &u),
            Err(LlmWasmError::UnknownModel(_))
        ));
    }

    #[cfg(feature = "llm-prices")]
    mod llm_prices_source {
        use super::*;

        #[test]
        fn long_context_tier_is_billed_above_the_threshold() {
            // gpt-5.4: $2.50 / $15 per 1M, $5 / $22.50 over 272k prompt tokens.
            let long = Usage::new(300_000, 1_000);
            let usd = cost_for_model("gpt-5.4", &long).unwrap();
            assert!((usd - (1.5 + 0.0225)).abs() < 1e-12, "{usd}");
            let flat = pricing_for_model("gpt-5.4").unwrap().cost(&long);
            assert!((flat - (0.75 + 0.015)).abs() < 1e-12, "{flat}");
            // the ledger uses the tiered price
            let mut ledger = CostLedger::new();
            assert!((ledger.record("gpt-5.4", 300_000, 1_000).unwrap() - usd).abs() < 1e-12);
            // below the threshold both agree
            let short = Usage::new(1_000, 1_000).with_reasoning(500);
            assert!(
                (cost_for_model("gpt-5.4", &short).unwrap()
                    - pricing_for_model("gpt-5.4").unwrap().cost(&short))
                .abs()
                    < 1e-12
            );
        }

        #[test]
        fn rows_keep_the_builtin_context_window() {
            let p = pricing_for_model("gpt-5.4").unwrap();
            assert_eq!(p.max_input_tokens, Some(1_050_000));
            assert_eq!(context_window("gpt-5.4"), Some(1_050_000));
            // llm-prices resolves provider prefixes and dated ids itself
            assert!(pricing_for_model("anthropic/claude-haiku-4-5-20251001").is_ok());
        }

        #[test]
        fn models_llm_prices_lacks_fall_back_to_the_builtin_table() {
            assert!(llm_prices::lookup("open-mistral-nemo").is_none());
            let p = pricing_for_model("open-mistral-nemo").unwrap();
            assert_eq!((p.provider, p.input_per_million), ("mistral", 0.3));
            assert!(cost_for_model("open-mistral-nemo", &Usage::new(1_000_000, 0)).is_ok());
        }

        #[test]
        fn cache_writes_on_rows_without_a_5m_price_use_the_flat_rate() {
            // gpt-6-sol lists 30m and 12h writes only; llm-prices cannot price
            // a 5m write there, so the first listed lifetime ($2.50) is used.
            let u = Usage::new(0, 0).with_cache_write(1_000_000);
            assert!((cost_for_model("gpt-6-sol", &u).unwrap() - 2.5).abs() < 1e-12);
        }

        #[test]
        fn from_llm_prices_maps_fields() {
            let haiku =
                ModelPricing::from_llm_prices(llm_prices::lookup("claude-haiku-4-5").unwrap())
                    .unwrap();
            assert_eq!(
                (haiku.cache_read_per_million, haiku.cache_write_per_million),
                (Some(0.1), Some(1.25))
            );
            assert_eq!(haiku.max_input_tokens, None);
            let embed = ModelPricing::from_llm_prices(
                llm_prices::lookup("text-embedding-3-small").unwrap(),
            )
            .unwrap();
            assert_eq!(embed.output_per_million, 0.0);
            let mut no_input = *llm_prices::lookup("gpt-4o").unwrap();
            no_input.input = None;
            assert!(ModelPricing::from_llm_prices(&no_input).is_none());
        }
    }
}
