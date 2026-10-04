//! JavaScript bindings (feature `js`), generated with
//! [wasm-bindgen](https://crates.io/crates/wasm-bindgen).
//!
//! Build for the browser or a worker with
//! `wasm-pack build --target web -- --features js`, or by hand with
//! `cargo build --release --target wasm32-unknown-unknown --features js`
//! followed by `wasm-bindgen --target web` (or `--target nodejs`).
//!
//! ```js
//! import init, { estimateCost, extractJson, renderTemplate, Ledger, Retry, Cache } from "./llm_wasm.js";
//! await init();
//! estimateCost("gpt-4o-mini", 1000, 500);        // 0.00045
//! extractJson('Sure! {"a": 1}');                 // { a: 1 }
//! renderTemplate("Hi {{name}}", { name: "Ada" }); // "Hi Ada"
//! const ledger = new Ledger(1.0);                 // $1 budget
//! ledger.record("gpt-4o", 1200, 300);             // throws once over budget
//! const retry = new Retry(3, 200, 5000);
//! if (retry.shouldRetry(1, 429)) await sleep(retry.delay(1, Math.random(), res.headers.get("retry-after")));
//! ```
//!
//! Every function that can fail throws a JavaScript `Error` with the message
//! of the matching [`LlmWasmError`].

use std::collections::HashMap;

use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::cache::TtlCache;
use crate::cost::{known_models, pricing_for_model, CostLedger};
use crate::format::{JsonFormatter, MarkdownFormatter};
use crate::guard::{ContentGuard, GuardChain, LengthGuard};
use crate::retry::RetryPolicy;
use crate::routing::{Router, RoutingRule};
use crate::template::TemplateEngine;
use crate::types::ChatRequest;
use crate::LlmWasmError;

fn js_err(e: LlmWasmError) -> JsError {
    JsError::new(&e.to_string())
}

fn from_js<T: serde::de::DeserializeOwned>(v: JsValue, what: &str) -> Result<T, JsError> {
    serde_wasm_bindgen::from_value(v).map_err(|e| JsError::new(&format!("invalid {what}: {e}")))
}

fn to_js<T: Serialize>(v: &T) -> Result<JsValue, JsError> {
    v.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(|e| JsError::new(&e.to_string()))
}

/// Cost in USD of a call to `model` with the given token counts.
#[wasm_bindgen(js_name = estimateCost)]
pub fn estimate_cost(model: &str, input_tokens: u32, output_tokens: u32) -> Result<f64, JsError> {
    Ok(pricing_for_model(model).map_err(js_err)?.cost_usd(input_tokens, output_tokens))
}

/// Every model name in the built-in price table.
#[wasm_bindgen(js_name = knownModels)]
pub fn known_models_js() -> Vec<String> {
    known_models().map(str::to_string).collect()
}

/// The first valid JSON object or array found in `text`, as a JavaScript value.
#[wasm_bindgen(js_name = extractJson)]
pub fn extract_json(text: &str) -> Result<JsValue, JsError> {
    to_js(&JsonFormatter::extract_json(text).map_err(js_err)?)
}

/// The contents of the first Markdown code fence in `text` (or `text` itself).
#[wasm_bindgen(js_name = stripCodeFence)]
pub fn strip_code_fence(text: &str) -> String {
    MarkdownFormatter::strip_code_fence(text)
}

/// Render a `{{variable}}` / `{{>partial}}` template. `vars` and `partials`
/// are plain objects of strings; `partials` may be omitted.
#[wasm_bindgen(js_name = renderTemplate)]
pub fn render_template(template: &str, vars: JsValue, partials: JsValue) -> Result<String, JsError> {
    let vars: HashMap<String, String> = from_js(vars, "vars")?;
    let mut engine = TemplateEngine::new();
    if !partials.is_undefined() && !partials.is_null() {
        let partials: HashMap<String, String> = from_js(partials, "partials")?;
        for (name, body) in &partials {
            engine.register_partial(name, body);
        }
    }
    engine.render(template, &vars).map_err(js_err)
}

/// Throw if `request` (`{model, messages: [{role, content}], ...}`) contains a
/// blocked term (case-insensitive) or more than `maxChars` characters.
#[wasm_bindgen(js_name = checkRequest)]
pub fn check_request(request: JsValue, blocklist: Vec<String>, max_chars: usize) -> Result<(), JsError> {
    let request: ChatRequest = from_js(request, "request")?;
    GuardChain::new()
        .add(ContentGuard::new(blocklist))
        .add(LengthGuard::new(max_chars))
        .check(&request)
        .map(|_| ())
        .map_err(js_err)
}

/// Pick a model for `request` from `rules` (an array of
/// `{condition, target_model}` as serialized by [`RoutingRule`]), or `fallback`.
#[wasm_bindgen(js_name = routeModel)]
pub fn route_model(request: JsValue, rules: JsValue, fallback: &str) -> Result<String, JsError> {
    let request: ChatRequest = from_js(request, "request")?;
    let rules: Vec<RoutingRule> = from_js(rules, "rules")?;
    let mut router = Router::new(fallback);
    for rule in rules {
        router.add_rule(rule);
    }
    Ok(router.route(&request).to_string())
}

/// A spend ledger with an optional USD budget.
#[wasm_bindgen]
pub struct Ledger {
    inner: CostLedger,
}

#[wasm_bindgen]
impl Ledger {
    /// `new Ledger()` for no limit, `new Ledger(1.5)` for a $1.50 budget.
    #[wasm_bindgen(constructor)]
    pub fn new(budget_usd: Option<f64>) -> Ledger {
        Ledger { inner: budget_usd.map_or_else(CostLedger::new, CostLedger::with_budget) }
    }

    /// Record a call; returns the new total. Throws for an unknown model or
    /// when the budget would be exceeded (nothing is recorded then).
    pub fn record(&mut self, model: &str, input_tokens: u32, output_tokens: u32) -> Result<f64, JsError> {
        self.inner.record(model, input_tokens, output_tokens).map_err(js_err)
    }

    /// Record a known dollar amount; returns the new total.
    #[wasm_bindgen(js_name = recordUsd)]
    pub fn record_usd(&mut self, cost_usd: f64) -> Result<f64, JsError> {
        self.inner.record_usd(cost_usd).map_err(js_err)
    }

    /// Total spent so far in USD.
    #[wasm_bindgen(getter)]
    pub fn total(&self) -> f64 {
        self.inner.total_usd()
    }

    /// Budget left in USD, or `undefined` without a budget.
    #[wasm_bindgen(getter)]
    pub fn remaining(&self) -> Option<f64> {
        self.inner.remaining_usd()
    }
}

/// Retry policy: which failures to retry and how long to wait.
#[wasm_bindgen]
pub struct Retry {
    inner: RetryPolicy,
}

#[wasm_bindgen]
impl Retry {
    /// `new Retry(maxAttempts, baseDelayMs, maxDelayMs)`. Throws on invalid values.
    #[wasm_bindgen(constructor)]
    pub fn new(max_attempts: u32, base_delay_ms: u32, max_delay_ms: u32) -> Result<Retry, JsError> {
        Ok(Retry { inner: RetryPolicy::new(max_attempts, base_delay_ms, max_delay_ms).map_err(js_err)? })
    }

    /// True if attempt number `attempt` (1-based) failed with a status worth retrying.
    #[wasm_bindgen(js_name = shouldRetry)]
    pub fn should_retry(&self, attempt: u32, status: u16) -> bool {
        self.inner.should_retry(attempt, status)
    }

    /// Milliseconds to wait. Pass `Math.random()` and the response's
    /// `Retry-After` header (or `null`).
    pub fn delay(&self, attempt: u32, random_unit: f64, retry_after: Option<String>) -> u32 {
        self.inner.delay_for_response(attempt, retry_after.as_deref(), random_unit)
    }
}

/// A response cache with a time-to-live and an optional size cap.
#[wasm_bindgen]
pub struct Cache {
    inner: TtlCache<String>,
}

#[wasm_bindgen]
impl Cache {
    /// `new Cache(ttlMs)` or `new Cache(ttlMs, maxEntries)`.
    #[wasm_bindgen(constructor)]
    pub fn new(ttl_ms: f64, max_entries: Option<usize>) -> Cache {
        let inner = match max_entries {
            Some(n) => TtlCache::with_capacity(ttl_ms, n),
            None => TtlCache::new(ttl_ms),
        };
        Cache { inner }
    }

    /// The cached value for `key`, or `undefined` if missing or expired.
    /// Pass the current time, for example `Date.now()`.
    pub fn get(&mut self, key: &str, now_ms: f64) -> Option<String> {
        self.inner.get(&key.to_string(), now_ms)
    }

    /// Store `value` under `key` at time `now_ms`.
    pub fn set(&mut self, key: &str, value: &str, now_ms: f64) {
        self.inner.set(key.to_string(), value.to_string(), now_ms);
    }

    /// Number of entries held (including expired ones not yet dropped).
    #[wasm_bindgen(getter)]
    pub fn size(&self) -> usize {
        self.inner.len()
    }
}

/// Like `extractJson`, but repairs almost-JSON (trailing commas, single
/// quotes, Python literals, cut-off output). Needs the `repair` feature.
#[cfg(feature = "repair")]
#[wasm_bindgen(js_name = extractJsonLenient)]
pub fn extract_json_lenient(text: &str) -> Result<JsValue, JsError> {
    to_js(&JsonFormatter::extract_json_lenient(text).map_err(js_err)?)
}

/// Render a Hugging Face chat template (the `chat_template` string from
/// `tokenizer_config.json`) for `messages` (`[{role, content}]`). Needs the
/// `jinja` feature.
#[cfg(feature = "jinja")]
#[wasm_bindgen(js_name = renderChatTemplate)]
pub fn render_chat_template(
    template: &str,
    messages: JsValue,
    add_generation_prompt: bool,
    bos_token: Option<String>,
    eos_token: Option<String>,
) -> Result<String, JsError> {
    let messages: Vec<crate::types::ChatMessage> = from_js(messages, "messages")?;
    crate::template::render_chat_template(
        template,
        &messages,
        add_generation_prompt,
        bos_token.as_deref().unwrap_or(""),
        eos_token.as_deref().unwrap_or(""),
    )
    .map_err(js_err)
}

/// `text` with API keys, private keys, emails and card numbers replaced by
/// `[REDACTED:<kind>]`. Needs the `secrets` feature.
#[cfg(feature = "secrets")]
#[wasm_bindgen(js_name = redactSecrets)]
pub fn redact_secrets(text: &str) -> String {
    crate::secrets::SecretGuard::redact().redact_text(text)
}

/// Decodes a streamed (SSE) chat response chunk by chunk. Needs the `stream`
/// feature.
///
/// ```js
/// const dec = new StreamDecoder();
/// for await (const piece of response.body) {
///   for (const delta of dec.push(piece)) output.textContent += delta;
/// }
/// ```
#[cfg(feature = "stream")]
#[wasm_bindgen(js_name = StreamDecoder)]
pub struct JsStreamDecoder {
    inner: crate::stream::StreamDecoder,
}

#[cfg(feature = "stream")]
#[wasm_bindgen(js_class = StreamDecoder)]
impl JsStreamDecoder {
    /// A new decoder.
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsStreamDecoder {
        JsStreamDecoder { inner: crate::stream::StreamDecoder::new() }
    }

    /// Feed the next bytes of the body (a `Uint8Array`); returns the text
    /// deltas they completed. Throws on malformed events or provider errors.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<String>, JsError> {
        let chunks = self.inner.push(bytes).map_err(js_err)?;
        Ok(chunks.into_iter().map(|c| c.delta).filter(|d| !d.is_empty()).collect())
    }

    /// True once the stream ended (`[DONE]` or `message_stop`).
    #[wasm_bindgen(getter)]
    pub fn finished(&self) -> bool {
        self.inner.is_finished()
    }

    /// All text received so far.
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.inner.text().to_string()
    }

    /// `{input_tokens, output_tokens}` if the provider reported usage, else `undefined`.
    #[wasm_bindgen(getter)]
    pub fn usage(&self) -> Result<JsValue, JsError> {
        match self.inner.usage() {
            Some(u) => to_js(&u),
            None => Ok(JsValue::UNDEFINED),
        }
    }
}

#[cfg(feature = "stream")]
impl Default for JsStreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}
