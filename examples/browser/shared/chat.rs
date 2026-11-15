//! The part of the browser examples that is the same in every framework: call
//! an OpenAI-compatible `/chat/completions` endpoint from wasm with
//!
//! - retries on 429 and 5xx, waiting real browser time (`setTimeout` through
//!   gloo-timers) for the jittered backoff or the server's `Retry-After`;
//! - a TTL cache keyed by model and messages, so asking the same thing twice
//!   within five minutes costs nothing;
//! - cost tracking: the response's `usage` (cache reads included) priced for
//!   the model and booked in a `CostLedger` with a budget, and no request
//!   sent once the budget is spent.
//!
//! The Leptos and Dioxus apps include this file with `#[path]` and only add
//! the UI around [`ask`].

use std::cell::RefCell;
use std::rc::Rc;

use gloo_net::http::Request;
use gloo_timers::future::TimeoutFuture;
use llm_wasm::cache::{cache_key, TtlCache};
use llm_wasm::cost::{cost_for_model, CostLedger};
use llm_wasm::format::MarkdownFormatter;
use llm_wasm::retry::RetryPolicy;
use llm_wasm::types::{ChatMessage, ChatRequest, Role, Usage};

/// Where to send requests. The key stays in this tab's memory. Do not ship a
/// real provider key to browsers: point `base_url` at your own proxy, or at a
/// local OpenAI-compatible server (Ollama, LM Studio, llama.cpp) for testing.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self { base_url: "https://api.openai.com/v1".into(), api_key: String::new(), model: "gpt-4o-mini".into() }
    }
}

/// State that lives as long as the page: the answer cache and the ledger.
pub struct Session {
    cache: TtlCache,
    ledger: CostLedger,
    retry: RetryPolicy,
}

impl Session {
    /// Five-minute cache of up to 200 answers, and a $0.50 budget.
    pub fn new() -> Self {
        Self {
            cache: TtlCache::with_capacity(5.0 * 60_000.0, 200),
            ledger: CostLedger::with_budget(0.50),
            retry: RetryPolicy::new(4, 500, 8_000).unwrap_or_else(|_| RetryPolicy::exponential()),
        }
    }

    pub fn spent_usd(&self) -> f64 {
        self.ledger.total_usd()
    }

    pub fn budget_usd(&self) -> f64 {
        self.ledger.budget_usd().unwrap_or(f64::INFINITY)
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

/// One answer and what it took to get it.
#[derive(Clone, Debug, PartialEq)]
pub struct Reply {
    pub text: String,
    /// Served from the cache: no request was sent.
    pub cached: bool,
    /// HTTP attempts made (0 when cached).
    pub attempts: u32,
    pub usage: Usage,
    /// `None` when the model is in no price table.
    pub cost_usd: Option<f64>,
}

/// Ask `prompt`. The session is borrowed only between awaits, so the UI can
/// read it (for the spend line) while a request is in flight.
pub async fn ask(settings: &Settings, session: &Rc<RefCell<Session>>, prompt: &str) -> Result<Reply, String> {
    let request = ChatRequest::new(settings.model.clone(), vec![ChatMessage::new(Role::User, prompt)]);
    let messages = serde_json::to_string(&request.messages).map_err(|e| e.to_string())?;
    let key = cache_key(&settings.model, &messages);

    let retry = {
        let mut s = session.borrow_mut();
        if let Some(text) = s.cache.get(&key, js_sys::Date::now()) {
            return Ok(Reply { text, cached: true, attempts: 0, usage: Usage::default(), cost_usd: Some(0.0) });
        }
        if s.ledger.exceeded_budget() || s.ledger.remaining_usd() == Some(0.0) {
            return Err(format!("Budget spent: ${:.4} of ${:.2}. No request sent.", s.spent_usd(), s.budget_usd()));
        }
        s.retry.clone()
    };

    let body = serde_json::to_string(&request).map_err(|e| e.to_string())?;
    let (json, attempts) = post_with_retry(settings, &retry, &body).await?;

    let text = json["choices"][0]["message"]["content"].as_str().unwrap_or_default().trim().to_string();
    let text = MarkdownFormatter::strip_code_fence(&text);
    let usage = Usage::from_json(&json).unwrap_or_default();
    let cost_usd = cost_for_model(&settings.model, &usage).ok();

    let mut s = session.borrow_mut();
    if let Some(usd) = cost_usd {
        // Paid already, so it is booked even if it crosses the budget.
        s.ledger.record_paid_usd(usd).map_err(|e| e.to_string())?;
    }
    s.cache.set(key, text.clone(), js_sys::Date::now());
    Ok(Reply { text, cached: false, attempts, usage, cost_usd })
}

/// POST the body, retrying what `policy` says is retryable. Network errors
/// (offline, CORS, DNS) are retried like a 503.
async fn post_with_retry(settings: &Settings, policy: &RetryPolicy, body: &str) -> Result<(serde_json::Value, u32), String> {
    let url = format!("{}/chat/completions", settings.base_url.trim_end_matches('/'));
    let mut attempt = 1;
    loop {
        let mut req = Request::post(&url).header("Content-Type", "application/json");
        if !settings.api_key.is_empty() {
            req = req.header("Authorization", &format!("Bearer {}", settings.api_key));
        }
        let sent = req.body(body.to_string()).map_err(|e| e.to_string())?.send().await;
        let (status, retry_after, error) = match sent {
            Ok(resp) if resp.ok() => {
                let json = resp.json::<serde_json::Value>().await.map_err(|e| e.to_string())?;
                return Ok((json, attempt));
            }
            Ok(resp) => {
                let status = resp.status();
                let retry_after = resp.headers().get("retry-after");
                (status, retry_after, format!("HTTP {status}: {}", resp.text().await.unwrap_or_default()))
            }
            Err(e) => (503, None, format!("network error: {e}")),
        };
        if !policy.should_retry(attempt, status) {
            return Err(format!("{error} (after {attempt} attempt(s))"));
        }
        let wait_ms = policy.delay_for_response(attempt, retry_after.as_deref(), js_sys::Math::random());
        if wait_ms > 60_000 {
            // A server asking for minutes is better reported than waited on in a UI.
            return Err(format!("{error} (server asked to retry in {} s)", wait_ms / 1000));
        }
        TimeoutFuture::new(wait_ms).await;
        attempt += 1;
    }
}
