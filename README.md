# llm-wasm

[![CI](https://github.com/Mattbusel/llm-wasm/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/llm-wasm/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/llm-wasm.svg)](https://crates.io/crates/llm-wasm)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

The logic layer around an LLM call, in pure Rust that builds for `wasm32-unknown-unknown`: response cache, retry policy, request guards, model routing, cost ledger, prompt templates and JSON extraction.

In a browser, a Cloudflare Worker or another edge runtime you usually cannot pull in Tokio or a full HTTP SDK. `llm-wasm` has no I/O and no clock of its own (you pass `now_ms`), so the same code runs on the host and in WASM. You make the HTTP call; this crate decides whether to make it, which model to use, what it costs, and how to parse the answer.

## Modules

| Module | What it gives you |
|---|---|
| `cache` | `TtlCache` (time-to-live cache, caller supplies the time), `cache_key(model, messages_json)` via FNV-1a |
| `retry` | `RetryPolicy`: exponential backoff with deterministic jitter, retries on 429 / 500 / 502 / 503 / 504 only |
| `guard` | `Guard` trait, `ContentGuard` (case-insensitive blocklist), `LengthGuard` (total characters), `GuardChain` that can block or rewrite a request |
| `routing` | `Router` with ordered `RoutingRule`s (`Always`, `MessageCountExceeds`, `ModelNameContains`, `MaxTokensBelow`) and a fallback model |
| `cost` | `pricing_for_model` for a built-in table of Claude and GPT-4o models, `CostLedger` with an optional hard USD budget |
| `template` | `TemplateEngine` for `{{variable}}` and `{{>partial}}` substitution |
| `format` | `JsonFormatter::extract_json` (pulls JSON out of chatty model output), `MarkdownFormatter::strip_code_fence` |
| `types` | `ChatMessage`, `ChatRequest`, `ChatResponse`, `StreamChunk`, `Role` |

## Quick start

```bash
cargo add llm-wasm
```

```rust
use std::collections::HashMap;
use llm_wasm::cache::{cache_key, TtlCache};
use llm_wasm::cost::CostLedger;
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

    // 2. Guard it.
    let guards = GuardChain::new()
        .add(ContentGuard::new(vec!["password".into()]))
        .add(LengthGuard::new(20_000));
    guards.check(&request)?; // Err(GuardBlocked) if a guard says no

    // 3. Pick a model.
    let mut router = Router::new("gpt-4o-mini");
    router.add_rule(RoutingRule::new(RoutingCondition::MessageCountExceeds(10), "claude-sonnet-4-6"));
    let model = router.route(&request).to_string();

    // 4. Check the cache (you supply the clock, e.g. Date.now() in JS).
    let now_ms = 1_700_000_000_000.0;
    let mut cache = TtlCache::new(60_000.0);
    let key = cache_key(&model, &serde_json::to_string(&request.messages).unwrap_or_default());
    if cache.get(key, now_ms).is_none() {
        // 5. Call the model yourself; on failure ask the policy whether to retry.
        let policy = RetryPolicy::exponential();
        if policy.should_retry(1, 429) {
            let _wait_ms = policy.delay_for_attempt(1);
        }
        let reply = "Sure! Here it is: {\"population\": 2102650}".to_string();
        cache.set(key, reply, now_ms);
    }

    // 6. Account for it and parse it.
    let mut ledger = CostLedger::with_budget(1.00);
    ledger.record(&model, 40, 20)?; // Err(BudgetExceeded) past $1.00
    let reply = cache.get(key, now_ms).unwrap_or_default();
    let json = JsonFormatter::extract_json(&reply)?;
    println!("{model}: {json} (spent ${:.6})", ledger.total_usd());
    Ok(())
}
```

Build for WebAssembly:

```bash
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown --release
```

## Design

- No `unwrap`, `expect` or `panic` (denied by Clippy lints in `Cargo.toml`); every fallible call returns `LlmWasmError`.
- Nothing reads the system clock or random numbers, so behavior is deterministic and identical on every target.
- The crate type is `cdylib` + `rlib`, and `wasm-bindgen` / `js-sys` are pulled in only for `wasm32` targets.

## Status and limitations

Version 0.1. The test suite has 182 tests (unit, integration and pipeline).

- No `#[wasm_bindgen]` exports are defined yet: use it as a Rust dependency of your own WASM crate, not directly from JavaScript.
- The pricing table is hard-coded to five models (`claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`, `gpt-4o`, `gpt-4o-mini`); other model names return an error from `CostLedger::record`.
- Templates support substitution and partials only, no loops or conditionals.

```bash
cargo test
```

## License

MIT, see [LICENSE](LICENSE).

---

Part of a set of Rust crates for LLM agents, see [rust-crates](https://github.com/Mattbusel/rust-crates).
