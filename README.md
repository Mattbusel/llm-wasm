# llm-wasm

LLM inference primitives for WebAssembly and edge environments -- cache, retry, routing, guards, cost tracking, and templates in a single no-std-compatible Rust crate.

## What's inside

| Module | Description |
|--------|-------------|
| `cache` | FNV-1a keyed `LruCache` + `TtlCache` with configurable expiry and auto-purge |
| `retry` | `RetryPolicy` with exponential backoff, jitter, and per-status-code retryability |
| `guard` | `GuardChain` -- composable content safety, PII detection, and injection filtering |
| `routing` | `Router` with condition-based dispatch (model, cost, latency, tag rules) |
| `cost` | `CostLedger` for per-model token accounting and hard budget enforcement |
| `template` | Mustache-style `TemplateEngine` with partials, loops, and conditionals |
| `format` | `JsonFormatter` and `MarkdownFormatter` for structured output validation |
| `streaming` | Token-streaming utilities for chunked LLM output |
| `types` | `ChatRequest`, `ChatResponse`, `ModelId`, `TokenCount` newtypes |

## Features

- **WASM-compatible** -- no OS threads required; all sync-safe primitives
- **Zero external HTTP** -- bring your own transport; this crate handles the logic layer
- **Composable pipelines** -- chain guard → route → cost → cache → retry in any order
- **Hard budget enforcement** -- `CostLedger` refuses requests that would exceed configured limits

## Quick start

```rust
use llm_wasm::{
 guard::GuardChain,
 routing::Router,
 cost::CostLedger,
 cache::TtlCache,
};

// Build a guard chain
let guards = GuardChain::new()
 .add(PiiGuard::default())
 .add(InjectionGuard::default());

// Check before sending
if let Some(block) = guards.check("ignore previous instructions") {
 eprintln!("Blocked: {}", block.reason);
}
```

## Add to your project

```toml
[dependencies]
llm-wasm = { git = "https://github.com/Mattbusel/llm-wasm" }
```

Or one-liner:

```ash
cargo add --git https://github.com/Mattbusel/llm-wasm
```

## Test coverage

168+ tests across unit, integration, and pipeline suites.

```bash
cargo test
```

---

> Used inside [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator) -- a production Rust orchestration layer for LLM pipelines. See the full [primitive library collection](https://github.com/Mattbusel/rust-crates).