# llm-wasm

The bookkeeping around an LLM API call (what it costs, whether to retry, whether to block it, which model to use, how to read a streamed answer, how to pull JSON out of it) as plain Rust that runs the same in a browser, an edge worker or a normal program.

[![crates.io](https://img.shields.io/crates/v/llm-wasm.svg)](https://crates.io/crates/llm-wasm)
[![docs.rs](https://img.shields.io/docsrs/llm-wasm)](https://docs.rs/llm-wasm)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://gitlab.com/mattbusel/llm-wasm/-/blob/main/LICENSE)

In a browser or a Cloudflare Worker you usually cannot pull in Tokio or a full HTTP SDK. This crate does no I/O and reads no clock (you pass the time in), so it builds for `wasm32-unknown-unknown` unchanged. You make the HTTP call; `llm-wasm` decides whether to make it, which model to use, what it costs and how to read the answer. With the `js` feature it is also callable straight from JavaScript.

## Install

```bash
cargo add llm-wasm
# streaming, JSON repair, chat templates, secret redaction, JS bindings:
cargo add llm-wasm --features stream,repair,jinja,secrets,js
```

## In ten lines

```rust
use llm_wasm::cost::CostLedger;
use llm_wasm::format::JsonFormatter;
use llm_wasm::types::{ChatMessage, ChatRequest, Role};

let req = ChatRequest::new("gpt-4o-mini", vec![ChatMessage::new(Role::User, "List 2 cities as JSON")]);
assert!(serde_json::to_string(&req).unwrap().contains(r#"{"role":"user","content":"List 2 cities as JSON"}"#)); // OpenAI body
let reply = "Sure! {\"cities\": [\"Paris\", \"Lyon\"]} Anything else?";
let mut ledger = CostLedger::with_budget(0.01);
ledger.record("gpt-4o-mini", 12, 9).unwrap();                       // real price, $0.0000072
assert_eq!(JsonFormatter::extract_json(reply).unwrap()["cities"][1], "Lyon");
```

## Quickstart (Rust)

```rust
use std::collections::HashMap;
use llm_wasm::cache::{cache_key, TtlCache};
use llm_wasm::cost::{pricing_for_model, CostLedger};
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

    // 2. Guard it: blocked words, maximum length in characters.
    let guards = GuardChain::new()
        .add(ContentGuard::whole_words(vec!["password".into()]))
        .add(LengthGuard::new(20_000));
    guards.check(&request)?; // Err(GuardBlocked) if a guard says no

    // 3. Pick a model.
    let mut router = Router::new("gpt-4o-mini");
    router.add_rule(RoutingRule::new(RoutingCondition::MessageCountExceeds(10), "claude-sonnet-4-5"));
    let model = router.route(&request).to_string();

    // 4. Check the cache (you supply the clock, e.g. Date.now() in JS).
    let now_ms = 1_700_000_000_000.0;
    let mut cache = TtlCache::with_capacity(60_000.0, 1_000); // 60 s, at most 1,000 entries
    let key = cache_key(&model, &serde_json::to_string(&request.messages)?);
    if cache.get(&key, now_ms).is_none() {
        // 5. Call the model yourself. If it fails, ask the policy what to do.
        let policy = RetryPolicy::exponential();
        if policy.should_retry(1, 429) {
            // Pass a random number in [0, 1) and the Retry-After header, if any.
            let wait_ms = policy.delay_for_response(1, Some("2"), 0.37);
            assert_eq!(wait_ms, 2_000);
        }
        let reply = "Sure! Here it is: {\"population\": 2102650}".to_string();
        cache.set(key, reply, now_ms);
    }

    // 6. Account for it (real per-model prices) and parse the answer.
    println!("{model} costs ${} per million input tokens", pricing_for_model(&model)?.input_per_million);
    let mut ledger = CostLedger::with_budget(1.00);
    ledger.record(&model, 40, 20)?; // Err(BudgetExceeded) past $1.00
    let reply = cache.get(&key, now_ms).unwrap_or_default();
    let json = JsonFormatter::extract_json(&reply)?;
    println!("{model}: {json} (spent ${:.6})", ledger.total_usd());
    Ok(())
}
```

The same program is `examples/quickstart.rs`: `cargo run --example quickstart`.

## Quickstart (JavaScript, feature `js`)

Build the module (needs [wasm-bindgen-cli](https://crates.io/crates/wasm-bindgen-cli) at the same version as the `wasm-bindgen` crate in your `Cargo.lock`, or use [wasm-pack](https://github.com/drager/wasm-pack)):

```bash
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown --features js,stream,repair
wasm-bindgen --target nodejs --out-dir pkg target/wasm32-unknown-unknown/release/llm_wasm.wasm
node examples/node/smoke.mjs pkg   # calls every export and checks the results
```

```js
const llm = require("./pkg/llm_wasm.js");

llm.estimateCost("gpt-4o-mini", 1000, 500);          // 0.00045 (USD)
llm.extractJson('Sure! {"a": [1, 2]}');               // { a: [1, 2] }
llm.extractJsonLenient("{'a': [1, 2,], 'ok': True");   // { a: [1, 2], ok: true }  (feature repair)

// Stream an answer as it arrives (feature stream). Works for OpenAI, Anthropic
// and OpenAI-compatible APIs; pieces may split lines or characters anywhere.
const dec = new llm.StreamDecoder();
for await (const piece of response.body) {
  for (const delta of dec.push(piece)) out.textContent += delta;
}
dec.usage;                                             // { input_tokens, output_tokens } if sent

const ledger = new llm.Ledger(1.0);                   // $1 budget
ledger.record("gpt-4o", 1200, 300);                   // throws once over budget
const retry = new llm.Retry(3, 200, 5000);
retry.delay(1, Math.random(), response.headers.get("retry-after"));
const cache = new llm.Cache(60000, 1000);             // 60 s TTL, LRU, 1,000 entries
cache.set(promptKey, answer, Date.now());
llm.redactSecrets(userText);                          // feature secrets
llm.renderChatTemplate(tokenizerConfig.chat_template, messages, true, "<s>", "</s>"); // feature jinja
```

Use `--target web` (and `await init()`) for browsers and workers.

## Why this and not something else

- **Provider SDKs** (async-openai, the official JS SDKs) need an HTTP client and usually Tokio or Node; they do not build for a plain `wasm32-unknown-unknown` worker. This crate is only the logic, so it runs wherever Rust compiles.
- **Doing it by hand** is where the bugs are: this crate's 0.1 had a stack overflow on self-including templates, byte-counting length limits, a cache key two different prompts could share, and quadratic JSON extraction. 0.2 fixes those and is property-tested.
- **Agent frameworks** (rig, genai) are larger and do the HTTP for you; use them if you are on a server with Tokio.

## Already using another crate or format?

- **OpenAI-style chat APIs**: `ChatRequest` serializes to the Chat Completions body (`role` in lowercase, unset options left out), and old capitalised JSON still reads.
- **OpenAI / Anthropic streaming**: `StreamDecoder` reads both SSE formats (parsing by [sse-core](https://crates.io/crates/sse-core)).
- **Hugging Face models**: `render_chat_template` runs the `chat_template` from `tokenizer_config.json` with [minijinja](https://crates.io/crates/minijinja) (Python string methods and `raise_exception` included), checked against the Llama 3 and Mistral templates.
- **LiteLLM**: the price table is generated from LiteLLM's price list (`scripts/update_prices.py`).

## Features

| Feature | Default | What it adds | Extra dependencies | `.wasm` size with `js` (release, after wasm-bindgen, before wasm-opt) |
|---|---|---|---|---|
| (none) | yes | cost, retry, guards, routing, cache, templates, JSON extraction, types | serde, serde_json, thiserror, sha2, lru | |
| `js` | no | `#[wasm_bindgen]` exports: `estimateCost`, `knownModels`, `extractJson`, `stripCodeFence`, `renderTemplate`, `checkRequest`, `routeModel`, classes `Ledger`, `Retry`, `Cache` | wasm-bindgen, serde-wasm-bindgen | 242 KB |
| `stream` | no | `stream::StreamDecoder` (+ JS `StreamDecoder`) | [sse-core](https://crates.io/crates/sse-core), bytes | 273 KB |
| `repair` | no | `JsonFormatter::extract_json_lenient` (+ JS `extractJsonLenient`) | [json5](https://crates.io/crates/json5) | 340 KB |
| `secrets` | no | `secrets::SecretGuard`: block or redact API keys, private keys, emails, card numbers (+ JS `redactSecrets`) | [regex](https://crates.io/crates/regex) | 672 KB |
| `jinja` | no | `template::render_jinja`, `render_chat_template` (+ JS `renderChatTemplate`) | [minijinja](https://crates.io/crates/minijinja), minijinja-contrib | 1,578 KB |
| all of the above | | | | 2,103 KB |

## Examples

| Example | Shows | Run |
|---|---|---|
| `quickstart` | the whole request path: template, guards, routing, cache, retry, cost, JSON | `cargo run --example quickstart` |
| `stream_decode` | OpenAI and Anthropic streams cut into 7-byte pieces | `cargo run --example stream_decode --features stream` |
| `repair_json` | strict vs lenient extraction on five real-looking answers | `cargo run --example repair_json --features repair` |
| `chat_template` | Llama 3 prompt from its own template, with secret redaction first | `cargo run --example chat_template --features jinja,secrets` |
| `node/smoke.mjs` | every JavaScript export, from Node | see the JavaScript quickstart |

## Modules

| Module | What it gives you |
|---|---|
| `cost` | `pricing_for_model` over 232 chat models (OpenAI, Anthropic, Gemini, Mistral, DeepSeek, xAI) from [LiteLLM's price list](https://github.com/BerriAI/litellm) (MIT); `CostLedger` with an optional hard USD budget and `record_usd` |
| `retry` | `RetryPolicy`: exponential backoff, `delay_with_jitter` (you pass a random number), `delay_for_response` (honours `Retry-After` seconds); retries 429 / 500 / 502 / 503 / 504 only |
| `guard` | `Guard` trait, `ContentGuard` (substring or `whole_words`, Unicode case-insensitive), `LengthGuard` (characters, not bytes), `GuardChain` that can block or rewrite a request |
| `secrets` | `SecretGuard` (feature `secrets`) |
| `routing` | `Router` with ordered `RoutingRule`s (`Always`, `MessageCountExceeds`, `ModelNameContains`, `MaxTokensBelow`) and a fallback model |
| `cache` | `TtlCache` (time-to-live, least-recently-used eviction, caller supplies the time), `cache_key(model, messages_json)` (SHA-256) |
| `template` | `TemplateEngine` for `{{variable}}` and `{{>partial}}`; `render_jinja` / `render_chat_template` (feature `jinja`) |
| `format` | `JsonFormatter::extract_json`, `extract_json_lenient` (feature `repair`), `MarkdownFormatter::strip_code_fence` |
| `stream` | `StreamDecoder` (feature `stream`) |
| `types` | `ChatMessage`, `ChatRequest`, `ChatResponse`, `StreamChunk`, `Role` |

To refresh the price table: `python scripts/update_prices.py`, then rebuild. `cost::PRICES_SNAPSHOT_DATE` says when it was generated.

## Performance

`benches/vs_alternatives.rs` (Intel i7-13700KF, Windows 11, release build, other work running, so rough):

| Task | llm-wasm 0.2 | comparison |
|---|---|---|
| extract JSON from a chatty 2 KB answer | 12.5 µs | 0.1.1: 13.4 µs |
| extract JSON from text with 20,000 stray `{` | 3.9 ms (stops at a 4 MB scan budget) | 0.1.1: 187 ms (quadratic) |
| lenient extraction of an answer cut off halfway | 20.9 µs | |
| render a 4-variable `{{var}}` template | 240 ns built-in | minijinja `render_jinja`: 2.5 µs (parses the template each call) |
| cache key + set + get with a 1 KB prompt | 563 ns (SHA-256, LRU) | 0.1.1: 969 ns (FNV-1a, HashMap) |
| decode a 500-chunk OpenAI stream in 64-byte pieces | 517 µs (86 MB/s) | |

Reproduce with `cargo bench --bench vs_alternatives --features repair,jinja,stream`.

## Design

- No `unwrap`, `expect` or `panic` in library code (denied by Clippy lints); every fallible call returns `LlmWasmError`. Property tests feed random input to every parser.
- Nothing reads the system clock or a random number generator, so behaviour is deterministic and identical on every target. Where randomness helps (retry jitter) you pass it in.
- Crate type is `cdylib` + `rlib`, so the same crate is a Rust dependency and a `.wasm` module.

## Limitations

- `extract_json_lenient` does not insert missing commas between members, and gives up on JSON that only starts after more than 4 MB worth of scanning past stray brackets.
- `{{var}}` templates support substitution and partials only (nesting 32 deep); use `render_jinja` for loops and conditionals.
- `SecretGuard` matches known key formats; it is a safety net, not a guarantee.
- `Retry-After` in HTTP-date form is not parsed (there is no clock); the backoff is used instead.
- Gemini models with long-context surcharges are priced at their base rate.

Run the tests with `cargo test --all-features`. Contributions are welcome, see [CONTRIBUTING.md](https://gitlab.com/mattbusel/llm-wasm/-/blob/main/CONTRIBUTING.md).

## License

MIT, see [LICENSE](https://gitlab.com/mattbusel/llm-wasm/-/blob/main/LICENSE).
