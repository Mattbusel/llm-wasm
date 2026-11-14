# Changelog

## 0.3.0 (unreleased)

Cache-aware cost, thread-safe guards and routers, and an API that can grow without breaking you again.

### Added
- `types::Usage { input, output, cache_read, cache_write, reasoning }` with `Usage::new(input, output)`, `with_*` setters, `total_input`, `total_output`, `total`, and `+` / `+=` for summing a run. The rustdoc maps Anthropic and OpenAI usage fields onto it. `ChatResponse::usage()` and `From<StreamUsage> for Usage` convert existing counts.
- `ModelPricing::cost(&Usage)`: prices prompt-cache reads and writes at their own rates (falling back to the input rate when the provider lists none) and bills reasoning tokens at the output rate. Before, a call that read 180k tokens from Anthropic's prompt cache was priced as if all 180k were fresh input: `claude-opus-5-5` with 3k input, 180k cache read, 20k cache write and 1.2k output costs $0.172, not $0.836.
- `ModelPricing` now exposes what the price table already held: `provider`, `cache_read_per_million`, `cache_write_per_million` (both `Option<f64>`) and `max_input_tokens` (`Option<u32>`). `ModelPricing::new(input, output)` plus `with_provider`, `with_cache_read`, `with_cache_write` and `with_max_input_tokens` build pricing for models missing from the table.
- `CostLedger::record_usage(model, &Usage)`: record a call with its cache and reasoning split; returns the new total and enforces the budget like `record`.
- `cost::context_window(model)`: the model's maximum input tokens from the table.
- `LlmWasmError::UnknownModel(String)`, returned by `pricing_for_model`, `CostLedger::record` and `record_usage` for models not in the table.
- `LlmWasmError::Provider { kind: Option<String>, message: String }`: `StreamDecoder` now reports provider `error` events with this variant, with the provider's error type or code (`"overloaded_error"`, `"rate_limit_exceeded"`) in `kind`, so callers can tell a provider overload from their own misconfiguration.
- `Debug` for `GuardChain` (lists guard names) and `Router`; `Debug` and `Clone` for `GuardResult`, `ContentGuard` and `LengthGuard`; `Clone` for `Router`.
- `tests/send_sync.rs`: compile-time checks that `GuardChain`, `Router`, `CostLedger`, `TtlCache`, `StreamDecoder` (feature `stream`) and `SecretGuard` (feature `secrets`) are `Send + Sync`, plus a test that shares a `GuardChain` across threads in an `Arc`.

### Changed (breaking)
- `trait Guard: Send + Sync`. A `GuardChain` can now sit in `Arc` state of an axum, tower or actix server or in a `static`; before, it was `!Send` and `!Sync` and failed to compile there.
- `#[non_exhaustive]` on `LlmWasmError`, `GuardResult`, `RoutingCondition`, `Role`, `ModelPricing` and the new `Usage`. Future variants and fields can ship in minor releases.
- `pricing_for_model` and `CostLedger::record` return `UnknownModel` instead of `InvalidConfig { field: "model", .. }` for an unknown model. Its message is now `Unknown model '<name>': no pricing data available` (the JS bindings throw it too).
- `StreamDecoder::push` returns `Provider` instead of `InvalidConfig { field: "stream", .. }` for provider error events.
- `thiserror` 1 -> 2. Dependents already on thiserror 2 no longer build two copies of it (and of its proc-macro stack). MSRV stays 1.88.
- `ModelPricing::cost_usd` is now `cost(&Usage::new(input, output))`; the result is unchanged.

### Migrating from 0.2
- Custom guards: a guard must be `Send + Sync`. Plain data, `String`, `Vec`, `regex::Regex` and `Arc<...>` are; replace `Rc<T>` with `Arc<T>` and `RefCell<T>` / `Cell<T>` with `Mutex<T>` / atomics.
- `match` on `LlmWasmError`, `GuardResult`, `RoutingCondition` or `Role` outside this crate needs a final `_ => ...` arm.
- `ModelPricing { input_per_million: a, output_per_million: b }` no longer compiles outside the crate: use `ModelPricing::new(a, b)`.
- Code that matched `InvalidConfig { field, .. }` with `field == "model"` should match `UnknownModel(_)`; code that matched `field == "stream"` should match `Provider { .. }`.
- To price prompt-cached calls correctly, switch `ledger.record(model, input, output)` to `ledger.record_usage(model, &Usage::new(input, output).with_cache_read(cached).with_cache_write(written))`. For OpenAI, `input` is `prompt_tokens - cached_tokens`.

## 0.2.1 (2026-10-21)

- Republished with the rewritten README (crate docs on docs.rs come from it). No code changes.

## 0.2.0 (2026-10-03)

### Added
- Feature `stream`: `StreamDecoder` reads streamed (SSE) answers from OpenAI, Anthropic and OpenAI-compatible APIs in pieces of any size (lines and multi-byte characters may be split anywhere; property-tested), with usage counts and provider error events. SSE framing by sse-core. JS class `StreamDecoder`.
- Feature `repair`: `JsonFormatter::extract_json_lenient` repairs almost-JSON (trailing commas, single quotes, unquoted keys, comments, Python literals, raw newlines in strings, cut-off output) with json5 and a bounded pre-pass. JS `extractJsonLenient`. jsonrepair was tried first and rejected: property tests found a panic and an input that drove memory past 20 GB.
- Feature `jinja`: `render_jinja` and `render_chat_template` (Hugging Face `chat_template` strings, with Python string methods and `raise_exception`), via minijinja with a fuel limit. Checked against the Llama 3 and Mistral templates. JS `renderChatTemplate`.
- Feature `secrets`: `SecretGuard` blocks or redacts API keys (OpenAI, Anthropic, AWS, GitHub, Google, Slack), private keys, emails and Luhn-valid card numbers. JS `redactSecrets`.
- `ContentGuard::whole_words`: match whole words only (`"ass"` no longer blocks `"class"`).
- `Role::Tool`; `ChatRequest` serializes to the OpenAI Chat Completions body.
- Property tests for every parser (no panics on random input; embedded JSON always found; stream split invariance; retry delay bounds), `benches/vs_alternatives.rs` against 0.1.1 and minijinja, examples `stream_decode`, `repair_json`, `chat_template`, CONTRIBUTING.md, issue templates, MSRV job in CI.
- Feature `js`: JavaScript bindings via wasm-bindgen: `estimateCost`, `knownModels`, `extractJson` (returns a JS object), `stripCodeFence`, `renderTemplate`, `checkRequest`, `routeModel`, and the classes `Ledger`, `Retry`, `Cache`. `examples/node/smoke.mjs` exercises every export from Node; CI builds the module, runs wasm-bindgen and runs it.
- Price table of 232 chat models (OpenAI, Anthropic, Gemini, Mistral, DeepSeek, xAI) generated from LiteLLM's maintained price list (MIT) by `scripts/update_prices.py`, looked up by binary search; `known_models()`, `PRICES_SNAPSHOT_DATE`; `provider/model` names accepted.
- `CostLedger::record_usd` and `remaining_usd`; `record` returns the new total.
- `RetryPolicy::delay_with_jitter` (caller passes a random number) and `delay_for_response` (honours `Retry-After` seconds); `parse_retry_after_seconds`.
- `TtlCache::with_capacity`: a size cap that drops expired entries first, then the oldest.
- `examples/quickstart.rs`; the README is the crate docs and its quickstart is a doctest.

### Fixed
- The cache key was a 64-bit FNV-1a hash of `"{model}::{messages}"`: collisions are easy to construct, so in a shared cache one user could be served another's answer, and `("a::b", "c")` equalled `("a", "b::c")`. Keys are now SHA-256 over length-prefixed fields (`CacheKey`). The JS `Cache` stores the full key string.
- `TtlCache::with_capacity` evicted the oldest insert even if it was read constantly; it is now least-recently-used (lru crate).
- `extract_json` tried every bracket and scanned to the end each time: 187 ms on a text with 20,000 stray `{`, now 3.9 ms (bounded scan budget).
- `ContentGuard` with an empty term blocked every request.
- `Role` serialized as `"User"` and unset options as `null`, which OpenAI-style APIs reject; now lowercase and omitted. Old JSON still reads.
- Wrong prices: `claude-opus-4-6` was $15 / $75 per million tokens (Anthropic charges $5 / $25) and `claude-haiku-4-5` was $0.80 / $4 (actual $1 / $5). Only five models were known at all.
- A template partial that included itself (directly or through another partial) recursed until the stack overflowed, killing the process or WASM instance. Nesting is now limited to 32 levels and returns `TemplateError`.
- `LengthGuard` and `ChatRequest::total_content_chars` counted bytes, not characters, so non-English text was blocked early (10 accented letters counted as 20).
- Retry "jitter" depended only on the attempt number, so every client waited exactly as long as every other one; the docs also described the range wrongly. Real jitter is now available with caller-supplied randomness.
- The `wasm32` dependencies (`wasm-bindgen`, `js-sys`, `wasm-bindgen-futures`, `serde-wasm-bindgen`, `getrandom`) were pulled in for every wasm build but nothing used them. They are gone; `wasm-bindgen` and `serde-wasm-bindgen` return as optional deps of the `js` feature.
- `CostLedger` kept every entry forever and summed them on each call; it now keeps a running total.

### Changed
- `CostLedger::record` returns `Result<f64, _>` (the new total) instead of `Result<(), _>`.
- `InvalidConfig` error text uses a colon instead of a dash.
- `cache_key` returns `CacheKey` instead of `u64`; `TtlCache` is generic over the key type (default `CacheKey`) and `get` takes `&K`.
- MSRV is 1.88 (json5 uses let chains), checked in CI.

## 0.1.1

- First published version.
