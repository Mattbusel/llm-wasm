# Changelog

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
