//! Compile-time checks that the stateful types can be shared across threads,
//! for example in `Arc` state of an axum or tower service, or a `static`.
//! If one of these stops being `Send + Sync`, this file stops compiling.

use llm_wasm::cache::{CacheKey, TtlCache};
use llm_wasm::cost::{CostLedger, ModelPricing};
use llm_wasm::guard::{ContentGuard, GuardChain, GuardResult, LengthGuard};
use llm_wasm::retry::RetryPolicy;
use llm_wasm::routing::{Router, RoutingCondition, RoutingRule};
use llm_wasm::template::TemplateEngine;
use llm_wasm::types::{ChatRequest, Usage};
use llm_wasm::LlmWasmError;

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn core_types_are_send_and_sync() {
    assert_send_sync::<GuardChain>();
    assert_send_sync::<ContentGuard>();
    assert_send_sync::<LengthGuard>();
    assert_send_sync::<GuardResult>();
    assert_send_sync::<Box<dyn llm_wasm::guard::Guard>>();
    assert_send_sync::<Router>();
    assert_send_sync::<RoutingRule>();
    assert_send_sync::<RoutingCondition>();
    assert_send_sync::<CostLedger>();
    assert_send_sync::<ModelPricing>();
    assert_send_sync::<Usage>();
    assert_send_sync::<TtlCache>();
    assert_send_sync::<TtlCache<CacheKey>>();
    assert_send_sync::<TtlCache<String>>();
    assert_send_sync::<RetryPolicy>();
    assert_send_sync::<TemplateEngine>();
    assert_send_sync::<ChatRequest>();
    assert_send_sync::<LlmWasmError>();
}

#[cfg(feature = "stream")]
#[test]
fn stream_decoder_is_send_and_sync() {
    assert_send_sync::<llm_wasm::stream::StreamDecoder>();
}

#[cfg(feature = "secrets")]
#[test]
fn secret_guard_is_send_and_sync() {
    assert_send_sync::<llm_wasm::secrets::SecretGuard>();
}

#[test]
fn guard_chain_works_from_another_thread() {
    use std::sync::Arc;
    let chain = Arc::new(GuardChain::new().add(ContentGuard::whole_words(vec!["password".into()])));
    let shared = Arc::clone(&chain);
    let blocked = std::thread::spawn(move || {
        let req = ChatRequest::new(
            "m",
            vec![llm_wasm::types::ChatMessage::new(
                llm_wasm::types::Role::User,
                "my password",
            )],
        );
        shared.check(&req).is_err()
    })
    .join()
    .unwrap();
    assert!(blocked);
}
