//! Integration test: content guard + length guard in chain.

use llm_wasm::guard::{ContentGuard, GuardChain, LengthGuard};
use llm_wasm::types::{ChatMessage, ChatRequest, Role};
use llm_wasm::LlmWasmError;

fn user_req(content: &str) -> ChatRequest {
    ChatRequest::new("model", vec![ChatMessage::new(Role::User, content)])
}

#[test]
fn test_chain_allows_clean_short_message() {
    let chain = GuardChain::new()
        .add(ContentGuard::new(vec!["spam".into(), "violence".into()]))
        .add(LengthGuard::new(200));

    let result = chain.check(&user_req("Hello, how are you?")).unwrap();
    assert!(result.is_none(), "clean message should be allowed");
}

#[test]
fn test_chain_blocks_on_content_before_length_checked() {
    let chain = GuardChain::new()
        .add(ContentGuard::new(vec!["badword".into()]))
        .add(LengthGuard::new(1000)); // length would pass, but content blocks first

    let result = chain.check(&user_req("this contains badword in it"));
    assert!(
        matches!(result, Err(LlmWasmError::GuardBlocked { ref guard, .. }) if guard == "content"),
        "content guard should fire first"
    );
}

#[test]
fn test_chain_blocks_on_length_when_content_clean() {
    let chain = GuardChain::new()
        .add(ContentGuard::new(vec!["spam".into()]))
        .add(LengthGuard::new(10));

    let result = chain.check(&user_req("this message is definitely longer than ten characters"));
    assert!(
        matches!(result, Err(LlmWasmError::GuardBlocked { ref guard, .. }) if guard == "length"),
        "length guard should fire"
    );
}

#[test]
fn test_chain_multiple_messages_content_checked_all() {
    let chain = GuardChain::new()
        .add(ContentGuard::new(vec!["forbidden".into()]))
        .add(LengthGuard::new(500));

    let req = ChatRequest::new(
        "model",
        vec![
            ChatMessage::new(Role::System, "You are helpful."),
            ChatMessage::new(Role::User, "Tell me something"),
            ChatMessage::new(Role::Assistant, "Sure! This contains forbidden content."),
        ],
    );
    let result = chain.check(&req);
    assert!(matches!(result, Err(LlmWasmError::GuardBlocked { .. })));
}

#[test]
fn test_empty_chain_always_allows() {
    let chain = GuardChain::new();
    let result = chain.check(&user_req("anything goes")).unwrap();
    assert!(result.is_none());
}
