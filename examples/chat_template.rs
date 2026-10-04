//! Build the exact prompt a local model expects from its own Hugging Face
//! chat template, then check the request for leaked secrets.
//!
//! Run with `cargo run --example chat_template --features jinja,secrets`.

use llm_wasm::guard::{ContentGuard, GuardChain};
use llm_wasm::secrets::SecretGuard;
use llm_wasm::template::render_chat_template;
use llm_wasm::types::{ChatMessage, ChatRequest, Role};

// The chat_template field of Meta-Llama-3-8B-Instruct's tokenizer_config.json.
const LLAMA3: &str = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}";

fn main() -> Result<(), llm_wasm::LlmWasmError> {
    let messages = vec![
        ChatMessage::new(Role::System, "You are a terse assistant."),
        ChatMessage::new(Role::User, "My key is sk-proj-AbCdEf0123456789XyZ0123456789. Is it valid?"),
    ];

    // Redact secrets and refuse whole-word blocklisted terms before anything leaves.
    let guards = GuardChain::new()
        .add(SecretGuard::redact())
        .add(ContentGuard::whole_words(vec!["password".into()]));
    let request = ChatRequest::new("llama-3-8b", messages);
    let request = guards.check(&request)?.unwrap_or(request);

    let prompt = render_chat_template(LLAMA3, &request.messages, true, "<|begin_of_text|>", "<|eot_id|>")?;
    println!("{prompt}");
    Ok(())
}
