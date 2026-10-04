//! Decode a streamed answer as it arrives, whatever the provider. The bytes
//! below are what OpenAI and Anthropic send; in a worker they come from
//! `fetch` in pieces of any size.
//!
//! Run with `cargo run --example stream_decode --features stream`.

use llm_wasm::stream::StreamDecoder;

const OPENAI: &str = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"content\":\"Paris is \"}}]}\n\n\
data: {\"choices\":[{\"delta\":{\"content\":\"the capital.\"},\"finish_reason\":\"stop\"}]}\n\n\
data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":5}}\n\n\
data: [DONE]\n\n";

const ANTHROPIC: &str = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":12,\"output_tokens\":1}}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Paris is \"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"the capital.\"}}\n\n\
event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n\
event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

fn main() -> Result<(), llm_wasm::LlmWasmError> {
    for (name, body) in [("OpenAI", OPENAI), ("Anthropic", ANTHROPIC)] {
        let mut decoder = StreamDecoder::new();
        print!("{name:>9}: ");
        // 7-byte pieces: lines and JSON are split mid-way, like a real network.
        for piece in body.as_bytes().chunks(7) {
            for chunk in decoder.push(piece)? {
                print!("{}", chunk.delta);
            }
        }
        println!("  [finished: {}, usage: {:?}]", decoder.is_finished(), decoder.usage());
    }
    Ok(())
}
