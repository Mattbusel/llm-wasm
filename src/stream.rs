//! Decode streamed model responses (feature `stream`).
//!
//! OpenAI, Anthropic, Mistral, DeepSeek, Groq, OpenRouter, Ollama's OpenAI
//! endpoint and most other chat APIs stream answers as Server-Sent Events.
//! In a browser or a worker you get those bytes in arbitrary pieces from
//! `fetch`: a piece can end in the middle of a line, or in the middle of a
//! multi-byte character. [`StreamDecoder`] takes the pieces as they come,
//! parses the SSE framing with [sse-core](https://crates.io/crates/sse-core)
//! (a zero-I/O decoder that follows the WHATWG spec), and turns each event into
//! a [`StreamChunk`] with the text delta, whatever the provider.
//!
//! ```
//! use llm_wasm::stream::StreamDecoder;
//!
//! let mut d = StreamDecoder::new();
//! // A piece of an OpenAI stream, cut in the middle of a line:
//! let mut text = String::new();
//! for c in d.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\ndata: {\"choi")? {
//!     text.push_str(&c.delta);
//! }
//! for c in d.push(b"ces\":[{\"delta\":{\"content\":\"lo\"}}]}\n\ndata: [DONE]\n\n")? {
//!     text.push_str(&c.delta);
//! }
//! assert_eq!(text, "Hello");
//! assert!(d.is_finished());
//! # Ok::<(), llm_wasm::LlmWasmError>(())
//! ```

use crate::error::LlmWasmError;
use crate::types::StreamChunk;
use bytes::BytesMut;
use serde_json::Value;
use sse_core::{SseDecoder, SseEvent};

/// Token counts reported at the end of a stream, when the provider sends them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StreamUsage {
    /// Prompt tokens.
    pub input_tokens: u32,
    /// Completion tokens.
    pub output_tokens: u32,
}

/// Incremental decoder for streamed chat responses.
#[derive(Debug)]
pub struct StreamDecoder {
    sse: SseDecoder,
    buf: BytesMut,
    finished: bool,
    usage: Option<StreamUsage>,
    text: String,
}

fn u32_at(v: &Value, path: &[&str]) -> Option<u32> {
    let mut cur = v;
    for p in path {
        cur = cur.get(*p)?;
    }
    cur.as_u64().and_then(|n| u32::try_from(n).ok())
}

impl Default for StreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamDecoder {
    /// A decoder with sse-core's default event size limit.
    pub fn new() -> Self {
        Self { sse: SseDecoder::new(), buf: BytesMut::new(), finished: false, usage: None, text: String::new() }
    }

    /// Feed the next piece of the response body and get the chunks it
    /// completed. Pieces may split lines and multi-byte characters anywhere.
    ///
    /// Recognised event payloads: OpenAI-style `choices[0].delta.content`
    /// (and `finish_reason`, `usage`), the `[DONE]` sentinel, Anthropic
    /// `content_block_delta` / `message_delta` / `message_stop` / `error`.
    /// Events without text (role announcements, pings) produce nothing.
    ///
    /// # Errors
    /// [`LlmWasmError::Serialization`] if an event's data is not JSON, or an
    /// event exceeds the size limit. A provider `error` event becomes
    /// [`LlmWasmError::InvalidConfig`] with the provider's message.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<StreamChunk>, LlmWasmError> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some(ev) = self.sse.next(&mut self.buf) {
            let ev = ev.map_err(|e| LlmWasmError::Serialization(format!("stream event: {e}")))?;
            let SseEvent::Message(msg) = ev else { continue };
            if self.finished {
                continue;
            }
            let data = msg.data.trim();
            if data.is_empty() {
                continue;
            }
            if data == "[DONE]" {
                self.finished = true;
                out.push(StreamChunk { delta: String::new(), finished: true });
                continue;
            }
            let v: Value = serde_json::from_str(data)
                .map_err(|e| LlmWasmError::Serialization(format!("stream event is not JSON: {e}")))?;
            if let Some(chunk) = self.handle(&msg.event, &v)? {
                out.push(chunk);
            }
        }
        Ok(out)
    }

    fn handle(&mut self, event: &str, v: &Value) -> Result<Option<StreamChunk>, LlmWasmError> {
        let kind = v.get("type").and_then(Value::as_str).unwrap_or(event);
        // Anthropic
        match kind {
            "content_block_delta" => {
                let text = v.pointer("/delta/text").and_then(Value::as_str).unwrap_or("");
                return Ok(self.emit(text, false));
            }
            "message_start" => {
                if let Some(i) = u32_at(v, &["message", "usage", "input_tokens"]) {
                    self.usage.get_or_insert_with(StreamUsage::default).input_tokens = i;
                }
                return Ok(None);
            }
            "message_delta" => {
                if let Some(o) = u32_at(v, &["usage", "output_tokens"]) {
                    self.usage.get_or_insert_with(StreamUsage::default).output_tokens = o;
                }
                return Ok(None);
            }
            "message_stop" => {
                self.finished = true;
                return Ok(Some(StreamChunk { delta: String::new(), finished: true }));
            }
            "error" => {
                let msg = v.pointer("/error/message").and_then(Value::as_str).unwrap_or("unknown error");
                return Err(LlmWasmError::InvalidConfig { field: "stream".into(), reason: format!("provider error: {msg}") });
            }
            _ => {}
        }
        // OpenAI-style (also errors sent inside the stream)
        if let Some(err) = v.get("error") {
            let msg = err.get("message").and_then(Value::as_str).unwrap_or("unknown error");
            return Err(LlmWasmError::InvalidConfig { field: "stream".into(), reason: format!("provider error: {msg}") });
        }
        if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(StreamUsage {
                input_tokens: u32_at(u, &["prompt_tokens"]).unwrap_or(0),
                output_tokens: u32_at(u, &["completion_tokens"]).unwrap_or(0),
            });
        }
        let Some(choice) = v.get("choices").and_then(|c| c.get(0)) else { return Ok(None) };
        let text = choice.pointer("/delta/content").and_then(Value::as_str).unwrap_or("");
        let done = choice.get("finish_reason").is_some_and(|f| !f.is_null());
        Ok(self.emit(text, done))
    }

    fn emit(&mut self, text: &str, finished: bool) -> Option<StreamChunk> {
        if text.is_empty() && !finished {
            return None;
        }
        self.text.push_str(text);
        Some(StreamChunk { delta: text.to_string(), finished })
    }

    /// True once the stream signalled its end (`[DONE]` or `message_stop`).
    /// Events after that are ignored.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Everything received so far, concatenated.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Token counts, if the provider sent them (OpenAI with
    /// `stream_options.include_usage`, Anthropic always).
    pub fn usage(&self) -> Option<StreamUsage> {
        self.usage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPENAI: &str = "data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Bon\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"jour \u{1F44B}\"},\"finish_reason\":null}]}\n\n\
data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n\
data: {\"id\":\"c1\",\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":3,\"total_tokens\":12}}\n\n\
data: [DONE]\n\n";

    const ANTHROPIC: &str = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"usage\":{\"input_tokens\":25,\"output_tokens\":1}}}\n\n\
event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
event: ping\ndata: {\"type\": \"ping\"}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n\
event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" there\"}}\n\n\
event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":15}}\n\n\
event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

    fn run(stream: &[u8], piece: usize) -> (String, StreamDecoder) {
        let mut d = StreamDecoder::new();
        let mut text = String::new();
        for part in stream.chunks(piece) {
            for c in d.push(part).unwrap() {
                text.push_str(&c.delta);
            }
        }
        (text, d)
    }

    #[test]
    fn openai_stream_any_split() {
        for piece in [1, 2, 3, 7, 64, 10_000] {
            let (text, d) = run(OPENAI.as_bytes(), piece);
            assert_eq!(text, "Bonjour \u{1F44B}", "piece size {piece}");
            assert!(d.is_finished());
            assert_eq!(d.usage(), Some(StreamUsage { input_tokens: 9, output_tokens: 3 }));
        }
    }

    #[test]
    fn anthropic_stream_any_split() {
        for piece in [1, 5, 4096] {
            let (text, d) = run(ANTHROPIC.as_bytes(), piece);
            assert_eq!(text, "Hello there");
            assert!(d.is_finished());
            assert_eq!(d.usage(), Some(StreamUsage { input_tokens: 25, output_tokens: 15 }));
            assert_eq!(d.text(), "Hello there");
        }
    }

    #[test]
    fn crlf_and_comments_and_after_done() {
        let s = ": keep-alive\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\r\n\r\ndata: [DONE]\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"ignored\"}}]}\r\n\r\n";
        let (text, d) = run(s.as_bytes(), 3);
        assert_eq!(text, "a");
        assert!(d.is_finished());
    }

    #[test]
    fn errors_are_reported() {
        let mut d = StreamDecoder::new();
        assert!(d.push(b"data: not json\n\n").is_err());
        let mut d = StreamDecoder::new();
        let e = d.push(b"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n").unwrap_err();
        assert!(e.to_string().contains("Overloaded"));
        let mut d = StreamDecoder::new();
        assert!(d.push(b"data: {\"error\":{\"message\":\"rate limited\"}}\n\n").unwrap_err().to_string().contains("rate limited"));
    }

    #[test]
    fn incomplete_event_waits_for_more() {
        let mut d = StreamDecoder::new();
        assert!(d.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n").unwrap().is_empty());
        assert_eq!(d.push(b"\n").unwrap().len(), 1);
    }
}
