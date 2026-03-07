//! Core data types shared across all llm-wasm modules.
//!
//! These types are plain Rust structs with serde support — no WASM-specific
//! dependencies so they compile identically on host and wasm32 targets.

/// The role of a participant in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Role {
    /// System-level instructions.
    System,
    /// Human turn.
    User,
    /// Model response.
    Assistant,
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    /// Who sent this message.
    pub role: Role,
    /// Text content of the message.
    pub content: String,
}

impl ChatMessage {
    /// Construct a new message.
    ///
    /// # Arguments
    /// * `role` — sender role
    /// * `content` — message text
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self { role, content: content.into() }
    }
}

/// A request to an LLM chat endpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatRequest {
    /// Model identifier (e.g. `"claude-sonnet-4-6"`).
    pub model: String,
    /// Ordered list of messages forming the conversation.
    pub messages: Vec<ChatMessage>,
    /// Maximum tokens to generate. `None` uses the model default.
    pub max_tokens: Option<u32>,
    /// Sampling temperature in `[0.0, 2.0]`. `None` uses the model default.
    pub temperature: Option<f32>,
}

impl ChatRequest {
    /// Construct a minimal request.
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            max_tokens: None,
            temperature: None,
        }
    }

    /// Total character count across all messages.
    pub fn total_content_chars(&self) -> usize {
        self.messages.iter().map(|m| m.content.len()).sum()
    }
}

/// A completed response from an LLM.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatResponse {
    /// Generated text content.
    pub content: String,
    /// Model that produced this response.
    pub model: String,
    /// Number of prompt tokens consumed.
    pub input_tokens: u32,
    /// Number of completion tokens generated.
    pub output_tokens: u32,
}

/// A single chunk from a streaming response.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StreamChunk {
    /// Incremental text delta.
    pub delta: String,
    /// `true` when this is the final chunk.
    pub finished: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_new() {
        let m = ChatMessage::new(Role::User, "hello");
        assert_eq!(m.role, Role::User);
        assert_eq!(m.content, "hello");
    }

    #[test]
    fn test_chat_request_total_content_chars() {
        let req = ChatRequest::new(
            "model",
            vec![
                ChatMessage::new(Role::User, "hello"),
                ChatMessage::new(Role::Assistant, "world!"),
            ],
        );
        assert_eq!(req.total_content_chars(), 11);
    }

    #[test]
    fn test_role_serialization_roundtrip() {
        let r = Role::System;
        let json = serde_json::to_string(&r).unwrap();
        let back: Role = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn test_chat_response_fields() {
        let r = ChatResponse {
            content: "hi".into(),
            model: "gpt-4o".into(),
            input_tokens: 10,
            output_tokens: 5,
        };
        assert_eq!(r.input_tokens + r.output_tokens, 15);
    }

    #[test]
    fn test_stream_chunk_finished_flag() {
        let chunk = StreamChunk { delta: "done".into(), finished: true };
        assert!(chunk.finished);
    }
}
