//! Core data types shared across all llm-wasm modules.
//!
//! These types are plain Rust structs with serde support: no WASM-specific
//! dependencies so they compile identically on host and wasm32 targets.

/// The role of a participant in a chat conversation.
///
/// Serialized in lowercase (`"system"`, `"user"`, `"assistant"`, `"tool"`),
/// as the OpenAI-style chat APIs expect. The capitalised names written by
/// 0.1 (`"User"`) are still accepted when reading.
///
/// Marked `#[non_exhaustive]` since 0.3: a `match` on it needs a `_` arm, so
/// new roles can be added in a minor release.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Role {
    /// System-level instructions.
    #[serde(alias = "System", alias = "developer")]
    System,
    /// Human turn.
    #[serde(alias = "User")]
    User,
    /// Model response.
    #[serde(alias = "Assistant")]
    Assistant,
    /// A tool result sent back to the model.
    #[serde(alias = "Tool")]
    Tool,
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
    /// * `role`: sender role
    /// * `content`: message text
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

/// A request to an LLM chat endpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatRequest {
    /// Model identifier (e.g. `"claude-sonnet-4-6"`).
    pub model: String,
    /// Ordered list of messages forming the conversation.
    pub messages: Vec<ChatMessage>,
    /// Maximum tokens to generate. `None` uses the model default (and is left
    /// out of the JSON).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature in `[0.0, 2.0]`. `None` uses the model default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
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
        self.messages
            .iter()
            .map(|m| m.content.chars().count())
            .sum()
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

impl ChatResponse {
    /// The token counts of this response as a [`Usage`] (no cache or
    /// reasoning split), ready for [`crate::cost::ModelPricing::cost`].
    pub fn usage(&self) -> Usage {
        Usage::new(u64::from(self.input_tokens), u64::from(self.output_tokens))
    }
}

/// Token usage of one model call, split the way providers bill it.
///
/// Every field counts tokens billed at a different rate, so a token belongs to
/// exactly one field:
///
/// | field | billed at | Anthropic usage | OpenAI usage |
/// |---|---|---|---|
/// | `input` | input price | `input_tokens` | `prompt_tokens - prompt_tokens_details.cached_tokens` |
/// | `cache_read` | cache-read price | `cache_read_input_tokens` | `prompt_tokens_details.cached_tokens` |
/// | `cache_write` | cache-write price | `cache_creation_input_tokens` | (not reported) |
/// | `output` | output price | `output_tokens` | `completion_tokens - completion_tokens_details.reasoning_tokens` |
/// | `reasoning` | output price | (included in `output_tokens`) | `completion_tokens_details.reasoning_tokens` |
///
/// Anthropic's `input_tokens` already excludes cached tokens; OpenAI's
/// `prompt_tokens` includes them, so subtract. Splitting reasoning out of
/// `output` is optional (both are billed at the output price); it only keeps
/// the count visible.
///
/// ```
/// use llm_wasm::types::Usage;
/// let u = Usage::new(1_200, 300).with_cache_read(50_000).with_cache_write(2_000);
/// assert_eq!(u.total(), 53_500);
/// ```
///
/// `#[non_exhaustive]`: build it with [`Usage::new`] and the `with_*` setters
/// (or [`Default`]), so fields can be added without a breaking release.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct Usage {
    /// Uncached input (prompt) tokens.
    pub input: u64,
    /// Visible output (completion) tokens.
    pub output: u64,
    /// Input tokens served from the provider's prompt cache.
    pub cache_read: u64,
    /// Input tokens written to the provider's prompt cache.
    pub cache_write: u64,
    /// Reasoning ("thinking") tokens not already counted in `output`.
    pub reasoning: u64,
}

impl Usage {
    /// Uncached input and visible output tokens; the other counts are zero.
    pub fn new(input: u64, output: u64) -> Self {
        Self {
            input,
            output,
            ..Self::default()
        }
    }

    /// Set the uncached input token count.
    pub fn with_input(mut self, tokens: u64) -> Self {
        self.input = tokens;
        self
    }

    /// Set the visible output token count.
    pub fn with_output(mut self, tokens: u64) -> Self {
        self.output = tokens;
        self
    }

    /// Set the number of input tokens read from the prompt cache.
    pub fn with_cache_read(mut self, tokens: u64) -> Self {
        self.cache_read = tokens;
        self
    }

    /// Set the number of input tokens written to the prompt cache.
    pub fn with_cache_write(mut self, tokens: u64) -> Self {
        self.cache_write = tokens;
        self
    }

    /// Set the number of reasoning tokens (not already in `output`).
    pub fn with_reasoning(mut self, tokens: u64) -> Self {
        self.reasoning = tokens;
        self
    }

    /// All prompt-side tokens: `input + cache_read + cache_write`.
    pub fn total_input(&self) -> u64 {
        self.input
            .saturating_add(self.cache_read)
            .saturating_add(self.cache_write)
    }

    /// All generated tokens: `output + reasoning`.
    pub fn total_output(&self) -> u64 {
        self.output.saturating_add(self.reasoning)
    }

    /// Every token in the call (saturating).
    pub fn total(&self) -> u64 {
        self.total_input().saturating_add(self.total_output())
    }

    /// Read the `usage` of an OpenAI-compatible or Anthropic response.
    ///
    /// `json` is either the whole response body or its `usage` object. The
    /// shape is detected from the field names:
    ///
    /// - OpenAI Chat Completions and compatible APIs (`prompt_tokens`,
    ///   `completion_tokens`, `prompt_tokens_details.cached_tokens`,
    ///   `completion_tokens_details.reasoning_tokens`): cached tokens are
    ///   moved from `input` to `cache_read`, reasoning from `output` to
    ///   `reasoning`. DeepSeek's `prompt_cache_hit_tokens` is read too.
    /// - OpenAI Responses API (`input_tokens`, `output_tokens`,
    ///   `input_tokens_details`, `output_tokens_details`): the same split.
    /// - Anthropic Messages (`input_tokens`, `output_tokens`,
    ///   `cache_read_input_tokens`, `cache_creation_input_tokens`): taken as
    ///   is, since Anthropic's `input_tokens` already excludes the cache.
    ///
    /// Missing or non-numeric fields count as zero, and the parts never
    /// exceed the totals they are subtracted from, so a malformed body gives
    /// smaller counts, never a panic. Returns `None` when `json` has neither
    /// shape.
    ///
    /// ```
    /// use llm_wasm::types::Usage;
    /// let body = serde_json::json!({
    ///     "choices": [],
    ///     "usage": {
    ///         "prompt_tokens": 2_000, "completion_tokens": 300,
    ///         "prompt_tokens_details": {"cached_tokens": 1_500},
    ///         "completion_tokens_details": {"reasoning_tokens": 100}
    ///     }
    /// });
    /// let u = Usage::from_json(&body).unwrap();
    /// assert_eq!(u, Usage::new(500, 200).with_cache_read(1_500).with_reasoning(100));
    /// ```
    pub fn from_json(json: &serde_json::Value) -> Option<Usage> {
        let u = json.get("usage").filter(|u| u.is_object()).unwrap_or(json);
        let n = |v: Option<&serde_json::Value>| v.and_then(serde_json::Value::as_u64).unwrap_or(0);
        let at = |path: &[&str]| n(path.iter().try_fold(u, |v, k| v.get(k)));
        if u.get("prompt_tokens").is_some() || u.get("completion_tokens").is_some() {
            let prompt = at(&["prompt_tokens"]);
            let completion = at(&["completion_tokens"]);
            let cached = at(&["prompt_tokens_details", "cached_tokens"])
                .max(at(&["prompt_cache_hit_tokens"]))
                .min(prompt);
            let reasoning = at(&["completion_tokens_details", "reasoning_tokens"]).min(completion);
            return Some(
                Usage::new(prompt - cached, completion - reasoning)
                    .with_cache_read(cached)
                    .with_reasoning(reasoning),
            );
        }
        if u.get("input_tokens").is_none() && u.get("output_tokens").is_none() {
            return None;
        }
        let input = at(&["input_tokens"]);
        let output = at(&["output_tokens"]);
        if u.get("input_tokens_details").is_some() || u.get("output_tokens_details").is_some() {
            // OpenAI Responses API: input_tokens includes the cached part.
            let cached = at(&["input_tokens_details", "cached_tokens"]).min(input);
            let reasoning = at(&["output_tokens_details", "reasoning_tokens"]).min(output);
            return Some(
                Usage::new(input - cached, output - reasoning)
                    .with_cache_read(cached)
                    .with_reasoning(reasoning),
            );
        }
        Some(
            Usage::new(input, output)
                .with_cache_read(at(&["cache_read_input_tokens"]))
                .with_cache_write(at(&["cache_creation_input_tokens"])),
        )
    }
}

impl std::ops::Add for Usage {
    type Output = Usage;

    /// Field-wise saturating sum, for totalling a conversation or an agent run.
    fn add(self, rhs: Usage) -> Usage {
        Usage {
            input: self.input.saturating_add(rhs.input),
            output: self.output.saturating_add(rhs.output),
            cache_read: self.cache_read.saturating_add(rhs.cache_read),
            cache_write: self.cache_write.saturating_add(rhs.cache_write),
            reasoning: self.reasoning.saturating_add(rhs.reasoning),
        }
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Usage) {
        *self = *self + rhs;
    }
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
    fn test_usage_builders_totals_and_sum() {
        let u = Usage::new(10, 5)
            .with_cache_read(100)
            .with_cache_write(20)
            .with_reasoning(7);
        assert_eq!(
            (u.total_input(), u.total_output(), u.total()),
            (130, 12, 142)
        );
        let mut sum = Usage::default();
        sum += u;
        sum += Usage::new(1, 1);
        assert_eq!(
            sum,
            Usage::new(11, 6)
                .with_cache_read(100)
                .with_cache_write(20)
                .with_reasoning(7)
        );
        assert_eq!((Usage::new(u64::MAX, 0) + Usage::new(1, 0)).input, u64::MAX);
        // missing fields read as zero
        let back: Usage = serde_json::from_str(r#"{"input":3,"cache_read":4}"#).unwrap();
        assert_eq!(back, Usage::new(3, 0).with_cache_read(4));
        let r = ChatResponse {
            content: String::new(),
            model: "m".into(),
            input_tokens: 9,
            output_tokens: 2,
        };
        assert_eq!(r.usage(), Usage::new(9, 2));
    }

    #[test]
    fn test_usage_from_json_shapes() {
        use serde_json::json;
        // Anthropic: input_tokens already excludes the cache
        let a = json!({"usage": {"input_tokens": 30, "output_tokens": 9,
            "cache_read_input_tokens": 1_000, "cache_creation_input_tokens": 200}});
        assert_eq!(
            Usage::from_json(&a),
            Some(
                Usage::new(30, 9)
                    .with_cache_read(1_000)
                    .with_cache_write(200)
            )
        );
        // OpenAI Responses API: the cached part is inside input_tokens
        let r = json!({"input_tokens": 1_000, "output_tokens": 50,
            "input_tokens_details": {"cached_tokens": 800}, "output_tokens_details": {"reasoning_tokens": 20}});
        assert_eq!(
            Usage::from_json(&r),
            Some(Usage::new(200, 30).with_cache_read(800).with_reasoning(20))
        );
        // DeepSeek reports cache hits at the top level
        let d =
            json!({"prompt_tokens": 100, "completion_tokens": 5, "prompt_cache_hit_tokens": 60});
        assert_eq!(
            Usage::from_json(&d),
            Some(Usage::new(40, 5).with_cache_read(60))
        );
        // nonsense never underflows; unknown shapes are None
        let bad = json!({"prompt_tokens": 1, "completion_tokens": "x", "prompt_tokens_details": {"cached_tokens": 9}});
        assert_eq!(
            Usage::from_json(&bad),
            Some(Usage::new(0, 0).with_cache_read(1))
        );
        assert_eq!(Usage::from_json(&json!({"choices": []})), None);
        assert_eq!(Usage::from_json(&json!(null)), None);
        // a body whose usage is null falls back to reading the body itself
        assert_eq!(Usage::from_json(&json!({"usage": null})), None);
    }

    #[test]
    fn test_stream_chunk_finished_flag() {
        let chunk = StreamChunk {
            delta: "done".into(),
            finished: true,
        };
        assert!(chunk.finished);
    }

    #[test]
    fn test_request_serializes_to_openai_chat_body() {
        // 0.1 wrote "User" and nulls, which OpenAI-style APIs reject.
        let req = ChatRequest::new(
            "gpt-4o-mini",
            vec![
                ChatMessage::new(Role::System, "Be brief."),
                ChatMessage::new(Role::User, "Hi"),
            ],
        );
        let body = serde_json::to_value(&req).unwrap();
        assert_eq!(
            body,
            serde_json::json!({"model": "gpt-4o-mini", "messages": [
                {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]})
        );
        // old capitalised JSON and OpenAI's "developer" role still read
        let old: ChatRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"User","content":"a"},{"role":"developer","content":"b"}],"max_tokens":null,"temperature":null}"#,
        ).unwrap();
        assert_eq!(old.messages[0].role, Role::User);
        assert_eq!(old.messages[1].role, Role::System);
    }
}
