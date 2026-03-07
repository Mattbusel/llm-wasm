//! # Module: Format
//!
//! ## Responsibility
//! Post-process LLM text output by extracting embedded JSON or stripping
//! Markdown code fences.
//!
//! ## Guarantees
//! - `extract_json` finds the first balanced `{...}` or `[...]` block
//! - `strip_code_fence` returns inner content without the fence lines
//! - No external parsing libraries required

use crate::error::LlmWasmError;

/// Utilities for extracting and validating JSON from LLM output.
pub struct JsonFormatter;

impl JsonFormatter {
    /// Extract the first valid JSON object or array from a text string.
    ///
    /// Scans forward for `{` or `[`, then finds the matching close brace/bracket
    /// by counting depth. The extracted slice is then validated with `serde_json`.
    ///
    /// # Arguments
    /// * `text` — raw text that may contain prose before/after the JSON
    ///
    /// # Returns
    /// The first parseable `serde_json::Value` found.
    ///
    /// # Errors
    /// Returns [`LlmWasmError::Serialization`] if no valid JSON is found.
    ///
    /// # Panics
    /// This function never panics.
    pub fn extract_json(text: &str) -> Result<serde_json::Value, LlmWasmError> {
        let bytes = text.as_bytes();
        for start in 0..bytes.len() {
            let open = bytes[start];
            let close = match open {
                b'{' => b'}',
                b'[' => b']',
                _ => continue,
            };

            // Walk forward counting depth
            let mut depth: i32 = 0;
            let mut in_string = false;
            let mut escape_next = false;
            let mut end_idx = None;

            for (i, &b) in bytes[start..].iter().enumerate() {
                if escape_next {
                    escape_next = false;
                    continue;
                }
                if b == b'\\' && in_string {
                    escape_next = true;
                    continue;
                }
                if b == b'"' {
                    in_string = !in_string;
                    continue;
                }
                if in_string {
                    continue;
                }
                if b == open {
                    depth += 1;
                } else if b == close {
                    depth -= 1;
                    if depth == 0 {
                        end_idx = Some(start + i + 1);
                        break;
                    }
                }
            }

            if let Some(end) = end_idx {
                let candidate = &text[start..end];
                if let Ok(value) = serde_json::from_str(candidate) {
                    return Ok(value);
                }
            }
        }
        Err(LlmWasmError::Serialization("no valid JSON found in text".into()))
    }

    /// Return `true` if `text` is a valid JSON value.
    pub fn is_valid_json(text: &str) -> bool {
        serde_json::from_str::<serde_json::Value>(text).is_ok()
    }
}

/// Utilities for stripping Markdown formatting from LLM output.
pub struct MarkdownFormatter;

impl MarkdownFormatter {
    /// Return `true` if `text` contains a Markdown code fence (` ``` `).
    pub fn has_code_fence(text: &str) -> bool {
        text.contains("```")
    }

    /// Remove the outermost ` ```lang ... ``` ` fence and return the inner content.
    ///
    /// If no fence is present the input is returned unchanged.
    ///
    /// Only the first fence block is stripped. The optional language tag on the
    /// opening fence line is discarded.
    ///
    /// # Arguments
    /// * `text` — raw LLM output, possibly wrapped in a code fence
    ///
    /// # Returns
    /// Inner content with leading/trailing whitespace trimmed, or the original
    /// text if no fence was detected.
    ///
    /// # Panics
    /// This function never panics.
    pub fn strip_code_fence(text: &str) -> String {
        // Find opening fence
        let open_start = match text.find("```") {
            Some(i) => i,
            None => return text.to_string(),
        };

        // The opening fence line ends at the next newline
        let after_open_fence = &text[open_start + 3..];
        let open_line_end = after_open_fence.find('\n').unwrap_or(after_open_fence.len());
        let inner_start = open_start + 3 + open_line_end + 1; // skip the newline

        // Find closing fence
        let remaining = &text[inner_start.min(text.len())..];
        match remaining.find("```") {
            Some(close_rel) => {
                let inner = &remaining[..close_rel];
                inner.trim().to_string()
            }
            None => text.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_formatter_extract_valid_json() {
        let value = JsonFormatter::extract_json(r#"{"key": "value"}"#).unwrap();
        assert_eq!(value["key"], "value");
    }

    #[test]
    fn test_json_formatter_extract_from_prose() {
        let text = r#"Sure! Here is the data: {"status": "ok", "count": 42} Hope that helps."#;
        let value = JsonFormatter::extract_json(text).unwrap();
        assert_eq!(value["status"], "ok");
        assert_eq!(value["count"], 42);
    }

    #[test]
    fn test_json_formatter_extract_array() {
        let text = "Result: [1, 2, 3]";
        let value = JsonFormatter::extract_json(text).unwrap();
        assert!(value.is_array());
    }

    #[test]
    fn test_json_formatter_invalid_returns_error() {
        let result = JsonFormatter::extract_json("no json here at all");
        assert!(matches!(result, Err(LlmWasmError::Serialization(_))));
    }

    #[test]
    fn test_json_formatter_is_valid_json_true() {
        assert!(JsonFormatter::is_valid_json(r#"{"a": 1}"#));
    }

    #[test]
    fn test_json_formatter_is_valid_json_false() {
        assert!(!JsonFormatter::is_valid_json("{bad json}"));
    }

    #[test]
    fn test_markdown_strip_code_fence() {
        let text = "```json\n{\"key\": \"val\"}\n```";
        let stripped = MarkdownFormatter::strip_code_fence(text);
        assert_eq!(stripped, r#"{"key": "val"}"#);
    }

    #[test]
    fn test_markdown_no_fence_unchanged() {
        let text = "plain text without fences";
        assert_eq!(MarkdownFormatter::strip_code_fence(text), text);
    }

    #[test]
    fn test_markdown_has_code_fence_true() {
        assert!(MarkdownFormatter::has_code_fence("```rust\nfn main() {}\n```"));
    }

    #[test]
    fn test_markdown_has_code_fence_false() {
        assert!(!MarkdownFormatter::has_code_fence("no fences here"));
    }

    #[test]
    fn test_markdown_strip_fence_no_language_tag() {
        let text = "```\nhello\n```";
        assert_eq!(MarkdownFormatter::strip_code_fence(text), "hello");
    }

    #[test]
    fn test_json_formatter_nested_object() {
        let text = r#"{"outer": {"inner": 99}}"#;
        let value = JsonFormatter::extract_json(text).unwrap();
        assert_eq!(value["outer"]["inner"], 99);
    }
}
