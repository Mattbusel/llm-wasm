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

#[cfg(feature = "repair")]
#[cfg_attr(docsrs, doc(cfg(feature = "repair")))]
pub mod repair;

/// Each `{` / `[` tried as the start of a JSON value is scanned forward to
/// its closing bracket (or the end of the text), so a long text full of
/// stray brackets took quadratic time in 0.1 (170 ms for 20,000 `{`).
/// Extraction now stops starting new tries once this many bytes have been
/// scanned in total. Ordinary model answers never get near it; a value that
/// only appears after thousands of unclosed brackets is not found.
pub const SCAN_BUDGET_BYTES: usize = 4 * 1024 * 1024;

/// Start positions of `{` / `[` (or only `only`), cut off once the scans
/// from earlier starts used up [`SCAN_BUDGET_BYTES`] (assuming the worst
/// case: each scan runs to the end of the text).
fn candidate_starts(bytes: &[u8], only: Option<u8>) -> impl Iterator<Item = usize> + '_ {
    let mut spent = 0usize;
    bytes
        .iter()
        .enumerate()
        .filter(move |&(_, &b)| only.map_or(b == b'{' || b == b'[', |o| b == o))
        .map(|(i, _)| i)
        .take_while(move |&i| {
            let ok = spent < SCAN_BUDGET_BYTES;
            spent = spent.saturating_add(bytes.len() - i);
            ok
        })
}

/// A backslash before a non-ASCII character is never a valid JSON escape;
/// models produce it in Windows paths (`C:\été`). Keep the backslash by
/// escaping it, instead of letting JSON5 drop it.
#[cfg(feature = "repair")]
fn escape_stray_backslashes(s: &str) -> std::borrow::Cow<'_, str> {
    let mut prev_backslashes = 0usize;
    let needs = s.chars().any(|c| {
        let hit = !c.is_ascii() && prev_backslashes % 2 == 1;
        prev_backslashes = if c == '\\' { prev_backslashes + 1 } else { 0 };
        hit
    });
    if !needs {
        return std::borrow::Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len() + 8);
    let mut run = 0usize;
    for c in s.chars() {
        if !c.is_ascii() && run % 2 == 1 {
            out.push('\\');
        }
        run = if c == '\\' { run + 1 } else { 0 };
        out.push(c);
    }
    std::borrow::Cow::Owned(out)
}

/// End (exclusive) of the bracketed value starting at `start`, tracking
/// double-quoted strings and escapes; `None` if it never closes.
fn balanced_end(bytes: &[u8], start: usize) -> Option<usize> {
    let open = bytes[start];
    let close = if open == b'{' { b'}' } else { b']' };
    let mut depth: i32 = 0;
    let mut in_string = false;
    let mut escape_next = false;
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
                return Some(start + i + 1);
            }
        }
    }
    None
}

/// Utilities for extracting and validating JSON from LLM output.
pub struct JsonFormatter;

impl JsonFormatter {
    /// Extract the first valid JSON object or array from a text string.
    ///
    /// Scans forward for `{` or `[`, then finds the matching close brace/bracket
    /// by counting depth. The extracted slice is then validated with `serde_json`.
    ///
    /// # Arguments
    /// * `text`: raw text that may contain prose before/after the JSON
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
        for start in candidate_starts(bytes, None) {
            if let Some(end) = balanced_end(bytes, start) {
                if let Ok(value) = serde_json::from_str(&text[start..end]) {
                    return Ok(value);
                }
            }
        }
        Err(LlmWasmError::Serialization("no valid JSON found in text".into()))
    }

    /// Like [`extract_json`](Self::extract_json), but when no strictly valid
    /// JSON is found, repairs the almost-JSON models often produce, using
    /// [json5](https://crates.io/crates/json5) plus a bounded pre-pass:
    /// trailing commas, single quotes, unquoted keys, comments, Python
    /// `True`/`None`, raw newlines inside strings, and output cut off
    /// mid-object (see [`repair`]). Missing commas
    /// between members are not repaired. Needs the `repair` feature.
    ///
    /// Valid JSON is taken first, in text order, exactly as by `extract_json`.
    /// A bracket that never closes is repaired from there to the end, because
    /// cut-off output contains everything after it. Otherwise each bracketed
    /// span is repaired on its own, objects before arrays (so prose like
    /// "[citation needed]" is not taken for an array). Only an object or
    /// array is accepted, so plain prose is still an error rather than being
    /// "repaired" into a JSON string.
    ///
    /// ```rust
    /// use llm_wasm::format::JsonFormatter;
    /// let v = JsonFormatter::extract_json_lenient("Sure: {'city': 'Paris', 'tags': ['eu',], 'big': True")?;
    /// assert_eq!(v, serde_json::json!({"city": "Paris", "tags": ["eu"], "big": true}));
    /// assert!(JsonFormatter::extract_json_lenient("The answer is Paris.").is_err());
    /// # Ok::<(), llm_wasm::LlmWasmError>(())
    /// ```
    ///
    /// # Errors
    /// [`LlmWasmError::Serialization`] if nothing usable is found.
    #[cfg(feature = "repair")]
    #[cfg_attr(docsrs, doc(cfg(feature = "repair")))]
    pub fn extract_json_lenient(text: &str) -> Result<serde_json::Value, LlmWasmError> {
        let bytes = text.as_bytes();
        let repaired = |slice: &str| repair::repair(&escape_stray_backslashes(slice));
        // In text order: the first valid value wins, as in `extract_json`; a
        // bracket that never closes means the output was cut off, and
        // everything after it is inside it, so it is repaired right away.
        let mut broken = Vec::new();
        for start in candidate_starts(bytes, None) {
            match balanced_end(bytes, start) {
                Some(end) => {
                    if let Ok(v) = serde_json::from_str(&text[start..end]) {
                        return Ok(v);
                    }
                    broken.push((start, end));
                }
                None => {
                    if let Some(v) = repaired(&text[start..]) {
                        return Ok(v);
                    }
                }
            }
        }
        // Then repair the closed-but-invalid spans, objects before arrays.
        broken.sort_by_key(|&(start, _)| (bytes[start] != b'{', start));
        for (start, end) in broken {
            if let Some(v) = repaired(&text[start..end]) {
                return Ok(v);
            }
        }
        Err(LlmWasmError::Serialization("no JSON object or array found in text, even after repair".into()))
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
    /// * `text`: raw LLM output, possibly wrapped in a code fence
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

    #[cfg(feature = "repair")]
    #[test]
    fn test_lenient_repairs_common_model_mistakes() {
        use serde_json::json;
        let cases = [
            (r#"{"a": 1, "b": [1,2,],}"#, json!({"a": 1, "b": [1, 2]})),
            (r#"{'a': 'x'}"#, json!({"a": "x"})),
            ("{a: 1}", json!({"a": 1})),
            ("{\"a\": 1 // one\n}", json!({"a": 1})),
            (r#"{"a": True, "b": None}"#, json!({"a": true, "b": null})),
            (r#"{"a": 1, "b": "hel"#, json!({"a": 1, "b": "hel"})),
            ("```json\n{\"items\": [{\"id\": 1}, {\"id\": 2", json!({"items": [{"id": 1}, {"id": 2}]})),
            ("{\"a\": \"line1\nline2\"}", json!({"a": "line1\nline2"})),
            ("[citation needed] {\"a\": 1,}", json!({"a": 1})),
            ("{'a': 1} hope that helps", json!({"a": 1})),
            ("{'a': 1} {'b': 2}", json!({"a": 1})),
            ("Use {name} as a placeholder. {\"a\": 1}", json!({"a": 1})),
            ("[1, 2,]", json!([1, 2])),
        ];
        for (input, want) in cases {
            assert_eq!(JsonFormatter::extract_json_lenient(input).unwrap(), want, "input {input:?}");
        }
        // valid JSON is returned untouched, prose is still an error
        assert_eq!(JsonFormatter::extract_json_lenient(r#"x {"n": 12345678901234567890} y"#).unwrap(), json!({"n": 12345678901234567890u64}));
        assert!(JsonFormatter::extract_json_lenient("The answer is Paris.").is_err());
        assert!(JsonFormatter::extract_json_lenient("").is_err());
        // strict extraction still refuses broken JSON
        assert!(JsonFormatter::extract_json(r#"{'a': 1}"#).is_err());
    }

    #[test]
    fn test_many_brackets_do_not_take_quadratic_time() {
        // 0.1 tried every bracket as a start and scanned to the end each time.
        let text = "{".repeat(200_000);
        let t = std::time::Instant::now();
        assert!(JsonFormatter::extract_json(&text).is_err());
        assert!(t.elapsed().as_secs_f64() < 2.0, "{:?}", t.elapsed());
        // within the budget, JSON after many stray brackets is still found
        let text = format!("{} {{\"a\": 1}}", "{".repeat(1_000));
        assert_eq!(JsonFormatter::extract_json(&text).unwrap(), serde_json::json!({"a": 1}));
    }

    #[cfg(feature = "repair")]
    #[test]
    fn test_backslash_before_non_ascii_does_not_crash() {
        // found by proptest (with jsonrepair, which panicked here); kept as a regression test
        // non-ASCII character
        let input = r#"{"path": "C:\été", 'x': 1,"#;
        let v = JsonFormatter::extract_json_lenient(input).unwrap();
        assert_eq!(v["path"], r"C:\été");
        assert_eq!(v["x"], 1);
        // an already escaped backslash is left alone
        assert_eq!(escape_stray_backslashes(r"a\\é"), r"a\\é");
        assert_eq!(escape_stray_backslashes(r"a\é"), r"a\\é");
    }
}
