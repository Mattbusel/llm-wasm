//! Repair of almost-JSON (feature `repair`).
//!
//! Parsing is done by [json5](https://crates.io/crates/json5), which accepts
//! single quotes, unquoted keys, comments, trailing commas, hex numbers and
//! `Infinity`/`NaN`. Before parsing, one linear pass fixes what JSON5 does not
//! cover but models often produce: Python `True` / `False` / `None`, raw line
//! breaks inside strings, and output cut off mid-value (unterminated strings
//! and unclosed brackets are closed; a dangling key or value is dropped).
//!
//! Every step is bounded: one scan of the input plus at most
//! [`MAX_TRIM_RETRIES`] re-parses, so hostile input cannot make it loop or
//! allocate without limit.

/// How many times a cut-off value is shortened to the previous complete
/// member before giving up.
pub const MAX_TRIM_RETRIES: usize = 8;

struct Scan {
    /// Normalised text (Python literals and raw newlines fixed).
    out: String,
    /// Open brackets at the end of the text, outermost first.
    stack: Vec<u8>,
    /// Ended inside a string with this quote character.
    open_quote: Option<char>,
    /// Recent commas outside strings: (byte offset in `out`, stack at that point).
    commas: Vec<(usize, Vec<u8>)>,
}

fn scan(input: &str) -> Scan {
    let mut out = String::with_capacity(input.len() + 8);
    let mut stack: Vec<u8> = Vec::new();
    let mut quote: Option<char> = None;
    let mut escape = false;
    let mut commas: Vec<(usize, Vec<u8>)> = Vec::new();
    let mut chars = input.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if let Some(q) = quote {
            if escape {
                escape = false;
                out.push(c);
            } else if c == '\\' {
                escape = true;
                out.push(c);
            } else if c == q {
                quote = None;
                out.push(c);
            } else if c == '\n' {
                out.push_str("\\n");
            } else if c == '\r' {
                out.push_str("\\r");
            } else if c == '\t' {
                out.push_str("\\t");
            } else {
                out.push(c);
            }
            continue;
        }
        match c {
            '"' | '\'' => {
                quote = Some(c);
                out.push(c);
            }
            '{' | '[' => {
                stack.push(c as u8);
                out.push(c);
            }
            '}' | ']' => {
                stack.pop();
                out.push(c);
            }
            ',' => {
                if commas.len() == MAX_TRIM_RETRIES {
                    commas.remove(0);
                }
                commas.push((out.len(), stack.clone()));
                out.push(c);
            }
            c if c.is_ascii_alphabetic() => {
                // Read a whole identifier and map Python literals.
                let start = i;
                let mut end = i + c.len_utf8();
                while let Some(&(j, d)) = chars.peek() {
                    if d.is_ascii_alphanumeric() || d == '_' {
                        end = j + d.len_utf8();
                        chars.next();
                    } else {
                        break;
                    }
                }
                let word = &input[start..end];
                out.push_str(match word {
                    "True" => "true",
                    "False" => "false",
                    "None" => "null",
                    other => other,
                });
            }
            _ => out.push(c),
        }
    }
    if escape {
        // a lone backslash at the very end of a cut-off string
        out.pop();
    }
    Scan { out, stack, open_quote: quote, commas }
}

fn close(mut text: String, stack: &[u8]) -> String {
    // drop a dangling separator
    loop {
        let t = text.trim_end();
        if t.ends_with(',') || t.ends_with(':') {
            let n = t.len() - 1;
            text.truncate(n);
        } else {
            text.truncate(t.len());
            break;
        }
    }
    for &b in stack.iter().rev() {
        text.push(if b == b'{' { '}' } else { ']' });
    }
    text
}

fn parse(text: &str) -> Option<serde_json::Value> {
    json5::from_str::<serde_json::Value>(text).ok().filter(|v| v.is_object() || v.is_array())
}

/// Repair and parse one candidate that starts at `{` or `[`.
pub(crate) fn repair(candidate: &str) -> Option<serde_json::Value> {
    let s = scan(candidate);
    let mut full = s.out.clone();
    if let Some(q) = s.open_quote {
        full.push(q);
    }
    if let Some(v) = parse(&close(full, &s.stack)) {
        return Some(v);
    }
    // Cut back to earlier complete members (drops a half-written key or value).
    for (pos, stack) in s.commas.iter().rev() {
        if let Some(v) = parse(&close(s.out[..*pos].to_string(), stack)) {
            return Some(v);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closes_and_trims() {
        assert_eq!(close("{\"a\": 1,".into(), b"{"), "{\"a\": 1}");
        assert_eq!(close("[1, [2".into(), b"[["), "[1, [2]]");
        assert_eq!(repair("{\"a\": {\"b\": 1, \"c"), Some(serde_json::json!({"a": {"b": 1}})));
        assert_eq!(repair("{\"a\": {\"b\": 1, \"c\": "), Some(serde_json::json!({"a": {"b": 1}})));
        assert_eq!(repair("{\"s\": \"ends in backslash \\"), Some(serde_json::json!({"s": "ends in backslash "})));
    }

    #[test]
    fn python_words_only_outside_strings() {
        assert_eq!(repair("{'a': True, 'b': 'None of it', 'Nonesuch': None}"), Some(serde_json::json!({"a": true, "b": "None of it", "Nonesuch": null})));
    }
}
