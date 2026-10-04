//! Property tests: no panics on any input, and invariants that must hold.
#![allow(clippy::unwrap_used)]

use std::collections::HashMap;

use llm_wasm::format::JsonFormatter;
use llm_wasm::retry::RetryPolicy;
use llm_wasm::template::TemplateEngine;
use proptest::prelude::*;

fn json_value() -> impl Strategy<Value = serde_json::Value> {
    let leaf = prop_oneof![
        Just(serde_json::Value::Null),
        any::<bool>().prop_map(serde_json::Value::Bool),
        any::<i32>().prop_map(serde_json::Value::from),
        "[a-z {}\\[\\]\"\\\\]{0,8}".prop_map(serde_json::Value::from),
    ];
    leaf.prop_recursive(3, 20, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..4).prop_map(serde_json::Value::Array),
            prop::collection::btree_map("[a-z]{1,4}", inner, 0..4).prop_map(|m| serde_json::Value::Object(m.into_iter().collect())),
        ]
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// A JSON object or array wrapped in prose without brackets is found exactly.
    #[test]
    fn extract_finds_embedded_json(v in json_value(), before in "[a-zA-Z .,!?\n]{0,40}", after in "[a-zA-Z .,!?\n]{0,40}") {
        prop_assume!(v.is_object() || v.is_array());
        let text = format!("{before}{}{after}", serde_json::to_string(&v).unwrap());
        prop_assert_eq!(JsonFormatter::extract_json(&text).unwrap(), v);
    }

    #[test]
    fn extract_never_panics(s in ".{0,200}") {
        let _ = JsonFormatter::extract_json(&s);
        let _ = llm_wasm::format::MarkdownFormatter::strip_code_fence(&s);
    }

    #[test]
    fn retry_delays_stay_within_bounds(base in 1u32..10_000, extra in 0u32..100_000, attempt in 0u32..100, r in any::<f64>()) {
        let max = base + extra;
        let p = RetryPolicy::new(5, base, max).unwrap();
        prop_assert!(p.delay_for_attempt(attempt) <= max);
        prop_assert!(p.delay_with_jitter(attempt, r) <= max);
        if r.is_finite() && (0.0..=1.0).contains(&r) && attempt >= 1 {
            // equal jitter never waits less than half the capped delay
            let capped = (u64::from(base) << (attempt - 1).min(30)).min(u64::from(max));
            prop_assert!(u64::from(p.delay_with_jitter(attempt, r)) * 2 + 1 >= capped);
        }
    }

    #[test]
    fn template_never_panics(t in ".{0,80}", key in "[a-z]{1,5}", val in ".{0,20}") {
        let mut ctx = HashMap::new();
        ctx.insert(key, val);
        let _ = TemplateEngine::new().render(&t, &ctx);
    }
}

#[cfg(feature = "repair")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Repair never panics, and never changes JSON that was already valid.
    #[test]
    fn lenient_agrees_with_strict_on_valid_json(v in json_value(), prose in "[a-zA-Z .,\n]{0,30}") {
        prop_assume!(v.is_object() || v.is_array());
        let text = format!("{prose}{}", serde_json::to_string(&v).unwrap());
        prop_assert_eq!(JsonFormatter::extract_json_lenient(&text).unwrap(), v);
    }

    /// Output cut off at any byte still gives a usable value or a clean error.
    #[test]
    fn lenient_handles_any_truncation(v in json_value(), cut in 0.0f64..1.0) {
        prop_assume!(v.is_object() || v.is_array());
        let full = serde_json::to_string(&v).unwrap();
        let mut at = ((full.len() as f64) * cut) as usize;
        while !full.is_char_boundary(at) { at -= 1; }
        if let Ok(r) = JsonFormatter::extract_json_lenient(&full[..at]) {
            prop_assert!(r.is_object() || r.is_array());
        }
        prop_assert_eq!(JsonFormatter::extract_json_lenient(&full).unwrap(), v);
    }

    #[test]
    fn lenient_never_panics(s in ".{0,200}") {
        let _ = JsonFormatter::extract_json_lenient(&s);
    }
}

#[cfg(feature = "stream")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// However the body is cut into pieces, the decoded text is the same.
    #[test]
    fn stream_split_invariance(words in prop::collection::vec("[a-zA-Z \u{e9}\u{1F600}]{0,6}", 1..12), cuts in prop::collection::vec(1usize..9, 1..40)) {
        let mut body = String::new();
        for w in &words {
            let chunk = serde_json::json!({"choices": [{"delta": {"content": w}}]});
            body.push_str(&format!("data: {chunk}\n\n"));
        }
        body.push_str("data: [DONE]\n\n");
        let bytes = body.as_bytes();
        let mut d = llm_wasm::stream::StreamDecoder::new();
        let mut out = String::new();
        let mut pos = 0;
        let mut i = 0;
        while pos < bytes.len() {
            let n = cuts[i % cuts.len()].min(bytes.len() - pos);
            for c in d.push(&bytes[pos..pos + n]).unwrap() {
                out.push_str(&c.delta);
            }
            pos += n;
            i += 1;
        }
        prop_assert_eq!(out, words.concat());
        prop_assert!(d.is_finished());
    }

    #[test]
    fn stream_never_panics(bytes in prop::collection::vec(any::<u8>(), 0..300)) {
        let mut d = llm_wasm::stream::StreamDecoder::new();
        let _ = d.push(&bytes);
    }
}
