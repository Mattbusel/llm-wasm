//! Integration test: render template, format JSON output.

use std::collections::HashMap;

use llm_wasm::format::{JsonFormatter, MarkdownFormatter};
use llm_wasm::template::TemplateEngine;

fn ctx(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

#[test]
fn test_render_then_extract_json() {
    let engine = TemplateEngine::new();
    // Template produces a JSON string after substitution
    let template = r#"{"model": "{{model}}", "prompt": "{{prompt}}"}"#;
    let rendered = engine
        .render(template, &ctx(&[("model", "gpt-4o"), ("prompt", "hello")]))
        .unwrap();

    let value = JsonFormatter::extract_json(&rendered).unwrap();
    assert_eq!(value["model"], "gpt-4o");
    assert_eq!(value["prompt"], "hello");
}

#[test]
fn test_render_partial_then_strip_fence() {
    let mut engine = TemplateEngine::new();
    engine.register_partial("code_block", "```python\n{{code}}\n```");

    let rendered = engine
        .render("{{>code_block}}", &ctx(&[("code", "print('hello')")]))
        .unwrap();

    assert!(MarkdownFormatter::has_code_fence(&rendered));
    let stripped = MarkdownFormatter::strip_code_fence(&rendered);
    assert_eq!(stripped, "print('hello')");
}

#[test]
fn test_extract_json_from_prose_response() {
    // Simulate an LLM response that wraps JSON in prose
    let llm_output = "Sure! Here is the result:\n\n{\"answer\": 42, \"confidence\": 0.95}\n\nLet me know if you need more.";
    let value = JsonFormatter::extract_json(llm_output).unwrap();
    assert_eq!(value["answer"], 42);
}

#[test]
fn test_template_with_unknown_var_leaves_placeholder() {
    let engine = TemplateEngine::new();
    let out = engine
        .render("Hello {{name}}, your score is {{score}}.", &ctx(&[("name", "Alice")]))
        .unwrap();
    assert!(out.contains("Alice"));
    assert!(out.contains("{{score}}")); // unknown var left verbatim
}

#[test]
fn test_render_json_template_multiple_vars() {
    let engine = TemplateEngine::new();
    let template = r#"[{"role": "{{role}}", "content": "{{content}}"}]"#;
    let rendered = engine
        .render(
            template,
            &ctx(&[("role", "user"), ("content", "What is 2+2?")]),
        )
        .unwrap();

    let value = JsonFormatter::extract_json(&rendered).unwrap();
    assert!(value.is_array());
    assert_eq!(value[0]["role"], "user");
}
