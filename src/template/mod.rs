//! # Module: Template
//!
//! ## Responsibility
//! Minimal `{{variable}}` and `{{>partial}}` substitution engine with no
//! external dependencies. Loops and conditionals are intentionally out of scope.
//!
//! ## Guarantees
//! - Unknown `{{variable}}` placeholders are left verbatim in output
//! - Missing partials produce a [`LlmWasmError::TemplateError`]
//! - No heap allocation beyond the output string

use std::collections::HashMap;

use crate::error::LlmWasmError;

/// How deep partials may include other partials before rendering stops.
pub const MAX_PARTIAL_DEPTH: usize = 32;

/// Simple `{{variable}}` / `{{>partial}}` template engine.
///
/// # Example
/// ```rust
/// use std::collections::HashMap;
/// use llm_wasm::template::TemplateEngine;
///
/// let engine = TemplateEngine::new();
/// let mut ctx = HashMap::new();
/// ctx.insert("name".into(), "world".into());
/// let out = engine.render("Hello, {{name}}!", &ctx).unwrap();
/// assert_eq!(out, "Hello, world!");
/// ```
pub struct TemplateEngine {
    partials: HashMap<String, String>,
}

impl Default for TemplateEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TemplateEngine {
    /// Create a new engine with no registered partials.
    pub fn new() -> Self {
        Self { partials: HashMap::new() }
    }

    /// Register a named partial template.
    ///
    /// # Arguments
    /// * `name`: partial name, referenced as `{{>name}}` in templates
    /// * `template`: the template text for this partial
    pub fn register_partial(&mut self, name: &str, template: &str) {
        self.partials.insert(name.to_string(), template.to_string());
    }

    /// Render a template string by substituting `{{key}}` and `{{>partial}}` tokens.
    ///
    /// Unknown `{{key}}` placeholders are left unchanged.
    /// Unknown `{{>partial}}` references return a [`LlmWasmError::TemplateError`].
    ///
    /// # Arguments
    /// * `template`: template string containing `{{...}}` tokens
    /// * `context`: variable bindings
    ///
    /// # Errors
    /// Returns [`LlmWasmError::TemplateError`] if a `{{>partial}}` references an
    /// unregistered partial name.
    ///
    /// # Panics
    /// This function never panics.
    pub fn render(
        &self,
        template: &str,
        context: &HashMap<String, String>,
    ) -> Result<String, LlmWasmError> {
        self.render_depth(template, context, 0)
    }

    fn render_depth(
        &self,
        template: &str,
        context: &HashMap<String, String>,
        depth: usize,
    ) -> Result<String, LlmWasmError> {
        if depth > MAX_PARTIAL_DEPTH {
            return Err(LlmWasmError::TemplateError(format!(
                "partials nested more than {MAX_PARTIAL_DEPTH} deep (is a partial including itself?)"
            )));
        }
        let mut output = String::with_capacity(template.len());
        let mut remaining = template;

        while let Some(open) = remaining.find("{{") {
            // Emit text before the tag
            output.push_str(&remaining[..open]);
            remaining = &remaining[open + 2..];

            let close = remaining.find("}}").ok_or_else(|| {
                LlmWasmError::TemplateError("unclosed '{{' tag".into())
            })?;

            let tag = &remaining[..close];
            remaining = &remaining[close + 2..];

            if let Some(partial_name) = tag.strip_prefix('>') {
                let partial_name = partial_name.trim();
                let partial_body = self.partials.get(partial_name).ok_or_else(|| {
                    LlmWasmError::TemplateError(format!(
                        "unknown partial '>{partial_name}'"
                    ))
                })?;
                // Recursively render the partial with the same context
                let rendered = self.render_depth(partial_body, context, depth + 1)?;
                output.push_str(&rendered);
            } else {
                let key = tag.trim();
                match context.get(key) {
                    Some(value) => output.push_str(value),
                    None => {
                        // Leave unknown variables verbatim
                        output.push_str("{{");
                        output.push_str(tag);
                        output.push_str("}}");
                    }
                }
            }
        }

        // Emit any trailing text after the last tag
        output.push_str(remaining);
        Ok(output)
    }
}

/// Render a full Jinja2 template (loops, conditionals, filters) with
/// [minijinja](https://crates.io/crates/minijinja). Needs the `jinja` feature.
///
/// Undefined variables render as empty strings, as in Jinja2. Rendering is
/// limited by minijinja's fuel counter, so a template from an untrusted
/// source cannot run forever.
///
/// ```rust
/// let out = llm_wasm::template::render_jinja(
///     "{% for c in cities %}{{ loop.index }}. {{ c | upper }}\n{% endfor %}",
///     &serde_json::json!({"cities": ["paris", "lyon"]}),
/// )?;
/// assert_eq!(out, "1. PARIS\n2. LYON\n");
/// # Ok::<(), llm_wasm::LlmWasmError>(())
/// ```
///
/// # Errors
/// [`LlmWasmError::TemplateError`] with minijinja's message for syntax and
/// render errors (including `raise_exception(...)` calls).
#[cfg(feature = "jinja")]
#[cfg_attr(docsrs, doc(cfg(feature = "jinja")))]
pub fn render_jinja(template: &str, context: &serde_json::Value) -> Result<String, LlmWasmError> {
    jinja_env()
        .render_str(template, context)
        .map_err(|e| LlmWasmError::TemplateError(format!("{e:#}")))
}

#[cfg(feature = "jinja")]
fn jinja_env() -> minijinja::Environment<'static> {
    let mut env = minijinja::Environment::new();
    // Python string methods ("x".strip(), .split(), dict.items(), ...) that
    // Hugging Face chat templates use.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
    });
    env.set_fuel(Some(5_000_000));
    env
}

/// Turn a conversation into the exact prompt text a local model expects,
/// using the model's own chat template: the `chat_template` string from its
/// Hugging Face `tokenizer_config.json` (Llama, Mistral, Qwen, Gemma, Phi, ...).
/// Needs the `jinja` feature.
///
/// The template sees `messages` (each with lowercase `role` and `content`),
/// `add_generation_prompt`, `bos_token` and `eos_token`, as in the
/// `transformers` library.
///
/// ```rust
/// use llm_wasm::template::render_chat_template;
/// use llm_wasm::types::{ChatMessage, Role};
///
/// // ChatML, used by Qwen and others
/// let chatml = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";
/// let prompt = render_chat_template(chatml, &[ChatMessage::new(Role::User, "Hi")], true, "", "")?;
/// assert_eq!(prompt, "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
/// # Ok::<(), llm_wasm::LlmWasmError>(())
/// ```
///
/// # Errors
/// [`LlmWasmError::TemplateError`] for template errors, including the
/// template's own `raise_exception` checks (for example "roles must alternate").
#[cfg(feature = "jinja")]
#[cfg_attr(docsrs, doc(cfg(feature = "jinja")))]
pub fn render_chat_template(
    template: &str,
    messages: &[crate::types::ChatMessage],
    add_generation_prompt: bool,
    bos_token: &str,
    eos_token: &str,
) -> Result<String, LlmWasmError> {
    let ctx = serde_json::json!({
        "messages": messages,
        "add_generation_prompt": add_generation_prompt,
        "bos_token": bos_token,
        "eos_token": eos_token,
    });
    render_jinja(template, &ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn test_template_render_substitutes_variable() {
        let engine = TemplateEngine::new();
        let out = engine.render("Hello, {{name}}!", &ctx(&[("name", "Alice")])).unwrap();
        assert_eq!(out, "Hello, Alice!");
    }

    #[test]
    fn test_template_render_unknown_variable_leaves_placeholder() {
        let engine = TemplateEngine::new();
        let out = engine.render("Value: {{missing}}", &ctx(&[])).unwrap();
        assert_eq!(out, "Value: {{missing}}");
    }

    #[test]
    fn test_template_render_multiple_variables() {
        let engine = TemplateEngine::new();
        let out = engine
            .render("{{a}} and {{b}}", &ctx(&[("a", "foo"), ("b", "bar")]))
            .unwrap();
        assert_eq!(out, "foo and bar");
    }

    #[test]
    fn test_template_render_partial_substitution() {
        let mut engine = TemplateEngine::new();
        engine.register_partial("greeting", "Hello, {{name}}!");
        let out = engine
            .render("{{>greeting}} Welcome.", &ctx(&[("name", "Bob")]))
            .unwrap();
        assert_eq!(out, "Hello, Bob! Welcome.");
    }

    #[test]
    fn test_template_render_unknown_partial_errors() {
        let engine = TemplateEngine::new();
        let result = engine.render("{{>missing}}", &ctx(&[]));
        assert!(matches!(result, Err(LlmWasmError::TemplateError(_))));
    }

    #[test]
    fn test_template_render_no_tags_passthrough() {
        let engine = TemplateEngine::new();
        let out = engine.render("plain text", &ctx(&[])).unwrap();
        assert_eq!(out, "plain text");
    }

    #[test]
    fn test_template_render_unclosed_tag_errors() {
        let engine = TemplateEngine::new();
        let result = engine.render("Hello {{world", &ctx(&[]));
        assert!(matches!(result, Err(LlmWasmError::TemplateError(_))));
    }

    #[test]
    fn test_template_render_partial_with_variable() {
        let mut engine = TemplateEngine::new();
        engine.register_partial("sig", "-- {{author}}");
        let out = engine
            .render("Body. {{>sig}}", &ctx(&[("author", "Team")]))
            .unwrap();
        assert_eq!(out, "Body. -- Team");
    }

    #[test]
    fn test_recursive_partial_is_an_error_not_a_stack_overflow() {
        // 0.1.x recursed forever on a partial that includes itself and
        // crashed the process (or the WASM instance) with a stack overflow.
        let mut engine = TemplateEngine::new();
        engine.register_partial("loop", "again {{>loop}}");
        engine.register_partial("a", "{{>b}}");
        engine.register_partial("b", "{{>a}}");
        let ctx = HashMap::new();
        assert!(matches!(engine.render("{{>loop}}", &ctx), Err(LlmWasmError::TemplateError(_))));
        assert!(matches!(engine.render("{{>a}}", &ctx), Err(LlmWasmError::TemplateError(_))));
        // Legitimate nesting still works.
        engine.register_partial("outer", "[{{>inner}}]");
        engine.register_partial("inner", "x");
        assert_eq!(engine.render("{{>outer}}", &ctx).unwrap(), "[x]");
    }

    #[cfg(feature = "jinja")]
    mod jinja {
        use super::super::*;
        use crate::types::{ChatMessage, Role};

        const LLAMA3: &str = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}";
        const MISTRAL: &str = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}";

        fn convo() -> Vec<ChatMessage> {
            vec![
                ChatMessage::new(Role::User, "  What is 2+2? "),
                ChatMessage::new(Role::Assistant, "4"),
                ChatMessage::new(Role::User, "And 3+3?"),
            ]
        }

        #[test]
        fn llama3_template() {
            let out = render_chat_template(LLAMA3, &convo(), true, "<|begin_of_text|>", "<|eot_id|>").unwrap();
            assert_eq!(
                out,
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is 2+2?<|eot_id|>\
<|start_header_id|>assistant<|end_header_id|>\n\n4<|eot_id|>\
<|start_header_id|>user<|end_header_id|>\n\nAnd 3+3?<|eot_id|>\
<|start_header_id|>assistant<|end_header_id|>\n\n"
            );
        }

        #[test]
        fn mistral_template_and_raise_exception() {
            let out = render_chat_template(MISTRAL, &convo(), false, "<s>", "</s>").unwrap();
            assert_eq!(out, "<s>[INST]   What is 2+2?  [/INST]4</s>[INST] And 3+3? [/INST]");
            let bad = vec![ChatMessage::new(Role::User, "a"), ChatMessage::new(Role::User, "b")];
            let err = render_chat_template(MISTRAL, &bad, false, "<s>", "</s>").unwrap_err();
            assert!(err.to_string().contains("roles must alternate"), "{err}");
        }

        #[test]
        fn python_methods_and_errors() {
            let out = render_jinja("{{ x.strip().upper() }}|{{ ', '.join(xs) }}", &serde_json::json!({"x": " hi ", "xs": ["a", "b"]})).unwrap();
            assert_eq!(out, "HI|a, b");
            assert!(render_jinja("{% for %}", &serde_json::json!({})).is_err());
            assert_eq!(render_jinja("[{{ missing }}]", &serde_json::json!({})).unwrap(), "[]");
            // a runaway template stops instead of hanging
            assert!(render_jinja("{% for i in range(100000) %}{% for j in range(100000) %}x{% endfor %}{% endfor %}", &serde_json::json!({})).is_err());
        }
    }
}
