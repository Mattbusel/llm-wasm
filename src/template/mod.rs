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
    /// * `name` — partial name, referenced as `{{>name}}` in templates
    /// * `template` — the template text for this partial
    pub fn register_partial(&mut self, name: &str, template: &str) {
        self.partials.insert(name.to_string(), template.to_string());
    }

    /// Render a template string by substituting `{{key}}` and `{{>partial}}` tokens.
    ///
    /// Unknown `{{key}}` placeholders are left unchanged.
    /// Unknown `{{>partial}}` references return a [`LlmWasmError::TemplateError`].
    ///
    /// # Arguments
    /// * `template` — template string containing `{{...}}` tokens
    /// * `context` — variable bindings
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
                let rendered = self.render(partial_body, context)?;
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
}
