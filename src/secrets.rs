//! Keep secrets out of prompts (feature `secrets`).
//!
//! [`SecretGuard`] looks for API keys, private keys, email addresses and
//! payment card numbers in a request before it is sent, using the
//! [regex](https://crates.io/crates/regex) crate. It either blocks the request
//! or replaces each match with a placeholder such as `[REDACTED:openai_key]`.
//! Card numbers must pass the Luhn checksum, so order numbers and phone
//! numbers are not flagged.
//!
//! ```
//! use llm_wasm::guard::{GuardChain};
//! use llm_wasm::secrets::SecretGuard;
//! use llm_wasm::types::{ChatMessage, ChatRequest, Role};
//!
//! let req = ChatRequest::new("gpt-4o", vec![ChatMessage::new(Role::User,
//!     "my key is sk-proj-AbCdEf0123456789XyZ0123456789 and my card 4111 1111 1111 1111")]);
//! let cleaned = GuardChain::new().add(SecretGuard::redact()).check(&req)?.expect("rewritten");
//! assert_eq!(cleaned.messages[0].content, "my key is [REDACTED:api_key] and my card [REDACTED:card_number]");
//! # Ok::<(), llm_wasm::LlmWasmError>(())
//! ```

use crate::guard::{Guard, GuardResult};
use crate::types::ChatRequest;
use regex::Regex;

/// What a [`SecretGuard`] does when it finds something.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecretAction {
    /// Refuse the request.
    Block,
    /// Replace each match with `[REDACTED:<kind>]` and let the request through.
    Redact,
}

struct Rule {
    kind: &'static str,
    re: Regex,
    luhn: bool,
}

/// Detects secrets and personal data in requests.
pub struct SecretGuard {
    rules: Vec<Rule>,
    action: SecretAction,
}

impl std::fmt::Debug for SecretGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecretGuard")
            .field("kinds", &self.rules.iter().map(|r| r.kind).collect::<Vec<_>>())
            .field("action", &self.action)
            .finish()
    }
}

// (kind, pattern, needs Luhn check). Compiled with `(?-u)`: ASCII classes are
// all these formats need, and leaving out the Unicode tables keeps the wasm
// module about 400 KB smaller.
const RULES: &[(&str, &str, bool)] = &[
    ("private_key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----(?su:.)*?-----END [A-Z ]*PRIVATE KEY-----", false),
    // OpenAI (sk-, sk-proj-), Anthropic (sk-ant-), OpenRouter (sk-or-) and similar
    ("api_key", r"\bsk-[A-Za-z0-9_-]{20,}", false),
    ("aws_access_key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", false),
    ("github_token", r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})", false),
    ("google_api_key", r"\bAIza[0-9A-Za-z_-]{35}", false),
    ("slack_token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}", false),
    ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b", false),
    ("card_number", r"\b\d(?:[ -]?\d){12,18}\b", true),
];

fn luhn_ok(s: &str) -> bool {
    let digits: Vec<u32> = s.chars().filter_map(|c| c.to_digit(10)).collect();
    if !(13..=19).contains(&digits.len()) {
        return false;
    }
    let sum: u32 = digits
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &d)| if i % 2 == 1 { let x = d * 2; if x > 9 { x - 9 } else { x } } else { d })
        .sum();
    sum % 10 == 0
}

impl SecretGuard {
    fn with_action(action: SecretAction) -> Self {
        let rules = RULES
            .iter()
            .filter_map(|&(kind, pat, luhn)| Regex::new(&format!("(?-u){pat}")).ok().map(|re| Rule { kind, re, luhn }))
            .collect();
        Self { rules, action }
    }

    /// Block any request that contains a secret.
    pub fn block() -> Self {
        Self::with_action(SecretAction::Block)
    }

    /// Replace secrets with `[REDACTED:<kind>]` and let the request through.
    pub fn redact() -> Self {
        Self::with_action(SecretAction::Redact)
    }

    /// Stop looking for one kind (`"email"`, `"card_number"`, `"api_key"`,
    /// `"aws_access_key"`, `"github_token"`, `"google_api_key"`,
    /// `"slack_token"`, `"private_key"`), for example to allow email addresses.
    pub fn allow(mut self, kind: &str) -> Self {
        self.rules.retain(|r| r.kind != kind);
        self
    }

    /// Every finding in `text` as `(kind, matched text)`, in rule order.
    pub fn find<'t>(&self, text: &'t str) -> Vec<(&'static str, &'t str)> {
        let mut out = Vec::new();
        for r in &self.rules {
            for m in r.re.find_iter(text) {
                if !r.luhn || luhn_ok(m.as_str()) {
                    out.push((r.kind, m.as_str()));
                }
            }
        }
        out
    }

    /// `text` with every finding replaced by `[REDACTED:<kind>]`.
    pub fn redact_text(&self, text: &str) -> String {
        let mut out = text.to_string();
        for r in &self.rules {
            out = r
                .re
                .replace_all(&out, |caps: &regex::Captures<'_>| {
                    let m = caps.get(0).map_or("", |m| m.as_str());
                    if !r.luhn || luhn_ok(m) { format!("[REDACTED:{}]", r.kind) } else { m.to_string() }
                })
                .into_owned();
        }
        out
    }
}

impl Guard for SecretGuard {
    fn name(&self) -> &str {
        "secrets"
    }

    fn check(&self, request: &ChatRequest) -> GuardResult {
        let found: Vec<&'static str> = request.messages.iter().flat_map(|m| self.find(&m.content)).map(|(k, _)| k).collect();
        if found.is_empty() {
            return GuardResult::Allow;
        }
        match self.action {
            SecretAction::Block => {
                let mut kinds = found;
                kinds.dedup();
                GuardResult::Block { reason: format!("request contains {}", kinds.join(", ")) }
            }
            SecretAction::Redact => {
                let mut req = request.clone();
                for m in &mut req.messages {
                    m.content = self.redact_text(&m.content);
                }
                GuardResult::Modify(req)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(text: &str) -> Vec<&'static str> {
        SecretGuard::block().find(text).into_iter().map(|(k, _)| k).collect()
    }

    #[test]
    fn finds_real_shapes() {
        assert_eq!(kinds("key sk-ant-api03-AbCdEfGhIjKlMnOpQrStUv-0123"), ["api_key"]);
        assert_eq!(kinds("AKIAIOSFODNN7EXAMPLE"), ["aws_access_key"]);
        assert_eq!(kinds("ghp_0123456789abcdefghijABCDEFGHIJ0123456789"), ["github_token"]);
        assert_eq!(kinds("mail me: ada.lovelace+llm@example.co.uk"), ["email"]);
        assert_eq!(kinds("card 4111-1111-1111-1111 exp 12/29"), ["card_number"]);
        assert_eq!(kinds("-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----"), ["private_key"]);
    }

    #[test]
    fn avoids_common_false_positives() {
        // words that contain "sk-", order numbers, phone numbers, version strings
        assert!(kinds("our risk-assessment-framework-for-small-teams doc").is_empty());
        assert!(kinds("order 1234 5678 9012 3456 shipped").is_empty()); // fails Luhn
        assert!(kinds("call +1 415 555 0132").is_empty());
        assert!(kinds("version 1.2.3, see docs@ section").is_empty());
        assert!(kinds("task-sk-12 done").is_empty());
    }

    #[test]
    fn block_and_redact() {
        let req = ChatRequest::new("m", vec![crate::types::ChatMessage::new(crate::types::Role::User, "token xoxb-1234567890-abcdef here")]);
        assert!(matches!(SecretGuard::block().check(&req), GuardResult::Block { reason } if reason.contains("slack_token")));
        match SecretGuard::redact().check(&req) {
            GuardResult::Modify(r) => assert_eq!(r.messages[0].content, "token [REDACTED:slack_token] here"),
            _ => panic!("expected a rewrite"),
        }
        let clean = ChatRequest::new("m", vec![crate::types::ChatMessage::new(crate::types::Role::User, "hello")]);
        assert!(matches!(SecretGuard::block().check(&clean), GuardResult::Allow));
        assert!(SecretGuard::block().allow("email").find("a@b.example.com").is_empty());
    }

    #[test]
    fn every_rule_compiles() {
        // a pattern that fails to compile would be skipped silently
        assert_eq!(SecretGuard::block().rules.len(), RULES.len());
    }

    #[test]
    fn luhn() {
        assert!(luhn_ok("4111111111111111"));
        assert!(luhn_ok("5500 0000 0000 0004"));
        assert!(!luhn_ok("4111111111111112"));
        assert!(!luhn_ok("12345"));
    }
}
