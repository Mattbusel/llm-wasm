//! llm-wasm in a Dioxus web app.
//!
//! ```text
//! cd examples/browser/dioxus-chat
//! dx serve --platform web   # or: cargo build --target wasm32-unknown-unknown
//! ```
//!
//! Enter an OpenAI-compatible base URL, key and model, ask something, and
//! watch the attempt count, cache hits and spend. The request logic
//! (retries with real browser delays, TTL cache, cost ledger) is in
//! `../shared/chat.rs`, shared with the Leptos example.

use std::cell::RefCell;
use std::rc::Rc;

use dioxus::prelude::*;

#[path = "../../shared/chat.rs"]
mod chat;

use chat::{ask, Reply, Session, Settings};

fn main() {
    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    let defaults = Settings::default();
    let mut base_url = use_signal(|| defaults.base_url.clone());
    let mut api_key = use_signal(|| defaults.api_key.clone());
    let mut model = use_signal(|| defaults.model.clone());
    let mut prompt = use_signal(|| String::from("Name three uses of WebAssembly, one line each."));
    let mut busy = use_signal(|| false);
    let mut answer = use_signal(|| None::<Result<Reply, String>>);
    let session = use_hook(|| Rc::new(RefCell::new(Session::new())));
    let mut spent = use_signal(|| (0.0_f64, 0.50_f64));

    let send = {
        let session = Rc::clone(&session);
        move |_| {
            let settings = Settings { base_url: base_url(), api_key: api_key(), model: model() };
            let question = prompt();
            let session = Rc::clone(&session);
            busy.set(true);
            spawn(async move {
                let result = ask(&settings, &session, &question).await;
                spent.set((session.borrow().spent_usd(), session.borrow().budget_usd()));
                answer.set(Some(result));
                busy.set(false);
            });
        }
    };

    let (s, b) = spent();
    rsx! {
        main { style: "max-width: 42rem; margin: 2rem auto; font-family: system-ui, sans-serif; padding: 0 1rem;",
            h1 { "llm-wasm + Dioxus" }
            label { "Base URL " input { value: "{base_url}", oninput: move |e| base_url.set(e.value()) } }
            br {}
            label { "API key " input { r#type: "password", value: "{api_key}", oninput: move |e| api_key.set(e.value()) } }
            br {}
            label { "Model " input { value: "{model}", oninput: move |e| model.set(e.value()) } }
            br {}
            textarea { rows: "4", style: "width: 100%;", value: "{prompt}", oninput: move |e| prompt.set(e.value()) }
            br {}
            button { onclick: send, disabled: busy(), if busy() { "Asking..." } else { "Ask" } }
            p { "Spent ${s:.6} of ${b:.2}" }
            match answer() {
                Some(Ok(reply)) => rsx! {
                    div {
                        pre { style: "white-space: pre-wrap;", "{reply.text}" }
                        small { "{describe(&reply)}" }
                    }
                },
                Some(Err(e)) => rsx! { p { style: "color: crimson;", "{e}" } },
                None => rsx! {},
            }
        }
    }
}

fn describe(reply: &Reply) -> String {
    if reply.cached {
        return "From the cache: no request, $0".into();
    }
    let cost = reply.cost_usd.map_or_else(|| "model not in the price table".into(), |c| format!("${c:.6}"));
    format!(
        "{} attempt(s), {} input + {} cached + {} output tokens, {cost}",
        reply.attempts, reply.usage.input, reply.usage.cache_read, reply.usage.total_output()
    )
}
