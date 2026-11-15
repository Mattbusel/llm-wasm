//! llm-wasm in a Leptos (client-side rendered) app.
//!
//! ```text
//! cd examples/browser/leptos-chat
//! trunk serve            # or: cargo build --target wasm32-unknown-unknown
//! ```
//!
//! Enter an OpenAI-compatible base URL, key and model, ask something, and
//! watch the attempt count, cache hits and spend. The request logic
//! (retries with real browser delays, TTL cache, cost ledger) is in
//! `../shared/chat.rs`, shared with the Dioxus example.

use std::cell::RefCell;
use std::rc::Rc;

use leptos::prelude::*;
use leptos::task::spawn_local;

#[path = "../../shared/chat.rs"]
mod chat;

use chat::{ask, Reply, Session, Settings};

#[component]
fn App() -> impl IntoView {
    let defaults = Settings::default();
    let (base_url, set_base_url) = signal(defaults.base_url);
    let (api_key, set_api_key) = signal(defaults.api_key);
    let (model, set_model) = signal(defaults.model);
    let (prompt, set_prompt) = signal(String::from("Name three uses of WebAssembly, one line each."));
    let (busy, set_busy) = signal(false);
    let (answer, set_answer) = signal_local(None::<Result<Reply, String>>);
    let (spent, set_spent) = signal((0.0_f64, 0.0_f64));

    let session = Rc::new(RefCell::new(Session::new()));
    set_spent.set((session.borrow().spent_usd(), session.borrow().budget_usd()));

    let send = move |_| {
        let settings = Settings { base_url: base_url.get(), api_key: api_key.get(), model: model.get() };
        let question = prompt.get();
        let session = Rc::clone(&session);
        set_busy.set(true);
        spawn_local(async move {
            let result = ask(&settings, &session, &question).await;
            set_spent.set((session.borrow().spent_usd(), session.borrow().budget_usd()));
            set_answer.set(Some(result));
            set_busy.set(false);
        });
    };

    view! {
        <main style="max-width: 42rem; margin: 2rem auto; font-family: system-ui, sans-serif; padding: 0 1rem;">
            <h1>"llm-wasm + Leptos"</h1>
            <label>"Base URL " <input prop:value=base_url on:input=move |ev| set_base_url.set(event_target_value(&ev)) /></label>
            <br />
            <label>"API key " <input type="password" prop:value=api_key on:input=move |ev| set_api_key.set(event_target_value(&ev)) /></label>
            <br />
            <label>"Model " <input prop:value=model on:input=move |ev| set_model.set(event_target_value(&ev)) /></label>
            <br />
            <textarea rows="4" style="width: 100%;" prop:value=prompt on:input=move |ev| set_prompt.set(event_target_value(&ev)) />
            <br />
            <button on:click=send disabled=busy>{move || if busy.get() { "Asking..." } else { "Ask" }}</button>
            <p>{move || { let (s, b) = spent.get(); format!("Spent ${s:.6} of ${b:.2}") }}</p>
            {move || answer.get().map(|r| match r {
                Ok(reply) => view! {
                    <div>
                        <pre style="white-space: pre-wrap;">{reply.text.clone()}</pre>
                        <small>{describe(&reply)}</small>
                    </div>
                }.into_any(),
                Err(e) => view! { <p style="color: crimson;">{e}</p> }.into_any(),
            })}
        </main>
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

fn main() {
    leptos::mount::mount_to_body(App);
}
