//! Pull structured data out of model answers that are not quite JSON.
//!
//! Run with `cargo run --example repair_json --features repair`.

use llm_wasm::format::JsonFormatter;

fn main() {
    let answers = [
        "Here you go:\n```json\n{\"city\": \"Paris\", \"tags\": [\"eu\", \"fr\",],}\n```",
        "{'city': 'Paris', 'capital': True, 'mayor': None}",
        "Result: {\"items\": [{\"id\": 1}, {\"id\": 2}, {\"id\"",
        "{\"note\": \"line one\nline two\"}",
        "I could not find any data, sorry.",
    ];
    for a in answers {
        let strict = JsonFormatter::extract_json(a).map(|v| v.to_string());
        let lenient = JsonFormatter::extract_json_lenient(a).map(|v| v.to_string());
        println!("strict:  {strict:?}\nlenient: {lenient:?}\n");
    }
}
