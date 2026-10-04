//! llm-wasm 0.2 against 0.1.1 and against the obvious alternatives.
//!
//! - JSON extraction from a chatty answer, and from a text with 20,000 stray
//!   `{` (0.1.1 tried every bracket and scanned to the end each time)
//! - lenient extraction of cut-off output (feature `repair`)
//! - `{{var}}` templates: built-in engine vs minijinja (feature `jinja`)
//! - cache key + set + get: SHA-256 + LRU (0.2) vs FNV-1a + HashMap (0.1.1)
//! - SSE decoding of a 500-chunk OpenAI stream fed in 64-byte pieces (feature `stream`)
//!
//! Run: `cargo bench --bench vs_alternatives --features repair,jinja,stream`
#![allow(clippy::unwrap_used)]

use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn json(c: &mut Criterion) {
    let mut g = c.benchmark_group("extract_json");
    let answer = format!("Sure! Here is the data you asked for:\n```json\n{}\n```\nLet me know.", serde_json::json!({
        "cities": (0..50).map(|i| serde_json::json!({"name": format!("city {i}"), "pop": i * 1000})).collect::<Vec<_>>()
    }));
    g.bench_function("chatty answer / 0.2", |b| b.iter(|| black_box(llm_wasm::format::JsonFormatter::extract_json(&answer).unwrap())));
    g.bench_function("chatty answer / 0.1.1", |b| b.iter(|| black_box(llm_wasm_011::format::JsonFormatter::extract_json(&answer).unwrap())));
    let brackets = format!("{} {{\"a\": 1}}", "{".repeat(20_000));
    g.sample_size(10);
    g.bench_function("20k stray brackets / 0.2", |b| b.iter(|| black_box(llm_wasm::format::JsonFormatter::extract_json(&brackets).is_ok())));
    g.bench_function("20k stray brackets / 0.1.1", |b| b.iter(|| black_box(llm_wasm_011::format::JsonFormatter::extract_json(&brackets).is_ok())));
    #[cfg(feature = "repair")]
    {
        let cut = &answer[..answer.len() / 2];
        g.bench_function("cut-off answer / 0.2 lenient", |b| {
            b.iter(|| black_box(llm_wasm::format::JsonFormatter::extract_json_lenient(cut).unwrap()))
        });
    }
    g.finish();
}

fn templates(c: &mut Criterion) {
    let mut g = c.benchmark_group("template {{var}}");
    let tpl = "You are {{role}}. Answer in {{lang}}. Context: {{ctx}}. Question: {{q}}";
    let mut vars = HashMap::new();
    for (k, v) in [("role", "a helpful assistant"), ("lang", "French"), ("ctx", "the Paris metro"), ("q", "how late does it run?")] {
        vars.insert(k.to_string(), v.to_string());
    }
    let engine = llm_wasm::template::TemplateEngine::new();
    g.bench_function("built-in engine", |b| b.iter(|| black_box(engine.render(tpl, &vars).unwrap())));
    #[cfg(feature = "jinja")]
    {
        let ctx = serde_json::to_value(&vars).unwrap();
        g.bench_function("minijinja (render_jinja, parses each time)", |b| {
            b.iter(|| black_box(llm_wasm::template::render_jinja(tpl, &ctx).unwrap()))
        });
    }
    g.finish();
}

fn cache(c: &mut Criterion) {
    let mut g = c.benchmark_group("cache key + set + get, 1 KB prompt");
    let messages = "x".repeat(1_000);
    g.bench_function("0.2 (SHA-256 key, LRU)", |b| {
        let mut cache = llm_wasm::cache::TtlCache::with_capacity(60_000.0, 1_000);
        let mut i = 0u64;
        b.iter(|| {
            i += 1;
            let k = llm_wasm::cache::cache_key("gpt-4o", &messages);
            cache.set(k, "answer".into(), i as f64);
            black_box(cache.get(&k, i as f64))
        })
    });
    g.bench_function("0.1.1 (FNV-1a key, HashMap)", |b| {
        let mut cache = llm_wasm_011::cache::TtlCache::new(60_000.0);
        let mut i = 0u64;
        b.iter(|| {
            i += 1;
            let k = llm_wasm_011::cache::cache_key("gpt-4o", &messages);
            cache.set(k, "answer".into(), i as f64);
            black_box(cache.get(k, i as f64))
        })
    });
    g.finish();
}

#[cfg(feature = "stream")]
fn stream(c: &mut Criterion) {
    let mut body = String::new();
    for i in 0..500 {
        let chunk = serde_json::json!({"id": "c", "choices": [{"index": 0, "delta": {"content": format!("tok{i} ")}, "finish_reason": null}]});
        body.push_str(&format!("data: {chunk}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    let bytes = body.into_bytes();
    let mut g = c.benchmark_group("SSE decode, 500 chunks");
    g.throughput(criterion::Throughput::Bytes(bytes.len() as u64));
    g.bench_function("StreamDecoder, 64-byte pieces", |b| {
        b.iter(|| {
            let mut d = llm_wasm::stream::StreamDecoder::new();
            for piece in bytes.chunks(64) {
                black_box(d.push(piece).unwrap());
            }
            black_box(d.text().len())
        })
    });
    g.finish();
}

#[cfg(not(feature = "stream"))]
fn stream(_: &mut Criterion) {}

criterion_group!(benches, json, templates, cache, stream);
criterion_main!(benches);
