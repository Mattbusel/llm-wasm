// Calls every JavaScript export of llm-wasm (feature `js`) and checks the results.
//
//   cargo build --release --target wasm32-unknown-unknown --features js
//   wasm-bindgen --target nodejs --out-dir pkg target/wasm32-unknown-unknown/release/llm_wasm.wasm
//   node examples/node/smoke.mjs [path/to/pkg]
//
// Exits with code 1 on the first failed check.
import { createRequire } from "node:module";
import assert from "node:assert/strict";
import path from "node:path";

const require = createRequire(import.meta.url);
const pkg = path.resolve(process.argv[2] ?? "pkg");
const llm = require(path.join(pkg, "llm_wasm.js"));

let checks = 0;
function check(name, fn) {
  fn();
  checks += 1;
  console.log(`ok ${checks} - ${name}`);
}

check("estimateCost uses the real price", () => {
  // gpt-4o-mini: $0.15 in / $0.60 out per million tokens
  assert.ok(Math.abs(llm.estimateCost("gpt-4o-mini", 1000, 500) - 0.00045) < 1e-12);
  assert.throws(() => llm.estimateCost("no-such-model", 1, 1), /unknown model/);
});

check("knownModels lists the table", () => {
  const models = llm.knownModels();
  assert.ok(models.length > 100);
  assert.ok(models.includes("claude-sonnet-4-5"));
});

check("extractJson returns a JS object", () => {
  assert.deepEqual(llm.extractJson('Sure! Here: {"a": [1, 2], "b": null} done'), { a: [1, 2], b: null });
  assert.throws(() => llm.extractJson("no json here"));
});

check("stripCodeFence", () => {
  assert.equal(llm.stripCodeFence("```json\n{\"x\":1}\n```"), '{"x":1}');
});

check("renderTemplate with vars and partials", () => {
  assert.equal(llm.renderTemplate("Hi {{name}}", { name: "Ada" }), "Hi Ada");
  assert.equal(llm.renderTemplate("{{>sig}}", { name: "Ada" }, { sig: "-- {{name}}" }), "-- Ada");
  assert.throws(() => llm.renderTemplate("{{>loop}}", {}, { loop: "{{>loop}}" }), /nested/);
});

check("checkRequest blocks and allows", () => {
  const req = { model: "gpt-4o", messages: [{ role: "user", content: "my password is hunter2" }], max_tokens: null, temperature: null };
  assert.throws(() => llm.checkRequest(req, ["password"], 1000), /blocked by guard 'content'/);
  assert.throws(() => llm.checkRequest(req, [], 5), /guard 'length'/);
  llm.checkRequest(req, ["secret"], 1000);
});

check("routeModel follows rules", () => {
  const req = { model: "gpt-4o", messages: [{ role: "user", content: "a" }, { role: "user", content: "b" }], max_tokens: null, temperature: null };
  const rules = [{ condition: { MessageCountExceeds: 1 }, target_model: "claude-sonnet-4-5" }];
  assert.equal(llm.routeModel(req, rules, "gpt-4o-mini"), "claude-sonnet-4-5");
  assert.equal(llm.routeModel(req, [], "gpt-4o-mini"), "gpt-4o-mini");
});

check("Ledger enforces the budget", () => {
  const ledger = new llm.Ledger(0.001);
  ledger.record("gpt-4o-mini", 1000, 0);
  assert.ok(Math.abs(ledger.total - 0.00015) < 1e-12);
  assert.throws(() => ledger.record("gpt-4o", 1_000_000, 0), /budget exceeded/);
  assert.ok(Math.abs(ledger.remaining - 0.00085) < 1e-12);
  assert.equal(new llm.Ledger().remaining, undefined);
});

check("Retry honours Retry-After and jitter", () => {
  const retry = new llm.Retry(3, 1000, 10000);
  assert.equal(retry.shouldRetry(1, 429), true);
  assert.equal(retry.shouldRetry(1, 400), false);
  assert.equal(retry.shouldRetry(3, 503), false);
  assert.equal(retry.delay(1, 0.0, "7"), 7000);
  assert.equal(retry.delay(1, 0.0, null), 500);
  assert.equal(retry.delay(1, 1.0, undefined), 1000);
  assert.throws(() => new llm.Retry(0, 1, 1));
});

check("Cache expires and caps", () => {
  const cache = new llm.Cache(1000, 2);
  cache.set("a", "1", 0);
  cache.set("b", "2", 10);
  cache.set("c", "3", 20);
  assert.equal(cache.size, 2);
  assert.equal(cache.get("a", 30), undefined);
  assert.equal(cache.get("c", 30), "3");
  assert.equal(cache.get("c", 5000), undefined);
});

// Optional features: only checked when the module was built with them.
if (llm.extractJsonLenient) {
  check("extractJsonLenient repairs almost-JSON", () => {
    assert.deepEqual(llm.extractJsonLenient("Sure: {'a': [1, 2,], 'ok': True"), { a: [1, 2], ok: true });
    assert.throws(() => llm.extractJsonLenient("no json"));
  });
}
if (llm.renderChatTemplate) {
  check("renderChatTemplate (ChatML)", () => {
    const chatml = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";
    assert.equal(llm.renderChatTemplate(chatml, [{ role: "user", content: "Hi" }], true), "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
  });
}
if (llm.redactSecrets) {
  check("redactSecrets", () => {
    assert.equal(llm.redactSecrets("key sk-proj-AbCdEf0123456789XyZ0123456789 ok"), "key [REDACTED:api_key] ok");
  });
}
if (llm.StreamDecoder) {
  check("StreamDecoder handles any split, including inside an emoji", () => {
    const body = new TextEncoder().encode('data: {"choices":[{"delta":{"content":"Bon"}}]}\n\ndata: {"choices":[{"delta":{"content":"jour \u{1F44B}"}}]}\n\ndata: [DONE]\n\n');
    const dec = new llm.StreamDecoder();
    let out = "";
    for (let i = 0; i < body.length; i += 3) for (const d of dec.push(body.subarray(i, i + 3))) out += d;
    assert.equal(out, "Bonjour \u{1F44B}");
    assert.equal(dec.finished, true);
    assert.equal(dec.text, out);
  });
}

console.log(`all ${checks} checks passed`);
