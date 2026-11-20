// Run by tests/test_editor_client_state.py: the editor's storage helper and API wrapper.
import assert from "node:assert/strict";

const memory = new Map();
globalThis.localStorage = {
  getItem: (k) => (memory.has(k) ? memory.get(k) : null),
  setItem: (k, v) => memory.set(k, String(v)),
  removeItem: (k) => memory.delete(k),
};
globalThis.document = {
  querySelector: () => null,
  querySelectorAll: () => [],
  documentElement: { classList: { contains: () => false } },
};

const { store, api, ApiError, onConnection } = await import(process.argv[2]);

const GARBAGE = ["5", "-5", "{}", "[]", "null", '"text"', "true", "[1,{},null]",
  '{"a":1}', "not json at all", "1e999", '{"__proto__":{"x":1}}'];

for (const raw of GARBAGE) {
  memory.set("pixlang.k", raw);
  const list = store.getList("k");
  assert.ok(Array.isArray(list), `getList(${raw})`);
  assert.doesNotThrow(() => new Set(store.getList("k", (x) => typeof x === "string")));
  const obj = store.getObject("k");
  assert.ok(obj !== null && typeof obj === "object" && !Array.isArray(obj), `getObject(${raw})`);
  assert.equal(typeof store.getBool("k", false), "boolean");
  const n = store.getNumber("k", 1, { min: 0.5, max: 2 });
  assert.ok(Number.isFinite(n) && n >= 0.5 && n <= 2, `getNumber(${raw}) -> ${n}`);
  assert.ok(["a", "b"].includes(store.getString("k", "a", ["a", "b"])), `getString(${raw})`);
}

memory.set("pixlang.k", '["x", 3, "y"]');
assert.deepEqual(store.getList("k", (x) => typeof x === "string"), ["x", "y"]);
memory.set("pixlang.k", "1.5");
assert.equal(store.getNumber("k", 1, { min: 0.5, max: 2 }), 1.5);
memory.set("pixlang.k", "99");
assert.equal(store.getNumber("k", 1, { min: 0.5, max: 2 }), 2);
assert.equal(store.getList("missing").length, 0);

globalThis.localStorage = { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); } };
assert.deepEqual(store.getObject("k"), {});
assert.doesNotThrow(() => store.set("k", 1));

const seen = [];
onConnection((s) => seen.push(s));
const reply = (status, body) => ({ ok: status < 400, status, statusText: "x", json: async () => body });

globalThis.fetch = async () => { throw new TypeError("Failed to fetch"); };
await assert.rejects(api("/api/info"), (e) => e instanceof ApiError && e.kind === "offline"
  && e.message === "Server not reachable");

globalThis.fetch = (url, opts) => new Promise((_, reject) => {
  opts.signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
});
await assert.rejects(api("/api/run", { code: "" }, { timeout: 20 }),
  (e) => e instanceof ApiError && e.kind === "timeout");

globalThis.fetch = async () => reply(403, { error: "Missing or invalid editor token" });
await assert.rejects(api("/api/info"), (e) => e.kind === "auth" && /reload/i.test(e.message));

globalThis.fetch = async () => reply(404, { error: "nope" });
await assert.rejects(api("/api/x"), (e) => e.kind === "http" && e.status === 404 && e.message === "nope");

globalThis.fetch = async () => reply(200, { ok: 1 });
assert.deepEqual(await api("/api/info"), { ok: 1 });

const callerStop = new AbortController();
globalThis.fetch = (url, opts) => new Promise((_, reject) => {
  opts.signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
});
const pending = api("/api/slow", undefined, { signal: callerStop.signal });
callerStop.abort();
await assert.rejects(pending, (e) => e.name === "AbortError");

assert.equal(seen[0], "offline");
assert.ok(seen.includes("auth") && seen.includes("ok"));
console.log("client state ok");
