// pixlang/editor/static/js/util.js — small shared helpers.

const TOKEN = document.querySelector('meta[name="pixlang-token"]')?.content || "";

/** JSON API call carrying the per-process editor token. */
export async function api(path, body) {
  const opts = { headers: { "X-PixLang-Token": TOKEN }, cache: "no-store" };
  if (body !== undefined) {
    opts.method = "POST";
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch { data = null; }
  if (!res.ok) {
    const err = new Error((data && data.error) || `${res.status} ${res.statusText}`);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

/** Fire-and-forget POST that survives page unload (fetch keepalive). */
export function apiKeepalive(path, body = {}) {
  try {
    fetch(path, {
      method: "POST", keepalive: true,
      headers: { "X-PixLang-Token": TOKEN, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).catch(() => {});
  } catch { /* unloading */ }
}

/** localStorage wrapper that never throws. */
export const store = {
  get(key, fallback = null) {
    try {
      const v = localStorage.getItem("pixlang." + key);
      return v === null ? fallback : JSON.parse(v);
    } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem("pixlang." + key, JSON.stringify(value)); } catch { /* blocked */ }
  },
  del(key) {
    try { localStorage.removeItem("pixlang." + key); } catch { /* blocked */ }
  },
};

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/** Create an element: h("div", {class: "x", onclick}, child, "text") */
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "class") el.className = v;
      else if (k === "style" && typeof v === "object") {
        for (const [sk, sv] of Object.entries(v)) {
          if (sk.startsWith("--")) el.style.setProperty(sk, sv); else el.style[sk] = sv;
        }
      }
      else if (k === "dataset") Object.assign(el.dataset, v);
      else el.setAttribute(k, v === true ? "" : String(v));
    }
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

const SVGNS = "http://www.w3.org/2000/svg";
export function svg(tag, attrs, ...children) {
  const el = document.createElementNS(SVGNS, tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

/** <svg class="icon"><use href="#i-name"/></svg> */
export function icon(name, cls = "") {
  const s = svg("svg", { class: ("icon " + cls).trim(), "aria-hidden": "true" });
  s.append(svg("use", { href: "#i-" + name }));
  return s;
}

export function debounce(fn, ms) {
  let t = null;
  const d = (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  d.cancel = () => clearTimeout(t);
  return d;
}

export const isMac = document.documentElement.classList.contains("is-mac");
export const modKey = isMac ? "⌘" : "Ctrl";

/** Rewrite <kbd data-kbd="mod+enter"> for the current platform. */
export function localizeKbd(root = document) {
  for (const k of $$("kbd[data-kbd]", root)) {
    k.textContent = fmtKeys(k.dataset.kbd);
  }
}
export function fmtKeys(spec) {
  return spec.split("+").map(p => ({
    mod: isMac ? "⌘" : "Ctrl", shift: isMac ? "⇧" : "Shift", alt: isMac ? "⌥" : "Alt",
    enter: "↵", k: "K", s: "S", p: "P", "/": "/", f: "F", b: "B", j: "J",
  }[p] ?? p.toUpperCase())).join(isMac ? "" : "+");
}

/** Six-hue category rail colours. */
const CAT_INDEX = {
  IO: 1, "I/O": 1, Depth: 3, Bitwise: 5, LUT: 3, Channels: 2, Raw: 1, Plugin: 6, Geometry: 2, Color: 3, Threshold: 4, Filter: 4, Morphology: 5,
  Analysis: 6, Composition: 2, Annotation: 3, Variable: 4, Control: 6,
  Include: 1, Assert: 5, ROI: 2, Function: 4, Statement: 6,
};
export function catVar(cat) {
  let i = CAT_INDEX[cat];
  if (!i) {
    let hsh = 0;
    for (const ch of String(cat || "")) hsh = (hsh * 31 + ch.charCodeAt(0)) >>> 0;
    i = (hsh % 6) + 1;
  }
  return `var(--cat-${i})`;
}

export function fmtMs(ms) {
  if (ms == null || isNaN(ms)) return "—";
  if (ms < 1) return ms.toFixed(2) + " ms";
  if (ms < 100) return ms.toFixed(1) + " ms";
  if (ms < 10000) return Math.round(ms) + " ms";
  return (ms / 1000).toFixed(1) + " s";
}

/** Simple subsequence fuzzy match → score (higher better) or -1, plus match indices. */
export function fuzzy(query, text) {
  if (!query) return { score: 0, idx: [] };
  const q = query.toLowerCase(), t = text.toLowerCase();
  const direct = t.indexOf(q);
  if (direct >= 0) {
    const idx = Array.from({ length: q.length }, (_, i) => direct + i);
    return { score: 100 - direct + (direct === 0 ? 50 : 0) - t.length * 0.1, idx };
  }
  let ti = 0, score = 0, last = -1;
  const idx = [];
  for (const ch of q) {
    ti = t.indexOf(ch, ti);
    if (ti < 0) return { score: -1, idx: [] };
    score += ti === last + 1 ? 5 : 1;
    if (ti === 0 || /[\s_/.-]/.test(t[ti - 1])) score += 4;
    idx.push(ti); last = ti; ti++;
  }
  return { score: score - t.length * 0.1, idx };
}

export function highlightMatches(text, idx) {
  const frag = document.createDocumentFragment();
  if (!idx || !idx.length) { frag.append(text); return frag; }
  const set = new Set(idx);
  let buf = "", inMark = false;
  const flush = () => {
    if (!buf) return;
    frag.append(inMark ? h("mark", null, buf) : document.createTextNode(buf));
    buf = "";
  };
  for (let i = 0; i < text.length; i++) {
    const m = set.has(i);
    if (m !== inMark) { flush(); inMark = m; }
    buf += text[i];
  }
  flush();
  return frag;
}

export function toast(message, kind = "ok", ms = 2200) {
  const host = $("#toasts");
  if (!host) return;
  const el = h("div", { class: "toast " + kind }, icon(kind === "err" ? "x-circle" : "check-circle", "icon-sm"), message);
  host.append(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity 200ms"; }, ms);
  setTimeout(() => el.remove(), ms + 250);
}
