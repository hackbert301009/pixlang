// pixlang/editor/static/js/layout.js
// Docked, resizable panes (CSS grid + small splitter logic), persisted sizes,
// sidebar views, bottom panel tabs, right-pane tabs, mobile view switching.

import { $, $$, store } from "./util.js";

const app = () => $("#app");
const root = document.documentElement;

const SIZES = {
  sidebar: { varName: "--sidebar-w", axis: "x", min: 180, max: 480, def: 248, unit: "px" },
  right:   { varName: "--right-w",   axis: "x", min: 300, max: 0.68, def: 0.44, unit: "frac", invert: true },
  bottom:  { varName: "--bottom-h",  axis: "y", min: 90,  max: 0.7,  def: 200, unit: "px", invert: true },
  flow:    { varName: "--flow-h",    axis: "y", min: 120, max: 0.8,  def: 0.38, unit: "frac", invert: true },
};

function containerFor(key) {
  if (key === "bottom") return $(".center");
  if (key === "flow") return $(".right");
  return $("#workbench");
}

function applySize(key, value) {
  const s = SIZES[key];
  if (s.unit === "frac") root.style.setProperty(s.varName, (value * 100).toFixed(2) + "%");
  else root.style.setProperty(s.varName, Math.round(value) + "px");
}

export function initLayout({ onResize } = {}) {
  const saved = store.get("layout", {}) || {};
  for (const key of Object.keys(SIZES)) {
    if (typeof saved[key] === "number") applySize(key, saved[key]);
  }

  for (const sp of $$(".splitter[data-split]")) {
    const key = sp.dataset.split;
    const s = SIZES[key];
    if (!s) continue;
    const commit = (v) => {
      applySize(key, v);
      const cur = store.get("layout", {}) || {};
      cur[key] = v;
      store.set("layout", cur);
      onResize && onResize();
    };
    const compute = (clientPos) => {
      const c = containerFor(key).getBoundingClientRect();
      const total = s.axis === "x" ? c.width : c.height;
      const start = s.axis === "x" ? c.left : c.top;
      let px;
      if (key === "sidebar") px = clientPos - start - 48;
      else px = s.invert ? (start + total - clientPos) : (clientPos - start);
      const maxPx = s.max <= 1 ? s.max * total : s.max;
      px = Math.max(s.min, Math.min(maxPx, px));
      return s.unit === "frac" ? px / total : px;
    };
    sp.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      sp.setPointerCapture(e.pointerId);
      sp.classList.add("dragging");
      document.body.classList.add("resizing", s.axis === "x" ? "col" : "row");
      const move = (ev) => {
        const v = compute(s.axis === "x" ? ev.clientX : ev.clientY);
        applySize(key, v);
        onResize && onResize();
      };
      const up = (ev) => {
        sp.releasePointerCapture(e.pointerId);
        sp.classList.remove("dragging");
        document.body.classList.remove("resizing", "col", "row");
        sp.removeEventListener("pointermove", move);
        sp.removeEventListener("pointerup", up);
        commit(compute(s.axis === "x" ? ev.clientX : ev.clientY));
      };
      sp.addEventListener("pointermove", move);
      sp.addEventListener("pointerup", up);
    });
    sp.addEventListener("dblclick", () => commit(s.def));
    sp.addEventListener("keydown", (e) => {
      const keys = s.axis === "x" ? ["ArrowLeft", "ArrowRight"] : ["ArrowUp", "ArrowDown"];
      if (!keys.includes(e.key)) return;
      e.preventDefault();
      const c = containerFor(key).getBoundingClientRect();
      const total = s.axis === "x" ? c.width : c.height;
      const cur = (store.get("layout", {}) || {})[key] ?? s.def;
      let px = s.unit === "frac" ? cur * total : cur;
      let delta = (e.key === keys[1] ? 1 : -1) * (e.shiftKey ? 64 : 16);
      if (s.invert) delta = -delta;
      px += delta;
      const maxPx = s.max <= 1 ? s.max * total : s.max;
      px = Math.max(s.min, Math.min(maxPx, px));
      commit(s.unit === "frac" ? px / total : px);
    });
  }

  // Bottom panel collapse state, flow pane visibility
  const ui = store.get("ui", {}) || {};
  if (ui.bottom === "collapsed") app().dataset.bottom = "collapsed";
  if (ui.flow === "hidden") app().dataset.flow = "hidden";
  if (ui.sidebar === "closed" || window.innerWidth < 1024) app().dataset.sidebar = "closed";
  $("#act-flow").setAttribute("aria-selected", String(app().dataset.flow !== "hidden"));
}

function saveUi(patch) {
  const ui = store.get("ui", {}) || {};
  Object.assign(ui, patch);
  store.set("ui", ui);
}

// ── Sidebar views ───────────────────────────────────────────────────────────
export function showSide(name, { toggle = true } = {}) {
  const a = app();
  const current = a.dataset.side || "files";
  const open = a.dataset.sidebar !== "closed";
  if (toggle && open && current === name && window.innerWidth >= 640) {
    a.dataset.sidebar = "closed";
    saveUi({ sidebar: "closed" });
    syncActivity();
    return;
  }
  a.dataset.side = name;
  a.dataset.sidebar = "open";
  if (window.innerWidth >= 1024) saveUi({ sidebar: "open", side: name });
  for (const v of $$(".side-view")) v.hidden = v.id !== "side-" + name;
  syncActivity();
}
export function closeSidebarIfOverlay() {
  if (window.innerWidth < 1024 && window.innerWidth >= 640) {
    app().dataset.sidebar = "closed";
    syncActivity();
  }
}
function syncActivity() {
  const a = app();
  const open = a.dataset.sidebar !== "closed";
  for (const b of $$(".act[data-side]")) {
    if (b.dataset.side === "flow") continue;
    b.setAttribute("aria-selected", String(open && b.dataset.side === (a.dataset.side || "files")));
  }
}

export function toggleFlowPane(force) {
  const a = app();
  const hidden = force === undefined ? a.dataset.flow !== "hidden" : !force;
  if (window.innerWidth < 1024) {
    showRightTab("flow");
    if (window.innerWidth < 640) setView("flow");
    return;
  }
  if (hidden) a.dataset.flow = "hidden"; else delete a.dataset.flow;
  $("#act-flow").setAttribute("aria-selected", String(!hidden));
  saveUi({ flow: hidden ? "hidden" : "shown" });
}

// ── Bottom tabs ─────────────────────────────────────────────────────────────
export function showBottom(name) {
  for (const t of $$(".bottom-panel .tab")) {
    const on = t.dataset.bottom === name;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
  }
  for (const v of $$(".bp-view")) v.hidden = v.id !== "bp-" + name;
  if (app().dataset.bottom === "collapsed") toggleBottom(false);
  store.set("bottomTab", name);
}
export function toggleBottom(collapse) {
  const a = app();
  const c = collapse === undefined ? a.dataset.bottom !== "collapsed" : collapse;
  if (c) a.dataset.bottom = "collapsed"; else delete a.dataset.bottom;
  $("#btn-toggle-bottom").setAttribute("aria-label", c ? "Expand panel" : "Collapse panel");
  saveUi({ bottom: c ? "collapsed" : "open" });
}

export function showRightTab(name) {
  app().dataset.rightTab = name;
  for (const t of $$(".right-tabs .tab")) {
    const on = t.dataset.right === name;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
  }
}

// ── Mobile views ────────────────────────────────────────────────────────────
export function setView(name) {
  const a = app();
  a.dataset.view = name;
  for (const b of $$(".mobile-tabs [data-view]")) b.setAttribute("aria-selected", String(b.dataset.view === name));
  if (name === "files") {
    a.dataset.sidebar = "open";
    for (const v of $$(".side-view")) v.hidden = v.id !== "side-" + (a.dataset.side || "files");
  }
  if (name === "problems") showBottom("problems");
  window.dispatchEvent(new Event("pixlang:view"));
}

/** Roving arrow-key focus for a tablist. */
export function rovingTabs(listEl, selector) {
  listEl.addEventListener("keydown", (e) => {
    const horiz = listEl.getAttribute("aria-orientation") !== "vertical";
    const next = horiz ? "ArrowRight" : "ArrowDown";
    const prev = horiz ? "ArrowLeft" : "ArrowUp";
    if (![next, prev, "Home", "End"].includes(e.key)) return;
    const items = $$(selector, listEl).filter(el => el.offsetParent !== null);
    const i = items.indexOf(document.activeElement);
    if (i < 0) return;
    let n = i;
    if (e.key === next) n = (i + 1) % items.length;
    if (e.key === prev) n = (i - 1 + items.length) % items.length;
    if (e.key === "Home") n = 0;
    if (e.key === "End") n = items.length - 1;
    e.preventDefault();
    items[n].focus();
    if (listEl.getAttribute("role") === "tablist") items[n].click();
  });
}

/** Arrow-key navigation for a vertical list of buttons. */
export function listKeys(listEl, selector = ".item") {
  listEl.addEventListener("keydown", (e) => {
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(e.key)) return;
    const items = $$(selector, listEl);
    const i = items.indexOf(document.activeElement);
    let n = i;
    if (e.key === "ArrowDown") n = Math.min(items.length - 1, i + 1);
    if (e.key === "ArrowUp") n = Math.max(0, i - 1);
    if (e.key === "Home") n = 0;
    if (e.key === "End") n = items.length - 1;
    e.preventDefault();
    items[n]?.focus();
  });
}
