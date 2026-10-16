// pixlang/editor/static/js/palette.js — Cmd/Ctrl+K command palette.

import { $, h, icon, fuzzy, highlightMatches, fmtKeys, catVar } from "./util.js";

/**
 * getItems() → [{id, group, title, sub, icon, keys, rail, run}]
 */
export function createPalette(getItems) {
  const overlay = $("#palette");
  const input = $("#palette-input");
  const list = $("#palette-list");
  let items = [];
  let shown = [];
  let sel = 0;
  let lastFocus = null;

  function open(prefill = "") {
    lastFocus = document.activeElement;
    items = getItems();
    overlay.hidden = false;
    input.value = prefill;
    filter();
    requestAnimationFrame(() => { input.focus(); input.select(); });
  }
  function close() {
    overlay.hidden = true;
    if (lastFocus && document.contains(lastFocus)) lastFocus.focus();
  }
  function filter() {
    const q = input.value.trim();
    if (!q) {
      shown = items.filter(i => i.pinned !== false).slice(0, 60).map(i => ({ ...i, idx: [] }));
    } else {
      shown = items.map(i => {
        const m = fuzzy(q, i.title);
        const ms = i.sub ? fuzzy(q, i.sub) : { score: -1 };
        const score = Math.max(m.score, ms.score >= 0 ? ms.score - 30 : -1);
        return { ...i, idx: m.score >= 0 ? m.idx : [], score: score + (i.weight || 0) };
      }).filter(i => i.score >= 0).sort((a, b) => b.score - a.score).slice(0, 60);
    }
    sel = 0;
    render();
  }
  function render() {
    list.replaceChildren();
    if (!shown.length) {
      list.append(h("div", { class: "pl-empty" }, "No matches"));
      input.removeAttribute("aria-activedescendant");
      return;
    }
    let group = null;
    shown.forEach((it, i) => {
      if (!input.value.trim() && it.group !== group) {
        group = it.group;
        list.append(h("div", { class: "pl-group", role: "presentation" }, group));
      }
      const row = h("button", {
        class: "pl-item", role: "option", id: "pl-" + i, tabindex: -1,
        "aria-selected": String(i === sel),
        onclick: () => choose(i),
        onmousemove: () => { if (sel !== i) { sel = i; mark(); } },
      },
      it.rail ? h("span", { class: "rail", style: { "--rail": catVar(it.rail) } }) : icon(it.icon || "command", "icon-sm"),
      h("span", { class: "pl-title" }, highlightMatches(it.title, it.idx)),
      it.sub ? h("span", { class: "pl-sub" }, it.sub) : h("span", { class: "pl-sub" }),
      it.keys ? h("span", { class: "pl-kbd" }, it.keys.split(" ").map(k => h("kbd", null, fmtKeys(k)))) : null);
      list.append(row);
    });
    mark();
  }
  function mark() {
    for (const el of list.querySelectorAll(".pl-item")) el.setAttribute("aria-selected", "false");
    const cur = list.querySelector("#pl-" + sel);
    if (cur) {
      cur.setAttribute("aria-selected", "true");
      cur.scrollIntoView({ block: "nearest" });
      input.setAttribute("aria-activedescendant", cur.id);
    }
  }
  function choose(i) {
    const it = shown[i];
    if (!it) return;
    close();
    setTimeout(() => it.run(), 0);
  }

  input.addEventListener("input", filter);
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { sel = Math.min(shown.length - 1, sel + 1); mark(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { sel = Math.max(0, sel - 1); mark(); e.preventDefault(); }
    else if (e.key === "Enter") { choose(sel); e.preventDefault(); }
    else if (e.key === "Escape") { close(); e.preventDefault(); }
    else if (e.key === "Tab") { e.preventDefault(); }
  });
  overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });

  return { open, close, isOpen: () => !overlay.hidden };
}
