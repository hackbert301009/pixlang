// pixlang/editor/static/js/preview.js
// Image preview: wheel zoom around the cursor, drag to pan, fit / 1:1,
// before/after compare slider, and a strip of run checkpoints.

import { $, $$, h, fmtMs, store } from "./util.js";

export function createPreview() {
  const vp = $("#pv-viewport");
  const stage = $("#pv-stage");
  const img = $("#pv-img");
  const imgB = $("#pv-img-before");
  const handle = $("#pv-handle");
  const empty = $("#pv-empty");
  const stale = $("#pv-stale");
  const staleText = $("#pv-stale-text");
  const busy = $("#pv-busy");
  const meta = $("#pv-meta");
  const strip = $("#pv-strip");
  const zoomBtn = $("#pv-zoom");
  const dl = $("#pv-download");
  const modes = $$("#preview-modes [data-mode]");

  const st = {
    mode: store.get("previewMode", "output"),
    result: null,
    stepKey: "final",
    scale: 1, tx: 0, ty: 0, fit: true,
    nat: { w: 0, h: 0 },
    split: 0.5,
  };

  // ── transform ────────────────────────────────────────────────────────────
  function apply() {
    stage.style.transform = `translate(${st.tx}px, ${st.ty}px) scale(${st.scale})`;
    stage.style.width = st.nat.w + "px";
    stage.style.height = st.nat.h + "px";
    zoomBtn.textContent = st.fit ? "Fit" : Math.round(st.scale * 100) + "%";
    vp.classList.toggle("pixelated", st.scale >= 2);
    if (st.mode === "compare") positionHandle();
  }
  function fit() {
    const r = vp.getBoundingClientRect();
    if (!st.nat.w || !r.width) return;
    const pad = 16;
    const s = Math.min((r.width - pad * 2) / st.nat.w, (r.height - pad * 2) / st.nat.h, 8);
    st.scale = Math.max(0.02, s);
    st.tx = (r.width - st.nat.w * st.scale) / 2;
    st.ty = (r.height - st.nat.h * st.scale) / 2;
    st.fit = true;
    apply();
  }
  function zoomAt(factor, cx, cy) {
    const r = vp.getBoundingClientRect();
    if (cx == null) { cx = r.width / 2; cy = r.height / 2; }
    const ns = Math.max(0.05, Math.min(32, st.scale * factor));
    const k = ns / st.scale;
    st.tx = cx - (cx - st.tx) * k;
    st.ty = cy - (cy - st.ty) * k;
    st.scale = ns;
    st.fit = false;
    apply();
  }
  function oneToOne() {
    const r = vp.getBoundingClientRect();
    st.scale = 1;
    st.tx = (r.width - st.nat.w) / 2;
    st.ty = (r.height - st.nat.h) / 2;
    st.fit = false;
    apply();
  }

  vp.addEventListener("wheel", (e) => {
    if (!st.nat.w) return;
    e.preventDefault();
    const r = vp.getBoundingClientRect();
    const f = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0018));
    zoomAt(f, e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });

  let drag = null;
  vp.addEventListener("pointerdown", (e) => {
    if (!st.nat.w || e.button !== 0 || e.target.closest("#pv-handle")) return;
    drag = { x: e.clientX, y: e.clientY, tx: st.tx, ty: st.ty };
    vp.setPointerCapture(e.pointerId);
    vp.classList.add("panning");
  });
  vp.addEventListener("pointermove", (e) => {
    if (!drag) return;
    st.tx = drag.tx + (e.clientX - drag.x);
    st.ty = drag.ty + (e.clientY - drag.y);
    st.fit = false;
    apply();
  });
  const endDrag = () => { drag = null; vp.classList.remove("panning"); };
  vp.addEventListener("pointerup", endDrag);
  vp.addEventListener("pointercancel", endDrag);
  vp.addEventListener("dblclick", () => (st.fit ? oneToOne() : fit()));
  vp.addEventListener("keydown", (e) => {
    if (!st.nat.w) return;
    const step = 40;
    if (e.key === "+" || e.key === "=") zoomAt(1.25);
    else if (e.key === "-") zoomAt(0.8);
    else if (e.key === "0") fit();
    else if (e.key === "1") oneToOne();
    else if (e.key === "ArrowLeft") { st.tx += step; st.fit = false; apply(); }
    else if (e.key === "ArrowRight") { st.tx -= step; st.fit = false; apply(); }
    else if (e.key === "ArrowUp") { st.ty += step; st.fit = false; apply(); }
    else if (e.key === "ArrowDown") { st.ty -= step; st.fit = false; apply(); }
    else return;
    e.preventDefault();
  });
  new ResizeObserver(() => { if (st.fit) fit(); else apply(); }).observe(vp);

  $("#pv-zoom-in").addEventListener("click", () => zoomAt(1.25));
  $("#pv-zoom-out").addEventListener("click", () => zoomAt(0.8));
  $("#pv-fit").addEventListener("click", fit);
  zoomBtn.addEventListener("click", () => (st.fit ? oneToOne() : fit()));

  // ── compare slider ───────────────────────────────────────────────────────
  function positionHandle() {
    const left = st.tx + st.nat.w * st.scale * st.split;
    handle.style.left = left + "px";
    const pct = st.split * 100;
    img.style.clipPath = `inset(0 0 0 ${pct}%)`;
    handle.setAttribute("aria-valuenow", String(Math.round(pct)));
  }
  let hdrag = false;
  handle.addEventListener("pointerdown", (e) => { hdrag = true; handle.setPointerCapture(e.pointerId); e.stopPropagation(); });
  handle.addEventListener("pointermove", (e) => {
    if (!hdrag) return;
    const r = vp.getBoundingClientRect();
    const x = e.clientX - r.left;
    st.split = Math.max(0, Math.min(1, (x - st.tx) / (st.nat.w * st.scale)));
    positionHandle();
  });
  handle.addEventListener("pointerup", () => { hdrag = false; });
  handle.addEventListener("keydown", (e) => {
    const d = e.shiftKey ? 0.1 : 0.02;
    if (e.key === "ArrowLeft") st.split = Math.max(0, st.split - d);
    else if (e.key === "ArrowRight") st.split = Math.min(1, st.split + d);
    else if (e.key === "Home") st.split = 0;
    else if (e.key === "End") st.split = 1;
    else return;
    e.preventDefault();
    e.stopPropagation();
    positionHandle();
  });

  // ── modes ────────────────────────────────────────────────────────────────
  for (const b of modes) b.addEventListener("click", () => setMode(b.dataset.mode));
  $("#preview-modes").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const enabled = modes.filter(m => !m.disabled);
    const i = enabled.indexOf(document.activeElement);
    const n = enabled[(i + (e.key === "ArrowRight" ? 1 : -1) + enabled.length) % enabled.length];
    n.focus(); n.click();
  });

  function setMode(mode) {
    st.mode = mode;
    store.set("previewMode", mode);
    for (const b of modes) {
      const on = b.dataset.mode === mode;
      b.setAttribute("aria-selected", String(on));
      b.tabIndex = on ? 0 : -1;
    }
    render();
  }

  function current() {
    const r = st.result;
    if (!r) return null;
    if (st.mode === "steps") {
      const all = stepsOf(r);
      return all.find(s => s.key === st.stepKey) || all[all.length - 1] || null;
    }
    return r.image ? { key: "final", label: "Output", ...r.image } : null;
  }

  function stepsOf(r) {
    const out = [];
    if (r.before) out.push({ key: "before", label: "Input", kind: "LOAD", ...r.before });
    for (const c of r.checkpoints || []) out.push({ key: c.key, label: c.name, kind: "CP", ...c });
    if (r.image) out.push({ key: "final", label: r.ok ? "Output" : "Partial", kind: r.ok ? "OUT" : "ERR", ...r.image });
    return out;
  }

  function render() {
    const r = st.result;
    const cur = current();
    const compare = st.mode === "compare" && r && r.before && r.image;
    const hasImg = !!cur;
    empty.hidden = hasImg;
    vp.classList.toggle("empty", !hasImg);
    img.hidden = !hasImg;
    imgB.hidden = !compare;
    handle.hidden = !compare;
    $("#pv-tag-before").hidden = !compare;
    $("#pv-tag-after").hidden = !compare;
    if (!compare) img.style.clipPath = "";
    modes.find(m => m.dataset.mode === "compare").disabled = !(r && r.before && r.image);
    modes.find(m => m.dataset.mode === "steps").disabled = !(r && stepsOf(r).length > 1);

    if (hasImg) {
      const url = cur.url;
      const newNat = cur.width !== st.nat.w || cur.height !== st.nat.h;
      st.nat = { w: cur.width, h: cur.height };
      if (img.getAttribute("src") !== url) img.src = url;
      img.width = cur.width; img.height = cur.height;
      if (compare) {
        imgB.src = r.before.url;
        imgB.width = cur.width; imgB.height = cur.height;
        imgB.style.width = cur.width + "px"; imgB.style.height = cur.height + "px";
      }
      img.style.width = cur.width + "px"; img.style.height = cur.height + "px";
      if (newNat || st.fit) fit(); else apply();
      dl.href = url;
      dl.download = (cur.label || "output").replace(/[^\w.-]+/g, "_") + ".png";
      dl.setAttribute("aria-disabled", "false");
      const parts = [
        h("span", { class: "lbl" }, cur.label || "Output"), h("span", { class: "sep" }, "·"),
        `${cur.width}×${cur.height}`, h("span", { class: "sep" }, "·"),
        `${cur.channels}ch`, h("span", { class: "sep" }, "·"), cur.dtype,
      ];
      if (r && r.stats && cur.key === "final") parts.push(h("span", { class: "sep" }, "·"), fmtMs(r.stats.total_ms));
      meta.replaceChildren(...parts);
    } else {
      st.nat = { w: 0, h: 0 };
      img.removeAttribute("src");
      dl.removeAttribute("href");
      dl.setAttribute("aria-disabled", "true");
      meta.textContent = r && !r.ok ? "No image produced" : "—";
    }
    renderStrip();
  }

  function renderStrip() {
    const r = st.result;
    const steps = r ? stepsOf(r) : [];
    if (steps.length < 2) { strip.replaceChildren(); return; }
    const sel = st.mode === "steps" ? (current()?.key) : (st.mode === "compare" ? null : "final");
    strip.replaceChildren(...steps.map(s => {
      const b = h("button", {
        class: "thumb", role: "option", "aria-selected": String(s.key === sel),
        title: `${s.label} · ${s.width}×${s.height} · ${s.channels}ch`,
        onclick: () => { st.stepKey = s.key; setMode("steps"); },
      },
      h("img", { src: s.thumb_url, alt: "", loading: "lazy", draggable: "false" }),
      h("span", { class: "tkind" }, s.kind),
      h("span", { class: "tcap" }, s.label));
      return b;
    }));
    strip.onkeydown = (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      const items = $$(".thumb", strip);
      const i = items.indexOf(document.activeElement);
      const n = items[Math.max(0, Math.min(items.length - 1, i + (e.key === "ArrowRight" ? 1 : -1)))];
      n?.focus(); n?.click(); e.preventDefault();
    };
  }

  setMode(st.mode);

  return {
    setResult(result) {
      st.result = result && (result.image || result.before) ? result : (result || null);
      if (st.mode === "steps" && result && !stepsOf(result).some(s => s.key === st.stepKey)) st.stepKey = "final";
      if (st.mode === "compare" && !(result && result.before && result.image)) st.mode = "output";
      setMode(st.mode);
    },
    setBusy(on) { busy.hidden = !on; },
    setStale(kind, text) {
      if (!kind) { stale.hidden = true; vp.classList.remove("stale"); return; }
      stale.hidden = false;
      stale.className = "pv-stale " + kind;
      staleText.textContent = text;
      vp.classList.toggle("stale", kind === "error");
    },
    hasImage: () => !!(st.result && st.result.image),
    fit,
  };
}
