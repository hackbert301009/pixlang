// pixlang/editor/static/js/flowchart.js
// Vertical flow diagram with nested blocks (IF/ELIF/ELSE lanes, loops with
// back-edges, ROI/WITH frames, collapsible FUNC groups).
// Input: the `tree` from /api/parse. Pure SVG, themed via CSS variables.

import { svg, catVar, fmtMs } from "./util.js";

const NODE_H = 30;
const VGAP = 18;
const HGAP = 20;
const PAD = 14;
const LABEL_GAP = 26;     // space above branch lanes for their labels
const PILL_H = 18;
const MIN_W = 112;
const MAX_W = 280;

let charW = 7.2;          // measured from the mono font once fonts load
function measureFont() {
  try {
    const c = document.createElement("canvas").getContext("2d");
    c.font = '12px "JetBrains Mono", ui-monospace, monospace';
    charW = c.measureText("MMMMMMMMMM").width / 10 || 7.2;
  } catch { /* keep default */ }
}

const collapsed = new Set();   // FUNC group ids the user collapsed (by line)
const expandedFuncs = new Set();

function nodeText(item) {
  const name = item.label || item.name || "";
  const args = item.args || "";
  return { name, args };
}

function stmtWidth(item) {
  const { name, args } = nodeText(item);
  const w = 12 + 10 + (name.length + (args ? args.length + 1 : 0)) * charW + 52;
  return Math.max(MIN_W, Math.min(MAX_W, Math.round(w)));
}

function isCollapsedFunc(item) {
  return item.kind === "block" && item.func && !expandedFuncs.has(item.line);
}

// ── Measure ────────────────────────────────────────────────────────────────
function measureSeq(items) {
  let w = 0, h = 0;
  items.forEach((it, i) => {
    measure(it);
    w = Math.max(w, it._w);
    h += it._h + (i ? VGAP : 0);
  });
  return { w, h };
}

function measure(item) {
  const hw = stmtWidth(item) + (item.kind === "block" ? 14 : 0);
  if (item.kind !== "block" || isCollapsedFunc(item)) {
    item._w = hw; item._h = NODE_H; item._hw = hw;
    return;
  }
  const lanes = item.branches.map(br => {
    const m = measureSeq(br.items || []);
    const lw = Math.max(m.w, Math.max(64, (br.label || "").length * 6.5 + 20));
    return { w: lw, h: br.items && br.items.length ? m.h : 8 };
  });
  item._lanes = lanes;
  const inner = lanes.reduce((s, l) => s + l.w, 0) + HGAP * Math.max(0, lanes.length - 1);
  const loopRoom = item.loop ? 26 : 0;
  item._w = Math.max(hw + loopRoom * 2, inner + PAD * 2 + loopRoom * 2);
  item._hw = hw;
  const maxH = Math.max(...lanes.map(l => l.h), 8);
  item._h = NODE_H + LABEL_GAP + maxH + 20 + PILL_H + 6;
}

// ── Render ─────────────────────────────────────────────────────────────────
export function renderFlowchart(svgEl, tree, opts = {}) {
  measureFont();
  const ctx = {
    edges: svg("g", { class: "f-edges" }),
    nodes: svg("g", { class: "f-nodes" }),
    labels: svg("g", { class: "f-labels" }),
    run: opts.run || null,
    onJump: opts.onJump || (() => {}),
    rerender: () => renderFlowchart(svgEl, tree, opts),
    order: [],
  };
  while (svgEl.firstChild) svgEl.firstChild.remove();
  svgEl.append(defs());

  const items = tree || [];
  const main = items;
  const m = measureSeq(main);
  const width = Math.max(m.w, 160) + 48;
  const top = 16;

  // START pill
  const cx = width / 2;
  const startW = 52;
  ctx.nodes.append(
    svg("rect", { class: "fn-start", x: cx - startW / 2, y: top, width: startW, height: 20, rx: 10 }),
    svg("text", { class: "fn-start-text", x: cx, y: top + 14, "text-anchor": "middle" }, "START"),
  );
  let y = top + 20 + VGAP;
  if (main.length) line(ctx, cx, top + 20, cx, y, { arrow: true });
  const bottom = drawSeq(ctx, main, cx, y);
  const height = (main.length ? bottom : y) + 24;

  svgEl.append(ctx.edges, ctx.nodes, ctx.labels);
  const scale = opts.scale || 1;
  svgEl.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svgEl.setAttribute("width", Math.round(width * scale));
  svgEl.setAttribute("height", Math.round(height * scale));
  wireKeyboard(svgEl, ctx);
  return { width, height, count: ctx.order.length };
}

function defs() {
  const d = svg("defs");
  d.append(
    svg("marker", { id: "fa", viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" },
      svg("path", { d: "M1 1.2 L7 4 L1 6.8 Z", class: "fe-arrow" })),
    svg("marker", { id: "fa-loop", viewBox: "0 0 8 8", refX: 7, refY: 4, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" },
      svg("path", { d: "M1 1.2 L7 4 L1 6.8 Z", class: "fe-arrow loop" })),
  );
  return d;
}

function line(ctx, x1, y1, x2, y2, o = {}) {
  ctx.edges.append(svg("path", {
    class: "fe" + (o.loop ? " loop" : ""),
    d: `M${x1} ${y1} L${x2} ${y2}`,
    "marker-end": o.arrow ? (o.loop ? "url(#fa-loop)" : "url(#fa)") : null,
  }));
}
function path(ctx, d, o = {}) {
  ctx.edges.append(svg("path", {
    class: "fe" + (o.loop ? " loop" : ""), d,
    "marker-end": o.arrow ? (o.loop ? "url(#fa-loop)" : "url(#fa)") : null,
  }));
}
/** Orthogonal path with rounded corners through points. */
function ortho(points, r = 6) {
  let d = `M${points[0][0]} ${points[0][1]}`;
  for (let i = 1; i < points.length - 1; i++) {
    const [x0, y0] = points[i - 1], [x1, y1] = points[i], [x2, y2] = points[i + 1];
    const d1 = Math.hypot(x1 - x0, y1 - y0), d2 = Math.hypot(x2 - x1, y2 - y1);
    const rr = Math.min(r, d1 / 2, d2 / 2);
    if (rr < 0.5) { d += ` L${x1} ${y1}`; continue; }
    const ax = x1 - (x1 - x0) / d1 * rr, ay = y1 - (y1 - y0) / d1 * rr;
    const bx = x1 + (x2 - x1) / d2 * rr, by = y1 + (y2 - y1) / d2 * rr;
    d += ` L${ax} ${ay} Q${x1} ${y1} ${bx} ${by}`;
  }
  const last = points[points.length - 1];
  return d + ` L${last[0]} ${last[1]}`;
}
function edgeLabel(ctx, x, y, text, anchor = "middle") {
  if (!text) return;
  const w = text.length * 5.6 + 8;
  const bx = anchor === "start" ? x - 2 : anchor === "end" ? x - w + 2 : x - w / 2;
  ctx.labels.append(
    svg("rect", { class: "fe-label-bg", x: bx, y: y - 9, width: w, height: 13, rx: 3 }),
    svg("text", { class: "fe-label", x: anchor === "start" ? x + 2 : anchor === "end" ? x - 2 : x, y: y + 1, "text-anchor": anchor }, text),
  );
}

function drawSeq(ctx, items, cx, y) {
  let prevBottom = null;
  for (const it of items) {
    if (prevBottom !== null) line(ctx, cx, prevBottom, cx, y, { arrow: true });
    const b = drawItem(ctx, it, cx, y);
    prevBottom = b;
    y = b + VGAP;
  }
  return prevBottom === null ? y : prevBottom;
}

function drawItem(ctx, it, cx, y) {
  if (it.kind !== "block") {
    drawNode(ctx, it, cx, y, it._w);
    return y + NODE_H;
  }
  if (isCollapsedFunc(it)) {
    drawNode(ctx, it, cx, y, it._hw, { chevron: "right", toggle: true });
    return y + NODE_H;
  }
  const x0 = cx - it._w / 2;
  const frameTop = y + NODE_H / 2;
  const bottom = y + it._h;
  const pillY = bottom - PILL_H - 2;
  ctx.edges.append(svg("rect", {
    class: "fb-frame" + (it.loop ? " loop" : "") + (it.func ? " func" : ""),
    x: x0 + 0.5, y: frameTop, width: it._w - 1, height: pillY + PILL_H / 2 - frameTop, rx: 10,
  }));
  drawNode(ctx, it, cx, y, it._hw, it.func ? { chevron: "down", toggle: true } : {});

  // lanes
  const lanes = it._lanes;
  const inner = lanes.reduce((s, l) => s + l.w, 0) + HGAP * Math.max(0, lanes.length - 1);
  let lx = cx - inner / 2;
  const laneTop = y + NODE_H + LABEL_GAP;
  const splitY = y + NODE_H + 10;
  const maxH = Math.max(...lanes.map(l => l.h), 8);
  const mergeY = laneTop + maxH + 10;
  it.branches.forEach((br, i) => {
    const lw = lanes[i].w;
    const lcx = lx + lw / 2;
    const items = br.items || [];
    const lbl = br.label === "body" || br.label === "region" ? "" : br.label;
    // header → lane
    if (items.length) {
      path(ctx, ortho([[cx, y + NODE_H], [cx, splitY], [lcx, splitY], [lcx, laneTop]]), { arrow: true });
      const end = drawSeq(ctx, items, lcx, laneTop);
      path(ctx, ortho([[lcx, end], [lcx, mergeY], [cx, mergeY], [cx, pillY]]), { arrow: lanes.length === 1 || lcx === cx });
    } else {
      path(ctx, ortho([[cx, y + NODE_H], [cx, splitY], [lcx, splitY], [lcx, mergeY], [cx, mergeY], [cx, pillY]]), { arrow: lcx === cx });
    }
    if (lbl) edgeLabel(ctx, lcx, splitY + 11, truncate(lbl, Math.floor(lw / 5.6)));
    lx += lw + HGAP;
  });
  if (lanes.length > 1) line(ctx, cx, mergeY, cx, pillY, { arrow: true });

  // footer pill
  const ft = it.footer || "END";
  const pw = ft.length * 6.4 + 16;
  ctx.nodes.append(
    svg("rect", { class: "fb-pill", x: cx - pw / 2, y: pillY, width: pw, height: PILL_H, rx: PILL_H / 2 }),
    svg("text", { class: "fb-pill-text", x: cx, y: pillY + 12.5, "text-anchor": "middle" }, ft),
  );

  // loop back-edge (right side)
  if (it.loop) {
    const rx = cx + it._hw / 2;
    const ex = x0 + it._w - 12;
    path(ctx, ortho([[cx + pw / 2, pillY + PILL_H / 2], [ex, pillY + PILL_H / 2], [ex, y + NODE_H / 2], [rx + 2, y + NODE_H / 2]]),
      { loop: true, arrow: true });
    const cnt = it.type === "repeat" && it.args ? "×" + it.args : "loop";
    edgeLabel(ctx, ex - 4, (pillY + y + NODE_H) / 2, cnt, "end");
  }
  return bottom;
}

function truncate(s, n) {
  if (!s) return "";
  return s.length > n ? s.slice(0, Math.max(1, n - 1)) + "…" : s;
}

function drawNode(ctx, it, cx, y, w, o = {}) {
  const x = cx - w / 2;
  const run = ctx.run;
  const lt = run && run.line_times ? run.line_times[String(it.line)] : null;
  const visited = run && run.visited ? run.visited.has(it.line) : false;
  const isErr = run && run.errorLine && run.errorLine === it.line;
  let cls = "fn";
  if (it.kind === "block" || it.type !== "command") cls += " kw";
  if (run && run.done) {
    if (isErr) cls += " error";
    else if (!visited) cls += " skipped";
  }
  if (o.toggle) cls += " fg-toggle";

  const { name, args } = nodeText(it);
  const timeText = lt ? fmtMs(lt.ms) + (lt.count > 1 ? ` ×${lt.count}` : "") : "";
  const timeW = timeText ? timeText.length * 5.6 + 6 : 0;
  const chevW = o.chevron ? 14 : 0;
  const avail = w - 12 - 10 - 10 - timeW - chevW;
  const maxChars = Math.max(3, Math.floor(avail / charW));
  let nm = name, ar = args;
  if (nm.length > maxChars) { nm = truncate(nm, maxChars); ar = ""; }
  else if (ar && nm.length + 1 + ar.length > maxChars) ar = truncate(ar, maxChars - nm.length - 1);

  const status = isErr ? "error" : (run && run.done ? (visited ? "ran" : "not executed") : "");
  const tip = [`Line ${it.line}`, `${name}${args ? " " + args : ""}`, it.category, timeText, status].filter(Boolean).join(" · ");
  const g = svg("g", {
    class: cls, tabindex: 0, role: "treeitem",
    "aria-label": tip, "data-line": it.line,
    "aria-expanded": o.toggle ? String(o.chevron === "down") : null,
    style: `--rail:${catVar(it.category)}`,
  });
  g.append(svg("title", null, tip));
  g.append(svg("rect", { class: "fn-box", x: x + 0.5, y: y + 0.5, width: w - 1, height: NODE_H - 1, rx: 7 }));
  g.append(svg("rect", { class: "fn-rail", x: x + 5, y: y + 8, width: 2.5, height: NODE_H - 16, rx: 1.25 }));
  let tx = x + 14;
  if (o.chevron) {
    const cy = y + NODE_H / 2;
    const p = o.chevron === "down" ? `M${tx} ${cy - 2} l3.5 3.5 l3.5 -3.5` : `M${tx + 1.5} ${cy - 3.5} l3.5 3.5 l-3.5 3.5`;
    g.append(svg("path", { d: p, fill: "none", stroke: "currentColor", "stroke-width": 1.5, "stroke-linecap": "round", "stroke-linejoin": "round", style: "color:var(--text-3)" }));
    tx += chevW;
  }
  const t = svg("text", { x: tx, y: y + 19.5 });
  t.append(svg("tspan", { class: "fn-name" }, nm));
  if (ar) t.append(svg("tspan", { class: "fn-args", dx: 6 }, ar));
  g.append(t);
  if (timeText) g.append(svg("text", { class: "fn-time", x: x + w - 10, y: y + 19, "text-anchor": "end" }, timeText));
  if (run && run.done && (visited || isErr) && !timeText) {
    g.append(svg("circle", { class: "fn-status", cx: x + w - 12, cy: y + NODE_H / 2, r: 3 }));
  }

  const activate = (ev) => {
    if (o.toggle && (ev.type === "dblclick" || ev.target.closest("path") || ev.key === " ")) {
      if (expandedFuncs.has(it.line)) expandedFuncs.delete(it.line); else expandedFuncs.add(it.line);
      ctx.rerender();
      return;
    }
    ctx.onJump(it.line);
  };
  g.addEventListener("click", activate);
  g.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(e); }
  });
  ctx.nodes.append(g);
  ctx.order.push(g);
}

function wireKeyboard(svgEl, ctx) {
  svgEl.onkeydown = (e) => {
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(e.key)) return;
    const nodes = ctx.order;
    if (!nodes.length) return;
    const i = nodes.indexOf(document.activeElement);
    let n = i;
    if (e.key === "ArrowDown") n = Math.min(nodes.length - 1, i + 1);
    if (e.key === "ArrowUp") n = Math.max(0, i - 1);
    if (e.key === "Home") n = 0;
    if (e.key === "End") n = nodes.length - 1;
    e.preventDefault();
    nodes[n]?.focus();
  };
}

/** Highlight nodes whose line matches the editor cursor. */
export function setActiveLine(svgEl, line) {
  for (const g of svgEl.querySelectorAll(".fn.active")) g.classList.remove("active");
  if (!line) return;
  for (const g of svgEl.querySelectorAll(`.fn[data-line="${line}"]`)) g.classList.add("active");
}
