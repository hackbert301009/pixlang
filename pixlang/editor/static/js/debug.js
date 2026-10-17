// pixlang/editor/static/js/debug.js
// Step debugger controller: session lifecycle (start / long-poll / control),
// toolbar state, the Debug side view (call stack, variables, watch, buffers,
// breakpoints), and the glue to the editor, preview, flow diagram and output.

import { $, $$, h, icon, api, apiKeepalive, store, toast, fmtMs, fmtKeys, debounce } from "./util.js";
import { showSide } from "./layout.js";
import { markDebug } from "./flowchart.js";
import {
  getBreakpoints, setBreakpoints, toggleBreakpoint, setDebugLine, scrollToLine, setReadOnly,
} from "./debug-editor.js";

const ACTIVE = new Set(["starting", "running", "paused"]);
const FINAL = new Set(["finished", "error", "stopped"]);

const KEYS = {
  start: "mod+shift+d", continue: "f8", step_over: "f10", step_into: "f11",
  step_out: "shift+f11", pause: "f6", stop: "shift+f5", toggle: "f9",
};
const LABELS = {
  continue: "Continue", step_over: "Step Over", step_into: "Step Into", step_out: "Step Out",
  pause: "Pause", stop: "Stop",
};

const REASONS = {
  breakpoint: "Breakpoint", step: "Step", entry: "Start of program", pause: "Paused",
  condition_error: "Breakpoint condition failed", end: "End of program",
};

/** "mod+shift+d" -> "Control+Shift+D Meta+Shift+D" (aria-keyshortcuts syntax). */
function ariaKeys(spec) {
  const part = (p, mod) => (p === "mod" ? mod : p === "shift" ? "Shift" : p.toUpperCase());
  const mods = spec.includes("mod") ? ["Control", "Meta"] : [""];
  return mods.map((m) => spec.split("+").map((p) => part(p, m)).join("+")).join(" ");
}

/** Which control buttons are usable in each state. */
const ENABLED = {
  idle: [], finished: [], error: [], stopped: [],
  starting: ["stop"],
  running: ["pause", "stop"],
  paused: ["continue", "step_over", "step_into", "step_out", "stop"],
};

export function createDebugger(deps) {
  const D = {
    sid: null, state: "idle", seq: 0, snap: null, code: "", loop: 0, sel: 0,
    watch: [], open: new Set(), changed: new Set(), prevVals: new Map(),
    outText: "", clearTimer: null, starting: false, widened: null,
    collapsed: new Set(store.get("dbgCollapsed", [])),
    startedAt: 0, pauseN: -1,
  };
  const view = () => deps.view();
  const app = () => $("#app");

  // ── Persistence (per file) ────────────────────────────────────────────────
  const loadMap = (name) => store.get(name, {}) || {};
  function saveMap(name, key, value) {
    const m = loadMap(name);
    if (value && value.length) m[key] = value; else delete m[key];
    store.set(name, m);
  }
  const persistBreakpoints = () => saveMap("breakpoints", deps.fileKey(),
    getBreakpoints(view().state).map(({ line, condition }) => ({ line, condition })));
  const persistWatch = () => saveMap("watch", deps.fileKey(), D.watch);

  function restoreForFile() {
    const saved = loadMap("breakpoints")[deps.fileKey()] || [];
    const doc = view().state.doc;
    setBreakpoints(view(), saved.filter((b) => b.line >= 1 && b.line <= doc.lines)
      .map((b) => ({ line: b.line, condition: b.condition || null, verified: verifiedLine(b.line) })));
    D.watch = (loadMap("watch")[deps.fileKey()] || []).filter((x) => typeof x === "string");
    renderAll();
  }

  /** Client-side guess whether a line can hold a breakpoint (server confirms on start). */
  function verifiedLine(line) {
    const lines = deps.stmtLines();
    if (!lines || !lines.size) return true;
    return lines.has(line);
  }

  // ── Session state ─────────────────────────────────────────────────────────
  const isActive = () => ACTIVE.has(D.state);
  const hasSession = () => !!D.sid && D.state !== "idle";

  function setState(state) {
    D.state = state;
    app().dataset.debug = state;
    renderToolbar();
    renderStatus();
    renderPreviewBadge();
    renderStatusBar();
    deps.onStateChange && deps.onStateChange(state);
  }

  async function start() {
    if (D.starting) return;
    D.starting = true;
    const v = view();
    try {
      await stopQuietly();
      resetSession();
      const code = v.state.doc.toString();
      const body = {
        code, breakpoints: getBreakpoints(v.state).map(({ line, condition }) => ({ line, condition })),
        watch: D.watch, ...deps.runBody(),
      };
      D.code = code;
      D.startedAt = Date.now();
      setState("starting");
      let snap;
      try {
        snap = await api("/api/debug/start", body);
      } catch (e) {
        setState("idle");
        toast(e.message, "err");
        return;
      }
      D.sid = snap.session_id;
      D.loop += 1;
      lockEditor(true);
      widenSidebar();
      if (window.innerWidth >= 1024) showSide("debug", { toggle: false });
      deps.appendOutput([h("span", { class: "o-dim" }, `── ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}  debug  ${deps.fileName()} ${"─".repeat(16)}\n`)]);
      apply(snap);
      if (!FINAL.has(snap.state)) poll(D.loop);
    } finally {
      D.starting = false;
    }
  }

  async function stopQuietly() {
    if (!D.sid || !isActive()) return;
    const sid = D.sid;
    D.loop += 1;
    try { await api(`/api/debug/${sid}/control`, { action: "stop", settle: 0.2 }); } catch { /* gone */ }
  }

  function resetSession() {
    clearTimeout(D.clearTimer);
    D.sid = null; D.snap = null; D.seq = 0; D.sel = 0; D.outText = ""; D.pauseN = -1;
    D.changed = new Set(); D.prevVals = new Map();
    setDebugLine(view(), null);
    markDebug(deps.flowSvg(), null);
  }

  /** Back to the idle UI (file switch, edit after a finished run). */
  function reset() {
    if (isActive()) { stopQuietly(); }
    D.loop += 1;
    resetSession();
    lockEditor(false);
    restoreSidebar();
    setState("idle");
    deps.setPausedPreview && deps.setPausedPreview(null);
    renderAll();
  }

  async function poll(loopId) {
    let failures = 0;
    while (loopId === D.loop && D.sid) {
      let snap;
      try {
        snap = await api(`/api/debug/${D.sid}/state?wait=${D.seq}&timeout=20`);
        failures = 0;
      } catch (e) {
        if (loopId !== D.loop) return;
        if (e.status === 404 || ++failures > 3) {
          toast("Lost connection to the debug session", "err");
          lockEditor(false);
          setDebugLine(view(), null);
          setState("stopped");
          return;
        }
        await new Promise((r) => setTimeout(r, 400 * failures));
        continue;
      }
      if (loopId !== D.loop) return;
      apply(snap);
      if (FINAL.has(snap.state)) return;
    }
  }

  async function control(action) {
    if (!D.sid) return;
    if (action === "stop") {
      if (!isActive()) return;
    } else if (!(ENABLED[D.state] || []).includes(action)) {
      return;
    }
    // Optimistic: the buttons go idle immediately so a double-press cannot double-step.
    if (action !== "stop" && action !== "pause") { D.state = "running"; renderToolbar(); }
    try {
      const snap = await api(`/api/debug/${D.sid}/control`, { action });
      apply(snap);
    } catch (e) {
      if (e.status !== 409) toast(e.message, "err");
      else if (D.snap) setState(D.snap.state);
    }
  }

  // ── Applying a server snapshot ────────────────────────────────────────────
  function apply(snap) {
    if (snap.session_id !== D.sid && D.sid !== null) return;
    if (D.sid === null) D.sid = snap.session_id;
    if (snap.seq < D.seq) return;
    const newPause = snap.state === "paused" && snap.pause !== D.pauseN;
    D.seq = snap.seq;
    D.snap = snap;
    if (snap.state === "paused") D.pauseN = snap.pause;

    if (snap.state !== "running") $("#dbg-scroll").classList.remove("stale");
    if (snap.state === "paused") {
      clearTimeout(D.clearTimer);
      if (newPause) { D.sel = 0; trackChanges(snap); }
      setState("paused");
      renderAll();
      if (newPause) afterPause(snap);
    } else if (snap.state === "running" || snap.state === "starting") {
      setState(snap.state);
      renderBreakpointsList();
      syncGutterFlags(snap);
      clearTimeout(D.clearTimer);
      // Keep the band for quick steps; drop it (and dim the stale panels) if the program keeps running.
      D.clearTimer = setTimeout(() => {
        if (D.state !== "running") return;
        setDebugLine(view(), null);
        $("#dbg-scroll").classList.add("stale");
        markDebug(deps.flowSvg(), D.snap ? { current: null, visited: new Set(D.snap.visited_lines || []) } : null);
      }, 250);
    } else if (FINAL.has(snap.state)) {
      clearTimeout(D.clearTimer);
      setDebugLine(view(), null);
      lockEditor(false);
      D.sel = 0;
      setState(snap.state);
      trackChanges(snap);
      renderAll();
      finish(snap);
    }
  }

  function afterPause(snap) {
    const loc = snap.location;
    const line = loc && loc.line;
    if (line) {
      setDebugLine(view(), line);
      scrollToLine(view(), line);
      $("#sb-cursor").textContent = `Ln ${line}, Col ${(loc.col || 1)}`;
    }
    const text = line ? `Paused at line ${line}${snap.reason ? ", " + (REASONS[snap.reason] || snap.reason).toLowerCase() : ""}` : "Paused";
    announce(text);
    markDebug(deps.flowSvg(), { current: line, visited: new Set(snap.visited_lines || []) });
    deps.onTiming && deps.onTiming(snap.stats);
    appendStdout(snap.stdout || "");
    updatePreview(snap);
    if (snap.message) toast(snap.message, "err", 3600);
  }

  function finish(snap) {
    const r = snap.result;
    appendStdout((r && r.stdout) || snap.stdout || "");
    markDebug(deps.flowSvg(), null);
    if (!r) return;
    if (snap.state === "stopped") {
      announce(r.stop_reason === "idle" ? "Debug session stopped after inactivity" : "Debug session stopped");
      if (r.stop_reason === "idle") toast("Debug session ended after a period of inactivity", "err", 4000);
      deps.onStopped && deps.onStopped(r, D.code);
    } else {
      announce(snap.state === "error" ? `Debug session failed${r.error_info && r.error_info.line ? " at line " + r.error_info.line : ""}` : "Debug session finished");
      deps.onFinal(r, D.code, { debug: true, quiet: false });
      const ei = r.error_info;
      if (ei && ei.line && ei.kind !== "syntax") scrollToLine(view(), ei.line, { center: true });
    }
  }

  function appendStdout(text) {
    if (text === D.outText) return;
    const delta = text.startsWith(D.outText) ? text.slice(D.outText.length) : text;
    D.outText = text;
    if (delta) deps.appendOutput([delta.endsWith("\n") ? delta : delta + "\n"]);
  }

  function updatePreview(snap) {
    const cps = (snap.buffers || []).filter((b) => b.url).map((b) => ({
      name: b.name, key: b.key, url: b.url, thumb_url: b.thumb_url, width: b.width,
      height: b.height, channels: b.channels, dtype: b.dtype,
    }));
    deps.preview.setResult({
      ok: true, image: snap.image ? { ...snap.image, label: "Current image", kind: "CUR" } : null,
      before: snap.before, checkpoints: cps, stats: snap.stats || {},
    });
    deps.preview.setStale(null);
    renderPreviewBadge();
  }

  function trackChanges(snap) {
    const cur = new Map();
    const stack = snap.stack || [];
    const depth = Math.max(0, stack.length - 1);
    for (const v of (snap.variables && snap.variables.globals) || []) cur.set("g:" + v.name, v.value);
    ((snap.variables && snap.variables.frames) || []).forEach((vars, i) => {
      for (const v of vars) cur.set(`f${depth - i}:${v.name}`, v.value);
    });
    const changed = new Set();
    if (D.prevVals.size) {
      for (const [k, val] of cur) if (D.prevVals.get(k) !== val) changed.add(k);
    }
    D.changed = changed;
    D.prevVals = cur;
  }

  // ── Editor integration ────────────────────────────────────────────────────
  function lockEditor(on) {
    setReadOnly(view(), on);
    $(".editor-pane").classList.toggle("debugging", on);
  }

  function syncGutterFlags(snap) {
    const server = new Map((snap.breakpoints || []).map((b) => [b.line, b]));
    const local = getBreakpoints(view().state);
    let dirty = false;
    const merged = local.map((b) => {
      const s = server.get(b.line);
      const verified = s ? s.verified : b.verified;
      if (verified !== b.verified) dirty = true;
      return { ...b, verified };
    });
    if (dirty) setBreakpoints(view(), merged);
  }

  function widenSidebar() {
    const root = document.documentElement;
    const w = parseFloat(getComputedStyle(app().querySelector(".sidebar")).width) || 0;
    if (w && w < 288 && window.innerWidth >= 1280 && D.widened === null) {
      D.widened = root.style.getPropertyValue("--sidebar-w");
      root.style.setProperty("--sidebar-w", "288px");
    }
  }
  function restoreSidebar() {
    if (D.widened === null) return;
    const root = document.documentElement;
    if (D.widened) root.style.setProperty("--sidebar-w", D.widened); else root.style.removeProperty("--sidebar-w");
    D.widened = null;
  }

  // ── Breakpoints ───────────────────────────────────────────────────────────
  function bpChanged() {
    persistBreakpoints();
    renderBreakpointsList();
    if (D.sid && isActive()) {
      const list = getBreakpoints(view().state).map(({ line, condition }) => ({ line, condition }));
      api(`/api/debug/${D.sid}/breakpoints`, { breakpoints: list })
        .then((snap) => { if (snap.session_id === D.sid) { syncGutterFlags(snap); D.snap = { ...D.snap, breakpoints: snap.breakpoints }; renderBreakpointsList(); } })
        .catch(() => { /* session ended meanwhile */ });
    }
  }

  function toggleLine(line) {
    const list = toggleBreakpoint(view(), line);
    const bp = list.find((b) => b.line === line);
    if (bp) {
      bp.verified = verifiedLine(line);
      setBreakpoints(view(), list);
    }
    bpChanged();
    announce(bp ? `Breakpoint set on line ${line}` : `Breakpoint removed from line ${line}`);
  }

  function cursorLine() {
    const st = view().state;
    return st.doc.lineAt(st.selection.main.head).number;
  }

  async function editCondition(line) {
    line = line || cursorLine();
    const existing = getBreakpoints(view().state).find((b) => b.line === line);
    const r = await deps.dialog({
      title: `Breakpoint condition · line ${line}`,
      body: "Pause here only when this expression is true. Leave it empty for an unconditional breakpoint.",
      input: { value: (existing && existing.condition) || "", placeholder: "$i == 2" },
      actions: [{ label: "Cancel", value: "cancel", kind: "btn-ghost" }, { label: "Save", value: "ok", kind: "btn-primary", default: true }],
    });
    if (r.action !== "ok") return;
    const cond = r.value.trim() || null;
    const list = getBreakpoints(view().state).filter((b) => b.line !== line);
    list.push({ line, condition: cond, verified: verifiedLine(line) });
    setBreakpoints(view(), list);
    bpChanged();
  }

  function clearBreakpoints() {
    setBreakpoints(view(), []);
    bpChanged();
  }

  // ── Watch ─────────────────────────────────────────────────────────────────
  function syncWatch() {
    persistWatch();
    renderWatch();
    if (D.sid && isActive()) {
      api(`/api/debug/${D.sid}/watch`, { exprs: D.watch })
        .then((snap) => { if (snap.session_id === D.sid) apply({ ...snap, seq: Math.max(snap.seq, D.seq) }); })
        .catch(() => {});
    }
  }
  function addWatch(expr) {
    expr = (expr || "").trim();
    if (!expr || D.watch.includes(expr)) return;
    D.watch.push(expr);
    syncWatch();
  }
  function removeWatch(i) { D.watch.splice(i, 1); syncWatch(); }
  function editWatch(i, expr) {
    expr = expr.trim();
    if (!expr) { removeWatch(i); return; }
    if (D.watch.includes(expr) && D.watch[i] !== expr) { removeWatch(i); return; }
    D.watch[i] = expr;
    syncWatch();
  }

  // ── Announcements ─────────────────────────────────────────────────────────
  function announce(text) {
    const el = $("#dbg-live");
    if (el) el.textContent = text;
  }

  // ── Rendering: toolbar / status ───────────────────────────────────────────
  const enabled = (a) => (ENABLED[D.state] || []).includes(a);

  function renderToolbar() {
    const grp = $("#debug-group");
    grp.dataset.state = D.state;
    for (const a of Object.keys(LABELS)) {
      const b = $("#dbg-" + a.replace("_", "-"));
      const on = enabled(a);
      b.disabled = !on;
      b.setAttribute("aria-disabled", String(!on));
    }
    const btn = $("#btn-debug");
    const active = isActive();
    btn.classList.toggle("restart", active);
    btn.querySelector("span").textContent = active ? "Restart" : "Debug";
    btn.querySelector("use").setAttribute("href", active ? "#i-refresh" : "#i-bug");
    btn.title = `${active ? "Restart debugging" : "Start debugging"} (${fmtKeys(KEYS.start)})`;
    btn.disabled = D.state === "starting";
    // Normal runs are unavailable while the interpreter belongs to the debugger.
    const run = $("#btn-run");
    run.classList.toggle("blocked", active);
    run.setAttribute("aria-disabled", String(active));
    run.title = active ? "Stop debugging to run the pipeline" : "";
  }

  function stateLabel() {
    const s = D.snap;
    const line = s && s.location && s.location.line;
    switch (D.state) {
      case "starting": return { title: "Starting…", sub: "Launching the debug session" };
      case "running": return { title: "Running", sub: "Executing until the next breakpoint" };
      case "paused": return {
        title: line ? `Paused at line ${line}` : "Paused",
        sub: [REASONS[s.reason] || null, s.stack && s.stack[0] ? s.stack[0].name : null].filter(Boolean).join(" · "),
      };
      case "finished": return { title: "Finished", sub: s && s.result ? `${fmtMs(s.result.stats.total_ms)} · ${s.result.stats.commands_executed} commands` : "" };
      case "error": {
        const ei = s && s.result && s.result.error_info;
        const msg = ei ? ei.message.replace(/^(\[PixLang\]\s*)?Error at line \d+(, col \d+)?( in \S+)?:\s*/i, "") : "";
        return { title: ei && ei.line ? `Failed at line ${ei.line}` : "Failed", sub: msg };
      }
      case "stopped": return { title: "Stopped", sub: s && s.result && s.result.stop_reason === "idle" ? "Ended after inactivity" : "Session ended by you" };
      default: return { title: "Not debugging", sub: "" };
    }
  }

  function renderStatus() {
    const box = $("#dbg-status");
    if (!box) return;
    box.dataset.state = D.state;
    const { title, sub } = stateLabel();
    const running = D.state === "running" || D.state === "starting";
    const idle = D.state === "idle";
    $("#dbg-status-dot").replaceChildren(running ? icon("loader", "icon-sm spin") : "");
    $("#dbg-status-title").textContent = title;
    const subEl = $("#dbg-status-sub");
    subEl.textContent = sub;
    subEl.hidden = !sub;
    $("#dbg-start").hidden = running || D.state === "paused";
    $("#dbg-start-label").textContent = idle ? "Start debugging" : "Debug again";
  }

  function renderPreviewBadge() {
    const el = $("#pv-debug");
    if (!el) return;
    const s = D.snap;
    const line = s && s.location && s.location.line;
    if (D.state === "paused") { el.hidden = false; el.dataset.state = "paused"; el.querySelector(".pv-debug-text").textContent = line ? `Paused at line ${line}` : "Paused"; }
    else if (D.state === "running" || D.state === "starting") { el.hidden = false; el.dataset.state = "running"; el.querySelector(".pv-debug-text").textContent = "Running…"; }
    else el.hidden = true;
  }

  function renderStatusBar() {
    const sb = $("#sb-run");
    const s = D.snap;
    const line = s && s.location && s.location.line;
    if (D.state === "paused") { sb.className = "sb-item sb-run dbg"; sb.replaceChildren(icon("bug", "icon-xs"), line ? `Paused at line ${line}` : "Paused"); }
    else if (D.state === "running" || D.state === "starting") { sb.className = "sb-item sb-run dbg"; sb.replaceChildren(icon("loader", "icon-xs spin"), "Debugging…"); }
  }

  // ── Rendering: side view ──────────────────────────────────────────────────
  function renderAll() {
    renderStatus();
    renderStack();
    renderVars();
    renderWatch();
    renderBuffers();
    renderBreakpointsList();
    renderToolbar();
  }

  function setBadge(id, n) {
    const el = $("#" + id);
    el.textContent = n ? String(n) : "";
    el.hidden = !n;
  }

  const empty = (iconName, text) => h("div", { class: "dbg-empty" }, icon(iconName, "icon-sm"), text);

  function renderStack() {
    const host = $("#dbg-stack");
    const stack = (D.snap && D.snap.stack) || [];
    setBadge("dbg-stack-count", stack.length);
    if (!stack.length) {
      host.replaceChildren(empty("layers", D.state === "idle" ? "Not running."
        : isActive() ? "Waiting for the next pause…" : "No active call stack."));
      return;
    }
    host.replaceChildren(h("div", { role: "listbox", "aria-label": "Call stack", class: "dbg-frames" },
      stack.map((f, i) => h("button", {
        class: "dbg-frame" + (i === D.sel ? " selected" : "") + (i === 0 ? " top" : ""),
        role: "option", "aria-selected": String(i === D.sel), title: `${f.name} — line ${f.line || "?"}`,
        onclick: () => selectFrame(i),
      },
      icon(i === 0 ? "arrow-right" : "corner", "icon-xs"),
      h("span", { class: "dbg-frame-name" }, f.name),
      h("span", { class: "dbg-frame-loc" }, f.file ? `${f.unit || "file"}:${f.line ?? "?"}` : `L${f.line ?? "?"}`)))));
  }

  function selectFrame(i) {
    D.sel = i;
    const f = D.snap.stack[i];
    const cur = D.snap.location && D.snap.location.line;
    if (f && f.line && !f.file) {
      setDebugLine(view(), cur, i === 0 ? null : f.line);
      scrollToLine(view(), f.line, { center: false });
    }
    renderStack();
    renderVars();
  }

  // Variable tree --------------------------------------------------------------
  const CHIP = { number: "num", string: "str", bool: "bool", buffer: "buf", pointer: "ptr", list: "list", other: "obj" };

  function varRow(v, depth, key, changedKey) {
    const kids = v.children && v.children.length ? v.children : null;
    const open = kids && D.open.has(key);
    const typeText = v.kind === "buffer" ? v.type.replace("buffer(", "buf·").replace(")", "")
      : v.kind === "list" ? `${v.type}[${v.length}]` : v.type;
    const changed = changedKey && D.changed.has(changedKey);
    const row = h("div", {
      class: "dbg-var" + (changed ? " changed" : ""), role: "treeitem", tabindex: -1,
      "aria-level": depth + 1, "aria-expanded": kids ? String(!!open) : null,
      "data-key": key, style: { "--depth": depth },
      title: `${v.name} = ${v.value}  (${v.type})`,
      onclick: () => { if (kids) toggleOpen(key); },
    },
    kids ? icon(open ? "chevron-down" : "chevron-right", "icon-xs dbg-chev") : h("span", { class: "dbg-chev-pad" }),
    h("span", { class: "dbg-var-name" }, v.name),
    h("span", { class: "dbg-var-val k-" + v.kind }, v.value),
    h("span", { class: "dbg-chip k-" + v.kind }, typeText || CHIP[v.kind]),
    depth === 0 && changedKey ? h("button", {
      class: "dbg-var-act", "aria-label": `Watch ${v.name}`, title: "Add to watch", tabindex: -1,
      onclick: (e) => { e.stopPropagation(); addWatch("$" + v.name); },
    }, icon("eye", "icon-xs")) : null);
    const out = [row];
    if (open) {
      v.children.forEach((c, i) => out.push(...varRow(c, depth + 1, `${key}/${i}`, null)));
      if (v.truncated) out.push(h("div", { class: "dbg-more", style: { "--depth": depth + 1 } }, `… ${v.length - v.children.length} more`));
    }
    return out;
  }

  function toggleOpen(key) {
    if (D.open.has(key)) D.open.delete(key); else D.open.add(key);
    const focus = document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.key : null;
    renderVars();
    renderWatch();
    if (focus) { const el = $(`.dbg-var[data-key="${CSS.escape(focus)}"]`); if (el) el.focus(); }
  }

  function renderVars() {
    const host = $("#dbg-vars");
    const snap = D.snap;
    const vars = snap && snap.variables;
    const stack = (snap && snap.stack) || [];
    const locals = vars && vars.frames[D.sel] ? vars.frames[D.sel] : [];
    const globals = vars ? vars.globals : [];
    setBadge("dbg-vars-count", locals.length + globals.length);
    if (!locals.length && !globals.length) {
      host.replaceChildren(empty("variable", D.state === "idle" ? "Start debugging to inspect variables."
        : isActive() ? "Waiting for the next pause…" : "No variables were set."));
      return;
    }
    const depth = Math.max(0, stack.length - 1);
    const frameIdx = depth - D.sel;
    const groups = [];
    if (locals.length) {
      const name = stack[D.sel] ? stack[D.sel].name.split("(")[0] : "Locals";
      groups.push(["Locals", name, locals.flatMap((v) => varRow(v, 0, `l${frameIdx}:${v.name}`, `f${frameIdx}:${v.name}`))]);
    }
    if (globals.length) groups.push(["Globals", `${globals.length}`, globals.flatMap((v) => varRow(v, 0, `g:${v.name}`, `g:${v.name}`))]);
    host.replaceChildren(h("div", { class: "dbg-tree", role: "tree", "aria-label": "Variables" },
      groups.flatMap(([label, sub, rows]) => [
        h("div", { class: "dbg-group", role: "presentation" }, label, h("span", null, sub)),
        ...rows,
      ])));
  }

  // Watch ----------------------------------------------------------------------
  function watchResult(expr) {
    const list = (D.snap && D.snap.watch) || [];
    return list.find((w) => w.expr === expr) || null;
  }

  function renderWatch() {
    const host = $("#dbg-watch-list");
    setBadge("dbg-watch-count", D.watch.length);
    if (!D.watch.length) {
      host.replaceChildren(h("div", { class: "dbg-empty slim" }, "Evaluate expressions at every pause, e.g. ", h("code", null, "$i * 2"), " or ", h("code", null, "_.width"), "."));
      return;
    }
    const live = D.state === "paused" || FINAL.has(D.state);
    host.replaceChildren(h("div", { class: "dbg-tree", role: "tree", "aria-label": "Watch expressions" },
      D.watch.flatMap((expr, i) => {
        const r = live ? watchResult(expr) : null;
        const key = `w:${expr}`;
        const kids = r && r.ok && r.children && r.children.length ? r.children : null;
        const open = kids && D.open.has(key);
        const row = h("div", {
          class: "dbg-var dbg-watch" + (r && r.ok === false ? " err" : ""), role: "treeitem", tabindex: -1,
          "aria-level": 1, "aria-expanded": kids ? String(!!open) : null, "data-key": key,
          style: { "--depth": 0 },
          title: r ? (r.ok ? `${expr} = ${r.value}` : `${expr}: ${r.error}`) : expr,
          onclick: () => { if (kids) toggleOpen(key); },
          ondblclick: () => beginEditWatch(i, row),
        },
        kids ? icon(open ? "chevron-down" : "chevron-right", "icon-xs dbg-chev") : h("span", { class: "dbg-chev-pad" }),
        h("span", { class: "dbg-var-name mono-expr" }, expr),
        r && r.ok ? h("span", { class: "dbg-var-val k-" + r.kind }, r.value)
          : r ? h("span", { class: "dbg-var-val error" }, r.error)
            : h("span", { class: "dbg-var-val pending" }, "not evaluated"),
        r && r.ok ? h("span", { class: "dbg-chip k-" + r.kind }, r.kind === "buffer" ? r.type.replace("buffer(", "buf·").replace(")", "") : r.type) : null,
        h("button", { class: "dbg-var-act always", "aria-label": `Remove ${expr}`, title: "Remove", onclick: (e) => { e.stopPropagation(); removeWatch(i); } }, icon("x", "icon-xs")));
        const out = [row];
        if (open) kids.forEach((c, j) => out.push(...varRow(c, 1, `${key}/${j}`, null)));
        return out;
      })));
  }

  function beginEditWatch(i, row) {
    const input = h("input", { class: "dbg-inline-input mono", value: D.watch[i], "aria-label": "Edit watch expression", spellcheck: "false" });
    const done = (commit) => {
      input.onblur = null;
      if (commit) editWatch(i, input.value); else renderWatch();
    };
    input.onkeydown = (e) => {
      if (e.key === "Enter") { e.preventDefault(); done(true); }
      else if (e.key === "Escape") { e.preventDefault(); done(false); }
      e.stopPropagation();
    };
    input.onblur = () => done(true);
    row.replaceChildren(input);
    input.focus();
    input.select();
  }

  // Buffers --------------------------------------------------------------------
  function renderBuffers() {
    const host = $("#dbg-buffers");
    const list = (D.snap && D.snap.buffers) || [];
    setBadge("dbg-buffers-count", list.length);
    if (!list.length) {
      host.replaceChildren(empty("image", D.state === "idle" ? "Buffers and checkpoints appear here."
        : "No buffers or checkpoints yet."));
      return;
    }
    host.replaceChildren(h("div", { class: "dbg-bufs" }, list.map((b) => h("button", {
      class: "dbg-buf" + (b.changed ? " changed" : ""), title: `${b.name} · ${b.value}${b.changed ? " · changed" : ""}`,
      "aria-label": `${b.name}, ${b.value}${b.changed ? ", changed" : ""}`,
      onclick: () => { if (b.key) { deps.preview.showStep(b.key); } },
    },
    b.thumb_url ? h("img", { src: b.thumb_url, alt: "", loading: "lazy", draggable: "false" })
      : h("span", { class: "dbg-buf-noimg" }, icon("hash", "icon-sm")),
    (b.changed || b.new) ? h("span", { class: "dbg-buf-flag" }, b.new ? "new" : "changed") : null,
    h("span", { class: "dbg-buf-name" }, b.name),
    h("span", { class: "dbg-buf-meta" }, b.value)))));
  }

  // Breakpoints ----------------------------------------------------------------
  function renderBreakpointsList() {
    const host = $("#dbg-bps");
    const list = getBreakpoints(view().state);
    setBadge("dbg-bps-count", list.length);
    $("#dbg-clear-bps").disabled = !list.length;
    if (!list.length) {
      host.replaceChildren(h("div", { class: "dbg-empty slim" }, "Click the editor gutter or press ", h("kbd", null, fmtKeys(KEYS.toggle)), " to add one."));
      return;
    }
    const server = new Map(((D.snap && D.snap.breakpoints) || []).map((b) => [b.line, b]));
    const doc = view().state.doc;
    host.replaceChildren(...list.map((b) => {
      const text = b.line <= doc.lines ? doc.line(b.line).text.trim() : "";
      const sv = server.get(b.line);
      const err = sv && sv.error;
      const unverified = b.verified === false;
      return h("div", { class: "dbg-bp" + (unverified ? " unverified" : "") },
        h("button", {
          class: "dbg-bp-main", title: unverified ? "No statement on this line — it will not pause" : text,
          onclick: () => deps.jump(b.line),
        },
        h("span", { class: "bp-dot" + (b.condition ? " cond" : "") + (unverified ? " unverified" : "") }),
        h("span", { class: "dbg-bp-line" }, "Ln " + b.line),
        h("span", { class: "dbg-bp-text" }, text || "(empty)"),
        b.condition ? h("span", { class: "dbg-chip cond" + (err ? " err" : ""), title: err || `Condition: ${b.condition}` }, "if " + b.condition) : null),
        h("button", { class: "dbg-var-act always", "aria-label": `Edit condition for line ${b.line}`, title: "Edit condition", onclick: () => editCondition(b.line) }, icon("pencil", "icon-xs")),
        h("button", { class: "dbg-var-act always", "aria-label": `Remove breakpoint on line ${b.line}`, title: "Remove", onclick: () => { toggleLine(b.line); } }, icon("x", "icon-xs")));
    }));
  }

  // ── Section collapse + tree keyboard ──────────────────────────────────────
  function wireSections() {
    for (const head of $$(".dbg-sec-head")) {
      const id = head.closest(".dbg-sec").dataset.sec;
      const apply = () => {
        const closed = D.collapsed.has(id);
        head.setAttribute("aria-expanded", String(!closed));
        head.closest(".dbg-sec").classList.toggle("closed", closed);
      };
      apply();
      head.addEventListener("click", () => {
        if (D.collapsed.has(id)) D.collapsed.delete(id); else D.collapsed.add(id);
        store.set("dbgCollapsed", [...D.collapsed]);
        apply();
      });
    }
    // Variable / watch trees: arrow keys move, Left/Right fold, Delete removes a watch.
    // Roving tabindex: exactly one row per tree is a tab stop.
    for (const tree of [$("#dbg-vars"), $("#dbg-watch-list")]) {
      tree.addEventListener("keydown", (e) => {
        const row = e.target.closest && e.target.closest(".dbg-var");
        if (!row || e.target.tagName === "INPUT") return;
        const rows = $$(".dbg-var", tree);
        const i = rows.indexOf(row);
        const own = e.target === row;
        let n = i;
        if (e.key === "ArrowDown") n = Math.min(rows.length - 1, i + 1);
        else if (e.key === "ArrowUp") n = Math.max(0, i - 1);
        else if (e.key === "Home") n = 0;
        else if (e.key === "End") n = rows.length - 1;
        else if (own && e.key === "ArrowRight" && row.getAttribute("aria-expanded") === "false") { row.click(); e.preventDefault(); return; }
        else if (own && e.key === "ArrowLeft" && row.getAttribute("aria-expanded") === "true") { row.click(); e.preventDefault(); return; }
        else if (own && (e.key === "Enter" || e.key === " ") && row.hasAttribute("aria-expanded")) { row.click(); e.preventDefault(); return; }
        else if (own && e.key === "Delete" && row.classList.contains("dbg-watch")) { removeWatch($$(".dbg-watch", tree).indexOf(row)); e.preventDefault(); return; }
        else return;
        e.preventDefault();
        rows[n].focus();
      });
      tree.addEventListener("focusin", (e) => {
        const row = e.target.closest && e.target.closest(".dbg-var");
        if (!row) return;
        for (const r of $$(".dbg-var[tabindex='0']", tree)) r.tabIndex = -1;
        row.tabIndex = 0;
      });
      new MutationObserver(() => {
        const first = $(".dbg-var", tree);
        if (first && !$(".dbg-var[tabindex='0']", tree)) first.tabIndex = 0;
      }).observe(tree, { childList: true, subtree: true });
    }
  }

  // ── Keyboard ──────────────────────────────────────────────────────────────
  function handleKey(e) {
    const mod = deps.isMac ? e.metaKey : e.ctrlKey;
    const k = e.key;
    if (mod && e.shiftKey && !e.altKey && k.toLowerCase() === "d") { e.preventDefault(); start(); return true; }
    if (e.ctrlKey || e.metaKey || e.altKey) return false;
    if (k === "F9") { e.preventDefault(); toggleLine(cursorLine()); return true; }
    if (!hasSession() || !isActive()) return false;
    const act = k === "F5" ? (e.shiftKey ? "stop" : "continue")
      : k === "F8" ? "continue"
        : k === "F10" ? "step_over"
          : k === "F11" ? (e.shiftKey ? "step_out" : "step_into")
            : k === "F6" ? "pause" : null;
    if (!act) return false;
    e.preventDefault();
    control(act);
    return true;
  }

  // ── Palette ───────────────────────────────────────────────────────────────
  function paletteItems() {
    const items = [
      { group: "Debug", title: isActive() ? "Restart debugging" : "Start debugging", icon: "bug", keys: KEYS.start, run: start, weight: 18 },
    ];
    for (const a of ["continue", "step_over", "step_into", "step_out", "pause", "stop"]) {
      if (!enabled(a)) continue;
      items.push({ group: "Debug", title: (a === "stop" ? "Stop debugging" : LABELS[a]), icon: a === "stop" ? "stop" : a.replace("_", "-"), keys: KEYS[a], run: () => control(a), weight: 12 });
    }
    items.push(
      { group: "Debug", title: "Toggle breakpoint on this line", icon: "circle-dot", keys: KEYS.toggle, run: () => toggleLine(cursorLine()) },
      { group: "Debug", title: "Edit breakpoint condition…", icon: "pencil", run: () => editCondition() },
      { group: "Debug", title: "Remove all breakpoints", icon: "trash", run: clearBreakpoints },
      { group: "Debug", title: "Add watch expression…", icon: "eye", run: promptWatch },
      { group: "Debug", title: "Show debug panel", icon: "bug", run: () => showSide("debug", { toggle: false }) },
    );
    return items;
  }

  async function promptWatch() {
    const r = await deps.dialog({
      title: "Add watch expression",
      body: "Any PixLang expression. It is evaluated read-only at every pause.",
      input: { value: "", placeholder: "$i * 2" },
      validate: (v) => (v.trim() ? null : "Enter an expression"),
      actions: [{ label: "Cancel", value: "cancel", kind: "btn-ghost" }, { label: "Add", value: "ok", kind: "btn-primary", default: true }],
    });
    if (r.action === "ok") { addWatch(r.value); showSide("debug", { toggle: false }); }
  }

  /** The document changed: breakpoints moved with it; a finished session is stale. */
  const settleEdit = debounce(() => { persistBreakpoints(); renderBreakpointsList(); }, 350);
  function onDocChange() {
    if (FINAL.has(D.state)) reset();
    settleEdit();
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  function init() {
    wireSections();
    $("#btn-debug").addEventListener("click", start);
    $("#dbg-start").addEventListener("click", start);
    for (const a of Object.keys(LABELS)) {
      const b = $("#dbg-" + a.replace("_", "-"));
      b.title = `${LABELS[a]} (${fmtKeys(KEYS[a])})`;
      b.setAttribute("aria-label", LABELS[a]);
      b.setAttribute("aria-keyshortcuts", ariaKeys(KEYS[a]));
      b.addEventListener("click", () => control(a));
    }
    $("#dbg-clear-bps").addEventListener("click", clearBreakpoints);
    $("#dbg-watch-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const inp = $("#dbg-watch-input");
      addWatch(inp.value);
      inp.value = "";
    });
    window.addEventListener("beforeunload", () => { if (isActive()) apiKeepalive("/api/debug/stop"); });
    // A reloaded page must not leave a paused session holding the interpreter.
    api("/api/debug/stop", {}).catch(() => {});
    setState("idle");
    renderAll();
  }

  return {
    init, start, control, handleKey, paletteItems, reset, restoreForFile, onDocChange,
    persistNow: () => { settleEdit.cancel(); persistBreakpoints(); },
    isActive, hasSession, toggleLine, editCondition, clearBreakpoints, addWatch,
    onGutterToggle: toggleLine, onGutterCondition: editCondition,
    state: () => D.state,
    isFinal: () => FINAL.has(D.state),
    refreshFlow() {
      if (D.state === "paused" && D.snap) {
        markDebug(deps.flowSvg(), { current: D.snap.location && D.snap.location.line, visited: new Set(D.snap.visited_lines || []) });
      }
    },
    refreshBreakpointsView: renderBreakpointsList,
    announce,
  };
}
