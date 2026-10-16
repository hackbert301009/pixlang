// pixlang/editor/static/js/app.js
// Main entry point: boots the editor, wires the chrome, drives runs.

import {
  EditorView, EditorState, Prec, keymap, lineNumbers, highlightActiveLineGutter,
  highlightSpecialChars, drawSelection, dropCursor, rectangularSelection, crosshairCursor,
  highlightActiveLine, placeholder, history, defaultKeymap, historyKeymap, indentWithTab,
  indentOnInput, bracketMatching, autocompletion, closeBrackets, closeBracketsKeymap,
  completionKeymap, search, searchKeymap, highlightSelectionMatches, openSearchPanel,
  lintKeymap, snippet, Compartment,
} from "../vendor/cm6.js";

import {
  pixlangSupport, pixlangTheme, pixlangCompletion, pixlangHover, pixlangLinter,
  runtimeErrorField, setRuntimeError, flashField, jumpTo, forceLinting,
} from "./lang.js";
import { renderFlowchart, setActiveLine } from "./flowchart.js";
import { createPreview } from "./preview.js";
import { createPalette } from "./palette.js";
import {
  initLayout, showSide, showBottom, toggleBottom, toggleFlowPane, showRightTab, setView,
  rovingTabs, listKeys, closeSidebarIfOverlay,
} from "./layout.js";
import {
  api, store, $, $$, h, icon, debounce, localizeKbd, catVar, fmtMs, toast, isMac,
} from "./util.js";

// ── State ────────────────────────────────────────────────────────────────────
const S = {
  lang: { keywords: [], commands: [], operators: [] },
  info: {},
  files: [],
  examples: [],
  file: { kind: "untitled", path: null, name: "untitled.pxl" },
  savedCode: "",
  dirty: false,
  lint: [],
  runtime: null,          // problem object for the last failed run
  lastRun: null,          // last /api/run response + codeAtRun
  running: false,
  pending: false,
  auto: store.get("autoRun", false),
  tree: [],
  flowScale: store.get("flowScale", 1),
};

let view = null;
let preview = null;
let palette = null;

// ── Editor ───────────────────────────────────────────────────────────────────
function createEditor(doc) {
  const appKeys = Prec.highest(keymap.of([
    { key: "Mod-Enter", run: () => { runPipeline(); return true; } },
    { key: "Mod-s", run: () => { saveFile(); return true; }, preventDefault: true },
    { key: "Mod-k", run: () => { palette.open(); return true; } },
    { key: "Mod-Shift-p", run: () => { palette.open(); return true; } },
  ]));
  view = new EditorView({
    parent: $("#codemirror-host"),
    state: EditorState.create({
      doc,
      extensions: [
        appKeys,
        lineNumbers(),
        highlightActiveLineGutter(),
        highlightSpecialChars(),
        history(),
        drawSelection(),
        dropCursor(),
        EditorState.allowMultipleSelections.of(true),
        indentOnInput(),
        bracketMatching(),
        closeBrackets(),
        autocompletion({ override: [pixlangCompletion(S.lang)], icons: true, activateOnTypingDelay: 60, closeOnBlur: true }),
        rectangularSelection(),
        crosshairCursor(),
        highlightActiveLine(),
        highlightSelectionMatches(),
        search({ top: true }),
        keymap.of([
          ...closeBracketsKeymap, ...defaultKeymap, ...searchKeymap, ...historyKeymap,
          ...completionKeymap, ...lintKeymap, indentWithTab,
        ]),
        pixlangSupport(S.lang),
        pixlangTheme,
        pixlangHover(S.lang),
        pixlangLinter({ getRuntime: () => S.runtime, onResult: onLintResult }),
        runtimeErrorField,
        flashField,
        placeholder("# Start typing a PixLang pipeline — try LOAD, then press Ctrl/⌘+Space"),
        EditorView.contentAttributes.of({ "aria-label": "PixLang source code" }),
        EditorView.updateListener.of(onEditorUpdate),
      ],
    }),
  });
}

function onEditorUpdate(u) {
  if (u.docChanged) {
    setDirty(view.state.doc.toString() !== S.savedCode);
    scheduleParse();
    persistDraft();
    if (S.auto) autoRun();
    else if (S.lastRun && S.lastRun.ok && preview.hasImage()) preview.setStale("outdated", "Code changed since last run");
  }
  if (u.selectionSet || u.docChanged) {
    const pos = u.state.selection.main.head;
    const line = u.state.doc.lineAt(pos);
    $("#sb-cursor").textContent = `Ln ${line.number}, Col ${pos - line.from + 1}`;
    setActiveLine($("#flow-svg"), line.number);
  }
}

function setDoc(code) {
  view.dispatch({
    changes: { from: 0, to: view.state.doc.length, insert: code },
    selection: { anchor: 0 },
    effects: [setRuntimeError.of(null), EditorView.scrollIntoView(0)],
  });
}

// ── Dirty / file identity ────────────────────────────────────────────────────
function setDirty(d) {
  S.dirty = d;
  $("#dirty-dot").hidden = !d;
  $("#sb-autosave").textContent = S.file.kind === "example" ? "Example (read-only)" : (d ? "Unsaved changes" : (S.file.kind === "workspace" ? "Saved" : "Not saved"));
  document.title = (d ? "● " : "") + S.file.name + " — PixLang";
  const cur = $(`#file-list .item[aria-selected="true"] .dirty-dot`);
  if (cur) cur.hidden = !d;
}

function setFile(file, code) {
  S.file = file;
  S.savedCode = file.kind === "untitled" ? "" : code;
  S.runtime = null;
  S.lastRun = null;
  setDoc(code);
  setDirty(file.kind === "untitled" ? code.trim().length > 0 && code !== S.template : false);
  const dir = file.kind === "example" ? "examples /" : (file.path && file.path.includes("/") ? file.path.slice(0, file.path.lastIndexOf("/")) + " /" : "");
  $("#crumb-dir").textContent = dir;
  $("#crumb-file").textContent = file.name;
  $("#crumb").title = file.kind === "workspace" ? `${S.info.workspace}/${file.path}` : file.name;
  store.set("session", { file });
  renderFiles();
  renderExamples();
  preview.setResult(null);
  preview.setStale(null);
  renderTiming();
  renderProblems();
  scheduleParse(0);
}

const persistDraft = debounce(() => {
  const sess = { file: S.file };
  if (S.dirty) sess.draft = view.state.doc.toString();
  store.set("session", sess);
}, 400);

window.addEventListener("beforeunload", (e) => {
  if (S.dirty && S.file.kind !== "untitled") { e.preventDefault(); e.returnValue = ""; }
});

async function confirmDiscard() {
  if (!S.dirty || S.file.kind === "example" && !S.dirty) return true;
  if (S.file.kind === "untitled" && !view.state.doc.toString().trim()) return true;
  const r = await dialog({
    title: "Unsaved changes",
    body: `Save changes to ${S.file.name} before continuing?`,
    actions: [
      { label: "Cancel", value: "cancel", kind: "btn-ghost" },
      { label: "Discard", value: "discard", kind: "btn-secondary" },
      { label: "Save", value: "save", kind: "btn-primary", default: true },
    ],
  });
  if (r.action === "save") return await saveFile();
  return r.action === "discard";
}

// ── Files ────────────────────────────────────────────────────────────────────
async function loadFiles() {
  try {
    const d = await api("/api/files");
    S.files = d.files || [];
    $("#ws-name span").textContent = d.name || d.root;
    $("#ws-name").title = d.root;
  } catch (e) { S.files = []; }
  renderFiles();
}

function renderFiles() {
  const list = $("#file-list");
  if (!S.files.length) {
    list.replaceChildren(h("div", { class: "list-empty" }, "No .pxl files in this workspace yet.", h("br"),
      h("button", { class: "btn btn-secondary btn-sm", style: { marginTop: "12px" }, onclick: newFile }, icon("plus", "icon-sm"), "New file")));
    return;
  }
  const groups = new Map();
  for (const f of S.files) {
    const dir = f.path.includes("/") ? f.path.slice(0, f.path.lastIndexOf("/")) : "";
    if (!groups.has(dir)) groups.set(dir, []);
    groups.get(dir).push(f);
  }
  const out = [];
  const dirs = [...groups.keys()].sort((a, b) => (a === "" ? -1 : b === "" ? 1 : a.localeCompare(b)));
  for (const dir of dirs) {
    if (dir) out.push(h("div", { class: "list-label", role: "presentation" }, icon("folder", "icon-xs"), dir));
    for (const f of groups.get(dir)) {
      const active = S.file.kind === "workspace" && S.file.path === f.path;
      out.push(h("button", {
        class: "item" + (dir ? " indent" : ""), role: "option", "aria-selected": String(active),
        title: f.path, onclick: () => openWorkspaceFile(f.path),
      }, icon("file-code", "icon-sm"),
      h("span", { class: "item-title" }, f.name),
      h("span", { class: "dirty-dot", hidden: !(active && S.dirty), "aria-label": "unsaved" })));
    }
  }
  list.replaceChildren(...out);
}

async function openWorkspaceFile(path, { force = false } = {}) {
  if (!force && !(await confirmDiscard())) return;
  try {
    const d = await api("/api/files/read", { path });
    setFile({ kind: "workspace", path: d.path, name: d.path.split("/").pop() }, d.code);
    closeSidebarIfOverlay();
    if (window.innerWidth < 640) setView("code");
    runPipeline({ quiet: true });
  } catch (e) {
    toast(e.message, "err");
  }
}

async function openExample(name) {
  if (!(await confirmDiscard())) return;
  try {
    const d = await api(`/api/examples/${encodeURIComponent(name)}`);
    setFile({ kind: "example", path: null, example: name, name: name + ".pxl" }, d.code);
    closeSidebarIfOverlay();
    if (window.innerWidth < 640) setView("code");
    runPipeline({ quiet: true });
  } catch (e) { toast(e.message, "err"); }
}

async function newFile() {
  if (!(await confirmDiscard())) return;
  const r = await dialog({
    title: "New pipeline",
    body: "File path inside the workspace.",
    input: { value: suggestName("pipeline"), placeholder: "pipelines/my_pipeline.pxl" },
    validate: validatePath,
    actions: [{ label: "Cancel", value: "cancel", kind: "btn-ghost" }, { label: "Create", value: "ok", kind: "btn-primary", default: true }],
  });
  if (r.action !== "ok") return;
  const path = normPath(r.value);
  try {
    const { code } = await api("/api/template", { path });
    await api("/api/files/save", { path, code, create: true });
    await loadFiles();
    await openWorkspaceFile(path, { force: true });
    toast(`Created ${path}`);
  } catch (e) { toast(e.message, "err"); }
}

function suggestName(base) {
  const names = new Set(S.files.map(f => f.path));
  if (!names.has(base + ".pxl")) return base + ".pxl";
  for (let i = 2; i < 100; i++) if (!names.has(`${base}_${i}.pxl`)) return `${base}_${i}.pxl`;
  return base + "_new.pxl";
}
function normPath(p) {
  p = p.trim().replace(/\\/g, "/").replace(/^\.\//, "");
  if (!p.toLowerCase().endsWith(".pxl")) p += ".pxl";
  return p;
}
function validatePath(v) {
  const p = (v || "").trim();
  if (!p) return "Enter a file name";
  if (p.startsWith("/") || /(^|\/)\.\.(\/|$)/.test(p)) return "Path must stay inside the workspace";
  return null;
}

async function saveFile({ as = false } = {}) {
  const code = view.state.doc.toString();
  let path = S.file.kind === "workspace" && !as ? S.file.path : null;
  let create = false;
  if (!path) {
    const r = await dialog({
      title: S.file.kind === "example" ? "Save example as" : "Save pipeline",
      body: "Choose a path inside the workspace.",
      input: { value: suggestName(S.file.kind === "example" ? S.file.example : "pipeline"), placeholder: "my_pipeline.pxl" },
      validate: validatePath,
      actions: [{ label: "Cancel", value: "cancel", kind: "btn-ghost" }, { label: "Save", value: "ok", kind: "btn-primary", default: true }],
    });
    if (r.action !== "ok") return false;
    path = normPath(r.value);
    create = true;
  }
  try {
    const d = await api("/api/files/save", { path, code, create });
    const wasOther = S.file.kind !== "workspace" || S.file.path !== d.path;
    S.file = { kind: "workspace", path: d.path, name: d.path.split("/").pop() };
    S.savedCode = code;
    if (wasOther) {
      $("#crumb-file").textContent = S.file.name;
      $("#crumb-dir").textContent = d.path.includes("/") ? d.path.slice(0, d.path.lastIndexOf("/")) + " /" : "";
      await loadFiles();
    }
    setDirty(false);
    store.set("session", { file: S.file });
    toast(`Saved ${d.path}`);
    return true;
  } catch (e) {
    if (e.status === 409) toast("A file with that name already exists", "err");
    else toast(e.message, "err");
    return false;
  }
}

// ── Commands & examples sidebars ─────────────────────────────────────────────
function renderCommands() {
  const q = $("#cmd-search").value.trim().toLowerCase();
  const groups = new Map();
  for (const c of S.lang.commands) {
    if (q && !(c.name.toLowerCase().includes(q) || (c.summary || "").toLowerCase().includes(q) || c.category.toLowerCase().includes(q))) continue;
    if (!groups.has(c.category)) groups.set(c.category, []);
    groups.get(c.category).push(c);
  }
  const out = [];
  for (const cat of [...groups.keys()].sort()) {
    out.push(h("div", { class: "list-label", role: "presentation", style: { "--rail": catVar(cat) } }, h("span", { class: "rail-dot" }), cat));
    for (const c of groups.get(cat)) {
      out.push(h("button", {
        class: "item item-cmd", role: "option", "aria-selected": "false",
        style: { "--rail": catVar(cat) }, title: c.usage,
        onclick: () => insertCommand(c),
      }, h("span", { class: "rail" }),
      h("span", { class: "item-main" },
        h("span", { class: "item-title" }, c.name),
        h("span", { class: "item-sub" }, c.summary || c.usage))));
    }
  }
  if (!out.length) out.push(h("div", { class: "list-empty" }, "No commands match."));
  $("#cmd-list").replaceChildren(...out);
  $("#cmd-count").textContent = String(S.lang.commands.length);
}

function insertCommand(c) {
  const req = (c.params || []).filter(p => p.name && !p.optional && !p.name.endsWith("…"));
  const tpl = c.name + req.map(p => /str/i.test(p.type || "") ? ` "\${${p.name}}"` : ` \${${p.name}}`).join("");
  const st = view.state;
  const line = st.doc.lineAt(st.selection.main.head);
  const indent = line.text.match(/^\s*/)[0];
  let from, to, text;
  if (!line.text.trim()) { from = line.from; to = line.to; text = indent + tpl; }
  else { from = to = line.to; text = "\n" + indent + tpl; }
  snippet(text)({ state: view.state, dispatch: (tr) => view.dispatch(tr) }, null, from, to);
  view.focus();
  closeSidebarIfOverlay();
  if (window.innerWidth < 640) setView("code");
}

async function loadExamples() {
  try { S.examples = (await api("/api/examples")).examples || []; } catch { S.examples = []; }
  renderExamples();
}
function renderExamples() {
  const out = S.examples.map(ex => {
    const active = S.file.kind === "example" && S.file.example === ex.name;
    return h("button", {
      class: "item", role: "option", "aria-selected": String(active),
      onclick: () => openExample(ex.name), title: ex.description,
    }, icon("book", "icon-sm"),
    h("span", { class: "item-main" },
      h("span", { class: "item-title" }, ex.name.replace(/_/g, " ")),
      ex.description ? h("span", { class: "item-sub" }, ex.description) : null));
  });
  $("#example-list").replaceChildren(...(out.length ? out : [h("div", { class: "list-empty" }, "No examples found.")]));
}

// ── Problems / output / timing ───────────────────────────────────────────────
function onLintResult(problems) {
  S.lint = problems.filter(p => p.source === "lint");
  renderProblems();
}

function allProblems() {
  const out = [];
  if (S.runtime) out.push({ ...S.runtime, source: "run" });
  out.push(...S.lint);
  return out;
}

function renderProblems() {
  const probs = allProblems();
  const errs = probs.filter(p => p.severity === "error").length;
  const warns = probs.filter(p => p.severity === "warning").length;
  $("#count-err").textContent = errs;
  $("#count-warn").textContent = warns;
  $(".diag-err").classList.toggle("has", errs > 0);
  $(".diag-warn").classList.toggle("has", warns > 0);
  $("#btn-diag").classList.toggle("ok", errs === 0);
  $(".diag-err use").setAttribute("href", errs ? "#i-x-circle" : "#i-check-circle");
  $("#btn-diag").setAttribute("aria-label", `${errs} errors, ${warns} warnings`);
  const badge = $("#badge-problems");
  badge.textContent = probs.length;
  badge.className = "badge" + (errs ? " err" : warns ? " warn" : "");
  $("#badge-mobile").hidden = errs === 0;

  const host = $("#bp-problems");
  if (!probs.length) {
    host.replaceChildren(h("div", { class: "bp-empty" }, icon("check-circle", "icon-sm"), "No problems detected"));
    return;
  }
  host.replaceChildren(...probs.map(p => {
    const sev = p.severity || "error";
    const ic = sev === "error" ? "x-circle" : sev === "warning" ? "alert" : "info";
    const loc = p.line ? `Ln ${p.line}${p.col ? ", Col " + p.col : ""}` : "";
    return h("button", {
      class: `problem ${sev}` + (p.source === "run" ? " runtime" : ""),
      onclick: () => { if (p.line) { if (window.innerWidth < 640) setView("code"); jumpTo(view, p.line, p.col); } },
      "aria-label": `${sev}: ${p.message} ${loc}`,
    }, icon(ic, "icon-sm"),
    h("span", { class: "problem-msg" }, p.message, h("span", { class: "problem-code" }, p.source === "run" ? (p.kindLabel || "runtime") : p.code)),
    h("span", { class: "problem-loc" }, loc));
  }));
}

function appendOutput(nodes) {
  const pre = $("#output-pre");
  pre.append(...nodes);
  const view = $("#bp-output");
  view.scrollTop = view.scrollHeight;
}

function renderTiming() {
  const host = $("#bp-timing");
  const r = S.lastRun;
  const lt = r && r.stats ? r.stats.line_times || {} : {};
  const rows = Object.entries(lt).map(([line, v]) => ({ line: +line, ...v })).sort((a, b) => a.line - b.line);
  if (!rows.length) {
    host.replaceChildren(h("div", { class: "bp-empty" }, icon("timer", "icon-sm"), r ? "No commands were executed." : "Run the pipeline to see per-command timing."));
    return;
  }
  const max = Math.max(...rows.map(r => r.ms), 0.001);
  const byName = new Map(S.lang.commands.map(c => [c.name, c]));
  host.replaceChildren(h("div", { class: "timing" },
    h("div", { class: "timing-head" },
      h("span", null, `${r.stats.commands_executed} commands executed`),
      h("span", null, "Total ", h("strong", null, fmtMs(r.stats.total_ms)))),
    rows.map(row => h("button", {
      class: "trow", title: `Line ${row.line}: ${row.name}`,
      style: { "--rail": catVar(byName.get(row.name)?.category || "Other") },
      onclick: () => jumpTo(view, row.line),
    }, h("span", { class: "tl" }, "L" + row.line),
    h("span", { class: "tn" }, row.name, row.count > 1 ? h("small", null, "×" + row.count) : null),
    h("span", { class: "tb" }, h("span", { style: { width: Math.max(1, (row.ms / max) * 100) + "%" } })),
    h("span", { class: "tv" }, fmtMs(row.ms))))));
}

// ── Flow ─────────────────────────────────────────────────────────────────────
const scheduleParse = (() => {
  let t = null, seq = 0;
  return (delay = 300) => {
    clearTimeout(t);
    t = setTimeout(async () => {
      const my = ++seq;
      const code = view.state.doc.toString();
      let d;
      try { d = await api("/api/parse", { code }); } catch { return; }
      if (my !== seq) return;
      if (d.error) {
        $("#flow-sub").textContent = "Parse error";
        if (!S.tree.length) showFlowEmpty(d.error_info ? `Line ${d.error_info.line}: ${d.error_info.message}` : d.error, true);
        return;
      }
      S.tree = d.tree || [];
      drawFlow();
    }, delay);
  };
})();

function showFlowEmpty(text, isErr = false) {
  const e = $("#flow-empty");
  e.hidden = false;
  e.textContent = text;
  e.classList.toggle("error", isErr);
  $("#flow-svg").replaceChildren();
  $("#flow-svg").setAttribute("width", 0);
  $("#flow-svg").setAttribute("height", 0);
}

function drawFlow() {
  if (!S.tree.length) { showFlowEmpty("Nothing to show yet — add some commands."); $("#flow-sub").textContent = ""; return; }
  $("#flow-empty").hidden = true;
  const r = S.lastRun;
  const sameCode = r && r.codeAtRun === view.state.doc.toString();
  const run = sameCode ? {
    done: true,
    line_times: r.stats ? r.stats.line_times : {},
    visited: new Set(r.visited_lines || []),
    errorLine: r.error_info && r.error_info.kind !== "syntax" ? r.error_info.line : null,
  } : null;
  const res = renderFlowchart($("#flow-svg"), S.tree, {
    run, scale: S.flowScale,
    onJump: (line) => { if (window.innerWidth < 640) setView("code"); jumpTo(view, line); },
  });
  $("#flow-sub").textContent = `${res.count} steps` + (run ? " · last run" : "");
  setActiveLine($("#flow-svg"), view.state.doc.lineAt(view.state.selection.main.head).number);
}

// ── Run ──────────────────────────────────────────────────────────────────────
const autoRun = debounce(() => runPipeline({ quiet: true, auto: true }), 700);

async function runPipeline({ quiet = false, auto = false } = {}) {
  if (S.running) { S.pending = true; return; }
  autoRun.cancel();
  S.running = true;
  const btn = $("#btn-run");
  btn.classList.add("running");
  btn.setAttribute("aria-busy", "true");
  preview.setBusy(true);
  $("#sb-run").className = "sb-item sb-run";
  $("#sb-run").replaceChildren(icon("loader", "icon-xs spin"), "Running…");

  const code = view.state.doc.toString();
  const body = { code };
  if (S.file.kind === "workspace") body.path = S.file.path;
  if (S.file.kind === "example") body.example = S.file.example;
  const started = new Date();
  let r;
  try {
    r = await api("/api/run", body);
  } catch (e) {
    r = { ok: false, error: e.message, error_info: { kind: "network", message: e.message }, stats: {} };
  }
  r.codeAtRun = code;
  S.lastRun = r;
  S.running = false;
  btn.classList.remove("running");
  btn.removeAttribute("aria-busy");
  preview.setBusy(false);

  const ei = r.error_info;
  const kindLabel = ei ? ({ syntax: "syntax", runtime: "runtime", timeout: "timeout", assert: "assert", busy: "busy", network: "network" }[ei.kind] || ei.kind) : null;
  S.runtime = ei ? {
    severity: "error", line: ei.line || null, col: ei.col || null, message: ei.message,
    code: ei.kind, kindLabel,
  } : null;
  view.dispatch({ effects: setRuntimeError.of(ei && ei.kind !== "syntax" ? ei.line : null) });
  forceLinting(view);
  renderProblems();

  // Preview
  preview.setResult(r);
  if (!ei) preview.setStale(null);
  else if (r.image) preview.setStale("error", ei.line ? `Failed at line ${ei.line} — showing partial result` : "Run failed — showing partial result");
  else preview.setStale("error", ei.kind === "syntax" ? `Syntax error on line ${ei.line ?? "?"}` : "Run failed — no image produced");

  // Output
  const ts = started.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const out = [h("span", { class: "o-dim" }, `── ${ts}  ${S.file.name} ${"─".repeat(24)}\n`)];
  if (r.stdout) out.push(r.stdout.endsWith("\n") ? r.stdout : r.stdout + "\n");
  for (const pe of r.plugin_errors || []) out.push(h("span", { class: "o-err" }, `plugin: ${pe}\n`));
  if (ei) out.push(h("span", { class: "o-err" }, `✗ ${ei.line ? `Line ${ei.line}${ei.col ? ":" + ei.col : ""} ` : ""}${ei.raw || ei.message}\n`));
  else out.push(h("span", { class: "o-ok" }, `✓ Finished in ${fmtMs(r.stats.total_ms)} · ${r.stats.commands_executed} commands\n`));
  appendOutput(out);
  renderTiming();
  drawFlow();

  // Status bar
  const sb = $("#sb-run");
  if (ei) { sb.className = "sb-item sb-run err"; sb.replaceChildren(icon("x-circle", "icon-xs"), `Last run failed${ei.line ? " · line " + ei.line : ""}`); }
  else { sb.className = "sb-item sb-run ok"; sb.replaceChildren(icon("check-circle", "icon-xs"), `Last run ✓ ${fmtMs(r.stats.total_ms)}`); }

  if (ei && !quiet && !auto) {
    if (window.innerWidth >= 640) showBottom("problems");
  }
  if (S.pending) { S.pending = false; runPipeline({ quiet: true, auto: true }); }
}

// ── Dialog ───────────────────────────────────────────────────────────────────
function dialog({ title, body, input, actions, validate }) {
  return new Promise((resolve) => {
    const ov = $("#dialog");
    const inp = $("#dialog-input");
    const err = $("#dialog-error");
    const prev = document.activeElement;
    $("#dialog-title").textContent = title;
    $("#dialog-body").textContent = body || "";
    err.hidden = true;
    inp.hidden = !input;
    if (input) { inp.value = input.value || ""; inp.placeholder = input.placeholder || ""; }
    const done = (action) => {
      if (action !== "cancel" && input && validate) {
        const msg = validate(inp.value);
        if (msg) { err.textContent = msg; err.hidden = false; inp.focus(); return; }
      }
      ov.hidden = true;
      ov.onkeydown = null;
      if (prev && document.contains(prev)) prev.focus();
      resolve({ action, value: inp.value });
    };
    const btns = actions.map(a => h("button", { class: "btn " + (a.kind || "btn-secondary"), onclick: () => done(a.value) }, a.label));
    $("#dialog-actions").replaceChildren(...btns);
    ov.hidden = false;
    ov.onkeydown = (e) => {
      if (e.key === "Escape") { e.preventDefault(); done("cancel"); }
      else if (e.key === "Enter" && (e.target === inp || !input)) {
        e.preventDefault();
        done((actions.find(a => a.default) || actions[actions.length - 1]).value);
      } else if (e.key === "Tab") {
        const f = [...(input ? [inp] : []), ...btns];
        const i = f.indexOf(document.activeElement);
        e.preventDefault();
        f[(i + (e.shiftKey ? -1 : 1) + f.length) % f.length].focus();
      }
    };
    ov.onmousedown = (e) => { if (e.target === ov) done("cancel"); };
    requestAnimationFrame(() => {
      if (input) {
        inp.focus();
        const dot = inp.value.lastIndexOf(".");
        inp.setSelectionRange(0, dot > 0 ? dot : inp.value.length);
      } else btns[btns.length - 1].focus();
    });
  });
}

// ── Menu ─────────────────────────────────────────────────────────────────────
function menuItems() {
  return [
    { label: "New file…", icon: "plus", run: newFile },
    { label: "Open file…", icon: "file-code", keys: "mod+k", run: () => palette.open("") },
    { label: "Save", icon: "save", keys: "mod+s", run: () => saveFile() },
    { label: "Save as…", icon: "save", run: () => saveFile({ as: true }) },
    "-",
    { label: "Find", icon: "search", keys: "mod+f", run: () => { view.focus(); openSearchPanel(view); } },
    { label: "Toggle theme", icon: "theme", run: toggleTheme },
    { label: "Keyboard shortcuts", icon: "command", run: showShortcuts },
  ];
}
function openMenu() {
  const m = $("#app-menu");
  const btn = $("#btn-menu");
  if (!m.hidden) { closeMenu(); return; }
  const items = menuItems().map(it => it === "-" ? h("div", { class: "menu-sep", role: "separator" }) :
    h("button", { class: "menu-item", role: "menuitem", tabindex: -1, onclick: () => { closeMenu(); it.run(); } },
      icon(it.icon, "icon-sm"), it.label, it.keys ? h("kbd", { "data-kbd": it.keys }) : null));
  m.replaceChildren(...items);
  localizeKbd(m);
  const r = btn.getBoundingClientRect();
  m.style.left = r.left + "px";
  m.style.top = r.bottom + 6 + "px";
  m.hidden = false;
  btn.setAttribute("aria-expanded", "true");
  const focusables = $$(".menu-item", m);
  focusables[0].focus();
  m.onkeydown = (e) => {
    const i = focusables.indexOf(document.activeElement);
    if (e.key === "ArrowDown") { focusables[(i + 1) % focusables.length].focus(); e.preventDefault(); }
    else if (e.key === "ArrowUp") { focusables[(i - 1 + focusables.length) % focusables.length].focus(); e.preventDefault(); }
    else if (e.key === "Escape") { closeMenu(); btn.focus(); }
  };
  setTimeout(() => document.addEventListener("mousedown", outside), 0);
}
function outside(e) { if (!e.target.closest("#app-menu") && !e.target.closest("#btn-menu")) closeMenu(); }
function closeMenu() {
  $("#app-menu").hidden = true;
  $("#btn-menu").setAttribute("aria-expanded", "false");
  document.removeEventListener("mousedown", outside);
}

function showShortcuts() {
  const rows = [
    ["Run pipeline", "mod+enter"], ["Save", "mod+s"], ["Command palette", "mod+k"],
    ["Autocomplete", "ctrl+space"], ["Find / replace", "mod+f"], ["Toggle comment", "mod+/"],
    ["Zoom preview", "+ / − / 0 / 1"],
  ];
  const fmt = (k) => k.includes("/") && !k.startsWith("mod") ? k : k.split("+").map(p => ({ mod: isMac ? "⌘" : "Ctrl", enter: "↵", ctrl: "Ctrl", space: "Space", shift: "⇧" }[p] || p.toUpperCase())).join(isMac ? "" : "+");
  dialog({
    title: "Keyboard shortcuts",
    body: rows.map(([a, k]) => `${a}: ${fmt(k)}`).join("   ·   "),
    actions: [{ label: "Close", value: "cancel", kind: "btn-primary", default: true }],
  });
}

// ── Theme ────────────────────────────────────────────────────────────────────
function toggleTheme() {
  const cur = document.documentElement.getAttribute("data-theme");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("pixlang.theme", next); } catch { /* blocked */ }
  syncThemeIcon();
}
function syncThemeIcon() {
  const dark = document.documentElement.getAttribute("data-theme") === "dark";
  $("#btn-theme").setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  $("#btn-theme use").setAttribute("href", dark ? "#i-sun" : "#i-moon");
}

// ── Palette items ────────────────────────────────────────────────────────────
function paletteItems() {
  const items = [
    { group: "Actions", title: "Run pipeline", icon: "play", keys: "mod+enter", run: () => runPipeline(), weight: 20 },
    { group: "Actions", title: "Save", icon: "save", keys: "mod+s", run: () => saveFile(), weight: 15 },
    { group: "Actions", title: "Save as…", icon: "save", run: () => saveFile({ as: true }) },
    { group: "Actions", title: "New file…", icon: "plus", run: newFile, weight: 10 },
    { group: "Actions", title: S.auto ? "Turn auto-run off" : "Turn auto-run on", icon: "refresh", run: () => setAuto(!S.auto) },
    { group: "Actions", title: "Toggle theme", icon: "theme", run: toggleTheme },
    { group: "Actions", title: "Find in file", icon: "search", keys: "mod+f", run: () => { view.focus(); openSearchPanel(view); } },
    { group: "Actions", title: "Show problems", icon: "list", run: () => { window.innerWidth < 640 ? setView("problems") : showBottom("problems"); } },
    { group: "Actions", title: "Show output", icon: "terminal", run: () => showBottom("output") },
    { group: "Actions", title: "Show timing", icon: "timer", run: () => showBottom("timing") },
    { group: "Actions", title: "Toggle flow pane", icon: "flow", run: () => toggleFlowPane() },
    { group: "Actions", title: "Go to line…", icon: "hash", run: gotoLine },
    { group: "Actions", title: "Keyboard shortcuts", icon: "command", run: showShortcuts },
  ];
  for (const f of S.files) items.push({ group: "Files", title: f.path, icon: "file-code", run: () => openWorkspaceFile(f.path), weight: 5 });
  for (const ex of S.examples) items.push({ group: "Examples", title: ex.name, sub: ex.description, icon: "book", run: () => openExample(ex.name), pinned: false });
  for (const c of S.lang.commands) items.push({ group: "Insert command", title: c.name, sub: c.summary || c.usage, rail: c.category, run: () => insertCommand(c), pinned: false });
  return items;
}

async function gotoLine() {
  const r = await dialog({
    title: "Go to line", body: `1 – ${view.state.doc.lines}`,
    input: { value: "", placeholder: "Line number" },
    validate: (v) => (/^\d+(:\d+)?$/.test(v.trim()) ? null : "Enter a line number, e.g. 12 or 12:4"),
    actions: [{ label: "Cancel", value: "cancel", kind: "btn-ghost" }, { label: "Go", value: "ok", kind: "btn-primary", default: true }],
  });
  if (r.action !== "ok") return;
  const [l, c] = r.value.trim().split(":").map(Number);
  jumpTo(view, l, c);
}

function setAuto(on) {
  S.auto = on;
  store.set("autoRun", on);
  $("#btn-auto").setAttribute("aria-checked", String(on));
  if (on) autoRun();
}

// ── Wiring ───────────────────────────────────────────────────────────────────
function wire() {
  $("#btn-run").addEventListener("click", () => runPipeline());
  $("#btn-auto").addEventListener("click", () => setAuto(!S.auto));
  $("#btn-auto").setAttribute("aria-checked", String(S.auto));
  $("#btn-save").addEventListener("click", () => saveFile());
  $("#btn-theme").addEventListener("click", toggleTheme);
  $("#btn-palette").addEventListener("click", () => palette.open());
  $("#btn-menu").addEventListener("click", openMenu);
  $("#btn-diag").addEventListener("click", () => (window.innerWidth < 640 ? setView("problems") : showBottom("problems")));
  $("#btn-new-file").addEventListener("click", newFile);
  $("#btn-refresh-files").addEventListener("click", loadFiles);
  $("#act-shortcuts").addEventListener("click", showShortcuts);
  $("#cmd-search").addEventListener("input", renderCommands);
  $("#cmd-search").addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { $("#cmd-list .item")?.focus(); e.preventDefault(); }
  });
  $("#btn-clear-output").addEventListener("click", () => $("#output-pre").replaceChildren());
  $("#btn-toggle-bottom").addEventListener("click", () => toggleBottom());

  for (const b of $$(".act[data-side]")) {
    b.addEventListener("click", () => (b.dataset.side === "flow" ? toggleFlowPane() : showSide(b.dataset.side)));
  }
  for (const t of $$(".bottom-panel .tab")) t.addEventListener("click", () => showBottom(t.dataset.bottom));
  for (const t of $$(".right-tabs .tab")) t.addEventListener("click", () => { showRightTab(t.dataset.right); if (t.dataset.right === "flow") drawFlow(); });
  for (const t of $$(".mobile-tabs [data-view]")) t.addEventListener("click", () => setView(t.dataset.view));
  rovingTabs($(".bottom-panel .tabs"), ".tab");
  rovingTabs($(".right-tabs"), ".tab");
  rovingTabs($(".activitybar"), ".act");
  rovingTabs($(".mobile-tabs"), "[data-view]");
  listKeys($("#file-list"));
  listKeys($("#cmd-list"));
  listKeys($("#example-list"));

  $("#flow-zoom-in").addEventListener("click", () => { S.flowScale = Math.min(2, S.flowScale * 1.15); store.set("flowScale", S.flowScale); drawFlow(); });
  $("#flow-zoom-out").addEventListener("click", () => { S.flowScale = Math.max(0.5, S.flowScale / 1.15); store.set("flowScale", S.flowScale); drawFlow(); });

  document.addEventListener("keydown", (e) => {
    const mod = isMac ? e.metaKey : e.ctrlKey;
    if (!mod) return;
    const k = e.key.toLowerCase();
    if (k === "s") { e.preventDefault(); saveFile(); }
    else if (k === "enter") { e.preventDefault(); runPipeline(); }
    else if (k === "k" || (k === "p" && e.shiftKey)) { e.preventDefault(); palette.isOpen() ? palette.close() : palette.open(); }
  });
  // Close the overlay sidebar when clicking into the editor on medium screens.
  $(".center").addEventListener("mousedown", closeSidebarIfOverlay);

  try {
    window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", (ev) => {
      let stored = null;
      try { stored = localStorage.getItem("pixlang.theme"); } catch { /* blocked */ }
      if (!stored) { document.documentElement.setAttribute("data-theme", ev.matches ? "light" : "dark"); syncThemeIcon(); }
    });
  } catch { /* old browser */ }
  window.addEventListener("pixlang:view", () => { if (S.tree.length) drawFlow(); preview.fit(); });
}

// ── Boot ─────────────────────────────────────────────────────────────────────
async function boot() {
  localizeKbd();
  syncThemeIcon();
  preview = createPreview();
  palette = createPalette(paletteItems);
  initLayout({ onResize: () => {} });

  const [lang, info] = await Promise.all([
    api("/api/language").catch(() => null),
    api("/api/info").catch(() => ({})),
  ]);
  if (lang) S.lang = lang;
  S.info = info || {};
  $("#sb-commands").textContent = `${S.lang.commands.length} commands`;

  // Restore session (file + unsaved draft)
  const sess = store.get("session", null);
  let file = { kind: "untitled", path: null, name: "untitled.pxl" };
  let code = null;
  S.template = (await api("/api/template", {}).catch(() => ({ code: "" }))).code;
  if (sess && sess.file && sess.file.kind === "workspace") {
    try {
      const d = await api("/api/files/read", { path: sess.file.path });
      file = { kind: "workspace", path: d.path, name: d.path.split("/").pop() };
      code = d.code;
    } catch { /* gone */ }
  } else if (sess && sess.file && sess.file.kind === "example") {
    try {
      const d = await api(`/api/examples/${encodeURIComponent(sess.file.example)}`);
      file = sess.file; code = d.code;
    } catch { /* gone */ }
  }
  if (code === null) code = S.template;

  createEditor("");
  wire();
  showSide(store.get("ui", {})?.side || "files", { toggle: false });
  if (window.innerWidth < 1024) { $("#app").dataset.sidebar = "closed"; showSide.call(null, "files", { toggle: false }); $("#app").dataset.sidebar = "closed"; }
  showBottom(store.get("bottomTab", "problems"));
  if ((store.get("ui", {}) || {}).bottom === "collapsed") toggleBottom(true);
  setFile(file, code);
  if (sess && sess.draft && sess.file && JSON.stringify(sess.file) === JSON.stringify(file) && sess.draft !== code) {
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: sess.draft } });
  }
  renderCommands();
  loadFiles();
  loadExamples();
  renderProblems();
  renderTiming();
  view.focus();
  runPipeline({ quiet: true });
}

boot();
