// pixlang/editor/static/js/app.js
// Main entry point — initialises CodeMirror, wires toolbar, drives all panels.

import { EditorView, keymap, lineNumbers, highlightActiveLine,
         drawSelection, ViewPlugin, Decoration, WidgetType }
  from "@codemirror/view";
import { EditorState, StateField, StateEffect, RangeSetBuilder }
  from "@codemirror/state";
import { defaultKeymap, indentWithTab, history, historyKeymap }
  from "@codemirror/commands";
import { StreamLanguage, syntaxHighlighting, defaultHighlightStyle }
  from "@codemirror/language";
import { oneDark } from "@codemirror/theme-one-dark";

import { renderFlowchart }   from "./flowchart.js";
import { makeDraggable, initCollapsibleConsole } from "./panels.js";
import { saveWorkspace, loadWorkspace, exportWorkspaceFile } from "./workspace.js";

// ── PixLang syntax definition ────────────────────────────────────────────────

const KEYWORDS = new Set([
  "SET","IF","ENDIF","REPEAT","END","INCLUDE","ASSERT",
  "ROI","ROI_RESET","LOAD_GLOB","SAVE_EACH"
]);

const pixlangLanguage = StreamLanguage.define({
  token(stream) {
    if (stream.match(/^#[^\n]*/))                         return "comment";
    if (stream.match(/^"[^"]*"/))                         return "string";
    if (stream.match(/^\$[A-Za-z_]\w*/))                  return "variableName";
    if (stream.match(/^(==|!=|<=|>=|<|>)/))               return "operator";
    if (stream.match(/^-?\d+\.\d+/))                      return "number";
    if (stream.match(/^-?\d+/))                           return "number";
    if (stream.match(/^\\\s*$/))                          return "operator"; // continuation
    // WORD — classify as keyword, command, or ident
    const word = stream.match(/^[A-Za-z_]\w*/);
    if (word) {
      const w = word[0];
      if (KEYWORDS.has(w))                                return "keyword";
      if (/^[A-Z][A-Z0-9_]*[A-Z0-9]$/.test(w) || /^[A-Z]{2,}$/.test(w))
                                                          return "function";
      return "variableName2";
    }
    stream.next();
    return null;
  },
  languageData: { commentTokens: { line: "#" } }
});

// ── Lint decoration ──────────────────────────────────────────────────────────

const addLintMarks  = StateEffect.define();
const clearLintMarks = StateEffect.define();

const lintField = StateField.define({
  create: () => Decoration.none,
  update(decos, tr) {
    decos = decos.map(tr.changes);
    for (const e of tr.effects) {
      if (e.is(clearLintMarks)) decos = Decoration.none;
      if (e.is(addLintMarks))   decos = e.value;
    }
    return decos;
  },
  provide: f => EditorView.decorations.from(f),
});

function buildLintDecos(issues, doc) {
  const builder = new RangeSetBuilder();
  const sorted  = [...issues].filter(i => i.line > 0).sort((a, b) => a.line - b.line);
  for (const issue of sorted) {
    if (issue.line > doc.lines) continue;
    const line  = doc.line(issue.line);
    const cls   = `cm-lint-${issue.severity}`;
    builder.add(line.from, line.to, Decoration.mark({ class: cls }));
  }
  return builder.finish();
}

// ── Global state ─────────────────────────────────────────────────────────────

let editorView    = null;
let currentFile   = "untitled.pxl";
let lintTimer     = null;
let parseTimer    = null;
let previewZoom   = 1.0;
let allCommands   = [];

// ── CodeMirror initialisation ────────────────────────────────────────────────

function initCodeMirror(initialCode) {
  const host = document.getElementById("codemirror-host");

  const updateListener = EditorView.updateListener.of(view => {
    if (!view.docChanged) return;
    scheduleLint();
    scheduleParse();
  });

  editorView = new EditorView({
    state: EditorState.create({
      doc: initialCode,
      extensions: [
        history(),
        lineNumbers(),
        highlightActiveLine(),
        drawSelection(),
        keymap.of([
          ...defaultKeymap,
          ...historyKeymap,
          indentWithTab,
          { key: "F5",       run: () => { runPipeline(); return true; } },
          { key: "Ctrl-s",   run: () => { saveWorkspaceNow(); return true; } },
          { key: "Ctrl-l",   run: () => { lintPipeline();   return true; } },
        ]),
        pixlangLanguage,
        syntaxHighlighting(defaultHighlightStyle),
        oneDark,
        lintField,
        updateListener,
        EditorView.theme({
          "&": { height: "100%" },
          ".cm-scroller": { overflow: "auto" },
        }),
      ],
    }),
    parent: host,
  });
}

// ── Debounced helpers ─────────────────────────────────────────────────────────

function scheduleLint() {
  clearTimeout(lintTimer);
  lintTimer = setTimeout(lintPipeline, 800);
}

function scheduleParse() {
  clearTimeout(parseTimer);
  parseTimer = setTimeout(refreshFlowchart, 600);
}

// ── Run ───────────────────────────────────────────────────────────────────────

async function runPipeline() {
  const code = editorView.state.doc.toString();
  setStatus("Running…", "");

  let data;
  try {
    const resp = await fetch("/api/run", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ code, filename: currentFile }),
    });
    data = await resp.json();
  } catch (e) {
    setStatus("Network error", "error");
    appendConsole(`Network error: ${e.message}`, "error");
    return;
  }

  // Console output
  if (data.stdout) appendConsole(data.stdout.trim(), "stdout");
  if (data.error)  appendConsole(`✗ ${data.error}`, "error");

  // Stats line
  if (data.stats && data.stats.commands_executed > 0) {
    const ms   = (data.stats.total_ms || 0).toFixed(1);
    const cmds = data.stats.commands_executed;
    appendConsole(`${cmds} commands · ${ms} ms`, "stat");
  }

  // Image preview
  if (data.image_b64) {
    showPreview(data.image_b64);
  }

  if (data.error) {
    setStatus(`✗ ${data.error.split("\n")[0]}`, "error");
  } else {
    const ms = data.stats ? (data.stats.total_ms || 0).toFixed(0) : "?";
    setStatus(`✓ Done  ${ms} ms`, "ok");
  }
}

// ── Lint ──────────────────────────────────────────────────────────────────────

async function lintPipeline() {
  const code = editorView.state.doc.toString();
  let data;
  try {
    const resp = await fetch("/api/lint", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ code }),
    });
    data = await resp.json();
  } catch { return; }

  const issues = data.issues || [];

  // Apply decorations
  const decos = buildLintDecos(issues, editorView.state.doc);
  editorView.dispatch({
    effects: [clearLintMarks.of(null), addLintMarks.of(decos)],
  });

  // Show summary in status bar if issues exist
  const errors   = issues.filter(i => i.severity === "error").length;
  const warnings = issues.filter(i => i.severity === "warning").length;
  if (errors || warnings) {
    setStatus(
      `${errors ? `✗ ${errors} error${errors > 1 ? "s" : ""}` : ""}` +
      `${errors && warnings ? "  " : ""}` +
      `${warnings ? `⚠ ${warnings} warning${warnings > 1 ? "s" : ""}` : ""}`,
      errors ? "error" : ""
    );
  }
}

// ── Flow diagram ──────────────────────────────────────────────────────────────

async function refreshFlowchart() {
  const code = editorView.state.doc.toString();
  let data;
  try {
    const resp = await fetch("/api/parse", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ code }),
    });
    data = await resp.json();
  } catch { return; }

  renderFlowchart(data, document.getElementById("flow-svg"), jumpToLine);
}

// ── Image preview ─────────────────────────────────────────────────────────────

function showPreview(b64) {
  const panel = document.getElementById("preview-panel");
  const img   = document.getElementById("preview-img");
  panel.style.display = "flex";
  img.src = `data:image/png;base64,${b64}`;
  img.onload = () => {
    document.getElementById("preview-info").textContent =
      `${img.naturalWidth} × ${img.naturalHeight} px`;
  };
  previewZoom = 1.0;
  img.style.transform = "scale(1)";
}

function updateZoom(delta) {
  previewZoom = Math.max(0.1, Math.min(8, previewZoom + delta));
  document.getElementById("preview-img").style.transform =
    `scale(${previewZoom})`;
}

// ── Jump to line ─────────────────────────────────────────────────────────────

function jumpToLine(lineNumber) {
  if (!lineNumber || lineNumber < 1) return;
  const doc = editorView.state.doc;
  if (lineNumber > doc.lines) return;
  const line = doc.line(lineNumber);
  editorView.dispatch({
    selection: { anchor: line.from },
    effects:   EditorView.scrollIntoView(line.from, { y: "center" }),
  });
  editorView.focus();
}

// ── Console ───────────────────────────────────────────────────────────────────

function appendConsole(text, cls = "stdout") {
  const out = document.getElementById("console-output");
  // Expand console if collapsed
  const panel = document.getElementById("panel-console");
  if (panel.classList.contains("collapsed")) panel.classList.remove("collapsed");

  const lines = String(text).split("\n");
  for (const line of lines) {
    const el  = document.createElement("span");
    el.className = `console-line ${cls}`;
    el.textContent = line;
    out.appendChild(el);
    out.appendChild(document.createTextNode("\n"));
  }
  out.scrollTop = out.scrollHeight;
}

function clearConsole() {
  document.getElementById("console-output").innerHTML = "";
}

// ── Status bar ────────────────────────────────────────────────────────────────

function setStatus(msg, cls = "") {
  const el = document.getElementById("toolbar-status");
  el.textContent  = msg;
  el.className    = cls;
}

// ── Commands docs panel ───────────────────────────────────────────────────────

async function loadCommands() {
  try {
    const resp = await fetch("/api/commands");
    const data = await resp.json();
    allCommands = data.commands || [];
  } catch { return; }
  renderCommands(allCommands);
}

function renderCommands(cmds) {
  const list = document.getElementById("cmd-list");
  list.innerHTML = "";

  // Group by category
  const groups = {};
  for (const cmd of cmds) {
    (groups[cmd.category] = groups[cmd.category] || []).push(cmd);
  }

  for (const [cat, items] of Object.entries(groups)) {
    const hdr = document.createElement("div");
    hdr.className   = "cmd-category-header";
    hdr.textContent = cat;
    list.appendChild(hdr);

    for (const cmd of items) {
      const card = document.createElement("div");
      card.className = `cmd-card cat-${cmd.category}`;
      card.innerHTML = `
        <div class="cmd-name">${cmd.name}</div>
        <div class="cmd-doc">${escHtml(cmd.doc)}</div>
        <div class="cmd-sig">${escHtml(cmd.signature)}</div>
      `;
      card.addEventListener("click", () => insertCommand(cmd));
      list.appendChild(card);
    }
  }
}

function insertCommand(cmd) {
  const snippet = cmd.example || cmd.name;
  const { from, to } = editorView.state.selection.main;
  const insertPos     = editorView.state.doc.lineAt(to).to;
  editorView.dispatch({
    changes: { from: insertPos, insert: "\n" + snippet },
    selection: { anchor: insertPos + 1 + snippet.length },
  });
  editorView.focus();
}

// ── Examples panel ────────────────────────────────────────────────────────────

async function loadExamples() {
  let data;
  try {
    const resp = await fetch("/api/examples");
    data = await resp.json();
  } catch { return; }

  const list = document.getElementById("example-list");
  list.innerHTML = "";
  for (const ex of data.examples || []) {
    const item = document.createElement("div");
    item.className = "example-item";
    item.innerHTML = `
      <div class="example-name">${escHtml(ex.name)}</div>
      <div class="example-desc">${escHtml(ex.description)}</div>
    `;
    item.addEventListener("click", () => loadExample(ex.name));
    list.appendChild(item);
  }
}

async function loadExample(name) {
  let data;
  try {
    const resp = await fetch(`/api/examples/${name}`);
    data = await resp.json();
  } catch { return; }
  if (!data.code) return;

  editorView.dispatch({
    changes: { from: 0, to: editorView.state.doc.length, insert: data.code },
  });
  currentFile = `${name}.pxl`;
  setStatus(`Loaded: ${name}.pxl`, "ok");
  clearConsole();
  refreshFlowchart();
}

// ── Workspace ─────────────────────────────────────────────────────────────────

function getEditorState() {
  return {
    code:             editorView.state.doc.toString(),
    filename:         currentFile,
    cursor:           { from: editorView.state.selection.main.from },
    preview_visible:  document.getElementById("preview-panel").style.display !== "none",
    preview_position: {
      left: document.getElementById("preview-panel").style.left || "auto",
      top:  document.getElementById("preview-panel").style.top  || "auto",
    },
    last_saved: new Date().toISOString(),
  };
}

function restoreEditorState(ws) {
  if (!ws) return;
  if (ws.code !== undefined) {
    editorView.dispatch({
      changes: { from: 0, to: editorView.state.doc.length, insert: ws.code },
    });
  }
  if (ws.filename) currentFile = ws.filename;
  if (ws.cursor)   editorView.dispatch({ selection: { anchor: ws.cursor.from || 0 } });
  const panel = document.getElementById("preview-panel");
  if (ws.preview_visible) panel.style.display = "flex";
  if (ws.preview_position) {
    if (ws.preview_position.left !== "auto") panel.style.left = ws.preview_position.left;
    if (ws.preview_position.top  !== "auto") panel.style.top  = ws.preview_position.top;
  }
}

async function saveWorkspaceNow() {
  await saveWorkspace(getEditorState());
  setStatus("Workspace saved", "ok");
  setTimeout(() => setStatus(""), 2000);
}

// ── File load (from disk via hidden <input>) ──────────────────────────────────

function setupFileLoad() {
  const input = document.getElementById("file-load-input");
  input.addEventListener("change", async () => {
    const file = input.files[0];
    if (!file) return;
    const text = await file.text();
    if (file.name.endsWith(".json")) {
      try {
        const ws = JSON.parse(text);
        restoreEditorState(ws);
        setStatus(`Loaded workspace: ${file.name}`, "ok");
      } catch { setStatus("Invalid workspace JSON", "error"); }
    } else {
      editorView.dispatch({
        changes: { from: 0, to: editorView.state.doc.length, insert: text },
      });
      currentFile = file.name;
      setStatus(`Opened: ${file.name}`, "ok");
    }
    input.value = "";
    refreshFlowchart();
  });
}

// ── Utility ───────────────────────────────────────────────────────────────────

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

const NEW_PIPELINE = `# New pipeline
LOAD "examples/sample.jpg"
RESIZE 640 480
GRAYSCALE
THRESHOLD_OTSU
FIND_CONTOURS
DRAW_BOUNDING_BOXES
SAVE "output/result.png"
`;

// ── Boot ──────────────────────────────────────────────────────────────────────

async function boot() {
  // Load saved workspace or start with template
  const ws = await loadWorkspace();
  const initialCode = (ws && ws.code) ? ws.code : NEW_PIPELINE;

  initCodeMirror(initialCode);

  if (ws) restoreEditorState(ws);

  // Wire toolbar buttons
  document.getElementById("btn-run").addEventListener("click", runPipeline);
  document.getElementById("btn-lint").addEventListener("click", lintPipeline);
  document.getElementById("btn-save").addEventListener("click", saveWorkspaceNow);
  document.getElementById("btn-load").addEventListener("click", () =>
    document.getElementById("file-load-input").click()
  );
  document.getElementById("btn-new").addEventListener("click", () => {
    editorView.dispatch({
      changes: { from: 0, to: editorView.state.doc.length, insert: NEW_PIPELINE },
    });
    currentFile = "untitled.pxl";
    clearConsole();
    refreshFlowchart();
  });

  // Console controls
  document.getElementById("console-toggle").addEventListener("click", () => {
    document.getElementById("panel-console").classList.toggle("collapsed");
  });
  document.getElementById("btn-console-clear").addEventListener("click", e => {
    e.stopPropagation();
    clearConsole();
  });

  // Flow diagram refresh button
  document.getElementById("btn-flow-refresh").addEventListener("click", refreshFlowchart);

  // Preview controls
  document.getElementById("btn-preview-close").addEventListener("click", () => {
    document.getElementById("preview-panel").style.display = "none";
  });
  document.getElementById("btn-preview-zoom-in").addEventListener("click", () => updateZoom(0.25));
  document.getElementById("btn-preview-zoom-out").addEventListener("click", () => updateZoom(-0.25));
  document.getElementById("btn-preview-reset").addEventListener("click", () => {
    previewZoom = 1.0;
    document.getElementById("preview-img").style.transform = "scale(1)";
  });

  // Preview drag
  makeDraggable(
    document.getElementById("preview-panel"),
    document.getElementById("preview-header")
  );

  // Console collapsible
  initCollapsibleConsole(
    document.getElementById("console-toggle"),
    document.getElementById("panel-console")
  );

  // Command search filter
  document.getElementById("cmd-search").addEventListener("input", e => {
    const q = e.target.value.trim().toLowerCase();
    const filtered = q
      ? allCommands.filter(c =>
          c.name.toLowerCase().includes(q) ||
          c.doc.toLowerCase().includes(q)  ||
          c.category.toLowerCase().includes(q))
      : allCommands;
    renderCommands(filtered);
  });

  // File load input
  setupFileLoad();

  // Export workspace button (Ctrl+Shift+S)
  document.addEventListener("keydown", e => {
    if (e.ctrlKey && e.shiftKey && e.key === "S") {
      e.preventDefault();
      exportWorkspaceFile(getEditorState());
    }
  });

  // Load remote data
  await Promise.all([loadCommands(), loadExamples()]);

  // Initial flow diagram
  refreshFlowchart();
  setStatus("Ready", "ok");
  setTimeout(() => setStatus(""), 3000);
}

boot();
