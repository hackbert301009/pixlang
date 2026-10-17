// pixlang/editor/static/js/lang.js
// PixLang language support for CodeMirror 6: tokenizer, highlight style,
// autocomplete, hover docs, lint integration, runtime error marks, theme.
// The keyword / command lists come from the server (/api/language) so new
// language features highlight without touching this file.

import {
  StreamLanguage, HighlightStyle, syntaxHighlighting, LanguageSupport, indentUnit,
  tags as t, EditorView, hoverTooltip, Decoration, StateField, StateEffect,
  snippetCompletion, linter, lintGutter, forceLinting,
} from "../vendor/cm6.js";
import { api, h, catVar } from "./util.js";

// Keywords that open / close / redirect control flow get the control colour.
const CONTROL = new Set([
  "IF", "ELIF", "ELSE", "ENDIF", "WHILE", "FOR", "TO", "STEP", "BREAK", "CONTINUE",
  "REPEAT", "END", "RETURN", "FUNC", "ENDFUNC", "ROI", "ROI_RESET", "WITH", "ENDWITH",
]);
// Block openers / closers used for auto-indent.
const OPENERS = new Set(["IF", "ELIF", "ELSE", "WHILE", "FOR", "REPEAT", "FUNC", "ROI", "WITH"]);
const CLOSERS = new Set(["ENDIF", "END", "ENDFUNC", "ROI_RESET", "ELIF", "ELSE", "ENDWITH"]);

/**
 * Build the StreamLanguage for a given language description.
 * lang = {keywords: [...], commands: [{name,...}], ...}
 */
export function pixlangLanguage(lang) {
  const keywords = new Set(lang.keywords || []);
  const commands = new Set((lang.commands || []).map(c => c.name));

  return StreamLanguage.define({
    name: "pixlang",
    startState: () => ({ inString: false, afterFunc: false, lastSig: "", depth: 0, first: true }),
    token(stream, state) {
      // ── inside a string: chunks and escapes
      if (state.inString) {
        if (stream.match(/^\\(x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)/)) return "pxEscape";
        if (stream.match(/^[^"\\]+/)) return "pxString";
        if (stream.eat('"')) { state.inString = false; state.lastSig = "v"; return "pxString"; }
        stream.next();
        return "pxString";
      }
      if (stream.sol()) { state.lastSig = ""; state.first = true; }
      if (stream.eatSpace()) return null;
      const first = state.first;
      state.first = false;

      if (stream.match(/^#.*/)) return "pxComment";
      if (stream.eat('"')) { state.inString = true; return "pxString"; }
      if (stream.match(/^\\\s*$/)) return "pxOp";              // line continuation

      if (stream.match(/^\$[A-Za-z_]\w*/)) { state.lastSig = "v"; return "pxVar"; }
      if (stream.match(/^&[A-Za-z_]\w*/)) { state.lastSig = "v"; return "pxPointer"; }
      // *p deref: '*' glued to a name and not after a value
      if (state.lastSig !== "v" && stream.match(/^\*(?=[$A-Za-z_])/)) { return "pxDeref"; }

      if (stream.match(/^(0[xX][0-9a-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|(\d[\d_]*)?\.\d+([eE][+-]?\d+)?|\d[\d_]*([eE][+-]?\d+)?)/)) {
        state.lastSig = "v"; return "pxNumber";
      }
      if (stream.match(/^(==|!=|<=|>=|<<|>>|&&|\|\||->)/)) { state.lastSig = "o"; return "pxCompare"; }
      if (stream.match(/^[-+*/%<>=!&|^~]/)) { state.lastSig = "o"; return "pxOp"; }
      if (stream.match(/^[()[\]{},:;.]/)) { state.lastSig = stream.current() === ")" || stream.current() === "]" ? "v" : "o"; return "pxPunc"; }

      if (stream.match(/^_(?![\w])/)) { state.lastSig = "v"; return "pxRegister"; }

      const m = stream.match(/^[A-Za-z_]\w*/);
      if (m) {
        const w = m[0];
        state.lastSig = "v";
        if (state.afterFunc) { state.afterFunc = false; return "pxFuncDef"; }
        if (w === "TRUE" || w === "FALSE" || w === "true" || w === "false") return "pxBool";
        if (keywords.has(w)) {
          if (w === "FUNC") state.afterFunc = true;
          if (first) {
            if (CLOSERS.has(w)) state.depth = Math.max(0, state.depth - 1);
            if (OPENERS.has(w)) state.depth += 1;
          }
          state.lastSig = "k";
          return CONTROL.has(w) ? "pxControl" : "pxKeyword";
        }
        if (commands.has(w)) return "pxCommand";
        if (stream.peek() === "(") return "pxCall";
        if (/^[A-Z][A-Z0-9_]+$/.test(w)) return "pxCommandUnknown";
        return "pxIdent";
      }
      stream.next();
      return "pxInvalid";
    },
    indent(state, textAfter, cx) {
      const w = (textAfter.trim().match(/^[A-Z_]+/) || [""])[0];
      const d = CLOSERS.has(w) ? Math.max(0, state.depth - 1) : state.depth;
      return d * cx.unit;
    },
    languageData: {
      commentTokens: { line: "#" },
      closeBrackets: { brackets: ["(", "[", '"'] },
      indentOnInput: /^\s*(ENDIF|END|ENDFUNC|ELSE|ELIF|ROI_RESET|ENDWITH)\b$/,
    },
    tokenTable: {
      pxComment: t.lineComment,
      pxString: t.string,
      pxEscape: t.escape,
      pxVar: t.special(t.variableName),
      pxPointer: t.special(t.name),
      pxDeref: t.derefOperator,
      pxNumber: t.number,
      pxCompare: t.compareOperator,
      pxOp: t.operator,
      pxPunc: t.punctuation,
      pxRegister: t.atom,
      pxBool: t.bool,
      pxKeyword: t.keyword,
      pxControl: t.controlKeyword,
      pxCommand: t.function(t.variableName),
      pxCommandUnknown: t.macroName,
      pxCall: t.function(t.propertyName),
      pxFuncDef: t.function(t.definition(t.variableName)),
      pxIdent: t.variableName,
      pxInvalid: t.invalid,
    },
  });
}

export const pixlangHighlight = HighlightStyle.define([
  { tag: t.lineComment, color: "var(--syn-comment)", fontStyle: "italic" },
  { tag: t.string, color: "var(--syn-string)" },
  { tag: t.escape, color: "var(--syn-escape)", fontWeight: "500" },
  { tag: t.special(t.variableName), color: "var(--syn-var)" },
  { tag: t.special(t.name), color: "var(--syn-pointer)" },
  { tag: t.derefOperator, color: "var(--syn-pointer)", fontWeight: "700" },
  { tag: t.atom, color: "var(--syn-pointer)", fontWeight: "700" },
  { tag: t.number, color: "var(--syn-number)" },
  { tag: t.bool, color: "var(--syn-number)" },
  { tag: [t.operator, t.compareOperator], color: "var(--syn-op)" },
  { tag: t.punctuation, color: "var(--syn-op)" },
  { tag: t.keyword, color: "var(--syn-keyword)", fontWeight: "500" },
  { tag: t.controlKeyword, color: "var(--syn-control)", fontWeight: "600" },
  { tag: t.function(t.variableName), color: "var(--syn-command)", fontWeight: "500" },
  { tag: t.macroName, color: "var(--syn-command)", textDecoration: "underline dotted", textUnderlineOffset: "3px" },
  { tag: t.function(t.propertyName), color: "var(--syn-funcdef)" },
  { tag: t.function(t.definition(t.variableName)), color: "var(--syn-funcdef)", fontWeight: "600" },
  { tag: t.variableName, color: "var(--syn-ident)" },
  { tag: t.invalid, color: "var(--syn-invalid)" },
]);

// ── Editor chrome theme (CSS variables → follows light/dark) ────────────────
export const pixlangTheme = EditorView.theme({
  "&": { height: "100%", fontSize: "13px", backgroundColor: "var(--bg-2)", color: "var(--syn-ident)" },
  ".cm-scroller": { fontFamily: "var(--font-mono)", lineHeight: "1.7", fontFeatureSettings: '"calt" 0, "liga" 0' },
  ".cm-content": { padding: "12px 0 40vh", caretColor: "var(--accent)" },
  ".cm-line": { padding: "0 16px 0 12px" },
  ".cm-cursor, .cm-dropCursor": { borderLeft: "2px solid var(--accent)" },
  "&.cm-focused .cm-selectionBackground, .cm-selectionBackground": { backgroundColor: "var(--editor-sel-blur)" },
  "&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground": { backgroundColor: "var(--editor-sel)" },
  ".cm-content ::selection": { backgroundColor: "transparent" },
  ".cm-activeLine": { backgroundColor: "var(--editor-active)" },
  ".cm-selectionMatch": { backgroundColor: "var(--editor-match)", borderRadius: "2px" },
  ".cm-matchingBracket, &.cm-focused .cm-matchingBracket": { backgroundColor: "var(--accent-soft)", outline: "1px solid var(--accent-ring)", borderRadius: "2px", color: "inherit" },
  ".cm-gutters": { backgroundColor: "var(--bg-2)", color: "var(--text-3)", border: "none", paddingLeft: "4px" },
  ".cm-lineNumbers .cm-gutterElement": { padding: "0 8px 0 12px", minWidth: "36px", fontSize: "12px", fontVariantNumeric: "tabular-nums" },
  ".cm-activeLineGutter": { backgroundColor: "transparent", color: "var(--text-1)" },
  ".cm-foldGutter .cm-gutterElement": { color: "var(--text-3)" },
  ".cm-gutter-lint": { width: "16px" },
  ".cm-gutter-lint .cm-gutterElement": { padding: "0 0 0 2px", display: "flex", alignItems: "center" },
  ".cm-lint-marker": { width: "8px", height: "8px", borderRadius: "50%", content: "none" },
  ".cm-lint-marker-error": { backgroundColor: "var(--danger)", boxShadow: "0 0 0 3px var(--danger-soft)" },
  ".cm-lint-marker-warning": { backgroundColor: "var(--warn)", boxShadow: "0 0 0 3px var(--warn-soft)" },
  ".cm-lint-marker-info": { backgroundColor: "var(--info)", boxShadow: "0 0 0 3px var(--info-soft)" },
  ".cm-lintRange-error": { backgroundImage: "none", textDecoration: "underline wavy var(--danger)", textUnderlineOffset: "4px", textDecorationThickness: "1px", textDecorationSkipInk: "none" },
  ".cm-lintRange-warning": { backgroundImage: "none", textDecoration: "underline wavy var(--warn)", textUnderlineOffset: "4px", textDecorationThickness: "1px", textDecorationSkipInk: "none" },
  ".cm-lintRange-info": { backgroundImage: "none", textDecoration: "underline dotted var(--info)", textUnderlineOffset: "4px" },
  ".cm-tooltip": { backgroundColor: "var(--bg-2)", color: "var(--text-1)", border: "1px solid var(--border-2)", borderRadius: "8px", boxShadow: "var(--shadow-2)", overflow: "hidden" },
  ".cm-tooltip-lint": { padding: "4px" },
  ".cm-diagnostic": { padding: "6px 10px 6px 12px", fontFamily: "var(--font-ui)", fontSize: "12.5px", borderLeftWidth: "3px", borderRadius: "4px", lineHeight: "1.5" },
  ".cm-diagnostic-error": { borderLeftColor: "var(--danger)" },
  ".cm-diagnostic-warning": { borderLeftColor: "var(--warn)" },
  ".cm-diagnostic-info": { borderLeftColor: "var(--info)" },
  ".cm-diagnosticSource": { color: "var(--text-3)", fontFamily: "var(--font-mono)", fontSize: "11px", marginTop: "2px" },
  ".cm-tooltip-autocomplete": { padding: "4px", minWidth: "300px" },
  ".cm-tooltip-autocomplete > ul": { fontFamily: "var(--font-mono)", fontSize: "12.5px", maxHeight: "300px" },
  ".cm-tooltip-autocomplete > ul > li": { display: "flex", alignItems: "center", gap: "8px", padding: "4px 8px !important", borderRadius: "6px", lineHeight: "1.4" },
  ".cm-tooltip-autocomplete > ul > li[aria-selected]": { backgroundColor: "var(--accent-soft)", color: "var(--text-1)" },
  ".cm-completionLabel": { color: "var(--text-1)" },
  ".cm-completionMatchedText": { textDecoration: "none", color: "var(--accent-text)", fontWeight: "700" },
  ".cm-completionDetail": { marginLeft: "auto", paddingLeft: "16px", color: "var(--text-3)", fontStyle: "normal", fontFamily: "var(--font-ui)", fontSize: "11px" },
  ".cm-completionIcon": { width: "16px", padding: "0", opacity: "1", fontSize: "11px", display: "inline-flex", alignItems: "center", justifyContent: "center", borderRadius: "4px", height: "16px", flex: "none" },
  ".cm-completionIcon-function::after": { content: "'ƒ'", color: "var(--syn-command)" },
  ".cm-completionIcon-keyword::after": { content: "'K'", color: "var(--syn-keyword)", fontFamily: "var(--font-ui)", fontWeight: "700", fontSize: "10px" },
  ".cm-completionIcon-variable::after": { content: "'$'", color: "var(--syn-var)" },
  ".cm-completionIcon-constant::after": { content: "'●'", color: "var(--syn-string)", fontSize: "8px" },
  ".cm-completionIcon-method::after": { content: "'λ'", color: "var(--syn-funcdef)" },
  ".cm-tooltip.cm-completionInfo": { padding: "0", marginLeft: "4px", maxWidth: "380px" },
  ".cm-panels": { backgroundColor: "var(--bg-1)", color: "var(--text-1)" },
  ".cm-panels-bottom": { borderTop: "1px solid var(--border)" },
  ".cm-panels-top": { borderBottom: "1px solid var(--border)" },
  ".cm-search": { display: "flex", flexWrap: "wrap", alignItems: "center", gap: "6px", padding: "8px 12px", fontFamily: "var(--font-ui)", fontSize: "12px" },
  ".cm-search br": { display: "none" },
  ".cm-search label": { display: "inline-flex", alignItems: "center", gap: "4px", color: "var(--text-2)" },
  ".cm-textfield": { height: "26px", padding: "0 8px", borderRadius: "6px", border: "1px solid var(--border-2)", backgroundColor: "var(--bg-inset)", color: "var(--text-1)", fontFamily: "var(--font-mono)", fontSize: "12px", outline: "none" },
  ".cm-textfield:focus": { borderColor: "var(--accent)", boxShadow: "0 0 0 3px var(--accent-soft)" },
  ".cm-button": { height: "26px", padding: "0 10px", borderRadius: "6px", border: "1px solid var(--border-2)", backgroundImage: "none", backgroundColor: "var(--bg-3)", color: "var(--text-1)", fontFamily: "var(--font-ui)", fontSize: "12px", cursor: "pointer" },
  ".cm-button:hover": { backgroundColor: "var(--bg-4)" },
  ".cm-search [name=close]": { marginLeft: "auto", color: "var(--text-3)", fontSize: "16px", cursor: "pointer", background: "none", border: "none" },
  ".cm-searchMatch": { backgroundColor: "var(--warn-soft)", outline: "1px solid color-mix(in srgb, var(--warn) 50%, transparent)", borderRadius: "2px" },
  ".cm-searchMatch-selected": { backgroundColor: "color-mix(in srgb, var(--warn) 35%, transparent)" },
  ".cm-foldPlaceholder": { backgroundColor: "var(--bg-3)", border: "1px solid var(--border-2)", color: "var(--text-3)", borderRadius: "4px", padding: "0 6px" },
  ".cm-placeholder": { color: "var(--text-3)" },
  ".cm-runtime-error-line": { backgroundColor: "var(--danger-soft)", boxShadow: "inset 2px 0 0 var(--danger)" },
  ".cm-flash-line": { animation: "px-flash 900ms ease-out" },
  "@keyframes px-flash": { from: { backgroundColor: "var(--accent-soft)" }, to: { backgroundColor: "transparent" } },
});

// ── Runtime error line decoration ───────────────────────────────────────────
export const setRuntimeError = StateEffect.define();
export const runtimeErrorField = StateField.define({
  create: () => Decoration.none,
  update(deco, tr) {
    deco = deco.map(tr.changes);
    for (const e of tr.effects) {
      if (e.is(setRuntimeError)) {
        const line = e.value;
        if (!line || line < 1 || line > tr.state.doc.lines) deco = Decoration.none;
        else {
          const l = tr.state.doc.line(line);
          deco = Decoration.set([Decoration.line({ class: "cm-runtime-error-line" }).range(l.from)]);
        }
      }
    }
    return deco;
  },
  provide: f => EditorView.decorations.from(f),
});

const flashEffect = StateEffect.define();
export const flashField = StateField.define({
  create: () => Decoration.none,
  update(deco, tr) {
    deco = deco.map(tr.changes);
    for (const e of tr.effects) if (e.is(flashEffect)) {
      deco = e.value == null ? Decoration.none
        : Decoration.set([Decoration.line({ class: "cm-flash-line" }).range(e.value)]);
    }
    return deco;
  },
  provide: f => EditorView.decorations.from(f),
});

/** Move the cursor to line/col, scroll it into view and flash the line. */
export function jumpTo(view, line, col) {
  const doc = view.state.doc;
  const ln = Math.max(1, Math.min(line || 1, doc.lines));
  const l = doc.line(ln);
  const pos = Math.min(l.from + Math.max(0, (col || 1) - 1), l.to);
  view.dispatch({
    selection: { anchor: pos },
    effects: [EditorView.scrollIntoView(pos, { y: "center" }), flashEffect.of(l.from)],
  });
  view.focus();
  setTimeout(() => view.dispatch({ effects: flashEffect.of(null) }), 950);
}

// ── Diagnostics helpers ─────────────────────────────────────────────────────
function diagRange(doc, line, col) {
  const ln = Math.max(1, Math.min(line || 1, doc.lines));
  const l = doc.line(ln);
  if (col && col >= 1) {
    const from = Math.min(l.from + col - 1, l.to);
    const rest = doc.sliceString(from, l.to);
    const m = rest.match(/^("[^"]*"?|[$&]?[\w.]+|\S)/);
    const to = Math.min(from + (m ? m[0].length : 1), l.to);
    return { from, to: Math.max(to, from) };
  }
  const text = doc.sliceString(l.from, l.to);
  const lead = text.length - text.trimStart().length;
  const trail = text.trimEnd().length;
  return { from: l.from + lead, to: l.from + Math.max(trail, lead) };
}

/**
 * Lint extension: fetches /api/lint, merges the last runtime error, reports
 * everything to onResult(problems) for the Problems panel.
 */
export function pixlangLinter({ getRuntime, onResult }) {
  return [
    linter(async (view) => {
      const doc = view.state.doc;
      const code = doc.toString();
      let issues = [];
      try {
        issues = (await api("/api/lint", { code })).issues || [];
      } catch (e) {
        issues = [];
      }
      if (view.state.doc.toString() !== code) return []; // stale, a newer pass follows
      const problems = issues.map(i => ({ ...i, source: "lint" }));
      const rt = getRuntime();
      if (rt && rt.line) problems.unshift({ ...rt, source: "run" });
      onResult(problems);
      const diags = [];
      for (const p of problems) {
        if (!p.line) continue;
        if (p.line > doc.lines) continue;
        const { from, to } = diagRange(doc, p.line, p.col);
        diags.push({
          from, to,
          severity: p.severity === "warning" ? "warning" : p.severity === "info" ? "info" : "error",
          message: p.message,
          source: p.source === "run" ? (p.kindLabel || "Runtime") : p.code,
        });
      }
      return diags;
    }, { delay: 350 }),
    lintGutter({ hoverTime: 150 }),
  ];
}
export { forceLinting };

// ── Command docs card (hover + completion info) ─────────────────────────────
export function commandDoc(cmd) {
  const params = (cmd.params || []).filter(p => p.name);
  return h("div", { class: "px-doc" },
    h("div", { class: "px-doc-head" },
      h("span", { class: "px-doc-name" }, cmd.name),
      h("span", { class: "px-doc-cat", style: { "--rail": catVar(cmd.category) } }, cmd.category)),
    h("div", { class: "px-doc-usage" }, cmd.usage || cmd.name),
    cmd.summary ? h("div", { class: "px-doc-sum" }, cmd.summary) : null,
    params.length ? h("ul", { class: "px-doc-params" },
      params.map(p => h("li", null,
        h("code", null, p.name),
        p.type ? ` · ${p.type}` : "",
        p.default != null ? ` · default ${p.default}` : (p.optional ? " · optional" : ""),
        p.doc ? ` — ${p.doc}` : ""))) : null,
    cmd.doc ? h("div", { class: "px-doc-body" }, cmd.doc) : null,
  );
}

export function pixlangHover(lang) {
  const byName = new Map((lang.commands || []).map(c => [c.name, c]));
  return hoverTooltip((view, pos) => {
    const line = view.state.doc.lineAt(pos);
    const text = line.text;
    let s = pos - line.from, e = s;
    while (s > 0 && /\w/.test(text[s - 1])) s--;
    while (e < text.length && /\w/.test(text[e])) e++;
    if (s === e) return null;
    const word = text.slice(s, e);
    const cmd = byName.get(word);
    if (!cmd) return null;
    if (/[$&]/.test(text[s - 1] || "")) return null;
    return {
      pos: line.from + s, end: line.from + e, above: true,
      create: () => ({ dom: commandDoc(cmd) }),
    };
  }, { hoverTime: 350 });
}

// ── Autocomplete ────────────────────────────────────────────────────────────
function snippetFor(cmd) {
  const req = (cmd.params || []).filter(p => p.name && !p.optional && !p.name.endsWith("…"));
  if (!req.length) return cmd.name;
  const parts = req.map(p => {
    const nm = p.name.replace(/[{}$\\]/g, "");
    return /str/i.test(p.type || "") || /path|name|file|text|mode/i.test(nm) && /str/i.test(p.type || "str")
      ? `"\${${nm}}"` : `\${${nm}}`;
  });
  return cmd.name + " " + parts.join(" ");
}

function collectSymbols(doc) {
  const text = doc.toString();
  const vars = new Set(), funcs = new Set(), bufs = new Set();
  let m;
  const reSet = /^\s*(?:SET|GLOBAL|EXPORT)\s+\$?([A-Za-z_]\w*)/gm;
  while ((m = reSet.exec(text))) vars.add(m[1]);
  const reFor = /^\s*FOR\s+\$?([A-Za-z_]\w*)/gm;
  while ((m = reFor.exec(text))) vars.add(m[1]);
  const reRef = /\$([A-Za-z_]\w*)/g;
  while ((m = reRef.exec(text))) vars.add(m[1]);
  const reFunc = /^\s*FUNC\s+([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?/gm;
  while ((m = reFunc.exec(text))) {
    funcs.add(m[1]);
    for (const p of (m[2] || "").split(/[\s,]+/)) if (p) vars.add(p.replace(/^\$/, ""));
  }
  const reBuf = /^\s*(?:ALLOC|BUF|CHECKPOINT)\s+"?([A-Za-z_][\w-]*)"?/gm;
  while ((m = reBuf.exec(text))) bufs.add(m[1]);
  return { vars, funcs, bufs };
}

export function pixlangCompletion(lang) {
  const keywords = lang.keywords || [];
  const builtinVars = lang.builtin_vars || ["ITER", "ITER1"];
  const cmdOptions = (lang.commands || []).map(cmd => snippetCompletion(snippetFor(cmd), {
    label: cmd.name,
    type: "function",
    detail: cmd.category,
    info: () => commandDoc(cmd),
    boost: 1,
  }));
  const kwOptions = keywords.map(k => ({ label: k, type: "keyword", detail: "keyword", boost: CONTROL.has(k) ? 0 : -1 }));

  return (context) => {
    const line = context.state.doc.lineAt(context.pos);
    const before = line.text.slice(0, context.pos - line.from);
    if (/^\s*#/.test(before)) return null;                     // comment
    const quotes = (before.match(/"/g) || []).length;
    const sym = collectSymbols(context.state.doc);

    // Inside a string: offer buffer / checkpoint names
    if (quotes % 2 === 1) {
      const w = context.matchBefore(/[\w-]*/);
      if (!sym.bufs.size || (!context.explicit && !/(OVERLAY|BLEND|CHECKPOINT|USE|FREE)\s+"[\w-]*$/.test(before))) return null;
      return { from: w.from, options: [...sym.bufs].map(b => ({ label: b, type: "constant", detail: "buffer" })), validFor: /^[\w-]*$/ };
    }

    const dollar = context.matchBefore(/\$\w*/);
    if (dollar) {
      const names = new Set([...sym.vars, ...builtinVars]);
      return {
        from: dollar.from,
        options: [...names].map(v => ({ label: "$" + v, type: "variable", detail: builtinVars.includes(v) ? "built-in" : "variable" })),
        validFor: /^\$\w*$/,
      };
    }

    const word = context.matchBefore(/[A-Za-z_]\w*/);
    if (!word && !context.explicit) return null;
    const from = word ? word.from : context.pos;
    const atStart = /^\s*$/.test(line.text.slice(0, from - line.from));

    if (atStart) {
      const funcs = [...sym.funcs].map(f => ({ label: f, type: "method", detail: "FUNC" }));
      return { from, options: [...cmdOptions, ...kwOptions, ...funcs], validFor: /^\w*$/ };
    }
    // Argument position: bare variable names, FUNC names, buffers, keywords like TO/STEP
    const opts = [
      ...[...sym.vars].map(v => ({ label: v, type: "variable", detail: "variable" })),
      ...[...sym.funcs].map(f => ({ label: f, type: "method", detail: "FUNC" })),
      ...[...sym.bufs].map(b => ({ label: b, type: "constant", detail: "buffer" })),
      ...kwOptions.filter(k => ["TO", "STEP", "AS", "FROM", "WITH"].includes(k.label)),
    ];
    if (!opts.length) return null;
    return { from, options: opts, validFor: /^\w*$/ };
  };
}

export function pixlangSupport(lang) {
  return new LanguageSupport(pixlangLanguage(lang), [syntaxHighlighting(pixlangHighlight), indentUnit.of("    ")]);
}
