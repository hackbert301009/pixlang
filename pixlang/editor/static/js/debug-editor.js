// pixlang/editor/static/js/debug-editor.js
// CodeMirror pieces of the step debugger: breakpoint gutter (click to toggle,
// hover preview), the "current line" band + arrow, and the read-only switch.
// Pure editor plumbing — the session logic lives in debug.js.

import {
  EditorView, EditorState, StateField, StateEffect, Decoration, RangeSet, GutterMarker,
  gutter, Compartment,
} from "../vendor/cm6.js";

const SVGNS = "http://www.w3.org/2000/svg";

// ── Breakpoints ──────────────────────────────────────────────────────────────

/** Replace all breakpoints: [{line, condition?, verified?}] (1-based lines). */
export const setBreakpointsEffect = StateEffect.define();
/** Highlight the paused line / the selected call-site: {line, site}. */
export const setDebugLineEffect = StateEffect.define();

class BreakpointMarker extends GutterMarker {
  constructor(bp, current, ghost = false) {
    super();
    this.bp = bp;
    this.current = current;
    this.ghost = ghost;
  }
  eq(o) {
    return this.current === o.current
      && this.ghost === o.ghost
      && !!this.bp === !!o.bp
      && (!this.bp || (this.bp.condition === o.bp.condition && this.bp.verified === o.bp.verified));
  }
  toDOM() {
    const wrap = document.createElement("span");
    wrap.className = "dbg-mark";
    if (this.bp) {
      const dot = document.createElement("span");
      const cls = ["bp-dot"];
      if (this.bp.condition) cls.push("cond");
      if (this.bp.verified === false) cls.push("unverified");
      dot.className = cls.join(" ");
      wrap.append(dot);
    }
    if (this.ghost) {
      const dot = document.createElement("span");
      dot.className = "bp-dot ghost";
      wrap.append(dot);
    }
    if (this.current) {
      const svg = document.createElementNS(SVGNS, "svg");
      svg.setAttribute("viewBox", "0 0 12 12");
      svg.setAttribute("class", "dbg-arrow");
      svg.setAttribute("aria-hidden", "true");
      const p = document.createElementNS(SVGNS, "path");
      p.setAttribute("d", "M2.5 2.2v7.6a.7.7 0 0 0 1.07.6l6-3.8a.7.7 0 0 0 0-1.2l-6-3.8a.7.7 0 0 0-1.07.6Z");
      svg.append(p);
      wrap.append(svg);
    }
    return wrap;
  }
}

const bpField = StateField.define({
  create: () => RangeSet.empty,
  update(set, tr) {
    set = set.map(tr.changes);
    for (const e of tr.effects) {
      if (!e.is(setBreakpointsEffect)) continue;
      const ranges = [];
      const doc = tr.state.doc;
      for (const bp of [...e.value].sort((a, b) => a.line - b.line)) {
        if (bp.line < 1 || bp.line > doc.lines) continue;
        ranges.push(new BpValue(bp.condition || null, bp.verified !== false).range(doc.line(bp.line).from));
      }
      set = RangeSet.of(ranges, true);
    }
    return set;
  },
});

// RangeSet needs RangeValue subclasses; GutterMarker is one, so carry the data on a marker.
class BpValue extends GutterMarker {
  constructor(condition, verified) {
    super();
    this.condition = condition;
    this.verified = verified;
  }
  eq(o) { return o.condition === this.condition && o.verified === this.verified; }
}

const lineField = StateField.define({
  create: () => ({ cur: null, site: null }),
  update(v, tr) {
    if (tr.docChanged) v = { cur: v.cur == null ? null : tr.changes.mapPos(v.cur), site: v.site == null ? null : tr.changes.mapPos(v.site) };
    for (const e of tr.effects) {
      if (!e.is(setDebugLineEffect)) continue;
      const doc = tr.state.doc;
      const at = (n) => (n && n >= 1 && n <= doc.lines ? doc.line(n).from : null);
      v = { cur: at(e.value.line), site: at(e.value.site) };
    }
    return v;
  },
  provide: (f) => EditorView.decorations.from(f, (v) => {
    const out = [];
    if (v.site != null && v.site !== v.cur) out.push(Decoration.line({ class: "cm-debug-site" }).range(v.site));
    if (v.cur != null) out.push(Decoration.line({ class: "cm-debug-line" }).range(v.cur));
    return Decoration.set(out, true);
  }),
});

/** The line under the mouse in the gutter (for the "click to add" preview dot). */
const setHoverEffect = StateEffect.define();
const hoverField = StateField.define({
  create: () => null,
  update(v, tr) {
    if (v != null && tr.docChanged) v = tr.changes.mapPos(v);
    for (const e of tr.effects) if (e.is(setHoverEffect)) v = e.value;
    return v;
  },
});

function breakpointAt(state, from) {
  let found = null;
  state.field(bpField).between(from, from, (_f, _t, value) => { found = value; return false; });
  return found;
}

/** Current breakpoints as [{line, condition, verified}]. */
export function getBreakpoints(state) {
  const out = [];
  state.field(bpField).between(0, state.doc.length, (from, _to, value) => {
    out.push({ line: state.doc.lineAt(from).number, condition: value.condition, verified: value.verified });
  });
  return out;
}

export function setBreakpoints(view, list) {
  view.dispatch({ effects: setBreakpointsEffect.of(list) });
}

/** Toggle (or force on/off) the breakpoint of a line; returns the new list. */
export function toggleBreakpoint(view, line, force) {
  const list = getBreakpoints(view.state);
  const i = list.findIndex((b) => b.line === line);
  const want = force === undefined ? i < 0 : force;
  if (want && i < 0) list.push({ line, condition: null, verified: true });
  else if (!want && i >= 0) list.splice(i, 1);
  setBreakpoints(view, list);
  return list;
}

export function setDebugLine(view, line, site = null) {
  view.dispatch({ effects: setDebugLineEffect.of({ line, site }) });
}

export function scrollToLine(view, line, { center = false } = {}) {
  if (!line || line < 1 || line > view.state.doc.lines) return;
  const pos = view.state.doc.line(line).from;
  view.dispatch({ effects: EditorView.scrollIntoView(pos, center ? { y: "center" } : { y: "nearest", yMargin: 72 }) });
}

/** Gutter shown left of the line numbers. handlers: {toggle(line), condition(line)} */
export function debugGutter(handlers) {
  const spacer = new BreakpointMarker({ condition: null, verified: true }, true);
  return [
    bpField,
    lineField,
    hoverField,
    gutter({
      class: "cm-debug-gutter",
      lineMarker(view, line) {
        const bp = breakpointAt(view.state, line.from);
        const cur = view.state.field(lineField).cur === line.from;
        if (bp || cur) return new BreakpointMarker(bp, cur);
        return view.state.field(hoverField) === line.from ? new BreakpointMarker(null, false, true) : null;
      },
      lineMarkerChange: (u) => u.startState.field(bpField) !== u.state.field(bpField)
        || u.startState.field(lineField) !== u.state.field(lineField)
        || u.startState.field(hoverField) !== u.state.field(hoverField),
      initialSpacer: () => spacer,
      domEventHandlers: {
        mousemove(view, line) {
          if (view.state.field(hoverField) !== line.from) view.dispatch({ effects: setHoverEffect.of(line.from) });
          return false;
        },
        mouseleave(view) {
          if (view.state.field(hoverField) != null) view.dispatch({ effects: setHoverEffect.of(null) });
          return false;
        },
        mousedown(view, line, ev) {
          if (ev.button !== 0) return false;
          ev.preventDefault();
          handlers.toggle(view.state.doc.lineAt(line.from).number);
          return true;
        },
        contextmenu(view, line, ev) {
          ev.preventDefault();
          handlers.condition(view.state.doc.lineAt(line.from).number);
          return true;
        },
      },
    }),
  ];
}

// ── Read-only while a session is live ────────────────────────────────────────

export const readOnlyCompartment = new Compartment();

export function readOnlyExtension(onBlocked) {
  let last = 0;
  const poke = () => {
    const now = Date.now();
    if (now - last > 2500) { last = now; onBlocked(); }
  };
  return [
    readOnlyCompartment.of(EditorState.readOnly.of(false)),
    EditorView.domEventHandlers({
      keydown(e, view) {
        if (view.state.readOnly && (e.key.length === 1 || e.key === "Backspace" || e.key === "Delete" || e.key === "Enter")
            && !e.ctrlKey && !e.metaKey && !e.altKey) poke();
        return false;
      },
      paste(_e, view) { if (view.state.readOnly) poke(); return false; },
      drop(_e, view) { if (view.state.readOnly) poke(); return false; },
    }),
  ];
}

export function setReadOnly(view, on) {
  view.dispatch({ effects: readOnlyCompartment.reconfigure(EditorState.readOnly.of(on)) });
}
