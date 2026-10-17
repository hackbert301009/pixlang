// Single ESM entry for the editor's CodeMirror bundle.
// Everything is re-exported from ONE file so the browser loads exactly one
// copy of @codemirror/state (several copies break its instanceof checks with
// "Unrecognized extension value in extension set").
export * from "@codemirror/state";
export * from "@codemirror/view";
export * from "@codemirror/language";
export * from "@codemirror/commands";
export * from "@codemirror/autocomplete";
export * from "@codemirror/lint";
export * from "@codemirror/search";
export { tags, Tag, styleTags, tagHighlighter, highlightTree, classHighlighter } from "@lezer/highlight";
