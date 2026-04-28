// pixlang/editor/static/js/workspace.js
// Workspace persistence: localStorage (fast) + server (survives browser clear).

const LS_KEY = "pixlang_workspace";

/**
 * Save workspace state to localStorage AND the server.
 * The server call is best-effort; failure is silently ignored.
 */
export async function saveWorkspace(state) {
  const ws = { ...state, last_saved: new Date().toISOString() };

  // localStorage — synchronous, instant
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(ws));
  } catch { /* quota exceeded — ignore */ }

  // Server — async, survives browser data clear
  try {
    await fetch("/api/workspace/save", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ workspace: ws }),
    });
  } catch { /* offline — ignore */ }
}

/**
 * Load the last saved workspace.
 * Tries localStorage first (fastest), then falls back to the server.
 * Returns the workspace object or null if nothing is saved.
 */
export async function loadWorkspace() {
  // 1. localStorage
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) {
      const ws = JSON.parse(raw);
      if (ws && typeof ws === "object") return ws;
    }
  } catch { /* corrupt JSON — fall through */ }

  // 2. Server fallback
  try {
    const resp = await fetch("/api/workspace/load");
    const data = await resp.json();
    if (data.workspace) return data.workspace;
  } catch { /* network error */ }

  return null;
}

/**
 * Trigger a browser file download of the workspace as a JSON file.
 * Used by Ctrl+Shift+S.
 */
export function exportWorkspaceFile(state) {
  const ws   = { ...state, last_saved: new Date().toISOString() };
  const blob = new Blob([JSON.stringify(ws, null, 2)], { type: "application/json" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = "pixlang_workspace.json";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
