// pixlang/editor/static/js/connection.js — tells the user when the server is gone
// or the page's token is stale, and notices on its own when the server is back.

import { api, onConnection, $, h } from "./util.js";

const RETRY_MS = 3000;
const PROBE_TIMEOUT_MS = 4000;

export function watchConnection({ onRecovered }) {
  const bar = $("#conn-banner");
  if (!bar) return;
  let state = "ok";
  let timer = null;

  const probe = () => api("/api/info", undefined, { timeout: PROBE_TIMEOUT_MS }).catch(() => {});

  function render() {
    if (state === "ok") { bar.hidden = true; return; }
    bar.hidden = false;
    if (state === "offline") {
      bar.replaceChildren(h("span", null, "Server not reachable — retrying…"));
    } else {
      bar.replaceChildren(
        h("span", null, "Session expired — reload the page. Unsaved edits are kept."),
        h("button", { class: "btn btn-secondary", onclick: () => location.reload() }, "Reload"));
    }
  }

  onConnection((next) => {
    if (next === state) return;
    const wasDown = state !== "ok";
    state = next;
    render();
    clearInterval(timer);
    timer = null;
    // a restarted server answers again but rejects our old token, so only "offline" is worth retrying
    if (state === "offline") timer = setInterval(probe, RETRY_MS);
    if (state === "ok" && wasDown) onRecovered();
  });
}
