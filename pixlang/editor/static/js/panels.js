// pixlang/editor/static/js/panels.js
// Draggable floating panel + collapsible console.

/**
 * Make a panel draggable by its handle element.
 * Clears CSS right/bottom once the user starts dragging so
 * left/top positioning takes over cleanly.
 */
export function makeDraggable(panel, handle) {
  let dragging = false;
  let startX, startY, startLeft, startTop;

  function onDown(e) {
    // Only primary mouse button or single-touch
    if (e.type === "mousedown" && e.button !== 0) return;
    dragging = true;

    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    startX    = clientX;
    startY    = clientY;
    startLeft = panel.offsetLeft;
    startTop  = panel.offsetTop;

    // Remove right/bottom so left/top take full control
    panel.style.right  = "auto";
    panel.style.bottom = "auto";

    e.preventDefault();
  }

  function onMove(e) {
    if (!dragging) return;
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    const dx = clientX - startX;
    const dy = clientY - startY;

    // Clamp to viewport
    const maxLeft = window.innerWidth  - panel.offsetWidth  - 4;
    const maxTop  = window.innerHeight - panel.offsetHeight - 4;
    panel.style.left = `${Math.max(0, Math.min(maxLeft, startLeft + dx))}px`;
    panel.style.top  = `${Math.max(0, Math.min(maxTop,  startTop  + dy))}px`;
  }

  function onUp() { dragging = false; }

  handle.addEventListener("mousedown",  onDown,  { passive: false });
  handle.addEventListener("touchstart", onDown,  { passive: false });
  document.addEventListener("mousemove", onMove);
  document.addEventListener("touchmove", onMove, { passive: false });
  document.addEventListener("mouseup",   onUp);
  document.addEventListener("touchend",  onUp);
}

/**
 * Toggle the console panel collapsed/expanded when its header is clicked.
 * The header click is already wired in app.js; this just exposes the helper.
 */
export function initCollapsibleConsole(header, panel) {
  // The class toggle is handled in app.js via panel.classList.toggle("collapsed").
  // This function is a hook point for any additional setup (e.g. animations).
  // Currently a no-op placeholder so app.js can import it cleanly.
}
