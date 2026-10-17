// Applied before first paint (classic script) so there is no theme flash.
(function () {
  var theme = null;
  try { theme = localStorage.getItem("pixlang.theme"); } catch (e) { /* storage blocked */ }
  if (theme !== "light" && theme !== "dark") {
    theme = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  document.documentElement.setAttribute("data-theme", theme);
  if (/Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent)) {
    document.documentElement.classList.add("is-mac");
  }
})();
