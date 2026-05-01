document.addEventListener("DOMContentLoaded", function () {
  var resultEl = document.querySelector("[data-md-component='search-result']");
  if (!resultEl) return;

  new MutationObserver(function () {
    resultEl.querySelectorAll("details:not([open])").forEach(function (d) {
      d.open = true;
    });
  }).observe(resultEl, { childList: true, subtree: true });
});
