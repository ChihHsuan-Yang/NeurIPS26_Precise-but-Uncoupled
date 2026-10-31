/* Optional citation convenience. Links and content work without JavaScript. */
(function () {
  "use strict";
  document.querySelectorAll("[data-copy]").forEach(function (button) {
    button.addEventListener("click", async function () {
      var source = document.getElementById(button.dataset.copy);
      var status = button.nextElementSibling;
      if (!source || !status) return;
      try {
        await navigator.clipboard.writeText(source.innerText);
        status.textContent = "Copied.";
      } catch (error) {
        var range = document.createRange();
        range.selectNodeContents(source);
        var selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        status.textContent = "Text selected. Use your browser’s Copy command.";
      }
    });
  });
})();
