/* Pure text helpers for the dashboard.
   Kept in their own file with no DOM access so they can be unit tested
   directly. escapeHtml and highlight are the only places where document
   text or model output becomes HTML, so they are the ones that must be
   provably safe. */
(function (root) {
  "use strict";

  /** Escape every character that could start markup. */
  function escapeHtml(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  /** Wrap `span` in <mark> inside `text`.
   *
   *  Both are escaped first and the search happens on the escaped
   *  strings, so the only markup in the result is the <mark> pair this
   *  function adds. Document text and model output can never inject
   *  tags, and a span that is not present is simply not highlighted.
   */
  function highlight(text, span) {
    var safeText = escapeHtml(text);
    if (!span) { return safeText; }

    var safeSpan = escapeHtml(span);
    if (!safeSpan) { return safeText; }

    var at = safeText.indexOf(safeSpan);
    if (at === -1) { return safeText; }

    return safeText.slice(0, at) + "<mark>" + safeSpan + "</mark>" +
      safeText.slice(at + safeSpan.length);
  }

  function extensionOf(name) {
    var parts = String(name || "").split(".");
    return parts.length > 1 ? parts.pop().toLowerCase() : "";
  }

  function formatSize(bytes) {
    if (bytes < 1024) { return bytes + " B"; }
    if (bytes < 1024 * 1024) { return (bytes / 1024).toFixed(1) + " KB"; }
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }

  function titleCase(value) {
    return String(value || "").replace(/_/g, " ").toLowerCase()
      .replace(/\b\w/g, function (c) { return c.toUpperCase(); });
  }

  var api = {
    escapeHtml: escapeHtml,
    highlight: highlight,
    extensionOf: extensionOf,
    formatSize: formatSize,
    titleCase: titleCase
  };

  // Browser: window.CrossDocText. Node (the unit tests): module.exports.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.CrossDocText = api;
  }
}(typeof window !== "undefined" ? window : this));
