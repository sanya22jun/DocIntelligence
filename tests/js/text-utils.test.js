/* Unit tests for the dashboard's text helpers, run with:
     node tests/js/text-utils.test.js
   These cover the only code path where document text or model output
   becomes HTML, so an escaping regression here would be an XSS hole. */
"use strict";

var path = require("path");
var utils = require(
  path.join(__dirname, "..", "..", "app", "static", "js", "text-utils.js")
);

var failures = 0;
var checks = 0;

function check(name, actual, expected) {
  checks += 1;
  var ok = actual === expected;
  if (!ok) {
    failures += 1;
    console.log("FAIL " + name);
    console.log("     expected: " + JSON.stringify(expected));
    console.log("     actual:   " + JSON.stringify(actual));
  }
}

function contains(name, haystack, needle, shouldContain) {
  checks += 1;
  var found = String(haystack).indexOf(needle) !== -1;
  if (found !== shouldContain) {
    failures += 1;
    console.log("FAIL " + name);
    console.log("     " + JSON.stringify(needle) +
      (shouldContain ? " should be in " : " should NOT be in ") +
      JSON.stringify(haystack));
  }
}

/* ---------------- escapeHtml ---------------- */

check("escapes angle brackets",
  utils.escapeHtml("<b>x</b>"), "&lt;b&gt;x&lt;/b&gt;");

check("escapes ampersand first",
  utils.escapeHtml("a & <b>"), "a &amp; &lt;b&gt;");

check("escapes both quote styles",
  utils.escapeHtml('He said "no" and \'maybe\''),
  "He said &quot;no&quot; and &#39;maybe&#39;");

check("null becomes empty", utils.escapeHtml(null), "");
check("undefined becomes empty", utils.escapeHtml(undefined), "");
check("numbers survive", utils.escapeHtml(42), "42");

contains("a script tag cannot survive escaping",
  utils.escapeHtml('<script>alert(1)</script>'), "<script", false);

contains("an event handler cannot close its attribute",
  utils.escapeHtml('" onmouseover="alert(1)'), '"', false);

/* ---------------- highlight ---------------- */

check("marks the span",
  utils.highlight("Employees receive 20 days of leave.", "20 days"),
  "Employees receive <mark>20 days</mark> of leave.");

check("no span leaves the text escaped only",
  utils.highlight("a < b", null), "a &lt; b");

check("empty span is ignored",
  utils.highlight("a < b", ""), "a &lt; b");

check("absent span is not highlighted",
  utils.highlight("Employees receive 20 days.", "30 days"),
  "Employees receive 20 days.");

check("marks only the first occurrence",
  utils.highlight("20 days then 20 days", "20 days"),
  "<mark>20 days</mark> days then 20 days".replace(
    "<mark>20 days</mark> days", "<mark>20 days</mark>"));

// A document crafted to break out of the highlight must not produce
// any tag other than the <mark> pair this function adds.
var hostile = utils.highlight(
  'Leave is <img src=x onerror="alert(1)"> 20 days', "20 days");
contains("hostile document text is neutralised", hostile, "<img", false);
contains("hostile document text keeps its mark", hostile, "<mark>20 days</mark>", true);
check("exactly one tag pair is added",
  (hostile.match(/<(?!\/?mark>)/g) || []).length, 0);

// A hostile span (i.e. hostile model output) must not inject markup
// either, and must not match after escaping.
var hostileSpan = utils.highlight("Employees receive 20 days.", "<script>");
contains("hostile span injects nothing", hostileSpan, "<script", false);
check("hostile span is simply not found",
  hostileSpan, "Employees receive 20 days.");

// Escaping must not shift the match: the span sits after an escaped
// character, so a naive implementation would mark the wrong slice.
check("span after an escaped character is located correctly",
  utils.highlight('Terms & conditions allow 20 days of leave.', "20 days"),
  "Terms &amp; conditions allow <mark>20 days</mark> of leave.");

check("span containing an escapable character still matches",
  utils.highlight('The limit is "20 days" exactly.', '"20 days"'),
  "The limit is <mark>&quot;20 days&quot;</mark> exactly.");

/* ---------------- small helpers ---------------- */

check("extension is lower-cased", utils.extensionOf("POLICY.PDF"), "pdf");
check("no extension gives empty", utils.extensionOf("README"), "");
check("size in KB", utils.formatSize(2048), "2.0 KB");
check("size in bytes", utils.formatSize(900), "900 B");
check("title case from a constant",
  utils.titleCase("NUMERICAL_CONFLICT"), "Numerical Conflict");

/* ---------------- report ---------------- */

console.log(
  (failures ? "FAILED" : "ok") + " - " + (checks - failures) + "/" + checks +
  " checks passed"
);
process.exit(failures ? 1 : 0);
