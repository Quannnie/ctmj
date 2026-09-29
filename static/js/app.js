/* ==========================================================================
   CJPS progressive enhancement
   --------------------------------------------------------------------------
   The form and the results work without JavaScript. This file only adds
   affordances that HTML cannot express: an animated submit state, animated
   probability counters, and client-side constraint hints.

   No dependencies, no framework, no build step.
   ========================================================================== */

(function () {
  "use strict";

  /** Respect the OS reduced-motion setting for every effect below. */
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------------------------------------------------------------------
     Probability counters + bar fills
     ------------------------------------------------------------------- */
  function animateCounters(root) {
    var nodes = root.querySelectorAll("[data-count-to]");
    if (!nodes.length) return;

    Array.prototype.forEach.call(nodes, function (node) {
      var target = parseFloat(node.getAttribute("data-count-to"));
      if (isNaN(target)) return;

      // Reduced motion: show the final value immediately.
      if (reduceMotion) {
        node.textContent = target.toFixed(1) + "%";
        return;
      }

      var duration = 900;
      var start = null;

      function frame(now) {
        if (start === null) start = now;
        var t = Math.min((now - start) / duration, 1);
        // easeOutCubic — fast start, gentle settle.
        var eased = 1 - Math.pow(1 - t, 3);
        node.textContent = (target * eased).toFixed(1) + "%";
        if (t < 1) requestAnimationFrame(frame);
      }

      requestAnimationFrame(frame);
    });
  }

  function fillBars(root) {
    var fills = root.querySelectorAll(".meter__fill");
    if (!fills.length) return;

    Array.prototype.forEach.call(fills, function (fill) {
      var pct = fill.getAttribute("data-fill") || "0";
      if (reduceMotion) {
        fill.style.width = pct + "%";
        return;
      }
      // Next frame, so the transition actually runs from 0.
      requestAnimationFrame(function () {
        requestAnimationFrame(function () {
          fill.style.width = pct + "%";
        });
      });
    });
  }

  /* ---------------------------------------------------------------------
     Form behaviour
     ------------------------------------------------------------------- */
  function initForm() {
    var form = document.getElementById("prediction-form");
    if (!form) return;

    var submit = form.querySelector("[data-submit]");
    var reset = form.querySelector("[data-reset]");

    if (submit) {
      form.addEventListener("submit", function () {
        if (!form.checkValidity()) return;
        // Disable after validation so the browser can still focus the first
        // invalid control.
        submit.setAttribute("aria-disabled", "true");
        var label = submit.querySelector("[data-submit-label]");
        if (label) label.textContent = "Đang tính toán…";
        var spinner = submit.querySelector("[data-spinner]");
        if (spinner) spinner.hidden = false;
      });
    }

    if (reset) {
      reset.addEventListener("click", function () {
        // Navigate so server-rendered results and errors are discarded too.
        window.location.href = form.getAttribute("action") || window.location.pathname;
      });
    }

    // Clear a field's error styling as soon as the user edits it, but keep
    // the message until the next submit so it is not silently ignored.
    Array.prototype.forEach.call(form.querySelectorAll("[aria-invalid]"), function (control) {
      control.addEventListener("input", function () {
        control.removeAttribute("aria-invalid");
      });
    });

    // Stepper buttons for the integer fields: fewer than 12 people should not
    // have to drag a number spinner.
    Array.prototype.forEach.call(form.querySelectorAll("[data-stepper]"), function (wrapper) {
      var input = wrapper.querySelector("input");
      if (!input) return;

      Array.prototype.forEach.call(wrapper.querySelectorAll("[data-delta]"), function (btn) {
        btn.addEventListener("click", function () {
          var delta = parseInt(btn.getAttribute("data-delta"), 10) || 0;
          var min = parseInt(input.getAttribute("min"), 10);
          var max = parseInt(input.getAttribute("max"), 10);
          var current = parseInt(input.value, 10);
          if (isNaN(current)) current = min || 0;
          var next = current + delta;
          if (!isNaN(min) && next < min) next = min;
          if (!isNaN(max) && next > max) next = max;
          input.value = next;
          input.dispatchEvent(new Event("input", { bubbles: true }));
          input.focus();
        });
      });
    });
  }

  /* ---------------------------------------------------------------------
     Results: focus management + scroll
     ------------------------------------------------------------------- */
  function initResults() {
    var results = document.getElementById("prediction-results");
    if (!results) return;

    fillBars(results);
    animateCounters(results);

    // Move focus to the results so screen-reader users are told the page
    // changed, then bring it into view.
    var hasTabindex = results.hasAttribute("tabindex");
    if (!hasTabindex) results.setAttribute("tabindex", "-1");
    results.focus({ preventScroll: true });

    if (reduceMotion) return;
    results.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  /* ---------------------------------------------------------------------
     Error summary focus (accessibility requirement for multi-error forms)
     ------------------------------------------------------------------- */
  function initErrorSummary() {
    var summary = document.querySelector("[data-error-summary]");
    if (!summary) return;
    if (!summary.hasAttribute("tabindex")) summary.setAttribute("tabindex", "-1");
    summary.focus();
  }

  /* ---------------------------------------------------------------------
     Boot
     ------------------------------------------------------------------- */
  function boot() {
    initForm();
    initResults();
    initErrorSummary();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
