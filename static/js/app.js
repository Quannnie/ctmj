/* ==========================================================================
   CJPS — progressive enhancement
   --------------------------------------------------------------------------
   The tool's guidance drove the scope here: *"Animate 1-2 key elements per view
   maximum"*, flagged as High severity under Excessive Motion, and *"Don't use
   back.out on dense data tables; the overshoot reads as sloppy on
   informational UI."* So this file animates exactly two things — the result
   bars and the probability counters — and leaves every hover state to CSS.

   Performance: only transform/opacity are animated; scroll and resize handlers
   are passive and collapsed into a single requestAnimationFrame callback, so
   they can never run more than once per frame.
   ========================================================================== */

(function () {
  "use strict";

  var REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)");

  function reduced() { return REDUCED.matches; }

  /* Coalesce event bursts into one call per frame. */
  function onFrame(fn) {
    var queued = false;
    return function () {
      if (queued) return;
      queued = true;
      requestAnimationFrame(function () {
        queued = false;
        fn();
      });
    };
  }

  /* ---------------------------------------------------------------------
     Scroll reveal
     ---------------------------------------------------------------------
     IntersectionObserver with a rAF-throttled scroll sweep as a backstop, so
     content can never be stranded at opacity 0 if the observer fails to fire.
     Offset is 8px and duration 300ms, matching the tool's stagger preset.
     ------------------------------------------------------------------- */
  function initReveal() {
    var all = document.querySelectorAll("[data-reveal]");
    if (!all.length) return;

    var pending = [];
    Array.prototype.forEach.call(all, function (el) { pending.push(el); });

    function reveal(el) {
      if (!el.classList.contains("is-visible")) el.classList.add("is-visible");
    }

    if (reduced() || !("IntersectionObserver" in window)) {
      pending.forEach(reveal);
      return;
    }

    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          reveal(entry.target);
          io.unobserve(entry.target);
        });
      },
      { rootMargin: "0px 0px -4% 0px", threshold: 0.05 }
    );
    pending.forEach(function (el) { io.observe(el); });

    var sweep = onFrame(function () {
      var limit = window.innerHeight * 0.96;
      for (var i = pending.length - 1; i >= 0; i--) {
        var el = pending[i];
        if (el.getBoundingClientRect().top < limit) {
          reveal(el);
          io.unobserve(el);
          pending.splice(i, 1);
        }
      }
    });

    window.addEventListener("scroll", sweep, { passive: true });
    window.addEventListener("resize", sweep, { passive: true });
    sweep();
  }

  /* ---------------------------------------------------------------------
     Result bars — stagger 0.03 (60ms), the tool's "keep it small" guidance
     ------------------------------------------------------------------- */
  function fillBars(root) {
    var fills = root.querySelectorAll(".meter__fill");
    if (!fills.length) return;

    Array.prototype.forEach.call(fills, function (fill, i) {
      var pct = fill.getAttribute("data-fill") || "0";
      if (reduced()) {
        fill.style.width = pct + "%";
        return;
      }
      var paint = function () {
        requestAnimationFrame(function () { fill.style.width = pct + "%"; });
      };
      if (i === 0) requestAnimationFrame(paint);
      else setTimeout(paint, i * 60);
    });
  }

  /* ---------------------------------------------------------------------
     Probability counters — 600ms, easeOutCubic
     ------------------------------------------------------------------- */
  function animateCounters(root) {
    var nodes = root.querySelectorAll("[data-count-to]");
    if (!nodes.length) return;

    Array.prototype.forEach.call(nodes, function (node) {
      var target = parseFloat(node.getAttribute("data-count-to"));
      if (isNaN(target)) return;

      if (reduced()) {
        node.textContent = target.toFixed(1) + "%";
        return;
      }

      var duration = 600;
      var start = null;

      function frame(now) {
        if (start === null) start = now;
        var t = Math.min((now - start) / duration, 1);
        var eased = 1 - Math.pow(1 - t, 3);
        node.textContent = (target * eased).toFixed(1) + "%";
        if (t < 1) requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    });
  }

  /* ---------------------------------------------------------------------
     Form
     ------------------------------------------------------------------- */
  function initForm() {
    var form = document.getElementById("prediction-form");
    if (!form) return;

    var submit = form.querySelector("[data-submit]");
    var reset = form.querySelector("[data-reset]");

    if (submit) {
      form.addEventListener("submit", function () {
        if (!form.checkValidity()) return;
        submit.setAttribute("aria-disabled", "true");
        submit.setAttribute("data-loading", "true");
        var label = submit.querySelector("[data-submit-label]");
        if (label) label.textContent = "Đang tính toán…";
        var spinner = submit.querySelector("[data-spinner]");
        if (spinner) spinner.hidden = false;
      });
    }

    if (reset) {
      reset.addEventListener("click", function () {
        window.location.href = form.getAttribute("action") || window.location.pathname;
      });
    }

    Array.prototype.forEach.call(form.querySelectorAll("[aria-invalid]"), function (control) {
      control.addEventListener("input", function () {
        control.removeAttribute("aria-invalid");
      });
    });

    // Stepper buttons
    Array.prototype.forEach.call(form.querySelectorAll("[data-stepper]"), function (wrapper) {
      var input = wrapper.querySelector("input");
      if (!input) return;

      Array.prototype.forEach.call(wrapper.querySelectorAll("[data-delta]"), function (btn) {
        btn.addEventListener("click", function () {
          var delta = parseInt(btn.getAttribute("data-delta"), 10) || 0;
          var min = parseInt(input.getAttribute("min"), 10);
          var max = parseInt(input.getAttribute("max"), 10);
          var current = parseInt(input.value, 10);
          if (isNaN(current)) current = isNaN(min) ? 0 : min;
          var next = current + delta;
          if (!isNaN(min) && next < min) next = min;
          if (!isNaN(max) && next > max) next = max;
          input.value = next;
          input.dispatchEvent(new Event("input", { bubbles: true }));
          input.focus({ preventScroll: true });
        });
      });
    });
  }

  /* ---------------------------------------------------------------------
     Results and error summary: focus first, then one smooth scroll
     ------------------------------------------------------------------- */
  function focusAndScroll(el) {
    if (!el.hasAttribute("tabindex")) el.setAttribute("tabindex", "-1");
    el.focus({ preventScroll: true });
    if (reduced()) return;
    var top = el.getBoundingClientRect().top + window.pageYOffset - 72;
    window.scrollTo({ top: Math.max(top, 0), behavior: "smooth" });
  }

  function initResults() {
    var results = document.getElementById("prediction-results");
    if (!results) return;
    fillBars(results);
    animateCounters(results);
    focusAndScroll(results);
  }

  function initErrorSummary() {
    var summary = document.querySelector("[data-error-summary]");
    if (summary) focusAndScroll(summary);
  }

  function boot() {
    initReveal();
    initForm();
    initResults();
    initErrorSummary();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
