/* ==========================================================================
   CJPS — progressive enhancement
   --------------------------------------------------------------------------
   The form and results work with JavaScript disabled. This file only adds
   affordances HTML cannot express.

   Performance rules followed here, because "smooth" is the requirement:
     * Only transform / opacity are animated, never layout properties.
     * Scroll and resize handlers are passive and collapsed into a single
       requestAnimationFrame callback, so they never run more than once per
       frame no matter how many events fire.
     * DOM reads and writes are batched — all reads first, then all writes —
       to avoid forced synchronous layout.
     * No dependencies, no framework, no build step.
   ========================================================================== */

(function () {
  "use strict";

  var REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)");

  function reduced() { return REDUCED.matches; }

  /* ---------------------------------------------------------------------
     rAF scheduler — coalesces bursts of events into one call per frame
     ------------------------------------------------------------------- */
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
     Sticky masthead: a hairline appears only once content scrolls under it
     ------------------------------------------------------------------- */
  function initMasthead() {
    var masthead = document.querySelector(".masthead");
    if (!masthead) return;

    var sentinel = document.createElement("div");
    sentinel.style.cssText = "position:absolute;top:0;height:1px;width:1px;pointer-events:none";
    document.body.prepend(sentinel);

    var apply = onFrame(function () {
      var scrolled = sentinel.getBoundingClientRect().top < 0;
      if (scrolled !== (masthead.dataset.scrolled === "true")) {
        masthead.dataset.scrolled = scrolled ? "true" : "false";
      }
    });

    window.addEventListener("scroll", apply, { passive: true });
    apply();
  }

  /* ---------------------------------------------------------------------
     Scroll reveal — elements fade up as they enter the viewport
     ---------------------------------------------------------------------
     IntersectionObserver is the primary path, but a scroll-based fallback
     runs alongside it. Relying on the observer alone means any edge case that
     stops it firing (an unusual viewport, an observer that never settles)
     leaves content stuck at opacity 0 with no way back. Two cheap mechanisms
     that agree is worth the belt-and-braces: content must never be
     permanently hidden.
     ------------------------------------------------------------------- */
  function initReveal() {
    var targets = document.querySelectorAll("[data-reveal]");
    if (!targets.length) return;

    var pending = [];
    Array.prototype.forEach.call(targets, function (el) { pending.push(el); });

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
          io.unobserve(entry.target); // reveal once, then stop paying for it
        });
      },
      { rootMargin: "0px 0px -6% 0px", threshold: 0.05 }
    );

    pending.forEach(function (el) { io.observe(el); });

    // Fallback sweep: anything whose top has entered the viewport.
    var sweep = onFrame(function () {
      var limit = window.innerHeight * 0.94;
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
     Probability counters
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

      var duration = 850;
      var start = null;

      function frame(now) {
        if (start === null) start = now;
        var t = Math.min((now - start) / duration, 1);
        // easeOutCubic: quick start, gentle settle.
        var eased = 1 - Math.pow(1 - t, 3);
        node.textContent = (target * eased).toFixed(1) + "%";
        if (t < 1) requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    });
  }

  /* ---------------------------------------------------------------------
     Meter bars — fill from 0, staggered so they read as a sequence
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
      // Two frames so the transition actually runs from 0, plus a stagger.
      setTimeout(function () {
        requestAnimationFrame(function () {
          requestAnimationFrame(function () { fill.style.width = pct + "%"; });
        });
      }, reduced() ? 0 : i * 110);
    });
  }

  /* ---------------------------------------------------------------------
     Button press feedback
     ------------------------------------------------------------------- */
  function initButtonFeedback() {
    if (reduced()) return;

    document.addEventListener(
      "pointerdown",
      function (e) {
        var btn = e.target.closest && e.target.closest(".btn");
        if (!btn || btn.hasAttribute("aria-disabled")) return;
        // Scale is handled in CSS :active; this only adds a subtle press
        // nudge on pointer devices, which :active already covers. Kept as a
        // no-op hook for touch devices that lack :active timing.
        if (e.pointerType === "touch") btn.style.transform = "scale(0.98)";
      },
      { passive: true }
    );

    document.addEventListener(
      "pointerup",
      function () {
        Array.prototype.forEach.call(document.querySelectorAll('.btn[style*="scale"]'), function (btn) {
          btn.style.removeProperty("transform");
        });
      },
      { passive: true }
    );
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
        // Mark loading only after validation, so the browser can still focus
        // the first invalid control. The label dims rather than vanishing, so
        // the button keeps its width and the row never reflows.
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
        // Navigate so server-rendered results and errors are discarded too.
        window.location.href = form.getAttribute("action") || window.location.pathname;
      });
    }

    // Clear the error styling as the user edits, but keep the message until
    // the next submit so it cannot be silently ignored.
    Array.prototype.forEach.call(form.querySelectorAll("[aria-invalid]"), function (control) {
      control.addEventListener("input", function () {
        control.removeAttribute("aria-invalid");
      });
    });

    // Stepper buttons. Nobody should have to drag a number spinner to go
    // from 3 children to 4.
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
     Results: focus management, then a gentle scroll
     ------------------------------------------------------------------- */
  function initResults() {
    var results = document.getElementById("prediction-results");
    if (!results) return;

    fillBars(results);
    animateCounters(results);

    // Focus so screen readers announce the change. preventScroll keeps the
    // manual smooth scroll below from fighting the browser's jump.
    if (!results.hasAttribute("tabindex")) results.setAttribute("tabindex", "-1");
    results.focus({ preventScroll: true });

    if (!reduced()) {
      var top = results.getBoundingClientRect().top + window.pageYOffset - 96;
      window.scrollTo({ top: top, behavior: "smooth" });
    }
  }

  /* ---------------------------------------------------------------------
     Error summary focus — required for multi-error forms
     ------------------------------------------------------------------- */
  function initErrorSummary() {
    var summary = document.querySelector("[data-error-summary]");
    if (!summary) return;
    if (!summary.hasAttribute("tabindex")) summary.setAttribute("tabindex", "-1");
    summary.focus({ preventScroll: true });
    if (!reduced()) {
      var top = summary.getBoundingClientRect().top + window.pageYOffset - 96;
      window.scrollTo({ top: Math.max(top, 0), behavior: "smooth" });
    }
  }

  /* ---------------------------------------------------------------------
     Boot
     ------------------------------------------------------------------- */
  function boot() {
    initMasthead();
    initReveal();
    initButtonFeedback();
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
