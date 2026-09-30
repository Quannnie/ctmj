/* ==========================================================================
   CJPS - progressive enhancement
   --------------------------------------------------------------------------
   Every feature here is an addition to a page that already works without it.
   The form submits, the results render, the bars show their values and the
   form reports its own errors as plain server-rendered HTML. Script only
   makes those things faster or nicer to operate.

   What it deliberately does not do: nothing here is required to read a
   result, and nothing animated here is load-bearing. The probability bars
   carry their final width in a --final custom property, so with this file
   blocked they are already correct rather than empty.
   ========================================================================== */

(function () {
  "use strict";

  var REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)");
  var STORAGE_KEY = "cjps-theme";

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
     Theme
     ---------------------------------------------------------------------
     Dark is the resolved default, so the buttons show the mode you would
     switch *to*. The stored choice wins; the pre-paint script in base.html has
     already set data-theme by the time this runs, so there is no flash and no
     second read of the media query.

     A write failure (private mode, storage disabled) leaves the toggle inert
     for the session rather than throwing — the page still works, it just
     forgets. The button is also disabled in that case so it does not look
     broken-but-clickable.
     ------------------------------------------------------------------- */
  function initTheme() {
    var root = document.documentElement;
    var buttons = document.querySelectorAll("[data-theme-set]");
    if (!buttons.length) return;

    function current() { return root.getAttribute("data-theme") === "light" ? "light" : "dark"; }

    function sync() {
      var mode = current();
      Array.prototype.forEach.call(buttons, function (btn) {
        btn.setAttribute("aria-pressed", btn.getAttribute("data-theme-set") === mode ? "true" : "false");
      });
    }

    function apply(mode) {
      root.setAttribute("data-theme", mode);
      try {
        localStorage.setItem(STORAGE_KEY, mode);
      } catch (e) {
        /* Storage unavailable: the choice holds for this page view only. */
      }
      sync();
    }

    /* Each button sets the mode it names.
       An earlier version made a click *toggle* whatever the current mode was,
       so pressing "dark mode" while already dark turned the page light. That
       is defensible only if the buttons are affordances rather than labels --
       and they carry aria-label and title, so they are labels. A labelled
       control that does the opposite of its label is worse than no control. */
    Array.prototype.forEach.call(buttons, function (btn) {
      btn.addEventListener("click", function () {
        apply(btn.getAttribute("data-theme-set"));
      });
    });

    /* Follow the system while the user has expressed no preference of their
       own. Once they pick a theme this stops, because an explicit choice
       outranks an OS setting that may only look like a choice. */
    var scheme = window.matchMedia("(prefers-color-scheme: light)");
    var onScheme = function (event) {
      var stored = null;
      try { stored = localStorage.getItem(STORAGE_KEY); } catch (e) { stored = null; }
      if (stored === "light" || stored === "dark") return;
      root.setAttribute("data-theme", event.matches ? "light" : "dark");
      sync();
    };
    if (scheme.addEventListener) scheme.addEventListener("change", onScheme);

    sync();
  }

  /* ---------------------------------------------------------------------
     Scroll reveal
     ---------------------------------------------------------------------
     IntersectionObserver with a rAF-throttled scroll sweep as a backstop, so
     content can never be stranded at opacity 0 if the observer fails to fire.
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

    /* Safety net, and it exists because this shipped broken without it.
       The observer only fires for elements that scroll into view, and the
       sweep only runs on a scroll or resize event. So a page that is never
       scrolled -- a full-page screenshot, a print, a headless render, or a
       reader who simply reaches the bottom by keyboard without generating a
       scroll -- leaves everything below the fold at opacity 0. That is
       content loss, not a cosmetic flaw, and it is not something a
       visual check catches on the sections that are still working.

       Bounded and delayed so it is never the thing that reveals a section in
       normal use: the observer beats it comfortably, and by the time this
       fires the page has had time to be read. */
    setTimeout(function () {
      pending.forEach(reveal);
      pending.length = 0;
    }, 2000);
  }

  /* ---------------------------------------------------------------------
     Result bars
     ---------------------------------------------------------------------
     CSS renders the bar at its --final width when this file is absent, and at
     zero when the js class is present. So the job here is only to move it from
     zero to --final, staggered, and to skip the movement entirely under
     prefers-reduced-motion rather than merely shortening it.
     ------------------------------------------------------------------- */
  function fillBars(root) {
    /* Two kinds of bar, same mechanic: the per-channel meters carry their
       width in data-fill, the single distribution bar in a --share custom
       property because it is one element rather than a list. */
    var fills = root.querySelectorAll(".meter__fill, .mass__fill");
    if (!fills.length) return;

    Array.prototype.forEach.call(fills, function (fill, i) {
      var pct = fill.getAttribute("data-fill");
      if (pct === null) {
        var share = getComputedStyle(fill).getPropertyValue("--share");
        pct = parseFloat(share);
      }

      if (isNaN(pct)) return;

      if (reduced()) {
        fill.style.width = pct + "%";
        return;
      }

      var paint = function () { fill.style.width = pct + "%"; };
      /* requestAnimationFrame alone batches to the next frame, but a short
         setTimeout staggers the bars so they read as a sequence rather than
         one block. The first moves immediately. */
      if (i === 0) requestAnimationFrame(paint);
      else setTimeout(paint, i * 60);
    });
  }

  /* ---------------------------------------------------------------------
     Probability counters
     ---------------------------------------------------------------------
     600ms, easeOutCubic.
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
     Form completion meter
     ---------------------------------------------------------------------
     Counts controls that hold a value, not just ones the user has touched. A
     select repopulated after a failed POST is genuinely filled, and telling
     the analyst otherwise is the kind of small lie that makes a progress
     indicator get ignored.
     ------------------------------------------------------------------- */
  function initProgress(form) {
    var host = form.querySelector("[data-progress]");
    if (!host) return;

    var countEl = host.querySelector("[data-progress-count]");
    var fillEl = host.querySelector("[data-progress-fill]");
    var hintEl = host.querySelector("[data-progress-hint]");
    var controls = form.querySelectorAll("select[required], input[required]");

    function filled(el) {
      if (el.type === "number") return el.value.trim() !== "";
      return el.value !== "";
    }

    function update() {
      var done = 0;
      Array.prototype.forEach.call(controls, function (el) {
        if (filled(el)) done++;
      });
      var total = controls.length;
      var pct = total ? (done / total) * 100 : 0;

      if (countEl) countEl.textContent = done + " / " + total;
      if (fillEl) fillEl.style.width = pct + "%";
      if (hintEl) {
        hintEl.textContent = done === total ? "đủ 12 trường" : "còn " + (total - done) + " trường";
      }
      host.setAttribute("data-complete", done === total ? "true" : "false");
    }

    var schedule = onFrame(update);
    Array.prototype.forEach.call(controls, function (el) {
      el.addEventListener("input", schedule);
      el.addEventListener("change", schedule);
    });
    update();
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

    /* Stepper buttons: real buttons, so they must honour min/max rather than
       letting the field hold a value the server will reject. */
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

    initProgress(form);
    initShortcuts(form);
  }

  /* ---------------------------------------------------------------------
     Keyboard
     ---------------------------------------------------------------------
     Ctrl/Cmd+Enter submits from anywhere in the form, and "/" jumps to the
     first field that still needs an answer. Twelve required fields is enough
     that tabbing from the top every time is the part of the form that actually
     gets tiring.

     Both are suppressed while a modifier other than the one they name is held,
     so Ctrl+Enter still means "new line" in a text field and "/" still means
     "/" while typing a touchpoint name.
     ------------------------------------------------------------------- */
  function initShortcuts(form) {
    function firstEmpty() {
      var controls = form.querySelectorAll("select[required], input[required]");
      for (var i = 0; i < controls.length; i++) {
        if (controls[i].value === "") return controls[i];
      }
      return null;
    }

    document.addEventListener("keydown", function (event) {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        if (form.contains(document.activeElement) || document.activeElement === document.body) {
          event.preventDefault();
          if (form.requestSubmit) form.requestSubmit();
          else form.submit();
        }
        return;
      }

      if (event.key !== "/" || event.ctrlKey || event.metaKey || event.altKey) return;
      var el = event.target;
      var tag = el && el.tagName ? el.tagName.toLowerCase() : "";
      var typing = tag === "input" || tag === "textarea" || tag === "select" || (el && el.isContentEditable);
      if (typing) return;

      var next = firstEmpty();
      if (!next) return;
      event.preventDefault();
      next.focus();
      if (typeof next.select === "function" && next.type === "text") next.select();
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

    /* On a two-column layout the rail is sticky and already on screen, so
       scrolling to it would jump the form out from under the reader for no
       reason. Only scroll when the rail has actually stacked below. */
    if (window.matchMedia("(min-width: 1080px)").matches) {
      results.focus({ preventScroll: true });
      return;
    }
    focusAndScroll(results);
  }

  function initErrorSummary() {
    var summary = document.querySelector("[data-error-summary]");
    if (summary) focusAndScroll(summary);
  }

  function boot() {
    initTheme();
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
