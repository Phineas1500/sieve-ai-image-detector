// Content script: finds images, requests analysis, renders badges + blur.

(() => {
  const MIN_SIZE = 64; // skip icons
  const seen = new WeakSet();
  const badges = new Map(); // img -> {badge, score}
  let settings = {
    enabled: true, threshold: 0.65, blur: true, minSize: 96, minDisplaySize: 100,
    scanMode: "auto",        // "auto" | "manual" (right-click only)
    badgeDisplay: "all",     // "all" | "flags"  (hide sub-threshold badges)
    flaggedAction: "blur",   // "badge" | "blur" | "hide" (slopblocker)
  };
  let flaggedCount = 0;
  let analyzedCount = 0;

  chrome.storage.sync.get(settings, (s) => {
    settings = { ...settings, ...s };
    init();
  });

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "sync") return;
    for (const [k, v] of Object.entries(changes)) settings[k] = v.newValue;
    // re-apply threshold to existing results
    for (const [img, st] of badges) { if (st.na) applyNotAnalysed(img, st.na); else applyResult(img, st.score, st.degraded, st.quality); }
  });

  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg?.kind === "aid:page-stats") {
      sendResponse({ analyzed: analyzedCount, flagged: flaggedCount });
    }
    if (msg?.kind === "aid:check-image" && msg.srcUrl) {
      // right-click "Check this image": bypass size gates for this one image
      document.querySelectorAll("img").forEach((img) => {
        if (img.currentSrc === msg.srcUrl) analyze(img, true);
      });
    }
  });

  function fetchable(img) {
    return !!img.currentSrc && !img.currentSrc.startsWith("blob:"); // blob: cannot be fetched cross-context
  }

  function eligibleNaturalSize(img) {
    const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
    return Math.min(w, h) >= Math.max(MIN_SIZE, settings.minSize);
  }

  // Avatars and thumbnails are often large source files displayed tiny —
  // badging them clutters feeds. Gate on on-screen size, not file size.
  function displayedLargeEnough(img) {
    const r = img.getBoundingClientRect();
    return Math.min(r.width, r.height) >= settings.minDisplaySize;
  }

  // If a skipped-small image later grows (lightbox, in-place expansion),
  // analyze it then.
  const growthWatched = new WeakSet();
  const growthObserver = new ResizeObserver((entries) => {
    for (const e of entries) {
      const img = e.target;
      if (displayedLargeEnough(img)) {
        growthObserver.unobserve(img);
        analyze(img);
      }
    }
  });

  function watchForGrowth(img) {
    if (growthWatched.has(img)) return;
    growthWatched.add(img);
    growthObserver.observe(img);
  }

  function coveredByStickyBar(img, x, y) {
    // Hide badges for images scrolled behind fixed/sticky page chrome
    // (e.g. Google's search bar) — our z-index would otherwise punch through.
    for (const el of document.elementsFromPoint(x, y)) {
      if (el.classList?.contains("aid-badge")) continue;
      if (el === img || img.contains(el) || el.contains(img)) return false;
      const pos = getComputedStyle(el).position;
      if (pos === "fixed" || pos === "sticky") return true;
      // anything else (same-tile overlays, wrappers) — keep walking down
    }
    return false;
  }

  function coversImage(candidate, imageRect) {
    const r = candidate.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return false;
    const overlapW = Math.max(0, Math.min(r.right, imageRect.right) - Math.max(r.left, imageRect.left));
    const overlapH = Math.max(0, Math.min(r.bottom, imageRect.bottom) - Math.max(r.top, imageRect.top));
    return overlapW * overlapH >= imageRect.width * imageRect.height * 0.8;
  }

  // Some sites (notably X/Twitter) keep the real <img> transparent and paint
  // the same asset through a sibling background-image div. Apply presentation
  // state to those visual twins as well as to the analyzed image.
  function visualTargets(img, previous = []) {
    const targets = [img, ...previous.filter((target) => target !== img && target.isConnected)];
    const parent = img.parentElement;
    if (!parent) return targets;
    const imageRect = img.getBoundingClientRect();
    if (imageRect.width < 1 || imageRect.height < 1) return targets;
    const candidates = [parent, ...parent.querySelectorAll("*")].slice(0, 65);
    for (const candidate of candidates) {
      if (targets.includes(candidate) || !coversImage(candidate, imageRect)) continue;
      const cs = getComputedStyle(candidate);
      if (cs.display === "none" || cs.visibility === "hidden" || cs.backgroundImage === "none") continue;
      targets.push(candidate);
    }
    return targets;
  }

  function effectiveFlaggedAction() {
    const action = settings.flaggedAction;
    // Compatibility for profiles that already stored the old popup's
    // `blur: false` while the newer options defaulted flaggedAction to blur.
    if (action === "blur" && settings.blur === false) return "badge";
    return ["badge", "blur", "hide"].includes(action)
      ? action
      : settings.blur ? "blur" : "badge";
  }

  function syncVisualState(img, st, isAI, action) {
    const blurred = isAI && action === "blur" && !st.revealed;
    const hidden = isAI && action === "hide";
    st.targets = visualTargets(img, st.targets);
    for (const target of st.targets) {
      if (target !== img) target.classList.add("aid-visual-twin");
      target.classList.toggle("aid-blurred", blurred);
      target.classList.toggle("aid-hidden", hidden);
    }
  }

  function positionBadge(img, badge) {
    const r = img.getBoundingClientRect();
    const offscreen =
      r.width === 0 || r.height === 0 ||
      r.bottom < 0 || r.top > window.innerHeight ||
      r.right < 0 || r.left > window.innerWidth;
    // Pages animate image stacks via opacity/visibility (crossfades,
    // scroll-driven zooms) — a laid-out but effectively invisible img must
    // not show a badge, or every frame of the stack badges at once. Check the
    // PARENT's visibility, not the img's own: X/Twitter keeps the real <img>
    // at opacity:0 over a background-image twin that shows the pixels, so the
    // img's own opacity says nothing about what the user sees.
    const anchor = img.parentElement || img;
    const invisible = typeof anchor.checkVisibility === "function" &&
      !anchor.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true,
                                opacityProperty: true, visibilityProperty: true });
    if (offscreen || invisible || coveredByStickyBar(img, r.left + 12, r.top + 12)) {
      badge.style.display = "none";
      return;
    }
    badge.style.display = "";
    badge.style.top = `${window.scrollY + r.top + 6}px`;
    badge.style.left = `${window.scrollX + r.left + 6}px`;
  }

  // A verdict is "flagged" only when the score clears the threshold AND the
  // input wasn't delivered in a degraded regime (heavy recompression or
  // upscaling — audit #41/#42), where a red badge would not be evidence-backed.
  function isFlagged(st) {
    return typeof st.score === "number" && st.score >= settings.threshold && !st.degraded;
  }

  function ensureBadge(img) {
    let st = badges.get(img);
    if (st) return st;
    const badge = document.createElement("div");
    badge.className = "aid-badge";
    badge.addEventListener("click", (e) => {
      e.stopPropagation();
      e.preventDefault();
      // clicking a blurred image reveals it; clicking a revealed one re-blurs
      st.revealed = img.classList.contains("aid-blurred");
      syncVisualState(img, st, isFlagged(st), "blur");
    });
    document.documentElement.appendChild(badge);
    st = { badge, score: null, degraded: false, na: null, revealed: false, action: null, targets: [] };
    badges.set(img, st);
    return st;
  }

  // Three states used to look identical (no badge): not reached yet, analysed
  // and clean, and not analysable. A quiet grey chip separates the last one
  // (audit #49). Only in the "all badges" display mode.
  const NA_TEXT = {
    "too-small": "Image too small to analyse",
    "degenerate-flat": "Flat / solid image — nothing to analyse",
    "degenerate-noise": "Noise-like image — nothing to analyse",
    "error": "Analysis failed repeatedly (see the extension log)",
  };
  function applyNotAnalysed(img, reason) {
    const st = ensureBadge(img);
    st.na = reason; st.score = null; st.degraded = false;
    delete img.dataset.aidScore;
    img.dataset.aidNa = reason; // diagnostics/tests
    st.badge.textContent = "not analysed";
    st.badge.className = "aid-badge aid-na";
    st.badge.title = NA_TEXT[reason] || NA_TEXT.error;
    st.badge.classList.toggle("aid-quiet", settings.badgeDisplay === "flags");
    syncVisualState(img, st, false, effectiveFlaggedAction());
    positionBadge(img, st.badge);
  }

  function applyResult(img, score, degraded = false, quality = null) {
    const st = ensureBadge(img);
    st.score = score; st.degraded = !!degraded; st.quality = quality; st.na = null;
    delete img.dataset.aidNa;
    img.dataset.aidScore = String(score); // exact score, for tests/tooling
    img.dataset.aidDegraded = String(!!degraded);
    if (quality) img.dataset.aidQuality = JSON.stringify(quality); else delete img.dataset.aidQuality; // {block, d12}, for tests/tooling
    const pct = Math.round(score * 100);
    const thr = settings.threshold;
    const flagged = isFlagged(st);
    // Above 50% the model leans AI even when below the flag threshold —
    // show that honestly instead of a bare number that reads as "safe".
    // A degraded input that clears the threshold is shown as unsure too.
    const isUnsure = !flagged && (score >= 0.5 || (score >= thr && st.degraded));
    st.badge.textContent = flagged ? `AI ${pct}%` : isUnsure ? `unsure ${pct}%` : `${pct}%`;
    st.badge.className = "aid-badge";
    st.badge.classList.toggle("aid-flagged", flagged);
    st.badge.classList.toggle("aid-unsure", isUnsure);
    st.badge.classList.toggle("aid-degraded", st.degraded && score >= thr);
    st.badge.classList.toggle("aid-quiet", !flagged && settings.badgeDisplay === "flags");
    st.badge.title = flagged
      ? (score < thr + 0.10
          ? `Likely AI-generated (confidence ${pct}% — near the ${Math.round(thr * 100)}% threshold: about 1 in 8 verdicts in this band are wrong on our benchmark). Click to toggle blur.`
          : `Likely AI-generated (confidence ${pct}%). Click to toggle blur.`)
      : st.degraded && score >= thr
        ? `Low-quality input (heavy recompression or upscaling) — AI confidence ${pct}%, shown as unsure because degraded images are not evidence-backed`
        : isUnsure
          ? `Uncertain — AI confidence ${pct}%, below the ${Math.round(thr * 100)}% flag threshold`
          : `Likely real (AI confidence ${pct}%)`;
    const act = effectiveFlaggedAction();
    if (st.action !== act) st.revealed = false;
    st.action = act;
    syncVisualState(img, st, flagged, act);
    positionBadge(img, st.badge);
  }

  async function analyze(img, force = false) {
    if (seen.has(img) || !fetchable(img)) return;
    if (!force) {
      if (!eligibleNaturalSize(img)) return;
      if (!displayedLargeEnough(img)) {
        watchForGrowth(img);
        return;
      }
    }
    seen.add(img);
    const url = img.currentSrc;
    try {
      const resp = await chrome.runtime.sendMessage({ kind: "aid:analyze", url });
      if (resp && typeof resp.score === "number") {
        analyzedCount++;
        if (resp.score >= settings.threshold && !resp.degraded) flaggedCount++;
        failures.delete(img);
        img.dataset.aidTta = String(!!resp.tta); // diagnostics/tests
        if (typeof resp.ms === "number") img.dataset.aidMs = String(resp.ms); // inference time, for latency tests
        applyResult(img, resp.score, !!resp.degraded, resp.quality || null);
      } else if (resp && resp.error && permanentError(resp.error)) {
        applyNotAnalysed(img, String(resp.error));
      } else if (resp && resp.error) {
        // Transient failure (fetch hiccup, GPU glitch): without a retry the
        // image stays in `seen` and is silently never analyzed again.
        if (!scheduleRetry(img)) applyNotAnalysed(img, "error");
      }
    } catch {
      // service worker restarting; will retry on next intersection
      seen.delete(img);
    }
  }

  // Errors that re-running cannot fix — retrying would loop forever.
  function permanentError(err) {
    return err === "too-small" || String(err).startsWith("degenerate");
  }

  const failures = new WeakMap(); // img -> consecutive transient failures
  function scheduleRetry(img) {
    const n = (failures.get(img) || 0) + 1;
    failures.set(img, n);
    if (n > 3) return false; // give up: three strikes with backoff
    setTimeout(() => {
      seen.delete(img);
      if (img.isConnected && fetchable(img)) analyze(img);
    }, 3000 * n);
    return true;
  }

  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          const img = e.target;
          if (img.complete && img.naturalWidth) analyze(img);
          else img.addEventListener("load", () => analyze(img), { once: true });
        }
      }
    },
    { rootMargin: "256px" }
  );

  function observeAll(root) {
    root.querySelectorAll?.("img").forEach((img) => io.observe(img));
  }

  let repositionStarted = false;
  function startRepositionLoop() {
    if (repositionStarted) return;
    repositionStarted = true;
    let ticking = false;
    const reposition = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        for (const [img, st] of badges) {
          // Feeds remove/virtualize images as you scroll; reclaim their badges
          // or they accumulate forever. Dropping from `seen` lets a re-attached
          // node re-analyze (cache makes that instant) — self-healing.
          if (!img.isConnected) {
            st.badge.remove();
            badges.delete(img);
            seen.delete(img);
            continue;
          }
          const stAction = effectiveFlaggedAction();
          syncVisualState(img, st, isFlagged(st), stAction);
          positionBadge(img, st.badge);
        }
        ticking = false;
      });
    };
    window.addEventListener("scroll", reposition, { passive: true });
    window.addEventListener("resize", reposition, { passive: true });
    setInterval(reposition, 1500); // catch layout shifts
  }

  function init() {
    if (!settings.enabled) return;
    startRepositionLoop(); // badges need positioning in every mode (incl. right-click results)
    if (settings.scanMode === "manual") return; // no auto-scan; right-click checks still work
    observeAll(document);

    const mo = new MutationObserver((muts) => {
      for (const m of muts) {
        if (m.type === "attributes" && m.target.tagName === "IMG") {
          seen.delete(m.target);
          io.observe(m.target);
        }
        for (const n of m.addedNodes) {
          if (n.nodeType !== 1) continue;
          if (n.tagName === "IMG") io.observe(n);
          else observeAll(n);
        }
      }
    });
    mo.observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["src", "srcset"],
    });
  }
})();
