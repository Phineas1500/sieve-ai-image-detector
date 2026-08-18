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
    for (const [img, st] of badges) applyResult(img, st.score);
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

  function applyResult(img, score) {
    let st = badges.get(img);
    if (!st) {
      const badge = document.createElement("div");
      badge.className = "aid-badge";
      badge.addEventListener("click", (e) => {
        e.stopPropagation();
        e.preventDefault();
        img.classList.toggle("aid-blurred");
      });
      document.documentElement.appendChild(badge);
      st = { badge, score };
      badges.set(img, st);
    }
    st.score = score;
    img.dataset.aidScore = String(score); // exact score, for tests/tooling
    const pct = Math.round(score * 100);
    const thr = settings.threshold;
    const isAI = score >= thr;
    // Above 50% the model leans AI even when below the flag threshold —
    // show that honestly instead of a bare number that reads as "safe".
    const isUnsure = !isAI && score >= 0.5;
    st.badge.textContent = isAI ? `AI ${pct}%` : isUnsure ? `unsure ${pct}%` : `${pct}%`;
    st.badge.classList.toggle("aid-flagged", isAI);
    st.badge.classList.toggle("aid-unsure", isUnsure);
    st.badge.classList.toggle("aid-quiet", !isAI && settings.badgeDisplay === "flags");
    st.badge.title = isAI
      ? `Likely AI-generated (confidence ${pct}%). Click to toggle blur.`
      : isUnsure
        ? `Uncertain — AI confidence ${pct}%, below the ${Math.round(thr * 100)}% flag threshold`
        : `Likely real (AI confidence ${pct}%)`;
    const act = settings.flaggedAction || (settings.blur ? "blur" : "badge");
    img.classList.toggle("aid-blurred", isAI && act === "blur");
    img.classList.toggle("aid-hidden", isAI && act === "hide");
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
        if (resp.score >= settings.threshold) flaggedCount++;
        img.dataset.aidTta = String(!!resp.tta); // diagnostics/tests
        applyResult(img, resp.score);
      }
    } catch {
      // service worker restarting; will retry on next intersection
      seen.delete(img);
    }
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
