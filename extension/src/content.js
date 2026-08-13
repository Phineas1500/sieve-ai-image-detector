// Content script: finds images, requests analysis, renders badges + blur.

(() => {
  const MIN_SIZE = 64; // skip icons
  const seen = new WeakSet();
  const badges = new Map(); // img -> {badge, score}
  let settings = { enabled: true, threshold: 0.65, blur: true, minSize: 96 };
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
  });

  function eligible(img) {
    if (!img.currentSrc) return false;
    if (img.currentSrc.startsWith("blob:")) return false; // cannot fetch cross-context
    const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
    return Math.min(w, h) >= Math.max(MIN_SIZE, settings.minSize);
  }

  function positionBadge(img, badge) {
    const r = img.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) {
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
    const isAI = score >= settings.threshold;
    st.badge.textContent = isAI ? `AI ${pct}%` : `${pct}%`;
    st.badge.classList.toggle("aid-flagged", isAI);
    st.badge.title = isAI
      ? `Likely AI-generated (confidence ${pct}%). Click to toggle blur.`
      : `Likely real (AI confidence ${pct}%)`;
    img.classList.toggle("aid-blurred", isAI && settings.blur);
    positionBadge(img, st.badge);
  }

  async function analyze(img) {
    if (seen.has(img) || !eligible(img)) return;
    seen.add(img);
    const url = img.currentSrc;
    try {
      const resp = await chrome.runtime.sendMessage({ kind: "aid:analyze", url });
      if (resp && typeof resp.score === "number") {
        analyzedCount++;
        if (resp.score >= settings.threshold) flaggedCount++;
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

  function init() {
    if (!settings.enabled) return;
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

    let ticking = false;
    const reposition = () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        for (const [img, st] of badges) positionBadge(img, st.badge);
        ticking = false;
      });
    };
    window.addEventListener("scroll", reposition, { passive: true });
    window.addEventListener("resize", reposition, { passive: true });
    setInterval(reposition, 1500); // catch layout shifts
  }
})();
