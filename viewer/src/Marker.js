import { CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

// Two CSS2DObjects per annotation:
//   dotObject      — hollow dot + connector line + static dim title, anchored at center
//   hoverLabelObject — full label, anchored at box-top; slides in on hover
export class Marker {
  constructor(annotation, { onHoverStart, onHoverEnd }) {
    this._forced = false;

    // ── dot + connector + static title ───────────────────────────────────
    const markerEl = document.createElement("div");
    markerEl.className = "marker";
    markerEl.style.setProperty("--dot-color", annotation.color);

    const title = document.createElement("div");
    title.className = "marker-title";
    title.textContent = annotation.label;

    const line = document.createElement("div");
    line.className = "marker-line";

    const dot = document.createElement("div");
    dot.className = "marker-dot";
    // No inline background — hollow style is fully CSS-driven via --dot-color.

    // Inner wrapper: CSS2DRenderer sets transform on markerEl; we animate this child
    // independently so there's no transform conflict.
    const inner = document.createElement("div");
    inner.className = "marker-inner";
    inner.appendChild(title);
    inner.appendChild(line);
    inner.appendChild(dot);
    markerEl.appendChild(inner);

    this.dotObject = new CSS2DObject(markerEl);
    this.dotObject.position.copy(annotation.position);
    this._markerEl = markerEl;
    this._markerInner = inner;
    this._titleEl = title;
    // Pre-computed size estimate for overlap detection — avoids per-frame getBoundingClientRect.
    // 7.2px per char (10px monospace + letter-spacing) + 18px padding/border; 34px tall.
    this._approxSize = { w: annotation.label.length * 7.2 + 18, h: 34 };

    // ── hover label anchored at box top-center ────────────────────────────
    const anchorEl = document.createElement("div");
    anchorEl.className = "hover-label-anchor";
    anchorEl.style.setProperty("--dot-color", annotation.color);

    const hoverLabel = document.createElement("div");
    hoverLabel.className = "hover-label";
    hoverLabel.textContent = annotation.label;
    anchorEl.appendChild(hoverLabel);

    this.hoverLabelObject = new CSS2DObject(anchorEl);
    this.hoverLabelObject.position.set(
      annotation.position.x,
      annotation.position.y + annotation.size.y / 2,
      annotation.position.z,
    );
    this._hoverAnchorEl = anchorEl;
    this._hoverLabelEl = hoverLabel;

    // ── hover events ──────────────────────────────────────────────────────
    markerEl.addEventListener("mouseenter", () => {
      title.classList.add("faded");
      hoverLabel.classList.add("visible");
      onHoverStart();
    });
    markerEl.addEventListener("mouseleave", () => {
      title.classList.remove("faded");
      hoverLabel.classList.remove("visible");
      onHoverEnd();
    });
  }

  get element() { return this._markerEl; }
  get approxSize() { return this._approxSize; }

  suppress(hidden) {
    if (this._forced) return;
    const v = hidden ? "hidden" : "";
    this._markerEl.style.visibility = v;
    this._hoverAnchorEl.style.visibility = v;
    if (hidden) {
      this._hoverLabelEl.classList.remove("visible");
      this._titleEl.classList.remove("faded");
    }
  }

  forceHide() {
    this._forced = true;
    this._markerEl.style.visibility = "hidden";
    this._hoverAnchorEl.style.visibility = "hidden";
    this._markerInner.classList.remove("marker-popping");
  }

  popIn() {
    this._forced = true;
    this._markerEl.style.visibility = "";
    this._hoverAnchorEl.style.visibility = "";
    this._markerInner.classList.remove("marker-popping");
    void this._markerInner.offsetWidth;
    this._markerInner.classList.add("marker-popping");
  }

  clearForce() {
    this._forced = false;
    this._markerInner.classList.remove("marker-popping");
  }
}
