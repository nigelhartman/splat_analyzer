import * as THREE from "three";
import { AnnotatedObject } from "./AnnotatedObject.js";

const _corner = new THREE.Vector3();

// Project one world-space point to viewport pixels { x, y }.
function toScreen(worldPos, camera, w, h) {
  _corner.copy(worldPos).project(camera);
  return { x: (_corner.x + 1) / 2 * w, y: (1 - _corner.y) / 2 * h, z: _corner.z };
}

// Manages the live set of AnnotatedObjects, rebuilding cleanly on reload.
export class AnnotationLayer {
  constructor(sceneManager, hooks = {}) {
    this.sceneManager = sceneManager;
    this.hooks = hooks;
    this.objects = [];
    this._lastUpdate = 0;
    this._revealing = false;
    this._revealTimers = [];
    sceneManager.addResizeListener((w, h) => {
      for (const o of this.objects) o.box.setResolution(w, h);
    });
    sceneManager.addFrameListener((camera) => this._update(camera));
  }

  rebuild(annotations) {
    this.clear();
    for (const annotation of annotations) {
      const obj = new AnnotatedObject(annotation, this.hooks);
      obj.addTo(this.sceneManager);
      this.objects.push(obj);
    }
  }

  clear() {
    this._cancelReveal();
    for (const o of this.objects) o.removeFrom(this.sceneManager);
    this.objects = [];
  }

  playReveal() {
    if (this._revealing || this.objects.length === 0) return;
    this._revealing = true;

    for (const obj of this.objects) obj.marker.forceHide();

    const count = this.objects.length;
    const interval = 2500 / count;
    const objects = this.objects;

    this._revealTimers.push(setTimeout(() => {
      objects.forEach((obj, i) => {
        this._revealTimers.push(setTimeout(() => {
          if (this.objects.includes(obj)) obj.marker.popIn();
        }, i * interval));
      });
      this._revealTimers.push(setTimeout(() => {
        for (const obj of objects) {
          if (this.objects.includes(obj)) obj.marker.clearForce();
        }
        this._revealing = false;
        this._revealTimers = [];
      }, count * interval + 400));
    }, 1000));
  }

  _cancelReveal() {
    for (const id of this._revealTimers) clearTimeout(id);
    this._revealTimers = [];
    this._revealing = false;
  }

  _update(camera) {
    if (this.objects.length === 0) return;

    // Throttle to ~30fps — this is a cosmetic overlay that doesn't need per-frame precision.
    const now = performance.now();
    if (now - this._lastUpdate < 33) return;
    this._lastUpdate = now;

    const W = window.innerWidth;
    const H = window.innerHeight;

    // ── 1. Compute screen-space AABBs of any currently-visible bounding boxes ──
    const boxRects = [];
    for (const obj of this.objects) {
      if (!obj.box.group.visible) continue;
      const { position, size, quaternion } = obj.annotation;
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      let anyVisible = false;
      for (let sx = -0.5; sx <= 0.5; sx += 1)
      for (let sy = -0.5; sy <= 0.5; sy += 1)
      for (let sz = -0.5; sz <= 0.5; sz += 1) {
        const wp = _corner.set(sx * size.x, sy * size.y, sz * size.z)
          .applyQuaternion(quaternion)
          .add(position);
        const s = toScreen(wp, camera, W, H);
        if (s.z < 1) { // in front of camera
          minX = Math.min(minX, s.x); maxX = Math.max(maxX, s.x);
          minY = Math.min(minY, s.y); maxY = Math.max(maxY, s.y);
          anyVisible = true;
        }
      }
      if (anyVisible) boxRects.push({ minX, maxX, minY, maxY, owner: obj });
    }

    // ── 2. Depth-sort markers (closest = highest priority) ──────────────────
    // Use approxSize instead of getBoundingClientRect to avoid forced layout reflows
    // (CSS2DRenderer writes transforms, then reading rects would flush the whole tree).
    const entries = this.objects.map(obj => {
      const s = toScreen(obj.annotation.position, camera, W, H);
      return { obj, z: s.z, screenX: s.x, screenY: s.y };
    });
    entries.sort((a, b) => a.z - b.z);

    // ── 3. Greedy label overlap prevention ──────────────────────────────────
    const EXPAND = 8;
    const COVER = 0.55; // hide if > 55% of farther marker's area is covered
    const shownRects = [];

    for (const entry of entries) {
      const { obj, screenX, screenY } = entry;

      // Check if this marker's center falls inside a visible box that isn't its own.
      let hiddenByBox = false;
      for (const br of boxRects) {
        if (br.owner === obj) continue;
        if (screenX >= br.minX && screenX <= br.maxX &&
            screenY >= br.minY && screenY <= br.maxY) {
          hiddenByBox = true;
          break;
        }
      }

      // Check if any already-shown marker significantly covers this one.
      let hiddenByOverlap = false;
      if (!hiddenByBox) {
        const { w, h } = obj.marker.approxSize;
        const ex = {
          left: screenX - w / 2 - EXPAND, right: screenX + w / 2 + EXPAND,
          top: screenY - h / 2 - EXPAND, bottom: screenY + h / 2 + EXPAND,
          width: w + EXPAND * 2, height: h + EXPAND * 2,
        };
        const area = ex.width * ex.height;
        for (const s of shownRects) {
          const ix = Math.min(s.right, ex.right) - Math.max(s.left, ex.left);
          const iy = Math.min(s.bottom, ex.bottom) - Math.max(s.top, ex.top);
          if (ix > 0 && iy > 0 && ix * iy / area > COVER) {
            hiddenByOverlap = true;
            break;
          }
        }
        if (!hiddenByOverlap) {
          shownRects.push({
            left: ex.left, right: ex.right,
            top: ex.top, bottom: ex.bottom,
          });
        }
      }

      obj.marker.suppress(hiddenByBox || hiddenByOverlap);
    }
  }
}
