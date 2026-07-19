import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { SparkRenderer } from "@sparkjsdev/spark";
import { CSS2DRenderer } from "three/addons/renderers/CSS2DRenderer.js";

// Splat rendering is fill-rate bound, so two knobs dominate performance:
// the render resolution (pixel ratio) and the LOD splat budget. Both are
// capped conservatively here; raise them for sharper output on strong GPUs.
const MAX_PIXEL_RATIO = 1.25;
const LOD_SPLAT_SCALE = 0.35; // fraction of the device's default LOD budget

// While the camera moves we drop to a lower resolution (motion hides the
// softness) and snap back to full sharpness once it settles.
const DRAG_PIXEL_RATIO = 0.75;
const SETTLE_MS = 180;

// Re-sorting millions of splats every frame is wasteful; ~30ms (≈33Hz) is
// imperceptible while cutting the per-frame sort cost.
const SORT_INTERVAL_MS = 30;

// Owns the entire render pipeline: scene, camera, renderers, controls and the
// animation loop. Nothing else in the app touches THREE's rendering directly.
export class SceneManager {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();

    this.camera = new THREE.PerspectiveCamera(60, this._aspect(), 0.01, 1000);
    this.camera.position.set(0, 2, 6);

    // antialias off: MSAA does nothing for alpha-blended splats and the wide
    // lines do their own edge smoothing, so it would only cost fill rate.
    this.renderer = new THREE.WebGLRenderer({ antialias: false });
    this._stillRatio = Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO);
    this._dragRatio = Math.min(window.devicePixelRatio, DRAG_PIXEL_RATIO);
    this._pixelRatio = this._stillRatio;
    this.renderer.setPixelRatio(this._stillRatio);
    this.renderer.setSize(this._width(), this._height());
    container.appendChild(this.renderer.domElement);

    this.labelRenderer = new CSS2DRenderer();
    this.labelRenderer.setSize(this._width(), this._height());
    this.labelRenderer.domElement.classList.add("css2d-layer");
    this.labelRenderer.domElement.style.position = "absolute";
    this.labelRenderer.domElement.style.top = "0";
    this.labelRenderer.domElement.style.left = "0";
    container.appendChild(this.labelRenderer.domElement);

    const spark = new SparkRenderer({
      renderer: this.renderer,
      lodSplatScale: LOD_SPLAT_SCALE,
      minSortIntervalMs: SORT_INTERVAL_MS,
    });
    this.scene.add(spark);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    // Render at reduced resolution during interaction, restore when it stops.
    this.controls.addEventListener("change", () => this._onCameraMove());

    this._resizeListeners = [];
    this._frameListeners = [];
    this.fps = 0;
    this._fpsFrames = 0;
    this._fpsLast = performance.now();
    window.addEventListener("resize", () => this._onResize());
  }

  add(object) { this.scene.add(object); }
  remove(object) { this.scene.remove(object); }

  addResizeListener(fn) { this._resizeListeners.push(fn); }
  addFrameListener(fn) { this._frameListeners.push(fn); }

  // Frame the camera and orbit target to a world-space bounding box.
  frameBounds(box) {
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1;

    this.controls.target.copy(center);
    const dir = new THREE.Vector3(0.4, 0.5, 1).normalize();
    this.camera.position.copy(center).addScaledVector(dir, maxDim * 1.8);
    this.camera.near = maxDim / 100;
    this.camera.far = maxDim * 100;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  start() {
    const loop = () => {
      requestAnimationFrame(loop);
      this._fpsFrames += 1;
      const now = performance.now();
      if (now - this._fpsLast >= 500) {
        this.fps = (this._fpsFrames * 1000) / (now - this._fpsLast);
        this._fpsFrames = 0;
        this._fpsLast = now;
      }
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      for (const fn of this._frameListeners) fn(this.camera);
      this.labelRenderer.render(this.scene, this.camera);
    };
    loop();
  }

  _onCameraMove() {
    if (this._pixelRatio !== this._dragRatio) this._setPixelRatio(this._dragRatio);
    clearTimeout(this._settleTimer);
    this._settleTimer = setTimeout(
      () => this._setPixelRatio(this._stillRatio),
      SETTLE_MS
    );
  }

  _setPixelRatio(ratio) {
    if (this._pixelRatio === ratio) return;
    this._pixelRatio = ratio;
    this.renderer.setPixelRatio(ratio);
    this.renderer.setSize(this._width(), this._height());
  }

  _onResize() {
    this.camera.aspect = this._aspect();
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this._width(), this._height());
    this.labelRenderer.setSize(this._width(), this._height());
    for (const fn of this._resizeListeners) fn(this._width(), this._height());
  }

  _width() { return this.container.clientWidth; }
  _height() { return this.container.clientHeight; }
  _aspect() { return this._width() / this._height(); }
}
