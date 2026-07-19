import * as THREE from "three";
import {
  SplatMesh,
  SplatEdit,
  SplatEditSdf,
  SplatEditSdfType,
  SplatEditRgbaBlendMode,
} from "@sparkjsdev/spark";

// Faint additive RGB applied to splats inside a hovered object's box.
const HIGHLIGHT_RGB = new THREE.Color(0.16, 0.16, 0.16);

async function readBodyWithProgress(body, total, onProgress, signal) {
  if (!body) throw new Error("Empty response body");
  const reader = body.getReader();
  const onAbort = () => reader.cancel("aborted");
  signal?.addEventListener("abort", onAbort);
  try {
    const chunks = [];
    let received = 0;

    while (true) {
      if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      onProgress?.(received, total);
    }

    const buffer = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return buffer.buffer;
  } finally {
    signal?.removeEventListener("abort", onAbort);
  }
}

function parseRadSplatCount(bytes) {
  if (bytes.length < 8) return null;
  // magic "RAD0"
  if (bytes[0] !== 0x52 || bytes[1] !== 0x41 || bytes[2] !== 0x44 || bytes[3] !== 0x30) {
    return null;
  }
  const jsonLen = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(4, true);
  if (jsonLen <= 0 || 8 + jsonLen > bytes.length) return null;
  try {
    const meta = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + jsonLen)));
    return Number.isFinite(meta.count) ? meta.count : null;
  } catch {
    return null;
  }
}

function disposeMesh(mesh) {
  if (!mesh) return;
  try {
    mesh.dispose?.();
  } catch {
    // Spark dispose can throw if partially initialized; ignore.
  }
}

// Loads and owns the SplatMesh. The π rotation about X here defines the
// viewer's world space; AnnotationParser mirrors it so boxes line up.
export class SplatView {
  constructor(sceneManager) {
    this.sceneManager = sceneManager;
    this.mesh = null;
    this.numSplats = 0;
    this._loadId = 0;
    this._abort = null;

    // One reusable edit: a box SDF re-placed over whichever object is hovered.
    // Lives in world space (Spark reads the SDF's matrixWorld), matching the
    // bounding boxes which are added to the scene at the same world transform.
    this._highlightSdf = new SplatEditSdf({
      type: SplatEditSdfType.BOX,
      color: HIGHLIGHT_RGB,
      opacity: 0, // leave alpha untouched; only add brightness
    });
    this._highlightEdit = new SplatEdit({
      rgbaBlendMode: SplatEditRgbaBlendMode.ADD_RGBA,
      sdfs: [this._highlightSdf],
    });
    this._highlighted = null;
  }

  clear() {
    this._invalidateLoads();
    this._removeMesh();
  }

  /**
   * @param {string | File} source
   * @param {{
   *   onProgress?: (loaded: number, total: number | null) => void,
   *   onProcessing?: () => void,
   * }} [opts]
   */
  async load(source, { onProgress, onProcessing } = {}) {
    // Drop the current splat immediately and cancel any in-flight load.
    this._invalidateLoads();
    this._removeMesh();

    const loadId = this._loadId;
    const abort = new AbortController();
    this._abort = abort;
    const { signal } = abort;

    const isUrl = typeof source === "string";
    const fileName = isUrl ? source.split("/").pop() : source.name;
    let buffer;

    try {
      if (isUrl) {
        const res = await fetch(source, { signal });
        if (!res.ok) {
          throw new Error(`Failed to fetch ${fileName}: ${res.status} ${res.statusText}`);
        }
        const total = Number(res.headers.get("Content-Length")) || null;
        onProgress?.(0, total);
        buffer = await readBodyWithProgress(res.body, total, onProgress, signal);
      } else {
        const total = source.size || null;
        onProgress?.(0, total);
        if (source.stream) {
          buffer = await readBodyWithProgress(source.stream(), total, onProgress, signal);
        } else {
          buffer = await source.arrayBuffer();
          onProgress?.(buffer.byteLength, total ?? buffer.byteLength);
        }
      }
    } catch (err) {
      if (err?.name === "AbortError" || loadId !== this._loadId) return;
      throw err;
    }

    if (loadId !== this._loadId) return;

    onProcessing?.();
    const bytes = new Uint8Array(buffer);
    // .rad files carry a pre-baked LOD tree (built with `npm run build-lod`);
    // any other format gets an LOD tree generated on-the-fly in a worker.
    const isBaked = fileName.toLowerCase().endsWith(".rad");
    const radCount = isBaked ? parseRadSplatCount(bytes) : null;

    return new Promise((resolve) => {
      if (loadId !== this._loadId) {
        resolve();
        return;
      }

      const mesh = new SplatMesh({
        fileBytes: bytes,
        fileName,
        lod: isBaked ? undefined : true,
        onLoad: (loaded) => {
          if (loadId !== this._loadId) {
            this.sceneManager.remove(loaded);
            disposeMesh(loaded);
            resolve();
            return;
          }
          loaded.rotation.x = Math.PI;
          loaded.edits = [];
          loaded.updateMatrixWorld(true);
          this.numSplats =
            radCount ??
            loaded.numSplats ??
            loaded.packedSplats?.numSplats ??
            loaded.splats?.numSplats ??
            0;
          const box = new THREE.Box3().setFromObject(loaded);
          this.sceneManager.frameBounds(box);
          resolve();
        },
      });

      if (loadId !== this._loadId) {
        disposeMesh(mesh);
        resolve();
        return;
      }

      this.mesh = mesh;
      this.sceneManager.add(mesh);
    });
  }

  // Brighten the splats inside one annotation's box, fading out over the
  // outer ~10% of the box so the core is uniform and the borders feather.
  highlight(annotation) {
    if (!this.mesh) return;
    this._highlighted = annotation;

    const sdf = this._highlightSdf;
    sdf.position.copy(annotation.position);
    sdf.quaternion.copy(annotation.quaternion);
    sdf.scale.set(
      annotation.size.x * 0.5,
      annotation.size.y * 0.5,
      annotation.size.z * 0.5
    );
    const minHalf = Math.min(sdf.scale.x, sdf.scale.y, sdf.scale.z);
    this._highlightEdit.softEdge = minHalf * 0.1;

    this.mesh.edits = [this._highlightEdit];
  }

  clearHighlight(annotation) {
    if (annotation && this._highlighted !== annotation) return;
    this._highlighted = null;
    if (this.mesh) this.mesh.edits = [];
  }

  _invalidateLoads() {
    this._loadId += 1;
    if (this._abort) {
      this._abort.abort();
      this._abort = null;
    }
  }

  _removeMesh() {
    if (this.mesh) {
      this.sceneManager.remove(this.mesh);
      disposeMesh(this.mesh);
      this.mesh = null;
    }
    this.numSplats = 0;
    this._highlighted = null;
  }
}
