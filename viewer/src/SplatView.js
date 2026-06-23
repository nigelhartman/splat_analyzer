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

// Loads and owns the SplatMesh. The π rotation about X here defines the
// viewer's world space; AnnotationParser mirrors it so boxes line up.
export class SplatView {
  constructor(sceneManager) {
    this.sceneManager = sceneManager;
    this.mesh = null;

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

  async load(file) {
    if (this.mesh) {
      this.sceneManager.remove(this.mesh);
      this.mesh = null;
    }
    this._highlighted = null;

    const bytes = new Uint8Array(await file.arrayBuffer());
    // .rad files carry a pre-baked LOD tree (built with `npm run build-lod`);
    // any other format gets an LOD tree generated on-the-fly in a worker.
    const isBaked = file.name.toLowerCase().endsWith(".rad");
    this.mesh = new SplatMesh({
      fileBytes: bytes,
      fileName: file.name,
      lod: isBaked ? undefined : true,
      onLoad: (mesh) => {
        mesh.rotation.x = Math.PI;
        mesh.edits = [];
        mesh.updateMatrixWorld(true);
        const box = new THREE.Box3().setFromObject(mesh);
        this.sceneManager.frameBounds(box);
      },
    });
    this.sceneManager.add(this.mesh);
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
}
