import * as THREE from "three";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

// One object's box: crisp wireframe edges plus a faint translucent fill.
// Hidden by default; revealed only while its marker is hovered.
export class BoundingBox {
  constructor(annotation) {
    const { size, position, quaternion, color } = annotation;
    const col = new THREE.Color(color);

    this.group = new THREE.Group();
    this.group.position.copy(position);
    this.group.quaternion.copy(quaternion);
    this.group.visible = false;

    const boxGeo = new THREE.BoxGeometry(size.x, size.y, size.z);

    const edges = new THREE.EdgesGeometry(boxGeo);
    const lineGeo = new LineSegmentsGeometry();
    lineGeo.setPositions(edges.attributes.position.array);
    this.material = new LineMaterial({
      color: col,
      linewidth: 0.6,
      resolution: new THREE.Vector2(window.innerWidth, window.innerHeight),
    });
    this.group.add(new LineSegments2(lineGeo, this.material));

    const fill = new THREE.Mesh(
      boxGeo,
      new THREE.MeshBasicMaterial({
        color: col,
        transparent: true,
        opacity: 0.06,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    this.group.add(fill);
  }

  show() { this.group.visible = true; }
  hide() { this.group.visible = false; }

  setResolution(w, h) { this.material.resolution.set(w, h); }
}
