import * as THREE from "three";
import { Annotation } from "./Annotation.js";

const PALETTE = [
  "#2f81f7", "#3fb950", "#db6d28", "#a371f7",
  "#f778ba", "#e3b341", "#56d4dd", "#f85149",
];

// The splat is rotated π about X in SplatView; annotations must follow the same
// mapping: position (x,-y,-z) and quaternion (x,-y,-z,w).
function toViewerSpace(obj) {
  const p = obj.position;
  const position = new THREE.Vector3(p.x, -p.y, -p.z);
  const s = obj.scale || obj.size || { x: 0.5, y: 0.5, z: 0.5 };
  const size = new THREE.Vector3(s.x, s.y, s.z);
  const r = obj.rotation || { x: 0, y: 0, z: 0, w: 1 };
  const quaternion = new THREE.Quaternion(r.x, -r.y, -r.z, r.w);
  return { position, size, quaternion };
}

export class AnnotationParser {
  static async parse(source) {
    const text =
      typeof source === "string"
        ? await fetch(source).then((r) => r.text())
        : await source.text();
    return AnnotationParser.fromText(text);
  }

  static fromText(text) {
    const data = JSON.parse(text);
    const objects = Array.isArray(data) ? data : data.objects || [];

    const colorByLabel = new Map();
    let colorIdx = 0;

    return objects
      .filter((o) => o && o.label && o.position)
      .map((o) => {
        if (!colorByLabel.has(o.label)) {
          colorByLabel.set(o.label, PALETTE[colorIdx++ % PALETTE.length]);
        }
        return new Annotation({
          label: o.label,
          color: colorByLabel.get(o.label),
          ...toViewerSpace(o),
        });
      });
  }
}
