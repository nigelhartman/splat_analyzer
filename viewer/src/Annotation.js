// Immutable annotation in viewer space. Coordinates are already mapped from the
// source JSON by AnnotationParser, so consumers use them as-is.
export class Annotation {
  constructor({ label, position, size, quaternion, color }) {
    this.label = label;
    this.position = position;   // THREE.Vector3
    this.size = size;           // THREE.Vector3 (full box extents)
    this.quaternion = quaternion; // THREE.Quaternion
    this.color = color;         // hex string
  }
}
