import { BoundingBox } from "./BoundingBox.js";
import { Marker } from "./Marker.js";

// Pairs one annotation's box and marker. Hovering the marker reveals the box
// and forwards hover events outward (so the splat itself can react).
export class AnnotatedObject {
  constructor(annotation, hooks = {}) {
    this.annotation = annotation;
    this.box = new BoundingBox(annotation);
    this.marker = new Marker(annotation, {
      onHoverStart: () => {
        this.box.show();
        hooks.onHoverStart?.(annotation);
      },
      onHoverEnd: () => {
        this.box.hide();
        hooks.onHoverEnd?.(annotation);
      },
    });
  }

  addTo(sceneManager) {
    sceneManager.add(this.box.group);
    sceneManager.add(this.marker.dotObject);
    sceneManager.add(this.marker.hoverLabelObject);
  }

  removeFrom(sceneManager) {
    sceneManager.remove(this.box.group);
    sceneManager.remove(this.marker.dotObject);
    sceneManager.remove(this.marker.hoverLabelObject);
  }
}
