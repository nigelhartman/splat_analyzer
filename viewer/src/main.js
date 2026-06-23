import { SceneManager } from "./SceneManager.js";
import { SplatView } from "./SplatView.js";
import { AnnotationParser } from "./AnnotationParser.js";
import { AnnotationLayer } from "./AnnotationLayer.js";
import { SettingsMenu } from "./SettingsMenu.js";

// Composition root: instantiate the parts and wire the data flow.
const sceneManager = new SceneManager(document.getElementById("viewport"));
const splatView = new SplatView(sceneManager);
const annotationLayer = new AnnotationLayer(sceneManager, {
  onHoverStart: (annotation) => splatView.highlight(annotation),
  onHoverEnd: (annotation) => splatView.clearHighlight(annotation),
});
const hint = document.getElementById("empty-hint");

new SettingsMenu({
  onSplatFile: async (file) => {
    hint.classList.add("hidden");
    await splatView.load(file);
  },
  onAnnotationsFile: async (file) => {
    const annotations = await AnnotationParser.parse(file);
    annotationLayer.rebuild(annotations);
  },
});

sceneManager.start();
