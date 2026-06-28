import { SceneManager } from "./SceneManager.js";
import { SplatView } from "./SplatView.js";
import { AnnotationParser } from "./AnnotationParser.js";
import { AnnotationLayer } from "./AnnotationLayer.js";
import { SettingsMenu } from "./SettingsMenu.js";

const DEFAULT_SPLAT_URL = "https://pub-91ba9769f7c94b59aae04c477ae1e261.r2.dev/candy-room-splat-lod.rad";
const DEFAULT_ANNOTATIONS_URL = "https://pub-91ba9769f7c94b59aae04c477ae1e261.r2.dev/candy-room-objectdata.json";

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

// Load default scene; hint stays until splat is ready
hint.classList.add("hidden");
splatView.load(DEFAULT_SPLAT_URL);
AnnotationParser.parse(DEFAULT_ANNOTATIONS_URL).then((annotations) =>
  annotationLayer.rebuild(annotations)
);
