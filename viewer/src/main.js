import { SceneManager } from "./SceneManager.js";
import { SplatView } from "./SplatView.js";
import { AnnotationParser } from "./AnnotationParser.js";
import { AnnotationLayer } from "./AnnotationLayer.js";
import { SettingsMenu } from "./SettingsMenu.js";

const PLAY_SVG = `<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5,3 19,12 5,21"/></svg>`;

// Bundled with the repo (see viewer/splats/) so the viewer works without any
// external bucket or proxy.
const EXAMPLES = [
  {
    id: "1",
    label: "CozyBedroom",
    splatUrl: "/splats/CozyBedroom.spz",
    annotationsUrl: "/splats/CozyBedroom.json",
  },
];

const sceneManager = new SceneManager(document.getElementById("viewport"));
const splatView = new SplatView(sceneManager);
const annotationLayer = new AnnotationLayer(sceneManager, {
  onHoverStart: (annotation) => splatView.highlight(annotation),
  onHoverEnd: (annotation) => splatView.clearHighlight(annotation),
});
const hint = document.getElementById("empty-hint");
const loading = document.getElementById("loading");
const loadingBarFill = document.getElementById("loading-bar-fill");
const loadingLabel = document.getElementById("loading-label");

let loadGeneration = 0;

function formatMb(bytes) {
  return (bytes / (1024 * 1024)).toFixed(1);
}

function showLoading(label = "Loading…") {
  loading.classList.remove("hidden");
  hint.classList.add("hidden");
  loadingBarFill.classList.remove("indeterminate");
  loadingBarFill.style.width = "0%";
  loadingLabel.textContent = label;
}

function setDownloadProgress(loaded, total) {
  loadingBarFill.classList.remove("indeterminate");
  if (total && total > 0) {
    const pct = Math.min(100, (loaded / total) * 100);
    loadingBarFill.style.width = `${pct}%`;
    loadingLabel.textContent = `${formatMb(loaded)} / ${formatMb(total)} MB`;
  } else {
    loadingBarFill.classList.add("indeterminate");
    loadingLabel.textContent = `${formatMb(loaded)} MB downloaded`;
  }
}

function setProcessing() {
  loadingBarFill.classList.remove("indeterminate");
  loadingBarFill.style.width = "100%";
  loadingLabel.textContent = "Processing splat…";
}

function hideLoading() {
  loading.classList.add("hidden");
  loadingBarFill.classList.remove("indeterminate");
  loadingBarFill.style.width = "0%";
}

async function loadExample(id) {
  const example = EXAMPLES.find((ex) => ex.id === id);
  if (!example) return;

  const gen = ++loadGeneration;
  settingsMenu.clearFileSelections();

  // Drop the previous scene immediately so old labels/splat don't linger.
  splatView.clear();
  annotationLayer.clear();
  showLoading("Loading…");

  // Fetch annotations in parallel, but don't show them until the splat is ready.
  const annotationsPromise = AnnotationParser.parse(example.annotationsUrl).catch((err) => {
    console.error(`Failed to load annotations for example ${id}:`, err);
    return null;
  });

  try {
    await splatView.load(example.splatUrl, {
      onProgress: (loaded, total) => {
        if (gen !== loadGeneration) return;
        setDownloadProgress(loaded, total);
      },
      onProcessing: () => {
        if (gen === loadGeneration) setProcessing();
      },
    });
    if (gen !== loadGeneration) return;

    const annotations = await annotationsPromise;
    if (gen !== loadGeneration) return;

    hideLoading();
    if (annotations) {
      annotationLayer.rebuild(annotations);
      annotationLayer.playReveal();
    }
  } catch (err) {
    console.error(`Failed to load example ${id}:`, err);
    if (gen === loadGeneration) {
      splatView.clear();
      annotationLayer.clear();
      hint.classList.remove("hidden");
      hideLoading();
    }
  }
}

function clearScene() {
  loadGeneration += 1;
  settingsMenu.clearFileSelections();
  splatView.clear();
  annotationLayer.clear();
  hideLoading();
  hint.classList.remove("hidden");
}

const settingsMenu = new SettingsMenu({
  examples: EXAMPLES,
  defaultExampleId: "1",
  onExampleSelect: (id) => {
    if (!id) {
      clearScene();
      return;
    }
    loadExample(id);
  },
  onSplatFile: async (file) => {
    const gen = ++loadGeneration;
    settingsMenu.setExampleId("");
    splatView.clear();
    showLoading("Loading…");
    try {
      await splatView.load(file, {
        onProgress: (loaded, total) => {
          if (gen !== loadGeneration) return;
          setDownloadProgress(loaded, total);
        },
        onProcessing: () => {
          if (gen === loadGeneration) setProcessing();
        },
      });
    } catch (err) {
      console.error("Failed to load splat file:", err);
      if (gen === loadGeneration) hint.classList.remove("hidden");
    } finally {
      if (gen === loadGeneration) hideLoading();
    }
  },
  onAnnotationsFile: async (file) => {
    settingsMenu.setExampleId("");
    try {
      const annotations = await AnnotationParser.parse(file);
      annotationLayer.rebuild(annotations);
    } catch (err) {
      console.error("Failed to load annotations:", err);
    }
  },
});

const playBtn = document.createElement("button");
playBtn.id = "play-btn";
playBtn.className = "fab";
playBtn.title = "Play Label appear animation";
playBtn.innerHTML = PLAY_SVG;
playBtn.addEventListener("click", () => annotationLayer.playReveal());
document.body.appendChild(playBtn);

sceneManager.start();

let lastStatsUpdate = 0;
sceneManager.addFrameListener(() => {
  const now = performance.now();
  if (now - lastStatsUpdate < 250) return;
  lastStatsUpdate = now;
  settingsMenu.updateStats({
    fps: sceneManager.fps,
    splatCount: splatView.numSplats,
    labelCount: annotationLayer.objects.length,
  });
});

loadExample("1");
