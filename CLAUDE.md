# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Splat Analyzer finds objects in a 3D Gaussian Splat (`.ply` or `.spz`) given a plain-English prompt, outputting 3D bounding boxes in `interactions.json`. No training required — it renders synthetic camera views, runs OWLv2 open-vocabulary detection on each frame, back-projects 2D boxes to 3D using per-pixel depth, then clusters them.

It has two distinct parts:
- **Python pipeline** — the analysis backend (CLI + FastAPI server)
- **`viewer/`** — a standalone browser-based splat viewer that overlays the bounding boxes from `interactions.json`

## Running the pipeline

**Mac (Apple Silicon):**
```bash
./install_mac.sh                 # one-time setup; compiles Metal shader
source .venv/bin/activate
python run_local.py --ply scene.ply --prompt "chair, table" --quality low
```

**NVIDIA GPU:**
```bash
source .venv/bin/activate
python run_local.py --ply scene.ply --prompt "chair, table" --quality high
```

**Server mode (FastAPI):**
```bash
source .venv/bin/activate
uvicorn server:app --reload
```

Output goes to `./out_<name>/` (override with `--job_dir`). Key files: `interactions.json`, `frames/`, `transforms.json`.

## Key CLI flags

`--quality {low,medium,high}` — 24 / 90 / 192 camera views  
`--score_threshold` (default 0.12) — OWLv2 per-frame confidence cutoff  
`--min_votes` (default 8) — frames an object must appear in  
`--min_peak_score` (default 0.40) — best single-frame score required  

Override OWLv2 device: `WMD_DEVICE=cpu python run_local.py ...`

## Viewer development and deployment

The viewer is pure ES-module JavaScript — no build step.

```bash
cd viewer
npm run dev                   # local dev — plain static file server (dev_server.py)
npx wrangler deploy           # deploy to Cloudflare Workers (shrill-dawn-2565.johannestscharn.workers.dev)
```

Sample data (CozyBedroom) is bundled in the repo at `viewer/splats/` and served as a static asset — the viewer loads it by default with no external bucket or proxy involved. Users can override via the gear menu. If you add more sample files: drop them in `viewer/splats/`, then update the `EXAMPLES` constant in `viewer/src/main.js`.

## Architecture

```
run_local.py          → thin CLI wrapper; calls pipeline.run_pipeline()
pipeline.py           → orchestrates the 5-step pipeline (render → detect → lift → cluster → write)
render_cameras.py     → camera placement (density-aware Poisson-disk sampler); SPZ→PLY converter;
                        writes frames/frame_XXXX.png, depth_XXXX.npy, transforms.json
renderers/
  base.py             → Renderer abstract base class; shared tensor conventions
  gsplat_backend.py   → CUDA backend (nerfstudio gsplat)
  gsplat_metal_backend.py → Apple Metal backend (gsplat-mps, third_party/)
config.py             → PipelineConfig dataclass; QUALITY_PRESETS; single source of truth for defaults
server.py             → FastAPI app; job queue (asyncio Semaphore, 1 job at a time); SQLite for API keys/settings
webapp/               → static HTML: index (upload form), debug (3D viewer), admin (panel)
viewer/               → standalone annotated splat viewer (deployed separately to Cloudflare Workers)
  src/main.js         → composition root; wires all modules together; loads default scene on startup
  src/SceneManager.js → owns Three.js scene, camera, SparkRenderer, OrbitControls, animation loop
  src/SplatView.js    → loads splat file (File or URL); manages SplatMesh lifecycle and hover highlights
  src/AnnotationParser.js → parses interactions.json (File or URL) into Annotation objects; applies π-X transform
  src/AnnotationLayer.js  → manages AnnotatedObjects; per-frame marker depth-sort and overlap suppression
  src/AnnotatedObject.js  → pairs one BoundingBox + Marker; wires hover events between them
  src/SettingsMenu.js → gear FAB + file-picker popover; emits File objects via callbacks
projects/             → example splats with pre-computed interactions.json and transforms.json
```

**Data flow in `pipeline.py`:**
1. `render_cameras.render_views()` → `transforms.json` + `frames/`
2. OWLv2 loaded onto CUDA/MPS/CPU; each frame produces raw 3D detections via `_unproject_box()`
3. `_cluster_detections()` uses fixed-anchor greedy clustering (not centroid-drift) to avoid adjacent same-label objects bleeding together
4. `interactions.json` written with `objects` (label, position, scale, frames) and `frame_annotations`

**Viewer coordinate system:** The splat is rotated π about X in `SplatView`. `AnnotationParser` mirrors this for all annotations: position `(x, -y, -z)`, quaternion `(x, -y, -z, w)`. Both must stay in sync if the rotation ever changes.

**Renderer conventions** (must be consistent across all backends):
- Quaternions: wxyz
- Camera poses: c2w, OpenCV convention (X right, Y down, Z forward); backends receive w2c
- Depth: camera-space Z in world units, 0 = no hit
- `gsplat` (CUDA) wants exp'd scales; `gsplat-mps` (Metal) wants raw log-scales with `glob_scale=1`

## Server deployment

```bash
cp .env.example .env   # fill in SERVER_IP, SSH_KEY, DOMAIN, CUDA_ARCH_LIST, ADMIN_PASSWORD
bash deploy.sh         # SSH to server, installs Docker + NVIDIA toolkit, builds image, starts Caddy
```

API keys are prefixed `wmd_`. Auth via `X-API-Key` header or `Authorization: Bearer`. Admin password is seeded from `.env` on first run; thereafter stored (PBKDF2) in `keys.db` on the server's persistent volume.

## Environment

- Python `.venv` at repo root; activate before running anything
- No test suite in the repo
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically in `pipeline.py` before torch import (required for OWLv2 on Apple Silicon)
- `TORCH_CUDA_ARCH_LIST` must be set before `pip install -r requirements.txt` on NVIDIA (controls gsplat CUDA compilation)
