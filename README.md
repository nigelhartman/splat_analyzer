# Splat Analyzer

https://github.com/user-attachments/assets/9a60e986-172f-4ac9-a47d-52a0f44c5017

Find objects in a 3D Gaussian Splat — **no manual annotation, no training**.

Give it a splat file (`.ply` or `.spz`) and a plain-English prompt like `"chair, table, monitor"`, and it returns a 3D bounding box (position + size) for each object it finds — ready to use as interaction zones in WebXR, games, or robotics.

```jsonc
// in:  scene.ply  +  prompt "chair, table"
// out: interactions.json
{
  "objects": [
    { "label": "chair", "position": { "x": -4.0, "y": 1.3, "z": -2.8 },
                         "size":     { "x":  2.1, "y": 2.1, "z":  2.1 } }
  ]
}
```

---

## Which way to run it?

Pick one — they all run the same pipeline:

| You have… | Use | Setup |
|---|---|---|
| **A Mac (Apple Silicon)** | Local CLI, Metal renderer | [`./install_mac.sh`](#run-on-a-mac-apple-silicon) |
| **A PC/Linux box with an NVIDIA GPU** | Local CLI, CUDA renderer | [`pip install`](#run-on-an-nvidia-gpu) |
| **A GPU server to host for others (e.g. an event)** | Web app + REST API + admin panel | [`deploy.sh`](#run-as-a-server) |

The renderer is picked automatically for your hardware (CUDA → `gsplat`, Apple Silicon → `gsplat-metal`). Detection (OWLv2) and the rest of the pipeline are identical everywhere.

---

## How it works

1. **Render** — synthetic camera views (RGB + depth) are rendered around the splat by a density-aware sampler.
2. **Detect** — the OWLv2 open-vocabulary model finds your prompt's objects in every frame (2D boxes).
3. **Lift to 3D** — each 2D box is back-projected into 3D using the per-pixel depth map.
4. **Cluster** — detections are fused across all views into one 3D box per object.
5. **Output** — `interactions.json` with a label, position, and size per object.

---

## Run on a Mac (Apple Silicon)

Renders on the Apple GPU via Metal — no NVIDIA hardware needed.

```bash
./install_mac.sh                 # creates .venv, installs deps, builds the Metal renderer
source .venv/bin/activate
python run_local.py --ply scene.ply --prompt "chair, table" --quality low
```

Requirements: macOS on Apple Silicon + Xcode command-line tools (`xcode-select --install`). The first build compiles a Metal shader (a few minutes, one time).

Notes:
- Rendering and detection both use the Apple GPU. If you ever hit an "op not implemented for MPS" error, set `WMD_DEVICE=cpu` to force CPU detection.
- It's slower than a CUDA box, so start with `--quality low`.

---

## Run on an NVIDIA GPU

```bash
python -m venv .venv && source .venv/bin/activate

# 1) Install torch matching your CUDA (see https://pytorch.org). Example, CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2) Set your GPU arch (table under Hardware), then install the rest:
export TORCH_CUDA_ARCH_LIST=8.9          # e.g. 8.9 for RTX 40xx / L40S
pip install -r requirements.txt

# 3) Run
python run_local.py --ply scene.ply --prompt "chair, table" --quality high
```

`gsplat` compiles CUDA extensions on install, so torch must be installed first.

---

## Local CLI output

Both local modes write to `./out_<name>/` (override with `--job_dir`):

- `interactions.json` — detected objects (label, position, size)
- `frames/` — rendered RGB + depth views
- `transforms.json` — camera poses

Options: `--quality {low,medium,high}`, `--score_threshold`, `--min_votes`, `--min_peak_score`. Run `python run_local.py --help` for details.

---

## Run as a server

Host the tool for other people — web upload form, REST API, and an admin panel — on an NVIDIA GPU box.

### 1. Configure

```bash
git clone https://github.com/nigelhartman/splat_analyzer.git
cd splat_analyzer
cp .env.example .env        # then edit it:
```

```
SERVER_IP=1.2.3.4           # server public IP
SERVER_USER=ubuntu          # SSH username
SSH_KEY=~/.ssh/my_key       # SSH private key
ADMIN_USER=admin            # admin panel username
ADMIN_PASSWORD=change-me    # initial admin password (change it — see below)
DOMAIN=your-domain.com      # DNS A record must point to SERVER_IP
CUDA_ARCH_LIST=8.9          # GPU compute capability (see Hardware)
```

### 2. Deploy

```bash
bash deploy.sh
```

Installs Docker + NVIDIA Container Toolkit, builds the image, starts the app and a Caddy HTTPS proxy. First run takes 15–20 min (compiles gsplat); later deploys take under a minute.

### 3. Pages

| URL | What |
|---|---|
| `/` | Upload form — submit a splat, see results |
| `/debug` | 3D viewer — splat + detected boxes overlaid |
| `/admin` | Admin panel — API keys, server stats, **change password** |
| `/docs` | Interactive API docs |

### 4. Set a password and an API key

Open `/admin`, sign in with your `.env` credentials, then:
- **Settings → Change admin password** to set a real password (the `.env` value is only the initial default).
- **API Keys → Create Key** to get a `wmd_…` key for the upload form and API.

The admin password and API keys are stored on the server's persistent data volume, so they survive redeploys.

---

## API

All endpoints take `X-API-Key: <key>` (or `Authorization: Bearer <key>`).

```bash
# Submit a job (returns a job_id)
curl -X POST https://your-domain.com/api/v1/detect \
  -H "X-API-Key: wmd_..." \
  -F "file=@scene.ply" \
  -F "prompt=sofa, chair, television"

# Poll status, then fetch the result
curl https://your-domain.com/api/v1/jobs/<job_id>/status -H "X-API-Key: wmd_..."
curl https://your-domain.com/api/v1/jobs/<job_id>/result -H "X-API-Key: wmd_..."
```

### Detection parameters (optional, CLI + API)

| Parameter | Default | Description |
|---|---|---|
| `quality` | `medium` | Camera coverage: `low` (24 views) · `medium` (90) · `high` (192) |
| `score_threshold` | `0.12` | Per-frame OWLv2 confidence cutoff |
| `min_votes` | `8` | Frames an object must appear in to be kept |
| `min_peak_score` | `0.40` | Best single-frame confidence required |

---

## Supported formats

- **`.ply`** — standard 3DGS (Nerfstudio, Gaussian Splatting, gsplat)
- **`.spz`** — Niantic / World Labs compressed format (v1–v3)

---

## Hardware

Developed and tested on an **NVIDIA L40S (48 GB)**; a `high`-quality job takes ~3–5 min there. Measured peak VRAM ~7 GB (OWLv2 dominates). A local NVIDIA box needs roughly:

- **GPU**: 8 GB VRAM minimum (12 GB comfortable) — e.g. RTX 3060 12 GB or better
- **RAM**: 16 GB min, 32 GB recommended · **Disk**: ~20 GB free · **CUDA**: 11.8 or 12.x

Set `CUDA_ARCH_LIST` (`.env`) / `TORCH_CUDA_ARCH_LIST` (local) to match your GPU:

| GPU | Arch |
|---|---|
| RTX 3070/3080/3090 | `8.6` |
| RTX 4080/4090 · L40S | `8.9` |
| A100 | `8.0` |
| H100 | `9.0` |

Apple Silicon Macs run via Metal (no CUDA); see [Run on a Mac](#run-on-a-mac-apple-silicon).

---

## Project structure

```
splat_analyzer/
├── pipeline.py              # Pipeline core — OWLv2 detection, clustering, output
├── render_cameras.py        # Camera placement, depth maps, SPZ→PLY converter
├── renderers/               # Pluggable GPU renderers
│   ├── gsplat_backend.py    #   CUDA (nerfstudio gsplat)
│   └── gsplat_metal_backend.py  # Apple Metal (gsplat-mps)
├── config.py                # Shared defaults + quality presets + renderer choice
├── run_local.py             # Local CLI entry point
├── server.py                # FastAPI app — REST API, admin, job queue
├── webapp/                  # index (upload) · debug (3D viewer) · admin (panel)
├── install_mac.sh           # Apple Silicon setup
├── requirements.txt         # CUDA / server deps
├── requirements-mac.txt     # Apple Silicon deps
├── Dockerfile · Caddyfile · deploy.sh · .env.example
```

---

## Tech stack

Python · FastAPI · gsplat / gsplat-mps · OWLv2 · PyTorch · Three.js · SparkJS · Docker · Caddy

---

## Acknowledgements

- **[Boxer](https://github.com/facebookresearch/boxer)** (Meta FAIR) — the inspiration for this project. Boxer lifts open-vocabulary 2D detections (OWLv2) into fused 3D bounding boxes from Project Aria captures. Splat Analyzer adapts that idea to Gaussian Splats: rather than Aria `.vrs` + BoxerNet, it renders synthetic views of any `.ply`/`.spz` splat and lifts the 2D boxes with camera-projection geometry.
- **[OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble)** (Google) — open-vocabulary 2D object detection.
- **[gsplat](https://github.com/nerfstudio-project/gsplat)** (Nerfstudio) and **[gsplat-mps](https://github.com/iffyloop/gsplat-mps)** — Gaussian Splat rasterization on CUDA and Apple Metal.
- **SparkJS** and **Three.js** — in-browser splat rendering for the 3D viewer.
