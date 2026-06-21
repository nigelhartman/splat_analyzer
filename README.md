# Splat Analyzer

Automated 3D object detection for Gaussian Splat scenes. Upload a `.ply` or `.spz` splat file, type a text prompt, and the pipeline returns 3D bounding boxes with world-space coordinates — ready to use as interaction zones in WebXR applications.

No manual annotation. No training. Any object label in plain English.

---

## How It Works

1. **Render** — gsplat renders synthetic camera views (RGB + depth) from positions chosen by a density-aware sampler (cameras spawn inside the splat bounding box, spaced apart, and biased away from the dense interior of objects). The number of views is set by the `quality` preset.
2. **Detect** — OWLv2 open-vocabulary model detects objects in every frame
3. **Lift to 3D** — 2D bounding boxes are back-projected using per-pixel depth maps
4. **Cluster** — detections are aggregated across frames using anchor-based clustering
5. **Output** — `interactions.json` with label, position, scale, and rotation per object

---

## Pages

| URL | Description |
|---|---|
| `/` | Upload form — submit a job and see results |
| `/debug` | Interactive 3D viewer — load splat + overlay bounding boxes |
| `/admin` | Admin panel — API key management, live server stats |
| `/docs` | Swagger API documentation |

---

## Three ways to run

- **Server (Docker)** — deploy to an NVIDIA GPU box with HTTPS, the upload form, the API, and the admin panel. Use this to host the tool for events or other people. See [Quick Setup](#quick-setup) below.
- **Local — NVIDIA GPU (CLI)** — run the pipeline on your own CUDA machine, no server or Docker. See [Run locally (NVIDIA GPU)](#run-locally-nvidia-gpu) below.
- **Local — Apple Silicon Mac (CLI)** — run on an M-series Mac via the Metal renderer. See [Run locally (Apple Silicon Mac)](#run-locally-apple-silicon-mac) below.

All three share the same core (`pipeline.py` + `render_cameras.py`); only the front door and the GPU renderer differ. The renderer is selected automatically (`config.renderer = auto`): CUDA → `gsplat`, Apple Silicon → `gsplat-metal`.

---

## Quick Setup

### Prerequisites

- A Linux GPU server with an NVIDIA GPU (see [Hardware](#hardware) below)
- Your domain's DNS A record pointing to the server IP
- SSH key access to the server

### 1. Clone and configure

```bash
git clone https://github.com/nigelhartman/splat_analyzer.git
cd splat_analyzer
cp .env.example .env
```

Edit `.env` with your values:

```
SERVER_IP=1.2.3.4          # your server's public IP
SERVER_USER=ubuntu         # SSH username
SSH_KEY=~/.ssh/my_key      # path to your SSH private key
ADMIN_USER=admin           # admin panel username
ADMIN_PASSWORD=changeme    # admin panel password — change this
DOMAIN=your-domain.com     # must point to SERVER_IP
CUDA_ARCH_LIST=8.9         # GPU compute capability (see below)
```

### 2. Deploy

```bash
bash deploy.sh
```

On first run this takes 15–20 minutes (compiles gsplat CUDA extensions). Subsequent deploys take under a minute.

The script handles everything remotely: installs Docker, NVIDIA Container Toolkit, builds the image, starts the app container, and starts Caddy with automatic HTTPS.

### 3. Create an API key

Open `https://your-domain.com/admin`, log in with your admin credentials, and create a key. You need the key to use the upload form and API.

---

## Run locally (NVIDIA GPU)

If you have a machine with an NVIDIA CUDA GPU, you can run the pipeline directly — no server, no Docker. It reuses the same rendering and detection code as the server.

### 1. Create an environment and install dependencies

gsplat compiles CUDA extensions against your local toolkit, so install a CUDA-matched **torch first**, then the rest:

```bash
python -m venv .venv && source .venv/bin/activate     # or use conda

# 1) Install torch matching your CUDA (see https://pytorch.org). Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2) Tell gsplat which GPU arch to build for (see the table under Hardware), then install the rest:
export TORCH_CUDA_ARCH_LIST=8.9     # e.g. 8.9 for RTX 40xx / L40S
pip install -r requirements.txt
```

### 2. Run a detection

```bash
python run_local.py --ply scene.ply --prompt "chair, table" --quality high
```

This writes results to `./out_scene/`:

- `interactions.json` — detected objects (label, position, scale, rotation)
- `frames/` — rendered RGB + depth views
- `transforms.json` — camera poses

Options: `--quality {low,medium,high}`, `--job_dir <dir>`, `--score_threshold`, `--min_votes`, `--min_peak_score`. Run `python run_local.py --help` for details.

---

## Run locally (Apple Silicon Mac)

On an M-series Mac (no NVIDIA GPU) the pipeline renders through **Metal/MPS** using [gsplat-mps](https://github.com/iffyloop/gsplat-mps), a Metal port of gsplat. One script sets everything up:

```bash
./install_mac.sh
```

It creates a `.venv`, installs the Python deps (`requirements-mac.txt`), and builds the Metal renderer. Requirements: macOS on Apple Silicon and Xcode command-line tools (`xcode-select --install`). The first build compiles a Metal shader and takes a few minutes.

Then run exactly as on CUDA:

```bash
source .venv/bin/activate
python run_local.py --ply scene.ply --prompt "chair, table" --quality low
```

Notes:
- The renderer is selected automatically (`renderer=auto` → `gsplat-metal` on Apple Silicon).
- **OWLv2 detection runs on CPU** on Mac, so it is slower than a CUDA box — start with `--quality low`.
- gsplat-mps is **AGPLv3** and an unmaintained fork; it is installed separately by the script (not bundled). Render quality is close to, but not identical to, the CUDA path.

---

## API

All endpoints require `X-API-Key: <key>` header (or `Authorization: Bearer <key>`).

```bash
# Submit a detection job
curl -X POST https://your-domain.com/api/v1/detect \
  -H "X-API-Key: wmd_..." \
  -F "file=@scene.ply" \
  -F "prompt=sofa, chair, television"

# Poll status
curl https://your-domain.com/api/v1/jobs/<job_id>/status \
  -H "X-API-Key: wmd_..."

# Get results
curl https://your-domain.com/api/v1/jobs/<job_id>/result \
  -H "X-API-Key: wmd_..."
```

Result format:

```json
{
  "objects": [
    {
      "label": "sofa",
      "position": { "x": 1.2, "y": 0.4, "z": -0.8 },
      "size": { "x": 2.1, "y": 0.9, "z": 1.0 }
    }
  ]
}
```

### Detection parameters (optional)

| Parameter | Default | Description |
|---|---|---|
| `quality` | `medium` | Camera coverage: `low` (24 views) · `medium` (90) · `high` (192) |
| `score_threshold` | `0.12` | Per-frame OWLv2 confidence cutoff |
| `min_votes` | `8` | Frames an object must appear in to be kept |
| `min_peak_score` | `0.40` | Best single-frame confidence required |

---

## Supported Formats

- **`.ply`** — standard 3DGS format (Nerfstudio, Gaussian Splatting, gsplat)
- **`.spz`** — Niantic/World Labs compressed format (gzip + custom quantization, v1–v3)

---

## Hardware

The pipeline is developed and tested on a **Nebius cloud instance with an NVIDIA L40S (48 GB VRAM)**. A full detection job at `high` quality (192 frames rendered + OWLv2 across all frames) takes approximately 3–5 minutes on this hardware; lower quality presets are proportionally faster.

**Running on a local computer has not been tested.** Measured peak VRAM on the L40S during a real job was **~7 GB** (gsplat rendering peaks at ~1.5 GB, OWLv2 peaks at ~7 GB when both are loaded). Based on this, a local machine would need at minimum:

- **GPU**: Any NVIDIA GPU with **8 GB VRAM** should work; 12 GB gives comfortable headroom. An RTX 3060 12 GB or better is a reasonable starting point.
- **RAM**: 16 GB minimum, 32 GB recommended
- **Disk**: 20 GB free (Docker image + model cache)
- **CUDA**: 11.8 or 12.x with matching PyTorch build

If you run on a GPU without fp16/bf16 support or below CUDA 11.8, the gsplat build will fail. Set `CUDA_ARCH_LIST` in `.env` to match your GPU:

| GPU | CUDA Arch |
|---|---|
| RTX 3070/3080/3090 | `8.6` |
| RTX 4080/4090 / L40S | `8.9` |
| A100 | `8.0` |
| H100 | `9.0` |

---

## Project Structure

```
splat_analyzer/
├── server.py          # FastAPI app — REST API, admin, job queue
├── pipeline.py        # Detection pipeline — OWLv2, clustering, output
├── render_cameras.py  # gsplat renderer — camera placement, depth maps, SPZ converter
├── config.py          # Shared pipeline defaults + quality presets
├── run_local.py       # Local CLI runner (no server / Docker needed)
├── webapp/
│   ├── index.html     # Upload form
│   ├── debug.html     # 3D viewer
│   └── admin.html     # Admin panel
├── Dockerfile
├── Caddyfile          # HTTPS reverse proxy (domain set via DOMAIN env var)
├── deploy.sh          # One-command deploy script
├── requirements.txt
└── .env.example
```

---

## Tech Stack

Python · FastAPI · gsplat · OWLv2 · PyTorch · Three.js · SparkJS 2.0 · Docker · Caddy
