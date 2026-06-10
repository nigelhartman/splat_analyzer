# Splat Analyzer

Automated 3D object detection for Gaussian Splat scenes. Upload a `.ply` or `.spz` splat file, type a text prompt, and the pipeline returns 3D bounding boxes with world-space coordinates — ready to use as interaction zones in WebXR applications.

No manual annotation. No training. Any object label in plain English.

---

## How It Works

1. **Render** — gsplat renders 180 synthetic camera views (RGB + depth) around the scene
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
| `score_threshold` | `0.12` | Per-frame OWLv2 confidence cutoff |
| `min_votes` | `8` | Frames an object must appear in to be kept |
| `min_peak_score` | `0.40` | Best single-frame confidence required |

---

## Supported Formats

- **`.ply`** — standard 3DGS format (Nerfstudio, Gaussian Splatting, gsplat)
- **`.spz`** — Niantic/World Labs compressed format (gzip + custom quantization, v1–v3)

---

## Hardware

The pipeline is developed and tested on a **Nebius cloud instance with an NVIDIA L40S (48 GB VRAM)**. A full detection job (180 frames rendered + OWLv2 across all frames) takes approximately 3–5 minutes on this hardware.

**Running on a local computer has not been tested.** Based on observed resource usage, a local machine would need at minimum:

- **GPU**: NVIDIA RTX 3090 or better (24 GB VRAM). The gsplat renderer and OWLv2 both run on GPU. Smaller cards may hit OOM on large splats or require reducing the number of rendered views.
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
├── render_cameras.py  # gsplat renderer — camera views, depth maps, SPZ converter
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
