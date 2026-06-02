import os
import sys
import uuid
import json
import time
import shutil
import logging
import asyncio
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="3D Gaussian Splat Interaction Generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WORK_DIR = Path("/tmp/workshop_jobs")
WORK_DIR.mkdir(parents=True, exist_ok=True)
JOB_TTL_SECONDS = 3600


@app.on_event("startup")
async def startup():
    asyncio.create_task(_cleanup_loop())


async def _cleanup_loop():
    while True:
        await asyncio.sleep(600)
        now = time.time()
        for d in WORK_DIR.iterdir():
            if d.is_dir() and (now - d.stat().st_mtime) > JOB_TTL_SECONDS:
                shutil.rmtree(d, ignore_errors=True)
                logger.info(f"Cleaned up old job {d.name}")


def _build_camera_groups(transforms: dict, job_id: str) -> list:
    """Parse transforms.json and build per-camera-position metadata for the frontend."""
    cam_positions = transforms.get("camera_positions", [])
    frames_meta   = transforms.get("frames", [])

    pos_frame_map = defaultdict(list)
    for i, fm in enumerate(frames_meta):
        pos_idx = fm.get("position_idx", 0)
        pos_frame_map[pos_idx].append(i)

    groups = []
    for pos_idx in sorted(pos_frame_map.keys()):
        fidx_list = pos_frame_map[pos_idx]
        pos = cam_positions[pos_idx] if pos_idx < len(cam_positions) else [0.0, 0.0, 0.0]
        first_fidx = fidx_list[0]
        first_fm   = frames_meta[first_fidx]
        groups.append({
            "idx":             pos_idx,
            "label":           f"Camera {pos_idx + 1}",
            "position":        pos,
            "first_transform": first_fm["transform_matrix"],
            "frame_indices":   fidx_list,
            "first_frame_url": f"/jobs/{job_id}/frames/frame_{first_fidx:04d}.png",
            "first_depth_url": f"/jobs/{job_id}/frames/depth_{first_fidx:04d}.png",
        })
    return groups


@app.post("/generate_interactions")
async def generate_interactions(
    file:   UploadFile = File(..., description=".ply Gaussian Splat file"),
    prompt: str        = Form(..., description="Comma-separated object labels"),
):
    job_id  = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True)

    ply_path = job_dir / "scene.ply"
    try:
        with open(ply_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    logger.info(f"[{job_id}] Received file={file.filename!r} prompt={prompt!r}")

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "pipeline.py",
        "--ply",     str(ply_path),
        "--prompt",  prompt,
        "--job_dir", str(job_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        logger.error(f"[{job_id}] pipeline failed:\n{stderr.decode()}")
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=stderr.decode()[-2000:])

    result_path = job_dir / "interactions.json"
    if not result_path.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Pipeline produced no output file.")

    with open(result_path) as f:
        interactions = json.load(f)

    ply_path.unlink(missing_ok=True)

    frames_dir = job_dir / "frames"

    # RGB frames
    rgb_frames = sorted(frames_dir.glob("frame_*.png"))
    frame_urls = [f"/jobs/{job_id}/frames/{p.name}" for p in rgb_frames]

    # Depth frames (colorized PNGs)
    depth_frames = sorted(frames_dir.glob("depth_*.png"))
    depth_urls   = [f"/jobs/{job_id}/frames/{p.name}" for p in depth_frames]

    # Camera groups from transforms.json
    camera_groups = []
    transforms_path = job_dir / "transforms.json"
    if transforms_path.exists():
        with open(transforms_path) as f:
            transforms = json.load(f)
        camera_groups = _build_camera_groups(transforms, job_id)

    logger.info(f"[{job_id}] Done — {len(interactions)} objects, "
                f"{len(rgb_frames)} frames, {len(camera_groups)} camera positions")

    return JSONResponse(content={
        "objects":       interactions,
        "job_id":        job_id,
        "frames":        frame_urls,
        "depth_frames":  depth_urls,
        "camera_groups": camera_groups,
    })


@app.get("/jobs/{job_id}/frames/{filename}")
async def get_frame(job_id: str, filename: str):
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    frame_path = WORK_DIR / job_id / "frames" / filename
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(frame_path, media_type="image/png")


@app.get("/health")
def health():
    return {"status": "ok"}
