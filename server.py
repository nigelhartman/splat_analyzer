import os
import sys
import uuid
import json
import time
import shutil
import logging
import asyncio
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
JOB_TTL_SECONDS = 3600  # keep frames for 1 hour


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


@app.post("/generate_interactions")
async def generate_interactions(
    file: UploadFile = File(..., description=".ply Gaussian Splat file"),
    prompt: str = Form(..., description="Comma-separated object labels, e.g. 'chair, desk'"),
):
    job_id = str(uuid.uuid4())
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
        "--ply", str(ply_path),
        "--prompt", prompt,
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

    # Delete the large .ply to save disk; keep frames + transforms
    ply_path.unlink(missing_ok=True)

    frames = sorted((job_dir / "frames").glob("frame_*.png"))
    frame_urls = [f"/jobs/{job_id}/frames/{p.name}" for p in frames]

    logger.info(f"[{job_id}] Done — {len(interactions)} interactions, {len(frames)} frames")
    return JSONResponse(content={
        "objects": interactions,
        "job_id": job_id,
        "frames": frame_urls,
    })


@app.get("/jobs/{job_id}/frames/{filename}")
async def get_frame(job_id: str, filename: str):
    # Prevent path traversal
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    frame_path = WORK_DIR / job_id / "frames" / filename
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(frame_path, media_type="image/png")


@app.get("/health")
def health():
    return {"status": "ok"}
