"""WorldModelData API — v1.0.0"""
import os, sys, uuid, json, time, shutil, sqlite3, secrets, logging, asyncio, zipfile
from io import BytesIO
from collections import defaultdict
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, Header, Query
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WORK_DIR = Path("/tmp/workshop_jobs")
DB_PATH  = Path("keys.db")
JOB_TTL  = 3600

ADMIN_USER     = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme")
WEBAPP_DIR     = Path("webapp")

JOB_STATUS     = {}  # {job_id: "pending"|"running"|"done"|"failed"}
ADMIN_SESSIONS = {}  # {token: expiry_epoch}


# ── Database ─────────────────────────────────────────────────────────────────
def _db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    with _db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS api_keys (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            key        TEXT    UNIQUE NOT NULL,
            label      TEXT    DEFAULT '',
            created_at TEXT    NOT NULL,
            last_used  TEXT,
            active     INTEGER NOT NULL DEFAULT 1
        )""")
        conn.commit()

def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── App lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    _init_db()
    asyncio.create_task(_cleanup_loop())
    yield

app = FastAPI(
    title="WorldModelData API",
    description=(
        "3D Gaussian Splat → 3D bounding-box detection pipeline.\n\n"
        "Pass your API key as `X-API-Key` header or `Authorization: Bearer <key>`."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── Auth helpers ──────────────────────────────────────────────────────────────
def _check_key(key: str | None) -> bool:
    if not key:
        return False
    with _db() as conn:
        row = conn.execute("SELECT id FROM api_keys WHERE key=? AND active=1", (key,)).fetchone()
        if not row:
            return False
        conn.execute("UPDATE api_keys SET last_used=? WHERE key=?", (_now(), key))
        conn.commit()
    return True

async def require_api_key(
    x_api_key:     str = Header(None, alias="X-API-Key"),
    authorization: str = Header(None),
) -> str:
    key = x_api_key
    if not key and authorization and authorization.startswith("Bearer "):
        key = authorization[7:]
    if not _check_key(key):
        raise HTTPException(401, "Invalid or missing API key")
    return key

async def require_admin(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Admin token required")
    token = authorization[7:]
    exp = ADMIN_SESSIONS.get(token)
    if not exp or exp < time.time():
        ADMIN_SESSIONS.pop(token, None)
        raise HTTPException(401, "Invalid or expired admin session")
    return token


# ── Cleanup loop ──────────────────────────────────────────────────────────────
async def _cleanup_loop():
    while True:
        await asyncio.sleep(600)
        now = time.time()
        try:
            for d in WORK_DIR.iterdir():
                if d.is_dir() and (now - d.stat().st_mtime) > JOB_TTL:
                    shutil.rmtree(d, ignore_errors=True)
                    JOB_STATUS.pop(d.name, None)
                    logger.info(f"Cleaned up job {d.name}")
        except Exception as exc:
            logger.warning(f"Cleanup error: {exc}")


# ── Pipeline runner ───────────────────────────────────────────────────────────
async def _run_pipeline(job_id: str, ply_path: Path, prompt: str, job_dir: Path):
    JOB_STATUS[job_id] = "running"
    logger.info(f"[{job_id}] Running pipeline prompt={prompt!r}")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "pipeline.py",
        "--ply", str(ply_path), "--prompt", prompt, "--job_dir", str(job_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.error(f"[{job_id}] Failed:\n{stderr.decode()}")
        (job_dir / "error.txt").write_text(stderr.decode()[-3000:])
        JOB_STATUS[job_id] = "failed"
    else:
        ply_path.unlink(missing_ok=True)
        JOB_STATUS[job_id] = "done"
        logger.info(f"[{job_id}] Done")


# ── Camera groups (for debug endpoint) ───────────────────────────────────────
def _build_camera_groups(transforms: dict, job_id: str) -> list:
    cam_positions = transforms.get("camera_positions", [])
    frames_meta   = transforms.get("frames", [])
    pos_frame_map = defaultdict(list)
    for i, fm in enumerate(frames_meta):
        pos_frame_map[fm.get("position_idx", 0)].append(i)
    groups = []
    for pos_idx in sorted(pos_frame_map.keys()):
        fidx_list = pos_frame_map[pos_idx]
        pos       = cam_positions[pos_idx] if pos_idx < len(cam_positions) else [0, 0, 0]
        first_fm  = frames_meta[fidx_list[0]]
        groups.append({
            "idx":             pos_idx,
            "label":           f"Camera {pos_idx + 1}",
            "position":        pos,
            "first_transform": first_fm["transform_matrix"],
            "frame_indices":   fidx_list,
            "first_frame_url": f"/api/v1/jobs/{job_id}/frames/frame_{fidx_list[0]:04d}.png",
            "first_depth_url": f"/api/v1/jobs/{job_id}/frames/depth_{fidx_list[0]:04d}.png",
        })
    return groups


# ── HTML pages ────────────────────────────────────────────────────────────────
@app.get("/",      include_in_schema=False)
async def root_page():  return FileResponse(WEBAPP_DIR / "index.html")

@app.get("/debug", include_in_schema=False)
async def debug_page(): return FileResponse(WEBAPP_DIR / "debug.html")

@app.get("/admin", include_in_schema=False)
async def admin_html(): return FileResponse(WEBAPP_DIR / "admin.html")


# ── Detection API ─────────────────────────────────────────────────────────────
@app.post(
    "/api/v1/detect",
    status_code=202,
    tags=["Detection"],
    summary="Submit a detection job",
)
async def detect(
    file:    UploadFile = File(..., description=".ply Gaussian Splat file"),
    prompt:  str        = Form(..., description="Comma-separated object labels"),
    api_key: str        = Depends(require_api_key),
):
    """Upload a `.ply` and a comma-separated prompt. Returns `job_id` immediately.
    Poll `/api/v1/jobs/{job_id}/status` until `done`, then fetch the result."""
    job_id  = str(uuid.uuid4())
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True)
    ply_path = job_dir / "scene.ply"
    try:
        with open(ply_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()
    JOB_STATUS[job_id] = "pending"
    asyncio.create_task(_run_pipeline(job_id, ply_path, prompt, job_dir))
    logger.info(f"[{job_id}] Queued — file={file.filename!r} prompt={prompt!r}")
    return {"job_id": job_id, "status": "pending"}


@app.get(
    "/api/v1/jobs/{job_id}/status",
    tags=["Detection"],
    summary="Poll job status",
)
async def job_status(job_id: str, api_key: str = Depends(require_api_key)):
    """Returns `{job_id, status}`. Status values: `pending` | `running` | `done` | `failed`."""
    status = JOB_STATUS.get(job_id)
    if status is None and not (WORK_DIR / job_id).exists():
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, "status": status or "unknown"}


@app.get(
    "/api/v1/jobs/{job_id}/result",
    tags=["Detection"],
    summary="Get detection result (clean format)",
)
async def get_result(job_id: str, api_key: str = Depends(require_api_key)):
    """Clean output with `label`, `position` (world-space centre x/y/z),
    and `size` (bounding-box dimensions in world units). Returns 202 while in progress."""
    status  = JOB_STATUS.get(job_id)
    job_dir = WORK_DIR / job_id
    if status is None and not job_dir.exists():
        raise HTTPException(404, "Job not found")
    if status in ("pending", "running"):
        return JSONResponse({"job_id": job_id, "status": status}, status_code=202)
    if status == "failed":
        ep = job_dir / "error.txt"
        raise HTTPException(500, ep.read_text()[-500:] if ep.exists() else "Pipeline failed")
    rp = job_dir / "interactions.json"
    if not rp.exists():
        raise HTTPException(500, "Result file missing")
    raw     = json.loads(rp.read_text())
    objects = raw.get("objects", raw) if isinstance(raw, dict) else raw
    clean   = [
        {
            "label":    o["label"],
            "position": o["position"],
            "size":     o.get("scale", o.get("size", {"x": 0.5, "y": 0.5, "z": 0.5})),
        }
        for o in objects
    ]
    return {"job_id": job_id, "status": "done", "objects": clean}


@app.get(
    "/api/v1/jobs/{job_id}/result/debug",
    tags=["Detection"],
    summary="Get full debug data",
)
async def get_result_debug(job_id: str, api_key: str = Depends(require_api_key)):
    """Full result including per-object frame lists, 2D box annotations, camera groups."""
    status  = JOB_STATUS.get(job_id)
    job_dir = WORK_DIR / job_id
    if status is None and not job_dir.exists():
        raise HTTPException(404, "Job not found")
    if status in ("pending", "running"):
        return JSONResponse({"job_id": job_id, "status": status}, status_code=202)
    if status == "failed":
        ep = job_dir / "error.txt"
        raise HTTPException(500, ep.read_text()[-500:] if ep.exists() else "Pipeline failed")
    rp = job_dir / "interactions.json"
    if not rp.exists():
        raise HTTPException(500, "Result file missing")
    raw               = json.loads(rp.read_text())
    objects           = raw.get("objects", raw) if isinstance(raw, dict) else raw
    frame_annotations = raw.get("frame_annotations", {}) if isinstance(raw, dict) else {}

    frames_dir   = job_dir / "frames"
    rgb_frames   = sorted(frames_dir.glob("frame_*.png")) if frames_dir.exists() else []
    depth_frames = sorted(frames_dir.glob("depth_*.png")) if frames_dir.exists() else []
    frame_urls   = [f"/api/v1/jobs/{job_id}/frames/{p.name}" for p in rgb_frames]
    depth_urls   = [f"/api/v1/jobs/{job_id}/frames/{p.name}" for p in depth_frames]

    camera_groups = []
    tp = job_dir / "transforms.json"
    if tp.exists():
        camera_groups = _build_camera_groups(json.loads(tp.read_text()), job_id)

    return {
        "job_id":            job_id,
        "status":            "done",
        "objects":           objects,
        "frames":            frame_urls,
        "depth_frames":      depth_urls,
        "camera_groups":     camera_groups,
        "frame_annotations": frame_annotations,
    }


@app.get("/api/v1/jobs/{job_id}/frames/{filename}", include_in_schema=False)
async def get_frame(
    job_id:   str,
    filename: str,
    key:      str = Query(None),
    x_api_key:     str = Header(None, alias="X-API-Key"),
    authorization: str = Header(None),
):
    # Accept key via query param (for <img src>) or header
    resolved = key or x_api_key
    if not resolved and authorization and authorization.startswith("Bearer "):
        resolved = authorization[7:]
    if not _check_key(resolved):
        raise HTTPException(401, "Invalid API key")
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise HTTPException(400, "Invalid filename")
    fp = WORK_DIR / job_id / "frames" / filename
    if not fp.exists():
        raise HTTPException(404, "Frame not found")
    return FileResponse(fp, media_type="image/png")


@app.get(
    "/api/v1/jobs/{job_id}/download",
    tags=["Detection"],
    summary="Download all job data as ZIP",
)
async def download_job(job_id: str, api_key: str = Depends(require_api_key)):
    """ZIP containing RGB frames, depth maps, interactions.json, and transforms.json."""
    job_dir = WORK_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, "Job not found")
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        fd = job_dir / "frames"
        if fd.exists():
            for f in sorted(fd.glob("*.png")):
                zf.write(f, f"frames/{f.name}")
        for name in ("interactions.json", "transforms.json"):
            p = job_dir / name
            if p.exists():
                zf.write(p, name)
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=splat_{job_id[:8]}.zip"},
    )


# ── Admin ─────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

class KeyCreateRequest(BaseModel):
    label: str = ""


@app.post("/api/v1/admin/login", tags=["Admin"], summary="Admin login")
async def admin_login(body: LoginRequest):
    """POST `{username, password}` → `{token}` valid for 1 hour."""
    if body.username != ADMIN_USER or body.password != ADMIN_PASSWORD:
        raise HTTPException(401, "Invalid credentials")
    token = secrets.token_urlsafe(32)
    ADMIN_SESSIONS[token] = time.time() + 3600
    return {"token": token}


@app.get("/api/v1/admin/keys", tags=["Admin"], summary="List all API keys")
async def list_keys(admin: str = Depends(require_admin)):
    with _db() as conn:
        rows = conn.execute(
            "SELECT id, label, key, created_at, last_used, active "
            "FROM api_keys ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/v1/admin/keys", tags=["Admin"], summary="Create a new API key")
async def create_key(body: KeyCreateRequest, admin: str = Depends(require_admin)):
    """Returns the full key — store it securely, it cannot be retrieved later."""
    key = "wmd_" + secrets.token_urlsafe(32)
    now = _now()
    with _db() as conn:
        cur = conn.execute(
            "INSERT INTO api_keys (key, label, created_at) VALUES (?,?,?)",
            (key, body.label.strip(), now),
        )
        conn.commit()
        row_id = cur.lastrowid
    return {"id": row_id, "key": key, "label": body.label, "created_at": now}


@app.delete("/api/v1/admin/keys/{key_id}", tags=["Admin"], summary="Revoke an API key")
async def revoke_key(key_id: int, admin: str = Depends(require_admin)):
    with _db() as conn:
        conn.execute("UPDATE api_keys SET active=0 WHERE id=?", (key_id,))
        conn.commit()
    return {"ok": True, "id": key_id}


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok", "version": "1.0.0"}
