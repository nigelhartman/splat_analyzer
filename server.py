"""WorldModelData API — v1.0.0"""
import os, sys, uuid, json, time, shutil, sqlite3, secrets, hashlib, logging, asyncio, zipfile
from io import BytesIO
from collections import defaultdict
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, Header, Query
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import PipelineConfig, QUALITY_PRESETS, WORLD_UP_CHOICES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WORK_DIR = Path("/tmp/workshop_jobs")
DB_PATH  = Path(os.getenv("DB_PATH", "keys.db"))
JOB_TTL  = 3600

ADMIN_USER     = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme")
WEBAPP_DIR     = Path("webapp")

JOB_STATUS      = {}   # {job_id: "pending"|"running"|"done"|"failed"}
JOB_QUEUE_ORDER: list = []  # job_ids in submission order while pending
_JOB_SEMAPHORE: asyncio.Semaphore | None = None  # enforces one job at a time
ADMIN_SESSIONS  = {}   # {token: expiry_epoch}


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
        conn.execute("""CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )""")
        # Seed the admin password (hashed) from ADMIN_PASSWORD on first run; after that the
        # stored value is the source of truth (changed via the admin panel).
        if not conn.execute("SELECT 1 FROM settings WHERE key='admin_password'").fetchone():
            conn.execute("INSERT INTO settings (key, value) VALUES ('admin_password', ?)",
                         (_hash_password(ADMIN_PASSWORD),))
        conn.commit()

def _get_setting(key: str) -> str | None:
    with _db() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None

def _set_setting(key: str, value: str):
    with _db() as conn:
        conn.execute("INSERT INTO settings (key, value) VALUES (?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        conn.commit()

def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ── Password hashing (stdlib PBKDF2 — no extra deps) ─────────────────────────
def _hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000)
    return f"pbkdf2_sha256$200000${salt.hex()}${dk.hex()}"

def _verify_password(pw: str, stored: str) -> bool:
    try:
        _algo, iters, salt_hex, hash_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt_hex), int(iters))
        return secrets.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


# ── App lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_):
    global _JOB_SEMAPHORE
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    _init_db()
    _JOB_SEMAPHORE = asyncio.Semaphore(1)
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
async def _run_pipeline_queued(job_id: str, ply_path: Path, prompt: str, job_dir: Path,
                               quality: str, world_up: str, score_threshold: float,
                               min_votes: int, min_peak_score: float):
    """Wait for the semaphore (one job at a time), then run the pipeline."""
    async with _JOB_SEMAPHORE:
        if job_id in JOB_QUEUE_ORDER:
            JOB_QUEUE_ORDER.remove(job_id)
        await _run_pipeline(job_id, ply_path, prompt, job_dir,
                            quality, world_up, score_threshold, min_votes, min_peak_score)

async def _run_pipeline(job_id: str, ply_path: Path, prompt: str, job_dir: Path,
                        quality: str, world_up: str, score_threshold: float,
                        min_votes: int, min_peak_score: float):
    JOB_STATUS[job_id] = "running"
    logger.info(f"[{job_id}] Running pipeline prompt={prompt!r} quality={quality} world_up={world_up} "
                f"score_threshold={score_threshold} min_votes={min_votes} min_peak_score={min_peak_score}")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "pipeline.py",
        "--ply", str(ply_path), "--prompt", prompt, "--job_dir", str(job_dir),
        "--quality", quality,
        "--world_up", world_up,
        "--score_threshold", str(score_threshold),
        "--min_votes", str(min_votes),
        "--min_peak_score", str(min_peak_score),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.error(f"[{job_id}] Failed:\n{stderr.decode()}")
        (job_dir / "error.txt").write_text(stderr.decode()[-3000:])
        JOB_STATUS[job_id] = "failed"
    else:
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
    file:             UploadFile = File(...,  description=".ply or .spz Gaussian Splat file"),
    prompt:           str        = Form(...,  description="Comma-separated object labels"),
    quality:          str        = Form(PipelineConfig().quality,
                                         description="Camera-coverage preset: low | medium | high"),
    score_threshold:  float      = Form(PipelineConfig().score_threshold,
                                         description="OWLv2 per-frame confidence threshold (0–1)"),
    min_votes:        int        = Form(PipelineConfig().min_votes,
                                         description="Minimum frames a cluster must appear in"),
    min_peak_score:   float      = Form(PipelineConfig().min_peak_score,
                                         description="Minimum best-frame score for a cluster to be kept (0–1)"),
    world_up:         str        = Form(PipelineConfig().world_up,
                                         description="Scene up-axis: y-down (standard 3DGS/COLMAP) | y-up"),
    api_key:          str        = Depends(require_api_key),
):
    """Upload a `.ply` or `.spz` file and a comma-separated prompt. Returns `job_id` immediately.
    Poll `/api/v1/jobs/{job_id}/status` until `done`, then fetch the result."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".ply", ".spz"):
        raise HTTPException(status_code=400, detail="Only .ply and .spz files are supported")
    if quality not in QUALITY_PRESETS:
        raise HTTPException(status_code=400,
                            detail=f"quality must be one of {sorted(QUALITY_PRESETS)}")
    if world_up not in WORLD_UP_CHOICES:
        raise HTTPException(status_code=400,
                            detail=f"world_up must be one of {sorted(WORLD_UP_CHOICES)}")
    job_id   = str(uuid.uuid4())
    job_dir  = WORK_DIR / job_id
    job_dir.mkdir(parents=True)
    scene_path = job_dir / f"scene{suffix}"
    try:
        with open(scene_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()
    JOB_STATUS[job_id] = "pending"
    JOB_QUEUE_ORDER.append(job_id)
    asyncio.create_task(_run_pipeline_queued(job_id, scene_path, prompt, job_dir,
                                             quality, world_up, score_threshold, min_votes, min_peak_score))
    logger.info(f"[{job_id}] Queued (position {len(JOB_QUEUE_ORDER)}) — "
                f"file={file.filename!r} prompt={prompt!r} quality={quality} world_up={world_up} "
                f"score_threshold={score_threshold} min_votes={min_votes} min_peak_score={min_peak_score}")
    return {"job_id": job_id, "status": "pending", "queue_position": len(JOB_QUEUE_ORDER)}


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
    result: dict = {"job_id": job_id, "status": status or "unknown"}
    if status == "pending":
        try:
            pos = JOB_QUEUE_ORDER.index(job_id) + 1
        except ValueError:
            pos = 1  # transitioning to running
        result["queue_position"] = pos
        result["queue_ahead"] = pos - 1
    return result


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


@app.get("/api/v1/jobs/{job_id}/splat", include_in_schema=False)
async def get_splat(
    job_id: str,
    key: str = Query(None),
    x_api_key: str = Header(None, alias="X-API-Key"),
    authorization: str = Header(None),
):
    resolved = key or x_api_key
    if not resolved and authorization and authorization.startswith("Bearer "):
        resolved = authorization[7:]
    if not _check_key(resolved):
        raise HTTPException(401, "Invalid API key")
    if "/" in job_id or ".." in job_id:
        raise HTTPException(400, "Invalid job ID")
    ply = WORK_DIR / job_id / "scene.ply"
    if not ply.exists():
        raise HTTPException(404, "Splat file not found or expired")
    return FileResponse(ply, media_type="application/octet-stream",
                        headers={"Content-Disposition": "inline; filename=scene.ply"})


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

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@app.post("/api/v1/admin/login", tags=["Admin"], summary="Admin login")
async def admin_login(body: LoginRequest):
    """POST `{username, password}` → `{token}` valid for 1 hour."""
    stored = _get_setting("admin_password")
    if body.username != ADMIN_USER or not stored or not _verify_password(body.password, stored):
        raise HTTPException(401, "Invalid credentials")
    token = secrets.token_urlsafe(32)
    ADMIN_SESSIONS[token] = time.time() + 3600
    return {"token": token}


@app.post("/api/v1/admin/change_password", tags=["Admin"], summary="Change the admin password")
async def change_password(body: ChangePasswordRequest, admin: str = Depends(require_admin)):
    """Verify the current password, then store a new one (hashed). Persists in the settings DB."""
    stored = _get_setting("admin_password")
    if not stored or not _verify_password(body.current_password, stored):
        raise HTTPException(401, "Current password is incorrect")
    if len(body.new_password) < 8:
        raise HTTPException(400, "New password must be at least 8 characters")
    _set_setting("admin_password", _hash_password(body.new_password))
    return {"ok": True}


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


# ── Stats helpers (no external deps — uses /proc + nvidia-smi) ────────────────
def _cpu_sample():
    with open("/proc/stat") as f:
        parts = f.readline().split()
    vals = [int(x) for x in parts[1:8]]
    idle = vals[3] + vals[4]  # idle + iowait
    return idle, sum(vals)

def _cpu_percent_sync() -> float:
    try:
        i1, t1 = _cpu_sample()
        time.sleep(0.2)
        i2, t2 = _cpu_sample()
        dt = t2 - t1
        return round((1.0 - (i2 - i1) / dt) * 100, 1) if dt else 0.0
    except Exception:
        return 0.0

def _memory_stats() -> dict:
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1]) * 1024
        total = info.get("MemTotal", 0)
        used  = total - info.get("MemAvailable", 0)
        return {
            "total_gb": round(total / 1_073_741_824, 1),
            "used_gb":  round(used  / 1_073_741_824, 1),
            "percent":  round(used / total * 100, 1) if total else 0.0,
        }
    except Exception:
        return {"total_gb": 0, "used_gb": 0, "percent": 0.0}

def _disk_stats() -> dict:
    try:
        st = os.statvfs("/")
        total = st.f_blocks * st.f_frsize
        free  = st.f_bavail * st.f_frsize
        used  = total - free
        return {
            "total_gb": round(total / 1_073_741_824, 1),
            "used_gb":  round(used  / 1_073_741_824, 1),
            "percent":  round(used / total * 100, 1) if total else 0.0,
        }
    except Exception:
        return {"total_gb": 0, "used_gb": 0, "percent": 0.0}

def _gpu_stats() -> list:
    try:
        import subprocess as _sp
        r = _sp.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode != 0:
            return []
        gpus = []
        for line in r.stdout.strip().split("\n"):
            if not line.strip():
                continue
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 5:
                mu, mt = int(p[2]), int(p[3])
                gpus.append({
                    "name":            p[0],
                    "utilization":     int(p[1]),
                    "memory_used_mb":  mu,
                    "memory_total_mb": mt,
                    "memory_percent":  round(mu / mt * 100, 1) if mt else 0.0,
                    "temperature":     int(p[4]),
                })
        return gpus
    except Exception:
        return []

def _collect_stats() -> dict:
    cpu_pct = _cpu_percent_sync()
    return {
        "cpu":    {"percent": cpu_pct},
        "memory": _memory_stats(),
        "disk":   _disk_stats(),
        "gpus":   _gpu_stats(),
        "jobs": {
            "active": sum(1 for s in JOB_STATUS.values() if s in ("pending", "running")),
            "total":  len(JOB_STATUS),
        },
    }


@app.get("/api/v1/admin/stats", tags=["Admin"], summary="Live server statistics")
async def admin_stats(admin: str = Depends(require_admin)):
    """CPU, memory, disk, GPU utilisation and active job counts."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _collect_stats)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok", "version": "1.0.0"}
