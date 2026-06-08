"""
Render a set of camera views from multiple random positions around a 3DGS scene.

Each position does a full panoramic sweep across azimuth and elevation.
Alongside every RGB frame a depth map is rendered and saved:
  <job_dir>/frames/frame_XXXX.png  — RGB
  <job_dir>/frames/depth_XXXX.png  — colorized depth (near=bright, far=dark)
  <job_dir>/frames/depth_XXXX.npy  — raw float depth in world units (for pipeline)

transforms.json gains two extra keys:
  "camera_positions" : [[x,y,z], ...]   one entry per unique camera position
  per-frame          : "position_idx"   which camera position this frame came from
                       "depth_path"     relative path to depth_XXXX.png
"""

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio

from gsplat import rasterization

# Number of camera views rendered in a single GPU call (RGB pass).
# Increase if you have lots of VRAM; decrease if you hit OOM.
RENDER_BATCH = 32
# Parallel threads for PNG / NPY disk writes.
IO_WORKERS = 4


# ---------------------------------------------------------------------------
# PLY loader
# ---------------------------------------------------------------------------

def _load_ply_gaussians(ply_path: str):
    from plyfile import PlyData

    plydata = PlyData.read(ply_path)
    v = plydata["vertex"]

    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    opacities = np.array(v["opacity"], dtype=np.float32)
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)

    sh_r = np.array(v["f_dc_0"], dtype=np.float32)
    sh_g = np.array(v["f_dc_1"], dtype=np.float32)
    sh_b = np.array(v["f_dc_2"], dtype=np.float32)
    sh_dc = np.stack([sh_r, sh_g, sh_b], axis=1)

    try:
        num_extra = sum(1 for name in v.data.dtype.names if name.startswith("f_rest_"))
        if num_extra > 0:
            extra = np.stack(
                [np.array(v[f"f_rest_{i}"], dtype=np.float32) for i in range(num_extra)],
                axis=1,
            )
            K = num_extra // 3
            extra_rgb = np.stack([
                extra[:, :K],
                extra[:, K:2*K],
                extra[:, 2*K:3*K],
            ], axis=2)
            sh_coeffs = np.concatenate([sh_dc[:, None, :], extra_rgb], axis=1)
        else:
            sh_coeffs = sh_dc[:, None, :]
    except Exception:
        sh_coeffs = sh_dc[:, None, :]

    total_sh = sh_coeffs.shape[1]
    sh_degree = int(math.isqrt(total_sh)) - 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "means":     torch.tensor(means,     device=device),
        "quats":     torch.tensor(quats,     device=device),
        "scales":    torch.exp(torch.tensor(scales,    device=device)),
        "opacities": torch.sigmoid(torch.tensor(opacities, device=device)),
        "sh":        torch.tensor(sh_coeffs, device=device),
        "sh_degree": sh_degree,
        "device":    device,
    }


# ---------------------------------------------------------------------------
# Scene bounds
# ---------------------------------------------------------------------------

def _scene_bounds(means: torch.Tensor):
    center = means.mean(dim=0)
    dists = torch.norm(means - center, dim=1)
    radius = dists.quantile(0.80).item()
    return center.cpu().numpy(), max(radius, 0.5)


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _lookat(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """Build c2w in OpenCV convention (X right, Y down, Z forward into scene)."""
    if up is None:
        up = np.array([0.0, 1.0, 0.0])
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
        x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = eye
    return c2w


def _generate_camera_positions(center: np.ndarray, radius: float,
                               n_positions: int, seed: int = 42) -> list:
    """
    Spread camera positions around the interior of the scene at different
    angles and heights so multi-view triangulation is possible.
    """
    rng = np.random.default_rng(seed)
    positions = []
    base_angles = np.linspace(0, 2 * math.pi, n_positions, endpoint=False)
    for angle in base_angles:
        r = rng.uniform(0.45, 0.75) * radius
        jitter = rng.uniform(-0.3, 0.3)
        height_offset = rng.uniform(-0.15, 0.08) * radius
        pos = center + np.array([
            r * math.cos(angle + jitter),
            height_offset,
            r * math.sin(angle + jitter),
        ], dtype=np.float32)
        positions.append(pos)
    return positions


def _build_poses(positions: list, n_azimuth: int, n_elevation: int,
                 el_min: float = -55.0, el_max: float = 40.0):
    """
    From each position do a full panoramic sweep over azimuth and elevation.
    Returns (all_poses, position_indices).
    """
    all_poses = []
    position_indices = []
    elevations = np.linspace(el_min, el_max, n_elevation, dtype=float)

    for pos_idx, pos in enumerate(positions):
        for elev in elevations:
            for i in range(n_azimuth):
                azim = 360.0 * i / n_azimuth
                el_r = math.radians(elev)
                az_r = math.radians(azim)
                look_dir = np.array([
                    math.cos(el_r) * math.sin(az_r),
                    math.sin(el_r),
                    math.cos(el_r) * math.cos(az_r),
                ], dtype=np.float32)
                target = pos + look_dir
                c2w = _lookat(pos, target)
                all_poses.append(c2w)
                position_indices.append(pos_idx)

    return all_poses, position_indices


# ---------------------------------------------------------------------------
# Depth helpers
# ---------------------------------------------------------------------------

def _depth_to_vis(depth_map: np.ndarray) -> np.ndarray:
    """Normalize depth to uint8 grayscale: near=bright (255), far=dark (0)."""
    valid = depth_map[depth_map > 0.01]
    if valid.size == 0:
        return np.zeros(depth_map.shape, dtype=np.uint8)
    d_min, d_max = float(valid.min()), float(depth_map.max())
    if d_max <= d_min:
        return np.full(depth_map.shape, 128, dtype=np.uint8)
    d_norm = np.clip((depth_map - d_min) / (d_max - d_min), 0.0, 1.0)
    return (255 * (1.0 - d_norm)).astype(np.uint8)


def _write_frame(frames_dir: Path, idx: int,
                 rgb: np.ndarray, depth_m: np.ndarray, depth_vis: np.ndarray):
    """Write one frame's RGB PNG, raw depth NPY, and colorized depth PNG. Runs in a thread."""
    imageio.imwrite(str(frames_dir / f"frame_{idx:04d}.png"), rgb)
    np.save(str(frames_dir / f"depth_{idx:04d}.npy"), depth_m)
    imageio.imwrite(str(frames_dir / f"depth_{idx:04d}.png"), depth_vis)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_views(ply_path: str, job_dir: str,
                 width: int = 512, height: int = 512,
                 n_positions: int = 10, n_azimuth: int = 6, n_elevation: int = 3):
    """
    Render RGB + depth from multiple camera positions and write:
      <job_dir>/frames/frame_XXXX.png
      <job_dir>/frames/depth_XXXX.png   (colorized, for display)
      <job_dir>/frames/depth_XXXX.npy   (raw float, for pipeline depth lookup)
      <job_dir>/transforms.json

    RGB frames are rasterized in batches of RENDER_BATCH so the GPU stays
    continuously loaded. Disk writes run in a thread pool overlapping with
    the next GPU batch.
    """
    job_dir = Path(job_dir)
    frames_dir = job_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render] Loading PLY: {ply_path}")
    g = _load_ply_gaussians(ply_path)
    device = g["device"]

    center, radius = _scene_bounds(g["means"])
    print(f"[render] Scene center={center.round(3)}, radius={radius:.3f}")

    # Camera intrinsics — 130° horizontal FoV
    fov_x = math.radians(130.0)
    fl_x = width / (2.0 * math.tan(fov_x / 2.0))
    fl_y = fl_x
    cx, cy = width / 2.0, height / 2.0

    K = torch.tensor(
        [[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]],
        dtype=torch.float32, device=device,
    )

    cam_positions = _generate_camera_positions(center, radius, n_positions)
    poses, position_indices = _build_poses(cam_positions, n_azimuth, n_elevation)
    total = len(poses)
    print(f"[render] {n_positions} positions × {n_azimuth} azimuth × {n_elevation} elevation = {total} views"
          f"  (batch={RENDER_BATCH}, io_workers={IO_WORKERS}, device={device})")

    # Build all w2c matrices on GPU in one shot — avoids per-frame tensor creation
    all_c2w = torch.tensor(np.stack(poses), device=device, dtype=torch.float32)  # (T, 4, 4)
    all_w2c = torch.linalg.inv(all_c2w)                                           # (T, 4, 4)

    # Homogeneous Gaussian means for depth z_cam = (w2c @ means_h.T)[2]
    N = g["means"].shape[0]
    means_h = torch.cat(
        [g["means"], torch.ones(N, 1, device=device)], dim=1
    ).contiguous()  # (N, 4)

    transforms = {
        "fl_x": fl_x, "fl_y": fl_y, "cx": cx, "cy": cy,
        "w": width, "h": height,
        "scene_center": center.tolist(),
        "scene_radius": float(radius),
        "camera_positions": [p.tolist() for p in cam_positions],
        "frames": [],
    }

    SH_C0 = 0.28209479177387814
    futures = []

    with ThreadPoolExecutor(max_workers=IO_WORKERS) as pool:
        for b0 in range(0, total, RENDER_BATCH):
            b1 = min(b0 + RENDER_BATCH, total)
            B = b1 - b0
            print(f"[render]  {b0}/{total} …")

            w2c_batch = all_w2c[b0:b1].contiguous()                        # (B, 4, 4)
            K_batch   = K.unsqueeze(0).expand(B, -1, -1).contiguous()      # (B, 3, 3)

            # ── Batched RGB render ─────────────────────────────────────────────
            # Single GPU rasterization call for all B views simultaneously.
            rgb_out, _, _ = rasterization(
                means=g["means"], quats=g["quats"], scales=g["scales"],
                opacities=g["opacities"], colors=g["sh"],
                viewmats=w2c_batch, Ks=K_batch,
                width=width, height=height,
                sh_degree=g["sh_degree"],
                near_plane=0.01, far_plane=1000.0,
            )
            # Move all B frames to CPU in one transfer
            rgb_cpu = (rgb_out.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # (B, H, W, 3)

            # ── Depth renders ──────────────────────────────────────────────────
            # Camera-space Z for every Gaussian across all B views in one bmm:
            # (B, 4, 4) @ (B, 4, N) → (B, 4, N) → row 2 → (B, N)
            z_batch = torch.bmm(
                w2c_batch,
                means_h.T.unsqueeze(0).expand(B, -1, -1).contiguous(),
            )[:, 2, :].clamp(min=0.01).contiguous()  # (B, N)

            # Depth rasterization is view-dependent (z_cam varies per view),
            # so we call rasterization once per view — but z_cam is already on GPU.
            depth_cpu = []
            for bi in range(B):
                dc = z_batch[bi].view(-1, 1, 1).expand(-1, 1, 3).contiguous()  # (N, 1, 3)
                dr, _, _ = rasterization(
                    means=g["means"], quats=g["quats"], scales=g["scales"],
                    opacities=g["opacities"], colors=dc,
                    viewmats=w2c_batch[bi:bi+1], Ks=K_batch[bi:bi+1],
                    width=width, height=height, sh_degree=0,
                    near_plane=0.01, far_plane=1000.0,
                )
                dm = np.maximum(
                    (dr[0, :, :, 0].cpu().numpy() - 0.5) / SH_C0, 0.0
                ).astype(np.float32)
                depth_cpu.append(dm)

            # ── Submit I/O to thread pool (overlaps with next GPU batch) ───────
            for bi, idx in enumerate(range(b0, b1)):
                futures.append(pool.submit(
                    _write_frame, frames_dir, idx,
                    rgb_cpu[bi], depth_cpu[bi], _depth_to_vis(depth_cpu[bi]),
                ))
                transforms["frames"].append({
                    "file_path":        f"frames/frame_{idx:04d}.png",
                    "depth_path":       f"frames/depth_{idx:04d}.png",
                    "transform_matrix": poses[idx].tolist(),
                    "position_idx":     int(position_indices[idx]),
                })

        # Drain futures — re-raises any disk write errors
        for f in as_completed(futures):
            f.result()

    transforms_path = job_dir / "transforms.json"
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"[render] Done. Wrote {total} RGB + depth frames + transforms.json")
    return str(transforms_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply",         required=True)
    parser.add_argument("--job_dir",     required=True)
    parser.add_argument("--width",       type=int, default=512)
    parser.add_argument("--height",      type=int, default=512)
    parser.add_argument("--n_positions", type=int, default=5)
    parser.add_argument("--n_azimuth",   type=int, default=6)
    parser.add_argument("--n_elevation", type=int, default=3)
    args = parser.parse_args()
    render_views(args.ply, args.job_dir, args.width, args.height,
                 args.n_positions, args.n_azimuth, args.n_elevation)
