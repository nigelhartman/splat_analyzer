"""
Render a set of spherical camera views around a 3D Gaussian Splat scene.

Outputs:
  <job_dir>/frames/frame_XXXX.png   — rendered RGB images
  <job_dir>/transforms.json         — NeRF-style camera metadata
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio

# ---------------------------------------------------------------------------
# gsplat imports
# ---------------------------------------------------------------------------
from gsplat import rasterization


def _load_ply_gaussians(ply_path: str):
    """Load 3DGS .ply into tensors required by gsplat.rasterization."""
    from plyfile import PlyData

    plydata = PlyData.read(ply_path)
    v = plydata["vertex"]

    means = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)

    # Opacity (logit-encoded in 3DGS files)
    opacities = np.array(v["opacity"], dtype=np.float32)

    # Scales (log-encoded)
    scales = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1).astype(np.float32)

    # Quaternions (wxyz in PLY)
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float32)

    # Spherical harmonics — DC term only (colours)
    sh_r = np.array(v["f_dc_0"], dtype=np.float32)
    sh_g = np.array(v["f_dc_1"], dtype=np.float32)
    sh_b = np.array(v["f_dc_2"], dtype=np.float32)
    sh_dc = np.stack([sh_r, sh_g, sh_b], axis=1)  # (N, 3)

    # Check for higher-degree SH coefficients
    try:
        num_extra = sum(1 for name in v.data.dtype.names if name.startswith("f_rest_"))
        if num_extra > 0:
            extra = np.stack(
                [np.array(v[f"f_rest_{i}"], dtype=np.float32) for i in range(num_extra)],
                axis=1,
            )  # (N, num_extra)
            # 3DGS stores SH rest coefficients channel-first: all R, then all G, then all B
            K = num_extra // 3  # rest coefficients per channel
            extra_rgb = np.stack([
                extra[:, :K],        # R coefficients
                extra[:, K:2*K],     # G coefficients
                extra[:, 2*K:3*K],   # B coefficients
            ], axis=2)               # (N, K, 3)
            sh_coeffs = np.concatenate([sh_dc[:, None, :], extra_rgb], axis=1)  # (N, K+1, 3)
        else:
            sh_coeffs = sh_dc[:, None, :]  # (N, 1, 3)
    except Exception:
        sh_coeffs = sh_dc[:, None, :]

    # Determine SH degree from total coefficient count: (degree+1)^2 == total
    total_sh = sh_coeffs.shape[1]
    sh_degree = int(math.isqrt(total_sh)) - 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return {
        "means": torch.tensor(means, device=device),
        "quats": torch.tensor(quats, device=device),
        "scales": torch.exp(torch.tensor(scales, device=device)),
        "opacities": torch.sigmoid(torch.tensor(opacities, device=device)),
        "sh": torch.tensor(sh_coeffs, device=device),
        "sh_degree": sh_degree,
        "device": device,
    }


def _scene_bounds(means: torch.Tensor):
    """Return (center, radius) using the inner 90% of points to ignore floater outliers."""
    center = means.mean(dim=0)
    dists = torch.norm(means - center, dim=1)
    # Use 80th-percentile radius so cameras orbit outside most real content
    # but aren't blown out by stray floater Gaussians at the reconstruction boundary
    radius = dists.quantile(0.80).item()
    return center.cpu().numpy(), max(radius, 0.5)


def _lookat(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """
    Construct a camera-to-world (c2w) matrix in OpenCV convention:
      X = right, Y = down (in image), Z = forward (into scene).
    gsplat's rasterization expects viewmats (w2c) in this convention,
    so Z must point toward the scene, not away from it.
    """
    if up is None:
        up = np.array([0.0, 1.0, 0.0])
    z = target - eye          # forward INTO the scene (+Z in OpenCV camera space)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)       # right
    if np.linalg.norm(x) < 1e-6:
        up = np.array([0.0, 0.0, 1.0])
        x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)        # down in image = world +Y (floor direction); correct OpenCV Y-down
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = eye
    return c2w


def _spherical_poses(center, radius, n_azimuth=12, n_elevation=3, dist_mult=3.0):
    """
    For indoor/room-scale scenes: place cameras INSIDE at the scene center
    and look outward in all directions (panoramic sweep).
    For object-scale scenes (dist_mult > 1): orbit outside looking inward.

    We detect indoor vs. object-scale by whether dist_mult < 1.5.
    Default dist_mult=3.0 → outdoor orbit.
    Call with dist_mult=0.0 to force the interior panoramic mode.
    """
    poses = []
    # Interior panoramic mode: cameras at center, looking outward at walls
    # eye slightly below scene center (approximates eye level if Y is up)
    eye_offset = np.array([0.0, -radius * 0.1, 0.0], dtype=np.float32)
    interior_eye = center + eye_offset

    elevations = np.linspace(-15, 15, n_elevation, dtype=float)
    for elev in elevations:
        for i in range(n_azimuth):
            azim = 360.0 * i / n_azimuth
            el_r = math.radians(elev)
            az_r = math.radians(azim)
            # Look outward from eye toward the walls/objects at radius distance
            look_dir = np.array([
                math.cos(el_r) * math.sin(az_r),
                math.sin(el_r),
                math.cos(el_r) * math.cos(az_r),
            ], dtype=np.float32)
            target = interior_eye + radius * look_dir
            c2w = _lookat(interior_eye, target)
            poses.append(c2w)
    return poses


def render_views(ply_path: str, job_dir: str, width: int = 512, height: int = 512):
    """
    Main entry point. Renders camera views and writes:
      <job_dir>/frames/frame_XXXX.png
      <job_dir>/transforms.json
    Returns path to transforms.json.
    """
    job_dir = Path(job_dir)
    frames_dir = job_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render] Loading PLY: {ply_path}")
    g = _load_ply_gaussians(ply_path)
    device = g["device"]

    center, radius = _scene_bounds(g["means"])
    print(f"[render] Scene center={center}, radius={radius:.3f}")

    # Camera intrinsics: 90° horizontal FoV — wide enough to see full walls/objects
    fov_x = math.radians(90.0)
    fl_x = width / (2.0 * math.tan(fov_x / 2.0))
    fl_y = fl_x  # square pixels
    cx, cy = width / 2.0, height / 2.0

    K = torch.tensor(
        [[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )

    poses = _spherical_poses(center, radius, n_azimuth=12, n_elevation=3)
    print(f"[render] Rendering {len(poses)} views …")

    transforms = {
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "scene_center": center.tolist(),
        "scene_radius": float(radius),
        "frames": [],
    }

    for idx, c2w in enumerate(poses):
        c2w_t = torch.tensor(c2w, device=device).unsqueeze(0)  # (1,4,4)

        # gsplat expects world-to-camera (w2c) as (1,4,4)
        w2c = torch.linalg.inv(c2w_t)

        render_colors, render_alphas, info = rasterization(
            means=g["means"],
            quats=g["quats"],
            scales=g["scales"],
            opacities=g["opacities"],
            colors=g["sh"],             # (N, K, 3) full SH coefficients
            viewmats=w2c,
            Ks=K.unsqueeze(0),
            width=width,
            height=height,
            sh_degree=g["sh_degree"],   # use actual degree from PLY (e.g. 3)
            near_plane=0.01,
            far_plane=1000.0,
        )

        # render_colors: (1, H, W, 3) float [0,1]
        img = (render_colors[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        frame_path = frames_dir / f"frame_{idx:04d}.png"
        imageio.imwrite(str(frame_path), img)

        transforms["frames"].append({
            "file_path": f"frames/frame_{idx:04d}.png",
            "transform_matrix": c2w.tolist(),
        })

    transforms_path = job_dir / "transforms.json"
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"[render] Done. Wrote {len(poses)} frames + transforms.json")
    return str(transforms_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--job_dir", required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()
    render_views(args.ply, args.job_dir, args.width, args.height)
