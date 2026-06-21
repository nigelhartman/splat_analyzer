"""
Renderer interface + shared Gaussian preparation.

A backend takes the raw arrays loaded from a .ply (means, quats wxyz, log-scales,
logit-opacities, SH coeffs) and renders a batch of camera views to RGB + depth.

Conventions (shared by every backend — must match `render_cameras.py`):
  • quaternions wxyz
  • poses are c2w (OpenCV: X right, Y down, Z forward); the backend receives w2c
  • intrinsics K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
  • depth is camera-space Z in world units, 0 where nothing was hit
"""

from __future__ import annotations

import numpy as np
import torch

# Rasterization clip planes (shared so every backend matches).
NEAR_PLANE = 0.01
FAR_PLANE = 1000.0
# Degree-0 spherical-harmonic constant, used by the depth-as-color trick.
SH_C0 = 0.28209479177387814


class Renderer:
    """Base class for a Gaussian-Splat rasterizer backend.

    Subclasses set ``name`` and ``device`` and implement ``render_rgb`` /
    ``render_depth``. ``prepare`` is shared: it applies the standard activations
    (exp on scales, sigmoid on opacities) and moves tensors onto the backend device.
    """

    name: str = "base"
    device: torch.device

    def _to_device(self, arrays: dict) -> dict:
        """Move raw .ply arrays onto the backend device, WITHOUT activations.

        Activations (exp on scales, sigmoid on opacity) differ per backend —
        gsplat (1.5.3) wants exp'd scales, gsplat-mps (0.1.x) wants raw log-scales
        with glob_scale=1 — so they are applied in each backend's ``prepare``.

        ``arrays`` keys: means (N,3), quats (N,4 wxyz), scales (N,3 log),
        opacities (N, logit), sh_coeffs (N,K,3), sh_degree (int).
        """
        d = self.device
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=d)
        return {
            "means":           t(arrays["means"]),
            "quats":           t(arrays["quats"]),
            "log_scales":      t(arrays["scales"]),      # raw log-scales
            "logit_opacities": t(arrays["opacities"]),   # raw logit opacities
            "sh":              t(arrays["sh_coeffs"]),
            "sh_degree":       int(arrays["sh_degree"]),
            "device":          d,
        }

    def prepare(self, arrays: dict) -> dict:
        """Return render-ready tensors on the backend device. Backends override
        to apply their activation convention; base just moves to device."""
        return self._to_device(arrays)

    def render_rgb(self, g: dict, w2c: torch.Tensor, K: torch.Tensor,
                   width: int, height: int) -> np.ndarray:
        """Render a batch of B views to RGB. Returns (B, H, W, 3) uint8."""
        raise NotImplementedError

    def render_depth(self, g: dict, w2c: torch.Tensor, K: torch.Tensor,
                     width: int, height: int) -> list[np.ndarray]:
        """Render depth for a batch of B views. Returns list of B (H, W) float32
        arrays, camera-space Z in world units (0 = no hit)."""
        raise NotImplementedError
