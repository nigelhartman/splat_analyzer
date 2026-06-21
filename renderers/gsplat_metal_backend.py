"""
gsplat-Metal (Apple Silicon / MPS) backend — for local rendering on M-series Macs.

Uses gsplat-mps (https://github.com/iffyloop/gsplat-mps, `opensplat-mps` branch), a
Metal/MPS port of gsplat v0.1.x. It installs under the package name ``gsplat`` (same as
the CUDA gsplat), so only one is present per machine — this module is imported lazily by
`renderers.get_renderer` only when the gsplat-metal backend is selected.

API differences vs. the CUDA gsplat 1.5.3 path (handled here):
  • 0.1.x SPLIT API: project_gaussians → spherical_harmonics → rasterize_gaussians,
    ONE view per call (no batch dim) — we loop over the batch.
  • scales passed RAW (log-scales) with glob_scale=1.0; the kernel exponentiates.
  • SH→RGB is a separate step we do ourselves (+0.5 DC offset, then clamp).
  • depth: rasterize_gaussians takes arbitrary channel counts, so we composite the
    per-Gaussian camera-space Z directly — no SH_C0 decode (cleaner than the CUDA trick).

⚠️ First import JIT-compiles a Metal extension (needs Xcode command-line tools; slow once).
⚠️ gsplat-mps is AGPLv3 — note for distribution. Smoke-test on a real M-series Mac.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import spherical_harmonics

from .base import Renderer, NEAR_PLANE

# Cull splats whose projected 3-sigma radius exceeds this fraction of the larger image
# dimension. gsplat 1.5.3 tames huge 2D radii; this 0.1.x fork does not, so a few
# near-camera / floater splats smear into streaks. Set None to disable.
MAX_RADIUS_FRAC = 1.0


class GsplatMetalRenderer(Renderer):
    name = "gsplat-metal"

    def __init__(self):
        mps = getattr(torch.backends, "mps", None)
        if not (mps and torch.backends.mps.is_available()):
            raise RuntimeError(
                "gsplat-metal backend requires Apple MPS (Metal). Use renderer='gsplat' on CUDA."
            )
        self.device = torch.device("mps")

    def prepare(self, arrays):
        # gsplat-mps's kernel uses LINEAR scales (scale_to_mat does not exp), so we
        # exp here just like the CUDA path; glob_scale stays 1.0. Opacity sigmoid'd (N,1).
        g = self._to_device(arrays)
        g["scales"] = torch.exp(g["log_scales"])
        g["opacities"] = torch.sigmoid(g["logit_opacities"]).reshape(-1, 1)
        return g

    @staticmethod
    def _intrinsics(K, width, height):
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        tile_bounds = ((width + 15) // 16, (height + 15) // 16, 1)
        return fx, fy, cx, cy, tile_bounds

    def _project(self, g, w2c_view, fx, fy, cx, cy, width, height, tile_bounds):
        # gsplat 0.1.x needs projmat = P @ w2c, where P is the OpenGL perspective built
        # from the intrinsics. project_pix perspective-divides by p_hom.w (must be z_cam)
        # and ndc2pix maps the result from NDC[-1,1] to pixels. Passing w2c alone (as the
        # fork's example does) gives w=1 → no divide/no focal → a degenerate ortho smear.
        # The principal point is applied by the kernel's ndc2pix(pp=cx,cy), so P omits cx/cy.
        P = torch.tensor([
            [2.0 * fx / width, 0.0,               0.0, 0.0],
            [0.0,              2.0 * fy / height, 0.0, 0.0],
            [0.0,              0.0,               1.0, 0.0],
            [0.0,              0.0,               1.0, 0.0],
        ], dtype=torch.float32, device=self.device)
        projmat = P @ w2c_view
        return project_gaussians(
            g["means"], g["scales"], 1.0, g["quats"],
            w2c_view, projmat,
            fx, fy, cx, cy, height, width, tile_bounds, NEAR_PLANE,
        )

    def _clamp_opacity(self, opacities, radii, width, height):
        """Zero the opacity of splats whose 2D radius exceeds the cap (#2: kill streaks).
        Returns opacities unchanged if MAX_RADIUS_FRAC is None."""
        if MAX_RADIUS_FRAC is None:
            return opacities
        cap = MAX_RADIUS_FRAC * max(width, height)
        keep = (radii.reshape(-1, 1) <= cap).to(opacities.dtype)
        return opacities * keep

    def render_rgb(self, g, w2c, K, width, height):
        fx, fy, cx, cy, tile_bounds = self._intrinsics(K, width, height)
        c2w = torch.linalg.inv(w2c)               # (B,4,4)
        centers = c2w[:, :3, 3]                   # (B,3) camera positions
        bg = torch.zeros(3, device=self.device)
        out = []
        for bi in range(w2c.shape[0]):
            xys, depths, radii, conics, n_hit, _ = self._project(
                g, w2c[bi], fx, fy, cx, cy, width, height, tile_bounds)
            op = self._clamp_opacity(g["opacities"], radii, width, height)
            viewdirs = F.normalize(g["means"] - centers[bi], dim=-1)
            colors = (spherical_harmonics(g["sh_degree"], viewdirs, g["sh"]) + 0.5).clamp(0, 1)
            rgb = rasterize_gaussians(
                xys, depths, radii, conics, n_hit, colors, op,
                height, width, bg)                # (H,W,3)
            out.append((rgb.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8))
        return np.stack(out)                       # (B,H,W,3)

    def render_depth(self, g, w2c, K, width, height):
        fx, fy, cx, cy, tile_bounds = self._intrinsics(K, width, height)
        bg1 = torch.zeros(1, device=self.device)
        out = []
        for bi in range(w2c.shape[0]):
            xys, depths, radii, conics, n_hit, _ = self._project(
                g, w2c[bi], fx, fy, cx, cy, width, height, tile_bounds)
            op = self._clamp_opacity(g["opacities"], radii, width, height)
            # Composite per-Gaussian camera-space Z as a 1-channel "color", AND get alpha.
            z = depths.reshape(-1, 1)
            dimg, alpha = rasterize_gaussians(
                xys, depths, radii, conics, n_hit, z, op,
                height, width, bg1, return_alpha=True)   # (H,W,1), (H,W[,1])
            # #1: normalize by accumulated alpha → true expected depth (no shallow bias).
            dnum = dimg[..., 0]                          # (H,W)
            a = alpha.reshape(height, width)
            depth = torch.where(a > 1e-6, dnum / a.clamp(min=1e-6),
                                torch.zeros_like(dnum))
            out.append(depth.detach().cpu().numpy().astype(np.float32))
        return out
