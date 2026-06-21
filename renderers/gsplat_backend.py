"""
gsplat (nerfstudio, CUDA) backend — the deployed server + CUDA desktops.

This is the original rasterization path extracted verbatim from render_cameras.py:
a single batched `rasterization` call for RGB, and the per-view depth-as-color trick
(camera-space Z passed as a degree-0 SH "color", then decoded with SH_C0).
"""

from __future__ import annotations

import numpy as np
import torch

from gsplat import rasterization

from .base import Renderer, NEAR_PLANE, FAR_PLANE, SH_C0


class GsplatRenderer(Renderer):
    name = "gsplat"

    def __init__(self):
        if not torch.cuda.is_available():
            # gsplat's rasterization is a compiled CUDA kernel; it cannot run on CPU.
            raise RuntimeError(
                "gsplat backend requires CUDA. On Apple Silicon use renderer='gsplat-metal'."
            )
        self.device = torch.device("cuda")

    def prepare(self, arrays):
        # gsplat 1.5.3 rasterization() wants activated scales + opacities.
        g = self._to_device(arrays)
        g["scales"] = torch.exp(g["log_scales"])
        g["opacities"] = torch.sigmoid(g["logit_opacities"])
        return g

    def render_rgb(self, g, w2c, K, width, height):
        B = w2c.shape[0]
        K_batch = K.unsqueeze(0).expand(B, -1, -1).contiguous()
        rgb_out, _, _ = rasterization(
            means=g["means"], quats=g["quats"], scales=g["scales"],
            opacities=g["opacities"], colors=g["sh"],
            viewmats=w2c, Ks=K_batch,
            width=width, height=height,
            sh_degree=g["sh_degree"],
            near_plane=NEAR_PLANE, far_plane=FAR_PLANE,
        )
        return (rgb_out.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # (B,H,W,3)

    def render_depth(self, g, w2c, K, width, height):
        B = w2c.shape[0]
        N = g["means"].shape[0]
        K_batch = K.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Camera-space Z for every Gaussian across all B views in one bmm.
        means_h = torch.cat(
            [g["means"], torch.ones(N, 1, device=self.device)], dim=1
        ).contiguous()  # (N, 4)
        z_batch = torch.bmm(
            w2c, means_h.T.unsqueeze(0).expand(B, -1, -1).contiguous(),
        )[:, 2, :].clamp(min=NEAR_PLANE).contiguous()  # (B, N)

        depth = []
        for bi in range(B):
            dc = z_batch[bi].view(-1, 1, 1).expand(-1, 1, 3).contiguous()  # (N,1,3)
            dr, _, _ = rasterization(
                means=g["means"], quats=g["quats"], scales=g["scales"],
                opacities=g["opacities"], colors=dc,
                viewmats=w2c[bi:bi + 1], Ks=K_batch[bi:bi + 1],
                width=width, height=height, sh_degree=0,
                near_plane=NEAR_PLANE, far_plane=FAR_PLANE,
            )
            dm = np.maximum(
                (dr[0, :, :, 0].cpu().numpy() - 0.5) / SH_C0, 0.0
            ).astype(np.float32)
            depth.append(dm)
        return depth
