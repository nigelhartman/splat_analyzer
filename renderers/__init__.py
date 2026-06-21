"""
Renderer backends — pluggable Gaussian-Splat rasterizers behind one interface.

The pipeline's camera placement, pose generation, depth-to-disk and transforms.json
authoring all live in `render_cameras.py` and are renderer-independent. Only the
actual rasterization of a batch of views is backend-specific, and that is what a
`Renderer` provides.

Backends
--------
- ``gsplat``        nerfstudio gsplat (CUDA). The deployed server + CUDA desktops.
- ``gsplat-metal``  a gsplat-API-compatible Metal/MPS fork. Apple-Silicon Macs.
- ``auto``          pick gsplat if CUDA is available, else gsplat-metal if MPS is
                    available, else error (CPU rasterization is not supported).

All backends consume the same activated Gaussian tensors (scales already exp'd,
opacities already sigmoid'd, SH coeffs raw) and return the same outputs, so
`render_cameras.py` is identical across platforms.
"""

from __future__ import annotations

from .base import Renderer, NEAR_PLANE, FAR_PLANE, SH_C0


def get_renderer(name: str = "auto") -> Renderer:
    """Resolve a renderer name to a constructed backend.

    ``auto`` resolves to gsplat on CUDA, gsplat-metal on Apple MPS, otherwise raises.
    """
    import torch

    requested = (name or "auto").lower()
    resolved = requested
    if requested == "auto":
        if torch.cuda.is_available():
            resolved = "gsplat"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            resolved = "gsplat-metal"
        else:
            raise RuntimeError(
                "renderer='auto' found neither CUDA nor Apple MPS. Gaussian-Splat "
                "rasterization needs a GPU — there is no CPU backend. Run on the CUDA "
                "server, a CUDA desktop, or an Apple-Silicon Mac."
            )

    if resolved in ("gsplat", "cuda"):
        from .gsplat_backend import GsplatRenderer
        return GsplatRenderer()
    if resolved in ("gsplat-metal", "metal", "mps"):
        from .gsplat_metal_backend import GsplatMetalRenderer
        return GsplatMetalRenderer()

    raise ValueError(f"unknown renderer: {name!r} (resolved to {resolved!r})")


__all__ = ["Renderer", "get_renderer", "NEAR_PLANE", "FAR_PLANE", "SH_C0"]
