"""
Shared pipeline configuration — the single source of truth for defaults used by
both the local CLI (run_local.py / pipeline.py) and the FastAPI server (server.py).

Keeping every default here prevents the kind of drift that crept in before
(e.g. n_positions defaulting to 10 in one place and 5 in another).

The user-facing knob is `quality` (low/medium/high), which expands to the three
internal camera counts via `apply_quality()`. The granular counts remain on the
dataclass for internal use but are not exposed in the UI/API.
"""

from dataclasses import dataclass, fields


# quality name -> (n_positions, n_azimuth, n_elevation)  → total frames
QUALITY_PRESETS = {
    "low":    (3, 4, 2),   # 24 frames  — fast preview
    "medium": (5, 6, 3),   # 90 frames  — balanced default
    "high":   (8, 8, 3),   # 192 frames — thorough coverage
}
DEFAULT_QUALITY = "medium"

# Scene up-axis conventions. Standard 3DGS/COLMAP scenes store +Y toward the
# floor ("y-down"); some exporters produce Y-up scenes, which render upside
# down unless world_up="y-up".
WORLD_UP_CHOICES = ("y-down", "y-up")


@dataclass
class PipelineConfig:
    # ── rendering ──────────────────────────────────────────────────────────
    width: int = 512
    height: int = 512
    renderer: str = "auto"                  # "auto" | "gsplat" (CUDA) | "gsplat-metal" (Apple MPS)
    world_up: str = "y-down"                # scene up-axis: "y-down" (standard 3DGS/COLMAP) | "y-up"
    quality: str = DEFAULT_QUALITY          # drives the camera counts below
    n_positions: int = 5
    n_azimuth: int = 6
    n_elevation: int = 3

    # ── camera placement (density-aware band-pass sampler) ─────────────────
    bbox_pct_lo: float = 1.0                # robust bounding-box lower percentile
    bbox_pct_hi: float = 99.0               # robust bounding-box upper percentile
    min_sep_frac: float = 0.12              # Poisson-disk min camera separation (frac of bbox diagonal)
    density_radius_frac: float = 0.05       # neighbour-count radius (frac of bbox diagonal)
    bandpass_alpha: float = 1.0             # band-pass width; larger = more permissive
    seed: int = 42                          # deterministic placement

    # ── detection / clustering ─────────────────────────────────────────────
    score_threshold: float = 0.12
    min_votes: int = 8
    min_peak_score: float = 0.40

    def apply_quality(self):
        """Expand the `quality` preset into the three camera-count fields."""
        if self.quality in QUALITY_PRESETS:
            self.n_positions, self.n_azimuth, self.n_elevation = QUALITY_PRESETS[self.quality]
        return self

    @classmethod
    def from_overrides(cls, **kw):
        """Build a config from a loose dict of overrides (ignores unknown/None keys)."""
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kw.items() if k in valid and v is not None}
        return cls(**filtered).apply_quality()
