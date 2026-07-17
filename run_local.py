"""
Local one-command runner for the WorldModelData pipeline.

Runs the *exact same* rendering + detection pipeline the server uses (no
duplicated logic — this is a thin front door over pipeline.run_pipeline) on a
machine with an NVIDIA CUDA GPU.

Example:
  python run_local.py --ply scene.ply --prompt "chair, table" --quality high

Outputs <job_dir>/interactions.json plus the rendered frames/ and transforms.json.
"""

import argparse
import sys
from pathlib import Path

import torch

import pipeline
from config import PipelineConfig, QUALITY_PRESETS, DEFAULT_QUALITY, WORLD_UP_CHOICES


def main():
    d = PipelineConfig()
    p = argparse.ArgumentParser(description="Run 3DGS object detection locally.")
    p.add_argument("--ply",     required=True, help="Path to a .ply or .spz Gaussian Splat file")
    p.add_argument("--prompt",  required=True, help='Comma-separated labels, e.g. "chair, table"')
    p.add_argument("--quality", choices=list(QUALITY_PRESETS.keys()), default=DEFAULT_QUALITY,
                   help="Camera-coverage preset (controls number of views)")
    p.add_argument("--job_dir", default=None, help="Output directory (default: ./out_<name>)")
    p.add_argument("--world_up", choices=list(WORLD_UP_CHOICES), default=d.world_up,
                   help="Scene up-axis: y-down (standard 3DGS/COLMAP) or y-up "
                        "(for scenes that otherwise render upside down)")
    p.add_argument("--score_threshold", type=float, default=d.score_threshold)
    p.add_argument("--min_votes",       type=int,   default=d.min_votes)
    p.add_argument("--min_peak_score",  type=float, default=d.min_peak_score)
    args = p.parse_args()

    if not Path(args.ply).exists():
        sys.exit(f"error: file not found: {args.ply}")

    if not torch.cuda.is_available():
        print("WARNING: no CUDA GPU detected — running on CPU will be extremely slow.\n"
              "         Install a CUDA-matched build of torch + gsplat for usable speed.",
              file=sys.stderr)

    job_dir = args.job_dir or f"out_{Path(args.ply).stem}"
    cfg = PipelineConfig.from_overrides(
        quality=args.quality,
        world_up=args.world_up,
        score_threshold=args.score_threshold,
        min_votes=args.min_votes,
        min_peak_score=args.min_peak_score,
    )

    objects = pipeline.run_pipeline(args.ply, args.prompt, job_dir, cfg)

    out = Path(job_dir)
    print(f"\nDone — {len(objects)} object(s) detected.")
    print(f"  results: {out / 'interactions.json'}")
    print(f"  frames:  {out / 'frames'}")


if __name__ == "__main__":
    main()
