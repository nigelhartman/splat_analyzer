"""
Master pipeline:
  1. Render spherical camera views from the .ply (render_cameras.py)
  2. Run OWLv2 open-vocabulary detection on every frame
  3. Back-project 2D boxes to 3D using camera matrices
  4. Cluster per-label detections → single 3D bounding box per object
  5. Write interactions.json in WebXR format

Usage:
  python pipeline.py --ply scene.ply --prompt "chair, desk" --job_dir /tmp/job123
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

import render_cameras


# ---------------------------------------------------------------------------
# 2-D → 3-D lifting helpers
# ---------------------------------------------------------------------------

def _unproject_box(box_2d, depth, K_inv, c2w):
    """
    Unproject a 2-D bounding box centre to a 3-D ray and place it at `depth`
    along that ray in world space.

    Args:
        box_2d: (x1, y1, x2, y2) in pixels
        depth: estimated depth (scene radius used as proxy)
        K_inv: (3,3) inverse camera intrinsic matrix (numpy)
        c2w:   (4,4) camera-to-world matrix (numpy)

    Returns:
        world_pos: (3,) xyz in world space
        box_wh:    (width_pixels, height_pixels) — used for scale estimation
    """
    x1, y1, x2, y2 = box_2d
    cx_px = (x1 + x2) / 2.0
    cy_px = (y1 + y2) / 2.0
    w_px = x2 - x1
    h_px = y2 - y1

    # Pixel → normalised camera ray
    p_cam_norm = K_inv @ np.array([cx_px, cy_px, 1.0])
    ray_cam = p_cam_norm / np.linalg.norm(p_cam_norm)

    # Scale to `depth` along the Z axis of the camera
    point_cam = ray_cam * (depth / ray_cam[2])

    # Camera → world
    point_world = (c2w[:3, :3] @ point_cam) + c2w[:3, 3]

    return point_world, (w_px, h_px)


def _pixel_size_to_world(w_px, h_px, depth, fl_x, fl_y):
    """Convert pixel bounding-box dimensions to metric world-space dimensions."""
    world_w = (w_px / fl_x) * depth
    world_h = (h_px / fl_y) * depth
    return world_w, world_h


def _cluster_detections(detections, eps_m=0.5, max_per_label=2,
                        min_votes=3, min_peak_score=0.35):
    """
    Score-weighted greedy clustering with false-positive suppression.

    A cluster is only kept when BOTH conditions hold:
      - min_votes  : at least this many frames contributed a detection at this location.
                     Real objects are seen consistently from multiple angles; stray
                     false-positives typically appear in only 1-2 frames.
      - min_peak_score : the single strongest detection in the cluster must exceed
                     this threshold. Filters clusters seeded only by borderline scores.

    Returns at most max_per_label clusters per label, ranked by peak score.
    """
    by_label = defaultdict(list)
    for det in detections:
        by_label[det["label"]].append(det)

    results = []
    for label, dets in by_label.items():
        # Seed clusters from the highest-scoring detections first
        dets = sorted(dets, key=lambda d: d["score"], reverse=True)
        positions = np.array([d["position"] for d in dets])
        scales    = np.array([d["scale"] for d in dets])
        scores    = np.array([d["score"] for d in dets])

        used = [False] * len(dets)
        clusters = []
        for i in range(len(dets)):
            if used[i]:
                continue
            cluster_idx = [i]
            centroid = positions[i].copy()
            for j in range(i + 1, len(dets)):
                if not used[j] and np.linalg.norm(centroid - positions[j]) < eps_m:
                    cluster_idx.append(j)
                    w = scores[cluster_idx]
                    centroid = (positions[cluster_idx] * w[:, None]).sum(0) / w.sum()
            for j in cluster_idx:
                used[j] = True

            peak_score = float(scores[cluster_idx].max())
            vote_count = len(cluster_idx)

            # Skip clusters that are too sparse or too low-confidence
            if vote_count < min_votes or peak_score < min_peak_score:
                continue

            cluster_pos   = (positions[cluster_idx] * scores[cluster_idx, None]).sum(0) / scores[cluster_idx].sum()
            cluster_scale = np.median(scales[cluster_idx], axis=0)
            cluster_scale = np.maximum(cluster_scale, 0.1)
            clusters.append((peak_score, vote_count, label, cluster_pos, cluster_scale))

        # Keep top-N by peak score
        clusters.sort(key=lambda c: c[0], reverse=True)
        for peak, votes, lbl, pos, scale in clusters[:max_per_label]:
            print(f"  [cluster] {lbl}: {votes} votes, peak={peak:.2f}, pos={pos.round(2)}")
            results.append((lbl, pos, scale))

    return results


# ---------------------------------------------------------------------------
# OWLv2 detection
# ---------------------------------------------------------------------------

def _run_owlv2(frames_dir: Path, labels: list[str], transforms: dict, scene_radius: float):
    """
    Run OWLv2 on all rendered frames.

    Returns list of raw detection dicts with 3D position/scale.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[pipeline] Loading OWLv2 on {device} …")
    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device)
    model.eval()

    fl_x = transforms["fl_x"]
    fl_y = transforms["fl_y"]
    cx  = transforms["cx"]
    cy  = transforms["cy"]
    W   = transforms["w"]
    H   = transforms["h"]

    K = np.array([
        [fl_x, 0,    cx],
        [0,    fl_y, cy],
        [0,    0,    1.0],
    ], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    # Texts must be one set per image when batching; we do one image at a time
    texts = [[f"a photo of a {lbl.strip()}" for lbl in labels]]

    raw_detections = []
    score_threshold = 0.12

    scene_center = np.array(transforms.get("scene_center", [0.0, 0.0, 0.0]))
    stored_radius = transforms.get("scene_radius", scene_radius)

    for frame_meta in transforms["frames"]:
        frame_path = frames_dir.parent / frame_meta["file_path"]
        if not frame_path.exists():
            continue

        c2w = np.array(frame_meta["transform_matrix"], dtype=np.float64)

        # Depth = distance from camera to scene centre.
        # For interior panoramic cameras (at the scene centre), this is near-zero,
        # so fall back to scene_radius (distance from centre to the walls/objects).
        cam_pos = c2w[:3, 3]
        cam_to_center = np.linalg.norm(cam_pos - scene_center)
        depth = cam_to_center if cam_to_center > stored_radius * 0.3 else stored_radius

        image = Image.open(frame_path).convert("RGB")
        inputs = processor(text=texts, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process into (boxes, scores, labels) — boxes in [x1,y1,x2,y2] pixel coords
        target_sizes = torch.tensor([[H, W]], device=device)
        results = processor.post_process_grounded_object_detection(
            outputs,
            threshold=score_threshold,
            target_sizes=target_sizes,
        )[0]

        boxes  = results["boxes"].cpu().numpy()   # (M, 4)
        scores = results["scores"].cpu().numpy()  # (M,)
        label_ids = results["labels"].cpu().numpy()  # (M,)

        for box, score, lid in zip(boxes, scores, label_ids):
            label = labels[lid].strip()
            world_pos, (w_px, h_px) = _unproject_box(box, depth, K_inv, c2w)
            world_w, world_h = _pixel_size_to_world(w_px, h_px, depth, fl_x, fl_y)
            # Use scene_radius as a rough depth proxy for the third dimension
            world_d = (world_w + world_h) / 2.0

            raw_detections.append({
                "label": label,
                "score": float(score),
                "position": world_pos,
                "scale": np.array([world_w, world_h, world_d]),
            })
            print(f"  [detect] {label} ({score:.2f}) @ {world_pos.round(3)} frame={frame_path.name}")

    return raw_detections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(ply_path: str, prompt: str, job_dir: str) -> list[dict]:
    job_dir = Path(job_dir)

    # ── Step 1: render views ────────────────────────────────────────────────
    print("[pipeline] Step 1: Rendering camera views …")
    transforms_path = render_cameras.render_views(ply_path, str(job_dir))

    with open(transforms_path) as f:
        transforms = json.load(f)

    frames_dir = job_dir / "frames"

    # Compute scene radius from the rendered camera distances
    cam_positions = np.array([
        frame["transform_matrix"]
        for frame in transforms["frames"]
    ])[:, :3, 3]
    scene_radius = float(np.linalg.norm(cam_positions, axis=1).mean())

    # ── Step 2: OWLv2 detection ─────────────────────────────────────────────
    labels = [l.strip() for l in prompt.split(",") if l.strip()]
    if not labels:
        raise ValueError("prompt must contain at least one label")

    print(f"[pipeline] Step 2: Detecting {labels} in {len(transforms['frames'])} frames …")
    raw_detections = _run_owlv2(frames_dir, labels, transforms, scene_radius)

    if not raw_detections:
        print("[pipeline] No detections above threshold.")
        interactions = []
    else:
        # ── Step 3: cluster and format ───────────────────────────────────────
        print(f"[pipeline] Step 3: Clustering {len(raw_detections)} raw detections …")
        clustered = _cluster_detections(
            raw_detections,
            eps_m=scene_radius * 0.5,
            max_per_label=2,
            min_votes=4,
            min_peak_score=0.40,
        )

        interactions = []
        for label, pos, scale in clustered:
            interactions.append({
                "label": label,
                "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "scale":    {"x": float(scale[0]), "y": float(scale[1]), "z": float(scale[2])},
            })

    output_path = job_dir / "interactions.json"
    with open(output_path, "w") as f:
        json.dump(interactions, f, indent=2)

    print(f"[pipeline] Done → {output_path} ({len(interactions)} objects)")
    return interactions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--job_dir", required=True)
    args = parser.parse_args()
    result = run_pipeline(args.ply, args.prompt, args.job_dir)
    print(json.dumps(result, indent=2))
