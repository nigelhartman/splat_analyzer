# Spike: Brush as a cross-platform renderer

**Branch:** `spike/brush-renderer`
**Goal:** Replace the CUDA-only `gsplat` rasterizer with [Brush](https://github.com/ArthurBrussee/brush)
(Rust + WGPU/Burn) so the rendering stage runs unchanged on:

- macOS / Apple Silicon (Metal, **no CUDA**) — local dev on Nigel's Mac
- Linux + NVIDIA L40S (the deployed Nebius server)
- other CUDA-enabled desktops

WGPU targets Metal/Vulkan/DX12, so one renderer *should* cover all three without
maintaining a CUDA-only path.

## Why this is non-trivial: the gsplat coupling

The pipeline depends on gsplat in exactly two places, both in `render_cameras.py`:

1. **RGB rasterization** — `render_cameras.py:553`
   `rasterization(means, quats, scales, opacities, colors=sh, viewmats, Ks, width, height, sh_degree, near, far)`

2. **Depth** — `render_cameras.py:577`. There is **no native depth output**. Depth is
   faked by passing camera-space Z as the "color" channel with `sh_degree=0`, rasterizing,
   then decoding: `depth = (rendered - 0.5) / SH_C0`. This trick relies on gsplat's exact
   alpha-compositing behavior.

Depth is **mandatory** — `pipeline.py` back-projects 2D OWLv2 boxes to 3D using
`frames/depth_XXXX.npy` (raw float, world units). A renderer that only emits RGB does not
replace gsplat for this pipeline.

Camera convention (must match for any backend): OpenCV (X right, Y down, Z forward),
poses are `c2w`, inverted to `w2c` for the rasterizer (`render_cameras.py:242,521`).
Intrinsics: 130° horizontal FoV, `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`.

Renderer-independent (stays as-is regardless of backend): SPZ→PLY conversion, PLY loading
for camera placement, the density-aware camera sampler, pose generation, transforms.json
authoring, and threaded PNG/NPY writing.

## Open feasibility questions (under investigation)

1. Does Brush's renderer expose a **per-pixel depth** map, or only RGB?
2. Can it render a **user-supplied set of camera poses** headlessly (no training)?
3. Does it ingest a Nerfstudio `transforms.json` and render exactly those frames? Convention?
4. Does it load a standard 3DGS `.ply` for rendering?
5. Integration path from Python: subprocess to `brush-cli`, C FFI, or a Rust port?
6. Confirmed running on Metal (Mac) **and** NVIDIA (Linux)?

→ Results recorded under "Findings" below before any code is written against Brush.

## Candidate integration architecture (pending findings)

Introduce a renderer seam so the CUDA dependency is isolated and a Brush backend can slot in:

- `config.PipelineConfig.renderer: "auto" | "gsplat" | "brush"` (`auto` = gsplat if CUDA else brush)
- A small backend interface that, given the loaded Gaussians (or `.ply` path), a list of
  `c2w` poses, intrinsics, and resolution, yields `(rgb_uint8, depth_float)` per view.
- `gsplat_backend`: the current batched `rasterization` loop + depth trick, moved verbatim
  (preserve the GPU/IO overlap via `RENDER_BATCH` batching).
- `brush_backend`: shape TBD by findings — most likely write `transforms.json`, invoke
  `brush-cli` once to render all poses to disk, read back RGB + depth.

The seam preserves `render_views`' camera placement and I/O; only the rasterization step swaps.

## Findings

Inspected Brush `main` (cloned to `/tmp/brush-src`). Evidence is file:line into that repo.

1. **Per-pixel depth — NOT produced anywhere.** The rasterize kernel accumulates RGB + alpha
   only (`crates/brush-render/src/kernels/rasterize.rs:74-77`); output is RGBA
   (`render.rs:247-252`), and `RenderAux` (`render_aux.rs:71-96`) has no depth field.
   `calc_tile_depth()` is a coarse per-*tile* splat-count heatmap, not per-pixel Z. Per-splat
   camera-space Z exists for sorting (`project_forward.rs:124`) but is never rasterized into a
   pixel buffer. → A depth buffer must be **added to the WGSL/CubeCL kernel** from scratch.
2. **Headless pose-driven render — does NOT exist.** `brush-cli` has no subcommands; it only
   trains (`apps/brush-cli/src/lib.rs:206-208`: "Only training is supported in the CLI"). The
   `eval_save_to_disk` path runs only inside the training loop, only over the dataset's eval
   split, **requires a ground-truth image per view** (for PSNR/SSIM), and writes RGB only.
3. **Nerfstudio `transforms.json` — parsed, but unusable for pose-only render.** The loader
   *skips any frame whose image file is missing* (`nerfstudio.rs:173-180`), so you can't feed it
   bare poses. Convention: OpenGL/NeRF **c2w** (`formats/mod.rs:126-131`) — note our pipeline is
   OpenCV, so a conversion would be needed.
4. **PLY input — works.** Standard 3DGS fields load via `load_splat_from_ply`, and the public
   `render_splats(splats, camera, img_size, ...) -> (Tensor, RenderAux)` is a clean entry point.
5. **Python access — none.** No PyO3/maturin. Only a C FFI (`apps/brush-c`) that is *train-only*
   (`train_and_save`). Integration = write a new Rust binary/shim around `render_splats`.
6. **Cross-platform — confirmed.** wgpu v29 via Burn → Metal on Mac, Vulkan/DX12 on NVIDIA, no
   CUDA. Needs Rust 1.88+. ⚠️ deps pin `burn` to git `main` and use author forks of wgpu/egui
   (unpinned branches) — a reproducible-build risk to vendor/pin.

## Verdict: **FEASIBLE WITH NON-TRIVIAL WORK** (a Rust project, not integration glue)

**As-is: not feasible** — two hard blockers: no per-pixel depth (and depth is mandatory for our
back-projection), and no headless pose-driven render path. Plus no Python binding.

To make it work requires, in order:
1. **Add a per-pixel depth accumulator to the rasterize kernel** (WGSL/CubeCL) — alpha-weighted
   expected depth or first-hit Z — and surface it through `RenderOutput`/`RenderAux`/`render_splats`.
   This is the largest item and needs GPU-kernel familiarity.
2. **Write a headless render binary** that builds `Camera` objects directly from our pose+intrinsics
   list (bypassing the image-required nerfstudio loader), loads the `.ply`, loops `render_splats`,
   and writes RGB + raw depth to disk.
3. **Expose to Python** (subprocess to that binary, or a PyO3/C shim).
4. **Pin/vendor** the forked wgpu/egui/burn-main deps for reproducible builds.

What's already in our favor: PLY loading, the genuinely cross-platform wgpu backend, and a clean
`render_splats` function. What must be built: depth + the headless render path.

### Lower-effort alternative if "one renderer everywhere" isn't worth the Rust project
Keep `gsplat` on CUDA (server + CUDA desktops) and add a **gsplat-API-compatible Metal fork**
(`gsplat-mps` / `splat-apple`) for the Mac only. Both keep the existing `rasterization()` call shape
**and the depth trick** (`render_cameras.py:577`) works unchanged. Two backends, but both share our
code path and depth Just Works — days, not a kernel project. The renderer seam in this branch
supports this too (`renderer: "auto" | "gsplat" | "gsplat-metal"`).

## DECISION (chosen): Metal gsplat fork for Mac, gsplat on CUDA elsewhere

Brush's Rust kernel project was rejected as too heavy for the payoff. We keep gsplat on CUDA and
add a gsplat-API Metal/MPS backend for the Mac, behind a renderer seam.

### Built on this branch so far
- `config.PipelineConfig.renderer: "auto" | "gsplat" | "gsplat-metal"` (auto = gsplat on CUDA,
  gsplat-metal on Apple MPS).
- `renderers/` package with a backend seam:
  - `base.Renderer` — shared `prepare()` (activations + device move) and the
    `render_rgb` / `render_depth` contract.
  - `gsplat_backend.GsplatRenderer` — the original CUDA path, extracted **verbatim** (batched RGB
    `rasterization` + the per-view depth-as-color trick). Behavior-preserving.
  - `gsplat_metal_backend.GsplatMetalRenderer` — **implemented** against gsplat-mps (see below).
- `render_cameras.py` refactored: `_load_ply_gaussians` → device-agnostic `_load_ply_arrays`
  (numpy); the render loop now calls `renderer.render_rgb/_depth`, preserving the GPU/IO overlap.
  No more top-level `from gsplat import` — so the module imports on a Mac without gsplat installed.

### Confirmed Metal fork: `gsplat-mps` (`opensplat-mps` branch)

Source investigation picked **gsplat-mps** over mlx-splat/msplat/splat-apple — it's the only
PyTorch/MPS option that takes raw per-Gaussian tensors, returns alpha, and accepts arbitrary
color-channel counts (so depth works). It installs under the package name **`gsplat`** (v0.1.3),
so it and the CUDA gsplat are mutually exclusive per machine — fine, since they live on different
boxes, and `get_renderer` imports each backend lazily.

`GsplatMetalRenderer` is implemented. API/convention deltas vs. the CUDA 1.5.3 path, all handled
inside the backend:
- **Split 0.1.x API, one view per call**: `project_gaussians → spherical_harmonics →
  rasterize_gaussians` (we loop the batch; no batched `rasterization()`).
- **Scales raw**: pass log-scales + `glob_scale=1.0`; the kernel exponentiates. (CUDA path exp's
  first.) → activation split now lives in each backend's `prepare()`.
- **SH→RGB done by us**: `spherical_harmonics(degree, viewdirs, sh) + 0.5`, then clamp; `viewdirs`
  from the camera center (`c2w[:3,3]`).
- **Depth is cleaner**: composite per-Gaussian camera-space Z directly as a 1-channel color — no
  `SH_C0` decode. Semantically the same alpha-weighted expected depth as the CUDA trick.
- ⚠️ **AGPLv3** (OpenSplat-derived Metal code) — confirm acceptable for distribution.
- ⚠️ Unmaintained (last real commit 2024-07); risk is bit-rot vs. current torch, not API churn.

### Mac setup (verified on M4 Pro, macOS, Python 3.12, torch 2.12.1, MPS)
```sh
git clone --recursive https://github.com/iffyloop/gsplat-mps.git   # --recursive: glm submodule
cd gsplat-mps                                                       # already on opensplat-mps
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch torchvision numpy plyfile imageio scipy
pip install -e . --no-build-isolation     # MUST use --no-build-isolation (setup.py imports torch)
# CRITICAL: remove the setup.py-built extension so it rebuilds via JIT with ABSOLUTE source paths
rm gsplat/csrc.so                          # else __FILE__ is relative → metal shader not found → SEGFAULT
# then, from the WorldModelData repo:
python render_cameras.py --ply scene.ply --job_dir /tmp/out --quality low   # auto → gsplat-metal
```

### LOCAL TEST RESULTS — ✅ WORKS (with fixes), quality caveat

Ran the full `render_cameras.py` on the real `scene.ply` (320,760 Gaussians, sh_degree 3) on the
M4 Pro via MPS. After three fixes (below) it produces **correct perspective RGB + valid varying
depth** (depth max ~0.5–1.5, median ~0.15–0.33; recognizable floor/edges/objects). Three bugs found
and fixed during testing — note all three contradicted the source-investigation's assumptions:

1. **SEGFAULT on first kernel call** (exit 139). gsplat-mps loads its `.metal` shader from a path
   derived from `__FILE__`, which the setup.py build baked in as **relative** → not found when CWD
   ≠ repo root → null library deref → segfault. **Fix:** delete the prebuilt `gsplat/csrc.so` so the
   extension rebuilds via `cpp_extension.load` with **absolute** source paths (now a documented
   install step above). One-time; the JIT build is cached.
2. **Scales convention** — kernel `scale_to_mat` uses **linear** scales (no `exp`). Investigation
   said "pass raw log-scales"; that produced giant garbage splats. **Fix:** `prepare()` exps the
   scales (same as CUDA path). *(applied in `gsplat_metal_backend.py`)*
3. **Projection matrix** — `project_pix` perspective-divides by `p_hom.w` and `ndc2pix` maps NDC→px,
   so `projmat` MUST be `P @ w2c` (P = OpenGL perspective from intrinsics). Investigation/example
   said "pass w2c for projmat too" → `w=1` → no divide/no focal → degenerate ortho smear. **Fix:**
   `_project()` builds `P @ w2c`. *(applied)* — this was the key correctness fix.

**Fix #1 (depth) + #2 (streaks) — IMPLEMENTED and validated against the server:**
- #1 `render_depth` now divides composited Z by accumulated alpha (`return_alpha=True`) → true
  expected depth, removing the shallow bias that pulled objects toward the camera.
- #2 `_clamp_opacity` zeros opacity for splats whose 2D radius exceeds `MAX_RADIUS_FRAC × max(W,H)`
  (default 1.0), emulating gsplat 1.5.3's large-radius handling → kills most streaks.

**Same-scene cross-check (scene.ply, prompt = the 9 workshop classes, quality=low):**
| object | Server (CUDA L40S) | Local (Mac gsplat-metal) | Δpos |
|---|---|---|---|
| chair | (-4.01, 1.31, -2.77) | (-4.11, 1.31, -2.74) | **~0.1 units** |

The shared detection's 3D position now matches within ~0.1 world units (was systematically shifted
before the depth fix). Local additionally returned a 2nd chair (-3.04,1.18,-2.96) and a couch
(-0.55,0.92,-2.29) the server missed at low quality, and local boxes run a bit smaller
(~1.5–2.0 vs ~2.1). These are render-fidelity / near-threshold differences, within the
"small differences OK" tolerance. Residual streaking remains on a few bright splats — lower
`MAX_RADIUS_FRAC` to tighten.

### Open (next)
- **Quality**: tackle the streaking (2D-radius clamp / large-splat handling) and compare a frame
  side-by-side against the CUDA server render of the same `scene.ply` + pose.
- **End-to-end**: run the full `run_local.py` (adds OWLv2) on Mac and check detections land.
- **Perf**: the Metal path projects twice per view (`render_rgb` + `render_depth`); cache the
  projection per batch if slow. Also no batched call in 0.1.x — it's a per-view loop.
- **Regression-check the CUDA path** on the server: the refactor is meant to be behavior-preserving
  — render a known scene pre/post and diff.
- **Decisions to confirm**: AGPLv3 acceptability; reliance on an unmaintained 2024 fork.
