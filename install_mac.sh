#!/usr/bin/env bash
# install_mac.sh — set up WorldModelData to run locally on Apple Silicon (Metal/MPS).
#
# Rendering uses gsplat-mps (a Metal port of gsplat); OWLv2 detection runs on CPU.
# Requirements: macOS on Apple Silicon + Xcode command-line tools.
#
#   ./install_mac.sh
#   source .venv/bin/activate
#   python run_local.py --ply scene.ply --prompt "chair, table" --quality low
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ── 0. sanity checks ─────────────────────────────────────────────────────────
if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This installer is for Apple Silicon (macOS arm64)."
  echo "On an NVIDIA CUDA machine, use:  pip install -r requirements.txt"
  exit 1
fi
if ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode command-line tools are required (to compile the Metal shader)."
  echo "Install them with:  xcode-select --install"
  exit 1
fi

# ── 1. pick a Python >= 3.10 ─────────────────────────────────────────────────
PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    if "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  fi
done
if [[ -z "$PY" ]]; then
  echo "Need Python >= 3.10. Install one, e.g.:  brew install python@3.12"
  exit 1
fi
echo ">> Using $($PY --version) ($(command -v "$PY"))"

# ── 2. venv + Python deps ────────────────────────────────────────────────────
$PY -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
pip install --upgrade pip wheel setuptools -q
echo ">> Installing Python dependencies (torch, OWLv2, …)"
pip install -q -r requirements-mac.txt

# ── 3. build the Metal renderer (gsplat-mps) ─────────────────────────────────
GS="third_party/gsplat-mps"
if [[ ! -d "$GS/.git" ]]; then
  echo ">> Cloning gsplat-mps (Metal fork of gsplat)"
  rm -rf "$GS"
  git clone --recursive -q https://github.com/iffyloop/gsplat-mps.git "$GS"
fi
echo ">> Building gsplat-mps (compiles a Metal extension — the first build takes a few minutes)"
# --no-build-isolation: its setup.py imports torch at build time, so it must see the torch
# we just installed (a fresh isolated build env would not have it).
pip install -e "$GS" --no-build-isolation -q
# The setup.py build bakes a RELATIVE shader path into csrc.so, which segfaults when the
# pipeline runs from any other directory. Removing it forces a JIT rebuild with ABSOLUTE
# paths on first use (cached afterwards). This is the one non-obvious step.
rm -f "$GS/gsplat/csrc.so"

echo ""
echo "✅ Setup complete. Run a detection:"
echo ""
echo "    source .venv/bin/activate"
echo "    python run_local.py --ply your_scene.ply --prompt \"chair, table\" --quality low"
echo ""
echo "Notes:"
echo "  • Renderer is auto-selected (renderer=auto → gsplat-metal on Apple Silicon)."
echo "  • OWLv2 detection runs on CPU here, so it is slower than a CUDA box — start with --quality low."
