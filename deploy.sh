#!/usr/bin/env bash
# deploy.sh -- upload and start the WorldModelData API server
# Reads SERVER_IP, SERVER_USER, SSH_KEY from .env in the same directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
SESSION="worldmodel"

# -- Load .env -----------------------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env not found. Create it from .env.example:"
  echo "  cp .env.example .env && nano .env"
  exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE"

: "${SERVER_IP:?SERVER_IP not set in .env}"
: "${SERVER_USER:?SERVER_USER not set in .env}"
: "${SSH_KEY:?SSH_KEY not set in .env}"

# Paths derived from SERVER_USER (defined after sourcing .env)
APP_DIR="/home/${SERVER_USER}/WorldModelData"
VENV="/home/${SERVER_USER}/venv"

# Expand ~ in SSH_KEY path
SSH_KEY="${SSH_KEY/#\~/$HOME}"

SSH_CMD="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no -o BatchMode=yes"
REMOTE="${SERVER_USER}@${SERVER_IP}"

echo "=================================================="
echo "  Deploying WorldModelData -> ${REMOTE}"
echo "=================================================="

# -- 1. Upload application files -----------------------------------------------
echo ""
echo ">> Uploading application files..."

${SSH_CMD} "${REMOTE}" "mkdir -p ${APP_DIR} /tmp/workshop_jobs"

if command -v rsync &>/dev/null; then
  rsync -az --progress \
    -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" \
    --exclude "*.ply" \
    --exclude "*.png" \
    --exclude "__pycache__" \
    --exclude ".DS_Store" \
    --exclude ".env" \
    --exclude "*.egg-info" \
    "${SCRIPT_DIR}/pipeline.py" \
    "${SCRIPT_DIR}/render_cameras.py" \
    "${SCRIPT_DIR}/server.py" \
    "${SCRIPT_DIR}/requirements.txt" \
    "${REMOTE}:${APP_DIR}/"
else
  for f in pipeline.py render_cameras.py server.py requirements.txt; do
    scp -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SCRIPT_DIR}/${f}" "${REMOTE}:${APP_DIR}/${f}"
  done
fi
echo "   OK: Files uploaded"

# -- 2. Remote one-time setup (idempotent) -------------------------------------
echo ""
echo ">> Running remote setup (skips if already done)..."

# Pass variables as the first lines of the remote script to avoid
# heredoc variable-expansion pitfalls on older bash versions.
${SSH_CMD} "${REMOTE}" bash -s << EOF
set -euo pipefail
RAPP="${APP_DIR}"
RVENV="${VENV}"

# System packages
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq 2>/dev/null
for pkg in python3 python3-pip python3-venv tmux ninja-build; do
  dpkg -s "\$pkg" &>/dev/null || sudo apt-get install -y -q "\$pkg"
done

# Python venv
if [[ ! -d "\${RVENV}" ]]; then
  echo "  Creating Python venv at \${RVENV}..."
  python3 -m venv "\${RVENV}"
fi

"\${RVENV}/bin/pip" install --quiet --upgrade pip

# Detect CUDA for the right PyTorch wheel
CUDA_TAG="cpu"
if command -v nvidia-smi &>/dev/null; then
  CUDA_TAG="cu121"
  echo "  GPU detected, using torch+cu121"
else
  echo "  WARNING: No GPU found -- installing CPU torch (pipeline will be slow)"
fi

# PyTorch (skip if already installed)
if ! "\${RVENV}/bin/python" -c "import torch" 2>/dev/null; then
  echo "  Installing PyTorch (\${CUDA_TAG})..."
  if [[ "\${CUDA_TAG}" == "cpu" ]]; then
    "\${RVENV}/bin/pip" install --quiet torch torchvision
  else
    "\${RVENV}/bin/pip" install --quiet torch torchvision \
      --index-url "https://download.pytorch.org/whl/\${CUDA_TAG}"
  fi
fi

# gsplat (compiles CUDA extensions -- can take ~5 min first time)
if ! "\${RVENV}/bin/python" -c "import gsplat" 2>/dev/null; then
  echo "  Installing gsplat (may compile CUDA code, ~5 min)..."
  "\${RVENV}/bin/pip" install --quiet gsplat
fi

# Remaining requirements (headless OpenCV on a server)
"\${RVENV}/bin/pip" install --quiet \
  fastapi "uvicorn[standard]" python-multipart \
  transformers numpy imageio plyfile \
  opencv-python-headless Pillow scipy

echo "  OK: Dependencies ready"
EOF

# -- 3. (Re)start API server in a tmux session ---------------------------------
echo ""
echo ">> (Re)starting API server..."

${SSH_CMD} "${REMOTE}" bash -s << EOF
set -euo pipefail
RVENV="${VENV}"
RAPP="${APP_DIR}"
RSESSION="${SESSION}"

# Kill any previous session
tmux kill-session -t "\${RSESSION}" 2>/dev/null && echo "  Stopped previous session" || true

# Start fresh session running uvicorn
tmux new-session -d -s "\${RSESSION}" -x 220 -y 50
tmux send-keys -t "\${RSESSION}" \
  "cd \${RAPP} && \${RVENV}/bin/uvicorn server:app --host 0.0.0.0 --port 8000 2>&1 | tee /tmp/worldmodel.log" \
  Enter

# Give uvicorn a moment to bind
sleep 3

# Health check
if "\${RVENV}/bin/python" -c \
   "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" \
   2>/dev/null; then
  echo "  OK: Server is up and healthy"
else
  echo "  WARNING: Server may still be starting -- check logs:"
  tail -20 /tmp/worldmodel.log 2>/dev/null || true
fi
EOF

# -- 4. Patch webapp API URL ---------------------------------------------------
echo ""
echo ">> Updating webapp/index.html API URL -> http://${SERVER_IP}:8000..."
if [[ "$(uname)" == "Darwin" ]]; then
  sed -i '' "s|const API = \"http://[^\"]*\"|const API = \"http://${SERVER_IP}:8000\"|" \
    "${SCRIPT_DIR}/webapp/index.html"
else
  sed -i "s|const API = \"http://[^\"]*\"|const API = \"http://${SERVER_IP}:8000\"|" \
    "${SCRIPT_DIR}/webapp/index.html"
fi
echo "   OK: Done"

# -- Summary -------------------------------------------------------------------
echo ""
echo "=================================================="
echo "  Deployment complete!"
echo ""
echo "  API endpoint  ->  http://${SERVER_IP}:8000"
echo "  Health check  ->  http://${SERVER_IP}:8000/health"
echo ""
echo "  View live logs:"
echo "    ssh -t -i ${SSH_KEY} ${REMOTE} 'tmux attach -t ${SESSION}'"
echo "  Detach from tmux: Ctrl-B then D"
echo ""
echo "  Start local webapp:"
echo "    cd webapp && bash start.sh"
echo "=================================================="
echo ""
echo "  NOTE: Ensure port 8000 is open in the cloud firewall / security group."
