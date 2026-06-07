#!/usr/bin/env bash
# deploy.sh -- upload and start the WorldModelData API server
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
SESSION="worldmodel"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env not found. Copy .env.example and fill in values."
  exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE"

: "${SERVER_IP:?SERVER_IP not set in .env}"
: "${SERVER_USER:?SERVER_USER not set in .env}"
: "${SSH_KEY:?SSH_KEY not set in .env}"

ADMIN_USER="${ADMIN_USER:-admin}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-changeme}"

APP_DIR="/home/${SERVER_USER}/WorldModelData"
VENV="/home/${SERVER_USER}/venv"
SSH_KEY="${SSH_KEY/#\~/$HOME}"
SSH_CMD="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no -o BatchMode=yes"
REMOTE="${SERVER_USER}@${SERVER_IP}"

echo "=================================================="
echo "  Deploying WorldModelData -> ${REMOTE}"
echo "=================================================="

# -- 1. Upload application files -----------------------------------------------
echo ""
echo ">> Uploading application files..."
${SSH_CMD} "${REMOTE}" "mkdir -p ${APP_DIR}/webapp /tmp/workshop_jobs"

if command -v rsync &>/dev/null; then
  rsync -az --progress \
    -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" \
    --exclude "*.ply" --exclude "*.npy" --exclude "__pycache__" \
    --exclude ".DS_Store" --exclude ".env" --exclude "*.egg-info" \
    "${SCRIPT_DIR}/pipeline.py" \
    "${SCRIPT_DIR}/render_cameras.py" \
    "${SCRIPT_DIR}/server.py" \
    "${SCRIPT_DIR}/requirements.txt" \
    "${REMOTE}:${APP_DIR}/"
  rsync -az --progress \
    -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" \
    --exclude ".DS_Store" \
    "${SCRIPT_DIR}/webapp/" \
    "${REMOTE}:${APP_DIR}/webapp/"
else
  for f in pipeline.py render_cameras.py server.py requirements.txt; do
    scp -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SCRIPT_DIR}/${f}" "${REMOTE}:${APP_DIR}/${f}"
  done
  for f in webapp/index.html webapp/debug.html webapp/admin.html; do
    scp -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SCRIPT_DIR}/${f}" "${REMOTE}:${APP_DIR}/${f}"
  done
fi
echo "   OK: Files uploaded"

# -- 2. Remote one-time setup (idempotent) -------------------------------------
echo ""
echo ">> Running remote setup (skips if already done)..."

${SSH_CMD} "${REMOTE}" bash -s << EOF
set -euo pipefail
RAPP="${APP_DIR}"
RVENV="${VENV}"

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq 2>/dev/null
for pkg in python3 python3-pip python3-venv tmux ninja-build; do
  dpkg -s "\$pkg" &>/dev/null || sudo apt-get install -y -q "\$pkg"
done

if [[ ! -d "\${RVENV}" ]]; then
  echo "  Creating Python venv at \${RVENV}..."
  python3 -m venv "\${RVENV}"
fi

"\${RVENV}/bin/pip" install --quiet --upgrade pip

CUDA_TAG="cpu"
if command -v nvidia-smi &>/dev/null; then
  CUDA_TAG="cu121"
  echo "  GPU detected, using torch+cu121"
else
  echo "  WARNING: No GPU found -- pipeline will be slow"
fi

if ! "\${RVENV}/bin/python" -c "import torch" 2>/dev/null; then
  echo "  Installing PyTorch (\${CUDA_TAG})..."
  if [[ "\${CUDA_TAG}" == "cpu" ]]; then
    "\${RVENV}/bin/pip" install --quiet torch torchvision
  else
    "\${RVENV}/bin/pip" install --quiet torch torchvision \
      --index-url "https://download.pytorch.org/whl/\${CUDA_TAG}"
  fi
fi

if ! "\${RVENV}/bin/python" -c "import gsplat" 2>/dev/null; then
  echo "  Installing gsplat (may compile CUDA code, ~5 min)..."
  "\${RVENV}/bin/pip" install --quiet gsplat
fi

"\${RVENV}/bin/pip" install --quiet \
  fastapi "uvicorn[standard]" python-multipart \
  "transformers>=4.37,<4.51" numpy imageio plyfile \
  opencv-python-headless Pillow scipy

echo "  OK: Dependencies ready"
EOF

# -- 3. (Re)start API server ---------------------------------------------------
echo ""
echo ">> (Re)starting API server..."

${SSH_CMD} "${REMOTE}" bash -s << EOF
set -euo pipefail
RVENV="${VENV}"
RAPP="${APP_DIR}"
RSESSION="${SESSION}"

tmux kill-session -t "\${RSESSION}" 2>/dev/null && echo "  Stopped previous session" || true

tmux new-session -d -s "\${RSESSION}" -x 220 -y 50
tmux send-keys -t "\${RSESSION}" \
  "cd \${RAPP} && export ADMIN_USER='${ADMIN_USER}' ADMIN_PASSWORD='${ADMIN_PASSWORD}' && \${RVENV}/bin/uvicorn server:app --host 0.0.0.0 --port 8000 2>&1 | tee /tmp/worldmodel.log" \
  Enter

sleep 3

if "\${RVENV}/bin/python" -c \
   "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" \
   2>/dev/null; then
  echo "  OK: Server is up and healthy"
else
  echo "  WARNING: Server may still be starting -- check logs:"
  tail -20 /tmp/worldmodel.log 2>/dev/null || true
fi
EOF

# -- Summary -------------------------------------------------------------------
echo ""
echo "=================================================="
echo "  Deployment complete!"
echo ""
echo "  Home page   ->  http://${SERVER_IP}:8000/"
echo "  Debug view  ->  http://${SERVER_IP}:8000/debug"
echo "  Admin panel ->  http://${SERVER_IP}:8000/admin"
echo "  API docs    ->  http://${SERVER_IP}:8000/docs"
echo "  Health      ->  http://${SERVER_IP}:8000/health"
echo ""
echo "  Admin credentials: ${ADMIN_USER} / ${ADMIN_PASSWORD}"
echo "  (Change ADMIN_USER and ADMIN_PASSWORD in .env)"
echo ""
echo "  View live logs:"
echo "    ssh -t -i ${SSH_KEY} ${REMOTE} 'tmux attach -t ${SESSION}'"
echo "  Detach: Ctrl-B then D"
echo "=================================================="
