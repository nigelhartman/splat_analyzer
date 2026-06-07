#!/usr/bin/env bash
# deploy.sh  ──  build & deploy WorldModelData as a Docker container
#
# On first run:
#   • Installs Docker if not present
#   • Installs NVIDIA Container Toolkit if a GPU is detected
#   • Builds the image (~15-20 min first time; subsequent builds use cache)
# On subsequent runs:
#   • Rebuilds only changed layers (usually < 1 min if only code changed)
#   • Replaces the running container in-place
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env not found."
  exit 1
fi
# shellcheck source=/dev/null
source "$ENV_FILE"

: "${SERVER_IP:?SERVER_IP not set in .env}"
: "${SERVER_USER:?SERVER_USER not set in .env}"
: "${SSH_KEY:?SSH_KEY not set in .env}"

ADMIN_USER="${ADMIN_USER:-admin}"
ADMIN_PASSWORD="${ADMIN_PASSWORD:-changeme}"
CUDA_ARCH_LIST="${CUDA_ARCH_LIST:-8.9}"

APP_DIR="/home/${SERVER_USER}/WorldModelData"
SSH_KEY="${SSH_KEY/#\~/$HOME}"
SSH_CMD="ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no -o BatchMode=yes"
REMOTE="${SERVER_USER}@${SERVER_IP}"

IMAGE="worldmodeldata:latest"
CONTAINER="worldmodeldata-app"

echo "=================================================="
echo "  Deploying WorldModelData (Docker)"
echo "  Target : ${REMOTE}"
echo "  Image  : ${IMAGE}"
echo "=================================================="

# ── 1. Upload source files ───────────────────────────────────────────────────
echo ""
echo ">> Uploading source files..."
${SSH_CMD} "${REMOTE}" "mkdir -p ${APP_DIR}/webapp"

if command -v rsync &>/dev/null; then
  rsync -az --progress \
    -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" \
    --exclude "*.ply" --exclude "*.npy" --exclude "__pycache__" \
    --exclude ".DS_Store" --exclude ".env" --exclude "*.egg-info" \
    --exclude ".git" \
    "${SCRIPT_DIR}/pipeline.py" \
    "${SCRIPT_DIR}/render_cameras.py" \
    "${SCRIPT_DIR}/server.py" \
    "${SCRIPT_DIR}/requirements.txt" \
    "${SCRIPT_DIR}/Dockerfile" \
    "${SCRIPT_DIR}/.dockerignore" \
    "${REMOTE}:${APP_DIR}/"
  rsync -az --progress \
    -e "ssh -i ${SSH_KEY} -o StrictHostKeyChecking=no" \
    --exclude ".DS_Store" \
    "${SCRIPT_DIR}/webapp/" \
    "${REMOTE}:${APP_DIR}/webapp/"
else
  for f in pipeline.py render_cameras.py server.py requirements.txt Dockerfile .dockerignore; do
    scp -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SCRIPT_DIR}/${f}" "${REMOTE}:${APP_DIR}/${f}"
  done
  for f in webapp/index.html webapp/debug.html webapp/admin.html; do
    scp -i "${SSH_KEY}" -o StrictHostKeyChecking=no "${SCRIPT_DIR}/${f}" "${REMOTE}:${APP_DIR}/${f}"
  done
fi
echo "   OK"

# ── 2. Install Docker (if missing) ───────────────────────────────────────────
echo ""
echo ">> Checking Docker installation..."
${SSH_CMD} "${REMOTE}" bash -s << 'REMOTE_EOF'
set -euo pipefail
if command -v docker &>/dev/null; then
  echo "  Docker $(docker --version | grep -oP '\d+\.\d+\.\d+' | head -1) already installed"
else
  echo "  Installing Docker via get.docker.com..."
  curl -fsSL https://get.docker.com | sudo sh
  sudo systemctl enable docker
  sudo systemctl start docker
  echo "  Docker installed: $(docker --version)"
fi
REMOTE_EOF

# ── 3. Install NVIDIA Container Toolkit (if GPU present and toolkit missing) ─
echo ""
echo ">> Checking NVIDIA Container Toolkit..."
${SSH_CMD} "${REMOTE}" bash -s << 'REMOTE_EOF'
set -euo pipefail
if ! command -v nvidia-smi &>/dev/null; then
  echo "  No GPU detected — skipping NVIDIA Container Toolkit"
  exit 0
fi
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if dpkg -s nvidia-container-toolkit &>/dev/null 2>&1; then
  echo "  NVIDIA Container Toolkit already installed"
  exit 0
fi
echo "  Installing NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
DIST=$(. /etc/os-release && echo "${ID}${VERSION_ID}")
curl -sL "https://nvidia.github.io/libnvidia-container/${DIST}/libnvidia-container.list" \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update -qq
sudo apt-get install -y -q nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
echo "  NVIDIA Container Toolkit installed"
REMOTE_EOF

# ── 4. Build Docker image ────────────────────────────────────────────────────
echo ""
echo ">> Building Docker image (first build ~15-20 min; subsequent builds use cache)..."
# Variables expanded locally before sending to remote
${SSH_CMD} "${REMOTE}" bash -s << EOF
set -euo pipefail
RAPP="${APP_DIR}"
ARCH="${CUDA_ARCH_LIST}"
echo "  Context : \${RAPP}"
echo "  CUDA arch: \${ARCH}"
sudo docker build \\
  --build-arg CUDA_ARCH_LIST="\${ARCH}" \\
  -t ${IMAGE} \\
  "\${RAPP}"
echo "  Image built: \$(sudo docker image inspect ${IMAGE} --format '{{.Size}}' | numfmt --to=iec)"
EOF

# ── 5. Replace running container ────────────────────────────────────────────
echo ""
echo ">> Replacing container..."
${SSH_CMD} "${REMOTE}" bash -s << EOF
set -euo pipefail
ADMIN_USER="${ADMIN_USER}"
ADMIN_PASSWORD="${ADMIN_PASSWORD}"

sudo docker stop ${CONTAINER} 2>/dev/null && echo "  Stopped old container" || true
sudo docker rm   ${CONTAINER} 2>/dev/null && echo "  Removed old container" || true

# Use --gpus all only when the GPU toolkit is available
GPU_FLAG=""
if command -v nvidia-smi &>/dev/null && sudo docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
  GPU_FLAG="--gpus all"
  echo "  GPU passthrough enabled"
fi

sudo docker run -d \\
  --name ${CONTAINER} \\
  --restart unless-stopped \\
  \${GPU_FLAG} \\
  -p 8000:8000 \\
  -v worldmodeldata-data:/data \\
  -v worldmodeldata-jobs:/tmp/workshop_jobs \\
  -e ADMIN_USER="\${ADMIN_USER}" \\
  -e ADMIN_PASSWORD="\${ADMIN_PASSWORD}" \\
  ${IMAGE}

echo "  Container started: \$(sudo docker inspect ${CONTAINER} --format '{{.Id}}' | head -c 12)"
EOF

# ── 6. Health check ──────────────────────────────────────────────────────────
echo ""
echo ">> Waiting for server to be ready..."
${SSH_CMD} "${REMOTE}" bash -s << 'REMOTE_EOF'
set -euo pipefail
for i in $(seq 1 15); do
  if python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" 2>/dev/null; then
    echo "  OK: Server is healthy (${i}s)"
    exit 0
  fi
  sleep 1
done
echo "  WARNING: Health check timed out. Container logs:"
sudo docker logs worldmodeldata-app --tail 30 2>&1 || true
REMOTE_EOF

# ── 7. Prune old images ──────────────────────────────────────────────────────
${SSH_CMD} "${REMOTE}" "sudo docker image prune -f --filter 'dangling=true' 2>/dev/null || true"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Deployment complete!"
echo ""
echo "  Home page   ->  http://${SERVER_IP}:8000/"
echo "  Debug view  ->  http://${SERVER_IP}:8000/debug"
echo "  Admin panel ->  http://${SERVER_IP}:8000/admin"
echo "  API docs    ->  http://${SERVER_IP}:8000/docs"
echo ""
echo "  Admin credentials: ${ADMIN_USER} / ${ADMIN_PASSWORD}"
echo "  (Change ADMIN_USER and ADMIN_PASSWORD in .env)"
echo ""
echo "  View logs:  ssh -i ${SSH_KEY} ${REMOTE} 'sudo docker logs -f ${CONTAINER}'"
echo "  Shell:      ssh -i ${SSH_KEY} ${REMOTE} 'sudo docker exec -it ${CONTAINER} bash'"
echo "=================================================="
