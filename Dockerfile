# Base: PyTorch + CUDA 12.1 devel (devel variant needed to compile gsplat CUDA extensions)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# System libs required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CUDA architecture(s) to compile gsplat for.
# Override at build time: docker build --build-arg CUDA_ARCH_LIST="8.9"
# Common values: 8.0=A100, 8.6=RTX30xx, 8.9=L40S/RTX40xx, 9.0=H100
ARG CUDA_ARCH_LIST="8.0;8.6;8.9"
ENV TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}"

# Install Python deps (torch/torchvision come from the base image).
# This layer is cached by Docker as long as requirements.txt doesn't change —
# subsequent deploys that only touch app code skip the ~15-min gsplat compile.
COPY requirements.txt .
RUN grep -vE '^(torch|torchvision)([>=<!]|$)' requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

# Application code (rebuilt on every deploy)
COPY server.py pipeline.py render_cameras.py config.py run_local.py ./
COPY renderers/ ./renderers/
COPY webapp/ ./webapp/

# Persistent data:
#   /data           → keys.db + HuggingFace model cache (mount as named volume)
#   /tmp/workshop_jobs → job scratch space (mount as named volume)
ENV DB_PATH="/data/keys.db"
ENV HF_HOME="/data/hf_cache"

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
