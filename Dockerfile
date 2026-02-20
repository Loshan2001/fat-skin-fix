# =============================================================
# Stage 1: Builder
# Compiles packages, clones repos, downloads config files.
# Nothing from this stage bloats the final runtime image.
# =============================================================
FROM nvidia/cuda:12.8.0-base-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_PREFER_BINARY=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8

# ── System deps needed to BUILD packages ─────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git git-lfs wget curl ca-certificates \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv python3-pip python3-distutils \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --upgrade pip setuptools wheel

# ── PyTorch (CUDA 12.8) ──────────────────────────────────────
# Installed first so it's a cacheable layer — only re-runs if version changes
RUN pip install torch==2.7.0 -f https://download.pytorch.org/whl/cu128/torch_stable.html

# ── ComfyUI ──────────────────────────────────────────────────
ARG COMFYUI_VERSION=latest
RUN pip install comfy-cli && \
    yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia

# ── Python runtime packages ───────────────────────────────────
RUN pip install \
    requests \
    websocket-client \
    websockets \
    sageattention \
    accelerate \
    transformers \
    insightface \
    onnxruntime-gpu==1.18.0 \
    boto3==1.35.74 \
    protobuf==4.25.1 \
    pydantic==2.10.6

# ── Vendored face_parsing custom node ────────────────────────
COPY custom_nodes/comfyui_face_parsing /comfyui/custom_nodes/comfyui_face_parsing
RUN pip install -r /comfyui/custom_nodes/comfyui_face_parsing/requirements.txt

# ── CNR (ComfyUI Registry) custom nodes ──────────────────────
RUN comfy --workspace /comfyui node install ComfyUI_LayerStyle_Advance@2.0.37 && \
    comfy --workspace /comfyui node install comfyui_essentials@1.1.0 && \
    comfy --workspace /comfyui node install seedvr2_videoupscaler@2.5.24 && \
    comfy --workspace /comfyui node install comfyui-custom-scripts@1.2.5

# ── Git-based custom nodes (all in one layer) ─────────────────
RUN git clone --depth=1 https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git \
        /comfyui/custom_nodes/ComfyUI_Comfyroll_CustomNodes && \
    cd /comfyui/custom_nodes/ComfyUI_Comfyroll_CustomNodes && \
    git fetch --depth=1 origin d78b780ae43fcf8c6b7c6505e6ffb4584281ceca && \
    git checkout d78b780ae43fcf8c6b7c6505e6ffb4584281ceca && \
    \
    git clone --depth=1 https://github.com/kijai/ComfyUI-Florence2.git \
        /comfyui/custom_nodes/ComfyUI-Florence2 && \
    \
    git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes.git \
        /comfyui/custom_nodes/ComfyUI-KJNodes && \
    cd /comfyui/custom_nodes/ComfyUI-KJNodes && \
    git fetch --depth=1 origin 50a0837f9aea602b184bbf6dbabf66ed2c7a1d22 && \
    git checkout 50a0837f9aea602b184bbf6dbabf66ed2c7a1d22 && \
    \
    git clone --depth=1 https://github.com/EllangoK/ComfyUI-post-processing-nodes.git \
        /comfyui/custom_nodes/ComfyUI-post-processing-nodes && \
    \
    git clone --depth=1 https://github.com/BadCafeCode/masquerade-nodes-comfyui.git \
        /comfyui/custom_nodes/masquerade-nodes-comfyui && \
    cd /comfyui/custom_nodes/masquerade-nodes-comfyui && \
    git fetch --depth=1 origin 432cb4d146a391b387a0cd25ace824328b5b61cf && \
    git checkout 432cb4d146a391b387a0cd25ace824328b5b61cf && \
    \
    git clone --depth=1 https://github.com/rgthree/rgthree-comfy.git \
        /comfyui/custom_nodes/rgthree-comfy && \
    cd /comfyui/custom_nodes/rgthree-comfy && \
    git fetch --depth=1 origin 8ff50e4521881eca1fe26aec9615fc9362474931 && \
    git checkout 8ff50e4521881eca1fe26aec9615fc9362474931

# ── Install all custom node requirements in one pass ─────────
RUN find /comfyui/custom_nodes -name requirements.txt \
    -not -path "*/comfyui_face_parsing/*" \
    -exec pip install -r {} \;

# ── Download face_parsing config files only (NOT the large model weights) ──
# The large model.safetensors is handled by fal's download_model_weights()
# in setup() so it benefits from fal's cross-container cache.
RUN mkdir -p /comfyui/models/face_parsing /comfyui/models/ultralytics/bbox && \
    wget -q -O /comfyui/models/face_parsing/config.json \
        "https://huggingface.co/jonathandinu/face-parsing/resolve/main/config.json" && \
    wget -q -O /comfyui/models/face_parsing/preprocessor_config.json \
        "https://huggingface.co/jonathandinu/face-parsing/resolve/main/preprocessor_config.json"

# ── Verify custom nodes are present ──────────────────────────
RUN echo "=== Custom nodes ===" && ls /comfyui/custom_nodes/


# =============================================================
# Stage 2: Runtime
# Lean final image — no compilers, no build tools, no git history.
# =============================================================
FROM nvidia/cuda:12.8.0-base-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/fal-volume/models/huggingface

# ── Only runtime system deps ──────────────────────────────────
# No build-essential, no python3.12-dev, no git-lfs
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv \
    git \
    ffmpeg \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# ── Copy installed Python packages from builder ───────────────
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# ── Copy ComfyUI (with custom nodes) from builder ────────────
COPY --from=builder /comfyui /comfyui

WORKDIR /comfyui
EXPOSE 8188