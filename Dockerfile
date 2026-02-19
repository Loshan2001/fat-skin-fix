FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

ARG COMFYUI_VERSION=latest
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_PREFER_BINARY=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8

# ---------------------------------------------------------
# System & Python Setup
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common git git-lfs wget curl ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ca-certificates \
    && git lfs install \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv python3-pip python3-distutils build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

RUN python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------
# PyTorch (CUDA 12.8)
# ---------------------------------------------------------
RUN pip install torch==2.7.0 -f https://download.pytorch.org/whl/cu128/torch_stable.html

# ---------------------------------------------------------
# ComfyUI Setup
# ---------------------------------------------------------
WORKDIR /opt
RUN pip install comfy-cli
RUN yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia

WORKDIR /comfyui

# ---------------------------------------------------------
# Extra dependencies
# ---------------------------------------------------------
RUN pip install requests websocket-client sageattention \
    accelerate transformers insightface onnxruntime-gpu==1.18.0

# FIX: Add missing websocket packages for fal run
RUN pip install websocket-client websockets

# ---------------------------------------------------------
# Skin v03 / ComfyUI Custom Nodes
# ---------------------------------------------------------

# Vendored comfyui_face_parsing (kept in repo to avoid network install issues)
COPY custom_nodes/comfyui_face_parsing /comfyui/custom_nodes/comfyui_face_parsing
RUN pip install -r /comfyui/custom_nodes/comfyui_face_parsing/requirements.txt

# Install CNR (ComfyUI Registry) packages - matching working RunPod snapshot versions
RUN comfy --workspace /comfyui node install ComfyUI_LayerStyle_Advance@2.0.37 \
    && comfy --workspace /comfyui node install comfyui_essentials@1.1.0 \
    && comfy --workspace /comfyui node install seedvr2_videoupscaler@2.5.24 \
    && comfy --workspace /comfyui node install comfyui-custom-scripts@1.2.5

# Verify CNR packages installed
RUN echo "=== Listing installed custom nodes ===" && ls -la /comfyui/custom_nodes/ | head -20 \
    && echo "=== Checking for face_parsing ===" && ls -la /comfyui/custom_nodes/ | grep -i face || echo "face_parsing not found in custom_nodes" \
    && echo "=== Checking for face_parsing package ===" && find /comfyui -name "*face*parsing*" -type d 2>/dev/null || echo "No face_parsing directories found"

# Git-based custom nodes with pinned commit hashes from snapshot
# 1. ComfyRoll Custom Nodes
RUN git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git /comfyui/custom_nodes/ComfyUI_Comfyroll_CustomNodes \
    && cd /comfyui/custom_nodes/ComfyUI_Comfyroll_CustomNodes \
    && git checkout d78b780ae43fcf8c6b7c6505e6ffb4584281ceca \
    && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# 2. ComfyUI Florence2
RUN git clone https://github.com/kijai/ComfyUI-Florence2.git /comfyui/custom_nodes/ComfyUI-Florence2 \
    && cd /comfyui/custom_nodes/ComfyUI-Florence2 \
    && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# 3. ComfyUI KJNodes (pinned hash from snapshot)
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git /comfyui/custom_nodes/ComfyUI-KJNodes \
    && cd /comfyui/custom_nodes/ComfyUI-KJNodes \
    && git checkout 50a0837f9aea602b184bbf6dbabf66ed2c7a1d22 \
    && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# 4. ComfyUI Post-Processing Nodes
RUN git clone https://github.com/EllangoK/ComfyUI-post-processing-nodes.git /comfyui/custom_nodes/ComfyUI-post-processing-nodes \
    && cd /comfyui/custom_nodes/ComfyUI-post-processing-nodes \
    && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# 5. Masquerade Nodes (pinned hash from snapshot)
RUN git clone https://github.com/BadCafeCode/masquerade-nodes-comfyui.git /comfyui/custom_nodes/masquerade-nodes-comfyui \
    && cd /comfyui/custom_nodes/masquerade-nodes-comfyui \
    && git checkout 432cb4d146a391b387a0cd25ace824328b5b61cf \
    && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# 6. rgthree – Power Lora Loader (pinned hash from snapshot)
RUN git clone https://github.com/rgthree/rgthree-comfy.git /comfyui/custom_nodes/rgthree-comfy \
    && cd /comfyui/custom_nodes/rgthree-comfy \
    && git checkout 8ff50e4521881eca1fe26aec9615fc9362474931 \
    && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# ---------------------------------------------------------
# Pre-download face_parsing models (required by Ryuukeisyou comfyui_face_parsing)
# ---------------------------------------------------------
RUN mkdir -p /comfyui/models/face_parsing /comfyui/models/ultralytics/bbox \
    && wget -q -O /comfyui/models/face_parsing/model.safetensors "https://huggingface.co/jonathandinu/face-parsing/resolve/main/model.safetensors" \
    && wget -q -O /comfyui/models/face_parsing/config.json "https://huggingface.co/jonathandinu/face-parsing/resolve/main/config.json" \
    && wget -q -O /comfyui/models/face_parsing/preprocessor_config.json "https://huggingface.co/jonathandinu/face-parsing/resolve/main/preprocessor_config.json" \
    && wget -q -O /comfyui/models/ultralytics/bbox/face_yolov8m.pt "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt"

# ---------------------------------------------------------
# fal Runtime Requirements
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
    boto3==1.35.74 \
    protobuf==4.25.1 \
    pydantic==2.10.6

ENV HF_HOME=/fal-volume/models/huggingface

WORKDIR /comfyui
EXPOSE 8188
