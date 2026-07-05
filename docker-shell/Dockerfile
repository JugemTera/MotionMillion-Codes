# MotionMillion-Codes runtime image
# Base: CUDA 11.8 + cuDNN8 (matches torch==2.4.1+cu118 in requirements.txt).
# Ubuntu 20.04 ships Python 3.8, matching the "Python 3.8.11" the repo is tested on.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # sklearn==0.0 is a deprecated stub; allow it just in case, though we strip it below.
    SKLEARN_ALLOW_DEPRECATED_SKLEARN=True

# --- System dependencies ------------------------------------------------------
# git/git-lfs: required for the `git+https://...` requirements (CLIP, human_body_prior)
#              and for prepare/download_T5-XL.sh.
# ffmpeg/libgl1/libglib2.0: needed by opencv-python, open3d, moviepy, imageio.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 \
        python3.8-dev \
        python3.8-distutils \
        python3-pip \
        build-essential \
        git \
        git-lfs \
        wget \
        curl \
        unzip \
        ca-certificates \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Make python3.8 the default `python`/`python3`.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Modern pip; keep setuptools<70 so legacy sdists (chumpy, sklearn-era pkgs) still build.
RUN python -m pip install --upgrade "pip==24.0" "setuptools<70" wheel

WORKDIR /workspace

# --- Python dependencies ------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt

# Build a Docker-friendly requirements file:
#   * drop `sklearn==0.0`               -> deprecated stub, scikit-learn is already pinned
#   * drop `mkl-fft/mkl-random/mkl-service` -> Intel-MKL conda artifacts; on pip they pin
#                                          numpy>=1.24.3 (conflicts with numpy==1.22.4) and
#                                          pull a multi-GB mkl/dpcpp runtime. Not needed here.
#   * pin torchvision / torchaudio      -> versions compatible with torch 2.4.1+cu118
#   * defer `chumpy`                    -> needs numpy present at build time (installed later)
RUN sed -E \
        -e '/^sklearn==0\.0/d' \
        -e '/^mkl-/d' \
        -e '/^chumpy==/d' \
        -e 's/^torchvision==.*/torchvision==0.19.1+cu118/' \
        -e 's/^torchaudio==.*/torchaudio==2.4.1+cu118/' \
        /tmp/requirements.txt > /tmp/requirements.docker.txt

# Pre-install the build/runtime deps that legacy sdists compile against.
RUN python -m pip install "numpy==1.22.4" "cython<3"

# Install everything (torch +cu118 wheels come from the PyTorch index).
RUN python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        -r /tmp/requirements.docker.txt

# chumpy last: it imports numpy in setup.py, so disable build isolation.
RUN python -m pip install --no-build-isolation "chumpy==0.70"

# Quick sanity check that the core stack imported cleanly and CUDA build is present.
RUN python -c "import torch, torchvision, torchaudio; \
print('torch', torch.__version__, '| cuda build', torch.version.cuda)"

CMD ["/bin/bash"]
