# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG TORCH_CUDA_ARCH_LIST=8.6

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PATH=/usr/local/cuda/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        gnupg \
        libbz2-dev \
        libffi-dev \
        libglib2.0-0 \
        liblzma-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsm6 \
        libsqlite3-dev \
        libssl-dev \
        libxml2-dev \
        libxext6 \
        libxrender1 \
        make \
        zlib1g-dev \
    && curl -fsSLo /tmp/Python-3.7.17.tgz https://www.python.org/ftp/python/3.7.17/Python-3.7.17.tgz \
    && tar -xzf /tmp/Python-3.7.17.tgz -C /tmp \
    && cd /tmp/Python-3.7.17 \
    && ./configure --enable-loadable-sqlite-extensions \
    && make -j"$(nproc)" \
    && make altinstall \
    && cd / \
    && rm -rf /tmp/Python-3.7.17 /tmp/Python-3.7.17.tgz \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLo /tmp/get-pip.py https://bootstrap.pypa.io/pip/3.7/get-pip.py \
    && python3.7 /tmp/get-pip.py \
        pip==23.3.2 \
        setuptools==67.8.0 \
        wheel==0.42.0 \
    && rm /tmp/get-pip.py

ENV PYTHON=python3.7 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN --mount=type=cache,target=/root/.cache/pip \
    python3.7 -m pip install numpy==1.19.5

RUN --mount=type=cache,target=/root/.cache/pip \
    python3.7 -m pip install \
        torch==1.10.1+cu111 \
        torchvision==0.11.2+cu111 \
        torchaudio==0.10.1+cu111 \
        --index-url https://download.pytorch.org/whl/cu111 \
        --extra-index-url https://pypi.org/simple

RUN --mount=type=cache,target=/root/.cache/pip \
    python3.7 -m pip install \
        mmcv-full==1.4.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html \
    && python3.7 -m pip install \
        mmdet==2.14.0 \
        mmsegmentation==0.14.1 \
    && python3.7 -m pip install \
        scikit-image==0.19.3 \
        grpcio==1.51.3 \
        tensorboard==2.11.2 \
        timm==0.4.12 \
        open3d-python==0.7.0.0

RUN git clone --branch v0.17.1 --depth 1 \
        https://github.com/open-mmlab/mmdetection3d.git /tmp/mmdetection3d \
    && cd /tmp/mmdetection3d \
    && FORCE_CUDA=1 python3.7 setup.py install \
        --single-version-externally-managed \
        --record=/tmp/mmdet3d-install-record.txt \
    && rm -rf /tmp/mmdetection3d

RUN --mount=type=cache,target=/root/.cache/pip \
    python3.7 -m pip install \
        lyft-dataset-sdk==0.0.8 \
        networkx==2.2 \
        numba==0.48.0 \
        nuscenes-devkit==1.1.9 \
        plyfile==0.9.0 \
        trimesh==2.35.39

WORKDIR /workspace
COPY SurroundOcc/extensions/chamfer_dist /tmp/SurroundOcc/extensions/chamfer_dist

RUN cd /tmp/SurroundOcc/extensions/chamfer_dist \
    && python3.7 setup.py install \
    && rm -rf /tmp/SurroundOcc

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 \
    && rm -rf /var/lib/apt/lists/*

CMD ["/bin/bash"]

# build the container:
# docker build -f Dockerfile -t surroundocc:cuda11.1 .

# start the container and enter a shell
# docker run --rm -it --gpus all -v "$PWD":/workspace -w /workspace surroundocc:cuda11.1 bash
