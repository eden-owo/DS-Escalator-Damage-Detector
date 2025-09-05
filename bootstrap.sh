#!/usr/bin/env bash
set -euo pipefail

# 環境變數
export DEBIAN_FRONTEND=noninteractive
export CUDA_VER=12.6

# 升級 pip
python3 -m pip install --upgrade pip

# 安裝必要套件
# apt-get update && apt-get install -y --no-install-recommends \
#     python3-pip python3-venv python3-dev build-essential git \
#     ffmpeg libgl1 libglib2.0-0 \
#  && rm -rf /var/lib/apt/lists/*

# 下載 repo
cd /opt
if [[ ! -d ultralytics ]]; then
  git clone --depth=1 https://github.com/ultralytics/ultralytics.git
fi

# 安裝 ultralytics 與 onnx 家族
cd /opt/ultralytics
# pip install .
pip install ultralytics==8.3.193
pip install onnx onnxslim onnxruntime

# 安裝 pyds 與 numpy
cd /opt
if [[ ! -f pyds-1.2.0-cp310-cp310-linux_x86_64.whl ]]; then
  wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.2.0/pyds-1.2.0-cp310-cp310-linux_x86_64.whl
fi
pip install ./pyds-1.2.0-cp310-cp310-linux_x86_64.whl

# DeepStream 7.1 is incompatible with NumPy 2.x, so force reinstall numpy==1.26.0
pip install --force-reinstall numpy==1.26.0

echo "[bootstrap] Done, environment ready."
