#!/usr/bin/env bash
set -euo pipefail

# 1. 確保啟動我們剛剛建立的 uv 虛擬環境
# 假設您是在 /apps 目錄下建立的 .venv
if [ -f "/apps/.venv/bin/activate" ]; then
    source /apps/.venv/bin/activate
else
    echo "警告: 找不到虛擬環境 /apps/.venv/bin/activate"
    exit 1
fi

# 環境變數
export DEBIAN_FRONTEND=noninteractive
export CUDA_VER=13.0

# 下載 repo
cd /opt
if [[ ! -d ultralytics ]]; then
  git clone --depth=1 https://github.com/ultralytics/ultralytics.git
fi

# 安裝 ultralytics 與 onnx 家族
cd /opt/ultralytics
uv pip install onnx onnxslim onnxruntime

echo "請注意：正在處理 DeepStream 8.0 的 pyds 安裝..."
# 【重要策略】：DeepStream 8.0 容器通常會把編譯好的 whl 檔直接放在系統內！
# 我們先尋找容器內建的 wheel 檔，而不是去 Github 下載舊版 7.1 的。
WHEL_PATH=$(find /opt/nvidia/deepstream/deepstream/ -name "pyds-*.whl" | head -n 1)

if [ -n "$WHEL_PATH" ]; then
    echo "找到內建的 pyds: $WHEL_PATH"
    uv pip install "$WHEL_PATH"
else
    echo "警告：在容器內找不到 DeepStream 8.0 專屬的 pyds wheel 檔。"
    echo "您可能需要從 deepstream_python_apps 的 master/DS8 分支手動編譯。"
    echo "開始從原始碼編譯 DeepStream 8.0 專屬的 pyds..."

    # 1. 安裝編譯 pyds 必須的系統依賴 (需要 root 權限)
    apt-get update && apt-get install -y --no-install-recommends \
        python3-dev python3-gi python3-gst-1.0 python-gi-dev \
        libgirepository1.0-dev libcairo2-dev cmake g++ build-essential \
        libglib2.0-dev libgstreamer1.0-dev

    # 2. 下載官方 Python Bindings 原始碼
    cd /opt
    if [[ ! -d deepstream_python_apps ]]; then
        git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
    fi

    # 3. 進入編譯目錄並初始化 C++ 綁定工具 (pybind11)
    cd /opt/deepstream_python_apps
    git submodule update --init
    cd bindings

    # 4. 自動抓取當前虛擬環境的 Python 次版本號 (例如 3.10 抓 10, 3.12 抓 12)
    PY_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

    # 5. 執行 CMake 與 Make 編譯
    mkdir -p build && cd build
    cmake .. -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=${PY_MINOR}
    make -j$(nproc) # 動用 GB10 全核心加速編譯！

    # 6. 直接透過 Python 的原始碼安裝方式 (setup.py) 裝進 uv 虛擬環境中
    echo "編譯完成！開始安裝 pyds 至虛擬環境..."
    cd /opt/deepstream_python_apps/bindings
    
    # 確保虛擬環境中有打包工具，然後將當前目錄的編譯成果安裝進去
    uv pip install wheel setuptools
    uv pip install .
fi

# DeepStream 7.1 & 8 is incompatible with NumPy 2.x, so force reinstall numpy==1.26.0
uv pip install --force-reinstall numpy==1.26.0
uv pip install "PyGObject==3.48.2"

echo "[bootstrap] Done, environment ready."
