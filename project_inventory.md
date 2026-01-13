# 專案環境盤點報告

## 1. 專案概覽
- **專案名稱**: DS-Escalator-Damage-Detector
- **目標**: 用於 YOLO-Pose 模型的 NVIDIA DeepStream SDK 應用程式（已針對 YOLO11-Pose 和 DeepStream 7.1 進行優化）。
- **核心技術**: C, Python, DeepStream SDK, TensorRT, YOLO-Pose。

## 2. Docker 環境
本專案設計於 Docker 容器內執行，以確保 DeepStream 相依套件的一致性。

- **基礎映像檔 (Base Image)**: `nvcr.io/nvidia/deepstream:7.1-triton-multiarch`
- **容器啟動指令 (標準版)**:
  ```bash
  docker run -it --privileged --rm \
             --net=host --ipc=host --gpus all \
             -e DISPLAY=$DISPLAY \
             -e CUDA_CACHE_DISABLE=0 \
             --device /dev/snd \
             -v /tmp/.X11-unix/:/tmp/.X11-unix \
             -v "$(pwd)":/apps \
             -w /apps \
             nvcr.io/nvidia/deepstream:7.1-triton-multiarch \
             bash
  ```

## 3. 環境設定 (容器內)
使用引導腳本 (Bootstrap script) 處理額外的相依套件安裝。

- **主要腳本**: `bootstrap.sh`
- **關鍵操作**:
    - 安裝 `ultralytics` (複製到 `/opt/ultralytics`)。
    - 安裝 ONNX 相關工具: `onnx`, `onnxslim`, `onnxruntime`。
    - 安裝 NVIDIA Python 綁定庫: `pyds-1.2.0-cp310-cp310-linux_x86_64.whl`。
    - **關鍵相依性**: 強制安裝 `numpy==1.26.0` (解決 DeepStream 7.1 與 NumPy 2.x 不相容的問題)。

## 4. 建置與編譯
專案需要編譯用於 YOLO-Pose 推論的自定義 C++ 外掛程式。

- **CUDA 版本**: 12.6 (適用於 DeepStream 7.1)。
- **編譯步驟**:
  ```bash
  # 編譯自定義解析程式庫
  make -C nvdsinfer_custom_impl_Yolo_pose
  # 編譯主程式
  make
  ```

## 5. 模型轉換環境
將 `.pt` 模型轉換為 DeepStream 使用的 `.onnx` 格式需要特定的 Python 環境。

- **位置**: `utils/` 目錄包含轉換腳本（例如 `export_yolo11_pose.py`）。
- **工作流程**:
    1. 設定 `ultralytics` 儲存庫。
    2. 安裝 `requirements.txt`。
    3. 執行 `export_yolo11_pose.py` 並帶上 `--dynamic` 參數。

## 6. 關鍵設定檔
- `config_infer_primary_yolo11_pose.txt`: YOLO11-Pose 的主要推論設定。
- `labels.txt`: 物件類別標籤。
- `Makefile`: C 應用程式的編譯指令。
- `Makefile`: C 應用程式的編譯指令。
- `mosquitto.conf`: MQTT Broker 的極簡開發配置。
    - `listener 1883 0.0.0.0`: 在埠口 1883 監聽，允許來自任何網路介面（如 Docker 內部網路）的連接。
    - `allow_anonymous true`: 允許匿名連線，無需帳密即可發布或接收 AI 事件訊息。
    - `log_dest stdout` & `log_type all`: 將所有行為日誌輸出到控制台，方便透過 `docker logs` 監控。
    - `persistence false`: 不保留歷史訊息到硬碟，適合即時性的 AI 警報。

## 7. E2E (End-to-End) 流水線特性
本專案是一個典型的端到端 AI 視覺解決方案，其流程涵蓋：
1.  **資料獲取**：從 RTSP 或影像檔自動獲取原始串流。
2.  **AI 感知**：整合 YOLO-Pose 進行即時推論與人體追蹤。
3.  **邏輯判定**：在 Pipeline 中實作自定義算法（如跌倒角度判定）。
4.  **業務輸出**：將感知結果轉化為實體動作，包括 MQTT 異警訊息、影像截圖存證以及 OSD 即時顯示。

這種閉環設計使得系統能從「看見影像」直接轉化為「送出告警」，無需額外的中間件處理。
