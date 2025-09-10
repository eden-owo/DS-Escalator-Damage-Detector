# Deepstream-Yolo-Fall-Detection (DeepStream 7.1)

NVIDIA DeepStream SDK application for YOLO-Pose models (Updated for DeepStream 7.1 and YOLO11-Pose)


> This is a continued implementation of the original project, updated to support DeepStream 7.1 and YOLO11-Pose.

> YOLO objetct detection models and other infos: https://github.com/marcoslucianops/DeepStream-Yolo

> Original YOLO-Pose models for DeepStream SDK 6.3 / 6.2 / 6.1.1 / 6.1 / 6.0.1 / 6.0: https://github.com/marcoslucianops/DeepStream-Yolo-Pose

---

## Verified models

* [YOLOv8-Pose](https://github.com/ultralytics/ultralytics)
* [YOLO11-Pose](https://github.com/ultralytics/ultralytics)

---

## Setup

### 1. Download the DeepStream-Yolo-Pose repo

Clone the repository and set environment.

```
git clone https://github.com/eden-owo/Deepstream-yolo-fall-detection.git
cd DeepStream-Yolo-Pose
```

### 2. Run in Docker
Prerequisites: NVIDIA driver, Docker, and nvidia-container-toolkit must be installed with GPU support.

(Optional) For X11 display, run: xhost +local:root (use xhost -local:root to revoke after testing).
```bash
xhost +local:root
```

Launch the Container
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

Run the setup script to install dependencies inside the container.

```bash
/apps/bootstrap.sh
```

### 3. Compile the libs

Export the CUDA_VER env according to your DeepStream version and platform:

* DeepStream 7.1 on x86_64 linux platform

  ```
  export CUDA_VER=12.6
  ```

* Compile the libs

  ```
  make -C nvdsinfer_custom_impl_Yolo_pose
  make
  ```

### 4. Python Bindings

  DeepStream 7.1 (x86_64): included in bootstrap.sh

  DeepStream ≤6.3: install pyds from  `NVIDIA-AI-IOT/deepstream_python_apps` 


### 5. Prepare and Export the Model to ONNX

Create a dedicated directory for the model, then follow [the documentation](./docs/) to set up the model and export it as an .onnx file.


### 6. Run

* C code

  ```
  ./deepstream -s file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 -c config_infer_primary_yolo11_pose.txt
  ```

* Python code

  ```
  python3 deepstream.py -s file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 -c config_infer_primary_yolo11_pose.txt
  ```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

Options

| Option | Example | Default | Description |
|--------|---------|---------|-------------|
| `-s`, `--source` | `rtsp://...` | – | Input source |
| `-c`, `--config-infer` | `config_infer.txt` | – | Inference config file |
| `-b`, `--streammux-batch-size` | `2` | `1` | Batch size |
| `-w`, `--streammux-width` | `1280` | `1920` | Frame width |
| `-e`, `--streammux-height` | `720` | `1080` | Frame height |
| `-g`, `--gpu-id` | `1` | `0` | GPU ID |
| `-f`, `--fps-interval` | `10` | `5` | FPS log interval |

##

### Config Notes

NMS
```
cluster-mode=4
IoU = 0.45
```

Threshold
```
[class-attrs-all]
pre-cluster-threshold=0.25
topk=300
```

## Reference: 
* https://github.com/marcoslucianops/DeepStream-Yolo-Pose
* https://github.com/NVIDIA-AI-IOT/deepstream_python_apps

## Project Lineage

This repository is a continuation of the unmaintained project  
[marcoslucianops/DeepStream-Yolo-Pose](https://github.com/marcoslucianops/DeepStream-Yolo-Pose).

- The **original upstream implementation** (for DeepStream 6.0–6.3) is preserved in branch [`legacy-upstream`](https://github.com/eden-owo/DeepStream-Yolo-Pose/tree/legacy-upstream)  
  and tagged as [`upstream-legacy-2023`](https://github.com/eden-owo/DeepStream-Yolo-Pose/releases/tag/upstream-legacy-2023).  
  This keeps the original history intact for reference and reproducibility.

- The **active development branch** is [`master`](https://github.com/eden-owo/DeepStream-Yolo-Pose/tree/master),  
  updated for **DeepStream 7.1** and supporting newer models such as **YOLO11-Pose**.

If you are looking for compatibility with DeepStream ≤ 6.3, please check the legacy branch/tag.  
For DeepStream 7.1 and newer development, use this repository’s master branch.
