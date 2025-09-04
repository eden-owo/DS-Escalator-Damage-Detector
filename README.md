# DeepStream-Yolo-Pose (DeepStream 7.1)

NVIDIA DeepStream SDK application for YOLO-Pose models (Updated for DeepStream 7.1)


> - YOLO objetct detection models and other infos: https://github.com/marcoslucianops/DeepStream-Yolo

> - Original YOLO-Pose models for DeepStream SDK 6.3 / 6.2 / 6.1.1 / 6.1 / 6.0.1 / 6.0: https://github.com/marcoslucianops/DeepStream-Yolo-Pose

---

## Verified models

* [YOLOv8-Pose](https://github.com/ultralytics/ultralytics)

---

## Setup

### 1. Run in Docker
> Prerequisites: NVIDIA driver, Docker, and nvidia-container-toolkit must be installed with GPU support.
> For X11 display, run: xhost +local:root (use xhost -local:root to revoke after testing).


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

### 2. Download the DeepStream-Yolo-Pose repo

After entering container, clone the repository and set environment.

```
git clone https://github.com/eden-owo/DeepStream-Yolo-Pose.git
cd DeepStream-Yolo-Pose
/apps/bootstrap.sh
```

### 3. Compile the libs

Export the CUDA_VER env according to your DeepStream version and platform:

* DeepStream 7.1 on x86 platform

  ```
  export CUDA_VER=12.6
  ```

* Compile the libs

  ```
  make -C nvdsinfer_custom_impl_Yolo_pose
  make
  ```

### 4. Python Bindings

  DeepStream 7.1 (x86): included in bootstrap.sh

  DeepStream ≤6.3: install pyds from  `NVIDIA-AI-IOT/deepstream_python_apps` 



### 5. Run

* C code

  ```
  ./deepstream -s file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 -c config_infer_primary_yoloV8_pose.txt
  ```

* Python code

  ```
  python3 deepstream.py -s file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 -c config_infer_primary_yoloV8_pose.txt
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
