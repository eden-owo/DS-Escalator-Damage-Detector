# DS-Escalator-Damage-Detector (DeepStream-based Escalator Damage Detection System)

Real-time **fall** and **suitcase drop** detection system built with **NVIDIA DeepStream 7.1** and **YOLO-Pose / YOLO-Seg**.  
Supports multi-object tracking and event publishing via **MQTT** for edge-to-cloud integration.

### Features
- 🧍‍♂️ Fall detection using YOLO-Pose keypoint analysis  
- 🧳 Suitcase drop detection via object tracking and separation logic  
- ⚙️ Optimized for Jetson Orin with TensorRT acceleration  
- 🔔 MQTT message publishing for remote alerts

---

## Verified models

Detection
* [YOLOv8](https://github.com/ultralytics/ultralytics)
* [YOLO11](https://github.com/ultralytics/ultralytics)

Pose estimation
* [YOLOv8-Pose](https://github.com/ultralytics/ultralytics)
* [YOLO11-Pose](https://github.com/ultralytics/ultralytics)

---

## Setup

### 1. Clone the Repository

  Download the project and set up the working environment.

```
git clone https://github.com/eden-owo/DS-Escalator-Damage-Detector.git
cd DS-Escalator-Damage-Detector
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
           --ipc=host --gpus all \
           -e DISPLAY=$DISPLAY \
           -e CUDA_CACHE_DISABLE=0 \
           --device /dev/snd \
           -v /tmp/.X11-unix/:/tmp/.X11-unix \
           -v "$(pwd)":/apps \
           --network ds-net \
           -w /apps 
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
  make -C nvdsinfer_custom_impl_Yolo
  make -C nvdsinfer_custom_impl_Yolo_pose
  make
  ```

### 4. Python Bindings

  DeepStream 7.1 (x86_64): included in bootstrap.sh


### 5. Prepare and Export the Model to ONNX

  Create a dedicated directory for the model, then follow [the documentation](./docs/) to set up the model and export it as an .onnx file.


### 6. Run MQTT Broker and Subscriber in separate Docker containers

  Before running the DeepStream application, start the MQTT broker and a subscriber for message monitoring.

* MQTT Broker

  ```  
  docker run --name mosq \
             --network ds-net \
             -it --rm -p 1883:1883 \
             -v "$PWD/mosquitto.conf:/mosquitto/config/mosquitto.conf" \
             eclipse-mosquitto
  ```

* MQTT Subscriber

  ```
  docker run --rm -it \
             --network ds-net eclipse-mosquitto mosquitto_sub \
             -h mosq -p 1883 \
             -t ds/events -v
  ```

### 7. Run deepstream.py as MQTT publisher

  Run the main script (MQTT publisher) in the container from Step 2:

  ```
  python3 deepstream.py -s file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4 -c config_infer_primary_yolo11_pose.txt
  ```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

Options

| Option | Example | Default | Description |
|--------|---------|---------|-------------|
| `-s`, `--source` | `rtsp://...` | – | Input source |
| `-cip`, `--config-infer-pose` | `config_infer_pose.txt` | – | Inference config file of Pose Dsitmaion |
| `-cid`, `--config-infer-detect` | `config_infer_detection.txt` | – | Inference config file of objection detection |
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
