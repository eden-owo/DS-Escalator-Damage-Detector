# YOLO11-Pose usage

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Edit the config_infer_primary_yolo11_pose file](#edit-the-config_infer_primary_yolo11_pose-file)

##

### Convert model

#### 1. Download the YOLO11 repo and install the requirements

```
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip3 install -r requirements.txt
python3 setup.py install
pip3 install onnx onnxsim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yolo11_pose.py` file from `DeepStream-Yolo-Pose/utils` directory to the `ultralytics` folder.

#### 3. Download the model

Download the `pt` file from [YOLO11](https://github.com/ultralytics/assets/releases/) releases (example for YOLO11s-Pose)

```
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-pose.pt
```

**NOTE**: You can use your custom model.

#### 4. Convert model

Generate the ONNX model file (example for YOLO11s-Pose)

```
python3 export_yolo11_pose.py -w yolo11s-pose.pt --dynamic
```

**NOTE**: To change the inference size (defaut: 640)

```
-s SIZE
--size SIZE
-s HEIGHT WIDTH
--size HEIGHT WIDTH
```

Example for 1280

```
-s 1280
```

or

```
-s 1280 1280
```

**NOTE**: To simplify the ONNX model (DeepStream >= 6.0)

```
--simplify
```

**NOTE**: To use dynamic batch-size (DeepStream >= 6.1)

```
--dynamic
```

**NOTE**: To use static batch-size (example for batch-size = 4)

```
--batch 4
```

#### 5. Copy generated files

Copy the generated ONNX model file to the `DeepStream-Yolo-Pose` folder.

##

### Edit the config_infer_primary_yolo11_pose file

Edit the `config_infer_primary_yolo11_pose.txt` file according to your model (example for YOLO11s-Pose)

```
[property]
...
onnx-file=yolo11s-pose.onnx
...
parse-bbox-func-name=NvDsInferParseYoloPose
...
```
