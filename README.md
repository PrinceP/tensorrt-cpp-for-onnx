
# <div align="center">TENSORRT CPP FOR ONNX</d>
 

Tensorrt codebase in c++ to inference for all major neural arch using onnx and dynamic batching


## <div align="left">NVIDIA Driver</d>

```bash

wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

sudo sh cuda_12.4.0_550.54.14_linux.run

```

## <div align="left">Docker</d>

```bash

sudo docker build -t trt_24.02_opencv  .

sudo docker run --rm --network="host" -v $(pwd):/app -it --runtime nvidia trt_24.02_opencv bash
```

## <div align="center">Models</div>

### <div align="left">YOLOV9</div>

<details>
<summary>Model Conversion</summary>

url = https://github.com/WongKinYiu/yolov9.git

commit 380284cb66817e9ffa30a80cad4c1b110897b2fb

- Clone the yolov9
```bash

git clone https://github.com/WongKinYiu/yolov9

python3 export.py --weights <model_version>.pt --include onnx_end2end

git clone https://github.com/PrinceP/tensorrt-cpp-for-onnx

// Move <model_version>-end2end.onnx file to 'examples/yolov9'
cp <model_version>-end2end.onnx /app/examples/yolov9

mkdir build
cd build
cmake ..
make -j4

./yolov9 /app/examples/yolov9/<model_version>-end2end.onnx /app/data/

// Check the results folder
```

</details>

<details>
<summary>Results</summary>

**Results  [YOLOv9-C, Batchsize = 2, Model size = 640x640]**

<div style="display: flex; justify-content: center;
padding: 10px">
    <img src="./results/v9_bus.jpg" width="100%"/>
</div>
<div style="display: flex; justify-content: center;
padding: 10px">
    <img src="./results/v9_zidane.jpg" width="100%"/>
</div>
<div style="display: flex; justify-content: center; padding: 10px">
    <img src="./results/v9_test.jpeg" width="100%"/>
</div>
</details>

### <div align="left">YOLOV8-Detect</div>

<details>
<summary>Model Conversion</summary>

url = https://github.com/ultralytics/ultralytics

ultralytics==8.1.24

- Install ultralytics package in python
```python

from ultralytics import YOLO

model = YOLO('yolov8s.pt')
model.export(format='onnx', dynamic=True)
```
```bash
git clone https://github.com/PrinceP/tensorrt-cpp-for-onnx

// Move <model_version>.onnx file to 'examples/yolov8'
cp <model_version>.onnx /app/examples/yolov8

mkdir build
cd build
cmake ..
make -j4

./yolov8-detect /app/examples/yolov8/<model_version>.onnx /app/data/

// Check the results folder
```

</details>

<details>
<summary>Results</summary>

**Results  [YOLOv8s, Batchsize = 2, Model size = 640x640]**

<div style="display: flex; justify-content: center;
padding: 10px">
    <img src="./results/v8_bus.jpg" width="100%"/>
</div>
<div style="display: flex; justify-content: center;
padding: 10px">
    <img src="./results/v8_zidane.jpg" width="100%"/>
</div>
<div style="display: flex; justify-content: center; padding: 10px">
    <img src="./results/v8_test.jpeg" width="100%"/>
</div>
</details>

### <div align="left">YOLOV8-Segment</div>

<details>
<summary>Model Conversion</summary>

url = https://github.com/ultralytics/ultralytics

ultralytics==8.1.24

- Install ultralytics package in python
```python

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')

# Export the model
model.export(format='onnx', dynamic=True)
```
```bash
git clone https://github.com/PrinceP/tensorrt-cpp-for-onnx

// Move <model_version>.onnx file to 'examples/yolov8'
cp <model_version>.onnx /app/examples/yolov8

mkdir build
cd build
cmake ..
make -j4

./yolov8-segment /app/examples/yolov8/<model_version>.onnx /app/data/

// Check the results folder
```

</details>

<details>
<summary>Results</summary>

**Results  [YOLOv8n, Batchsize = 2, Model size = 640x640]**

<div style="display: flex; justify-content: center;
padding: 10px">
    <img src="./results/v8seg_bus.jpg" width="100%"/>
</div>
<div style="display: flex; justify-content: center;
padding: 10px">
    <img src="./results/v8seg_zidane.jpg" width="100%"/>
</div>
<div style="display: flex; justify-content: center; padding: 10px">
    <img src="./results/v8seg_test.jpeg" width="100%"/>
</div>
</details>

### <div align="left">YOLOV8-Pose</div>

<details>
<summary>Model Conversion</summary>

url = https://github.com/ultralytics/ultralytics

ultralytics==8.1.24

- Install ultralytics package in python
```python

from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-pose.pt')

# Export the model
model.export(format='onnx', dynamic=True)
```
```bash
git clone https://github.com/PrinceP/tensorrt-cpp-for-onnx

// Move <model_version>.onnx file to 'examples/yolov8'
cp <model_version>.onnx /app/examples/yolov8

mkdir build
cd build
cmake ..
make -j4

./yolov8-pose /app/examples/yolov8/<model_version>.onnx /app/data/

// Check the results folder
```

</details>

<details>
<summary>Results</summary>

**Results  [YOLOv8n, Batchsize = 2, Model size = 640x640]**

<div style="display: flex; justify-content: center;
padding: 10px">
    <img src="./results/v8pose_bus.jpg" width="100%"/>
</div>
<div style="display: flex; justify-content: center;
padding: 10px">
    <img src="./results/v8pose_zidane.jpg" width="100%"/>
</div>
<div style="display: flex; justify-content: center; padding: 10px">
    <img src="./results/v8pose_test.jpeg" width="100%"/>
</div>
</details>


### <div align="left">NOTES</div>
<details>
<summary>Issues</summary>

-  Dynamic batching is supported. The batchsize and image sizes can be updated in the codebase.

- If size issue happens while building. Increase the workspaceSize

```bash
    Internal error: plugin node /end2end/EfficientNMS_TRT requires XXX bytes of scratch space, but only XXX is available. Try increasing the workspace size with IBuilderConfig::setMemoryPoolLimit().
```
```cpp
    config->setMaxWorkspaceSize(1U << 26) 
    //The current memory is 2^26 bytes
```
</details>
