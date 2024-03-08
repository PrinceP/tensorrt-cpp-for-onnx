# TENSORRT CPP FOR ONNX 

## Nvidia Driver

```bash

wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

sudo sh cuda_12.4.0_550.54.14_linux.run

```

## Docker

```bash

sudo docker build -t trt_21.12_opencv .

sudo docker run --rm --network="host" -v $(pwd):/app -it --runtime nvidia trt_24.02_opencv bash
```

## MODELS 

### YOLOV9

url = https://github.com/WongKinYiu/yolov9.git

commit 380284cb66817e9ffa30a80cad4c1b110897b2fb

#### Model conversion

- Clone the yolov9
```bash

python3 export.py --weights <model_version>.pt --include onnx_end2end

// Move <model_version>-end2end.onnx file to 'examples/yolov9'
cp <model_version>-end2end.onnx /app/examples/yolov9

mkdir build
cd build
cmake ..
make -j4

./yolov9 /app/examples/yolov9/<model_version>-end2end.onnx /app/data/

// Check the results folder
```
-----------
<div style="display: flex; justify-content: space-between;
    margin-right: 10px;">
    <img src="./results/v9_bus.jpg" width="50%"/>
    <img src="./results/v9_zidane.jpg" width="50%"/>
</div>
<div style="display: flex; justify-content: space-between;
    margin-right: 10px;">
    <img src="./results/v9_test.jpeg" width="100%"/>
</div>

-----------