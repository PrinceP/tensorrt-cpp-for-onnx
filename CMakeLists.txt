cmake_minimum_required(VERSION 3.7)
project(trtcpp)

add_definitions(-std=c++17)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)

# Set CUDA C++ standard
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")

set(CMAKE_BUILD_TYPE Release)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

# include and link dirs of cuda
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})


include_directories(${PROJECT_SOURCE_DIR}/include/)
include_directories(${PROJECT_SOURCE_DIR}/kernels/)

file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp)

file(GLOB_RECURSE PROCESS_SRCS ${PROJECT_SOURCE_DIR}/kernels/*.cu)
cuda_add_library(processplugin SHARED ${PROCESS_SRCS})
target_link_libraries(processplugin nvinfer cuda cudart)

cuda_add_executable(yolo-world examples/yolo-world/main.cpp ${SRCS})
target_link_libraries(yolo-world nvinfer)
target_link_libraries(yolo-world nvonnxparser)
target_link_libraries(yolo-world cuda)
target_link_libraries(yolo-world cudart)
target_link_libraries(yolo-world ${OpenCV_LIBS})
target_link_libraries(yolo-world nvinfer_plugin)
target_link_libraries(yolo-world processplugin)

cuda_add_executable(rt-detr examples/rt-detr/main.cpp ${SRCS})
target_link_libraries(rt-detr nvinfer)
target_link_libraries(rt-detr nvonnxparser)
target_link_libraries(rt-detr cuda)
target_link_libraries(rt-detr cudart)
target_link_libraries(rt-detr ${OpenCV_LIBS})
target_link_libraries(rt-detr nvinfer_plugin)
target_link_libraries(rt-detr processplugin)

cuda_add_executable(yolov10 examples/yolov10/main.cpp ${SRCS})
target_link_libraries(yolov10 nvinfer)
target_link_libraries(yolov10 nvonnxparser)
target_link_libraries(yolov10 cuda)
target_link_libraries(yolov10 cudart)
target_link_libraries(yolov10 ${OpenCV_LIBS})
target_link_libraries(yolov10 nvinfer_plugin)
target_link_libraries(yolov10 processplugin)

cuda_add_executable(yolov9 examples/yolov9/main.cpp ${SRCS})
target_link_libraries(yolov9 nvinfer)
target_link_libraries(yolov9 nvonnxparser)
target_link_libraries(yolov9 cuda)
target_link_libraries(yolov9 cudart)
target_link_libraries(yolov9 ${OpenCV_LIBS})
target_link_libraries(yolov9 nvinfer_plugin)
target_link_libraries(yolov9 processplugin)

cuda_add_executable(yolov8-detect examples/yolov8/main_detect.cpp ${SRCS})
target_link_libraries(yolov8-detect nvinfer)
target_link_libraries(yolov8-detect nvonnxparser)
target_link_libraries(yolov8-detect cuda)
target_link_libraries(yolov8-detect cudart)
target_link_libraries(yolov8-detect ${OpenCV_LIBS})
target_link_libraries(yolov8-detect nvinfer_plugin)
target_link_libraries(yolov8-detect processplugin)

cuda_add_executable(yolov8-segment examples/yolov8/main_segment.cpp ${SRCS})
target_link_libraries(yolov8-segment nvinfer)
target_link_libraries(yolov8-segment nvonnxparser)
target_link_libraries(yolov8-segment cuda)
target_link_libraries(yolov8-segment cudart)
target_link_libraries(yolov8-segment ${OpenCV_LIBS})
target_link_libraries(yolov8-segment nvinfer_plugin)
target_link_libraries(yolov8-segment processplugin)

cuda_add_executable(yolov8-pose examples/yolov8/main_pose.cpp ${SRCS})
target_link_libraries(yolov8-pose nvinfer)
target_link_libraries(yolov8-pose nvonnxparser)
target_link_libraries(yolov8-pose cuda)
target_link_libraries(yolov8-pose cudart)
target_link_libraries(yolov8-pose ${OpenCV_LIBS})
target_link_libraries(yolov8-pose nvinfer_plugin)
target_link_libraries(yolov8-pose processplugin)

cuda_add_executable(yolov8-obb examples/yolov8/main_obb.cpp ${SRCS})
target_link_libraries(yolov8-obb nvinfer)
target_link_libraries(yolov8-obb nvonnxparser)
target_link_libraries(yolov8-obb cuda)
target_link_libraries(yolov8-obb cudart)
target_link_libraries(yolov8-obb ${OpenCV_LIBS})
target_link_libraries(yolov8-obb nvinfer_plugin)
target_link_libraries(yolov8-obb processplugin)


cuda_add_executable(yolov8-classify examples/yolov8/main_classify.cpp ${SRCS})
target_link_libraries(yolov8-classify nvinfer)
target_link_libraries(yolov8-classify nvonnxparser)
target_link_libraries(yolov8-classify cuda)
target_link_libraries(yolov8-classify cudart)
target_link_libraries(yolov8-classify ${OpenCV_LIBS})
target_link_libraries(yolov8-classify nvinfer_plugin)
target_link_libraries(yolov8-classify processplugin)

cuda_add_executable(yolov5-face examples/yolov5-face/main.cpp ${SRCS})
target_link_libraries(yolov5-face nvinfer)
target_link_libraries(yolov5-face nvonnxparser)
target_link_libraries(yolov5-face cuda)
target_link_libraries(yolov5-face cudart)
target_link_libraries(yolov5-face ${OpenCV_LIBS})
target_link_libraries(yolov5-face nvinfer_plugin)
target_link_libraries(yolov5-face processplugin)

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)
