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


if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)
