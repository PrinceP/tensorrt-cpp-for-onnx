#ifndef __CROPRESIZE_H
#define __CROPRESIZE_H

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>

// For print inside CUDA kernel
#include <bits/stdc++.h>
#include <cstdlib>

struct AffineMatrix{
    float value[6];

    float mean_values[3];
    float scale_values[3];
};


void crop_resize_kernel_img(
    uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    cv::Rect crop,
    float* Imean_values,
    float* Iscale_values,
    int letterbox,
    float scale,
    size_t size,
    bool is_norm,
    cudaStream_t stream);

#endif  // __CROPRESIZE_H