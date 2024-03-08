#ifndef COMMON_HPP
#define COMMON_HPP

#include "opencv2/opencv.hpp"
#include <sys/stat.h>
#include <unistd.h>
#include "NvInfer.h"



inline int get_size_by_dims(const nvinfer1::Dims& dims)
{
	int size = 1;
	for (int i = 0; i < dims.nbDims; i++)
	{
		size *= dims.d[i];
	}
	return size;
}

inline int type_to_size(const nvinfer1::DataType& dataType)
{
	switch (dataType)
	{
	case nvinfer1::DataType::kFLOAT:
		return 4;
	case nvinfer1::DataType::kHALF:
		return 2;
	case nvinfer1::DataType::kINT32:
		return 4;
	case nvinfer1::DataType::kINT8:
		return 1;
	case nvinfer1::DataType::kBOOL:
		return 1;
	default:
		return 4;
	}
}

inline static float clamp(float val, float min, float max)
{
	return val > min ? (val < max ? val : max) : min;
}

struct Binding
{
	size_t size = 1;
	size_t dsize = 1;
	nvinfer1::Dims dims;
	std::string name;
};

#endif // COMMON_HPP