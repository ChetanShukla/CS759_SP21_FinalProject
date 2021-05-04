#ifndef CANNY_CUH
#define CANNY_CUH

#include "device_launch_parameters.h"
#include <stdint.h>
__global__ void convolution_kernel(const uint8_t* image, float* output, const float* mask, int imageRows, int imageCols,
	int outputRows, int outputCols, int maskDimension);

__global__ void magnitude_matrix_kernel(float* mag, const float* x, const float* y, const int height, const int width);

#endif