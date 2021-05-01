#ifndef CANNY_CUH
#define CANNY_CUH

// #include "cuda_runtime.h"

__global__ void convolution_kernel(float* image, float* output, float* mask, int imageRows, int imageCols, int outputRows, int outputCols, int maskRows, int maskCols);

__host__ void convolve(float* image, float* output, int imageRows, int imageCols, int outputRows, int outputCols);

#endif