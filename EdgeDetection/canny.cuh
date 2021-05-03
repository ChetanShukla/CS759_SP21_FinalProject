#ifndef CANNY_CUH
#define CANNY_CUH

// #include "cuda_runtime.h"

__global__ void convolution_kernel(const float* image, float* output, const float* mask, int imageRows, int imageCols, 
                                    int outputRows, int outputCols, int maskDimension);

__global__ void magnitude_matrix_kernel(float* mag, const float* x, const float* y, const int height, const int width);

__host__ void convolve(const float* image, float* xOutput, float* yOutput, const float* maskx, const float* masky, 
                        int imageRows, int imageCols, int outputRows, int outputCols, int maskDimension);

#endif