#ifndef CANNY_CUH
#define CANNY_CUH

__global__ void convolution_kernel(const float* image, float* output, const float* mask, int imageRows, int imageCols,
	int outputRows, int outputCols, int maskDimension);

__global__ void magnitude_matrix_kernel(float* mag, const float* x, const float* y, const int height, const int width);

#endif