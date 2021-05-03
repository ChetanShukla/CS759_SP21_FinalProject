#include "canny.cuh"
// #include "global.hpp"
#include "pixel.cuh"

using namespace std;

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <time.h>

#define BLOCK_SIZE 32
#define WA 256   
#define HA 256     
#define HC 7     
#define WC 7
#define WB (WA - WC + 1)
#define HB (HA - HC + 1)

#define sig 1

/**
=========================================== Kernel Convolution =========================================================

This function performs the convolution step on the given image array using the mask array that has been passed.
The output of this step is stored in the output array.

========================================================================================================================
**/
__global__ void convolution_kernel(const float* image, float* output, const float* mask, 
                                    int imageRows, int imageCols, int outputRows, int outputCols, 
                                    int maskDimension) {
    
    int col = blockIdx.x * (BLOCK_SIZE - WC + 1) + threadIdx.x;
	int row = blockIdx.y * (BLOCK_SIZE - WC + 1) + threadIdx.y;
	int row_i = row - WC + 1;
	int col_i = col - WC + 1;

	float tmp = 0;

	__shared__ float sharedMem[BLOCK_SIZE][BLOCK_SIZE];

	if (row_i < WA && row_i >= 0 && col_i < WA && col_i >= 0) {
		sharedMem[threadIdx.y][threadIdx.x] = image[col_i * WA + row_i];
	}
	else {
		sharedMem[threadIdx.y][threadIdx.x] = 0;
	}

	__syncthreads();

    if (threadIdx.y < (BLOCK_SIZE - WC + 1) && threadIdx.x < (BLOCK_SIZE - WC + 1) 
            && row < (WB - WC + 1) && col < (WB - WC + 1)) {
		for (int i = 0; i< WC;i++)
			for (int j = 0;j<WC;j++)
				tmp += sharedMem[threadIdx.y + i][threadIdx.x + j] * mask[j*WC + i];
        // TODO Check if this indexing is correct
        output[col*WB + row] = tmp;
        // Or should it be output[row*WB + col] = tmp; 
	}
}

__host__ void convolve(const float* image, float* xOutput, float* yOutput, const float* maskx, const float* masky, 
                        int imageRows, int imageCols, int outputRows, int outputCols, int maskDimension) {

    
    /*
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((WB - 1) / (BLOCK_SIZE - WC + 1), (WB - 1) / (BLOCK_SIZE - WC + 1));

	convolution_kernel <<< grid, threads >>> (image, xOutput, maskx, imageRows, imageCols, outputRows, outputCols, maskDimension, maskDimension);

	cudaEventRecord(start_G);

	convolution_kernel <<< grid, threads >>> (image, yOutput, masky, imageRows, imageCols, outputRows, outputCols, maskDimension, maskDimension);
	error = cudaGetLastError();
    
	cudaEventRecord(stop_G);

    cudaEventSynchronize(stop_G); 
    */
    
    // float* mag = getMagnitudeMatrix(height, width, xOutput, yOutput);

}

__global__ void magnitude_matrix_kernel(float* mag, const float* x, const float* y, const int height, const int width) {

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int array_upper_bound = width * height;

    if (index < array_upper_bound) {
        float mags = x[index] * x[index] + y[index] * y[index];
        mag[index] = mags;
    }
}

/*
float* getMagnitudeMatrix(unsigned int height, unsigned int width, float* x, float* y) {

    unsigned int magnitudeSize = height * width;
    unsigned int magnitudeMemorySize = sizeof(float) * magnitudeSize;

    float* mag = (float*)malloc(magnitudeMemorySize);    

    float mags;
	float maxVal = 0f;
	for (int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			mags = sqrt((x[i * width + j] * x[i * width + j]) + (y[i * width + j] * y[i * width + j]));

			if (mags > maxVal)
				maxVal = mags;

			mag[i * width + j] = mags;
		}
	}

	// Make sure all the magnitude values are between 0-255
    // TODO : We can use a custom kernel to perform this operation here
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
            mag[i * width + j] = mag[i * width + j] / maxVal * 255;
            
    return mag;        
}
*/
