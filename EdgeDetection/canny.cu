#include "canny.cuh"
#include "global.hpp"
#include "point.hpp"

using namespace std;

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <time.h>

#define BLOCK_SIZE 32
#define WA 512   
#define HA 512     
#define HC 3     
#define WC 3
#define WB (WA - WC + 1)
#define HB (HA - HC + 1)

#define sig 1

/**
=========================================== Kernel Convolution =========================================================

This function performs the convolution step on the given image array using the mask array that has been passed.
The output of this step is stored in the output array.

========================================================================================================================
**/
__global__ void convolution_kernel(float* image, float* output, float* mask, int imageRows, int imageCols, int outputRows, int outputCols, int maskRows, int maskCols) {
    
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

	if (threadIdx.y < (BLOCK_SIZE - WC + 1) && threadIdx.x < (BLOCK_SIZE - WC + 1) && row < (WB - WC + 1) && col < (WB - WC + 1))
	{
		for (int i = 0; i< WC;i++)
			for (int j = 0;j<WC;j++)
				tmp += sharedMem[threadIdx.y + i][threadIdx.x + j] * mask[j*WC + i];
        // TODO Check if this indexing is correct
        output[col*WB + row] = tmp;
	}
}

__host__ void convolve(float* image, float* xOutput, float* yOutput, int imageRows, int imageCols, int outputRows, int outputCols) {

    int dim = 6 * sig + 1, cent = dim / 2;
    
    unsigned int maskSize = dim * dim;
    unsigned int maskMemorySize = sizeof(float) * maskSize;

    // float maskx[dim][dim], masky[dim][dim]
    float* maskx = (float*)malloc(maskMemorySize);
    float* masky = (float*)malloc(maskMemorySize);


	// Use the Gausian 1st derivative formula to fill in the mask values
    float denominator = 2 * sig * sig; 
	for (int p = -cent; p <= cent; p++)
	{	
		for (int q = -cent; q <= cent; q++)
		{
            float numerator = (p * p + q * q);
            
            int rowIndex = p + cent;
            int colIndex = q + cent;

            // maskx[p+cent][q+cent] = q * exp(-1 * (numerator / denominator))
			maskx[rowIndex * maskSize + colIndex] = q * exp(-1 * (numerator / denominator));
            
            // masky[p+cent][q+cent] = p * exp(-1 * (numerator / denominator))
            masky[rowIndex * maskSize + colIndex] = p * exp(-1 * (numerator / denominator)); 
		}
	}

    cudaError_t error;
    cudaEvent_t start_G, stop_G;
    
    cudaEventCreate(&start_G);
	cudaEventCreate(&stop_G);
    
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((WB - 1) / (BLOCK_SIZE - WC + 1), (WB - 1) / (BLOCK_SIZE - WC + 1));

	convolution_kernel <<< grid, threads >>> (image, xOutput, maskx, imageRows, imageCols, outputRows, outputCols, maskSize, maskSize);

	cudaEventRecord(start_G);

	convolution_kernel <<< grid, threads >>> (image, yOutput, masky, imageRows, imageCols, outputRows, outputCols, maskSize, maskSize);
	error = cudaGetLastError();
    
    if (error != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in launching kernel\n", cudaGetErrorString(error));
		return EXIT_FAILURE;
	}

	error = cudaDeviceSynchronize();

	if (error != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s  in cudaDeviceSynchronize \n", cudaGetErrorString(error));
		return EXIT_FAILURE;
	}

	cudaEventRecord(stop_G);

    cudaEventSynchronize(stop_G); 
    
    float* mag = getMagnitudeMatrix(height, width, xOutput, yOutput);

}

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
