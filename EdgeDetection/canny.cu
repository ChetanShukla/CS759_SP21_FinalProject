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

/* ================================ Peaks Detection ================================
    The formula for the slope of a given line is Δy/Δx. We have Δy and Δx from the
    scanning convolution step before. We can get the slope by dividing the two.
    We'll store all the points that are greater than both its neighbors in the 
    direction of the slope into a vector. We can calculate the direction of the slope
    using the tan(x) function. We'll also store the peaks in a HashMap for 
    O(1) searches in the recursiveDT function later.
   ================================ Peaks Detection ================================
*/
 vector<Point*> peak_detection(double *mag, unordered_map<Point*, bool> peaks, double *x, double *y, int height, int width)
{
	double slope = 0;
	vector<Point*> v;
	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
            // The value of this can be used for [i][j]
            int equivalent1DCurrentIndex = i * width + j;
			// To avoid dividing by zero
            if (x[equivalent1DCurrentIndex] == 0)
                x[equivalent1DCurrentIndex] = 0.00001;

            slope = y[equivalent1DCurrentIndex] / x[equivalent1DCurrentIndex];
            Point* givenPoint = new Point(i, j);

            bool shouldInsert = false;

			// We're only looking for the peaks. If we're at a peak, store 255 in 'peaks'
			if (slope <= tan(22.5) && slope > tan(-22.5)) {
                if (mag[equivalent1DCurrentIndex] > mag[equivalent1DCurrentIndex - 1] 
                    && mag[equivalent1DCurrentIndex] > mag[equivalent1DCurrentIndex + 1]) {
                    shouldInsert = true;
				}
			}
			else if (slope <= tan(67.5) && slope > tan(22.5)) {
                if (mag[equivalent1DCurrentIndex] > mag[((i-1) * width) + j-1] 
                    && mag[equivalent1DCurrentIndex] > mag[((i+1) * width) + j+1]) {
                    shouldInsert = true;
				}
			}
			else if (slope <= tan(-22.5) && slope > tan(-67.5)) {
                if (mag[equivalent1DCurrentIndex] > mag[((i+1) * width) + j-1] 
                    && mag[equivalent1DCurrentIndex] > mag[((i-1) * width) + j+1]) {
                    shouldInsert = true;
				}
			}
			else {
                if (mag[equivalent1DCurrentIndex] > mag[((i-1) * width) + j] 
                    && mag[equivalent1DCurrentIndex] > mag[((i+1) * width) + j]) {
                    shouldInsert = true;
				}
            }
            
            if (shouldInsert == true) {
                v.push_back(givenPoint);
                peaks.insert(make_pair(givenPoint, true));
            }
		}
	}

	return v;
}

// ======================== Hysteresis & Double Thresholding ========================
// The points passed into this function are coming from the peaks vector. We'll start
// by searching around the current pixel for a pixel that made it to "final". If
// found, then we'll recursively search for a "series" of pixels that are in the mid
// range and swith all those to ON in final. We'll stop as soon as all the pixels are
// either already processed or less than the 'lo' threshold.
// ======================== Hysteresis & Double Thresholding ========================
void recursiveDoubleThresholding(double *mag, double *final, unordered_map<Point*, bool> visited, unordered_map<Point*, bool> peaks, 
                    int a, int b, int flag, int width, int height) {
    
                        // If the pixel value is < lo, out-of-bounds, or at a point we've visited before,
	// then exit the funciton.
	if (mag[a * width + b] < lo || a < 0 || b < 0 || a >= height || b >= width)
		return;

    Point* givenPoint = new Point(a, b);    
	if (visited.find(givenPoint) != visited.end())
		return;

	// Insert the current pixel so we know we've been here before.
	visited.insert(make_pair(givenPoint, true));

	// If flag = 0, that means that this is the first pixel of the "series" that
	// we're looking at. We're going to look for a pixel in "final" that's set to
	// ON. If we found one, assert the flag and break out of the loops.
	if (!flag)
	{
		for (int p = -1; p <= 1; p++)
		{
			for (int q = -1; q <= 1; q++)
			{
                int rowIndex = a+p;
                int colIndex = b+q;
                if (final[rowIndex * width + colIndex] == 255) {
					final[a * width + b] = 255;
					flag = 1;
					break;
				}
			}

			if (flag)
				break;
		}
	}
	
	// If flag is asserted, that means we found a pixel that's in final, all what
	// we have to do now is just search for pixels that are in the mid range.
	// Also, make sure that it's in the peaks to begin with.
	if (flag)
	{
		for (int p = -1; p <= 1; p++) {
			for (int q = -1; q <= 1; q++) {
                int rowIndex = a+p;
                int colIndex = b+q;
                Point* currentPoint = new Point(rowIndex, colIndex); 
				if (mag[rowIndex * width + colIndex] >= lo && peaks.find(currentPoint) != peaks.end()) {
                    
                    recursiveDoubleThresholding(mag, final, h, peaks, a+p, b+q, 1);
					final[a * width + b] = 255;
                }
			}
		}
	}

	return;
}