#include "hough.cuh"
#include <fstream>
#include <math_constants.h>
#define NUM_DEGREE_CALC 256
using namespace std;

__global__ void accumulate_edge_points(uint8_t* image, int image_size, uint8_t* edges_x, uint8_t* edges_y, unsigned int* edges_len) {
	//Setup shared memory variables
	__shared__ unsigned int sh_next;
	__shared__ unsigned int sh_global_write;
	__shared__ uint8_t sh_edges_x[256];
	__shared__ uint8_t sh_edges_y[256];

	//Initialize shared memory
	if (threadIdx.x == 0) {
		sh_next = 0;
	}
	__syncthreads();

	uint8_t image_x = threadIdx.x;
	uint8_t image_y = blockIdx.x;

	//Bring in the proper pixel from global memory
	int pixel = image[image_y * image_size + image_x];
	//If the pixel is part of an edge
	if (pixel == 1) {
		unsigned int write_ind = atomicAdd(&sh_next, (unsigned int)1);
		//Write the point to shared memory
		//Swap x and y b/c the image is really stored columnwise from the MATLAB
		//binary image export.
		sh_edges_x[write_ind] = image_y;
		sh_edges_y[write_ind] = image_x;
	}
	__syncthreads();

	//Figure out where we need to start writing our portion in global memory
	if (threadIdx.x == 0) {
		sh_global_write = atomicAdd(edges_len, sh_next);
	}
	__syncthreads();

	//Write our shared memory to global memory
	if (threadIdx.x < sh_next) {
		edges_x[sh_global_write + threadIdx.x] = sh_edges_x[threadIdx.x];
		edges_y[sh_global_write + threadIdx.x] = sh_edges_y[threadIdx.x];
	}
}

__global__ void hough(uint8_t* edges_x, uint8_t* edges_y, unsigned int* edges_len, int* global_acc) {
	// (Image_size/shrink)^2
	// (256/4) = 64 so 64^2
	__shared__ unsigned int sh_hough[64 * 64];
	__shared__ float sh_sin[NUM_DEGREE_CALC];
	__shared__ float sh_cos[NUM_DEGREE_CALC];

	//Initialize shared memory accumulator.
	for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
		sh_hough[i] = 0;
	}

	//Precompute sin and cos values.
	float spacing = 360.0 / NUM_DEGREE_CALC;
	for (int i = threadIdx.x; i < NUM_DEGREE_CALC; i += blockDim.x) {
		// The ability to use fast math here is advantagous over alternatives for
		// sin and cos in degree mode such as sincospif()
		sh_sin[i] = __sinf((i * spacing * CUDART_PI_F) / 180.0);
		sh_cos[i] = __cosf((i * spacing * CUDART_PI_F) / 180.0);
	}
	__syncthreads();

	for (int k = threadIdx.x; k < (*edges_len); k += blockDim.x) {
		// Pull in from global memory.
		// Coalesced and aligned for the win.
		// Shrink the image points to fit in the accumulator space.
		float point_x = edges_x[k] / 4.0;
		float point_y = edges_y[k] / 4.0;

		for (int i = 0; i < NUM_DEGREE_CALC; i++) {
			int a = round(point_y - (6 + blockIdx.x) * sh_sin[i]);
			int b = round(point_x - (6 + blockIdx.x) * sh_cos[i]);
			if (0 <= a && a < 64 && 0 <= b && b < 64) {
				atomicAdd(&sh_hough[a + b * 64], (unsigned int)1);
			}
		}
	}
	__syncthreads();

	//Write the accumulator out to global memory.
	int global_offset = 64 * 64 * blockIdx.x;
	for (int i = threadIdx.x; i < 64 * 64; i += blockDim.x) {
		global_acc[global_offset + i] = sh_hough[i];
	}
}