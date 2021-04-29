#include "hough.cuh"
#include <fstream>
#include <math_constants.h>
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
	// (Image_size/shrink)^2 * 3 different radius sizes
	// (256/4) = 64 so 64^2*3
	// Scary since this is max shared memory size. Would love to move this to uint16_t if
	// half precision atomic add was available on maxwell.
	__shared__ unsigned int hough[64 * 64 * 3];

	//Initialize shared memory.
	for (int i = threadIdx.x; i < 64 * 64 * 3; i += blockDim.x) {
		hough[i] = 0;
	}
	__syncthreads();

	for (int k = threadIdx.x; k < (*edges_len); k += blockDim.x) {
		// Pull in from global memory.
		// Coalesced and aligned for the win.
		uint8_t point_x = edges_x[k];
		uint8_t point_y = edges_y[k];

		// Shrink the image points to fit in the accumulator space.
		float shrunk_x = (float)point_x / 4;
		float shrunk_y = (float)point_y / 4;

		for (int i = 0; i < 360; i++) {
			// Sin and cos seem to be the big players in the speed of this function
			// due to the limited number of SFU's.
			// The ability to use fast math here is advantagous over alternatives for
			// sin and cos in degree mode such as sincospif()
			float sin_result = __sinf((i * CUDART_PI_F) / 180.0);
			float cos_result = __cosf((i * CUDART_PI_F) / 180.0);

			for (int j = 0; j < 3; j++) {
				int a = round(shrunk_y - (6 + j) * sin_result);
				int b = round(shrunk_x - (6 + j) * cos_result);
				if (0 <= a && a < 64 && 0 <= b && b < 64) {
					atomicAdd(&hough[a + b * 64 + 64 * 64 * j], (unsigned int)1);
				}
			}
		}
	}

	__syncthreads();

	//Write the accumulator out to global memory.
	for (int i = threadIdx.x; i < 64 * 64 * 3; i += blockDim.x) {
		global_acc[i] = hough[i];
	}
}