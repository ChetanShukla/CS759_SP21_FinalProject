#include "hough.cuh"
#include <fstream>
#include <math_constants.h>
using namespace std;

__global__ void accumulate_edge_points(uint8_t* image, int image_size, uint8_t* edges, unsigned int* edges_len) {
	//Setup shared memory variables
	extern __shared__ uint8_t sh_mem[];
	unsigned int* sh_next = (unsigned int*)sh_mem;
	unsigned int* sh_global_write = (unsigned int*)(sh_mem + 4);
	uint8_t* sh_edges = sh_mem + 8;

	//Initialize shared memory
	if (threadIdx.x == 0) {
		*sh_next = 0;
	}
	__syncthreads();

	int image_x = threadIdx.x;
	int image_y = blockIdx.x;

	//Bring in the proper pixel from global memory
	int pixel = image[image_y * image_size + image_x];
	//If the pixel is part of an edge
	if (pixel == 1) {
		unsigned int write_ind = atomicAdd(sh_next, (unsigned int)2);
		//Write the point to shared memory
		sh_edges[write_ind] = image_x + 1;
		sh_edges[write_ind + 1] = image_y + 1;
	}
	__syncthreads();

	//Figure out where we need to start writing our portion in global memory
	if (threadIdx.x == 0) {
		*sh_global_write = atomicAdd(edges_len, *sh_next);
	}
	__syncthreads();

	//Write our shared memory to global memory
	if (threadIdx.x < *sh_next) {
		edges[*sh_global_write + threadIdx.x] = sh_edges[threadIdx.x];
	}
}

__global__ void hough(uint8_t* edges, unsigned int* edges_len, int* global_acc) {
	//(Image_size/shrink)^2 * 3 different radius sizes
	//(256/4) = 64
	__shared__ unsigned int hough[64 * 64 * 3];

	//for (int i = 0; i < 64; i++) {
	//	for (int j = 0; j < 3; j++) {
	//		hough[threadIdx.x * 64 + i * 64 + j];
	//	}
	//}

	for (int i = threadIdx.x; i < 64 * 64 * 3; i += blockDim.x) {
		hough[i] = 0;
	}

	__syncthreads();

	for (int k = threadIdx.x; k < (*edges_len) / 2; k += blockDim.x) {
		uint8_t point_x = edges[k * 2];
		uint8_t point_y = edges[k * 2 + 1];

		float shrunk_y = (float)point_y / 4;
		float shrunk_x = (float)point_x / 4;

		/*if (k == 0) {
			printf("POINT: %d,%d\n", point_x, point_y);
		}*/

		for (int i = 1; i < 361; i++) {
			float sin_result = sinf((i * CUDART_PI_F) / 180);
			float cos_result = cosf((i * CUDART_PI_F) / 180);

			for (int j = 1; j <= 3; j++) {
				int a = round(shrunk_x - (5 + j) * sin_result);
				int b = round(shrunk_y - (5 + j) * cos_result);
				/*if (k == 0 && j == 1) {
					printf("%d,%d\n", a, b);
				}*/
				/*if (a == 6 && b == 0 && j == 1) {
					printf("%d", 1);
				}*/
				if (0 <= a && a < 64 && 0 <= b && b < 64) {
					atomicAdd(&hough[a + b * 64 + 64 * 64 * (j - 1)], (unsigned int)1);
					//hough[a + b * 64 + 64 * 64 * (j - 1)] += 1;
				}
			}
		}
	}

	__syncthreads();

	for (int i = threadIdx.x; i < 64 * 64 * 3; i += blockDim.x) {
		global_acc[i] = hough[i];
	}
}