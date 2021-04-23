#include "hough.cuh"
#include <fstream>
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
		sh_edges[write_ind] = image_x;
		sh_edges[write_ind + 1] = image_y;
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
	__shared__ int hough[64 * 64 * 3];

	//for (int i = 0; i < 64; i++) {
	//	for (int j = 0; j < 3; j++) {
	//		hough[threadIdx.x * 64 + i * 64 + j];
	//	}
	//}

	for (int i = threadIdx.x; i < 64 * 64 * 3; i += blockDim.x) {
		hough[i] = 0;
	}

	__syncthreads();

	for (int k = threadIdx.x; k < *edges_len; k += blockDim.x) {
		uint8_t point_x = edges[k * 2];
		uint8_t point_y = edges[k * 2 + 1];

		for (int i = 0; i < 360; i++) {
			float sin_result = sinpif(((float)i) / 180);
			float cos_result = cospif(((float)i) / 180);
			for (int j = 1; j <= 3; j++) {
				int a = round(point_y / 4 - (5 + j) * sin_result);
				int b = round(point_x / 4 - (5 + j) * cos_result);
				if (0 <= a && a < 64 && 0 <= b && b < 64) {
					atomicAdd(&hough[a * 64 + b * 64 + j], (int)1);
				}
			}
		}
	}

	__syncthreads();

	for (int i = threadIdx.x; i < 64 * 64 * 3; i += blockDim.x) {
		global_acc[i] = hough[i];
	}
}