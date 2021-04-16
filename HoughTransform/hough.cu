#include "hough.cuh"
#include <fstream>
using namespace std;

__global__ void find_edge_points(uint8_t* image, int image_size, uint8_t* edges, unsigned int* edges_len) {
	extern __shared__ uint8_t sh_mem[];
	unsigned int* sh_next = (unsigned int*)sh_mem;
	unsigned int* sh_global_write = (unsigned int*)(sh_mem + 4);
	uint8_t* sh_edges = sh_mem + 8;

	if (threadIdx.x == 0) {
		*sh_next = 0;
	}

	__syncthreads();

	int image_x = threadIdx.x;
	int image_y = blockIdx.x;

	int edge = image[image_y * image_size + image_x];

	if (edge == 1) {
		unsigned int write_ind = atomicAdd(sh_next, (unsigned int)2);

		sh_edges[write_ind] = image_x;
		sh_edges[write_ind + 1] = image_y;
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		*sh_global_write = atomicAdd(edges_len, *sh_next);
		//printf("%d,%d,%d\n", blockIdx.x, *sh_next, *sh_global_write);
	}
	__syncthreads();

	if (threadIdx.x < *sh_next) {
		//printf("Block: %d\tThread: %d\tGlobal: %d\tShared: %d\n", blockIdx.x, threadIdx.x, *sh_global_write + threadIdx.x, threadIdx.x);
		//printf("%d,%d,%d\n", blockIdx.x, threadIdx.x, *sh_global_write + threadIdx.x);
		edges[*sh_global_write + threadIdx.x] = sh_edges[threadIdx.x];
	}
}

__global__ void hough(uint8_t* edges, unsigned int* edges_len) {
	int idk = 2;
}