#include "hough.cuh"
#include <fstream>
using namespace std;

__global__ void find_edge_points(uint8_t* image, int image_size, uint8_t* edges, unsigned int* edges_len) {
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

//__global__ void hough(uint8_t* edges, unsigned int* edges_len) {
//	int idk = 2;
//}